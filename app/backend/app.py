import dataclasses
import io
import json
import logging
import mimetypes
import os
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast
# 图像处理
from PIL import Image
import base64
import httpx

from azure.cognitiveservices.speech import (
    ResultReason,
    SpeechConfig,
    SpeechSynthesisOutputFormat,
    SpeechSynthesisResult,
    SpeechSynthesizer,
)
from azure.identity.aio import (
    AzureDeveloperCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.httpx import (
    HTTPXClientInstrumentor,
)
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from quart import (
    Blueprint,
    Quart,
    abort,
    current_app,
    jsonify,
    make_response,
    request,
    send_file,
    send_from_directory,
)
from quart_cors import cors

from approaches.approach import Approach
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.promptmanager import PromptyManager
from approaches.retrievethenread import RetrieveThenReadApproach
from chat_history.cosmosdb import chat_history_cosmosdb_bp
from config import (
    CONFIG_AGENT_CLIENT,
    CONFIG_AGENTIC_RETRIEVAL_ENABLED,
    CONFIG_ASK_APPROACH,
    CONFIG_AUTH_CLIENT,
    CONFIG_CHAT_APPROACH,
    CONFIG_CHAT_HISTORY_BROWSER_ENABLED,
    CONFIG_CHAT_HISTORY_COSMOS_ENABLED,
    CONFIG_CREDENTIAL,
    CONFIG_DEFAULT_REASONING_EFFORT,
    CONFIG_GLOBAL_BLOB_MANAGER,
    CONFIG_INGESTER,
    CONFIG_LANGUAGE_PICKER_ENABLED,
    CONFIG_MULTIMODAL_ENABLED,
    CONFIG_OPENAI_CLIENT,
    CONFIG_QUERY_REWRITING_ENABLED,
    CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS,
    CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS,
    CONFIG_RAG_SEND_IMAGE_SOURCES,
    CONFIG_RAG_SEND_TEXT_SOURCES,
    CONFIG_REASONING_EFFORT_ENABLED,
    CONFIG_SEARCH_CLIENT,
    CONFIG_SEMANTIC_RANKER_DEPLOYED,
    CONFIG_SPEECH_INPUT_ENABLED,
    CONFIG_SPEECH_OUTPUT_AZURE_ENABLED,
    CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED,
    CONFIG_SPEECH_SERVICE_ID,
    CONFIG_SPEECH_SERVICE_LOCATION,
    CONFIG_SPEECH_SERVICE_TOKEN,
    CONFIG_SPEECH_SERVICE_VOICE,
    CONFIG_STREAMING_ENABLED,
    CONFIG_USER_BLOB_MANAGER,
    CONFIG_USER_UPLOAD_ENABLED,
    CONFIG_VECTOR_SEARCH_ENABLED,
)
from core.authentication import AuthenticationHelper
from core.sessionhelper import create_session_id
from decorators import authenticated, authenticated_path
from error import error_dict, error_response
from prepdocs import (
    OpenAIHost,
    setup_embeddings_service,
    setup_file_processors,
    setup_image_embeddings_service,
    setup_openai_client,
    setup_search_info,
)
from prepdocslib.blobmanager import AdlsBlobManager, BlobManager
from prepdocslib.embeddings import ImageEmbeddings
from prepdocslib.filestrategy import UploadUserFileStrategy
from prepdocslib.listfilestrategy import File

bp = Blueprint("routes", __name__, static_folder="static")
# 简单的内存图片缓存：key = 文件名，value = 原始二进制
IMAGE_CACHE: dict[str, bytes] = {}
# 每个会话最近一次使用的图片文件名：key = session_state 序列化后的字符串
SESSION_LAST_IMAGE: dict[str, str] = {}
# Fix Windows registry issue with mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


@bp.route("/")
async def index():
    return await bp.send_static_file("index.html")


# Empty page is recommended for login redirect to work.
# See https://github.com/AzureAD/microsoft-authentication-library-for-js/blob/dev/lib/msal-browser/docs/initialization.md#redirecturi-considerations for more information
@bp.route("/redirect")
async def redirect():
    return ""


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory(Path(__file__).resolve().parent / "static" / "assets", path)


@bp.route("/content/<path>")
@authenticated_path
async def content_file(path: str, auth_claims: dict[str, Any]):
    """
    Serve content files from blob storage from within the app to keep the example self-contained.
    *** NOTE *** if you are using app services authentication, this route will return unauthorized to all users that are not logged in
    if AZURE_ENFORCE_ACCESS_CONTROL is not set or false, logged in users can access all files regardless of access control
    if AZURE_ENFORCE_ACCESS_CONTROL is set to true, logged in users can only access files they have access to
    This is also slow and memory hungry.
    """
    # Remove page number from path, filename-1.txt -> filename.txt
    # This shouldn't typically be necessary as browsers don't send hash fragments to servers
    if path.find("#page=") > 0:
        path_parts = path.rsplit("#page=", 1)
        path = path_parts[0]
    current_app.logger.info("Opening file %s", path)
    blob_manager: BlobManager = current_app.config[CONFIG_GLOBAL_BLOB_MANAGER]

    # Get bytes and properties from the blob manager
    result = await blob_manager.download_blob(path)

    if result is None:
        current_app.logger.info("Path not found in general Blob container: %s", path)
        if current_app.config[CONFIG_USER_UPLOAD_ENABLED]:
            user_oid = auth_claims["oid"]
            user_blob_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
            result = await user_blob_manager.download_blob(path, user_oid=user_oid)
            if result is None:
                current_app.logger.exception("Path not found in DataLake: %s", path)

    if not result:
        abort(404)

    content, properties = result

    if not properties or "content_settings" not in properties:
        abort(404)

    mime_type = properties["content_settings"]["content_type"]
    if mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    # Create a BytesIO object from the bytes
    blob_file = io.BytesIO(content)
    return await send_file(blob_file, mimetype=mime_type, as_attachment=False, attachment_filename=path)


@bp.route("/ask", methods=["POST"])
@authenticated
async def ask(auth_claims: dict[str, Any]):
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})
    context["auth_claims"] = auth_claims
    try:
        approach: Approach = cast(Approach, current_app.config[CONFIG_ASK_APPROACH])
        r = await approach.run(
            request_json["messages"], context=context, session_state=request_json.get("session_state")
        )
        return jsonify(r)
    except Exception as error:
        return error_response(error, "/ask")


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


async def format_as_ndjson(r: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    try:
        async for event in r:
            yield json.dumps(event, ensure_ascii=False, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("Exception while generating response stream: %s", error)
        yield json.dumps(error_dict(error))


@bp.route("/chat", methods=["POST"])
@authenticated
async def chat(auth_claims: dict[str, Any]):
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})
    context["auth_claims"] = auth_claims
    try:
        approach: Approach = cast(Approach, current_app.config[CONFIG_CHAT_APPROACH])

        # If session state is provided, persists the session state,
        # else creates a new session_id depending on the chat history options enabled.
        session_state = request_json.get("session_state")
        if session_state is None:
            session_state = create_session_id(
                current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED],
                current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED],
            )
        result = await approach.run(
            request_json["messages"],
            context=context,
            session_state=session_state,
        )
        return jsonify(result)
    except Exception as error:
        return error_response(error, "/chat")


@bp.route("/chat/stream", methods=["POST"])
@authenticated
async def chat_stream(auth_claims: dict[str, Any]):
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})
    context["auth_claims"] = auth_claims
    try:
        approach: Approach = cast(Approach, current_app.config[CONFIG_CHAT_APPROACH])

        # If session state is provided, persists the session state,
        # else creates a new session_id depending on the chat history options enabled.
        session_state = request_json.get("session_state")
        if session_state is None:
            session_state = create_session_id(
                current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED],
                current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED],
            )
        result = await approach.run_stream(
            request_json["messages"],
            context=context,
            session_state=session_state,
        )
        response = await make_response(format_as_ndjson(result))
        response.timeout = None  # type: ignore
        response.mimetype = "application/json-lines"
        return response
    except Exception as error:
        return error_response(error, "/chat")


# # Send MSAL.js settings to the client UI
# @bp.route("/auth_setup", methods=["GET"])
# def auth_setup():
#     auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
#     return jsonify(auth_helper.get_auth_setup_for_client())
# Send MSAL.js settings to the client UI
@bp.route("/auth_setup", methods=["GET"])
def auth_setup():
    """
    本地开发模式：
    - 不使用 Azure AD / MSAL 认证
    - 告诉前端：不启用认证，允许未登录访问
    """
    return jsonify({
        "useAuthentication": False,
        "enableUnauthenticatedAccess": True,
        "enforceAccessControl": False,
    })

@bp.route("/api/image_chat", methods=["POST"])
async def image_chat():
    """
    最小多模态接口：
    - 接收一个图片 + 可选的文本 prompt
    - 调用本地 Qwen3-VL 做“看图回答”
    - 不依赖前端改动，你可以用 curl / Postman 直接访问
    """
    model = current_app.config.get("LOCAL_QWEN_MODEL")
    processor = current_app.config.get("LOCAL_QWEN_PROCESSOR")

    if model is None or processor is None:
        # 理论上不会出现，除非 setup_clients 没跑成功
        return jsonify({"error": "Qwen3-VL model not initialized"}), 500

    # 1) 读取上传的图片
    # 这里假设使用 multipart/form-data: image=..., prompt=...
    if "image" not in (await request.files):
        return jsonify({"error": "No image file uploaded. Use form field 'image'."}), 400

    files = await request.files  # Quart 的 request.files 需要 await（视版本而定）
    img_file = files["image"]

    form = await request.form
    prompt = form.get("prompt", "请描述这张图片。")

    # 2) 把文件读取为 PIL.Image
    # img_bytes = await img_file.read()
    img_bytes = img_file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    # 3) 构造 Qwen3-VL 所需的 messages（真正用上 type=image）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # Qwen3-VL 的 AutoProcessor 可以直接接收 PIL Image
                    "image": img,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # 4) 调用模型做多模态推理
    def _infer():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)

        # 去掉 prompt，保留新生成内容
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], out)
        ]
        texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts[0] if texts else ""

    try:
        answer_text = await asyncio.to_thread(_infer)
    except Exception as e:
        current_app.logger.exception("Qwen3-VL image_chat failed")
        # 这里我们可以走统一的 ChatResponse 结构，也可以简化一点
        return jsonify({"error": f"Qwen3-VL 推理失败: {e}"}), 500

    # 5) 返回一个简单 JSON（不用管前端 Chat 协议）
    # 因为这个接口你先通过 curl / Postman 自己调试
    return jsonify({
        "answer": answer_text,
        "prompt": prompt,
    })

'''
Question: What is the name of the colony shown?
Options:
A. Maryland
B. New Hampshire
C. Rhode Islan
D. Vermont
'''

@bp.route("/config", methods=["GET"])
def config():
    return jsonify(
        {
            "showMultimodalOptions": current_app.config[CONFIG_MULTIMODAL_ENABLED],
            "showSemanticRankerOption": current_app.config[CONFIG_SEMANTIC_RANKER_DEPLOYED],
            "showQueryRewritingOption": current_app.config[CONFIG_QUERY_REWRITING_ENABLED],
            "showReasoningEffortOption": current_app.config[CONFIG_REASONING_EFFORT_ENABLED],
            "streamingEnabled": current_app.config[CONFIG_STREAMING_ENABLED],
            "defaultReasoningEffort": current_app.config[CONFIG_DEFAULT_REASONING_EFFORT],
            "showVectorOption": current_app.config[CONFIG_VECTOR_SEARCH_ENABLED],
            "showUserUpload": current_app.config[CONFIG_USER_UPLOAD_ENABLED],
            "showLanguagePicker": current_app.config[CONFIG_LANGUAGE_PICKER_ENABLED],
            "showSpeechInput": current_app.config[CONFIG_SPEECH_INPUT_ENABLED],
            "showSpeechOutputBrowser": current_app.config[CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED],
            "showSpeechOutputAzure": current_app.config[CONFIG_SPEECH_OUTPUT_AZURE_ENABLED],
            "showChatHistoryBrowser": current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED],
            "showChatHistoryCosmos": current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED],
            "showAgenticRetrievalOption": current_app.config[CONFIG_AGENTIC_RETRIEVAL_ENABLED],
            "ragSearchTextEmbeddings": current_app.config[CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS],
            "ragSearchImageEmbeddings": current_app.config[CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS],
            "ragSendTextSources": current_app.config[CONFIG_RAG_SEND_TEXT_SOURCES],
            "ragSendImageSources": current_app.config[CONFIG_RAG_SEND_IMAGE_SOURCES],
        }
    )


@bp.route("/speech", methods=["POST"])
async def speech():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415

    speech_token = current_app.config.get(CONFIG_SPEECH_SERVICE_TOKEN)
    if speech_token is None or speech_token.expires_on < time.time() + 60:
        speech_token = await current_app.config[CONFIG_CREDENTIAL].get_token(
            "https://cognitiveservices.azure.com/.default"
        )
        current_app.config[CONFIG_SPEECH_SERVICE_TOKEN] = speech_token

    request_json = await request.get_json()
    text = request_json["text"]
    try:
        # Construct a token as described in documentation:
        # https://learn.microsoft.com/azure/ai-services/speech-service/how-to-configure-azure-ad-auth?pivots=programming-language-python
        auth_token = (
            "aad#"
            + current_app.config[CONFIG_SPEECH_SERVICE_ID]
            + "#"
            + current_app.config[CONFIG_SPEECH_SERVICE_TOKEN].token
        )
        speech_config = SpeechConfig(auth_token=auth_token, region=current_app.config[CONFIG_SPEECH_SERVICE_LOCATION])
        speech_config.speech_synthesis_voice_name = current_app.config[CONFIG_SPEECH_SERVICE_VOICE]
        speech_config.speech_synthesis_output_format = SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result: SpeechSynthesisResult = synthesizer.speak_text_async(text).get()
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return result.audio_data, 200, {"Content-Type": "audio/mp3"}
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            current_app.logger.error(
                "Speech synthesis canceled: %s %s", cancellation_details.reason, cancellation_details.error_details
            )
            raise Exception("Speech synthesis canceled. Check logs for details.")
        else:
            current_app.logger.error("Unexpected result reason: %s", result.reason)
            raise Exception("Speech synthesis failed. Check logs for details.")
    except Exception as e:
        current_app.logger.exception("Exception in /speech")
        return jsonify({"error": str(e)}), 500

@bp.post("/upload")
async def upload():
    """本地开发版上传接口：接收 file 字段，保存到 static/assets/uploads 下，并返回可访问 URL"""
    # 和 image_chat 一样的风格：await request.files（只获取一次，避免重复读取）
    request_files = await request.files
    current_app.logger.info("request.files keys=%s", list(request_files.keys()))
    file_storage = None

    # # 先尝试 "file"
    # if "file" in request_files:
    #     file_storage = request_files.getlist("file")[0]
    # # 再兼容 "files"
    # elif "files" in request_files:
    #     file_storage = request_files.getlist("files")[0]
    if "file" in request_files:
        files = request_files.getlist("file")
        file_storage = files[0] if files else None
    elif "files" in request_files:
        files = request_files.getlist("files")
        file_storage = files[0] if files else None

    if file_storage is None:
        return jsonify({"message": "No file part in the request", "status": "failed"}), 400

    try:
        # file = request_files.getlist("file")[0]
        file = file_storage 

        backend_root = Path(__file__).resolve().parent
        # 注意：这里是 static/assets/uploads，方便复用 /assets 路由
        upload_dir = backend_root / "static" / "assets" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename).suffix or ""
        timestamp = int(time.time() * 1000)
        safe_name = f"img_{timestamp}{suffix}"
        save_path = upload_dir / safe_name

        file_bytes = file.read()
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # 放进内存缓存，key 用 safe_name 即可（使用本模块顶部定义的全局 IMAGE_CACHE）
        IMAGE_CACHE[str(safe_name)] = file_bytes

        # 前端和 Qwen 用这个相对 URL
        public_url = f"/assets/uploads/{safe_name}"
        current_app.logger.info("Local upload saved file to %s, url=%s", save_path, public_url)
        return jsonify({"message": public_url, "status": "ok"}), 200
    except Exception as error:
        current_app.logger.error("Error uploading file: %s", error)
        return jsonify(
            {"message": "Error uploading file, check server logs for details.", "status": "failed"}
        ), 500

@bp.post("/delete_uploaded")
@authenticated
async def delete_uploaded(auth_claims: dict[str, Any]):
    request_json = await request.get_json()
    filename = request_json.get("filename")
    user_oid = auth_claims["oid"]
    adls_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
    await adls_manager.remove_blob(filename, user_oid)
    ingester: UploadUserFileStrategy = current_app.config[CONFIG_INGESTER]
    await ingester.remove_file(filename, user_oid)
    return jsonify({"message": f"File {filename} deleted successfully"}), 200


@bp.get("/list_uploaded")
@authenticated
async def list_uploaded(auth_claims: dict[str, Any]):
    """Lists the uploaded documents for the current user.
    Only returns files directly in the user's directory, not in subdirectories.
    Excludes image files and the images directory."""
    user_oid = auth_claims["oid"]
    adls_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
    files = await adls_manager.list_blobs(user_oid)
    return jsonify(files), 200

import asyncio
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

@bp.before_app_serving
async def setup_clients():
    current_app.logger.info("Local mode: loading local Qwen3-VL model")

    # 1. 模型路径
    model_path = os.getenv("LOCAL_QWEN3_PATH", "/root/autodl-tmp/qwen3")

    # 2. 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )

    # 2.1 如果存在 LoRA 适配器，则在基座模型上加载适配器权重
    adapter_path = "/root/autodl-tmp/models/lora/1"
    try:
        if os.path.isdir(adapter_path):
            current_app.logger.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            current_app.logger.info("LoRA adapter loaded successfully")
        else:
            current_app.logger.warning(
                "LoRA adapter path not found: %s, using base model only", adapter_path
            )
    except Exception as e:
        current_app.logger.exception("Failed to load LoRA adapter, fallback to base model: %s", e)

    processor = AutoProcessor.from_pretrained(model_path)
    current_app.logger.info("Qwen3-VL model loaded successfully")

    # 3. 定义本地推理 Approach
    class Qwen3LocalApproach:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
        async def run(
                self,
                messages,
                *,
                overrides=None,
                auth_claims=None,
                context=None,
                session_state=None,
                **kwargs,
        ):
            """
            本地 Qwen3-VL 的 run 接口（带简单记忆）：
            - 用 messages 里的多轮文本作为上下文
            - 把图片挂在“最后一条用户消息”上
            """

            # 为当前会话生成一个简单的 key，用于记住最近一次使用的图片
            # 在本地模式下 session_state 可能是 None/字符串/字典，这里做统一归一化：
            # - None、{} 等“空状态”都归为同一个 "default" 会话
            # - 字典则序列化为排序后的 JSON 字符串，保证稳定
            # - 其它类型直接用 str()
            if not session_state:
                session_key = "default"
            elif isinstance(session_state, dict):
                try:
                    session_key = json.dumps(session_state, sort_keys=True, ensure_ascii=False)
                except TypeError:
                    session_key = str(session_state)
            else:
                session_key = str(session_state)

            # ===== 1. 先把所有历史消息转成“只含文本”的多轮结构 =====
            converted_messages: list[dict[str, Any]] = []

            for m in messages or []:
                role = m.get("role", "user")
                content = m.get("content", "")

                if isinstance(content, list):
                    text_parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                    text = "\n".join(text_parts)
                else:
                    text = str(content or "")

                converted_messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": text or "你好",
                            }
                        ],
                    }
                )

            if not converted_messages:
                converted_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "你好"}],
                    }
                )

            # 当前轮用户文本 = 最后一条 user 消息里的文本（没有就用最后一条）
            user_text = "你好"
            for i in range(len(converted_messages) - 1, -1, -1):
                if converted_messages[i].get("role") == "user":
                    user_text = converted_messages[i]["content"][0]["text"]
                    break
            else:
                user_text = converted_messages[-1]["content"][0]["text"]

            # ===== 2. 从 context.images 里取第一张图片，并从 IMAGE_CACHE 还原为 PIL.Image =====
            image_attachments: list[dict[str, Any]] = []
            if isinstance(context, dict):
                ctx_images = context.get("images") or []
                for img in ctx_images:
                    if isinstance(img, dict) and img.get("url"):
                        image_attachments.append(
                            {
                                "url": str(img["url"]),
                                "name": img.get("name"),
                                "mimeType": img.get("mimeType"),
                            }
                        )
            current_app.logger.info("Qwen3LocalApproach context.images = %r", image_attachments)

            pil_img = None
            used_filename: str | None = None
            if image_attachments:
                raw_url = str(image_attachments[0]["url"])
                filename = raw_url.rsplit("/", 1)[-1]
                current_app.logger.info("Qwen3LocalApproach image filename = %r", filename)

                try:
                    img_bytes = IMAGE_CACHE.get(filename)
                    if img_bytes is None:
                        current_app.logger.warning("IMAGE_CACHE miss for %s, fallback to pure text", filename)
                    else:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        current_app.logger.info("Qwen3LocalApproach loaded image from IMAGE_CACHE for %s", filename)
                        used_filename = filename
                except Exception as e:
                    current_app.logger.exception("从 IMAGE_CACHE 恢复图片失败: %s", e)
                    pil_img = None

            # 如果本轮没有显式传图片，但该会话之前使用过图片，则复用上一次的图片
            if pil_img is None and session_key in SESSION_LAST_IMAGE:
                last_filename = SESSION_LAST_IMAGE[session_key]
                try:
                    img_bytes = IMAGE_CACHE.get(last_filename)
                    if img_bytes is None:
                        current_app.logger.warning(
                            "SESSION_LAST_IMAGE points to %s but IMAGE_CACHE miss, fallback to pure text",
                            last_filename,
                        )
                    else:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        used_filename = last_filename
                        current_app.logger.info(
                            "Qwen3LocalApproach reused last image %s for session %s", last_filename, session_key
                        )
                except Exception as e:
                    current_app.logger.exception("复用上一次会话图片失败: %s", e)
                    pil_img = None

            # 如果最终确定了本轮使用的图片文件名，则更新会话 -> 图片 的映射
            if pil_img is not None and used_filename is not None:
                SESSION_LAST_IMAGE[session_key] = used_filename

            # ===== 3. 把图片挂到“最后一条用户消息”的 content 里 =====
            target_idx = None
            for i in range(len(converted_messages) - 1, -1, -1):
                if converted_messages[i].get("role") == "user":
                    target_idx = i
                    break
            if target_idx is None:
                target_idx = len(converted_messages) - 1

            if pil_img is not None:
                new_content = list(converted_messages[target_idx]["content"])
                # 图片放在文本前面
                new_content.insert(
                    0,
                    {
                        "type": "image",
                        "image": pil_img,
                    },
                )
                converted_messages[target_idx]["content"] = new_content

            # 可选：加一条 system 提示
            qwen_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who answers science questions from elementary "
                                    "to high school. Read the question and context carefully, then first give the "
                                    "correct answer, and then provide a clear explanation of your reasoning so "
                                    "that a student can understand."
                        }
                    ],
                },
                *converted_messages,
            ]
            if len(qwen_messages) == 1:
                qwen_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "你好"}],
                    }
                )

            # 记录这一轮最终是否携带了图片，便于从日志中确认模型是否“看到了图”
            has_image = False
            for msg in qwen_messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "image":
                            has_image = True
                            break
                if has_image:
                    break
            current_app.logger.info(
                "Qwen3LocalApproach final has_image=%s, session_key=%r, session_last_image=%r",
                has_image,
                session_key,
                SESSION_LAST_IMAGE.get(session_key),
            )

            # ===== 4. 调用模型生成 =====
            def _generate():
                inputs = self.processor.apply_chat_template(
                    qwen_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=512)

                trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output)
                ]
                texts = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return texts[0] if texts else ""

            try:
                answer_text = await asyncio.to_thread(_generate)
            except Exception as e:
                current_app.logger.exception("Qwen3 推理失败")
                answer_text = f"Qwen3 推理失败：{e}"

            return {
                "message": {
                    "content": answer_text,
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                },
                "context": {
                    "data_points": {
                        "text": []
                    },
                    "thoughts": [
                        {
                            "title": "Local Qwen3-VL",
                            "description": "回答由本地 Qwen3-VL 模型生成。",
                            "props": None,
                        }
                    ],
                },
                "session_state": session_state or {},
            }
        
    qwen_approach = Qwen3LocalApproach(model, processor)

    # ===== 4. 关闭所有 Azure 相关客户端 & 功能开关 =====
    current_app.config[CONFIG_OPENAI_CLIENT] = None
    current_app.config[CONFIG_SEARCH_CLIENT] = None
    current_app.config[CONFIG_AGENT_CLIENT] = None
    current_app.config[CONFIG_AUTH_CLIENT] = None

    current_app.config[CONFIG_ASK_APPROACH] = qwen_approach
    current_app.config[CONFIG_CHAT_APPROACH] = qwen_approach

    # 暴露模型句柄，方便其他接口直接用
    current_app.config["LOCAL_QWEN_MODEL"] = model
    current_app.config["LOCAL_QWEN_PROCESSOR"] = processor

    current_app.config[CONFIG_SEMANTIC_RANKER_DEPLOYED] = False
    current_app.config[CONFIG_QUERY_REWRITING_ENABLED] = False
    current_app.config[CONFIG_DEFAULT_REASONING_EFFORT] = None
    current_app.config[CONFIG_REASONING_EFFORT_ENABLED] = False
    current_app.config[CONFIG_STREAMING_ENABLED] = False
    current_app.config[CONFIG_VECTOR_SEARCH_ENABLED] = False
    current_app.config[CONFIG_USER_UPLOAD_ENABLED] = False
    current_app.config[CONFIG_LANGUAGE_PICKER_ENABLED] = False
    current_app.config[CONFIG_SPEECH_INPUT_ENABLED] = False
    current_app.config[CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED] = False
    current_app.config[CONFIG_SPEECH_OUTPUT_AZURE_ENABLED] = False
    current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED] = False
    current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED] = False
    current_app.config[CONFIG_AGENTIC_RETRIEVAL_ENABLED] = False
    current_app.config[CONFIG_MULTIMODAL_ENABLED] = False  # 先关多模态，后面再一点点开
    current_app.config[CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS] = False
    current_app.config[CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS] = False
    current_app.config[CONFIG_RAG_SEND_TEXT_SOURCES] = False
    current_app.config[CONFIG_RAG_SEND_IMAGE_SOURCES] = False

    current_app.logger.info("Local Qwen3-VL mode enabled")

@bp.after_app_serving
async def close_clients():
    await current_app.config[CONFIG_SEARCH_CLIENT].close()
    await current_app.config[CONFIG_GLOBAL_BLOB_MANAGER].close_clients()
    if user_blob_manager := current_app.config.get(CONFIG_USER_BLOB_MANAGER):
        await user_blob_manager.close_clients()
    await current_app.config[CONFIG_CREDENTIAL].close()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.register_blueprint(chat_history_cosmosdb_bp)

    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        app.logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING is set, enabling Azure Monitor")
        configure_azure_monitor(
            instrumentation_options={
                "django": {"enabled": False},
                "psycopg2": {"enabled": False},
                "fastapi": {"enabled": False},
            }
        )
        # This tracks HTTP requests made by aiohttp:
        AioHttpClientInstrumentor().instrument()
        # This tracks HTTP requests made by httpx:
        HTTPXClientInstrumentor().instrument()
        # This tracks OpenAI SDK requests:
        OpenAIInstrumentor().instrument()
        # This middleware tracks app route requests:
        app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)  # type: ignore[assignment]

    # Log levels should be one of https://docs.python.org/3/library/logging.html#logging-levels
    # Set root level to WARNING to avoid seeing overly verbose logs from SDKS
    logging.basicConfig(level=logging.WARNING)
    # Set our own logger levels to INFO by default
    app_level = os.getenv("APP_LOG_LEVEL", "INFO")
    app.logger.setLevel(os.getenv("APP_LOG_LEVEL", app_level))
    logging.getLogger("scripts").setLevel(app_level)

    if allowed_origin := os.getenv("ALLOWED_ORIGIN"):
        allowed_origins = allowed_origin.split(";")
        if len(allowed_origins) > 0:
            app.logger.info("CORS enabled for %s", allowed_origins)
            cors(app, allow_origin=allowed_origins, allow_methods=["GET", "POST"])

    return app

# @bp.before_app_serving
# async def setup_clients():
#     # Replace these with your own values, either in environment variables or directly here
#     AZURE_STORAGE_ACCOUNT = os.environ["AZURE_STORAGE_ACCOUNT"]
#     AZURE_STORAGE_CONTAINER = os.environ["AZURE_STORAGE_CONTAINER"]
#     AZURE_IMAGESTORAGE_CONTAINER = os.environ.get("AZURE_IMAGESTORAGE_CONTAINER")
#     AZURE_USERSTORAGE_ACCOUNT = os.environ.get("AZURE_USERSTORAGE_ACCOUNT")
#     AZURE_USERSTORAGE_CONTAINER = os.environ.get("AZURE_USERSTORAGE_CONTAINER")
#     AZURE_SEARCH_SERVICE = os.environ["AZURE_SEARCH_SERVICE"]
#     AZURE_SEARCH_ENDPOINT = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"
#     AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
#     AZURE_SEARCH_AGENT = os.getenv("AZURE_SEARCH_AGENT", "")
#     # Shared by all OpenAI deployments
#     OPENAI_HOST = OpenAIHost(os.getenv("OPENAI_HOST", "azure"))
#     OPENAI_CHATGPT_MODEL = os.environ["AZURE_OPENAI_CHATGPT_MODEL"]
#     AZURE_OPENAI_SEARCHAGENT_MODEL = os.getenv("AZURE_OPENAI_SEARCHAGENT_MODEL")
#     AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT = os.getenv("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT")
#     OPENAI_EMB_MODEL = os.getenv("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002")
#     OPENAI_EMB_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMB_DIMENSIONS") or 1536)
#     OPENAI_REASONING_EFFORT = os.getenv("AZURE_OPENAI_REASONING_EFFORT")
#     # Used with Azure OpenAI deployments
#     AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
#     AZURE_OPENAI_CHATGPT_DEPLOYMENT = (
#         os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
#         if OPENAI_HOST in [OpenAIHost.AZURE, OpenAIHost.AZURE_CUSTOM]
#         else None
#     )
#     AZURE_OPENAI_EMB_DEPLOYMENT = (
#         os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT") if OPENAI_HOST in [OpenAIHost.AZURE, OpenAIHost.AZURE_CUSTOM] else None
#     )
#     AZURE_OPENAI_CUSTOM_URL = os.getenv("AZURE_OPENAI_CUSTOM_URL")
#     AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
#     AZURE_OPENAI_API_KEY_OVERRIDE = os.getenv("AZURE_OPENAI_API_KEY_OVERRIDE")
#     # Used only with non-Azure OpenAI deployments
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
#
#     AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
#     AZURE_USE_AUTHENTICATION = os.getenv("AZURE_USE_AUTHENTICATION", "").lower() == "true"
#     AZURE_ENFORCE_ACCESS_CONTROL = os.getenv("AZURE_ENFORCE_ACCESS_CONTROL", "").lower() == "true"
#     AZURE_ENABLE_UNAUTHENTICATED_ACCESS = os.getenv("AZURE_ENABLE_UNAUTHENTICATED_ACCESS", "").lower() == "true"
#     AZURE_SERVER_APP_ID = os.getenv("AZURE_SERVER_APP_ID")
#     AZURE_SERVER_APP_SECRET = os.getenv("AZURE_SERVER_APP_SECRET")
#     AZURE_CLIENT_APP_ID = os.getenv("AZURE_CLIENT_APP_ID")
#     AZURE_AUTH_TENANT_ID = os.getenv("AZURE_AUTH_TENANT_ID", AZURE_TENANT_ID)
#
#     KB_FIELDS_CONTENT = os.getenv("KB_FIELDS_CONTENT", "content")
#     KB_FIELDS_SOURCEPAGE = os.getenv("KB_FIELDS_SOURCEPAGE", "sourcepage")
#
#     AZURE_SEARCH_QUERY_LANGUAGE = os.getenv("AZURE_SEARCH_QUERY_LANGUAGE") or "en-us"
#     AZURE_SEARCH_QUERY_SPELLER = os.getenv("AZURE_SEARCH_QUERY_SPELLER") or "lexicon"
#     AZURE_SEARCH_SEMANTIC_RANKER = os.getenv("AZURE_SEARCH_SEMANTIC_RANKER", "free").lower()
#     AZURE_SEARCH_QUERY_REWRITING = os.getenv("AZURE_SEARCH_QUERY_REWRITING", "false").lower()
#     # This defaults to the previous field name "embedding", for backwards compatibility
#     AZURE_SEARCH_FIELD_NAME_EMBEDDING = os.getenv("AZURE_SEARCH_FIELD_NAME_EMBEDDING", "embedding")
#
#     AZURE_SPEECH_SERVICE_ID = os.getenv("AZURE_SPEECH_SERVICE_ID")
#     AZURE_SPEECH_SERVICE_LOCATION = os.getenv("AZURE_SPEECH_SERVICE_LOCATION")
#     AZURE_SPEECH_SERVICE_VOICE = os.getenv("AZURE_SPEECH_SERVICE_VOICE") or "en-US-AndrewMultilingualNeural"
#
#     USE_MULTIMODAL = os.getenv("USE_MULTIMODAL", "").lower() == "true"
#     RAG_SEARCH_TEXT_EMBEDDINGS = os.getenv("RAG_SEARCH_TEXT_EMBEDDINGS", "true").lower() == "true"
#     RAG_SEARCH_IMAGE_EMBEDDINGS = os.getenv("RAG_SEARCH_IMAGE_EMBEDDINGS", "true").lower() == "true"
#     RAG_SEND_TEXT_SOURCES = os.getenv("RAG_SEND_TEXT_SOURCES", "true").lower() == "true"
#     RAG_SEND_IMAGE_SOURCES = os.getenv("RAG_SEND_IMAGE_SOURCES", "true").lower() == "true"
#     USE_USER_UPLOAD = os.getenv("USE_USER_UPLOAD", "").lower() == "true"
#     ENABLE_LANGUAGE_PICKER = os.getenv("ENABLE_LANGUAGE_PICKER", "").lower() == "true"
#     USE_SPEECH_INPUT_BROWSER = os.getenv("USE_SPEECH_INPUT_BROWSER", "").lower() == "true"
#     USE_SPEECH_OUTPUT_BROWSER = os.getenv("USE_SPEECH_OUTPUT_BROWSER", "").lower() == "true"
#     USE_SPEECH_OUTPUT_AZURE = os.getenv("USE_SPEECH_OUTPUT_AZURE", "").lower() == "true"
#     USE_CHAT_HISTORY_BROWSER = os.getenv("USE_CHAT_HISTORY_BROWSER", "").lower() == "true"
#     USE_CHAT_HISTORY_COSMOS = os.getenv("USE_CHAT_HISTORY_COSMOS", "").lower() == "true"
#     USE_AGENTIC_RETRIEVAL = os.getenv("USE_AGENTIC_RETRIEVAL", "").lower() == "true"
#     USE_VECTORS = os.getenv("USE_VECTORS", "").lower() != "false"
#
#     # WEBSITE_HOSTNAME is always set by App Service, RUNNING_IN_PRODUCTION is set in main.bicep
#     RUNNING_ON_AZURE = os.getenv("WEBSITE_HOSTNAME") is not None or os.getenv("RUNNING_IN_PRODUCTION") is not None
#
#     # Use the current user identity for keyless authentication to Azure services.
#     # This assumes you use 'azd auth login' locally, and managed identity when deployed on Azure.
#     # The managed identity is setup in the infra/ folder.
#     azure_credential: AzureDeveloperCliCredential | ManagedIdentityCredential
#     azure_ai_token_provider: Callable[[], Awaitable[str]]
#     if RUNNING_ON_AZURE:
#         current_app.logger.info("Setting up Azure credential using ManagedIdentityCredential")
#         if AZURE_CLIENT_ID := os.getenv("AZURE_CLIENT_ID"):
#             # ManagedIdentityCredential should use AZURE_CLIENT_ID if set in env, but its not working for some reason,
#             # so we explicitly pass it in as the client ID here. This is necessary for user-assigned managed identities.
#             current_app.logger.info(
#                 "Setting up Azure credential using ManagedIdentityCredential with client_id %s", AZURE_CLIENT_ID
#             )
#             azure_credential = ManagedIdentityCredential(client_id=AZURE_CLIENT_ID)
#         else:
#             current_app.logger.info("Setting up Azure credential using ManagedIdentityCredential")
#             azure_credential = ManagedIdentityCredential()
#     elif AZURE_TENANT_ID:
#         current_app.logger.info(
#             "Setting up Azure credential using AzureDeveloperCliCredential with tenant_id %s", AZURE_TENANT_ID
#         )
#         azure_credential = AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
#     else:
#         current_app.logger.info("Setting up Azure credential using AzureDeveloperCliCredential for home tenant")
#         azure_credential = AzureDeveloperCliCredential(process_timeout=60)
#     azure_ai_token_provider = get_bearer_token_provider(
#         azure_credential, "https://cognitiveservices.azure.com/.default"
#     )
#
#     # Set the Azure credential in the app config for use in other parts of the app
#     current_app.config[CONFIG_CREDENTIAL] = azure_credential
#
#     # Set up clients for AI Search and Storage
#     search_client = SearchClient(
#         endpoint=AZURE_SEARCH_ENDPOINT,
#         index_name=AZURE_SEARCH_INDEX,
#         credential=azure_credential,
#     )
#     agent_client = KnowledgeAgentRetrievalClient(
#         endpoint=AZURE_SEARCH_ENDPOINT, agent_name=AZURE_SEARCH_AGENT, credential=azure_credential
#     )
#
#     # Set up the global blob storage manager (used for global content/images, but not user uploads)
#     global_blob_manager = BlobManager(
#         endpoint=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
#         credential=azure_credential,
#         container=AZURE_STORAGE_CONTAINER,
#         image_container=AZURE_IMAGESTORAGE_CONTAINER,
#     )
#     current_app.config[CONFIG_GLOBAL_BLOB_MANAGER] = global_blob_manager
#
#     # Set up authentication helper
#     search_index = None
#     if AZURE_USE_AUTHENTICATION:
#         current_app.logger.info("AZURE_USE_AUTHENTICATION is true, setting up search index client")
#         search_index_client = SearchIndexClient(
#             endpoint=AZURE_SEARCH_ENDPOINT,
#             credential=azure_credential,
#         )
#         search_index = await search_index_client.get_index(AZURE_SEARCH_INDEX)
#         await search_index_client.close()
#     auth_helper = AuthenticationHelper(
#         search_index=search_index,
#         use_authentication=AZURE_USE_AUTHENTICATION,
#         server_app_id=AZURE_SERVER_APP_ID,
#         server_app_secret=AZURE_SERVER_APP_SECRET,
#         client_app_id=AZURE_CLIENT_APP_ID,
#         tenant_id=AZURE_AUTH_TENANT_ID,
#         enforce_access_control=AZURE_ENFORCE_ACCESS_CONTROL,
#         enable_unauthenticated_access=AZURE_ENABLE_UNAUTHENTICATED_ACCESS,
#     )
#
#     if USE_SPEECH_OUTPUT_AZURE:
#         current_app.logger.info("USE_SPEECH_OUTPUT_AZURE is true, setting up Azure speech service")
#         if not AZURE_SPEECH_SERVICE_ID or AZURE_SPEECH_SERVICE_ID == "":
#             raise ValueError("Azure speech resource not configured correctly, missing AZURE_SPEECH_SERVICE_ID")
#         if not AZURE_SPEECH_SERVICE_LOCATION or AZURE_SPEECH_SERVICE_LOCATION == "":
#             raise ValueError("Azure speech resource not configured correctly, missing AZURE_SPEECH_SERVICE_LOCATION")
#         current_app.config[CONFIG_SPEECH_SERVICE_ID] = AZURE_SPEECH_SERVICE_ID
#         current_app.config[CONFIG_SPEECH_SERVICE_LOCATION] = AZURE_SPEECH_SERVICE_LOCATION
#         current_app.config[CONFIG_SPEECH_SERVICE_VOICE] = AZURE_SPEECH_SERVICE_VOICE
#         # Wait until token is needed to fetch for the first time
#         current_app.config[CONFIG_SPEECH_SERVICE_TOKEN] = None
#
#     openai_client, azure_openai_endpoint = setup_openai_client(
#         openai_host=OPENAI_HOST,
#         azure_credential=azure_credential,
#         azure_openai_service=AZURE_OPENAI_SERVICE,
#         azure_openai_custom_url=AZURE_OPENAI_CUSTOM_URL,
#         azure_openai_api_key=AZURE_OPENAI_API_KEY_OVERRIDE,
#         openai_api_key=OPENAI_API_KEY,
#         openai_organization=OPENAI_ORGANIZATION,
#     )
#
#     user_blob_manager = None
#     if USE_USER_UPLOAD:
#         current_app.logger.info("USE_USER_UPLOAD is true, setting up user upload feature")
#         if not AZURE_USERSTORAGE_ACCOUNT or not AZURE_USERSTORAGE_CONTAINER:
#             raise ValueError(
#                 "AZURE_USERSTORAGE_ACCOUNT and AZURE_USERSTORAGE_CONTAINER must be set when USE_USER_UPLOAD is true"
#             )
#         if not AZURE_ENFORCE_ACCESS_CONTROL:
#             raise ValueError("AZURE_ENFORCE_ACCESS_CONTROL must be true when USE_USER_UPLOAD is true")
#         user_blob_manager = AdlsBlobManager(
#             endpoint=f"https://{AZURE_USERSTORAGE_ACCOUNT}.dfs.core.windows.net",
#             container=AZURE_USERSTORAGE_CONTAINER,
#             credential=azure_credential,
#         )
#         current_app.config[CONFIG_USER_BLOB_MANAGER] = user_blob_manager
#
#         # Set up ingester
#         file_processors, figure_processor = setup_file_processors(
#             azure_credential=azure_credential,
#             document_intelligence_service=os.getenv("AZURE_DOCUMENTINTELLIGENCE_SERVICE"),
#             local_pdf_parser=os.getenv("USE_LOCAL_PDF_PARSER", "").lower() == "true",
#             local_html_parser=os.getenv("USE_LOCAL_HTML_PARSER", "").lower() == "true",
#             use_content_understanding=os.getenv("USE_CONTENT_UNDERSTANDING", "").lower() == "true",
#             content_understanding_endpoint=os.getenv("AZURE_CONTENTUNDERSTANDING_ENDPOINT"),
#             use_multimodal=USE_MULTIMODAL,
#             openai_client=openai_client,
#             openai_model=OPENAI_CHATGPT_MODEL,
#             openai_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT if OPENAI_HOST == OpenAIHost.AZURE else None,
#         )
#         search_info = setup_search_info(
#             search_service=AZURE_SEARCH_SERVICE, index_name=AZURE_SEARCH_INDEX, azure_credential=azure_credential
#         )
#
#         text_embeddings_service = None
#         if USE_VECTORS:
#             text_embeddings_service = setup_embeddings_service(
#                 open_ai_client=openai_client,
#                 openai_host=OPENAI_HOST,
#                 emb_model_name=OPENAI_EMB_MODEL,
#                 emb_model_dimensions=OPENAI_EMB_DIMENSIONS,
#                 azure_openai_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
#                 azure_openai_endpoint=azure_openai_endpoint,
#             )
#
#         image_embeddings_service = setup_image_embeddings_service(
#             azure_credential=azure_credential,
#             vision_endpoint=AZURE_VISION_ENDPOINT,
#             use_multimodal=USE_MULTIMODAL,
#         )
#         ingester = UploadUserFileStrategy(
#             search_info=search_info,
#             file_processors=file_processors,
#             embeddings=text_embeddings_service,
#             image_embeddings=image_embeddings_service,
#             search_field_name_embedding=AZURE_SEARCH_FIELD_NAME_EMBEDDING,
#             blob_manager=user_blob_manager,
#             figure_processor=figure_processor,
#         )
#         current_app.config[CONFIG_INGESTER] = ingester
#
#     image_embeddings_client = None
#     if USE_MULTIMODAL:
#         image_embeddings_client = ImageEmbeddings(AZURE_VISION_ENDPOINT, azure_ai_token_provider)
#
#     current_app.config[CONFIG_OPENAI_CLIENT] = openai_client
#     current_app.config[CONFIG_SEARCH_CLIENT] = search_client
#     current_app.config[CONFIG_AGENT_CLIENT] = agent_client
#     current_app.config[CONFIG_AUTH_CLIENT] = auth_helper
#
#     current_app.config[CONFIG_SEMANTIC_RANKER_DEPLOYED] = AZURE_SEARCH_SEMANTIC_RANKER != "disabled"
#     current_app.config[CONFIG_QUERY_REWRITING_ENABLED] = (
#         AZURE_SEARCH_QUERY_REWRITING == "true" and AZURE_SEARCH_SEMANTIC_RANKER != "disabled"
#     )
#     current_app.config[CONFIG_DEFAULT_REASONING_EFFORT] = OPENAI_REASONING_EFFORT
#     current_app.config[CONFIG_REASONING_EFFORT_ENABLED] = OPENAI_CHATGPT_MODEL in Approach.GPT_REASONING_MODELS
#     current_app.config[CONFIG_STREAMING_ENABLED] = (
#         OPENAI_CHATGPT_MODEL not in Approach.GPT_REASONING_MODELS
#         or Approach.GPT_REASONING_MODELS[OPENAI_CHATGPT_MODEL].streaming
#     )
#     current_app.config[CONFIG_VECTOR_SEARCH_ENABLED] = bool(USE_VECTORS)
#     current_app.config[CONFIG_USER_UPLOAD_ENABLED] = bool(USE_USER_UPLOAD)
#     current_app.config[CONFIG_LANGUAGE_PICKER_ENABLED] = ENABLE_LANGUAGE_PICKER
#     current_app.config[CONFIG_SPEECH_INPUT_ENABLED] = USE_SPEECH_INPUT_BROWSER
#     current_app.config[CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED] = USE_SPEECH_OUTPUT_BROWSER
#     current_app.config[CONFIG_SPEECH_OUTPUT_AZURE_ENABLED] = USE_SPEECH_OUTPUT_AZURE
#     current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED] = USE_CHAT_HISTORY_BROWSER
#     current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED] = USE_CHAT_HISTORY_COSMOS
#     current_app.config[CONFIG_AGENTIC_RETRIEVAL_ENABLED] = USE_AGENTIC_RETRIEVAL
#     current_app.config[CONFIG_MULTIMODAL_ENABLED] = USE_MULTIMODAL
#     current_app.config[CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS] = RAG_SEARCH_TEXT_EMBEDDINGS
#     current_app.config[CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS] = RAG_SEARCH_IMAGE_EMBEDDINGS
#     current_app.config[CONFIG_RAG_SEND_TEXT_SOURCES] = RAG_SEND_TEXT_SOURCES
#     current_app.config[CONFIG_RAG_SEND_IMAGE_SOURCES] = RAG_SEND_IMAGE_SOURCES
#
#     prompt_manager = PromptyManager()
#
#     # Set up the two default RAG approaches for /ask and /chat
#     # RetrieveThenReadApproach is used by /ask for single-turn Q&A
#
#     current_app.config[CONFIG_ASK_APPROACH] = RetrieveThenReadApproach(
#         search_client=search_client,
#         search_index_name=AZURE_SEARCH_INDEX,
#         agent_model=AZURE_OPENAI_SEARCHAGENT_MODEL,
#         agent_deployment=AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT,
#         agent_client=agent_client,
#         openai_client=openai_client,
#         chatgpt_model=OPENAI_CHATGPT_MODEL,
#         chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
#         embedding_model=OPENAI_EMB_MODEL,
#         embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
#         embedding_dimensions=OPENAI_EMB_DIMENSIONS,
#         embedding_field=AZURE_SEARCH_FIELD_NAME_EMBEDDING,
#         sourcepage_field=KB_FIELDS_SOURCEPAGE,
#         content_field=KB_FIELDS_CONTENT,
#         query_language=AZURE_SEARCH_QUERY_LANGUAGE,
#         query_speller=AZURE_SEARCH_QUERY_SPELLER,
#         prompt_manager=prompt_manager,
#         reasoning_effort=OPENAI_REASONING_EFFORT,
#         multimodal_enabled=USE_MULTIMODAL,
#         image_embeddings_client=image_embeddings_client,
#         global_blob_manager=global_blob_manager,
#         user_blob_manager=user_blob_manager,
#     )
#
#     # ChatReadRetrieveReadApproach is used by /chat for multi-turn conversation
#     current_app.config[CONFIG_CHAT_APPROACH] = ChatReadRetrieveReadApproach(
#         search_client=search_client,
#         search_index_name=AZURE_SEARCH_INDEX,
#         agent_model=AZURE_OPENAI_SEARCHAGENT_MODEL,
#         agent_deployment=AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT,
#         agent_client=agent_client,
#         openai_client=openai_client,
#         chatgpt_model=OPENAI_CHATGPT_MODEL,
#         chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
#         embedding_model=OPENAI_EMB_MODEL,
#         embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
#         embedding_dimensions=OPENAI_EMB_DIMENSIONS,
#         embedding_field=AZURE_SEARCH_FIELD_NAME_EMBEDDING,
#         sourcepage_field=KB_FIELDS_SOURCEPAGE,
#         content_field=KB_FIELDS_CONTENT,
#         query_language=AZURE_SEARCH_QUERY_LANGUAGE,
#         query_speller=AZURE_SEARCH_QUERY_SPELLER,
#         prompt_manager=prompt_manager,
#         reasoning_effort=OPENAI_REASONING_EFFORT,
#         multimodal_enabled=USE_MULTIMODAL,
#         image_embeddings_client=image_embeddings_client,
#         global_blob_manager=global_blob_manager,
#         user_blob_manager=user_blob_manager,
#     )