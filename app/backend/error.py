import logging

from openai import APIError
from quart import jsonify

ERROR_MESSAGE = """The app encountered an error processing your request.
If you are an administrator of the app, check the application logs for a full traceback.
Error type: {error_type}
"""
ERROR_MESSAGE_FILTER = """Your message contains content that was flagged by the OpenAI content filter."""

ERROR_MESSAGE_LENGTH = """Your message exceeded the context length limit for this OpenAI model. Please shorten your message or change your settings to retrieve fewer search results."""


def error_dict(error: Exception) -> dict:
    if isinstance(error, APIError) and error.code == "content_filter":
        return {"error": ERROR_MESSAGE_FILTER}
    if isinstance(error, APIError) and error.code == "context_length_exceeded":
        return {"error": ERROR_MESSAGE_LENGTH}
    return {"error": ERROR_MESSAGE.format(error_type=type(error))}

def error_response(error: Exception, route: str, status_code: int = 500):
    """
    为了防止前端白屏，必须返回一个完整的 ChatResponse 结构：
    {
        "answer": "...",
        "context": {
            "data_points": [],
            "thoughts": [...]
        },
        "session_state": {},
        "error": "...",
        "message": "..."
    }
    """

    logging.exception("Exception in %s: %s", route, error)

    error_type = type(error).__name__
    error_message = str(error)

    # content-filter 特殊逻辑保留
    if isinstance(error, APIError) and error.code == "content_filter":
        status_code = 400

    return (
        jsonify(
            {
                "answer": f"[后端异常：{error_type}] {error_message}",
                "context": {
                    "data_points": [],
                    "thoughts": [
                        {
                            "title": "Backend Error",
                            "description": f"{route} 发生后端异常：{error_type}: {error_message}",
                            "props": None,
                        }
                    ],
                },
                "session_state": {},

                # 额外返回原有字段（调试用）
                "error": error_type,
                "message": error_message,
            }
        ),
        status_code,
    )

# def error_response(error: Exception, route: str, status_code: int = 500):
#     logging.exception("Exception in %s: %s", route, error)
#     if isinstance(error, APIError) and error.code == "content_filter":
#         status_code = 400
#     return jsonify(error_dict(error)), status_code
