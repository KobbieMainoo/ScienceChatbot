import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from quart import abort, current_app, request

from config import CONFIG_AUTH_CLIENT, CONFIG_SEARCH_CLIENT
from core.authentication import AuthError
from error import error_response

def authenticated_path(route_fn: Callable[[str, dict[str, Any]], Any]):
    """
    本地开发版：不做任何访问控制，所有请求直接放行。
    """

    @wraps(route_fn)
    async def auth_handler(path: str = ""):
        # 永远允许访问，传入空的 auth_claims
        auth_claims: dict[str, Any] = {}
        return await route_fn(path, auth_claims)

    return auth_handler


_C = TypeVar("_C", bound=Callable[..., Any])


def authenticated(route_fn: _C) -> _C:
    """
    本地开发版：不做认证检查，始终传入空的 auth_claims
    """

    @wraps(route_fn)
    async def auth_handler(*args, **kwargs):
        auth_claims: dict[str, Any] = {}
        return await route_fn(auth_claims, *args, **kwargs)

    return cast(_C, auth_handler)

# def authenticated_path(route_fn: Callable[[str, dict[str, Any]], Any]):
#     """
#     Decorator for routes that request a specific file that might require access control enforcement
#     """
#
#     @wraps(route_fn)
#     async def auth_handler(path=""):
#         # If authentication is enabled, validate the user can access the file
#         auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
#         search_client = current_app.config[CONFIG_SEARCH_CLIENT]
#         authorized = False
#         try:
#             auth_claims = await auth_helper.get_auth_claims_if_enabled(request.headers)
#             authorized = await auth_helper.check_path_auth(path, auth_claims, search_client)
#         except AuthError:
#             abort(403)
#         except Exception as error:
#             logging.exception("Problem checking path auth %s", error)
#             return error_response(error, route="/content")
#
#         if not authorized:
#             abort(403)
#
#         return await route_fn(path, auth_claims)
#
#     return auth_handler
#
#
# _C = TypeVar("_C", bound=Callable[..., Any])
#
#
# def authenticated(route_fn: _C) -> _C:
#     """
#     Decorator for routes that might require access control. Unpacks Authorization header information into an auth_claims dictionary
#     """
#
#     @wraps(route_fn)
#     async def auth_handler(*args, **kwargs):
#         auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
#         try:
#             auth_claims = await auth_helper.get_auth_claims_if_enabled(request.headers)
#         except AuthError:
#             abort(403)
#
#         return await route_fn(auth_claims, *args, **kwargs)
#
#     return cast(_C, auth_handler)
