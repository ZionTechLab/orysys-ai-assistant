"""Application errors and global exception handlers.

Every error response has the same shape so the UI can show it consistently:
    {"error": {"code": "...", "message": "...", "request_id": "..."}}
Internal details (stack traces, provider messages) are logged, never returned.
"""

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging import get_logger

log = get_logger(__name__)


class AppError(Exception):
    """Base error with a stable code and an HTTP status."""

    status_code = 500
    code = "internal_error"
    message = "Something went wrong. Please try again."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message


class InvalidRequestError(AppError):
    status_code, code, message = 400, "invalid_request", "The request is not valid."


class AuthenticationError(AppError):
    status_code, code, message = 401, "unauthenticated", "Please log in."


class PermissionDeniedError(AppError):
    status_code, code, message = 403, "permission_denied", "You do not have access to this action."


class RateLimitedError(AppError):
    status_code, code, message = 429, "rate_limited", "Too many requests. Please wait and try again."

    def __init__(self, retry_after_seconds: int, message: str | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class UpstreamError(AppError):
    """A dependency (LLM, Pinecone, MCP) failed."""

    status_code, code, message = 503, "upstream_unavailable", "A dependent service is unavailable."


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _error_body(code: str, message: str, request: Request, details: list | None = None) -> dict:
    body = {"code": code, "message": message, "request_id": _request_id(request)}
    if details:
        body["details"] = details
    return {"error": body}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log.warning("app_error", code=exc.code, status=exc.status_code, message=exc.message)
        headers = {}
        if isinstance(exc, RateLimitedError):
            headers["Retry-After"] = str(exc.retry_after_seconds)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, request),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Return field + reason only; never echo the submitted input back.
        details = [{"field": ".".join(map(str, e["loc"])), "reason": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content=_error_body("validation_error", "Some fields are not valid.", request, details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body("http_error", str(exc.detail), request),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error_type=type(exc).__name__, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_error_body(AppError.code, AppError.message, request),
        )
