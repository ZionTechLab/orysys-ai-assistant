"""FastAPI application factory."""

import time
import uuid

from fastapi import FastAPI, Request

from app.api import auth, health
from app.config import get_settings
from app.errors import register_exception_handlers
from app.logging import bind_context, clear_context, configure_logging, get_logger

log = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # Every request gets an id. It is bound to all log lines, returned in the
        # X-Request-ID header and in error bodies, and attached to LangSmith traces.
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        clear_context()
        bind_context(request_id=request_id)

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=status,
                latency_ms=round((time.perf_counter() - start) * 1000),
                user_id=getattr(request.state, "user_id", None),
                role=getattr(request.state, "role", None),
            )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    return app


app = create_app()
