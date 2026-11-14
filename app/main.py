"""FastAPI application factory."""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api import admin, auth, chat, feedback, health
from app.config import Settings, get_settings
from app.errors import register_exception_handlers
from app.graph.builder import build_graph
from app.logging import bind_context, clear_context, configure_logging, get_logger
from app.memory.checkpointer import open_checkpointer

log = get_logger(__name__)


def _configure_langsmith(s: Settings) -> None:
    """LangSmith reads its settings from environment variables; copy them from .env-backed settings."""
    key = s.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_TRACING"] = "true" if (s.langsmith_tracing and key) else "false"
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    if key:
        os.environ["LANGSMITH_API_KEY"] = key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One checkpointer (SQLite memory) and one compiled graph for the whole process.
    async with open_checkpointer() as checkpointer:
        app.state.graph = build_graph(checkpointer)
        log.info("app_started", langsmith_tracing=os.environ.get("LANGSMITH_TRACING"))
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    _configure_langsmith(settings)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

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
    app.include_router(chat.router)
    app.include_router(feedback.router)
    app.include_router(admin.router)
    return app


app = create_app()
