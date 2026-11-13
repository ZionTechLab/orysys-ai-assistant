"""FastAPI application factory."""

import logging
import uuid

from fastapi import FastAPI, Request

from app.api import health
from app.config import get_settings
from app.errors import register_exception_handlers


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)  # replaced by structlog in task 1.2

    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        # Every request gets an id; it is returned in errors and (later) in logs and traces.
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    register_exception_handlers(app)
    app.include_router(health.router)
    return app


app = create_app()
