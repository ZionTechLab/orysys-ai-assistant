"""Structured logging (structlog). See docs/logging.md.

- JSON lines to stdout (console renderer when LOG_FORMAT=console).
- Per-request context (request_id, user_id, role, thread_id) is bound with
  contextvars and added to every line automatically.
- Secrets are redacted; message content is never logged (use `fingerprint`).
"""

import hashlib
import logging
import sys

import structlog

# Field names whose values must never appear in logs.
_SECRET_KEYS = {"password", "token", "access_token", "api_key", "authorization", "secret", "jwt"}


def _redact_secrets(_, __, event_dict: dict) -> dict:
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        _redact_secrets,
    ]
    renderer = (
        structlog.dev.ConsoleRenderer()
        if fmt == "console"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route third-party libraries (uvicorn, httpx) through the same format.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # httpx logs every outbound URL at INFO; keep it at WARNING to avoid noise.
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Our middleware logs requests; uvicorn's access log would duplicate them.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


def bind_context(**values) -> None:
    """Add fields to every log line for the rest of the current request."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def fingerprint(text: str) -> dict:
    """Describe user text without logging it: length + short hash to spot repeats."""
    return {"msg_len": len(text), "msg_hash": hashlib.sha256(text.encode()).hexdigest()[:12]}
