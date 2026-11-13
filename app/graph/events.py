"""Activity events streamed to the UI (Agent Activity Panel).

Nodes call `emit(...)`. Inside a graph run this goes to LangGraph's custom
stream; outside a run (unit tests) it is a no-op. Event names match
docs/logging.md so the panel, the logs and the traces read the same.
"""

from typing import Any

from langgraph.config import get_stream_writer

from app.logging import get_logger

log = get_logger("graph")


def emit(event: str, **data: Any) -> None:
    payload = {"event": event, **data}
    log.info(event, **data)
    try:
        get_stream_writer()(payload)
    except RuntimeError:
        pass  # not running inside a graph (e.g. unit test)
