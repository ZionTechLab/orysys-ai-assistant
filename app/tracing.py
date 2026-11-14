"""LangSmith spans for work that is neither an LLM call nor a LangGraph node.

LangGraph already traces every node, routing function and model call, and hybrid search has its
own `retriever` span. These add typed child spans for the rest, so a trace shows *what ran*:

  tool:<name>   run_type "tool"  – arguments, status (ok / denied / rejected / timeout ...), RBAC result
  mcp:<name>    run_type "tool"  – the call to the MCP server inside the tool span
  rlm_sandbox   run_type "chain" – the Python plan the sandbox executed and the documents it selected

Spans are created only when LangSmith tracing is on; with it off (tests) they cost nothing.
Data shown in a span is truncated – a trace is for inspecting, not a copy of every result.
"""

import json
from typing import Any

import langsmith as ls

MAX_TRACE_CHARS = 4000


def preview(value: Any, limit: int = MAX_TRACE_CHARS) -> Any:
    """Small values as they are; large ones as truncated JSON text."""
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + " …[truncated]"


def span(name: str, run_type: str = "chain", inputs: dict | None = None, metadata: dict | None = None,
         tags: list[str] | None = None):
    """Child span of the current trace (context manager). Call `.end(outputs=..., error=...)` to record the
    result; an exception inside the block is recorded as the span's error automatically."""
    return ls.trace(name, run_type=run_type, inputs=inputs or {}, metadata=metadata or {}, tags=tags or [])
