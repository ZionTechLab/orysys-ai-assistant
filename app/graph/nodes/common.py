"""Shared node helpers: the failure-isolating wrapper and JSON parsing."""

import functools
import json
import re
import time
from collections.abc import Awaitable, Callable

from langgraph.runtime import Runtime

from app.graph.events import emit
from app.graph.state import AgentState, UserContext

NodeFn = Callable[[AgentState, Runtime[UserContext]], Awaitable[dict]]


def node(name: str, on_error: Callable[[AgentState, Exception], dict] | None = None):
    """Wrap a graph node so a failure is recorded in state instead of crashing the run.

    - emits node_start / node_end (activity panel, logs)
    - on exception: emits node_error, appends to state.errors, and returns the
      node's safe fallback update (on_error) so the next node can still run.
    This is how one failing agent does not take the whole answer down.
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: AgentState, runtime: Runtime[UserContext]) -> dict:
            start = time.perf_counter()
            emit("node_start", node=name)
            try:
                update = await fn(state, runtime)
                emit("node_end", node=name, status="ok", latency_ms=_ms(start))
                return update
            except Exception as exc:  # noqa: BLE001 – isolation is the point of this wrapper
                emit("node_error", node=name, error_type=type(exc).__name__, latency_ms=_ms(start))
                fallback = on_error(state, exc) if on_error else {}
                return {
                    **fallback,
                    "errors": [{"node": name, "error_type": type(exc).__name__, "message": _safe(exc)}],
                }

        return wrapper

    return decorator


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _safe(exc: Exception) -> str:
    """Short, non-sensitive error text for state / UI (full detail goes to logs only)."""
    return str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output; tolerates ``` fences and leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    return json.loads(match.group(0))
