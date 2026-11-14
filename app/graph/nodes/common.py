"""Shared node helpers: the failure-isolating wrapper, JSON parsing and validated JSON calls."""

import functools
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from app.graph.events import emit
from app.graph.state import AgentState, UserContext

NodeFn = Callable[[AgentState, Runtime[UserContext]], Awaitable[dict | Command]]
M = TypeVar("M", bound=BaseModel)

JSON_ATTEMPTS = 3
JSON_RETRY = ("Your previous reply could not be used ({error}). Reply again with only the JSON object described "
              "in the instructions – all required fields, no prose, no code fences.")


async def invoke_json(llm, messages: list[BaseMessage], schema: type[M], node: str) -> M:
    """Ask for JSON, validate it against `schema`, and ask again (with the reason) when it is unusable.

    Seen live (14 Nov): the gateway sometimes returns truncated replies – '{"tool-index to preview…"}' or
    an object missing a required key. One bad reply must not decide the route. Network errors are not
    retried here (the client already retries them); after the last attempt the error goes to the node's
    fallback as before.
    """
    error: Exception | None = None
    for attempt in range(1, JSON_ATTEMPTS + 1):
        reply = await llm.ainvoke(messages)
        try:
            return schema.model_validate(parse_json(reply.content))
        except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
            error = exc
            emit("llm_output_invalid", node=node, attempt=attempt, error_type=type(exc).__name__,
                 retry=attempt < JSON_ATTEMPTS)
            reason = str(exc).splitlines()[0][:200]
            messages = [*messages, AIMessage(str(reply.content)[:2000]), HumanMessage(JSON_RETRY.format(error=reason))]
    raise error


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
            except GraphBubbleUp:
                raise  # interrupt() for human approval is control flow, not a failure
            except Exception as exc:  # noqa: BLE001 – isolation is the point of this wrapper
                emit("node_error", node=name, error_type=type(exc).__name__, latency_ms=_ms(start))
                fallback = on_error(state, exc) if on_error else {}
                error = [{"node": name, "error_type": type(exc).__name__, "message": _safe(exc)}]
                if isinstance(fallback, Command):  # nodes that route themselves keep routing on failure
                    return Command(goto=fallback.goto, update={**(fallback.update or {}), "errors": error})
                return {**fallback, "errors": error}

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
