"""/chat – runs the agent graph and streams progress + answer as Server-Sent Events.

Event types (see docs/architecture.md D3):
  node_start, node_end, node_error, route_selected, retrieval, response_generation,
  response_done            – agent activity (from nodes, via LangGraph custom stream)
  token                    – a piece of the answer
  memory_update            – the turn was saved; message count of the conversation
  error                    – the turn failed; message is safe to show
  done                     – final summary: thread_id, route, citations, run_id
"""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.errors import InvalidRequestError
from app.graph.state import UserContext, new_turn_state
from app.logging import bind_context, fingerprint, get_logger
from app.memory.checkpointer import thread_key
from app.security.auth import User, get_current_user

router = APIRouter(tags=["chat"])
log = get_logger(__name__)

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, default=str)}


def _graph_config(user: User, thread_id: str, request_id: str, run_id: uuid.UUID) -> dict:
    return {
        "configurable": {"thread_id": thread_key(user.user_id, thread_id)},
        # LangSmith: one trace per turn, searchable by request id, user and role.
        "run_id": run_id,
        "run_name": "chat_turn",
        "tags": [f"role:{user.role.value}"],
        "metadata": {"request_id": request_id, "user_id": user.user_id, "role": user.role.value,
                     "thread_id": thread_id},
    }


async def _stream_turn(request: Request, user: User, body: ChatRequest) -> AsyncIterator[dict]:
    graph = request.app.state.graph
    request_id = request.state.request_id
    run_id = uuid.uuid4()
    context = UserContext(user_id=user.user_id, role=user.role.value, clearance=tuple(user.clearance),
                          display_name=user.display_name, request_id=request_id)
    config = _graph_config(user, body.thread_id, request_id, run_id)
    summary: dict = {"route": None, "citations": []}

    try:
        async for mode, chunk in graph.astream(new_turn_state(body.message), config=config, context=context,
                                                stream_mode=["custom", "messages"]):
            if await request.is_disconnected():
                log.info("client_disconnected")
                return
            if mode == "custom":
                event = chunk.get("event", "activity")
                if event == "route_selected":
                    summary["route"] = chunk.get("route")
                elif event == "response_done":
                    summary["citations"] = chunk.get("citations", [])
                yield _sse(event, chunk)
            elif mode == "messages":
                msg, meta = chunk
                # Only streamed pieces of the final answer; the supervisor's JSON is not shown as answer text.
                if isinstance(msg, AIMessageChunk) and meta.get("langgraph_node") == "response_agent" and msg.content:
                    yield _sse("token", {"text": msg.content})
    except Exception as exc:  # noqa: BLE001 – the stream must end with a clean error event
        log.error("chat_turn_failed", error_type=type(exc).__name__, exc_info=exc)
        yield _sse("error", {"message": "Something went wrong while answering. Please try again.",
                             "request_id": request_id})
        return

    # Memory: the checkpointer has saved this turn; report what the conversation now holds.
    snapshot = await graph.aget_state(config)
    n_messages = len((snapshot.values or {}).get("messages", [])) if snapshot else 0
    yield _sse("memory_update", {"event": "memory_update", "thread_id": body.thread_id, "messages": n_messages})

    yield _sse("done", {"thread_id": body.thread_id, "request_id": request_id, "run_id": str(run_id), **summary})


@router.post("/chat")
async def chat(request: Request, body: ChatRequest, user: User = Depends(get_current_user)) -> EventSourceResponse:
    bind_context(thread_id=body.thread_id)
    log.info("chat_request", **fingerprint(body.message))
    return EventSourceResponse(_stream_turn(request, user, body), ping=15)


@router.get("/traces/{run_id}")
async def trace_url(run_id: uuid.UUID, user: User = Depends(get_current_user)) -> dict:
    """Link to the LangSmith trace of a turn (shown in the activity panel)."""
    import asyncio

    from langsmith import Client

    def lookup() -> str:
        client = Client()
        run = client.read_run(str(run_id))  # deprecated after Jan 2027 → client.runs.retrieve(project_id=...)
        if (run.extra or {}).get("metadata", {}).get("user_id") != user.user_id:
            raise PermissionError
        return client.get_run_url(run=run)

    try:
        return {"run_id": str(run_id), "url": await asyncio.to_thread(lookup)}
    except Exception:  # noqa: BLE001 – not found yet, other user's run, or LangSmith down
        return {"run_id": str(run_id), "url": None}


@router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str, user: User = Depends(get_current_user)) -> dict:
    """Conversation history for the logged-in user's thread only."""
    import re

    if not re.match(THREAD_ID_PATTERN, thread_id):
        raise InvalidRequestError("Invalid thread id.")
    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_key(user.user_id, thread_id)}})
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    return {
        "thread_id": thread_id,
        "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages],
    }
