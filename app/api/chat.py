"""/chat – runs the agent graph and streams progress + answer as Server-Sent Events.

Event types (see docs/architecture.md D3):
  node_start, node_end, node_error, route_selected, retrieval, response_generation,
  response_done            – agent activity (from nodes, via LangGraph custom stream)
  token                    – a piece of the answer
  memory_update            – the turn was saved; message count of the conversation
  approval_required        – the run is paused until the user approves or rejects a tool call
                             (POST /chat/resume continues it)
  error                    – the turn failed; message is safe to show
  done                     – final summary: thread_id, route, citations, run_id, status
"""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.api.feedback import remember_run
from app.errors import AppError, InvalidRequestError, PermissionDeniedError
from app.graph.state import UserContext, new_turn_state
from app.logging import bind_context, fingerprint, get_logger
from app.memory.checkpointer import thread_key
from app.security.auth import User, get_current_user
from app.security.rate_limit import CHAT_LIMITER, enforce, limit_for_role
from app.tools.registry import REGISTRY, ToolContext

router = APIRouter(tags=["chat"])
log = get_logger(__name__)

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)


class ResumeRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    approved: bool


async def rate_limited_user(user: User = Depends(get_current_user)) -> User:
    """Authenticate, then take one token from the user's bucket – before any agent or LLM work."""
    await enforce(CHAT_LIMITER, user.user_id, limit_for_role(user.role.value), scope="chat")
    return user


class ConflictError(AppError):
    status_code, code, message = 409, "approval_pending", "Approve or reject the pending action first."


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


def _context(user: User, request_id: str) -> UserContext:
    return UserContext(user_id=user.user_id, role=user.role.value, clearance=tuple(user.clearance),
                       display_name=user.display_name, request_id=request_id, department=user.department)


async def _pending_approval(graph, user: User, thread_id: str) -> dict | None:
    """The interrupt payload if this user's thread is paused waiting for approval."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_key(user.user_id, thread_id)}})
    interrupts = getattr(snapshot, "interrupts", ()) if snapshot else ()
    return dict(interrupts[0].value) if interrupts else None


async def _stream_turn(request: Request, user: User, thread_id: str, graph_input) -> AsyncIterator[dict]:
    """Run (or resume) the graph and stream its events. Used by /chat and /chat/resume."""
    graph = request.app.state.graph
    request_id = request.state.request_id
    run_id = uuid.uuid4()
    remember_run(run_id, user.user_id)  # so only this user can rate this answer (POST /feedback)
    config = _graph_config(user, thread_id, request_id, run_id)
    summary: dict = {"route": None, "citations": []}
    streamed_tokens = 0

    try:
        async for mode, chunk in graph.astream(graph_input, config=config, context=_context(user, request_id),
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
                    streamed_tokens += 1
                    yield _sse("token", {"text": msg.content})
    except Exception as exc:  # noqa: BLE001 – the stream must end with a clean error event
        log.error("chat_turn_failed", error_type=type(exc).__name__, exc_info=exc)
        yield _sse("error", {"message": "Something went wrong while answering. Please try again.",
                             "request_id": request_id})
        return

    pending = await _pending_approval(graph, user, thread_id)
    if pending:
        # Human-in-the-loop: the run is paused (state saved by the checkpointer) until /chat/resume.
        log.info("approval_required", tool=pending.get("tool"))
        yield _sse("approval_required", {"event": "approval_required", "thread_id": thread_id, **pending})
        yield _sse("done", {"thread_id": thread_id, "request_id": request_id, "run_id": str(run_id),
                            "status": "awaiting_approval", **summary})
        return

    # Memory: the checkpointer has saved this turn; report what the conversation now holds.
    snapshot = await graph.aget_state(config)
    values = (snapshot.values or {}) if snapshot else {}
    messages = values.get("messages", [])
    n_messages = len(messages)
    memory = values.get("summary") or {}  # rolling conversation summary (not the turn `summary` above)
    if not streamed_tokens and messages and messages[-1].type == "ai":
        # Answers not produced by a streaming model call (guard refusal, fallback message) are sent whole.
        yield _sse("token", {"text": messages[-1].content})
    yield _sse("memory_update", {"event": "memory_update", "thread_id": thread_id, "messages": n_messages,
                                 "summarized": memory.get("covered", 0), "summary_chars": len(memory.get("text", ""))})
    yield _sse("done", {"thread_id": thread_id, "request_id": request_id, "run_id": str(run_id),
                        "status": "complete", **summary})


@router.post("/chat")
async def chat(request: Request, body: ChatRequest, user: User = Depends(rate_limited_user)) -> EventSourceResponse:
    bind_context(thread_id=body.thread_id)
    log.info("chat_request", **fingerprint(body.message))
    if await _pending_approval(request.app.state.graph, user, body.thread_id):
        raise ConflictError()
    return EventSourceResponse(_stream_turn(request, user, body.thread_id, new_turn_state(body.message)), ping=15)


@router.post("/chat/resume")
async def resume(request: Request, body: ResumeRequest,
                 user: User = Depends(rate_limited_user)) -> EventSourceResponse:
    """Approve or reject the tool call the run is paused on, then continue streaming."""
    bind_context(thread_id=body.thread_id)
    pending = await _pending_approval(request.app.state.graph, user, body.thread_id)
    if not pending:
        raise InvalidRequestError("There is no action waiting for approval in this conversation.")
    try:  # the approver must be allowed to run the tool themselves
        REGISTRY.check(pending.get("tool", ""), ToolContext(user.user_id, user.role.value, tuple(user.clearance)))
    except (KeyError, PermissionError):
        raise PermissionDeniedError("You are not allowed to approve this action.") from None
    log.info("approval_decision", tool=pending.get("tool"), approved=body.approved)
    decision = Command(resume={"approved": body.approved, "by": user.user_id})
    return EventSourceResponse(_stream_turn(request, user, body.thread_id, decision), ping=15)


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
    values = snapshot.values if snapshot and snapshot.values else {}
    messages = values.get("messages", [])
    return {
        "thread_id": thread_id,
        "summary": (values.get("summary") or {}).get("text", ""),
        "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in messages],
    }
