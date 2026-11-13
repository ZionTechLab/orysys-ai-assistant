"""Shared graph state and per-run context.

State  = what agents read and write while answering (saved by the checkpointer,
         so it survives across turns of a conversation).
Context = who is asking (user id, role, clearance). Passed fresh on every run
         from the verified JWT and never stored in state, so nothing an agent
         writes – and nothing in an old checkpoint – can change a user's access.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

Route = Literal["retrieve", "chitchat"]


@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str
    clearance: tuple[str, ...]
    display_name: str = ""
    request_id: str = ""


class Plan(TypedDict, total=False):
    intent: str
    route: Route
    search_query: str
    departments: list[str]
    document_types: list[str]
    created_from: str  # ISO date
    created_to: str
    reasoning: str


class Evidence(TypedDict):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    document_type: str
    created_date: str


class NodeError(TypedDict):
    node: str
    error_type: str
    message: str


def _replace(_old: Any, new: Any) -> Any:
    return new


def _append(old: list | None, new: list | None) -> list:
    return (old or []) + (new or [])


class AgentState(TypedDict, total=False):
    # Conversation – appended, persisted across turns
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn working data – replaced every turn
    plan: Annotated[Plan, _replace]
    evidence: Annotated[list[Evidence], _replace]
    # Failures recorded by nodes instead of crashing the graph
    errors: Annotated[list[NodeError], _append]
    degraded: Annotated[dict[str, bool], _replace]


def new_turn_state(question: str) -> dict:
    """Input for one user turn: the new message, and per-turn fields reset."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(question)],
        "plan": {},
        "evidence": [],
        "degraded": {},
    }


@dataclass
class TurnSummary:
    """Small, UI-friendly summary of a finished turn."""

    route: str = ""
    citations: list[str] = field(default_factory=list)
    errors: list[NodeError] = field(default_factory=list)
