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

Route = Literal["retrieve", "research", "tool", "chitchat", "blocked"]


@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str
    clearance: tuple[str, ...]
    display_name: str = ""
    request_id: str = ""
    department: str = ""


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


def _merge_dict(old: dict | None, new: dict | None) -> dict:
    """Parallel workers may each set a flag; merge instead of overwrite. An empty dict resets."""
    if new == {}:
        return {}
    return {**(old or {}), **(new or {})}


def _findings(old: list | None, new: list | None) -> list:
    """Findings from parallel research workers are appended – but only within one turn.

    Each finding carries the turn id. When findings of a new turn arrive, those of
    older turns are dropped, so the checkpoint does not grow turn after turn.
    """
    if not new:
        return old or []
    turn = new[0]["turn_id"]
    return [f for f in (old or []) if f.get("turn_id") == turn] + new


class Finding(TypedDict, total=False):
    """What a research worker extracted from one document."""

    turn_id: str
    doc_id: str
    title: str
    date: str
    summary: str
    cause_category: str
    cause: str
    impact: str
    citations: list[str]
    sources: list[dict]  # sections the worker read: chunk_id, title, section, text (capped)
    batch_no: int
    error: str  # set instead of the fields above when the document could not be analysed


class Research(TypedDict, total=False):
    turn_id: str
    question: str
    plan_code: str
    plan_attempts: int
    used_fallback: bool
    doc_ids: list[str]
    batches: list[list[str]]
    themes: list[dict]  # filled by the aggregator: theme, doc_ids, count, explanation


class AgentState(TypedDict, total=False):
    # Conversation – appended, persisted across turns
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn working data – replaced every turn
    plan: Annotated[Plan, _replace]
    evidence: Annotated[list[Evidence], _replace]
    research: Annotated[Research, _replace]
    findings: Annotated[list[Finding], _findings]
    tool_call: Annotated[dict, _replace]    # {"tool", "args", "reason", "approved", "approved_by"}
    tool_result: Annotated[dict, _replace]  # {"tool", "status", "data", "message", "latency_ms"}
    guard: Annotated[dict, _replace]        # input guard verdict
    validation: Annotated[dict, _replace]   # validator verdict: ok, action, attempt, problems, grounding
    output_guard: Annotated[dict, _replace]  # output guard verdict: actions, brand
    # Memory – rolling summary of older turns; persists across turns (not reset by new_turn_state)
    summary: Annotated[dict, _replace]       # {"text", "covered": messages folded in, "turns": updates so far}
    # Failures recorded by nodes instead of crashing the graph
    errors: Annotated[list[NodeError], _append]
    degraded: Annotated[dict[str, bool], _merge_dict]


def new_turn_state(question: str) -> dict:
    """Input for one user turn: the new message, and per-turn fields reset."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(question)],
        "plan": {},
        "evidence": [],
        "research": {},
        "tool_call": {},
        "tool_result": {},
        "guard": {},
        "validation": {},
        "output_guard": {},
        "degraded": {},
    }


@dataclass
class TurnSummary:
    """Small, UI-friendly summary of a finished turn."""

    route: str = ""
    citations: list[str] = field(default_factory=list)
    errors: list[NodeError] = field(default_factory=list)
