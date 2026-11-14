"""Supervisor: understands intent and routes. Returns validated JSON, not free text."""

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, field_validator

from app.graph.events import emit
from app.graph.nodes.common import invoke_json, node
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.memory.summary import conversation_context
from app.retrieval.hybrid import NAMESPACES

DOC_TYPES = ("incident", "runbook", "architecture", "policy", "product_spec", "meeting_notes")


class SupervisorDecision(BaseModel):
    intent: str = Field(max_length=300)
    route: str
    search_query: str = Field(default="", max_length=500)
    departments: list[str] = []
    document_types: list[str] = []
    created_from: str = ""
    created_to: str = ""
    reasoning: str = Field(default="", max_length=500)

    @field_validator("route")
    @classmethod
    def _route(cls, v: str) -> str:
        return v if v in ("retrieve", "research", "tool", "chitchat") else "retrieve"  # unknown -> safest useful default

    @field_validator("departments")
    @classmethod
    def _departments(cls, v: list[str]) -> list[str]:
        return [d for d in v if d in NAMESPACES]  # drop anything the model invented

    @field_validator("document_types")
    @classmethod
    def _types(cls, v: list[str]) -> list[str]:
        return [t for t in v if t in DOC_TYPES]

    @field_validator("created_from", "created_to")
    @classmethod
    def _date(cls, v: str) -> str:
        try:
            return date.fromisoformat(v).isoformat() if v else ""
        except ValueError:
            return ""


def _fallback(state: AgentState, _exc: Exception) -> dict:
    """If routing fails, still search with the user's own words."""
    question = state["messages"][-1].content
    return {"plan": {"intent": "unknown (supervisor failed)", "route": "retrieve", "search_query": question}}


@node("supervisor", on_error=_fallback)
async def supervisor(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    # Memory: summary of older turns + recent messages verbatim, to resolve follow-up questions.
    summary, recent = conversation_context(state["messages"], state.get("summary"))
    convo = "\n".join(f"{m.type}: {m.content}" for m in recent)
    system = prompts.SUPERVISOR.replace("{today}", date.today().isoformat())
    user = runtime.context
    context = f"User: role {user.role}, department {user.department or 'unknown'}.\n"
    if summary:
        context += f"Summary of earlier conversation:\n{summary}\n"

    decision = await invoke_json(
        get_llm("strong"), [SystemMessage(system), HumanMessage(f"{context}\nConversation (latest last):\n{convo}")],
        SupervisorDecision, node="supervisor")
    if decision.route == "retrieve" and not decision.search_query:
        decision.search_query = state["messages"][-1].content

    plan = decision.model_dump()
    emit("route_selected", route=plan["route"], intent=plan["intent"],
         departments=plan["departments"], document_types=plan["document_types"],
         created_from=plan["created_from"], created_to=plan["created_to"])
    return {"plan": plan}
