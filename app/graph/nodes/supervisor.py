"""Supervisor: understands intent and routes. Returns validated JSON, not free text."""

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, field_validator

from app.graph.events import emit
from app.graph.nodes.common import node, parse_json
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.retrieval.hybrid import NAMESPACES

DOC_TYPES = ("incident", "runbook", "architecture", "policy", "product_spec", "meeting_notes")
HISTORY_TURNS = 6  # recent messages given to the supervisor to resolve follow-up questions


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
        return v if v in ("retrieve", "chitchat") else "retrieve"  # unknown route -> safest useful default

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
    history = state["messages"][-HISTORY_TURNS:]
    convo = "\n".join(f"{m.type}: {m.content}" for m in history)
    system = prompts.SUPERVISOR.replace("{today}", date.today().isoformat())

    reply = await get_llm("strong").ainvoke(
        [SystemMessage(system), HumanMessage(f"Conversation (latest last):\n{convo}")]
    )
    decision = SupervisorDecision.model_validate(parse_json(reply.content))
    if decision.route == "retrieve" and not decision.search_query:
        decision.search_query = state["messages"][-1].content

    plan = decision.model_dump()
    emit("route_selected", route=plan["route"], intent=plan["intent"],
         departments=plan["departments"], document_types=plan["document_types"],
         created_from=plan["created_from"], created_to=plan["created_to"])
    return {"plan": plan}
