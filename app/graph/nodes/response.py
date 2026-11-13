"""Response agent: writes the final answer from evidence only, streamed token by token."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.graph.events import emit
from app.graph.nodes.common import node
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm

HISTORY_TURNS = 6
CITATION = re.compile(r"\[([A-Z]+-[\w-]+#[\w-]+)\]")

FALLBACK_ANSWER = (
    "I'm sorry, I can't answer right now because a service I depend on is unavailable. "
    "Please try again in a few minutes."
)


def _fallback(_state: AgentState, _exc: Exception) -> dict:
    return {"messages": [AIMessage(FALLBACK_ANSWER)]}


def _documents_block(evidence: list[dict]) -> str:
    if not evidence:
        return "No documents matched this question."
    return "\n\n".join(
        f'<document id="{e["chunk_id"]}" title="{e["title"]}" type="{e["document_type"]}" date="{e["created_date"]}">\n'
        f'{e["text"]}\n</document>'
        for e in evidence
    )


def _notes(state: AgentState) -> str:
    degraded = state.get("degraded") or {}
    if degraded.get("retrieval_unavailable"):
        return "\nNOTE: document search is unavailable right now. Tell the user plainly that you could not search."
    return ""


@node("response_agent", on_error=_fallback)
async def response_agent(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    plan = state.get("plan", {})
    history = state["messages"][-HISTORY_TURNS:]

    if plan.get("route") == "chitchat":
        system = prompts.CHITCHAT
    else:
        system = f"{prompts.RESPONSE}{_notes(state)}\n\nDocuments:\n{_documents_block(state.get('evidence', []))}"

    emit("response_generation", route=plan.get("route"), evidence=len(state.get("evidence", [])))
    # Streaming model: tokens reach the UI through LangGraph's "messages" stream as they are generated.
    reply = await get_llm("strong", streaming=True).ainvoke([SystemMessage(system), *history[:-1],
                                                              HumanMessage(history[-1].content)])
    citations = sorted(set(CITATION.findall(reply.content)))
    emit("response_done", citations=citations, chars=len(reply.content))
    return {"messages": [AIMessage(reply.content)]}
