"""Response agent: writes the final answer from evidence only, streamed token by token."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.graph.events import emit
from app.graph.nodes.common import node
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.security.guards import sanitize_retrieved

HISTORY_TURNS = 6
CITATION = re.compile(r"\[([A-Z]+-[\w-]+#[\w-]+)\]")

FALLBACK_ANSWER = (
    "I'm sorry, I can't answer right now because a service I depend on is unavailable. "
    "Please try again in a few minutes."
)


def _fallback(_state: AgentState, _exc: Exception) -> dict:
    return {"messages": [AIMessage(FALLBACK_ANSWER)]}


def _documents_block(evidence: list[dict]) -> str:
    """Evidence wrapped as untrusted data. Text was already sanitised where it was retrieved."""
    if not evidence:
        return "No documents matched this question."
    return "\n\n".join(
        f'<document id="{e["chunk_id"]}" title="{e["title"]}" type="{e["document_type"]}" date="{e["created_date"]}">\n'
        f'{sanitize_retrieved(e["text"]).text}\n</document>'
        for e in evidence
    )


def _notes(state: AgentState) -> str:
    degraded = state.get("degraded") or {}
    if degraded.get("retrieval_unavailable"):
        return "\nNOTE: document search is unavailable right now. Tell the user plainly that you could not search."
    if degraded.get("research_unavailable"):
        return "\nNOTE: the multi-document research could not be completed. Tell the user plainly."
    return ""


@node("response_agent", on_error=_fallback)
async def response_agent(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    from app.graph.nodes.validator import validation_feedback

    plan = state.get("plan", {})
    messages = state["messages"]
    # On a validation retry the last message is our rejected answer: replace it (same id), and build
    # the prompt from the conversation before it.
    retry_of = messages[-1] if isinstance(messages[-1], AIMessage) else None
    if retry_of is not None:
        messages = messages[:-1]
    history = messages[-HISTORY_TURNS:]

    if plan.get("route") == "blocked":
        # Input guard blocked the message: fixed refusal, no model call – nothing to manipulate.
        emit("response_done", citations=[], chars=len(prompts.REFUSAL), refusal=True)
        return {"messages": [AIMessage(prompts.REFUSAL)]}

    if plan.get("route") == "chitchat":
        system = prompts.CHITCHAT
    else:
        addendum = ""
        if plan.get("route") == "tool":
            import json

            addendum = prompts.TOOL_RESPONSE_ADDENDUM.replace(
                "{tool_result}", json.dumps(state.get("tool_result") or {}, default=str)[:9000])
        elif plan.get("route") == "research":
            from app.graph.nodes.research import research_block

            addendum = prompts.RESEARCH_RESPONSE_ADDENDUM.format(research_block=research_block(state))
        system = (f"{prompts.RESPONSE}{_notes(state)}{addendum}{validation_feedback(state)}\n\n"
                  f"Documents:\n{_documents_block(state.get('evidence', []))}")

    emit("response_generation", route=plan.get("route"), evidence=len(state.get("evidence", [])),
         retry=retry_of is not None)
    # Streaming model: tokens reach the UI through LangGraph's "messages" stream as they are generated.
    reply = await get_llm("strong", streaming=True).ainvoke([SystemMessage(system), *history[:-1],
                                                              HumanMessage(history[-1].content)])
    citations = sorted(set(CITATION.findall(reply.content)))
    emit("response_done", citations=citations, chars=len(reply.content))
    if retry_of is not None:
        return {"messages": [AIMessage(reply.content, id=retry_of.id)]}  # replaces the rejected answer
    return {"messages": [AIMessage(reply.content)]}
