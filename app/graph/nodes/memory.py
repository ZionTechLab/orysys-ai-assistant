"""Memory node: last step of every turn. Folds older turns into the rolling summary (see app/memory/summary.py).

Runs after the answer has been streamed, so the user does not wait for it. If the summariser fails,
the previous summary is kept and the unsummarised messages are tried again next turn.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.config import get_settings
from app.graph.events import emit
from app.graph.nodes.common import node
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.memory.summary import messages_to_summarise, transcript


@node("memory")
async def memory_manager(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    messages = state.get("messages", [])
    summary = state.get("summary") or {}
    aged, covered = messages_to_summarise(messages, summary)
    if not aged:
        emit("memory_summary", action="kept", messages=len(messages), summarized=summary.get("covered", 0))
        return {}

    reply = await get_llm("fast").ainvoke([
        SystemMessage(prompts.MEMORY_SUMMARY),
        HumanMessage(f"Current summary:\n{summary.get('text') or '(none yet)'}\n\n"
                     f"Messages to add (oldest first):\n{transcript(aged)}"),
    ])
    text = str(reply.content).strip()[: get_settings().memory_summary_max_chars]
    if len(text) < 10:
        raise ValueError("the summariser returned an empty summary")
    emit("memory_summary", action="summarized", messages=len(messages), summarized=covered, added=len(aged),
         chars=len(text))
    return {"summary": {"text": text, "covered": covered, "turns": summary.get("turns", 0) + 1}}
