"""Retrieval agent: hybrid search with the user's clearance, keeps only relevant hits."""

from datetime import date

from langgraph.runtime import Runtime

from app.config import get_settings
from app.graph.events import emit
from app.graph.nodes.common import node
from app.graph.state import AgentState, UserContext
from app.retrieval.hybrid import SearchFilters, get_retriever
from app.security.guards import sanitize_retrieved


def _fallback(_state: AgentState, _exc: Exception) -> dict:
    return {"evidence": [], "degraded": {"retrieval_unavailable": True}}


def _filters(plan: dict) -> SearchFilters:
    def d(v: str) -> date | None:
        return date.fromisoformat(v) if v else None

    return SearchFilters(
        departments=plan.get("departments") or None,
        document_types=plan.get("document_types") or None,
        created_from=d(plan.get("created_from", "")),
        created_to=d(plan.get("created_to", "")),
    )


@node("retrieval_agent", on_error=_fallback)
async def retrieval_agent(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    s = get_settings()
    plan = state.get("plan", {})
    user = runtime.context
    query = plan.get("search_query") or state["messages"][-1].content

    # Clearance comes from the run context (verified JWT), never from the plan the LLM wrote.
    hits = await get_retriever().search(query, list(user.clearance), _filters(plan))
    relevant = [h for h in hits if h.score >= s.retrieval_min_score]

    emit("retrieval", query=query, hits=len(hits), relevant=len(relevant), min_score=s.retrieval_min_score,
         top=[{"chunk_id": h.chunk_id, "score": h.score} for h in relevant[:5]])

    # Retrieved text is untrusted: clean it before any model sees it, and make flags visible.
    evidence = []
    for h in relevant:
        checked = sanitize_retrieved(h.text)
        if checked.flags:
            emit("content_flagged", chunk_id=h.chunk_id, flags=checked.flags)
        evidence.append({"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "section": h.section,
                         "text": checked.text, "score": h.score, "document_type": h.document_type,
                         "created_date": h.created_date})
    return {"evidence": evidence}
