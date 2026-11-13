"""Retrieval agent: hybrid search with the user's clearance, keeps only relevant hits.

When the question names document ids ("what actions followed INC-2026-021?"), a second search
runs inside those documents only (metadata filter doc_id, same access filter) and its hits come
first: an id is one token among many, so ranking alone does not reliably put it on top.
"""

import asyncio
import re
from datetime import date

from langgraph.runtime import Runtime

from app.config import get_settings
from app.graph.events import emit
from app.graph.nodes.common import node
from app.graph.state import AgentState, UserContext
from app.retrieval.hybrid import SearchFilters, get_retriever
from app.security.guards import sanitize_retrieved


DOC_ID = re.compile(r"\b(?:INC-\d{4}-\d{3}|(?:RB|ARCH|POL|SPEC|MTG)-\d{3})\b", re.IGNORECASE)
MAX_NAMED_DOCS = 3
NAMED_DOC_HITS = 4  # sections taken from each named document


def named_doc_ids(*texts: str) -> list[str]:
    """Document ids mentioned in the question or the supervisor's search query."""
    ids = {m.group(0).upper() for t in texts if t for m in DOC_ID.finditer(t)}
    return sorted(ids)[:MAX_NAMED_DOCS]


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
    retriever = get_retriever()
    named = named_doc_ids(query, state["messages"][-1].content)
    searches = [retriever.search(query, list(user.clearance), _filters(plan))]
    if named:  # the user named documents: also search inside them (doc_id filter; access filter still applies)
        searches.append(retriever.search(query, list(user.clearance), SearchFilters(doc_ids=named),
                                         top_k=NAMED_DOC_HITS * len(named)))
    results = await asyncio.gather(*searches)
    hits, pinned = results[0], (results[1] if named else [])
    relevant = [h for h in hits if h.score >= s.retrieval_min_score]
    if pinned:
        pinned_ids = {h.chunk_id for h in pinned}
        relevant = pinned + [h for h in relevant if h.chunk_id not in pinned_ids]
        emit("retrieval_named_docs", doc_ids=named, sections=len(pinned))
    fallback = any(h.source == "local_bm25" for h in hits + pinned)
    if fallback:
        emit("retrieval_fallback", reason="vector search unavailable", source="local_bm25", hits=len(hits))

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
    return {"evidence": evidence, "degraded": {"pinecone_down": True} if fallback else {}}
