"""Hybrid search: dense (embeddings) + sparse (BM25), across department namespaces.

Score = alpha * dense_score + (1 - alpha) * sparse_score
Implemented by scaling the query vectors before a single dotproduct query, so
Pinecone ranks by the combined score.

Access control: the `access_level` filter is built from the *user's* clearance
(from the JWT), never from LLM output. Callers cannot widen it.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import date

from app.config import get_settings
from app.errors import UpstreamError
from app.logging import get_logger
from app.retrieval.bm25 import get_bm25
from app.retrieval.pinecone_store import PineconeStore

log = get_logger(__name__)

NAMESPACES = ("payments", "cards", "core-banking", "security", "hr")


@dataclass
class SearchFilters:
    """Optional narrowing chosen by the agent (on top of the mandatory access filter)."""

    departments: list[str] | None = None  # -> namespaces to query
    document_types: list[str] | None = None
    created_from: date | None = None
    created_to: date | None = None


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    department: str
    document_type: str
    access_level: str
    created_date: str

    def citation(self) -> str:
        return f"[{self.chunk_id}]"


def scale_hybrid(dense: list[float], sparse: dict, alpha: float) -> tuple[list[float], dict]:
    """Weight dense by alpha and sparse by (1 - alpha)."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    return (
        [v * alpha for v in dense],
        {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]},
    )


def build_filter(clearance: list[str], f: SearchFilters | None) -> dict:
    """Pinecone metadata filter. The access clause is always present."""
    if not clearance:
        raise ValueError("clearance must not be empty")
    clauses: list[dict] = [{"access_level": {"$in": list(clearance)}}]
    if f:
        if f.document_types:
            clauses.append({"document_type": {"$in": f.document_types}})
        if f.created_from:
            clauses.append({"created_int": {"$gte": int(f.created_from.strftime("%Y%m%d"))}})
        if f.created_to:
            clauses.append({"created_int": {"$lte": int(f.created_to.strftime("%Y%m%d"))}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _namespaces(f: SearchFilters | None) -> list[str]:
    if f and f.departments:
        unknown = set(f.departments) - set(NAMESPACES)
        if unknown:
            raise ValueError(f"unknown departments: {sorted(unknown)}")
        return list(f.departments)
    return list(NAMESPACES)


class HybridRetriever:
    def __init__(self, store: PineconeStore | None = None) -> None:
        self._s = get_settings()
        self._store = store or PineconeStore()

    async def search(
        self,
        query: str,
        clearance: list[str],
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> list[Hit]:
        top_k = top_k or self._s.retrieval_top_k
        alpha = self._s.hybrid_alpha if alpha is None else alpha
        namespaces = _namespaces(filters)
        metadata_filter = build_filter(clearance, filters)
        start = time.perf_counter()

        try:
            dense = (await self._store.embed([query], input_type="query"))[0]
            sparse = get_bm25().encode_queries(query)
            dense, sparse = scale_hybrid(dense, sparse, alpha)
            sparse_arg = sparse if sparse["indices"] else None

            # Query every namespace in parallel, then merge by combined score.
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        self._store.query(
                            namespace=ns, dense=dense, sparse=sparse_arg, top_k=top_k, metadata_filter=metadata_filter
                        )
                        for ns in namespaces
                    )
                ),
                timeout=self._s.retrieval_timeout_seconds,
            )
        except (asyncio.TimeoutError, OSError) as exc:
            log.warning("retrieval_failed", error_type=type(exc).__name__, namespaces=namespaces)
            raise UpstreamError("Document search is unavailable.") from exc

        matches = sorted((m for ns_matches in results for m in ns_matches), key=lambda m: m["score"], reverse=True)
        hits = [_to_hit(m) for m in matches[:top_k]]
        log.info(
            "retrieval",
            namespaces=namespaces,
            alpha=alpha,
            filter_keys=sorted(_filter_keys(metadata_filter)),
            hits=len(hits),
            top_doc=hits[0].doc_id if hits else None,
            latency_ms=round((time.perf_counter() - start) * 1000),
        )
        return hits


def _filter_keys(flt: dict) -> set[str]:
    if "$and" in flt:
        return {k for clause in flt["$and"] for k in clause}
    return set(flt)


def _to_hit(match: dict) -> Hit:
    md = match["metadata"]
    return Hit(
        chunk_id=match["id"],
        doc_id=md.get("doc_id", ""),
        title=md.get("title", ""),
        section=md.get("section", ""),
        text=md.get("text", ""),
        score=round(float(match["score"]), 4),
        department=md.get("department", ""),
        document_type=md.get("document_type", ""),
        access_level=md.get("access_level", ""),
        created_date=md.get("created_date", ""),
    )
