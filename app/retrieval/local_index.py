"""Local keyword search – the fallback when Pinecone is unavailable.

Scores the same chunks with the same fitted BM25 vocabulary as the sparse side of the hybrid
index, entirely in memory. Same access-level filter, same optional filters. Weaker than hybrid
(no semantic matching), so answers based on it are marked as degraded.
"""

from functools import lru_cache

from app.config import get_settings
from app.retrieval.bm25 import get_bm25
from app.retrieval.loader import Chunk, load_corpus


class LocalKeywordIndex:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        encoded = get_bm25().encode_documents([c.embed_text for c in chunks])
        self._vectors = [dict(zip(v["indices"], v["values"], strict=True)) for v in encoded]

    def search(self, query: str, clearance: list[str], filters=None, top_k: int = 8) -> list[tuple[Chunk, float]]:
        q = get_bm25().encode_queries(query)
        q_vec = dict(zip(q["indices"], q["values"], strict=True))
        scored = []
        for chunk, vec in zip(self._chunks, self._vectors, strict=True):
            d = chunk.doc
            if d.access_level not in clearance:  # same rule as the Pinecone filter
                continue
            if filters is not None:
                if filters.departments and d.department not in filters.departments:
                    continue
                if filters.document_types and d.document_type not in filters.document_types:
                    continue
                if filters.created_from and d.created_date < filters.created_from:
                    continue
                if filters.created_to and d.created_date > filters.created_to:
                    continue
            score = sum(w * vec.get(i, 0.0) for i, w in q_vec.items())
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda cs: cs[1], reverse=True)
        if not scored:
            return []
        # Normalise to 0..1 (best hit = 1) so the relevance threshold means roughly the same as for hybrid.
        top = scored[0][1]
        return [(c, round(s / top, 4)) for c, s in scored[:top_k]]


@lru_cache
def get_local_index() -> LocalKeywordIndex:
    _, chunks = load_corpus(get_settings().docs_dir)
    return LocalKeywordIndex(chunks)


def reload_local_index() -> None:
    get_local_index.cache_clear()
