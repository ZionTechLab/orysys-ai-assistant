"""Document catalog: the API the research agent (RLM) uses to explore the collection.

Recursive Language Model idea: instead of loading whole documents into the model's
context, the agent first looks at *what exists* (describe, list_docs), decides which
documents matter, and reads only the sections it needs (read_section).

Security:
- A Catalog is built per request for one user's clearance. Documents above it are
  invisible – not counted, listed, searchable or readable.
- "Not found" and "not visible" give the same error, so the catalog cannot be used
  to probe for documents a user may not see.

Size limits keep the agent from pulling a whole corpus into context.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from app.config import get_settings
from app.retrieval.loader import Chunk, DocMeta, load_corpus
from app.retrieval.models import slug

MAX_LIST = 200           # documents returned by one list_docs call
MAX_SECTION_CHARS = 2000  # characters returned by one read_section call
MAX_SEARCH_K = 10


class CatalogError(Exception):
    """Raised for requests the agent can recover from (unknown doc/section, bad argument)."""


@dataclass(frozen=True)
class Corpus:
    docs: dict[str, DocMeta]
    sections: dict[str, dict[str, Chunk]]  # doc_id -> section slug -> chunk


@lru_cache
def get_corpus() -> Corpus:
    """Loaded once per process from the same documents that are indexed (see decision A5)."""
    docs, chunks = load_corpus(get_settings().docs_dir)
    sections: dict[str, dict[str, Chunk]] = {d.doc_id: {} for d in docs}
    for c in chunks:
        sections[c.doc.doc_id][slug(c.section)] = c
    return Corpus({d.doc_id: d for d in docs}, sections)


def reload_corpus() -> None:
    """Call after an index sync so the catalog matches the index."""
    get_corpus.cache_clear()


def _parse_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise CatalogError(f"{name} must be YYYY-MM-DD") from None


class Catalog:
    def __init__(self, clearance: tuple[str, ...] | list[str], corpus: Corpus | None = None, retriever=None):
        if not clearance:
            raise ValueError("clearance must not be empty")
        corpus = corpus or get_corpus()
        self._clearance = tuple(clearance)
        self._docs = {i: d for i, d in corpus.docs.items() if d.access_level in self._clearance}
        self._sections = {i: corpus.sections[i] for i in self._docs}
        self._retriever = retriever
        # What the agent has done – shown in the activity panel and traces.
        self.usage: Counter = Counter()

    # ---------- explore ----------

    def describe(self) -> dict:
        """Shape of the collection the user can see. No document content."""
        self.usage["describe"] += 1
        docs = list(self._docs.values())
        dates = sorted(d.created_date for d in docs)
        return {
            "total_documents": len(docs),
            "by_document_type": dict(Counter(d.document_type for d in docs)),
            "by_department": dict(Counter(d.department for d in docs)),
            "date_range": [dates[0].isoformat(), dates[-1].isoformat()] if dates else None,
            "common_sections": [s for s, _ in Counter(
                c.section for secs in self._sections.values() for c in secs.values()).most_common(12)],
            "top_tags": [t for t, _ in Counter(t for d in docs for t in d.tags).most_common(25)],
        }

    def list_docs(
        self,
        document_type: str | None = None,
        department: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tag: str | None = None,
        title_contains: str | None = None,
        limit: int = MAX_LIST,
    ) -> list[dict]:
        """Document metadata (no content), newest first."""
        self.usage["list_docs"] += 1
        start, end = _parse_date(since, "since"), _parse_date(until, "until")
        out = []
        for d in sorted(self._docs.values(), key=lambda d: d.created_date, reverse=True):
            if document_type and d.document_type != document_type:
                continue
            if department and d.department != department:
                continue
            if start and d.created_date < start:
                continue
            if end and d.created_date > end:
                continue
            if tag and tag.lower() not in (t.lower() for t in d.tags):
                continue
            if title_contains and title_contains.lower() not in d.title.lower():
                continue
            out.append(self._info(d))
        return out[: max(1, min(int(limit), MAX_LIST))]

    def get_doc(self, doc_id: str) -> dict:
        self.usage["get_doc"] += 1
        return self._info(self._visible(doc_id))

    # ---------- read ----------

    def read_section(self, doc_id: str, section: str) -> str:
        """Text of one section (by heading or slug), capped at MAX_SECTION_CHARS."""
        self._visible(doc_id)
        chunk = self._sections[doc_id].get(slug(section))
        if chunk is None:
            available = [c.section for c in self._sections[doc_id].values()]
            raise CatalogError(f"{doc_id} has no section {section!r}; sections: {available}")
        self.usage["read_section"] += 1
        self.usage["chars_read"] += min(len(chunk.text), MAX_SECTION_CHARS)
        text = chunk.text
        return text if len(text) <= MAX_SECTION_CHARS else text[:MAX_SECTION_CHARS] + " …[truncated]"

    # ---------- search ----------

    async def search(self, query: str, top_k: int = 5, document_type: str | None = None,
                     department: str | None = None, since: str | None = None) -> list[dict]:
        """Hybrid search (Pinecone) limited to the user's clearance. Returns pointers, not full text."""
        from app.retrieval.hybrid import SearchFilters, get_retriever

        self.usage["search"] += 1
        retriever = self._retriever or get_retriever()  # shared, warm connections
        filters = SearchFilters(
            departments=[department] if department else None,
            document_types=[document_type] if document_type else None,
            created_from=_parse_date(since, "since"),
        )
        hits = await retriever.search(query, list(self._clearance), filters,
                                      top_k=max(1, min(int(top_k), MAX_SEARCH_K)))
        return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "section": h.section,
                 "score": h.score} for h in hits if h.doc_id in self._docs]

    # ---------- helpers ----------

    def _visible(self, doc_id: str) -> DocMeta:
        doc = self._docs.get(str(doc_id))
        if doc is None:
            raise CatalogError(f"document {doc_id!r} not found")  # same message if it exists but is not visible
        return doc

    def _info(self, d: DocMeta) -> dict:
        return {
            "doc_id": d.doc_id,
            "title": d.title,
            "document_type": d.document_type,
            "department": d.department,
            "created_date": d.created_date.isoformat(),
            "tags": list(d.tags),
            "severity": d.severity,
            "sections": [c.section for c in self._sections[d.doc_id].values()],
        }
