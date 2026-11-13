"""Load documents of any supported format and split them into section chunks.

Chunking by section (not fixed token windows) because the documents are
structured (Summary, Root Cause, Actions...). A section is a meaningful unit to
cite, and the RLM research agent can read one section at a time.
"""

from pathlib import Path

from app.retrieval.loaders import SUPPORTED_SUFFIXES, get_loader
from app.retrieval.models import Chunk, DocMeta, ParsedDocument, slug

__all__ = ["Chunk", "DocMeta", "build_chunks", "load_corpus"]


def build_chunks(parsed: ParsedDocument) -> list[Chunk]:
    doc = parsed.meta
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{slug(heading)}",
            doc=doc,
            section=heading,
            text=body,
            # The id is part of the indexed text, so "INC-2026-021" matches by keyword (BM25).
            embed_text=f"{doc.doc_id} {doc.title}\n{heading}\n{body}",
        )
        for heading, body in parsed.sections
        if body
    ]


def load_corpus(docs_dir: str | Path) -> tuple[list[DocMeta], list[Chunk]]:
    docs, chunks = [], []
    paths = sorted(p for p in Path(docs_dir).rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in paths:
        parsed = get_loader(path).load(path)
        docs.append(parsed.meta)
        chunks.extend(build_chunks(parsed))

    doc_ids = [d.doc_id for d in docs]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate doc_id – the same document exists in two formats")
    chunk_ids = [c.chunk_id for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk ids – two sections with the same heading in one document")
    return docs, chunks
