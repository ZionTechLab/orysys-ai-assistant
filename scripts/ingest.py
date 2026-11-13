"""Ingest data/docs into Pinecone (hybrid: dense + BM25 sparse), one namespace per department.

Run: .venv/Scripts/python scripts/ingest.py
Idempotent: creates the index if missing, record ids are chunk ids so re-running overwrites.
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402
from app.retrieval.bm25 import fit_bm25  # noqa: E402
from app.retrieval.loader import load_corpus  # noqa: E402
from app.retrieval.pinecone_store import PineconeStore  # noqa: E402

log = get_logger("ingest")


async def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, "console")

    # 1. Load and chunk
    docs, chunks = load_corpus(s.docs_dir)
    log.info("corpus_loaded", documents=len(docs), chunks=len(chunks))

    # 2. Sparse: fit BM25 on our corpus and save the vocabulary for query time
    bm25 = fit_bm25([c.embed_text for c in chunks], s.bm25_path)
    sparse = bm25.encode_documents([c.embed_text for c in chunks])
    log.info("bm25_fitted", path=s.bm25_path)

    # 3. Dense: Pinecone Inference embeddings
    store = PineconeStore()
    try:
        await store.ensure_index()
        dense = await store.embed([c.embed_text for c in chunks], input_type="passage")
        log.info("embedded", model=s.embed_model, vectors=len(dense))

        # 4. Group by namespace (department) and upsert
        by_ns: dict[str, list[dict]] = defaultdict(list)
        for chunk, d_vec, s_vec in zip(chunks, dense, sparse):
            record = {"id": chunk.chunk_id, "values": d_vec, "metadata": chunk.metadata()}
            if s_vec["indices"]:
                record["sparse_values"] = s_vec
            by_ns[chunk.doc.department].append(record)

        await asyncio.gather(*(store.upsert(recs, ns) for ns, recs in by_ns.items()))
        for ns, recs in sorted(by_ns.items()):
            log.info("namespace_upserted", namespace=ns, records=len(recs))

        # Serverless indexes are eventually consistent; wait until counts match.
        expected = {ns: len(recs) for ns, recs in by_ns.items()}
        for _ in range(30):
            counts = await store.namespace_counts()
            if all(counts.get(ns, 0) >= n for ns, n in expected.items()):
                break
            await asyncio.sleep(2)
        log.info("ingest_complete", namespaces=await store.namespace_counts())
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
