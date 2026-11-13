"""Keep the Pinecone index in sync with the documents folder (decision A11).

Upsert alone never removes anything, so a deleted or withdrawn document could
still be cited. A sync compares the documents on disk with what is in Pinecone:

  added     – doc_id not in the index          -> embed + upsert all chunks
  updated   – content/metadata hash changed    -> embed + upsert, delete removed sections
  deleted   – doc_id in the index, not on disk -> delete all its chunks
  unchanged – same hash                        -> no embedding; sparse vector refreshed if BM25 was re-fitted

Pinecone itself is the record of what is indexed (each chunk stores `doc_hash`),
so there is no separate manifest file that could drift.
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.logging import get_logger
from app.retrieval.bm25 import fit_bm25, reload_bm25
from app.retrieval.hybrid import NAMESPACES
from app.retrieval.loader import Chunk, load_corpus
from app.retrieval.pinecone_store import PineconeStore

log = get_logger(__name__)

UPDATE_CONCURRENCY = 8


@dataclass(frozen=True)
class DocState:
    doc_hash: str
    namespace: str
    chunk_ids: frozenset[str]


@dataclass
class SyncPlan:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    # namespace -> chunk ids to delete (removed docs, removed sections, docs that moved department)
    delete_ids: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.updated or self.deleted)


@dataclass
class SyncReport:
    added: list[str]
    updated: list[str]
    deleted: list[str]
    unchanged: list[str]
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    sparse_refreshed: int = 0
    bm25_refit: bool = False
    dry_run: bool = False
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "unchanged": len(self.unchanged),
            "chunks_upserted": self.chunks_upserted,
            "chunks_deleted": self.chunks_deleted,
            "sparse_refreshed": self.sparse_refreshed,
            "bm25_refit": self.bm25_refit,
            "dry_run": self.dry_run,
            "seconds": round(self.seconds, 1),
        }


def doc_hash(chunks: list[Chunk]) -> str:
    """Hash of everything that ends up in the index for one document.

    Metadata is included on purpose: changing a document's access level or
    department must re-index it, even if the text is the same.
    """
    payload = [
        {k: v for k, v in c.metadata().items() if k != "source_format"} | {"chunk_id": c.chunk_id}
        for c in sorted(chunks, key=lambda c: c.chunk_id)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def local_state(chunks: list[Chunk]) -> dict[str, DocState]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc.doc_id].append(c)
    return {
        doc_id: DocState(doc_hash(cs), cs[0].doc.department, frozenset(c.chunk_id for c in cs))
        for doc_id, cs in by_doc.items()
    }


def plan_sync(local: dict[str, DocState], remote: dict[str, DocState], force: bool = False) -> SyncPlan:
    """Pure comparison – no I/O, unit-tested."""
    plan = SyncPlan()
    deletes: dict[str, set[str]] = defaultdict(set)

    for doc_id, loc in sorted(local.items()):
        rem = remote.get(doc_id)
        if rem is None:
            plan.added.append(doc_id)
            continue
        if rem.doc_hash == loc.doc_hash and rem.namespace == loc.namespace and not force:
            plan.unchanged.append(doc_id)
            continue
        plan.updated.append(doc_id)
        if rem.namespace != loc.namespace:
            deletes[rem.namespace] |= rem.chunk_ids  # moved department: remove everything from the old namespace
        else:
            deletes[rem.namespace] |= rem.chunk_ids - loc.chunk_ids  # sections that no longer exist

    for doc_id, rem in sorted(remote.items()):
        if doc_id not in local:
            plan.deleted.append(doc_id)
            deletes[rem.namespace] |= rem.chunk_ids

    plan.delete_ids = {ns: sorted(ids) for ns, ids in deletes.items() if ids}
    return plan


async def remote_state(store: PineconeStore) -> dict[str, DocState]:
    """Read what is currently indexed: ids per namespace, then doc_id + doc_hash from metadata."""
    idx = await store.index()
    docs: dict[str, dict] = {}
    for ns in NAMESPACES:
        ids: list[str] = []
        async for page in idx.list(namespace=ns):
            ids.extend(page if isinstance(page, list) else [v.id for v in page.vectors])
        for start in range(0, len(ids), 100):
            fetched = await idx.fetch(ids=ids[start : start + 100], namespace=ns)
            for vid, vec in fetched.vectors.items():
                md = vec.metadata or {}
                doc_id = md.get("doc_id") or vid.split("#", 1)[0]
                entry = docs.setdefault(doc_id, {"hash": md.get("doc_hash", ""), "ns": ns, "ids": set()})
                entry["ids"].add(vid)
                if md.get("doc_hash") != entry["hash"]:
                    entry["hash"] = ""  # mixed hashes = partially updated doc; force a re-index
    return {d: DocState(e["hash"], e["ns"], frozenset(e["ids"])) for d, e in docs.items()}


async def sync_index(dry_run: bool = False, force: bool = False, docs_dir: str | Path | None = None) -> SyncReport:
    s = get_settings()
    start = time.perf_counter()
    _, chunks = load_corpus(docs_dir or s.docs_dir)
    local = local_state(chunks)

    store = PineconeStore()
    try:
        await store.ensure_index()
        remote = await remote_state(store)
        plan = plan_sync(local, remote, force=force)
        report = SyncReport(plan.added, plan.updated, plan.deleted, plan.unchanged, dry_run=dry_run)
        report.chunks_deleted = sum(len(v) for v in plan.delete_ids.values())

        bm25_missing = not Path(s.bm25_path).exists()
        if dry_run or not (plan.has_changes or bm25_missing):
            report.seconds = time.perf_counter() - start
            log.info("index_sync", **report.summary())
            return report

        # 1. BM25 term weights depend on the whole corpus, so re-fit on every change.
        bm25 = fit_bm25([c.embed_text for c in chunks], s.bm25_path)
        report.bm25_refit = True
        reload_bm25()  # the running API picks up the new vocabulary on its next query

        # 2. Delete removed documents / sections first.
        for ns, ids in plan.delete_ids.items():
            for i in range(0, len(ids), 1000):
                await (await store.index()).delete(ids=ids[i : i + 1000], namespace=ns)

        # 3. Embed + upsert new and changed documents.
        changed = set(plan.added) | set(plan.updated)
        to_upsert = [c for c in chunks if c.doc.doc_id in changed]
        if to_upsert:
            dense = await store.embed([c.embed_text for c in to_upsert], input_type="passage")
            sparse = bm25.encode_documents([c.embed_text for c in to_upsert])
            by_ns: dict[str, list[dict]] = defaultdict(list)
            for c, d_vec, s_vec in zip(to_upsert, dense, sparse, strict=True):
                rec = {"id": c.chunk_id, "values": d_vec,
                       "metadata": c.metadata() | {"doc_hash": local[c.doc.doc_id].doc_hash}}
                if s_vec["indices"]:
                    rec["sparse_values"] = s_vec
                by_ns[c.doc.department].append(rec)
            await asyncio.gather(*(store.upsert(recs, ns) for ns, recs in by_ns.items()))
            report.chunks_upserted = len(to_upsert)

        # 4. Unchanged documents keep their dense vectors; only refresh sparse values (no embedding cost).
        unchanged = [c for c in chunks if c.doc.doc_id in set(plan.unchanged)]
        if unchanged:
            report.sparse_refreshed = await _refresh_sparse(store, unchanged, bm25)

        report.seconds = time.perf_counter() - start
        log.info("index_sync", **report.summary())
        return report
    finally:
        await store.close()


async def _refresh_sparse(store: PineconeStore, chunks: list[Chunk], bm25) -> int:
    idx = await store.index()
    sem = asyncio.Semaphore(UPDATE_CONCURRENCY)
    sparse = bm25.encode_documents([c.embed_text for c in chunks])

    async def one(c: Chunk, s_vec: dict) -> int:
        if not s_vec["indices"]:
            return 0
        async with sem:
            await idx.update(id=c.chunk_id, sparse_values=s_vec, namespace=c.doc.department)
        return 1

    return sum(await asyncio.gather(*(one(c, v) for c, v in zip(chunks, sparse, strict=True))))
