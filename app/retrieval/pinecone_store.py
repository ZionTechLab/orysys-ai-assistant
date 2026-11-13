"""Async Pinecone access: index setup, embeddings (Pinecone Inference), upsert, query.

One serverless index with metric=dotproduct so each record holds a dense vector
and a sparse (BM25) vector. Dotproduct is required for hybrid queries: scaling the
two query vectors by alpha / (1-alpha) gives a weighted sum of both scores.
"""

import asyncio
from typing import Any, Literal

from pinecone import PineconeAsyncio, ServerlessSpec

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

EMBED_BATCH = 90  # llama-text-embed-v2 accepts up to 96 inputs per request


class PineconeStore:
    """Thin async wrapper. Create one per process and reuse it."""

    def __init__(self) -> None:
        self._s = get_settings()
        self._client: PineconeAsyncio | None = None
        self._index = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> PineconeAsyncio:
        if self._client is None:
            self._client = PineconeAsyncio(api_key=self._s.pinecone_api_key.get_secret_value())
        return self._client

    async def ensure_index(self) -> None:
        """Create the index if it does not exist (idempotent, used by the ingest script)."""
        pc = await self._get_client()
        existing = [i.name for i in (await pc.list_indexes())]
        if self._s.pinecone_index in existing:
            return
        log.info("pinecone_index_create", index=self._s.pinecone_index, dimension=self._s.embed_dimension)
        await pc.create_index(
            name=self._s.pinecone_index,
            dimension=self._s.embed_dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=self._s.pinecone_cloud, region=self._s.pinecone_region),
        )

    async def index(self):
        async with self._lock:
            if self._index is None:
                pc = await self._get_client()
                desc = await pc.describe_index(self._s.pinecone_index)
                self._index = pc.IndexAsyncio(host=desc.host)
            return self._index

    async def embed(self, texts: list[str], input_type: Literal["passage", "query"]) -> list[list[float]]:
        pc = await self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            result = await pc.inference.embed(
                model=self._s.embed_model,
                inputs=texts[start : start + EMBED_BATCH],
                parameters={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(item["values"] for item in result)
        return vectors

    async def upsert(self, records: list[dict[str, Any]], namespace: str) -> None:
        idx = await self.index()
        await idx.upsert(vectors=records, namespace=namespace, batch_size=50, show_progress=False)

    async def query(
        self,
        *,
        namespace: str,
        dense: list[float],
        sparse: dict | None,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[dict]:
        idx = await self.index()
        resp = await idx.query(
            namespace=namespace,
            vector=dense,
            sparse_vector=sparse,
            top_k=top_k,
            filter=metadata_filter,
            include_metadata=True,
        )
        return [{"id": m.id, "score": m.score, "metadata": dict(m.metadata or {})} for m in resp.matches]

    async def namespace_counts(self) -> dict[str, int]:
        idx = await self.index()
        stats = await idx.describe_index_stats()
        return {name: ns.vector_count for name, ns in (stats.namespaces or {}).items()}

    async def close(self) -> None:
        if self._index is not None:
            await self._index.close()
        if self._client is not None:
            await self._client.close()
        self._index = self._client = None
