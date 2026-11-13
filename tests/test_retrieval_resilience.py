"""Retriever resilience: stale connection -> reconnect + retry; persistent failure -> UpstreamError."""

import asyncio

import pytest

from app.errors import UpstreamError
from app.retrieval import hybrid
from app.retrieval.hybrid import HybridRetriever


class FakeBM25:
    def encode_queries(self, _q):
        return {"indices": [1], "values": [1.0]}


class FakeStore:
    """Fails the first `failures` embed calls, optionally by hanging instead of raising."""

    def __init__(self, failures: int, hang: bool = False):
        self.failures, self.hang, self.calls, self.resets = failures, hang, 0, 0

    async def embed(self, texts, input_type):
        self.calls += 1
        if self.calls <= self.failures:
            if self.hang:
                await asyncio.sleep(60)
            raise ConnectionAbortedError("connection aborted")
        return [[0.1, 0.2]]

    async def query(self, **kw):
        return [{"id": f"DOC-1#{kw['namespace']}", "score": 0.5,
                 "metadata": {"doc_id": "DOC-1", "access_level": "internal", "text": "t"}}]

    async def reset(self):
        self.resets += 1


@pytest.fixture(autouse=True)
def fake_bm25(monkeypatch):
    monkeypatch.setattr(hybrid, "get_bm25", lambda: FakeBM25())


@pytest.fixture
def fast_timeout(monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "retrieval_timeout_seconds", 0.2)


async def test_recovers_after_one_connection_error():
    store = FakeStore(failures=1)
    hits = await HybridRetriever(store).search("q", ["internal"])
    assert hits and store.resets == 1 and store.calls == 2


async def test_two_failures_fall_back_to_local_keyword_search():
    store = FakeStore(failures=2)
    hits = await HybridRetriever(store).search("PayBridge circuit breaker timeout", ["public", "internal"])
    assert store.resets == 2
    assert hits and all(h.source == "local_bm25" for h in hits)
    assert hits[0].doc_id in {"INC-2026-021", "INC-2026-062", "INC-2026-095", "RB-003", "MTG-003"}


async def test_local_fallback_keeps_the_access_filter():
    hits = await HybridRetriever(FakeStore(failures=2)).search("executive bonus deferred", ["public", "internal"])
    assert all(h.access_level in ("public", "internal") for h in hits)
    assert "POL-007" not in {h.doc_id for h in hits}


async def test_two_failures_raise_when_fallback_disabled(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "retrieval_local_fallback", False)
    store = FakeStore(failures=2)
    with pytest.raises(UpstreamError):
        await HybridRetriever(store).search("q", ["internal"])
    assert store.resets == 2


async def test_hanging_call_times_out_and_retries(fast_timeout):
    store = FakeStore(failures=1, hang=True)
    hits = await HybridRetriever(store).search("q", ["internal"])
    assert hits and store.resets == 1


async def test_bad_filters_are_not_retried():
    store = FakeStore(failures=0)
    with pytest.raises(ValueError):
        await HybridRetriever(store).search("q", [])  # empty clearance is a bug, not an outage
    assert store.calls == 0
