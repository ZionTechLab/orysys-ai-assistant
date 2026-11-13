"""Unit tests for loading, chunking and hybrid query building (no network)."""

from datetime import date

import pytest

from app.retrieval.hybrid import SearchFilters, _namespaces, build_filter, scale_hybrid
from app.retrieval.loader import load_corpus


@pytest.fixture(scope="module")
def corpus():
    return load_corpus("data/docs")


def test_corpus_loads_all_documents(corpus):
    docs, chunks = corpus
    assert len(docs) == 45
    assert len(chunks) > len(docs)
    assert all(c.chunk_id.startswith(c.doc.doc_id + "#") for c in chunks)


def test_chunk_metadata_has_filter_and_attribution_fields(corpus):
    _, chunks = corpus
    md = next(c for c in chunks if c.chunk_id == "INC-2026-007#root-cause").metadata()
    assert md["department"] == "payments"
    assert md["document_type"] == "incident"
    assert md["access_level"] == "internal"
    assert md["created_int"] == 20260130
    assert md["section"] == "Root Cause"
    assert "connection" in md["text"].lower()


def test_poisoned_and_restricted_docs_are_labelled(corpus):
    docs, _ = corpus
    by_id = {d.doc_id: d for d in docs}
    assert by_id["POL-007"].access_level == "restricted"
    assert by_id["MTG-005"].access_level == "internal"  # poisoned doc is visible to everyone, by design


def test_scale_hybrid_weights_both_sides():
    dense, sparse = scale_hybrid([1.0, 2.0], {"indices": [3], "values": [10.0]}, alpha=0.75)
    assert dense == [0.75, 1.5]
    assert sparse == {"indices": [3], "values": [2.5]}


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_scale_hybrid_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        scale_hybrid([1.0], {"indices": [], "values": []}, alpha)


def test_filter_always_contains_access_clause():
    assert build_filter(["public", "internal"], None) == {"access_level": {"$in": ["public", "internal"]}}


def test_filter_combines_agent_filters_with_access_clause():
    flt = build_filter(
        ["public", "internal"],
        SearchFilters(document_types=["incident"], created_from=date(2025, 11, 12), created_to=date(2026, 11, 12)),
    )
    clauses = flt["$and"]
    assert {"access_level": {"$in": ["public", "internal"]}} in clauses
    assert {"document_type": {"$in": ["incident"]}} in clauses
    assert {"created_int": {"$gte": 20251112}} in clauses
    assert {"created_int": {"$lte": 20261112}} in clauses


def test_empty_clearance_is_rejected():
    with pytest.raises(ValueError):
        build_filter([], None)


def test_namespaces_default_to_all_and_reject_unknown():
    assert len(_namespaces(None)) == 5
    assert _namespaces(SearchFilters(departments=["payments"])) == ["payments"]
    with pytest.raises(ValueError):
        _namespaces(SearchFilters(departments=["treasury"]))


def test_named_document_filter_keeps_the_access_clause():
    flt = build_filter(["public", "internal"], SearchFilters(doc_ids=["INC-2026-021"]))
    assert {"access_level": {"$in": ["public", "internal"]}} in flt["$and"]
    assert {"doc_id": {"$in": ["INC-2026-021"]}} in flt["$and"]


def test_chunks_index_their_document_id(corpus):
    """Ids must be in the indexed text, or "INC-2026-021" cannot match by keyword."""
    _, chunks = corpus
    assert all(c.embed_text.startswith(c.doc.doc_id + " ") for c in chunks)
