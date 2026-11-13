"""Sync planning: what gets added, updated, deleted (pure logic, no Pinecone)."""

from dataclasses import replace

import pytest

from app.retrieval.loader import load_corpus
from app.retrieval.sync import DocState, doc_hash, local_state, plan_sync


def state(h: str, ns: str, *ids: str) -> DocState:
    return DocState(h, ns, frozenset(ids))


def test_identical_index_means_no_changes():
    local = {"A": state("h1", "payments", "A#x", "A#y")}
    plan = plan_sync(local, dict(local))
    assert plan.unchanged == ["A"]
    assert not plan.has_changes
    assert plan.delete_ids == {}


def test_new_document_is_added():
    plan = plan_sync({"A": state("h1", "payments", "A#x")}, {})
    assert plan.added == ["A"]
    assert plan.delete_ids == {}


def test_removed_document_is_deleted_with_all_chunks():
    plan = plan_sync({}, {"A": state("h1", "payments", "A#x", "A#y")})
    assert plan.deleted == ["A"]
    assert plan.delete_ids == {"payments": ["A#x", "A#y"]}


def test_changed_document_deletes_only_removed_sections():
    local = {"A": state("h2", "payments", "A#x")}
    remote = {"A": state("h1", "payments", "A#x", "A#old")}
    plan = plan_sync(local, remote)
    assert plan.updated == ["A"]
    assert plan.delete_ids == {"payments": ["A#old"]}


def test_document_moved_to_other_department_is_removed_from_old_namespace():
    local = {"A": state("h2", "cards", "A#x")}
    remote = {"A": state("h1", "payments", "A#x")}
    plan = plan_sync(local, remote)
    assert plan.updated == ["A"]
    assert plan.delete_ids == {"payments": ["A#x"]}


def test_force_reindexes_unchanged_documents():
    local = {"A": state("h1", "payments", "A#x")}
    assert plan_sync(local, dict(local), force=True).updated == ["A"]


@pytest.fixture(scope="module")
def chunks():
    return load_corpus("data/docs")[1]


def test_hash_is_stable(chunks):
    assert local_state(chunks) == local_state(chunks)


def test_access_level_change_changes_the_hash(chunks):
    """A permission change must re-index the document even if the text is the same."""
    doc_chunks = [c for c in chunks if c.doc.doc_id == "INC-2026-007"]
    restricted = [replace(c, doc=replace(c.doc, access_level="restricted")) for c in doc_chunks]
    assert doc_hash(doc_chunks) != doc_hash(restricted)


def test_text_change_changes_the_hash(chunks):
    doc_chunks = [c for c in chunks if c.doc.doc_id == "INC-2026-007"]
    edited = [replace(doc_chunks[0], text=doc_chunks[0].text + " edited"), *doc_chunks[1:]]
    assert doc_hash(doc_chunks) != doc_hash(edited)
