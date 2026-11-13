"""Catalog API for the research agent: exploration, bounded reads, clearance (no network)."""

import pytest

from app.rlm.catalog import MAX_SECTION_CHARS, Catalog, CatalogError

VIEWER = ("public", "internal")
ANALYST = ("public", "internal", "confidential")
ADMIN = ("public", "internal", "confidential", "restricted")


@pytest.fixture
def viewer():
    return Catalog(VIEWER)


def test_describe_shows_shape_not_content(viewer):
    d = viewer.describe()
    assert d["total_documents"] == 39  # 45 minus 4 confidential and 2 restricted
    assert d["by_document_type"]["incident"] == 11  # INC-2026-088 is confidential
    assert "Root Cause" in d["common_sections"]
    assert all(len(str(v)) < 2000 for v in d.values())


def test_clearance_controls_what_exists():
    counts = [Catalog(c).describe()["total_documents"] for c in (VIEWER, ANALYST, ADMIN)]
    assert counts == [39, 43, 45]


def test_payment_outage_scenario_listing(viewer):
    """The RLM demo question: payment incidents in the last 12 months (to 12 Nov 2026)."""
    found = {d["doc_id"] for dept in ("payments", "cards")
             for d in viewer.list_docs(document_type="incident", department=dept, since="2025-11-12")}
    assert found == {"INC-2025-118", "INC-2026-007", "INC-2026-021", "INC-2026-034",
                     "INC-2026-049", "INC-2026-062", "INC-2026-078", "INC-2026-095"}
    assert "INC-2025-101" not in found  # October 2025 – outside the window


def test_list_docs_returns_metadata_only_newest_first(viewer):
    docs = viewer.list_docs(document_type="incident")
    assert "text" not in docs[0] and "sections" in docs[0]
    dates = [d["created_date"] for d in docs]
    assert dates == sorted(dates, reverse=True)


def test_list_filters(viewer):
    assert [d["doc_id"] for d in viewer.list_docs(tag="paybridge", document_type="incident")] == \
        ["INC-2026-095", "INC-2026-062", "INC-2026-021"]
    assert viewer.list_docs(title_contains="phishing") == []  # confidential incident is invisible
    assert len(viewer.list_docs(limit=3)) == 3


def test_read_section_by_heading_or_slug(viewer):
    by_heading = viewer.read_section("INC-2026-007", "Root Cause")
    by_slug = viewer.read_section("INC-2026-007", "root-cause")
    assert by_heading == by_slug and "connection" in by_heading.lower()
    assert viewer.usage["read_section"] == 2 and viewer.usage["chars_read"] > 0


def test_read_section_works_for_pdf_and_docx_documents(viewer):
    assert "pool" in viewer.read_section("INC-2026-078", "Root Cause").lower()  # PDF
    assert "replica" in viewer.read_section("RB-002", "Month-end checklist (T-2 days)").lower()  # DOCX


def test_unknown_section_lists_available_ones(viewer):
    with pytest.raises(CatalogError, match="Root Cause"):
        viewer.read_section("INC-2026-007", "Lessons learned")


def test_hidden_and_missing_documents_give_the_same_error(viewer):
    with pytest.raises(CatalogError) as hidden:
        viewer.read_section("POL-007", "Structure")  # restricted
    with pytest.raises(CatalogError) as missing:
        viewer.read_section("POL-999", "Structure")
    assert str(hidden.value).replace("POL-007", "X") == str(missing.value).replace("POL-999", "X")


def test_admin_can_read_restricted():
    assert "bonus" in Catalog(ADMIN).read_section("POL-007", "Structure").lower()


def test_section_reads_are_capped(viewer, monkeypatch):
    from app.rlm import catalog as mod

    monkeypatch.setattr(mod, "MAX_SECTION_CHARS", 50)
    text = viewer.read_section("INC-2026-021", "Summary")
    assert len(text) <= 50 + len(" …[truncated]") and text.endswith("[truncated]")
    assert MAX_SECTION_CHARS == 2000


def test_bad_date_is_a_recoverable_error(viewer):
    with pytest.raises(CatalogError, match="YYYY-MM-DD"):
        viewer.list_docs(since="last year")


def test_empty_clearance_rejected():
    with pytest.raises(ValueError):
        Catalog(())


async def test_search_respects_clearance_and_returns_pointers():
    class FakeRetriever:
        async def search(self, query, clearance, filters, top_k):
            self.clearance = clearance
            from app.retrieval.hybrid import Hit
            return [Hit("POL-007#structure", "POL-007", "t", "Structure", "x", 0.9, "hr", "policy", "restricted", ""),
                    Hit("RB-003#actions", "RB-003", "t", "Actions", "x", 0.5, "payments", "runbook", "internal", "")]

    fake = FakeRetriever()
    hits = await Catalog(VIEWER, retriever=fake).search("anything", top_k=50)
    assert fake.clearance == list(VIEWER)
    assert [h["doc_id"] for h in hits] == ["RB-003"]  # defence in depth: restricted hit dropped
    assert "text" not in hits[0]
