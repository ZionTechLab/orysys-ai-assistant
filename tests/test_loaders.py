"""Loaders: the same document gives the same sections and metadata from Markdown, DOCX and PDF."""

from pathlib import Path

import pytest

from app.retrieval.loader import build_chunks
from app.retrieval.loaders import get_loader

DOCS = Path("data/docs")
SRC = Path("data/samples_src")

CONVERTED = [
    ("RB-002", DOCS / "runbook/RB-002.docx", SRC / "runbook/RB-002.md", "docx"),
    ("POL-004", DOCS / "policy/POL-004.docx", SRC / "policy/POL-004.md", "docx"),
    ("INC-2026-078", DOCS / "incident/INC-2026-078.pdf", SRC / "incident/INC-2026-078.md", "pdf"),
    ("ARCH-001", DOCS / "architecture/ARCH-001.pdf", SRC / "architecture/ARCH-001.md", "pdf"),
]


@pytest.mark.parametrize("doc_id,converted,original,fmt", CONVERTED)
def test_converted_document_matches_markdown_original(doc_id, converted, original, fmt):
    new = get_loader(converted).load(converted)
    old = get_loader(original).load(original)

    assert new.meta.source_format == fmt
    for field in ("doc_id", "title", "department", "document_type", "access_level", "created_date", "tags"):
        assert getattr(new.meta, field) == getattr(old.meta, field), field

    # Same citation ids, so answers cite the same sections whatever the file format.
    assert [c.chunk_id for c in build_chunks(new)] == [c.chunk_id for c in build_chunks(old)]


def test_pdf_keeps_severity():
    parsed = get_loader(DOCS / "incident/INC-2026-078.pdf").load(DOCS / "incident/INC-2026-078.pdf")
    assert parsed.meta.severity == "SEV1"


def test_unsupported_format_is_rejected():
    with pytest.raises(ValueError):
        get_loader(Path("notes.xlsx"))


def test_document_without_access_level_is_rejected(tmp_path):
    bad = tmp_path / "BAD-1.md"
    bad.write_text("---\ndoc_id: BAD-1\ntitle: t\ndepartment: hr\ndocument_type: policy\n"
                   "created_date: 2026-01-01\n---\n## A\ntext\n", encoding="utf-8")
    with pytest.raises(ValueError, match="access_level"):
        get_loader(bad).load(bad)
