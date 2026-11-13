"""Convert a few sample Markdown documents to .docx and .pdf.

Demonstrates that ingestion works the same for Word and PDF files (decision A10).
The Markdown originals are moved to data/samples_src/ (kept in git, not ingested),
so each doc_id exists once in data/docs.

Run: .venv/Scripts/python scripts/make_sample_formats.py
"""

import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"
SRC = ROOT / "data" / "samples_src"

# doc_id -> target format
TARGETS = {
    "RB-002": "docx",
    "POL-004": "docx",
    "INC-2026-078": "pdf",
    "ARCH-001": "pdf",
}

# Characters the built-in PDF fonts (latin-1) cannot encode.
_PDF_SAFE = {"–": "-", "—": "-", "→": "->", "’": "'", "‘": "'", "“": '"', "”": '"', "≥": ">=", "≤": "<=", "×": "x"}


def _plain(text: str) -> str:
    """Remove inline markdown (bold, code) – Word/PDF carry formatting differently."""
    return re.sub(r"\*\*(.+?)\*\*|`(.+?)`", lambda m: m.group(1) or m.group(2), text)


def _pdf_text(text: str) -> str:
    for k, v in _PDF_SAFE.items():
        text = text.replace(k, v)
    return text


def _sections(content: str) -> list[tuple[str, list[str]]]:
    parts = re.split(r"^##\s+(.+)$", content, flags=re.MULTILINE)
    return [(h.strip(), [ln for ln in b.strip().splitlines() if ln.strip()]) for h, b in zip(parts[1::2], parts[2::2])]


def _created(meta: dict) -> datetime:
    d = meta["created_date"]
    d = d if isinstance(d, date) else date.fromisoformat(str(d))
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def to_docx(meta: dict, content: str, out: Path) -> None:
    doc = Document()
    props = doc.core_properties
    props.identifier = meta["doc_id"]
    props.title = meta["title"]
    props.subject = meta["department"]
    props.category = meta["document_type"]
    props.content_status = meta["access_level"]
    props.keywords = ", ".join(meta.get("tags", []))
    props.created = _created(meta)
    props.comments = meta.get("disclaimer", "")
    props.author = "Sample data generator"

    doc.add_heading(f"{meta['doc_id']} - {meta['title']}", level=1)
    doc.add_paragraph(meta.get("disclaimer", "")).italic = True
    for heading, lines in _sections(content):
        doc.add_heading(heading, level=2)
        for line in lines:
            if line.startswith("- "):
                doc.add_paragraph(_plain(line[2:]), style="List Bullet")
            elif re.match(r"^\d+\.\s", line):
                doc.add_paragraph(_plain(re.sub(r"^\d+\.\s", "", line)), style="List Number")
            else:
                doc.add_paragraph(_plain(line))
    doc.save(out)


def to_pdf(meta: dict, content: str, out: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _pdf_text(f"{meta['doc_id']} - {meta['title']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, _pdf_text(meta.get("disclaimer", "")), new_x="LMARGIN", new_y="NEXT")
    for heading, lines in _sections(content):
        pdf.ln(3)
        pdf.start_section(_pdf_text(heading), level=0)  # adds a bookmark used by the PDF loader
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, _pdf_text(heading), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            pdf.multi_cell(0, 5.5, _pdf_text(_plain(line)), new_x="LMARGIN", new_y="NEXT")
    tmp = out.with_suffix(".tmp.pdf")
    pdf.output(str(tmp))

    # Write metadata into the document info dictionary (custom keys allowed by the PDF spec).
    writer = PdfWriter(clone_from=PdfReader(str(tmp)))
    writer.add_metadata({
        "/Title": meta["title"],  # info dict supports Unicode; only page text is latin-1
        "/DocId": meta["doc_id"],
        "/Department": meta["department"],
        "/DocumentType": meta["document_type"],
        "/AccessLevel": meta["access_level"],
        "/CreatedDate": _created(meta).date().isoformat(),
        "/Keywords": ", ".join(meta.get("tags", [])),
        "/Severity": str(meta.get("severity", "")),
        "/Disclaimer": meta.get("disclaimer", ""),
    })
    with open(out, "wb") as fh:
        writer.write(fh)
    tmp.unlink()


def main() -> int:
    SRC.mkdir(parents=True, exist_ok=True)
    for doc_id, fmt in TARGETS.items():
        md = next(DOCS.rglob(f"{doc_id}.md"), None)
        if md is None:
            original = next(SRC.rglob(f"{doc_id}.md"), None)
            if original is None:
                print(f"skip {doc_id}: markdown not found")
                continue
            md_src = original  # already converted before; rebuild from the kept original
            target_dir = DOCS / original.parent.name
        else:
            md_src = SRC / md.parent.name / md.name
            md_src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(md), md_src)
            target_dir = md.parent

        post = frontmatter.load(md_src)
        out = target_dir / f"{doc_id}.{fmt}"
        (to_docx if fmt == "docx" else to_pdf)(dict(post.metadata), post.content, out)
        print(f"{doc_id}: {md_src.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
