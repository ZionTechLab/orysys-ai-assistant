"""Format loaders: file -> ParsedDocument (metadata + sections).

This is the only format-specific step of ingestion (decision A2). Chunking,
embeddings, BM25 and Pinecone work the same for every format.

| Format   | Sections from                         | Metadata from                                  |
|----------|---------------------------------------|------------------------------------------------|
| Markdown | `## ` headings                        | YAML front-matter                              |
| DOCX     | paragraphs styled "Heading 2"         | core properties (identifier, subject, ...)     |
| PDF      | bookmarks (outline); else one per page | document info dictionary (custom /DocId, ...) |

In production, metadata (especially access level) would come from the document
management system rather than the file itself.
"""

import re
from pathlib import Path
from typing import Protocol

import frontmatter

from app.retrieval.models import ParsedDocument, make_meta


class Loader(Protocol):
    suffixes: tuple[str, ...]

    def load(self, path: Path) -> ParsedDocument: ...


class MarkdownLoader:
    suffixes = (".md",)

    def load(self, path: Path) -> ParsedDocument:
        post = frontmatter.load(path)
        meta = make_meta(dict(post.metadata), str(path), "markdown")
        parts = re.split(r"^##\s+(.+)$", post.content, flags=re.MULTILINE)
        # parts = [preamble, heading1, body1, heading2, body2, ...]; preamble (title line) is skipped
        sections = [(h.strip(), b.strip()) for h, b in zip(parts[1::2], parts[2::2])]
        return ParsedDocument(meta, sections)


class DocxLoader:
    """Word documents. Section = text between paragraphs styled 'Heading 2'.

    Metadata mapping (Word: File > Info > Properties):
      identifier -> doc_id, title -> title, subject -> department,
      category -> document_type, content_status -> access_level,
      keywords -> tags (comma separated), created -> created_date
    """

    suffixes = (".docx",)

    def load(self, path: Path) -> ParsedDocument:
        from docx import Document  # imported lazily: only needed when .docx files exist

        doc = Document(str(path))
        p = doc.core_properties
        meta = make_meta(
            {
                "doc_id": p.identifier,
                "title": p.title,
                "department": p.subject,
                "document_type": p.category,
                "access_level": p.content_status,
                "tags": p.keywords,
                "created_date": p.created,
            },
            str(path),
            "docx",
        )

        sections: list[tuple[str, str]] = []
        heading, lines = None, []
        for para in doc.paragraphs:
            style = para.style.name if para.style is not None else ""
            text = para.text.strip()
            if style == "Heading 2":
                if heading:
                    sections.append((heading, "\n".join(lines).strip()))
                heading, lines = text, []
            elif heading and text:
                prefix = "- " if style.startswith("List") else ""
                lines.append(prefix + text)
        if heading:
            sections.append((heading, "\n".join(lines).strip()))
        return ParsedDocument(meta, sections)


class PdfLoader:
    """Text PDFs. Sections from bookmarks if present, else one section per page.

    Metadata from the document info dictionary: /Title plus custom keys
    /DocId, /Department, /DocumentType, /AccessLevel, /CreatedDate, /Keywords, /Severity.
    Scanned PDFs would need OCR first (not handled in the POC).
    """

    suffixes = (".pdf",)

    def load(self, path: Path) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        info = reader.metadata or {}
        meta = make_meta(
            {
                "doc_id": info.get("/DocId"),
                "title": info.get("/Title"),
                "department": info.get("/Department"),
                "document_type": info.get("/DocumentType"),
                "access_level": info.get("/AccessLevel"),
                "tags": info.get("/Keywords"),
                "created_date": info.get("/CreatedDate"),
                "severity": info.get("/Severity"),
            },
            str(path),
            "pdf",
        )

        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        headings = [item.title for item in _flatten(reader.outline)]
        if headings:
            sections = _split_on_headings(full_text, headings)
        else:
            sections = [(f"Page {i}", (pg.extract_text() or "").strip()) for i, pg in enumerate(reader.pages, 1)]
        return ParsedDocument(meta, [s for s in sections if s[1]])


def _flatten(outline) -> list:
    items = []
    for item in outline:
        if isinstance(item, list):
            items.extend(_flatten(item))
        else:
            items.append(item)
    return items


def _split_on_headings(text: str, headings: list[str]) -> list[tuple[str, str]]:
    """Cut the extracted text at each bookmarked heading line, in outline order."""
    positions = []
    cursor = 0
    for h in headings:
        m = re.compile(rf"^\s*{re.escape(h)}\s*$", re.MULTILINE).search(text, cursor)
        if m:
            positions.append((h, m.start(), m.end()))
            cursor = m.end()
    sections = []
    for i, (h, _, end) in enumerate(positions):
        stop = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        sections.append((h, text[end:stop].strip()))
    return sections


LOADERS: list[Loader] = [MarkdownLoader(), DocxLoader(), PdfLoader()]
SUPPORTED_SUFFIXES = tuple(s for loader in LOADERS for s in loader.suffixes)


def get_loader(path: Path) -> Loader:
    for loader in LOADERS:
        if path.suffix.lower() in loader.suffixes:
            return loader
    raise ValueError(f"No loader for {path.suffix} files ({path.name})")
