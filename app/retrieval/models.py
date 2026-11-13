"""Document and chunk models shared by all loaders."""

import re
from dataclasses import dataclass, field
from datetime import date, datetime

REQUIRED_FIELDS = ("doc_id", "title", "department", "document_type", "access_level", "created_date")
ACCESS_LEVELS = ("public", "internal", "confidential", "restricted")


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: date
    tags: tuple[str, ...] = ()
    severity: str | None = None
    path: str = ""
    source_format: str = "markdown"


@dataclass(frozen=True)
class ParsedDocument:
    """What every loader returns: metadata + ordered (heading, body) sections."""

    meta: DocMeta
    sections: list[tuple[str, str]]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<section-slug>" – also the citation id
    doc: DocMeta
    section: str
    text: str  # section body only
    embed_text: str = field(repr=False, default="")  # title + section + body, used for embeddings and BM25

    def metadata(self) -> dict:
        """Flat metadata stored in Pinecone (filters + attribution)."""
        d = self.doc
        meta = {
            "doc_id": d.doc_id,
            "title": d.title,
            "department": d.department,
            "document_type": d.document_type,
            "access_level": d.access_level,
            "created_date": d.created_date.isoformat(),
            # Pinecone range filters ($gte/$lte) need numbers, so the date is also stored as yyyymmdd.
            "created_int": int(d.created_date.strftime("%Y%m%d")),
            "tags": list(d.tags),
            "section": self.section,
            "text": self.text,
            "source_format": d.source_format,
        }
        if d.severity:
            meta["severity"] = d.severity
        return meta


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def make_meta(fields: dict, path: str, source_format: str) -> DocMeta:
    """Validate raw metadata from any format. A document without a valid access level is rejected."""
    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        raise ValueError(f"{path}: missing metadata fields {missing}")
    if fields["access_level"] not in ACCESS_LEVELS:
        raise ValueError(f"{path}: unknown access_level {fields['access_level']!r}")

    created = fields["created_date"]
    if isinstance(created, datetime):
        created = created.date()
    elif isinstance(created, str):
        created = date.fromisoformat(created[:10])

    tags = fields.get("tags") or ()
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    return DocMeta(
        doc_id=str(fields["doc_id"]),
        title=str(fields["title"]),
        department=str(fields["department"]),
        document_type=str(fields["document_type"]),
        access_level=str(fields["access_level"]),
        created_date=created,
        tags=tuple(tags),
        severity=fields.get("severity") or None,
        path=path,
        source_format=source_format,
    )
