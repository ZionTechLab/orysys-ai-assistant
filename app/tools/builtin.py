"""Built-in tools. Each one declares its roles, argument schema, timeout and approval need.

Role matrix (brief): Viewer – chat, search · Analyst – search, analytics, MCP · Admin – all tools.
Analytics (python_analysis) and MCP tools are registered by their own modules (tasks 2.10, 2.12).
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.tools.registry import REGISTRY, Tool, ToolContext

ALL_ROLES = frozenset({"viewer", "analyst", "admin"})
ADMIN = frozenset({"admin"})

DocType = Literal["incident", "runbook", "architecture", "policy", "product_spec", "meeting_notes"]
Department = Literal["payments", "cards", "core-banking", "security", "hr"]


# ---------- knowledge_search ----------

class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="what to search for")
    top_k: int = Field(default=5, ge=1, le=10, description="number of results")
    document_type: DocType | None = Field(default=None, description="optional document type filter")
    department: Department | None = Field(default=None, description="optional department filter")


async def knowledge_search(args: KnowledgeSearchArgs, ctx: ToolContext) -> list[dict]:
    from app.retrieval.hybrid import SearchFilters, get_retriever

    filters = SearchFilters(departments=[args.department] if args.department else None,
                            document_types=[args.document_type] if args.document_type else None)
    from app.security.guards import sanitize_retrieved

    hits = await get_retriever().search(args.query, list(ctx.clearance), filters, top_k=args.top_k)
    return [{"chunk_id": h.chunk_id, "title": h.title, "section": h.section, "score": h.score,
             "text": sanitize_retrieved(h.text).text[:600]} for h in hits]


# ---------- admin_reindex ----------

class ReindexArgs(BaseModel):
    dry_run: bool = Field(default=False, description="only report what would change")


async def admin_reindex(args: ReindexArgs, ctx: ToolContext) -> dict:
    """Sync the search index with the document library (decision A11) and reload in-memory caches."""
    from app.retrieval.sync import sync_index
    from app.rlm.catalog import reload_corpus

    report = await sync_index(dry_run=args.dry_run)
    if not args.dry_run:
        reload_corpus()  # research catalog sees the same documents as the index (BM25 is reloaded by the sync)
    return report.summary()


# ---------- admin_view_audit_log ----------

class AuditArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100, description="number of most recent entries")
    status: Literal["ok", "denied", "rejected", "invalid_args", "timeout", "error", "unknown_tool"] | None = Field(
        default=None, description="only entries with this status")


async def admin_view_audit_log(args: AuditArgs, ctx: ToolContext) -> list[dict]:
    entries = [e for e in REGISTRY.audit if args.status is None or e["status"] == args.status]
    return list(reversed(entries))[: args.limit]


def register_builtin_tools() -> None:
    if "knowledge_search" in REGISTRY.tools:
        return
    REGISTRY.register(Tool("knowledge_search", "Search the bank's documents and return matching sections with ids.",
                           ALL_ROLES, KnowledgeSearchArgs, knowledge_search, timeout_s=25))
    REGISTRY.register(Tool("admin_reindex", "Re-sync the search index with the document library: add new, update "
                                            "changed and remove deleted documents. Administrators only.",
                           ADMIN, ReindexArgs, admin_reindex, timeout_s=240, requires_approval=True))
    REGISTRY.register(Tool("admin_view_audit_log", "Show recent tool calls with user, role and outcome (including "
                                                   "denied calls). Administrators only.",
                           ADMIN, AuditArgs, admin_view_audit_log, timeout_s=5))


register_builtin_tools()
