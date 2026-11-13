"""MCP server exposing dummy enterprise data: employee directory, service catalog, incident records.

A separate service (its own process and port), as an enterprise system would be. The assistant
reaches it over MCP Streamable HTTP (app/tools/mcp_tools.py).

Security
  - Every HTTP request needs the shared service key (Authorization: Bearer <MCP_API_KEY>);
    without a key configured the server refuses to start.
  - Trusted-subsystem model: the assistant API authenticates the end user, enforces RBAC and passes
    the user's clearance (`caller_clearance`) and identity (X-Caller-* headers, for the audit log).
    Records above that clearance are never returned. Missing clearance -> least privilege.
  - Data minimisation: work contact details only; arguments are validated and result sizes capped.
  - Read-only: no tool changes data.

All data is fictional (mcp_server/data/*.json).
"""

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

log = logging.getLogger("mcp_server")

DATA_DIR = Path(__file__).parent / "data"
ACCESS_LEVELS = ("public", "internal", "confidential", "restricted")
DEFAULT_CLEARANCE = ("public", "internal")  # when the caller does not say: least privilege

Department = Literal["payments", "cards", "core-banking", "security", "hr"]
Clearance = Annotated[list[Literal["public", "internal", "confidential", "restricted"]] | None,
                      Field(description="Set by the calling application from the verified user, not by a model.")]

mcp = MCPServer(
    name="combank-enterprise-data",
    version="0.1.0",
    instructions="Read-only enterprise data for the internal knowledge assistant: employee directory, "
                 "service catalog (owners, on-call, status) and incident tickets. All data is fictional.",
)


@lru_cache
def _load(name: str) -> list[dict]:
    raw = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return raw[name]


def _employees() -> dict[str, dict]:
    return {e["employee_id"]: e for e in _load("employees")}


def _person(employee_id: str | None) -> dict | None:
    e = _employees().get(employee_id or "")
    return {"employee_id": e["employee_id"], "name": e["name"], "title": e["title"], "extension": e["extension"],
            "email": e["email"]} if e else None


def _matches(text: str, *fields: Any) -> bool:
    """Every word of the query appears in one of the fields (case-insensitive)."""
    haystack = " ".join(str(f) for f in fields if f).lower()
    return all(word in haystack for word in text.lower().split())


def _allowed(clearance: list[str] | None) -> set[str]:
    return set(clearance or DEFAULT_CLEARANCE) & set(ACCESS_LEVELS)


def _audit(ctx: Context | None, tool: str, returned: int, **args: Any) -> None:
    """One line per call, with the end user the assistant API acted for."""
    request = getattr(getattr(ctx, "request_context", None), "request", None) if ctx else None
    headers = getattr(request, "headers", {}) or {}
    log.info(json.dumps({"event": "mcp_tool_call", "tool": tool, "returned": returned,
                         "caller_id": headers.get("x-caller-id"), "caller_role": headers.get("x-caller-role"),
                         "request_id": headers.get("x-request-id"),
                         "args": {k: v for k, v in args.items() if v not in (None, "", [])}}))


# ---------- tools ----------

@mcp.tool(description="Look up staff by name, title, team or skill. Returns work contact details and manager.")
def employee_directory(
    query: Annotated[str, Field(max_length=100, description="name, title or team words; empty = everyone")] = "",
    department: Department | None = None,
    limit: Annotated[int, Field(ge=1, le=25)] = 10,
    ctx: Context | None = None,
) -> dict:
    staff = [e for e in _load("employees")
             if (not department or e["department"] == department)
             and _matches(query, e["name"], e["title"], e["team"], e["department"])]
    results = [{**{k: e[k] for k in ("employee_id", "name", "title", "department", "team", "email", "extension",
                                    "location")},
                "manager": _person(e["manager_id"])} for e in staff[:limit]]
    _audit(ctx, "employee_directory", len(results), query=query, department=department)
    return {"count": len(results), "total_matches": len(staff), "employees": results}


@mcp.tool(description="Service catalog: owner team, owner, current on-call engineer, tier, live status, "
                      "dependencies and the runbooks / architecture document for each service.")
def service_catalog(
    query: Annotated[str, Field(max_length=100, description="service name or id words; empty = all")] = "",
    department: Department | None = None,
    status: Literal["operational", "degraded", "maintenance"] | None = None,
    tier: Annotated[int | None, Field(ge=1, le=3)] = None,
    ctx: Context | None = None,
) -> dict:
    services = [s for s in _load("services")
                if (not department or s["department"] == department)
                and (not status or s["status"] == status)
                and (tier is None or s["tier"] == tier)
                and _matches(query, s["service_id"], s["name"], s["owner_team"], s["description"])]
    results = [{**{k: v for k, v in s.items() if k not in ("owner_id", "on_call_id")},
                "owner": _person(s["owner_id"]), "on_call": _person(s["on_call_id"])} for s in services]
    _audit(ctx, "service_catalog", len(results), query=query, department=department, status=status, tier=tier)
    return {"count": len(results), "services": results}


@mcp.tool(description="Incident tickets from the IT service management system: open and past incidents with "
                      "severity, status, times, duration and the linked incident report (report_doc).")
def incident_records(
    service: Annotated[str, Field(max_length=60, description="service id or name words; empty = all")] = "",
    status: Literal["open", "closed", "investigating", "monitoring"] | None = Field(
        default=None, description="'open' = not closed yet (investigating or monitoring)"),
    severity: Literal["SEV1", "SEV2", "SEV3"] | None = None,
    since: Annotated[str, Field(pattern=r"^(\d{4}-\d{2}-\d{2})?$", description="YYYY-MM-DD")] = "",
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
    caller_clearance: Clearance = None,
    ctx: Context | None = None,
) -> dict:
    allowed = _allowed(caller_clearance)
    services = {s["service_id"]: s for s in _load("services")}
    since_date = date.fromisoformat(since) if since else None

    def keep(t: dict) -> bool:
        svc = services.get(t["service_id"], {})
        if t["access_level"] not in allowed:
            return False
        if status == "open" and t["status"] == "closed" or status not in (None, "open") and t["status"] != status:
            return False
        if severity and t["severity"] != severity:
            return False
        if since_date and date.fromisoformat(t["opened_at"][:10]) < since_date:
            return False
        return _matches(service, t["service_id"], svc.get("name"))

    tickets = sorted((t for t in _load("incidents") if keep(t)), key=lambda t: t["opened_at"], reverse=True)
    results = [{**t, "service_name": services.get(t["service_id"], {}).get("name")} for t in tickets[:limit]]
    _audit(ctx, "incident_records", len(results), service=service, status=status, severity=severity, since=since,
           clearance=sorted(allowed))
    return {"count": len(results), "total_matches": len(tickets), "incidents": results}
