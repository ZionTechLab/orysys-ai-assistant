"""MCP tools – enterprise data from the MCP server (mcp_server/), for analysts and administrators.

Each server tool is wrapped as a registry tool, so MCP calls get the same controls as every other
tool: role check at binding and at execution, Pydantic argument validation, timeout, audit log.

Allow-list, not discovery: only the three tools below are offered to the model, with schemas
defined here. A new tool appearing on the server is not exposed automatically (protects against
a changed or compromised server adding tools – "tool poisoning").

Trusted subsystem: the API authenticates to the server with a service key and passes the user's
clearance from the verified JWT – never from model output. Results are filtered by clearance again
here (defence in depth). The data that comes back is untrusted text like any document.
"""

import json
from datetime import date
from typing import Literal

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field

from app import faults
from app.config import get_settings
from app.security.guards import sanitize_retrieved
from app.tools.registry import REGISTRY, Tool, ToolContext

ANALYSTS = frozenset({"analyst", "admin"})
Department = Literal["payments", "cards", "core-banking", "security", "hr"]


class McpUnavailable(ConnectionError):
    """The MCP server could not be reached or failed."""


class McpToolError(RuntimeError):
    """The MCP server rejected the call (e.g. invalid arguments)."""


# ---------- argument schemas (what the model may pass) ----------

class EmployeeArgs(BaseModel):
    query: str = Field(default="", max_length=100, description="name, title or team words, e.g. 'card switch'")
    department: Department | None = Field(default=None, description="optional department filter")
    limit: int = Field(default=10, ge=1, le=25)


class ServiceArgs(BaseModel):
    query: str = Field(default="", max_length=100, description="service name words, e.g. 'PayBridge'; empty = all")
    department: Department | None = None
    status: Literal["operational", "degraded", "maintenance"] | None = Field(
        default=None, description="only services with this live status")
    tier: int | None = Field(default=None, ge=1, le=3, description="1 = most critical")


class IncidentArgs(BaseModel):
    service: str = Field(default="", max_length=60, description="service name words, e.g. 'PayBridge'; empty = all")
    status: Literal["open", "closed", "investigating", "monitoring"] | None = Field(
        default=None, description="'open' = not closed yet")
    severity: Literal["SEV1", "SEV2", "SEV3"] | None = None
    since: date | None = Field(default=None, description="only tickets opened on or after this date (YYYY-MM-DD)")
    limit: int = Field(default=20, ge=1, le=50)


# ---------- client ----------

def _open_client(ctx: ToolContext) -> Client:
    """One short-lived MCP session per call (simple and safe for a POC; a pool would reuse sessions)."""
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.mcp_api_key.get_secret_value()}",
               # who the API acts for – recorded in the MCP server's audit log
               "X-Caller-Id": ctx.user_id, "X-Caller-Role": ctx.role, "X-Request-ID": ctx.request_id}
    http = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(s.mcp_timeout_seconds))
    return Client(streamable_http_client(s.mcp_server_url, http_client=http),
                  read_timeout_seconds=s.mcp_timeout_seconds)


def _first_error(exc: BaseException) -> BaseException:
    """anyio wraps transport failures in exception groups; report the real cause."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


async def call_mcp(tool: str, arguments: dict, ctx: ToolContext) -> dict:
    faults.check("mcp")  # demo: simulated outage at the boundary to the real dependency
    try:
        async with _open_client(ctx) as client:
            result = await client.call_tool(tool, arguments)
    except Exception as exc:  # noqa: BLE001 – connection refused, timeout, 401, protocol error ...
        cause = _first_error(exc)
        raise McpUnavailable(f"MCP server unavailable ({type(cause).__name__})") from cause
    text = "".join(getattr(c, "text", "") for c in result.content or [])
    if result.is_error:
        raise McpToolError(text[:200] or "the MCP tool reported an error")
    return result.structured_content or json.loads(text)


def _within_clearance(records: list[dict], clearance: tuple[str, ...]) -> list[dict]:
    """Defence in depth: never pass on a record above the user's clearance, whatever the server sent."""
    return [r for r in records if r.get("access_level", "internal") in clearance]


def _clean(value):
    """Server data is untrusted: same cleaning as retrieved documents (hidden comments, tag breakouts)."""
    if isinstance(value, str):
        return sanitize_retrieved(value).text
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


# ---------- tool handlers ----------

async def employee_directory(args: EmployeeArgs, ctx: ToolContext) -> dict:
    return _clean(await call_mcp("employee_directory", args.model_dump(exclude_none=True), ctx))


async def service_catalog(args: ServiceArgs, ctx: ToolContext) -> dict:
    return _clean(await call_mcp("service_catalog", args.model_dump(exclude_none=True), ctx))


async def incident_records(args: IncidentArgs, ctx: ToolContext) -> dict:
    payload = args.model_dump(exclude_none=True, mode="json")
    payload["caller_clearance"] = list(ctx.clearance)  # from the JWT, not from the model
    data = await call_mcp("incident_records", payload, ctx)
    data["incidents"] = _within_clearance(data.get("incidents", []), ctx.clearance)
    data["count"] = len(data["incidents"])
    return _clean(data)


async def list_server_tools() -> list[str]:
    """Names of the tools the server offers (health check)."""
    ctx = ToolContext(user_id="health_check", role="system", clearance=("public",))
    async with _open_client(ctx) as client:
        return [t.name for t in (await client.list_tools()).tools]


def register_mcp_tools() -> None:
    if "mcp_employee_directory" in REGISTRY.tools:
        return
    s = get_settings()
    timeout = s.mcp_timeout_seconds + 5
    REGISTRY.register(Tool(
        "mcp_employee_directory",
        "Live employee directory (MCP): find staff by name, title or team – work email, extension, manager.",
        ANALYSTS, EmployeeArgs, employee_directory, timeout_s=timeout))
    REGISTRY.register(Tool(
        "mcp_service_catalog",
        "Live service catalog (MCP): who owns a service, who is on call now, its live status (operational / "
        "degraded / maintenance), tier, dependencies and its runbooks.",
        ANALYSTS, ServiceArgs, service_catalog, timeout_s=timeout, search_fallback=True))
    REGISTRY.register(Tool(
        "mcp_incident_records",
        "Live incident tickets (MCP): open incidents right now, or past tickets with severity, status, "
        "duration and the linked incident report.",
        ANALYSTS, IncidentArgs, incident_records, timeout_s=timeout, search_fallback=True))


register_mcp_tools()
