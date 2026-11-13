"""Tool registry with role-based access control.

Every tool declares which roles may use it. RBAC is enforced twice:

  1. Binding  – `tools_for(role)`: the tool agent's LLM is only shown the tools the
                user's role allows, so it normally cannot even ask for others.
  2. Execution – `execute(...)`: every call is checked again against the role from
                the verified JWT (run context). A tool name the LLM invented, or one
                planted by a prompt injection, is denied here.

Arguments are validated with a Pydantic schema before the tool runs; every tool
call has a timeout; every attempt – allowed or denied – goes to the audit log.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from app.logging import get_logger

log = get_logger(__name__)

AUDIT_SIZE = 500


@dataclass(frozen=True)
class ToolContext:
    """Who is calling – always built from the verified JWT, never from LLM output."""

    user_id: str
    role: str
    clearance: tuple[str, ...]
    request_id: str = ""


@dataclass
class ToolResult:
    tool: str
    status: str  # ok | denied | invalid_args | timeout | error | unknown_tool
    data: Any = None
    message: str = ""
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict:
        return {"tool": self.tool, "status": self.status, "data": self.data, "message": self.message,
                "latency_ms": self.latency_ms}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    roles: frozenset[str]
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Awaitable[Any]]
    timeout_s: float = 20.0
    requires_approval: bool = False  # human-in-the-loop before running

    def schema_for_llm(self) -> dict:
        props = self.args_model.model_json_schema().get("properties", {})
        return {"name": self.name, "description": self.description,
                "args": {k: {kk: vv for kk, vv in v.items() if kk in ("type", "description", "enum", "default")}
                         for k, v in props.items()},
                "needs_approval": self.requires_approval}


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    audit: deque = field(default_factory=lambda: deque(maxlen=AUDIT_SIZE))

    def register(self, tool: Tool) -> Tool:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name} already registered")
        self.tools[tool.name] = tool
        return tool

    # ---------- layer 1: binding ----------

    def tools_for(self, role: str) -> list[Tool]:
        return [t for t in self.tools.values() if role in t.roles]

    def matrix(self) -> dict[str, list[str]]:
        """tool -> allowed roles (for docs / admin view)."""
        return {t.name: sorted(t.roles) for t in self.tools.values()}

    # ---------- layer 2: execution ----------

    def check(self, name: str, ctx: ToolContext) -> Tool:
        """Raise PermissionError / KeyError without running anything (used before asking for approval)."""
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(name)
        if ctx.role not in tool.roles:
            raise PermissionError(name)
        return tool

    async def execute(self, name: str, args: dict | None, ctx: ToolContext) -> ToolResult:
        start = time.perf_counter()
        try:
            tool = self.check(name, ctx)
        except KeyError:
            return self._finish(ToolResult(name, "unknown_tool", message=f"No tool named {name!r}."), ctx, start)
        except PermissionError:
            log.warning("rbac_denied", tool=name, role=ctx.role, user_id=ctx.user_id)
            return self._finish(ToolResult(name, "denied", message="Your role is not allowed to use this tool."),
                                ctx, start)

        try:
            parsed = tool.args_model.model_validate(args or {})
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return self._finish(ToolResult(name, "invalid_args", message=problems[:300]), ctx, start)

        try:
            data = await asyncio.wait_for(tool.handler(parsed, ctx), timeout=tool.timeout_s)
            return self._finish(ToolResult(name, "ok", data=data), ctx, start)
        except TimeoutError:
            return self._finish(ToolResult(name, "timeout", message=f"The tool did not finish within "
                                                                     f"{tool.timeout_s:.0f} s."), ctx, start)
        except Exception as exc:  # noqa: BLE001 – a failing tool must not take the conversation down
            log.error("tool_failed", tool=name, error_type=type(exc).__name__, exc_info=exc)
            return self._finish(ToolResult(name, "error", message=f"The tool failed ({type(exc).__name__})."),
                                ctx, start)

    def record_rejection(self, name: str, ctx: ToolContext) -> ToolResult:
        """A human rejected the call at the approval step."""
        return self._finish(ToolResult(name, "rejected", message="The request was not approved."), ctx,
                            time.perf_counter())

    def _finish(self, result: ToolResult, ctx: ToolContext, start: float) -> ToolResult:
        result.latency_ms = round((time.perf_counter() - start) * 1000)
        entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "user_id": ctx.user_id, "role": ctx.role,
                 "tool": result.tool, "status": result.status, "request_id": ctx.request_id,
                 "latency_ms": result.latency_ms}
        self.audit.append(entry)
        log.info("tool_audit", **{k: v for k, v in entry.items() if k != "ts"})
        return result


REGISTRY = Registry()
