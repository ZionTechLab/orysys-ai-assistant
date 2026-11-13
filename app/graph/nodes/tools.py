"""Tool agent: picks a tool, gets human approval when required, executes it with RBAC.

  tool_agent     LLM sees ONLY the tools the user's role allows (RBAC layer 1) and returns
                 {"tool", "args", "reason"} as JSON.
  approval_gate  tools marked requires_approval pause the run with LangGraph interrupt();
                 the API asks the user to approve or reject and resumes the run.
  tool_executor  REGISTRY.execute() re-checks the role (RBAC layer 2), validates arguments,
                 applies the timeout and writes the audit log.

Three separate nodes, so that resuming after approval re-runs only the gate – the LLM
is not asked again and cannot pick a different tool after the human said yes.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.graph.events import emit
from app.graph.nodes.common import node, parse_json
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.config import get_settings
from app.logging import get_logger
from app.security.guards import sanitize_retrieved
from app.tools import analysis, builtin, mcp_tools  # noqa: F401 – registers the tools
from app.tools.registry import REGISTRY, ToolContext

MAX_RESULT_CHARS = 8000
log = get_logger(__name__)


def _ctx(runtime: Runtime[UserContext]) -> ToolContext:
    c = runtime.context
    return ToolContext(user_id=c.user_id, role=c.role, clearance=tuple(c.clearance), request_id=c.request_id)


def _no_tool(_state: AgentState, _exc: Exception) -> dict:
    return {"tool_call": {"tool": None, "args": {}, "reason": "tool selection failed"}}


@node("tool_agent", on_error=_no_tool)
async def tool_agent(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    offered = REGISTRY.tools_for(runtime.context.role)  # RBAC layer 1: only permitted tools are shown
    if not offered:
        emit("tool_selected", tool=None, offered=[], reason="no tools for this role")
        return {"tool_call": {"tool": None, "args": {}, "reason": "no tools available for this role"}}

    tools_json = json.dumps([t.schema_for_llm() for t in offered], indent=1)
    reply = await get_llm("strong").ainvoke([
        SystemMessage(prompts.TOOL_AGENT.replace("{tools}", tools_json)),
        HumanMessage(state["messages"][-1].content),
    ])
    choice = parse_json(reply.content)
    tool = choice.get("tool")
    tool = None if tool in (None, "", "none") else str(tool)
    args = choice.get("args") if isinstance(choice.get("args"), dict) else {}
    emit("tool_selected", tool=tool, args=args, reason=str(choice.get("reason", ""))[:200],
         offered=[t.name for t in offered])
    return {"tool_call": {"tool": tool, "args": args, "reason": str(choice.get("reason", ""))[:200]}}


@node("approval_gate")
async def approval_gate(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    call = state.get("tool_call") or {}
    name = call.get("tool")
    if not name:
        return {}
    try:
        tool = REGISTRY.check(name, _ctx(runtime))
    except (KeyError, PermissionError):
        return {}  # the executor denies it and writes the audit entry – never ask a human to approve this
    if not tool.requires_approval:
        return {"tool_call": {**call, "approved": True}}

    # Pause the run. The API shows the request to the user; resuming returns their decision here.
    decision = interrupt({"tool": name, "args": call.get("args", {}), "reason": call.get("reason", ""),
                         "message": f"Approve running '{name}'?"})
    approved = bool(isinstance(decision, dict) and decision.get("approved"))
    emit("approval_decision", tool=name, approved=approved, by=(decision or {}).get("by"))
    return {"tool_call": {**call, "approved": approved, "approved_by": (decision or {}).get("by")}}


@node("tool_executor")
async def tool_executor(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    call = state.get("tool_call") or {}
    name = call.get("tool")
    if not name:
        return {"tool_result": {"tool": None, "status": "no_tool", "data": None,
                                "message": call.get("reason") or "No suitable tool for this request."}}

    ctx = _ctx(runtime)
    if call.get("approved") is False:
        result = REGISTRY.record_rejection(name, ctx)
    else:
        emit("tool_call", tool=name, args=call.get("args", {}))
        result = await REGISTRY.execute(name, call.get("args", {}), ctx)  # RBAC layer 2 + validation + timeout

    if result.status == "denied":
        emit("rbac_denied", tool=name, role=ctx.role)
    out = result.as_dict()
    text = json.dumps(out["data"], default=str)
    if len(text) > MAX_RESULT_CHARS:
        out["data"] = text[:MAX_RESULT_CHARS] + " …[truncated]"
    emit("tool_result", tool=name, status=result.status, latency_ms=result.latency_ms, message=result.message,
         preview=text[:300])

    tool = REGISTRY.tools.get(name)
    if result.status in ("error", "timeout") and tool is not None and tool.search_fallback:
        # Graceful degradation: the live system (e.g. the MCP server) is down – answer from the documents.
        evidence = await _search_documents(state, ctx)
        emit("tool_fallback", tool=name, reason=result.message, source="document search", evidence=len(evidence))
        return {"tool_result": out, "evidence": evidence, "degraded": {"mcp_down": True}}
    return {"tool_result": out}


async def _search_documents(state: AgentState, ctx: ToolContext) -> list[dict]:
    """Hybrid search with the user's question and clearance; [] if search is down as well."""
    from app.retrieval.hybrid import get_retriever

    try:
        hits = await get_retriever().search(state["messages"][-1].content, list(ctx.clearance), top_k=5)
    except Exception as exc:  # noqa: BLE001 – both systems down: the answer says the action failed
        log.warning("tool_fallback_search_failed", error_type=type(exc).__name__)
        return []
    min_score = get_settings().retrieval_min_score
    return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "section": h.section,
             "text": sanitize_retrieved(h.text).text, "score": h.score, "document_type": h.document_type,
             "created_date": h.created_date} for h in hits if h.score >= min_score]
