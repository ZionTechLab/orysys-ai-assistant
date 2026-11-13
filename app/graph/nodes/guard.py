"""Input guard: first node of every turn.

1. Rules (app/security/guards.py) – high-confidence attacks are blocked without any LLM call.
2. LLM classifier (fast model) – only for messages with softer suspicious markers that no rule
   matched, so normal questions do not pay the extra latency.
Blocked turns skip all agents and get a fixed, polite refusal (no LLM, nothing to manipulate).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.graph.events import emit
from app.graph.nodes.common import node, parse_json
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.logging import get_logger
from app.security.guards import detect_injection

log = get_logger(__name__)

CATEGORIES = {"instruction_override", "data_exfiltration", "tool_abuse"}


def _allow_on_error(_state: AgentState, _exc: Exception) -> dict:
    # Guard failure does not block the user: RBAC, clearance filters, untrusted-data wrapping and the
    # output guard still apply. The failure is recorded (errors) and visible in the panel.
    return {"guard": {"blocked": False, "checked_by": "error"}}


@node("input_guard", on_error=_allow_on_error)
async def input_guard(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    text = state["messages"][-1].content
    result = detect_injection(text)
    checked_by = "rules"

    if not result.blocked and result.suspicious:
        checked_by = "rules+llm"
        try:
            reply = await get_llm("fast").ainvoke([SystemMessage(prompts.INJECTION_CLASSIFIER),
                                                   HumanMessage(result.normalized)])
            verdict = parse_json(reply.content)
            if verdict.get("attack") in (True, "true") and verdict.get("category") in CATEGORIES:
                result.blocked, result.category, result.rule = True, verdict["category"], "llm_classifier"
        except Exception as exc:  # noqa: BLE001 – classifier unavailable: allow, record (defence in depth)
            checked_by = f"rules (classifier unavailable: {type(exc).__name__})"

    guard = {"blocked": result.blocked, "category": result.category, "rule": result.rule,
             "suspicious": result.suspicious, "checked_by": checked_by}
    emit("input_guard", **guard)
    if result.blocked:
        log.warning("input_blocked", category=result.category, rule=result.rule, role=runtime.context.role)
        return {"guard": guard, "plan": {"route": "blocked", "intent": f"blocked: {result.category}"}}
    return {"guard": guard}
