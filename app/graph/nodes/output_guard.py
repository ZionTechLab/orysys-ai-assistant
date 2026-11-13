"""Output guard: last node before the answer reaches the user.

1. Deterministic checks (secrets, system-prompt leakage, clearance leakage) – always.
2. Brand compliance (fast model) – only when risk markers appear in the answer.
   A violation is rewritten once by the strong model; if the rewrite is unavailable or still
   breaks a rule, a safe neutral message is used instead.
The final text replaces the answer in place (same message id), so memory holds the guarded version.
If the guard itself fails, it fails closed: the answer is withheld (unlike the input guard and the
validator, which fail open for availability – their failure does not expose anything).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.graph.events import emit
from app.graph.nodes.common import node, parse_json
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm
from app.logging import get_logger
from app.security.output_guard import check_output

log = get_logger(__name__)

SAFE_FALLBACK = ("I can't phrase an answer to that in line with the bank's communication guidelines. "
                 "For advice on investments, rates, compensation or other banks, please contact the relevant "
                 "department. I'm happy to help with questions about the bank's policies, systems and incidents.")


WITHHELD = ("I couldn't complete the final safety check on this answer, so I have withheld it. "
            "Please try again in a moment.")


def _withhold(state: AgentState, _exc: Exception) -> dict:
    """Fail closed. This is the last leak barrier: if it could not run, the answer is not shown.

    (A failing brand check is handled inside the node and does not get here – only an unexpected
    error in the guard itself does.)
    """
    result = {"checked": False, "actions": ["withheld: output check failed"], "brand": "skipped (error)"}
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or (state.get("plan") or {}).get("route") == "blocked":
        return {"output_guard": result}  # fixed refusals are not model output: nothing to withhold
    emit("output_guard", **result, answer=WITHHELD)  # the UI replaces the streamed text with this
    return {"output_guard": result, "messages": [AIMessage(WITHHELD, id=last.id)]}


async def _brand_compliant(text: str) -> tuple[bool | None, list[str]]:
    """True / False, or None if the check could not run."""
    try:
        reply = await get_llm("fast").ainvoke([SystemMessage(prompts.BRAND_CHECK), HumanMessage(text)])
        verdict = parse_json(reply.content)
    except Exception:  # noqa: BLE001
        return None, []
    violations = [str(v)[:200] for v in verdict.get("violations", []) if v][:5]
    return (verdict.get("compliant") in (True, "true") and not violations), violations


@node("output_guard", on_error=_withhold)
async def output_guard(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or (state.get("plan") or {}).get("route") == "blocked":
        result = {"checked": True, "actions": [], "brand": "not needed"}
        emit("output_guard", **result)
        return {"output_guard": result}

    checked = check_output(last.content, tuple(runtime.context.clearance))
    text, actions, brand = checked.text, list(checked.actions), "not needed"

    if checked.brand_markers:
        compliant, violations = await _brand_compliant(text)
        if compliant is None:
            brand = "check unavailable"
        elif compliant:
            brand = "compliant"
        else:
            brand = "rewritten"
            try:
                reply = await get_llm("strong").ainvoke([SystemMessage(prompts.BRAND_REWRITE), HumanMessage(text)])
                rewritten = reply.content.strip()
                still_ok, _ = await _brand_compliant(rewritten)
                if rewritten and still_ok is not False:
                    text = rewritten
                else:
                    text, brand = SAFE_FALLBACK, "replaced"
            except Exception:  # noqa: BLE001 – rewrite unavailable: do not show a non-compliant answer
                text, brand = SAFE_FALLBACK, "replaced"
            actions.append(f"brand_{brand}: " + "; ".join(violations))

    # Full detail (incl. hidden document ids) goes to the server log only; the activity panel shown to the
    # user must not reveal which restricted documents exist.
    public_actions = [a.split(":")[0] if a.startswith("clearance_leak_removed") else a for a in actions]
    result = {"checked": True, "actions": public_actions, "brand": brand, "markers": checked.brand_markers}
    if text != last.content:
        log.warning("output_modified", actions=actions)
        emit("output_guard", **result, answer=text)  # the UI replaces the streamed text with this
        return {"output_guard": result, "messages": [AIMessage(text, id=last.id)]}
    emit("output_guard", **result)
    return {"output_guard": result}
