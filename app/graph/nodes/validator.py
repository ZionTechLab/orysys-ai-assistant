"""Validator: checks the answer after it is written; one retry; last-resort repair.

Deterministic checks (always):
  - the answer is not empty
  - every [citation] exists in the evidence the response agent was actually given
    (hallucinated citations – a guardrail the brief asks for)
  - document answers (retrieve / research) cite at least one source
LLM grounding check (fast model, short timeout, only when the deterministic checks pass):
  - each factual claim is supported by the documents; if the check cannot run it is
    recorded as skipped and does not block the answer.

On failure: back to the response agent once, with the problems as feedback. The new answer
replaces the old one (same message id). If it fails again: invalid citations are removed
and a short note is added – the user never sees a reference that does not exist.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.graph.events import emit
from app.graph.nodes.common import node, parse_json
from app.graph.state import AgentState, UserContext
from app.llm import prompts
from app.llm.factory import get_llm

CITATION = re.compile(r"\[([A-Z]+-[\w-]+#[\w-]+)\]")
MAX_RETRIES = 1
REPAIR_NOTE = "\n\n_Some references could not be verified against the source documents and were removed._"


def allowed_citations(state: AgentState) -> set[str]:
    ids = {e["chunk_id"] for e in state.get("evidence", [])}
    data = (state.get("tool_result") or {}).get("data")
    if isinstance(data, list):  # knowledge_search results carry chunk ids
        ids |= {d["chunk_id"] for d in data if isinstance(d, dict) and "chunk_id" in d}
    return ids


def deterministic_problems(answer: str, route: str, allowed: set[str]) -> list[str]:
    problems = []
    if len(answer.strip()) < 15:
        problems.append("the answer is empty or too short")
    cited = set(CITATION.findall(answer))
    invented = sorted(cited - allowed)
    if invented:
        problems.append(f"these citations are not in the documents provided: {', '.join(invented)}")
    if route in ("retrieve", "research") and allowed and not cited:
        problems.append("the answer uses documents but cites none of them")
    return problems


async def grounding_check(answer: str, state: AgentState) -> tuple[str, list[str]]:
    evidence = state.get("evidence", [])
    if not evidence:
        return "skipped (no documents)", []
    # Title and date are part of what the answer may rely on (e.g. "the town hall on 20 October").
    docs = "\n\n".join(f"[{e['chunk_id']}] {e.get('title', '')} ({e.get('created_date', '')}), "
                       f"section {e.get('section', '')}:\n{e['text']}" for e in evidence)[:12000]
    try:
        reply = await get_llm("fast").ainvoke([SystemMessage(prompts.GROUNDING_CHECK),
                                               HumanMessage(f"Documents:\n{docs}\n\nAnswer:\n{answer}")])
        verdict = parse_json(reply.content)
    except Exception as exc:  # noqa: BLE001 – grounding is an extra check; do not block answers on it
        return f"skipped ({type(exc).__name__})", []
    unsupported = [str(c)[:200] for c in verdict.get("unsupported", []) if c][:5]
    grounded = verdict.get("grounded") in (True, "true") and not unsupported
    return ("passed" if grounded else "failed"), unsupported


def _accept(_state: AgentState, _exc: Exception) -> Command:
    return Command(goto="output_guard", update={"validation": {"ok": True, "action": "accepted", "grounding": "skipped (error)"}})


@node("validator", on_error=_accept)
async def validator(state: AgentState, runtime: Runtime[UserContext]) -> Command:
    route = (state.get("plan") or {}).get("route", "")
    last = state["messages"][-1]
    attempt = (state.get("validation") or {}).get("attempt", 0)

    llm_down = bool((state.get("degraded") or {}).get("llm_unavailable"))
    if route in ("blocked", "chitchat") or llm_down or not isinstance(last, AIMessage):
        # Fixed refusals, greetings and the no-LLM fallback message are not model-written answers:
        # retrying them would only call the unavailable model again.
        grounding = "not needed (language service unavailable)" if llm_down else "not needed"
        result = {"ok": True, "action": "accepted", "attempt": attempt, "problems": [], "grounding": grounding}
        emit("validation", **result)
        return Command(goto="output_guard", update={"validation": result})

    answer = last.content
    allowed = allowed_citations(state)
    problems = deterministic_problems(answer, route, allowed)
    if route not in ("retrieve", "research"):
        grounding = "not needed (tool result)"
    elif problems:
        grounding = "skipped (deterministic checks failed)"
    if not problems and route in ("retrieve", "research"):
        grounding, unsupported = await grounding_check(answer, state)
        if unsupported:
            problems.append("claims not supported by the documents: " + "; ".join(unsupported))

    if not problems:
        result = {"ok": True, "action": "accepted", "attempt": attempt, "problems": [], "grounding": grounding,
                  "citations": sorted(set(CITATION.findall(answer)))}
        emit("validation", **result)
        return Command(goto="output_guard", update={"validation": result})

    if attempt < MAX_RETRIES:
        result = {"ok": False, "action": "retry", "attempt": attempt + 1, "problems": problems, "grounding": grounding,
                  "previous_answer": answer}  # kept in case the rewrite comes back worse (e.g. empty)
        emit("validation", **{k: v for k, v in result.items() if k != "previous_answer"})
        return Command(goto="response_agent", update={"validation": result})

    # Last resort: never end worse than the first attempt. If the rewrite is empty, use the first answer.
    previous = (state.get("validation") or {}).get("previous_answer", "")
    base = answer if len(answer.strip()) >= 15 or len(previous.strip()) < 15 else previous
    # Remove citations that do not exist, keep the rest, and say so.
    repaired = CITATION.sub(lambda m: m.group(0) if m.group(1) in allowed else "", base)
    if repaired != base:
        repaired = re.sub(r"[ \t]{2,}", " ", repaired) + REPAIR_NOTE
    result = {"ok": False, "action": "repaired", "attempt": attempt, "problems": problems, "grounding": grounding}
    emit("validation", **result, answer=repaired)  # the UI replaces the streamed text with this
    return Command(goto="output_guard", update={"validation": result,
                                     "messages": [AIMessage(repaired, id=last.id)]})  # same id: replaces the answer


def validation_feedback(state: AgentState) -> str:
    """Extra prompt text for the response agent when it is asked to try again."""
    v = state.get("validation") or {}
    if v.get("action") != "retry":
        return ""
    return prompts.VALIDATION_FEEDBACK.format(problems=json.dumps(v.get("problems", []), indent=1))
