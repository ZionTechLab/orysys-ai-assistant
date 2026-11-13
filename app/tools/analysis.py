"""python_analysis tool – structured analysis over the documents the user may see (analyst, admin).

The tool agent passes only a question. Inside the tool:
  1. the LLM writes Python against the clearance-bound catalog plus analysis helpers,
  2. the sandbox (app/rlm/sandbox.py) runs it – same limits as research plans,
  3. sandbox errors go back to the LLM to repair the code (up to 3 attempts).
The result includes the code that ran, so every number is explainable.
"""

import re
import statistics

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import prompts
from app.rlm.catalog import Catalog
from app.rlm.sandbox import SandboxError, run_plan
from app.tools.registry import REGISTRY, Tool, ToolContext

ANALYSTS = frozenset({"analyst", "admin"})
ATTEMPTS = 3
_NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


# ---------- helpers available inside the analysis code ----------

def count_by(items, key) -> dict:
    """{value: count} for a key of each dict (lists, e.g. tags, count each element)."""
    counts: dict = {}
    for item in items:
        values = item.get(key) if isinstance(item, dict) else None
        for v in values if isinstance(values, list) else [values]:
            counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def mean(values) -> float | None:
    values = [float(v) for v in values]
    return round(statistics.fmean(values), 2) if values else None


def median(values) -> float | None:
    values = [float(v) for v in values]
    return round(statistics.median(values), 2) if values else None


def find_numbers(text: str) -> list[float]:
    """Numbers in text, e.g. '~22,400 declined ... 47 minutes' -> [22400.0, 47.0]."""
    return [float(n.replace(",", "")) for n in _NUMBER.findall(str(text))]


HELPERS = {"count_by": count_by, "mean": mean, "median": median, "find_numbers": find_numbers}


# ---------- tool ----------

class AnalysisArgs(BaseModel):
    question: str = Field(min_length=5, max_length=500,
                          description="the analysis to perform, e.g. 'number of incidents per department in 2026'")


def _code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


async def python_analysis(args: AnalysisArgs, ctx: ToolContext) -> dict:
    from app.llm.factory import get_llm

    catalog = Catalog(ctx.clearance)
    messages = [SystemMessage(prompts.ANALYSIS_CODER), HumanMessage(f"Question: {args.question}")]
    llm = get_llm("strong", long=True)
    attempts, code, error = [], "", None
    for attempt in range(1, ATTEMPTS + 1):
        reply = await llm.ainvoke(messages)
        code = _code(reply.content)
        run = await run_plan(code, catalog, extra_functions=HELPERS)
        attempts.append({"attempt": attempt, "ok": run.ok, "error": run.error})
        if run.ok:
            return {"question": args.question, "code": code, "result": run.result, "printed": run.stdout or None,
                    "attempts": attempts, "catalog_calls": {k: v for k, v in run.catalog_usage.items()
                                                            if k != "chars_read"}}
        error = run.error
        messages += [AIMessage(reply.content), HumanMessage(prompts.RESEARCH_PLAN_REPAIR.format(error=error))]
    raise SandboxError(f"analysis code failed after {ATTEMPTS} attempts: {error}")


def register_analysis_tool() -> None:
    if "python_analysis" in REGISTRY.tools:
        return
    REGISTRY.register(Tool(
        "python_analysis",
        "Answer a quantitative question about the documents (counts, trends, averages, durations, comparisons) by "
        "running generated Python in a sandbox over document metadata and sections. Pass the question only.",
        ANALYSTS, AnalysisArgs, python_analysis, timeout_s=300))


register_analysis_tool()
