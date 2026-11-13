"""Research agent (RLM) logic with fake LLMs – no network."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command, Send

from app.graph.nodes import research
from app.graph.state import UserContext, _findings
from app.rlm.catalog import Catalog, CatalogError

VIEWER = UserContext("viewer1", "viewer", ("public", "internal"))
RUNTIME = SimpleNamespace(context=VIEWER)
Q = "Summarize payment outages in the last year and recurring root causes"


class ScriptedLLM:
    """Returns the given replies in order; an Exception instance is raised instead."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def use_llm(monkeypatch, llm):
    monkeypatch.setattr(research, "get_llm", lambda *a, **k: llm)


GOOD_PLAN = '''```python
docs = []
for dept in ["payments", "cards"]:
    docs += list_docs(document_type="incident", department=dept, since="2025-11-12")
result = {"batches": chunk(sorted(d["doc_id"] for d in docs), 3)}
```'''


# ---------- planner ----------

async def test_planner_runs_generated_plan(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM(GOOD_PLAN))
    out = await research.research_planner({"messages": [HumanMessage(Q)]}, RUNTIME)
    r = out["research"]
    assert len(r["doc_ids"]) == 8 and "INC-2025-101" not in r["doc_ids"]
    assert [len(b) for b in r["batches"]] == [3, 3, 2]
    assert r["plan_attempts"] == 1 and not r["used_fallback"]


async def test_planner_repairs_a_failing_plan(monkeypatch):
    llm = ScriptedLLM("```python\nimport os\nresult = 1\n```", GOOD_PLAN)
    use_llm(monkeypatch, llm)
    out = await research.research_planner({"messages": [HumanMessage(Q)]}, RUNTIME)
    assert out["research"]["plan_attempts"] == 2
    repair_prompt = llm.calls[1][-1].content
    assert "Import" in repair_prompt  # the sandbox error was fed back to the model


async def test_planner_rejects_invisible_documents_and_repairs(monkeypatch):
    bad = '```python\nresult = {"batches": [["POL-007"]]}\n```'  # restricted for a viewer
    llm = ScriptedLLM(bad, GOOD_PLAN)
    use_llm(monkeypatch, llm)
    out = await research.research_planner({"messages": [HumanMessage(Q)]}, RUNTIME)
    assert "POL-007" not in out["research"]["doc_ids"]
    assert "not found" in llm.calls[1][-1].content


async def test_planner_falls_back_to_search_when_llm_is_down(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM(TimeoutError("gateway")))

    async def fake_search(self, query, top_k=5, **kw):
        return [{"doc_id": "INC-2026-021"}, {"doc_id": "INC-2026-062"}, {"doc_id": "INC-2026-021"}]

    monkeypatch.setattr(Catalog, "search", fake_search)
    out = await research.research_planner({"messages": [HumanMessage(Q)]}, RUNTIME)
    assert out["research"]["used_fallback"] is True
    assert out["research"]["batches"] == [["INC-2026-021", "INC-2026-062"]]


def test_validate_batches():
    cat = Catalog(VIEWER.clearance)
    assert research.validate_batches({"batches": [["RB-003", "RB-003"], ["RB-001"]]}, cat) == [["RB-003"], ["RB-001"]]
    for bad in ({"x": 1}, {"batches": [[]]}, {"batches": ["RB-003"]}):
        with pytest.raises(ValueError):
            research.validate_batches(bad, cat)
    with pytest.raises(CatalogError):
        research.validate_batches({"batches": [["POL-008"]]}, cat)
    too_many = {"batches": [[d["doc_id"] for d in cat.list_docs()]]}
    with pytest.raises(ValueError, match="maximum"):
        research.validate_batches(too_many, cat)


def test_fan_out_sends_one_task_per_batch():
    sends = research.fan_out({"research": {"turn_id": "t1", "question": Q, "batches": [["A"], ["B", "C"]]}})
    assert all(isinstance(s, Send) and s.node == "research_worker" for s in sends)
    assert [s.arg["batch"] for s in sends] == [["A"], ["B", "C"]]
    assert research.fan_out({"research": {"batches": []}}) == "response_agent"


# ---------- worker ----------

def finding_json(relevant=True, citations=("INC-2026-021#root-cause",)):
    return json.dumps({"relevant": relevant, "date": "2026-03-14", "summary": "s", "cause_category": "no circuit breaker",
                       "cause": "c", "impact": "i", "citations": list(citations)})


async def test_worker_recurses_on_large_batches():
    task = {"turn_id": "t1", "question": Q, "batch": ["A", "B", "C", "D", "E"], "batch_no": "1", "depth": 0}
    cmd = await research.research_worker(task, RUNTIME)
    assert isinstance(cmd, Command)
    assert [s.arg["batch"] for s in cmd.goto] == [["A", "B", "C"], ["D", "E"]]
    assert [s.arg["batch_no"] for s in cmd.goto] == ["1.1", "1.2"] and cmd.goto[0].arg["depth"] == 1


async def test_worker_stops_recursing_at_max_depth(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM(finding_json()))
    task = {"turn_id": "t1", "question": Q, "batch": ["INC-2026-021", "INC-2026-062", "INC-2026-095", "INC-2026-007"],
            "batch_no": "1.1.1", "depth": research.MAX_DEPTH}
    out = await research.research_worker(task, RUNTIME)
    assert len(out["findings"]) == 4  # analysed itself instead of splitting further


async def test_worker_drops_invented_citations(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM(finding_json(citations=["INC-2026-021#root-cause", "INC-9999#made-up"])))
    task = {"turn_id": "t1", "question": Q, "batch": ["INC-2026-021"], "batch_no": "1", "depth": 0}
    f = (await research.research_worker(task, RUNTIME))["findings"][0]
    assert f["citations"] == ["INC-2026-021#root-cause"]
    assert [s["chunk_id"] for s in f["sources"]] == ["INC-2026-021#root-cause"]


async def test_worker_marks_irrelevant_and_failed_documents(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM(finding_json(relevant=False)))
    task = {"turn_id": "t1", "question": Q, "batch": ["INC-2026-055", "POL-007"], "batch_no": "2", "depth": 0}
    found = {f["doc_id"]: f for f in (await research.research_worker(task, RUNTIME))["findings"]}
    assert found["INC-2026-055"]["relevant"] is False
    assert "not found" in found["POL-007"]["error"]  # restricted for viewer: failure marker, no content


async def test_one_bad_llm_reply_fails_only_that_document(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM("not json at all", finding_json()))
    task = {"turn_id": "t1", "question": Q, "batch": ["INC-2026-021", "INC-2026-062"], "batch_no": "1", "depth": 0}
    findings = (await research.research_worker(task, RUNTIME))["findings"]
    assert sum(1 for f in findings if f.get("error")) == 1
    assert sum(1 for f in findings if not f.get("error")) == 1


# ---------- aggregator ----------

def finding(doc, cat, turn="t1", **kw):
    return {"turn_id": turn, "doc_id": doc, "relevant": True, "date": "2026-01-01", "summary": "s",
            "cause_category": cat, "cause": "c", "impact": "i", "citations": [f"{doc}#root-cause"],
            "sources": [{"chunk_id": f"{doc}#root-cause", "title": "t", "section": "Root Cause", "date": "2026-01-01",
                         "text": "x"}], **kw}


async def test_aggregator_waits_until_all_documents_are_covered():
    state = {"research": {"turn_id": "t1", "doc_ids": ["A", "B"], "question": Q}, "findings": [finding("A", "x")]}
    cmd = await research.research_aggregator(state, RUNTIME)
    assert isinstance(cmd, Command) and cmd.goto == []


async def test_aggregator_counts_themes_in_code(monkeypatch):
    # The model groups, and even claims wrong counts – the code recounts from doc ids.
    reply = json.dumps({"themes": [
        {"theme": "Pool exhaustion", "doc_ids": ["A", "B", "Z-invented"], "count": 99},
        {"theme": "Certificates", "doc_ids": ["C", "A"]},  # A already assigned
    ]})
    use_llm(monkeypatch, ScriptedLLM(reply))
    state = {"research": {"turn_id": "t1", "doc_ids": ["A", "B", "C", "D"], "question": Q},
             "findings": [finding("A", "pool"), finding("B", "pool"), finding("C", "cert"),
                          {"turn_id": "t1", "doc_id": "D", "error": "analysis failed"}]}
    cmd = await research.research_aggregator(state, RUNTIME)
    themes = cmd.update["research"]["themes"]
    assert [(t["theme"], t["count"], t["doc_ids"]) for t in themes] == [
        ("Pool exhaustion", 2, ["A", "B"]), ("Certificates", 1, ["C"])]
    assert cmd.goto == "response_agent"
    assert {e["chunk_id"] for e in cmd.update["evidence"]} == {"A#root-cause", "B#root-cause", "C#root-cause"}


async def test_aggregator_falls_back_to_worker_labels(monkeypatch):
    use_llm(monkeypatch, ScriptedLLM(TimeoutError()))
    state = {"research": {"turn_id": "t1", "doc_ids": ["A", "B", "C"], "question": Q},
             "findings": [finding("A", "Pool"), finding("B", "pool "), finding("C", "cert")]}
    themes = (await research.research_aggregator(state, RUNTIME)).update["research"]["themes"]
    assert [(t["theme"], t["count"]) for t in themes] == [("pool", 2), ("cert", 1)]


def test_irrelevant_findings_are_not_themed():
    block_state = {"research": {"turn_id": "t1", "doc_ids": ["A", "B"], "themes": []},
                   "findings": [finding("A", "x"), finding("B", "y", relevant=False)]}
    block = research.research_block(block_state)
    assert "B: excluded – not relevant" in block


def test_findings_reducer_keeps_only_current_turn():
    old = [finding("A", "x", turn="t1")]
    merged = _findings(old, [finding("B", "y", turn="t2")])
    assert [f["doc_id"] for f in merged] == ["B"]
    assert [f["doc_id"] for f in _findings(merged, [finding("C", "z", turn="t2")])] == ["B", "C"]
