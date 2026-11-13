"""python_analysis tool: helpers, generated code in the sandbox, repair loop, RBAC (fake LLM)."""

from types import SimpleNamespace

import pytest

from app.tools import analysis
from app.tools.registry import REGISTRY, ToolContext

VIEWER = ToolContext("viewer1", "viewer", ("public", "internal"))
ANALYST = ToolContext("analyst1", "analyst", ("public", "internal", "confidential"))


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


@pytest.fixture
def llm(monkeypatch):
    def install(*replies):
        fake = ScriptedLLM(*replies)
        monkeypatch.setattr("app.llm.factory.get_llm", lambda *a, **k: fake)
        return fake
    return install


# ---------- helpers ----------

def test_helpers():
    docs = [{"department": "payments", "tags": ["a", "b"]}, {"department": "cards", "tags": ["a"]},
            {"department": "payments", "tags": []}]
    assert analysis.count_by(docs, "department") == {"payments": 2, "cards": 1}
    assert analysis.count_by(docs, "tags") == {"a": 2, "b": 1}
    assert analysis.mean([1, 2, 3, 4]) == 2.5 and analysis.median([5, 1, 3]) == 3
    assert analysis.mean([]) is None
    assert analysis.find_numbers("~22,400 declined over 47 minutes, 1.5 h") == [22400.0, 47.0, 1.5]


# ---------- tool ----------

COUNT_CODE = '''```python
docs = list_docs(document_type="incident", since="2026-01-01")
result = {"by_department": count_by(docs, "department"),
          "sev1": sorted(d["doc_id"] for d in docs if d["severity"] == "SEV1")}
```'''


async def test_analysis_runs_generated_code(llm):
    fake = llm(COUNT_CODE)
    r = await REGISTRY.execute("python_analysis", {"question": "incidents per department in 2026"}, ANALYST)
    assert r.ok, r.message
    assert r.data["result"]["by_department"]["payments"] >= 5
    assert r.data["result"]["sev1"] == ["INC-2026-021", "INC-2026-078"]
    assert "count_by" in r.data["code"] and r.data["attempts"] == [{"attempt": 1, "ok": True, "error": None}]
    assert len(fake.calls) == 1


async def test_analysis_repairs_failing_code(llm):
    fake = llm("```python\nimport os\nresult = 1\n```", COUNT_CODE)
    r = await REGISTRY.execute("python_analysis", {"question": "incidents per department in 2026"}, ANALYST)
    assert r.ok and len(r.data["attempts"]) == 2 and not r.data["attempts"][0]["ok"]
    assert "Import" in fake.calls[1][-1].content  # sandbox error fed back to the model


async def test_analysis_gives_up_after_three_attempts(llm):
    llm(*["```python\nwhile True:\n    pass\n```"] * 3)
    r = await REGISTRY.execute("python_analysis", {"question": "loop forever please"}, ANALYST)
    assert r.status == "error"


async def test_analysis_respects_clearance(llm):
    llm('```python\nresult = [d["doc_id"] for d in list_docs(document_type="policy")]\n```')
    r = await REGISTRY.execute("python_analysis", {"question": "list policies"}, ANALYST)
    assert "POL-007" not in r.data["result"] and "POL-008" not in r.data["result"]  # restricted
    assert "RB-008" not in r.data["result"]  # not a policy – sanity


async def test_viewer_cannot_use_analysis(llm):
    fake = llm(COUNT_CODE)
    r = await REGISTRY.execute("python_analysis", {"question": "incidents per department"}, VIEWER)
    assert r.status == "denied" and fake.calls == []  # denied before any LLM call


def test_tool_is_bound_only_for_analysts_and_admins():
    assert "python_analysis" not in [t.name for t in REGISTRY.tools_for("viewer")]
    assert "python_analysis" in [t.name for t in REGISTRY.tools_for("analyst")]
    assert "python_analysis" in [t.name for t in REGISTRY.tools_for("admin")]
