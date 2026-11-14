"""Citation parsing: single and grouped brackets, invented ids inside groups, repair."""

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.citations import find_citations, remove_citations
from app.graph.nodes import validator
from app.graph.state import UserContext

RUNTIME = type("R", (), {"context": UserContext("viewer1", "viewer", ("public", "internal"))})()


class FakeLLM:
    async def ainvoke(self, _messages):
        return AIMessage('{"grounded": true, "unsupported": []}')


def test_single_and_grouped_citations_are_found_in_order():
    text = ("Pool exhausted [INC-2026-034#summary, INC-2026-034#root-cause]; certificate expired "
            "[INC-2026-007#root-cause] [RB-003#steps; POL-009#eligibility and MTG-005#notes].")
    assert find_citations(text) == ["INC-2026-034#summary", "INC-2026-034#root-cause", "INC-2026-007#root-cause",
                                    "RB-003#steps", "POL-009#eligibility", "MTG-005#notes"]


def test_other_brackets_are_not_citations():
    text = "See [the runbook](https://example.test) and [1], [sic], [INC-2026-034] (no section)."
    assert find_citations(text) == []
    assert remove_citations(text, set()) == text


def test_invented_id_inside_a_group_is_removed_and_valid_ids_kept():
    text = "Cause [INC-2026-999#summary, INC-2026-034#impact] and [INC-2026-998#x]."
    assert remove_citations(text, {"INC-2026-034#impact"}) == "Cause [INC-2026-034#impact] and ."


def test_duplicates_are_reported_once():
    assert find_citations("[A-1#x] and again [A-1#x, B-2#y]") == ["A-1#x", "B-2#y"]


def answer_state(answer, attempt=0):
    ev = [{"chunk_id": c, "doc_id": c.split("#")[0], "title": "t", "section": "s", "text": "x"}
          for c in ("INC-2026-034#summary", "INC-2026-034#impact")]
    return {"messages": [HumanMessage("q"), AIMessage(answer, id="ans-1")], "plan": {"route": "research"},
            "evidence": ev, "validation": {"attempt": attempt} if attempt else {}}


async def test_validator_catches_an_invented_id_inside_a_group(monkeypatch):
    """Regression (14 Nov): this answer used to pass with problems == []."""
    monkeypatch.setattr(validator, "get_llm", lambda *a, **k: FakeLLM())
    answer = "Root cause found [INC-2026-034#summary, INC-2026-034#impact] and [INC-2026-999#summary, INC-2026-034#impact]."
    cmd = await validator.validator(answer_state(answer), RUNTIME)
    assert cmd.goto == "response_agent"
    assert "INC-2026-999#summary" in cmd.update["validation"]["problems"][0]


async def test_validator_repairs_a_group_on_the_second_failure(monkeypatch):
    monkeypatch.setattr(validator, "get_llm", lambda *a, **k: FakeLLM())
    answer = "Root cause found [INC-2026-999#summary, INC-2026-034#impact]."
    cmd = await validator.validator(answer_state(answer, attempt=1), RUNTIME)
    fixed = cmd.update["messages"][0].content
    assert cmd.update["validation"]["action"] == "repaired"
    assert "INC-2026-999" not in fixed and "[INC-2026-034#impact]" in fixed


async def test_accepted_answer_lists_grouped_citations(monkeypatch):
    monkeypatch.setattr(validator, "get_llm", lambda *a, **k: FakeLLM())
    cmd = await validator.validator(answer_state("Found [INC-2026-034#summary, INC-2026-034#impact]."), RUNTIME)
    assert cmd.update["validation"]["citations"] == ["INC-2026-034#impact", "INC-2026-034#summary"]
