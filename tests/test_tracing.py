"""LangSmith spans for tool calls: name, run type, inputs and the recorded outcome."""

import asyncio
from contextlib import contextmanager

import pytest
from pydantic import BaseModel

from app import tracing
from app.tools import registry as registry_module
from app.tools.registry import Registry, Tool, ToolContext

ANALYST = ToolContext("analyst1", "analyst", ("public", "internal"))
VIEWER = ToolContext("viewer1", "viewer", ("public",))


class NoArgs(BaseModel):
    pass


async def ok(_args, _ctx):
    return {"rows": 3}


async def boom(_args, _ctx):
    raise RuntimeError("down")


class FakeRun:
    def __init__(self, name, run_type, inputs, metadata, tags):
        self.name, self.run_type, self.inputs, self.metadata, self.tags = name, run_type, inputs, metadata, tags
        self.outputs, self.error = None, None

    def end(self, outputs=None, error=None):
        self.outputs, self.error = outputs, error


@pytest.fixture
def spans(monkeypatch):
    recorded: list[FakeRun] = []

    @contextmanager
    def fake_span(name, run_type="chain", inputs=None, metadata=None, tags=None):
        run = FakeRun(name, run_type, inputs, metadata, tags)
        recorded.append(run)
        yield run

    monkeypatch.setattr(registry_module, "span", fake_span)
    return recorded


@pytest.fixture
def reg():
    r = Registry()
    r.register(Tool("lookup", "ok tool", frozenset({"analyst"}), NoArgs, ok))
    r.register(Tool("broken", "failing tool", frozenset({"analyst"}), NoArgs, boom))
    return r


def test_tool_call_is_a_tool_span_with_args_and_result(reg, spans):
    asyncio.run(reg.execute("lookup", {}, ANALYST))
    (run,) = spans
    assert (run.name, run.run_type) == ("tool:lookup", "tool")
    assert run.inputs == {"tool": "lookup", "args": {}} and run.metadata["role"] == "analyst"
    assert run.outputs["status"] == "ok" and run.outputs["rbac"] == "allowed" and run.outputs["data"] == {"rows": 3}
    assert run.error is None


def test_denied_call_is_recorded_as_rbac_denied_not_as_an_error(reg, spans):
    asyncio.run(reg.execute("lookup", {}, VIEWER))
    assert spans[0].outputs["rbac"] == "denied" and spans[0].error is None


def test_failing_tool_marks_the_span_as_error(reg, spans):
    result = asyncio.run(reg.execute("broken", {}, ANALYST))
    assert result.status == "error" and spans[0].error.startswith("error:")


def test_human_rejection_is_traced(reg, spans):
    reg.record_rejection("lookup", ANALYST)
    assert spans[0].outputs["status"] == "rejected" and spans[0].tags == ["rejected_by_human"]


def test_preview_truncates_large_values():
    assert tracing.preview({"a": 1}) == {"a": 1}
    big = tracing.preview(["x" * 100] * 100, limit=50)
    assert isinstance(big, str) and big.endswith("…[truncated]")


def test_spans_are_safe_with_tracing_off():
    with tracing.span("tool:x", "tool", inputs={"a": 1}) as run:  # tests run with LANGSMITH_TRACING=false
        run.end(outputs={"ok": True})
