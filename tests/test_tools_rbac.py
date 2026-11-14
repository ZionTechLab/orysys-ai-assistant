"""Tool registry, RBAC (binding + execution), audit log, and human-in-the-loop approval."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.graph import builder
from app.graph.nodes import supervisor, tools
from app.graph.state import UserContext, new_turn_state
from app.tools.registry import REGISTRY, Registry, Tool, ToolContext

VIEWER = ToolContext("viewer1", "viewer", ("public", "internal"))
ANALYST = ToolContext("analyst1", "analyst", ("public", "internal", "confidential"))
ADMIN = ToolContext("admin1", "admin", ("public", "internal", "confidential", "restricted"))


class EchoArgs(BaseModel):
    text: str = Field(min_length=1, max_length=20)


async def echo(args, ctx):
    return {"echo": args.text, "by": ctx.user_id}


async def slow(args, ctx):
    await asyncio.sleep(5)


@pytest.fixture
def reg():
    r = Registry()
    r.register(Tool("echo", "echo text", frozenset({"analyst", "admin"}), EchoArgs, echo))
    r.register(Tool("slow", "slow", frozenset({"admin"}), EchoArgs, slow, timeout_s=0.1))
    return r


# ---------- registry ----------

def test_binding_shows_only_permitted_tools(reg):
    assert [t.name for t in reg.tools_for("viewer")] == []
    assert [t.name for t in reg.tools_for("analyst")] == ["echo"]
    assert [t.name for t in reg.tools_for("admin")] == ["echo", "slow"]


async def test_execution_rechecks_the_role(reg):
    denied = await reg.execute("echo", {"text": "hi"}, VIEWER)
    assert denied.status == "denied" and denied.data is None
    ok = await reg.execute("echo", {"text": "hi"}, ANALYST)
    assert ok.ok and ok.data == {"echo": "hi", "by": "analyst1"}


async def test_arguments_are_validated(reg):
    r = await reg.execute("echo", {"text": "x" * 50}, ADMIN)
    assert r.status == "invalid_args" and "text" in r.message


async def test_unknown_tool_and_timeout(reg):
    assert (await reg.execute("drop_tables", {}, ADMIN)).status == "unknown_tool"
    assert (await reg.execute("slow", {"text": "a"}, ADMIN)).status == "timeout"


async def test_every_attempt_is_audited(reg):
    await reg.execute("echo", {"text": "hi"}, VIEWER)
    await reg.execute("echo", {"text": "hi"}, ANALYST)
    assert [(e["user_id"], e["status"]) for e in reg.audit] == [("viewer1", "denied"), ("analyst1", "ok")]


def test_builtin_role_matrix():
    m = REGISTRY.matrix()
    assert m["knowledge_search"] == ["admin", "analyst", "viewer"]
    assert m["admin_reindex"] == ["admin"] and m["admin_view_audit_log"] == ["admin"]
    assert REGISTRY.tools["admin_reindex"].requires_approval


async def test_audit_log_tool_is_admin_only_and_filters():
    denied = await REGISTRY.execute("admin_view_audit_log", {}, ANALYST)
    assert denied.status == "denied"
    r = await REGISTRY.execute("admin_view_audit_log", {"status": "denied", "limit": 5}, ADMIN)
    assert r.ok and all(e["status"] == "denied" for e in r.data) and r.data[0]["user_id"] == "analyst1"


# ---------- tool nodes ----------

class FakeLLM:
    def __init__(self, reply):
        self.reply, self.seen = reply, None

    async def ainvoke(self, messages):
        self.seen = messages
        return SimpleNamespace(content=self.reply)


def runtime(ctx: ToolContext):
    return SimpleNamespace(context=UserContext(ctx.user_id, ctx.role, ctx.clearance))


async def test_tool_agent_only_offers_permitted_tools(monkeypatch):
    llm = FakeLLM(json.dumps({"tool": "knowledge_search", "args": {"query": "paybridge"}}))
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: llm)
    await tools.tool_agent({"messages": [HumanMessage("search paybridge")]}, runtime(VIEWER))
    shown = llm.seen[0].content
    assert "knowledge_search" in shown and "admin_reindex" not in shown


async def test_injected_admin_tool_is_denied_at_execution(monkeypatch):
    """Layer 2: even if the model picks a tool it was not shown (hallucination / prompt injection)."""
    state = {"tool_call": {"tool": "admin_view_audit_log", "args": {}, "reason": "injected"}}
    gate = await tools.approval_gate(state, runtime(VIEWER))
    assert gate == {}  # a human is never asked to approve a forbidden call
    out = await tools.tool_executor(state, runtime(VIEWER))
    assert out["tool_result"]["status"] == "denied" and out["tool_result"]["data"] is None


async def test_no_tool_and_rejection():
    out = await tools.tool_executor({"tool_call": {"tool": None, "reason": "nothing fits"}}, runtime(ADMIN))
    assert out["tool_result"]["status"] == "no_tool"
    out = await tools.tool_executor({"tool_call": {"tool": "admin_reindex", "args": {}, "approved": False}},
                                    runtime(ADMIN))
    assert out["tool_result"]["status"] == "rejected"


# ---------- human-in-the-loop through the real graph ----------

@pytest.fixture
def tool_graph(monkeypatch):
    """Graph with fake LLMs: supervisor routes to tool, tool agent picks admin_reindex (dry run)."""
    route = json.dumps({"intent": "reindex", "route": "tool", "search_query": "", "reasoning": "action"})
    pick = json.dumps({"tool": "admin_reindex", "args": {"dry_run": True}, "reason": "user asked to re-index"})
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: FakeLLM(route))
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: FakeLLM(pick))

    class Answer:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="done")

    from app.graph.nodes import response
    monkeypatch.setattr(response, "get_llm", lambda *a, **k: Answer())
    ran = []

    async def fake_reindex(args, ctx):
        ran.append(args.dry_run)
        return {"added": [], "dry_run": args.dry_run}

    original = REGISTRY.tools["admin_reindex"]  # Tool is frozen: swap in a copy with a fake handler
    REGISTRY.tools["admin_reindex"] = Tool(original.name, original.description, original.roles,
                                           original.args_model, fake_reindex, original.timeout_s, True)
    yield builder.build_graph(InMemorySaver()), ran
    REGISTRY.tools["admin_reindex"] = original


async def _run(graph, graph_input, ctx, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}}
    await graph.ainvoke(graph_input, config=cfg, context=runtime(ctx).context)
    return await graph.aget_state(cfg)


async def test_admin_tool_pauses_for_approval_then_runs(tool_graph):
    graph, ran = tool_graph
    snap = await _run(graph, new_turn_state("please re-index the documents"), ADMIN)
    assert snap.interrupts and snap.interrupts[0].value["tool"] == "admin_reindex"
    assert ran == []  # nothing ran before approval

    snap = await _run(graph, Command(resume={"approved": True, "by": "admin1"}), ADMIN)
    assert not snap.interrupts and ran == [True]
    assert snap.values["tool_result"]["status"] == "ok"


async def test_rejected_approval_runs_nothing(tool_graph):
    graph, ran = tool_graph
    await _run(graph, new_turn_state("please re-index"), ADMIN, thread="t2")
    snap = await _run(graph, Command(resume={"approved": False, "by": "admin1"}), ADMIN, thread="t2")
    assert ran == [] and snap.values["tool_result"]["status"] == "rejected"


async def test_viewer_is_never_asked_to_approve_admin_tool(tool_graph):
    graph, ran = tool_graph  # fake tool agent picks admin_reindex even for the viewer
    snap = await _run(graph, new_turn_state("please re-index"), VIEWER, thread="t3")
    assert not snap.interrupts and ran == []
    assert snap.values["tool_result"]["status"] == "denied"


# ---------- unusable model replies are retried (seen live: truncated JSON from the gateway) ----------

class _Replies:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        from types import SimpleNamespace
        return SimpleNamespace(content=self.replies.pop(0))


async def test_tool_agent_asks_again_after_a_truncated_reply(monkeypatch):
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    from app.graph.nodes import tools as tool_nodes
    from app.graph.state import UserContext

    llm = _Replies('{"tool-index to preview what would change."}',           # truncated – invalid JSON
                   '{"args": {"dry_run": true}, "reason": "preview"}',         # valid JSON, "tool" missing
                   '{"tool": "admin_reindex", "args": {"dry_run": true}, "reason": "preview"}')
    monkeypatch.setattr(tool_nodes, "get_llm", lambda *a, **k: llm)
    admin = SimpleNamespace(context=UserContext("admin1", "admin", ("public", "internal", "confidential", "restricted")))
    out = await tool_nodes.tool_agent({"messages": [HumanMessage("Re-index (dry run)")]}, admin)
    assert out["tool_call"] == {"tool": "admin_reindex", "args": {"dry_run": True}, "reason": "preview"}
    assert len(llm.prompts) == 3 and "could not be used" in llm.prompts[1][-1].content  # told what was wrong


async def test_tool_agent_falls_back_after_three_unusable_replies(monkeypatch):
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    from app.graph.nodes import tools as tool_nodes
    from app.graph.state import UserContext

    monkeypatch.setattr(tool_nodes, "get_llm", lambda *a, **k: _Replies("oops", "{broken", '{"reason": "x"}'))
    admin = SimpleNamespace(context=UserContext("admin1", "admin", ("public",)))
    out = await tool_nodes.tool_agent({"messages": [HumanMessage("Re-index")]}, admin)
    assert out["tool_call"]["tool"] is None and out["errors"][0]["node"] == "tool_agent"


async def test_supervisor_asks_again_after_an_unusable_reply(monkeypatch):
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    from app.graph.nodes import supervisor
    from app.graph.state import UserContext

    llm = _Replies('{"intent": "reindex", "rou', '{"intent": "reindex", "route": "tool"}')
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: llm)
    out = await supervisor.supervisor({"messages": [HumanMessage("Re-index the library")]},
                                      SimpleNamespace(context=UserContext("admin1", "admin", ("public",))))
    assert out["plan"]["route"] == "tool" and len(llm.prompts) == 2
