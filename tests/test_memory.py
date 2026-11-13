"""Conversation memory: rolling summary of older turns, window, user context – fake LLMs, no network."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import builder
from app.graph.nodes import memory, response, supervisor, validator
from app.graph.state import UserContext, new_turn_state
from app.memory.summary import conversation_context, messages_to_summarise

ANALYST = UserContext("analyst1", "analyst", ("public", "internal", "confidential"), department="payments")
RUNTIME = SimpleNamespace(context=ANALYST)


class RecordingLLM:
    """Returns a fixed reply and records the prompts it was given."""

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.prompts = reply, error, []

    async def ainvoke(self, messages):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


def convo(n_turns: int) -> list:
    msgs = []
    for i in range(n_turns):
        msgs += [HumanMessage(f"question {i}", id=f"h{i}"), AIMessage(f"answer {i}", id=f"a{i}")]
    return msgs


# ---------- window and summary arithmetic ----------

def test_short_conversation_is_not_summarised():
    assert messages_to_summarise(convo(4), None) == ([], 0)  # 8 messages: only 2 aged out of the 6-window


def test_older_messages_are_summarised_in_batches():
    aged, covered = messages_to_summarise(convo(5), None)  # 10 messages, window 6 -> 4 aged
    assert [m.content for m in aged] == ["question 0", "answer 0", "question 1", "answer 1"] and covered == 4
    aged, covered = messages_to_summarise(convo(6), {"text": "s", "covered": 4})  # only 2 new aged: wait
    assert aged == [] and covered == 4


def test_agents_see_summary_plus_everything_after_it_no_gap():
    msgs = convo(6) + [HumanMessage("new question")]
    summary, recent = conversation_context(msgs, {"text": "- asked about PayBridge", "covered": 4})
    assert summary == "- asked about PayBridge"
    assert recent[0].content == "question 2" and recent[-1].content == "new question"  # messages 4.. verbatim


def test_window_is_capped_if_summarising_keeps_failing():
    _, recent = conversation_context(convo(20), None)
    assert len(recent) == 16


# ---------- memory node ----------

async def test_memory_node_folds_aged_messages_into_the_summary(monkeypatch):
    llm = RecordingLLM("- User asked about INC-2026-021 (PayBridge timeouts); root cause: no circuit breaker.")
    monkeypatch.setattr(memory, "get_llm", lambda *a, **k: llm)
    out = await memory.memory_manager({"messages": convo(5), "summary": {}}, RUNTIME)
    assert out["summary"]["covered"] == 4 and "INC-2026-021" in out["summary"]["text"]
    assert "question 0" in llm.prompts[0] and "question 4" not in llm.prompts[0]  # recent turns not summarised


async def test_memory_node_extends_the_previous_summary(monkeypatch):
    llm = RecordingLLM("- earlier points\n- new points")
    monkeypatch.setattr(memory, "get_llm", lambda *a, **k: llm)
    out = await memory.memory_manager({"messages": convo(7), "summary": {"text": "- earlier points", "covered": 4,
                                                                         "turns": 1}}, RUNTIME)
    assert "- earlier points" in llm.prompts[0] and "question 2" in llm.prompts[0] and "question 1" not in llm.prompts[0]
    assert out["summary"]["covered"] == 8 and out["summary"]["turns"] == 2


async def test_memory_failure_keeps_the_old_summary(monkeypatch):
    monkeypatch.setattr(memory, "get_llm", lambda *a, **k: RecordingLLM(error=TimeoutError()))
    out = await memory.memory_manager({"messages": convo(5), "summary": {"text": "old", "covered": 0}}, RUNTIME)
    assert "summary" not in out and out["errors"][0]["node"] == "memory"


async def test_no_llm_call_while_nothing_has_aged(monkeypatch):
    llm = RecordingLLM("x")
    monkeypatch.setattr(memory, "get_llm", lambda *a, **k: llm)
    assert await memory.memory_manager({"messages": convo(2)}, RUNTIME) == {}
    assert llm.prompts == []


# ---------- agents use the summary and the user context ----------

async def test_supervisor_gets_summary_and_user_department(monkeypatch):
    llm = RecordingLLM('{"intent": "x", "route": "retrieve", "search_query": "q"}')
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: llm)
    msgs = convo(6) + [HumanMessage("And what fixed it?")]
    await supervisor.supervisor({"messages": msgs, "summary": {"text": "- asked about INC-2026-021", "covered": 4}},
                                RUNTIME)
    prompt = llm.prompts[0]
    assert "INC-2026-021" in prompt and "department payments" in prompt
    assert "question 0" not in prompt and "question 2" in prompt  # summarised turns not repeated verbatim


async def test_response_agent_treats_summary_as_context_not_source(monkeypatch):
    llm = RecordingLLM("Answer [RB-003#actions].")
    monkeypatch.setattr(response, "get_llm", lambda *a, **k: llm)
    state = {"messages": convo(6) + [HumanMessage("What fixed it?")], "plan": {"route": "retrieve"},
             "evidence": [], "summary": {"text": "- asked about INC-2026-021", "covered": 4}}
    await response.response_agent(state, RUNTIME)
    assert "NOT a source" in llm.prompts[0] and "INC-2026-021" in llm.prompts[0]


async def test_summary_survives_turns_in_the_checkpoint(monkeypatch):
    """End to end: after enough turns the graph stores a summary, and new turns do not reset it."""
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: RecordingLLM('{"intent": "hi", "route": "chitchat"}'))
    monkeypatch.setattr(response, "get_llm", lambda *a, **k: RecordingLLM("Hello! How can I help you today?"))
    monkeypatch.setattr(validator, "get_llm", lambda *a, **k: RecordingLLM('{"grounded": true, "unsupported": []}'))
    summariser = RecordingLLM("- greeted the assistant several times")
    monkeypatch.setattr(memory, "get_llm", lambda *a, **k: summariser)

    graph = builder.build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "mem-1"}}
    for i in range(5):
        await graph.ainvoke(new_turn_state(f"hello {i}"), config=cfg, context=ANALYST)
    values = (await graph.aget_state(cfg)).values
    assert len(values["messages"]) == 10 and values["summary"]["covered"] == 4  # full history kept
    assert len(summariser.prompts) == 1  # summarised once, at the turn where 4 messages had aged
    await graph.ainvoke(new_turn_state("hello again"), config=cfg, context=ANALYST)
    assert (await graph.aget_state(cfg)).values["summary"]["text"] == "- greeted the assistant several times"
