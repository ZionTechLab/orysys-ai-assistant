"""Fault injection and graceful degradation (no network: faults fire before any connection)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from app import faults
from app.graph.nodes import retrieval
from app.graph.state import UserContext
from app.llm.factory import GatewayChatModel
from app.main import app

VIEWER = UserContext("viewer1", "viewer", ("public", "internal"))


@pytest.fixture(autouse=True)
def all_faults_off():
    faults.set_faults({name: False for name in faults.KNOWN})
    yield
    faults.set_faults({name: False for name in faults.KNOWN})


def _token(client, user, password):
    return client.post("/auth/login", json={"username": user, "password": password}).json()["access_token"]


def test_only_admins_can_switch_faults():
    client = TestClient(app)
    viewer = {"Authorization": f"Bearer {_token(client, 'viewer1', 'Viewer@2026')}"}
    admin = {"Authorization": f"Bearer {_token(client, 'admin1', 'Admin@2026')}"}
    assert client.post("/admin/faults", json={"pinecone": True}, headers=viewer).status_code == 403
    assert not faults.active("pinecone")
    r = client.post("/admin/faults", json={"pinecone": True}, headers=admin)
    assert r.status_code == 200 and r.json()["pinecone"] is True
    assert client.get("/admin/faults", headers=admin).json() == {"pinecone": True, "llm": False, "mcp": False}


def test_unknown_fault_rejected():
    with pytest.raises(ValueError):
        faults.set_faults({"database": True})


async def test_llm_fault_fails_fast_without_network():
    faults.set_faults({"llm": True})
    model = GatewayChatModel(base_url="http://127.0.0.1:9", api_key="x", model="m", max_retries=0)
    with pytest.raises(faults.SimulatedOutage):
        await model.ainvoke([HumanMessage("hi")])
    with pytest.raises(faults.SimulatedOutage):
        async for _ in model.astream([HumanMessage("hi")]):
            pass


async def test_pinecone_fault_degrades_to_local_search():
    faults.set_faults({"pinecone": True})
    state = {"messages": [HumanMessage("How do we handle PayBridge degradation?")],
             "plan": {"search_query": "PayBridge degradation circuit breaker"}}
    out = await retrieval.retrieval_agent(state, SimpleNamespace(context=VIEWER))
    assert out["degraded"] == {"pinecone_down": True}
    assert out["evidence"], "local keyword search should still find documents"
    assert any(e["doc_id"] in ("RB-003", "INC-2026-021", "INC-2026-062") for e in out["evidence"])


async def test_supervisor_survives_llm_outage(monkeypatch):
    """LLM down: the supervisor's node fallback searches with the user's own words."""
    from app.graph.nodes import supervisor

    faults.set_faults({"llm": True})
    model = GatewayChatModel(base_url="http://127.0.0.1:9", api_key="x", model="m", max_retries=0)
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: model)
    out = await supervisor.supervisor({"messages": [HumanMessage("Why did PayBridge fail?")]},
                                      SimpleNamespace(context=VIEWER))
    assert out["plan"]["route"] == "retrieve" and out["errors"][0]["error_type"] == "SimulatedOutage"


def test_health_reports_simulated_faults(monkeypatch):
    from app.api import health

    async def ok():
        return "ok"

    for name in ("_check_llm_gateway", "_check_pinecone", "_check_mcp"):
        monkeypatch.setattr(health, name, ok)
    faults.set_faults({"mcp": True})
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded" and body["simulated_faults"] == ["mcp"]


async def test_llm_outage_lists_found_documents_instead_of_only_apologising(monkeypatch):
    from app.graph.nodes import response, validator

    faults.set_faults({"llm": True})
    model = GatewayChatModel(base_url="http://127.0.0.1:9", api_key="x", model="m", max_retries=0)
    monkeypatch.setattr(response, "get_llm", lambda *a, **k: model)
    evidence = [{"chunk_id": "RB-003#actions", "doc_id": "RB-003", "title": "PayBridge gateway degradation handling",
                 "section": "Actions", "text": "t", "score": 0.5, "document_type": "runbook", "created_date": "x"}]
    state = {"messages": [HumanMessage("How do we handle PayBridge degradation?")], "plan": {"route": "retrieve"},
             "evidence": evidence}
    out = await response.response_agent(state, SimpleNamespace(context=VIEWER))
    text = out["messages"][0].content
    assert "RB-003" in text and "[RB-003#actions]" in text and out["degraded"] == {"llm_unavailable": True}

    # The validator accepts it instead of retrying the unavailable model.
    state = {**state, "messages": state["messages"] + out["messages"], "degraded": out["degraded"]}
    cmd = await validator.validator(state, SimpleNamespace(context=VIEWER))
    assert cmd.goto == "output_guard" and "unavailable" in cmd.update["validation"]["grounding"]
