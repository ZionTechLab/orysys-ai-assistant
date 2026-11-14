"""POST /feedback: ownership, validation, update instead of duplicate, LangSmith outage – no network."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import feedback
from app.main import app

PASSWORDS = {"viewer1": "Viewer@2026", "analyst1": "Analyst@2026"}


class FakeLangSmith:
    """Records feedback; the second create for the same feedback id conflicts, like LangSmith."""

    def __init__(self, owner=None, down=False):
        self.owner, self.down = owner, down
        self.created, self.updated = {}, {}

    def read_run(self, run_id):
        if self.owner is None:
            raise RuntimeError("not found")
        from types import SimpleNamespace

        return SimpleNamespace(extra={"metadata": {"user_id": self.owner}})

    def create_feedback(self, run_id, key, *, score, comment, feedback_id, **_):
        from langsmith.utils import LangSmithConflictError

        if self.down:
            raise ConnectionError("langsmith down")
        if feedback_id in self.created:
            raise LangSmithConflictError("exists")
        self.created[feedback_id] = {"run_id": run_id, "key": key, "score": score, "comment": comment}

    def update_feedback(self, feedback_id, *, score, comment):
        self.updated[feedback_id] = {"score": score, "comment": comment}


@pytest.fixture
def client():
    return TestClient(app)


def login(client, user):
    token = client.post("/auth/login", json={"username": user, "password": PASSWORDS[user]}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_owner_can_rate_and_a_second_rating_updates_the_first(client, monkeypatch):
    ls = FakeLangSmith()
    monkeypatch.setattr(feedback, "_langsmith_client", lambda: ls)
    run_id = uuid.uuid4()
    feedback.remember_run(run_id, "viewer1")
    headers = login(client, "viewer1")

    r = client.post("/feedback", json={"run_id": str(run_id), "score": 1}, headers=headers)
    assert r.status_code == 200 and r.json()["stored"] is True
    (stored,) = ls.created.values()
    assert stored["key"] == "user_score" and stored["score"] == 1

    r = client.post("/feedback", json={"run_id": str(run_id), "score": 0, "comment": "wrong runbook"}, headers=headers)
    assert r.json()["stored"] is True and len(ls.created) == 1  # no duplicate
    assert list(ls.updated.values()) == [{"score": 0, "comment": "wrong runbook"}]


def test_users_cannot_rate_other_users_answers(client, monkeypatch):
    monkeypatch.setattr(feedback, "_langsmith_client", lambda: FakeLangSmith())
    run_id = uuid.uuid4()
    feedback.remember_run(run_id, "viewer1")
    r = client.post("/feedback", json={"run_id": str(run_id), "score": 0}, headers=login(client, "analyst1"))
    assert r.status_code == 403


def test_ownership_falls_back_to_the_run_metadata_after_a_restart(client, monkeypatch):
    ls = FakeLangSmith(owner="analyst1")
    monkeypatch.setattr(feedback, "_langsmith_client", lambda: ls)
    r = client.post("/feedback", json={"run_id": str(uuid.uuid4()), "score": 1}, headers=login(client, "analyst1"))
    assert r.status_code == 200 and r.json()["stored"] is True


def test_unknown_run_is_refused(client, monkeypatch):
    monkeypatch.setattr(feedback, "_langsmith_client", lambda: FakeLangSmith(owner=None))
    r = client.post("/feedback", json={"run_id": str(uuid.uuid4()), "score": 1}, headers=login(client, "viewer1"))
    assert r.status_code == 403


def test_langsmith_outage_is_reported_not_raised(client, monkeypatch):
    monkeypatch.setattr(feedback, "_langsmith_client", lambda: FakeLangSmith(down=True))
    run_id = uuid.uuid4()
    feedback.remember_run(run_id, "viewer1")
    r = client.post("/feedback", json={"run_id": str(run_id), "score": 1}, headers=login(client, "viewer1"))
    assert r.status_code == 200 and r.json()["stored"] is False


@pytest.mark.parametrize("body", [{"run_id": "not-a-uuid", "score": 1}, {"run_id": str(uuid.uuid4()), "score": 5},
                                  {"run_id": str(uuid.uuid4()), "score": 1, "comment": "x" * 501}])
def test_invalid_feedback_is_rejected(client, body):
    assert client.post("/feedback", json=body, headers=login(client, "viewer1")).status_code == 422


def test_feedback_needs_login(client):
    assert client.post("/feedback", json={"run_id": str(uuid.uuid4()), "score": 1}).status_code == 401
