"""Token-bucket rate limiting: algorithm with a fake clock, API behaviour, per-role thresholds."""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.security.rate_limit import Limit, TokenBucketLimiter, limit_for_role


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


# ---------- algorithm ----------

async def test_burst_up_to_capacity_then_refuse(clock):
    lim = TokenBucketLimiter(clock)
    limit = Limit(capacity=3, refill_per_sec=0.5)
    results = [await lim.take("u", limit) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results[:3]] == [2, 1, 0]
    assert results[3].retry_after == 2  # one token needs 1 / 0.5 = 2 s


async def test_tokens_refill_over_time(clock):
    lim = TokenBucketLimiter(clock)
    limit = Limit(capacity=2, refill_per_sec=1)
    await lim.take("u", limit)
    await lim.take("u", limit)
    assert not (await lim.take("u", limit)).allowed
    clock.now += 1.0
    assert (await lim.take("u", limit)).allowed
    clock.now += 100  # long idle: refill stops at capacity, no unlimited burst
    assert [(await lim.take("u", limit)).allowed for _ in range(3)] == [True, True, False]


async def test_users_have_separate_buckets(clock):
    lim = TokenBucketLimiter(clock)
    limit = Limit(capacity=1, refill_per_sec=0.1)
    assert (await lim.take("alice", limit)).allowed
    assert not (await lim.take("alice", limit)).allowed
    assert (await lim.take("bob", limit)).allowed


async def test_idle_buckets_are_evicted(clock):
    lim = TokenBucketLimiter(clock)
    await lim.take("u", Limit(1, 1))
    clock.now += 4000
    assert lim.evict_idle() == 1 and lim._buckets == {}


def test_role_overrides(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "rate_limit_role_overrides", {"admin": {"capacity": 30, "refill_per_sec": 0.5}})
    assert limit_for_role("admin") == Limit(30, 0.5)
    assert limit_for_role("viewer") == Limit(s.rate_limit_capacity, s.rate_limit_refill_per_sec)


# ---------- API ----------

def _client_and_headers(monkeypatch, capacity=2, refill=0.1):
    s = get_settings()
    monkeypatch.setattr(s, "rate_limit_capacity", capacity)
    monkeypatch.setattr(s, "rate_limit_refill_per_sec", refill)
    client = TestClient(app)
    token = client.post("/auth/login", json={"username": "viewer1", "password": "Viewer@2026"}).json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def test_chat_returns_429_with_retry_after_before_any_agent_runs(monkeypatch):
    client, headers = _client_and_headers(monkeypatch)
    calls = []

    async def no_graph(*a, **k):
        calls.append(1)
        if False:
            yield {}

    from app.api import chat as chat_api
    monkeypatch.setattr(app.state, "graph", object(), raising=False)  # TestClient without lifespan has no graph
    monkeypatch.setattr(chat_api, "_stream_turn", lambda *a, **k: no_graph())
    monkeypatch.setattr(chat_api, "_pending_approval", lambda *a, **k: _none())
    ok = [client.post("/chat", json={"thread_id": "rl-1", "message": "hi"}, headers=headers) for _ in range(3)]
    assert [r.status_code for r in ok] == [200, 200, 429]
    limited = ok[2]
    assert limited.headers["Retry-After"] == "10"
    assert limited.json()["error"]["code"] == "rate_limited"
    assert "10 seconds" in limited.json()["error"]["message"]
    assert len(calls) == 2  # the limited request never reached the graph


def test_malformed_requests_also_spend_tokens(monkeypatch):
    """Flooding the API with junk is limited too (FastAPI resolves the dependency before body validation fails)."""
    client, headers = _client_and_headers(monkeypatch)
    codes = [client.post("/chat", json={"thread_id": "rl-2", "message": ""}, headers=headers).status_code
             for _ in range(3)]
    assert codes == [422, 422, 429]


async def _none():
    return None


def test_login_is_limited_per_username():
    client = TestClient(app)
    codes = [client.post("/auth/login", json={"username": "admin1", "password": "wrong"}).status_code
             for _ in range(6)]
    assert codes == [401] * 5 + [429]
    # another username is not affected
    assert client.post("/auth/login", json={"username": "viewer1", "password": "Viewer@2026"}).status_code == 200
