from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def _login(username: str, password: str):
    return client.post("/auth/login", json={"username": username, "password": password})


@pytest.mark.parametrize(
    "username,password,role,clearance",
    [
        ("viewer1", "Viewer@2026", "viewer", ["public", "internal"]),
        ("analyst1", "Analyst@2026", "analyst", ["public", "internal", "confidential"]),
        ("admin1", "Admin@2026", "admin", ["public", "internal", "confidential", "restricted"]),
    ],
)
def test_login_returns_role_and_clearance(username, password, role, clearance):
    resp = _login(username, password)
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["role"] == role
    assert body["user"]["clearance"] == clearance

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["user_id"] == username


def test_wrong_password_and_unknown_user_give_same_error():
    wrong = _login("viewer1", "nope")
    unknown = _login("ghost", "nope")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_invalid_username_format_rejected():
    resp = _login("bad user; drop", "x")
    assert resp.status_code == 422


def test_me_requires_token():
    assert client.get("/auth/me").status_code == 401


def test_tampered_role_is_rejected():
    """A viewer cannot promote themselves by editing the token."""
    secret = "an-attacker-chosen-secret-that-is-not-the-servers"
    forged = jwt.encode({"sub": "viewer1", "role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}, secret)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_role_mismatch_with_server_record_is_rejected():
    """Even a correctly signed token is rejected if its role differs from the server's user list."""
    secret = get_settings().jwt_secret.get_secret_value()
    token = jwt.encode({"sub": "viewer1", "role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}, secret)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_expired_token_rejected():
    secret = get_settings().jwt_secret.get_secret_value()
    token = jwt.encode({"sub": "viewer1", "role": "viewer", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)}, secret)
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "expired" in resp.json()["error"]["message"].lower()
