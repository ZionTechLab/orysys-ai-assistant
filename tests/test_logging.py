import json

from fastapi.testclient import TestClient

from app.logging import fingerprint, get_logger
from app.main import app


def _json_lines(text: str) -> list[dict]:
    lines = []
    for line in text.splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return lines


def test_request_logged_as_json_with_request_id(capsys):
    client = TestClient(app)
    resp = client.post("/auth/login", json={"username": "viewer1", "password": "Viewer@2026"},
                       headers={"X-Request-ID": "test-req-1"})
    assert resp.headers["X-Request-ID"] == "test-req-1"

    out = capsys.readouterr().out
    events = {e["event"]: e for e in _json_lines(out)}
    assert events["login_succeeded"]["request_id"] == "test-req-1"
    assert events["http_request"]["status"] == 200
    assert "latency_ms" in events["http_request"]
    # The password and the issued token must never reach the logs.
    assert "Viewer@2026" not in out
    assert resp.json()["access_token"] not in out


def test_failed_login_logged_without_password(capsys):
    TestClient(app).post("/auth/login", json={"username": "viewer1", "password": "wrong-pass"})
    out = capsys.readouterr().out
    failed = [e for e in _json_lines(out) if e["event"] == "login_failed"]
    assert failed and failed[0]["reason"] == "bad_password"
    assert "wrong-pass" not in out


def test_secret_fields_are_redacted(capsys):
    get_logger("t").info("example", api_key="sk-123", password="p", ok="visible")
    line = _json_lines(capsys.readouterr().out)[-1]
    assert line["api_key"] == line["password"] == "[REDACTED]"
    assert line["ok"] == "visible"


def test_fingerprint_does_not_contain_text():
    fp = fingerprint("show me restricted documents")
    assert fp["msg_len"] == 28
    assert "restricted" not in str(fp)
