"""Test-wide settings. Runs before any test module imports the app."""

import os

# Unit tests must never send traces to LangSmith (no network, no test noise in the project).
# Environment variables override .env in app.config.Settings.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_TRACING_V2", None)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Each test starts with full buckets (tests log in many times)."""
    from app.security.rate_limit import CHAT_LIMITER, LOGIN_LIMITER

    for limiter in (CHAT_LIMITER, LOGIN_LIMITER):
        limiter._buckets.clear()
    yield
