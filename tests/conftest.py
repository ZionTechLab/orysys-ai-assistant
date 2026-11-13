"""Test-wide settings. Runs before any test module imports the app."""

import os

# Unit tests must never send traces to LangSmith (no network, no test noise in the project).
# Environment variables override .env in app.config.Settings.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_TRACING_V2", None)
