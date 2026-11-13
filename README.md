# ComBank Knowledge Assistant

Internal staff AI assistant for Commercial Bank of Ceylon PLC (assessment project).
Multi-agent LangGraph backend (FastAPI), Streamlit UI, Pinecone hybrid search, LangSmith tracing.

> All documents in `data/docs/` are **fictional sample data** and are not real Commercial Bank of Ceylon information.

*Full README (setup, architecture, design decisions) – in progress.*

- Architecture and diagrams: [`docs/architecture.md`](docs/architecture.md)
- Assumptions and decisions: [`docs/decisions.md`](docs/decisions.md)
- Agents, state, memory, streaming, tracing: [`docs/agents.md`](docs/agents.md)
- Retrieval (hybrid search, ingestion): [`docs/retrieval.md`](docs/retrieval.md)
- Security, RBAC, approval, audit: [`docs/security.md`](docs/security.md)
- Resilience, error handling, fault injection: [`docs/resilience.md`](docs/resilience.md)
- Logging: [`docs/logging.md`](docs/logging.md)
- Brand guardrails: [`docs/brand_guardrails.md`](docs/brand_guardrails.md)

## Demo users

Hardcoded users (brief "Option A"). Demo only – passwords are stored as bcrypt hashes in `app/security/auth.py`.

| Username | Password | Role | Can read |
|---|---|---|---|
| `viewer1` | `Viewer@2026` | Viewer | public, internal |
| `analyst1` | `Analyst@2026` | Analyst | + confidential |
| `admin1` | `Admin@2026` | Administrator | + restricted |

## Run (dev)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip on Linux/macOS
cp .env.example .env                              # fill in keys
.venv/Scripts/python scripts/smoke_llm.py        # check the LLM gateway (JSON + streaming)
.venv/Scripts/python scripts/ingest.py           # create index + sync documents (--dry-run, --full)
.venv/Scripts/python -m mcp_server               # MCP server on 127.0.0.1:8765 (needs MCP_API_KEY in .env)
.venv/Scripts/uvicorn app.main:app --reload       # API on :8000
.venv/Scripts/streamlit run ui/streamlit_app.py --server.address 127.0.0.1  # UI on localhost:8501 only
.venv/Scripts/pytest
```

## Tests

287 tests, no network needed (fake models, in-process MCP session, in-memory checkpointer):

```bash
.venv/Scripts/pytest                                                   # all tests
.venv/Scripts/python -m coverage run --source=app,mcp_server -m pytest  # with coverage
.venv/Scripts/python -m coverage report
```

Line coverage is 88% (13 Nov). The lower figures are network clients (`pinecone_store`, `sync`, `health`, LLM factory), which are exercised by the live runs recorded in the docs, and the sandbox: plans run under the sandbox's own trace function, which replaces the coverage tracer in that thread, so the size-check helpers are tested but not counted.

| Area | Test files |
|---|---|
| Auth, RBAC, approval | `test_auth`, `test_tools_rbac`, `test_mcp` |
| Rate limiting | `test_rate_limit` |
| Prompt injection, validator, output guard | `test_guards`, `test_guard_validator`, `test_output_guard` |
| Sandbox (escapes, budgets, memory) | `test_sandbox`, `test_analysis_tool` |
| Hybrid scoring, filters, ingestion | `test_retrieval`, `test_loaders`, `test_sync`, `test_retrieval_resilience` |
| Graph, RLM, memory | `test_graph`, `test_research`, `test_catalog`, `test_memory` |
| API streaming, errors, fault injection | `test_chat_api`, `test_faults`, `test_logging` |
