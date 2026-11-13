# ComBank Knowledge Assistant

Internal staff AI assistant for Commercial Bank of Ceylon PLC (assessment project).
Multi-agent LangGraph backend (FastAPI), Streamlit UI, Pinecone hybrid search, LangSmith tracing.

> All documents in `data/docs/` are **fictional sample data** and are not real Commercial Bank of Ceylon information.

*Full README (setup, architecture, design decisions) – in progress.*

- Architecture and diagrams: [`docs/architecture.md`](docs/architecture.md)
- Assumptions and decisions: [`docs/decisions.md`](docs/decisions.md)
- Retrieval (hybrid search, ingestion): [`docs/retrieval.md`](docs/retrieval.md)
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
.venv/Scripts/uvicorn app.main:app --reload
.venv/Scripts/pytest
```
