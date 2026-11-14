# Logging

## Mechanism

- **Library:** [structlog](https://www.structlog.org/) on top of Python `logging`.
- **Format:** one JSON object per line to stdout (container-friendly; any log platform can ingest it). In `APP_ENV=dev` a readable console format can be used instead.
- **Context:** `request_id`, `user_id`, `role` and `thread_id` are bound once per request (`contextvars`) and added to every line automatically, including lines from agents and tools.
- **Correlation:** the same `request_id` is returned to the client (`X-Request-ID` header and error body) and attached to the LangSmith trace, so an error seen by a user can be found in logs and traces.

## Logs vs LangSmith

| | Logs | LangSmith |
|---|---|---|
| Purpose | Operations: health, errors, security events, performance | Debugging AI behaviour: prompts, outputs, agent steps |
| Content | Metadata only | Full conversation content |
| Audience | Ops / security | AI engineers / evaluator |

Keeping conversation content out of logs keeps them small, safe to share with operations teams, and easy to query.

## Example

```json
{"ts":"2026-11-13T10:42:01Z","level":"info","event":"tool_call","request_id":"6f1c...","user_id":"analyst1","role":"analyst","thread_id":"t-12","node":"tool_agent","tool":"mcp_incidents","latency_ms":412,"status":"ok"}
```

## What is logged

| Area | Events |
|---|---|
| HTTP | `http_request` – method, path, status, latency |
| Auth | `login_succeeded`, `login_failed` (username only) |
| Security | `rate_limited`, `input_blocked` (rule), `rbac_denied` (user, tool), `output_guard_modified` |
| Agents | `node_start` / `node_end`, `route_selected`, `validation_result`, `node_retry` |
| Retrieval | `retrieval` – namespaces, filters, hit count, latency; `retrieval_fallback` (Pinecone → local BM25) |
| Tools / MCP | `tool_audit` (every attempt: user, role, tool, status, latency), `tool_unavailable` (dependency down), `tool_failed` (with stack trace); MCP server: `mcp_tool_call` with the end user and request id |
| Memory | `memory_summary` activity event (messages folded, summary size) |
| Feedback | `answer_feedback` (run id, score, comment length – never the comment), `feedback_not_stored` (LangSmith unavailable) |
| RLM | `rlm_plan`, `rlm_sandbox_error`, `rlm_batches`, `rlm_worker_done`, `rlm_aggregate` |
| Tools / LLM | `tool_call`, `tool_timeout`, `llm_call` – model, latency, tokens; `llm_fallback` |
| Errors | `app_error`, `unhandled_error` with stack trace (server side only, never returned to the client) |

## What is never logged

- Passwords, tokens, API keys (settings use `SecretStr`; a processor also redacts known secret field names).
- Full user messages and answers – only length and a short hash (`msg_len`, `msg_hash`) to correlate repeats.
- Document content.

## Levels

- `info` – normal events above.
- `warning` – handled problems: guard blocks, RBAC denials, rate limits, fallbacks, retries.
- `error` – unexpected exceptions, dependency failures after retries.
- Level set by `LOG_LEVEL`.
