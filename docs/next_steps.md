# Next Steps

| # | Step                           | How                                                                                                                                                                                              | Details                                |
| - | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| 1 | **SharePoint sync**      | A Microsoft Graph webhook reports a changed document, a worker runs the same sync as today, and the access level comes from SharePoint permissions. A nightly full sync catches anything missed. | `retrieval.md` – D16                |
| 2 | **Long sections**        | Split sections over ~1,200 characters into parts with a small overlap (`POL-009#eligibility-2`); `read_section` joins them back.                                                             | `retrieval.md` – Future enhancement |
| 3 | **Long-term memory**     | A per-user store (LangGraph`Store`) of preferences and frequent topics, read by the supervisor.                                                                                                | `agents.md` – D11                   |
| 4 | **Conversation history** | `GET /threads` lists a user's saved conversations; the sidebar shows them and a click reopens one.                                                                                             | `agents.md` – D11                   |
| 5 | **Reranker**             | Rerank the top ~20 search hits down to 8 with Pinecone's hosted reranker (`bge-reranker-v2-m3`). If it fails, keep the normal order. Re-tune the 0.25 score cut-off.                           | –                                     |
| 6 | **Docker Compose**       | Three services (api, ui, mcp) and a volume for the data, started with one command.                                                                                                               | `architecture.md` – D14             |
| 7 | **SSO integration**      | Replace the hardcoded users with the bank's identity provider (OpenID Connect, e.g. Entra ID or Keycloak). Role and clearance come from the user's groups. Only the login and `get_current_user` change. | A6 |
| 8 | **Redis and Postgres**   | Move conversation memory from SQLite to Postgres (LangGraph `AsyncPostgresSaver`), and rate limits, the audit log and fault switches from in-memory to Redis – so several API instances can run. | A6, A7 |

**Before real bank data** (see A1): self-hosted or region-pinned services, TLS, secrets in a vault, and a separate process for the sandbox.
