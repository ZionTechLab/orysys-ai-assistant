# Assumptions and Decisions

The main assumptions and design decisions, each with why and what it costs. Details are in the linked docs.

| #  | Topic                                                            |
| -- | ---------------------------------------------------------------- |
| A1 | [Audience, data and deployment](#a1-audience-data-and-deployment) |
| A2 | [Documents and search](#a2-documents-and-search)                  |
| A3 | [Model and gateway](#a3-model-and-gateway)                        |
| A4 | [Agents and failure handling](#a4-agents-and-failure-handling)    |
| A5 | [RLM research and the sandbox](#a5-rlm-research-and-the-sandbox)  |
| A6 | [Security and access control](#a6-security-and-access-control)    |
| A7 | [Memory](#a7-memory)                                              |
| A8 | [Observability and feedback](#a8-observability-and-feedback)      |
| A9 | [Scope: what's built and what's next](#a9-scope-whats-built-and-whats-next)         |

---

## A1. Audience, data and deployment

**Assumptions:**

- The assistant is for **Commercial Bank of Ceylon staff**, not customers – the brief talks about internal documents and employees. Tone and brand rules follow the bank's public website (`docs/brand_guardrails.md`); no logo or official imagery is used.
- **All data is fictional** and marked as such. That is why cloud services are acceptable here: Pinecone stores document text, and LangSmith traces hold questions, prompts and answers.
- It runs **on one machine** (127.0.0.1, no HTTPS), in **English only**.

**Trade-off:** With real bank data, the first things to change are where data goes (self-hosted or region-pinned services, masking what is traced), TLS and network controls, secrets in a vault instead of `.env`, and Sinhala / Tamil support. Customer concerns (KYC, PII masking) are out of scope.

## A2. Documents and search

**Decision:** 45 fictional documents, one folder per type under `data/docs/`:

| Type                   | Folder             | Count |
| ---------------------- | ------------------ | ----- |
| Incident reports       | `incident/`      | 12    |
| Runbooks               | `runbook/`       | 8     |
| Policies               | `policy/`        | 8     |
| Architecture documents | `architecture/`  | 7     |
| Product specifications | `product_spec/`  | 5     |
| Meeting notes          | `meeting_notes/` | 5     |

- Each file is Markdown with a header for department, type, access level (public, internal, confidential, restricted), date and tags. Four are Word or PDF instead (RB-002, POL-004 `.docx`; INC-2026-078, ARCH-001 `.pdf`) to show real formats load the same way (`app/retrieval/loaders.py`).
- The incidents share three planned root causes, so the research answer can be checked. MTG-005 hides an injected instruction for the security demo.
- **Search:** Pinecone, one index, a namespace per department, one chunk per section. Each search mixes meaning (dense, 60 %) and keywords (BM25, 40 %), with the user's access level as a filter. Document ids are in the indexed text, and a question that names an id also searches inside that document.
- **Sync:** a hash per document decides what to re-embed, and deleted documents are removed. Admins run it with `admin_reindex`. In production the documents would live in **SharePoint**: a Graph webhook reports changes, a worker runs the same sync, and the access level comes from SharePoint permissions (D16).

**Trade-off:** A small set of documents. Long sections are not split yet, the 60/40 mix and the 0.25 score cut-off were set by hand, and there is no reranker. Tables, images and scanned PDFs are not handled. Details: `docs/retrieval.md`.

## A3. Model and gateway

**Decision:** All model calls go through **9router**, an OpenAI-compatible gateway I already had running. In it I created a virtual model, `orysys-fast`: **DeepSeek v4.1 Flash, with Gemini as fallback**. The code asks for a `strong` or `fast` model; today both point to it (`.env`). The supervisor and tool agent reply in JSON checked against a schema – an unusable reply is asked again, up to 3 times. Calls time out after 20 s (120 s for long ones like research plans).

**Why:** It saved setup time and costs nothing. The app only knows a URL and a model name, so both can change without code changes.

**Trade-off:** It is a reasoning model, so it is slow – a research plan takes 40–60 s. Production would use LiteLLM or a managed gateway, and a small fast model for the `fast` tier.

## A4. Agents and failure handling

**Decision:** Each agent does one job, and LangGraph connects them (D1).

- **The user's access is never in the agents' hands.** Role and clearance come from the login token on every run. An agent can't change them, even if it's tricked.
- **One failure doesn't break everything.** If an agent fails, the error is recorded and it hands over a safe fallback, so an answer still comes. For example, if search is down, a local keyword search is used. Admins can switch off services to show this live.

**Why:** Small agents are easier to test and explain, and the brief asks that one failure doesn't spread (the "butterfly effect").

**Trade-off:** One exception: if the final safety check (the output guard) fails, the answer is held back. For a bank, showing nothing is safer than showing something unchecked.

## A5. RLM research and the sandbox

**Decision:** Big questions across many documents don't go into one huge prompt. Instead:

1. The model **writes a small Python program** to find the right documents.
2. The program runs in a **sandbox** – it can only use safe Python and only see documents the user is allowed to see.
3. The documents are **split into small batches**, and **sub-agents read them in parallel** – only the sections that matter. A batch that is too big is split again.
4. The model groups the causes into themes, and **the code does the counting**, so the numbers are exact.

The `python_analysis` tool uses the same sandbox. Diagram D4.

**Why:** This is the brief's RLM example. Reading only what's needed keeps prompts small, and each step can be seen in the panel and the trace.

**Trade-off:** It's slow on the free model – about 4 minutes for the example question. The sandbox runs inside the API process; production would run it in a separate, locked-down process.

## A6. Security and access control

| Control          | How                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Login            | three hardcoded users (brief Option A), bcrypt hashes, an 8-hour JWT with role and clearance                               |
| Documents        | the user's access level filters every search and the RLM catalog                                                           |
| Tools            | a tool is only shown to roles allowed to use it, checked again when it runs, and every attempt is audited                  |
| Approval         | sensitive tools (`admin_reindex`) pause the run with LangGraph `interrupt()` until a human approves                    |
| MCP              | a separate server; three allow-listed tools; a service key; results filtered by clearance again and treated as untrusted   |
| Prompt injection | clear attacks blocked by rules; suspicious wording sent to a small classifier; document text cleaned and marked as data    |
| Answers          | the validator checks every citation and grounding; the output guard removes secrets, prompt text and restricted references |
| Limits           | token bucket per user (burst 10, then 12 per minute, higher for admins), plus a login limiter                              |

**Trade-off:** No SSO – production would use the bank's identity provider. The requester can approve their own action; a bank would want a second person. Rules and markers will miss some wording, which is why access control never depends on them. Rate limits and the audit log live in one process (Redis in production). Details: `docs/security.md`.

## A7. Memory

**Decision:** A LangGraph SQLite checkpointer saves every turn, keyed by `user_id:thread_id`. Agents see the user's context, the last 6 messages word for word, and a rolling summary of older ones. Diagram D11.

**Why:** Follow-up questions work, conversations survive restarts, and another user can't open a conversation even with its id. The summary keeps long conversations fast and cheap.

**Trade-off:** The summary is lossy, so it is only context, never a source. SQLite is single-node (Postgres in production). Memory is per conversation – no memory across conversations yet.

## A8. Observability and feedback

**Decision:**

- **LangSmith** records every question as one trace – every agent step, model call, search and tool call, and the Python code the research agent wrote.
- **Logs** are JSON lines on the API's console. They show who, when and how long – never the question or the answer.
- The same **request id** is in the error message, the logs and the trace, so any problem can be traced back.
- Users can rate each answer 👍 / 👎, and the rating is saved on its trace.

**Why:** The brief asks to trace what the agents do and to use structured logs. With ratings on the trace, a bad answer leads straight to what went wrong.

**Trade-off:** To see what a user asked, you need LangSmith, not the logs. Using the ratings to improve answers is the next step. Details: `docs/logging.md`.

## A9. Scope: what's built and what's next

**Built :** agents that keep working when one fails, human approval for sensitive actions, and 👍 / 👎 feedback on answers.

**Next (planned, not built):**

- **Docker Compose** – start everything with one command.
- **Reranker** – better ordering of search results.
- **Long-term memory** – remember a user across conversations.
- **Conversation history** – reopen past chats (they are already saved).
- **Long sections** – split very long sections into smaller pieces.

**Why:** The brief says to focus on what matters for a POC first, so I spent the time on the main graded areas and on testing each feature live.
