# Assumptions and Decisions

Each entry: what we decided, why, and the trade-off. Referenced from the README and the demo.

---

## A1. Audience: internal staff assistant

**Decision:** The assistant is for Commercial Bank of Ceylon PLC employees, not customers.

**Why:** The brief describes *internal documents* (policies, runbooks, incident reports) and says *employees* need the assistant. The roles (Viewer, Analyst, Administrator) are staff roles.

**Brand guardrail:** The bot still represents the bank. Staff can screenshot or forward answers, so the output guard enforces a professional tone, no blaming individuals, no financial advice, no comments on competitors and no invented policies.

**Trade-off:** Customer-facing concerns (KYC, PII masking for customers, multilingual support) are out of scope.

## A2. Bot company and sample data

**Decision:** The assistant's persona, tone and brand guardrails are for **Commercial Bank of Ceylon PLC**. All sample documents are **fictional** and marked as such.

**Why:** The brief asks to consider brand value using a commercial bank as the bot's company. Brand rules are based on the bank's public website, not guessed. The sample documents describe outages, security incidents and executive pay; in a public repo they must not look like real bank documents.

**Trade-off:** No logo or official imagery is used – only the name, the tone and, in the UI, the colour palette and font of the public website (blue #095BCC / navy #0E2E5A / yellow #F7D700, Nunito; `.streamlit/config.toml`), so the demo feels like a bank tool. Every screen still states that the sample data is fictional.

## A3. Mock data

**Decision:** 45 hand-written documents with YAML front-matter (department, document type, access level, date, tags).

**Why:** The brief allows generated data. The incidents are designed with three recurring root causes (certificate expiry, month-end connection pool exhaustion, third-party gateway timeouts) so the RLM demo has a known, checkable answer. Includes restricted documents for RBAC and one poisoned document for the prompt-injection demo.

**Trade-off:** Small corpus; scale is shown through design (namespaces, batching), not volume.

## A4. LLM gateway: 9router

**Decision:** All LLM calls go through 9router (local OpenAI-compatible proxy) with multi-provider fallback, plus a small in-code fallback.

**Why:** Free-tier providers have daily limits; a router gives provider fallback quickly. The app only knows `LLM_BASE_URL`, so the gateway can be swapped.

**Trade-off:** LiteLLM is the more common production choice and would be the next step. 9router is aimed at developer tooling.

## A5. Model selection: two tiers, one model behind the gateway for now

**Decision:** The code asks for one of two tiers – `strong` (supervisor, tool agent, final answer, brand rewrite, analysis code) and `fast` (research workers, guards, grounding and brand checks, memory summary). Both currently map to the 9router combo `orysys-fast`, which serves `deepseek-v4.1-flash` (a reasoning model) with the gateway's provider fallbacks. The tier names are fixed in the code; the model behind each is one line in `.env` (`LLM_MODEL_STRONG`, `LLM_MODEL_FAST`).

**Why:** Free-tier quota decided the model: the combo gives a capable reasoning model for JSON routing, Python plan writing and grounded answers at no cost, and the gateway switches provider when one hits its limit. RLM makes many small calls, so the code is ready to send those to a cheaper model the moment one is available, without touching any agent.

**Trade-off:** A reasoning model is slow – a research plan takes 40–60 s and about 4,600 tokens, and simple routing still takes several seconds – which is why long generations have their own 120 s timeout (A14, A17). Using the same model for both tiers gives up the cost saving the tiers were designed for. With a paid account the `fast` tier would move to a small non-reasoning model (e.g. a Haiku- or Flash-class model) and `strong` would stay a reasoning model.

## A6. Authentication: hardcoded users with JWT (brief Option A)

**Decision:** Three demo users (viewer, analyst, admin) with bcrypt-hashed passwords; login returns a signed JWT (8 h) with role and clearance.

**Why:** The brief allows hardcoded users. It keeps the time for the agent work while still showing the real pattern: identity comes from a signed token, never from the chat or the LLM. On each request the user is reloaded from the server-side list, so an edited or stale role in a token is rejected.

**Trade-off:** No user management, refresh tokens or SSO. In production this would be Keycloak / the bank's identity provider (OIDC); only `get_current_user` would change.

## A7. Structured logging (structlog, JSON) separate from LangSmith

**Decision:** structlog JSON lines to stdout with `request_id`, `user_id`, `role`, `thread_id` on every line. Logs hold metadata only; conversation content goes to LangSmith. Details: `docs/logging.md`.

**Why:** The brief requires structured logging. JSON is searchable without parsing ("all RBAC denials for analyst1"). One `request_id` links the user's error, the log line and the trace. Not logging message content follows the bank's data classification (POL-002) and keeps logs safe to share.

**Trade-off:** To see what the user asked you need LangSmith, not the logs. No log shipping (e.g. ELK/Loki) in the POC; stdout is ready for it.

## A8. LLM access through one factory, JSON routing instead of native tool calling

**Decision:** `app/llm/factory.py` is the only place that creates models (`get_llm("strong" | "fast")`), using the OpenAI-compatible API, with client retries and an optional direct-provider fallback (`with_fallbacks`). The supervisor returns JSON validated by Pydantic rather than relying on the model's tool-calling.

**Why:** The smoke test (`scripts/smoke_llm.py`) showed the gateway handles plain calls, JSON output and streaming. JSON routing works the same across providers behind the router, is easy to validate and easy to show in traces.

**Trade-off:** First call latency through the gateway was ~10 s (later calls 2–5 s). Acceptable for a POC; streaming hides most of it.

**Update 14 Nov – validate and re-ask:** live, the gateway sometimes returned truncated JSON (`{"tool-index to preview…"}`) or an object missing a required key – 2 of 4 identical tool-agent calls in one test. One bad reply sent an admin's re-index request to "no tool". The supervisor and tool agent now go through `invoke_json` (`app/graph/nodes/common.py`): the reply is parsed and validated against a Pydantic schema, and if it is unusable the model is asked again with the reason, up to 3 attempts, before the node's fallback applies. The panel shows each retry (`llm_output_invalid`). After the fix: 6 of 6 live runs chose `admin_reindex` correctly. Cost: one extra call (a few seconds) only when a reply is bad.

## A9. Hybrid retrieval design

**Decision:** One Pinecone dotproduct index; dense `llama-text-embed-v2` + BM25 fitted on the corpus; `alpha = 0.6`; namespace per department; one chunk per document section; access filter from the user's token on every query. Details and diagrams D5/D6: `docs/retrieval.md`.

**Why:** Dense finds meaning ("outage" ≈ "failures"), BM25 finds exact bank terms and ids ("PayBridge", "INC-2026-021" – ids are in the indexed text since A28). Pinecone Inference keeps embeddings free of LLM quota. Section chunks give precise citations.

**Trade-off:** Sections are not split, so a very long section would become one oversized chunk – fine for the short sample documents; a size cap with overlap for long sections is designed as a future enhancement (`docs/retrieval.md`). Alpha is fixed, not tuned on an evaluation set. Hits below a relevance score of 0.25 are not used as evidence (`RETRIEVAL_MIN_SCORE`); the threshold was set by inspecting scores, not tuned either. No reranker (bonus item, deferred – A30).

## A10. Source documents in Markdown (production would add format loaders)

**Decision:** The sample corpus is Markdown with YAML front-matter. Ingestion is written so only the first step (file → sections + metadata) depends on the format.

**Why:** The brief allows generated mock data. Markdown headings give clean section chunks and precise citations; front-matter carries the metadata that RBAC and filtering need; files are readable in GitHub and the demo.

**Production path:** real bank documents are mostly `.docx`, `.pdf` and old `.doc`. Only the loader changes:

| Format | Section extraction | Metadata source |
|---|---|---|
| `.docx` | python-docx – Heading 1/2 styles → sections | Document properties or DMS metadata |
| `.pdf` (text) | pypdf / pdfplumber, or Docling for layout and tables | PDF properties or DMS metadata |
| Scanned PDF | OCR (Tesseract, Azure Document Intelligence) | DMS metadata |
| `.doc` | Convert to `.docx` (LibreOffice headless) first | DMS metadata |

Metadata – especially `access_level` – should come from the document management system (e.g. SharePoint permissions), not be inferred from content.

**Implemented (13 Nov):** pluggable loaders in `app/retrieval/loaders.py` – Markdown, DOCX (python-docx, Heading 2 → section, core properties → metadata) and text PDF (pypdf, bookmarks → sections, info dictionary → metadata). Four sample documents were converted (`scripts/make_sample_formats.py`): RB-002 and POL-004 to `.docx`, INC-2026-078 and ARCH-001 to `.pdf`. Tests confirm each produces the same metadata and the same citation ids as its Markdown original, and they are ingested and searchable like any other document.

**Trade-off:** Tables, images and scanned PDFs (OCR) are not handled. PDF metadata uses custom info-dictionary keys; a real system would read it from the DMS instead.

## A11. Keeping the index in sync with the documents

**Decision (POC):** Ingestion is a **sync**, not only an upsert: a content hash per document decides what to re-embed, and vectors of deleted documents or removed sections are deleted. BM25 is re-fitted on the full corpus at each sync and reloaded by the API. Admins trigger a sync with the `admin_reindex` tool; others cannot.

**Why:** Upsert alone never removes anything, so a withdrawn policy could still be cited – a real risk for a bank. Hashing avoids re-embedding unchanged documents.

**Production assumption: documents live in SharePoint.** The POC folder `data/docs` stands in for a SharePoint document library. In the live environment:
1. **Change notification:** a Microsoft Graph webhook subscription on the document library notifies the ingestion service when a file is added, updated or deleted.
2. **What changed:** the service calls the Graph *delta* query to get exactly the changed items since the last sync (webhooks only say "something changed").
3. **Queue + worker:** each changed item goes to a queue; a worker downloads the file, picks the loader by format (DOCX / PDF / ...), and runs the same sync per document: chunk → embed → upsert, or delete.
4. **Metadata from SharePoint, not the file:** department, document type and especially **access level** come from library columns, sensitivity labels and permissions. A permission change is also a change event, so access stays in step with SharePoint.
5. **Safety net:** a nightly full sync re-fits BM25 and catches missed notifications (Graph subscriptions expire and must be renewed).

The POC shows the same logic through `scripts/ingest.py` and the admin-only `admin_reindex` tool; only the trigger and the file source differ.

**Trade-off:** BM25 needs the whole corpus to compute term weights, so a single new document is only fully weighted after the next re-fit. Alternative: Pinecone's learned sparse model (`pinecone-sparse-english-v0`), which needs no fitting and is truly incremental; we kept BM25 because the brief asks for keyword/BM25 search.

## A12. User identity as run context, not graph state

**Decision:** Role and clearance are passed to the graph as LangGraph run context on every request, built from the verified JWT. They are not part of `AgentState` and never stored in checkpoints. The retrieval agent reads clearance only from the context.

**Why:** State is written by agents (LLM output) and persisted. If clearance lived there, a manipulated plan or an old checkpoint could change what a user can see. Context cannot be written by nodes. A unit test checks that a `clearance` value placed in the plan is ignored.

**Trade-off:** None significant; slightly more plumbing in the API.

## A13. Session memory: SQLite checkpointer, scoped by user

**Decision:** LangGraph `AsyncSqliteSaver` (`data/memory.db`), thread key `user_id:thread_id`; agents see a summary of older turns plus the recent messages (A27). Details and diagram D11: `docs/agents.md`.

**Why:** Survives turns and API restarts with no extra service. Scoping the key by the JWT user id means a guessed thread id cannot expose another user's conversation. A short window keeps prompts small and cheap; the supervisor rewrites follow-up questions into standalone search queries.

**Trade-off:** SQLite is single-node. Production would use the Postgres checkpointer (same interface).

## A14. Fail fast on slow dependencies

**Decision:** LLM calls time out after 20 s per attempt (2 retries); a search times out after 10 s and is retried once after recreating the Pinecone client. Every graph node has a safe fallback.

**Why:** Observed on 13 Nov: the gateway sometimes routes to a backend that hangs, and the office network had periods where new connections stalled for 30–60 s. With a 60 s timeout one turn could hang for minutes. With short timeouts the retry often lands on a healthy backend, and if not, the user gets an honest message instead of an endless spinner. Stale pooled connections after idle time are also handled by recreating the client.

**Trade-off:** A slow but working backend (> 20 s) is treated as failed. Streaming depends on which backend the gateway picks; when it picks a non-streaming one, the answer appears in one piece.

## A15. RLM catalog: explore metadata first, read sections on demand

**Decision:** The research agent gets a small catalog API (`describe`, `list_docs`, `get_doc`, `read_section`, `search`) bound to the user's clearance. Listing and description return metadata only; content is read one section at a time (max 2,000 chars). Metadata and section text come from the indexed documents loaded in memory; `search` uses Pinecone. Details: `docs/agents.md`.

**Why:** This is the core of the Recursive Language Model idea in the brief: the model plans over *what exists* and pulls only targeted sections, so a question over many documents does not need all of them in context. Binding the catalog to clearance means generated code (task 2.2) cannot reach documents the user may not see, whatever it tries. Same error for hidden and missing documents prevents probing.

**Trade-off:** The in-memory catalog fits a POC corpus; at thousands of documents the metadata would come from a database or the index (Pinecone `list` + `fetch`, as the sync already does) with the same API. One shared retriever per process: opening new connections was the slow, failure-prone part on 13 Nov.

## A16. Sandbox for generated research plans

**Decision:** The research agent's Python plans run in an in-process sandbox: AST allow-list → safe builtins + catalog functions only → line and time budget enforced by a trace function → JSON result cap. Failures are returned to the agent as fixable messages. Details: `docs/agents.md`.

**Why:** The brief asks the agent to "generate Python-based search plans". Generated code is untrusted (a poisoned document could try to steer it), so it must not reach the file system, network, environment, imports or Python internals, and must not run forever. All data access goes through the clearance-bound catalog, so even a malicious plan cannot see more than the user can.

**Trade-off – stated honestly:** an in-process Python sandbox is defence in depth, not a hard security boundary; Python has a long history of sandbox escapes, and a single expensive operation (e.g. a very large string multiplication) can still use memory before the budget check runs. Production would run plans in a separate process or container with OS-level limits (CPU, memory, no network, read-only file system – e.g. gVisor / Firecracker / a locked-down subprocess), keeping the same plan contract. Alternative considered: a small JSON plan DSL instead of Python – safer, but it does not meet the brief's "Python-based search plans".

**Update 13 Nov (memory):** the audit found that a plan like `"a" * 100000 * 100000` passed the static check and would allocate ~10 GB inside the API process. Added: size checks before `*`, `+`, `+=`, `*=`, `str.join` and `str.replace` run (max 1,000,000 items/characters per operation); padding methods, wide or computed f-string format specs and `%` formatting blocked; `sum()` for numbers only; and a memory watchdog in the trace function (stops the plan if the process grows by more than 256 MB) for gradual growth such as `list.extend` in a loop. 13 blow-up plans are now tests. Still defence in depth – the production answer remains a separate process with OS-level limits.

## A17. Research agent: plan → parallel recursive workers → deferred aggregator

**Decision:** Multi-document questions go to a research flow: an LLM-written Python plan selects documents (sandboxed, repaired up to 3 times, search fallback), LangGraph `Send` fans out one worker per batch, workers split oversized batches recursively (depth ≤ 2) and analyse each document with one small LLM call over targeted sections, and a deferred aggregator waits for full coverage, lets the LLM group causes and **counts in code**. Details and diagram D4: `docs/agents.md`.

**Why:** This is the brief's RLM example end to end (search → filter → batches → analyse separately → aggregate → summary) without loading documents into one context. Lessons from live testing shaped it: the fast model returned malformed JSON when asked about three documents at once (→ one call per document); the plan over-selected a non-payment incident by keyword (→ worker relevance flag); LangGraph's deferred node alone ran early when workers recursed (→ explicit completeness check, verified on a toy graph first).

**Trade-off:** ~75 s for the example question, mostly the reasoning model writing the plan; more LLM calls than one big prompt (1 plan + 9 documents + 1 grouping + 1 answer), but each is small, parallel and individually recoverable. The worker reads fixed preferred sections (Summary, Root Cause, Impact…) rather than asking the model which sections to read – cheaper and predictable, less flexible for unusual document types.

**A14 update:** short calls keep the 20 s timeout; long generations (plans, workers, grouping) use `LLM_LONG_TIMEOUT_SECONDS` = 120 s with one retry, because the gateway's reasoning model needs 40–60 s for a plan. The sandbox time limit is 20 s because `search()` network time counts against it.

## A18. Tool registry with RBAC enforced twice, plus an audit log

**Decision:** Tools live in one registry; each declares allowed roles, a Pydantic argument schema, a timeout and whether it needs approval. RBAC is enforced at binding (the tool agent's LLM sees only permitted tools) and again at execution (`REGISTRY.execute` re-checks the role from the JWT). Every attempt is audited. Details and diagram D8: `docs/security.md`.

**Why:** The brief requires that "tool execution must respect role permissions" and "the agent should not be able to bypass authorization". Binding alone relies on the model behaving; the execution check does not – a hallucinated or injected tool name is denied (unit-tested). The tool agent chooses with JSON, consistent with the supervisor (A8).

**Trade-off:** The audit log is in memory (last 500 entries) plus JSON log lines; production would persist it. Roles are coarse (three); production would map tools to fine-grained permissions from the identity provider.

## A19. Human-in-the-loop approval with LangGraph interrupts

**Decision:** Tools marked `requires_approval` (today `admin_reindex`) pause the run with LangGraph `interrupt()`. The API returns `approval_required`; `POST /chat/resume` continues with the decision. Tool choice, approval and execution are three nodes. New messages on a paused thread get 409.

**Why:** A bonus item in the brief, and the right control for actions that change shared state. Separate nodes mean resuming re-runs only the gate, so the model cannot change the tool after approval. The paused state is in the checkpointer, so it survives restarts. The approver must be allowed to run the tool; the thread key is scoped to the user, so nobody else can approve or even see the paused run.

**Trade-off:** The requester approves their own action (one admin). A bank would usually require a second person ("four-eyes") – the resume endpoint is where that check would go. Our node wrapper had to let `GraphInterrupt` pass through instead of treating it as a failure.

## A20. Prompt-injection protection: rules first, classifier for suspicious input, untrusted content

**Decision:** Every message is normalised and checked by explainable rules for the three categories in the brief, in two tiers. *Block* rules match only wording aimed at the assistant itself ("ignore your rules", "reveal your system prompt", "make me admin"); a match ends the turn with a fixed refusal (no LLM). *Refer* rules match the same topics in wording that normal staff questions share ("can I skip approval…", "debug mode"); they do not block but send the message to a fast-model classifier, as do softer suspicious markers. Retrieved text is sanitised before any model sees it (hidden HTML comments and tag breakouts removed, injection text flagged) and wrapped as data. Details and D7: `docs/security.md`.

**Why:** Rules are instant, free and easy to explain in an audit; the classifier catches rephrasings ("pretend you are the administrator…") without adding 5–40 s to every normal question on this gateway. A fixed refusal gives an attacker nothing to work with. Indirect injection through documents is the more realistic threat for a RAG system – the poisoned MTG-005 shows it.

**Trade-off:** Detection is never complete; false positives are possible for unusual wording. Referred questions pay one classifier call (3–9 s on this gateway). If the classifier is down, referred and suspicious messages are allowed (availability) – acceptable because access control does not depend on detection.

**Changed 13 Nov:** the first version blocked on every rule match. An audit found it refused 12 of 12 normal questions that share attack vocabulary ("Give me the root cause of the PayBridge outage" matched "give me … root"). A refusal of a normal work question is a worse failure for an internal assistant than a missed pattern, because RBAC and clearance filters still hold when a pattern is missed. Hence the two tiers, and a regression list of normal questions in the tests.

## A21. Answer validator with one rewrite

**Decision:** After every answer: deterministic checks (non-empty, citations exist in the evidence, document answers cite sources) and an LLM grounding check. On failure the response agent rewrites once with the problems as feedback; the rewrite replaces the first answer (same message id). A second failure removes invalid citations and adds a note.

**Why:** The brief asks for guardrails against hallucinated citations and invalid responses. Citation checks are exact and cheap; grounding catches invented facts. One retry bounds cost and latency.

**Trade-off:** The first answer is already streamed when it is validated, so a failed answer is visibly replaced in the UI ("revising…"). Validating before streaming would remove that but lose live streaming. The grounding check adds a model call per document answer; it is skipped (and recorded) when unavailable rather than blocking answers.

## A22. Output guard: deterministic leak checks + brand check on risk markers

**Decision:** The last node before the user. Always: redact secret formats and the configured secret values, remove sentences copied from our prompts, remove sentences naming documents above the user's clearance. Only when risk markers appear: a fast-model brand compliance check; a violation is rewritten once by the strong model, else replaced with a safe neutral message. Details: `docs/security.md`.

**Why:** The brief's guardrails include brand value (commercial bank as the bot's company), unauthorised access and invalid responses. Leak checks are exact and free, so they run on every answer; the brand check needs judgement, so a model does it – but only when an answer contains a reason to check, keeping normal answers fast. Clearance removal backs up the retrieval filters (defence in depth) and uses a neutral note so it cannot be used to confirm a restricted document exists.

**Trade-off:** Marker lists will miss some phrasings (e.g. an opinion with no trigger word); the response prompt carries the same rules, so the guard is a second line, not the only one. Removing whole sentences can make an answer terser. The fixed competitor list names other Sri Lankan banks for detection only.

## A23. Token-bucket rate limiting per user, plus a login limiter

**Decision:** In-memory token buckets: per user for `/chat` and `/chat/resume` (capacity 10, refill 0.2/s, per-role overrides from config), per username for `/auth/login` (5, then 1 per minute). Enforced as a dependency right after authentication; an empty bucket returns 429 with `Retry-After`. Details and D13: `docs/security.md`.

**Why:** The brief asks for token-bucket limiting with per-user limits, configurable thresholds and graceful errors. A token bucket allows short bursts (a user asking follow-ups quickly) while capping the sustained rate – which matters here because each turn can trigger several LLM calls and the free-tier quota is small. Checking before the graph means a refused request costs nothing. The login limiter is not in the brief but closes password guessing at no cost.

**Trade-off:** Buckets live in one process: with several API instances each would have its own buckets; production would keep the same interface on Redis (atomic script). Every request costs one token although a research question costs far more LLM calls than a greeting – weighting by route would be fairer but the route is only known after the supervisor runs.

## A24. python_analysis: generated code in the existing sandbox

**Decision:** The analysis tool takes a question; inside, the LLM writes Python over the clearance-bound catalog with a few analysis helpers, the RLM sandbox runs it, and errors are repaired up to 3 times. Analyst and admin only. Quantitative questions route to the tool; research handles content questions.

**Why:** The brief asks for a "Python analysis tool – structured analysis on retrieved data". Reusing the catalog and sandbox means one set of security limits for all generated code, and counting in code gives exact, explainable numbers (the code is returned with the result). Passing only a question to the tool keeps the tool agent's JSON small and leaves code generation to the long-timeout model.

**Trade-off:** Analysis works on metadata and section text, not on structured fields inside documents (e.g. outage minutes are parsed from text with `find_numbers`). A production version would extract such fields at ingestion. The validator cannot check computed numbers against documents – the code in the result is the audit trail instead.

**Also fixed (validator):** the last-resort repair now falls back to the first answer if the rewrite comes back empty – seen live once, the repaired answer was empty.

## A25. Graceful degradation everywhere, with fault injection to demonstrate it

**Decision:** Every dependency failure is contained at its boundary and turned into a weaker but honest answer (map and table: `docs/resilience.md`, diagram D9). New in this step: Pinecone falls back to in-memory BM25 over the same chunks and access rules; when the LLM is down the response lists the documents already found (extractive, no model needed) and the validator accepts it instead of retrying. Administrators can simulate Pinecone / LLM / MCP outages at runtime (`/admin/faults`, UI sidebar).

**Why:** The brief asks to demonstrate handling of LLM, vector DB, MCP, tool and request failures and that the application "degrades gracefully". Contained failures plus useful fallbacks avoid the cascading ("butterfly") effect named in the bonus points. Fault injection makes each path demonstrable on demand and repeatable; the checks sit at the real dependency boundary so everything above runs unchanged.

**Trade-off:** Fault injection only simulates hard outages, not latency. The fault switch is an admin-only operational endpoint – in production it would be removed or limited to non-production environments. Local BM25 is weaker than hybrid search (no semantic matching), which the answer says.

## A26. MCP server as a separate service, wrapped as allow-listed registry tools

**Decision:** A small MCP server (`mcp_server/`, official MCP Python SDK, Streamable HTTP) exposes fictional enterprise data – employee directory, service catalog (owner, on-call, live status), incident tickets. The assistant calls it through three registry tools (`mcp_*`) for analysts and admins. The server needs a shared service key; the API passes the user's clearance and identity (trusted subsystem); incident tickets carry an access level and are filtered on both sides. When the server is down, service and incident questions fall back to a document search.

**Why:** The brief asks for an MCP tool with dummy enterprise data that the agent invokes when needed, and for MCP failures to be handled. A separate process is how a real enterprise system would sit; wrapping the tools in the registry means MCP calls get the same RBAC, validation, timeout and audit as every other tool instead of a second, weaker path. Live data (who is on call now, open tickets) is something documents cannot answer, so the supervisor has a real reason to choose it.

**Trade-off:** Static allow-list instead of dynamic tool discovery – safer, but a new server tool needs a code change. One MCP session per call (simple; a pool would save the handshake). A shared key is not per-user authentication; production would use mTLS / OAuth and propagate the user's identity.

## A27. Rolling summary of older turns

**Decision:** Agents see the user's context, a rolling summary of older turns and the recent messages verbatim. A memory node at the end of each turn folds messages that left the 6-message window into the summary once at least 4 have aged. All messages stay in the checkpoint.

**Why:** The brief asks memory to keep user context, previous questions and relevant historical interactions across turns. A fixed window forgets the start of a long conversation; sending everything grows cost and latency every turn. The summary keeps document ids and topics, which is what follow-ups refer to ("the first incident I asked about"). Running it after the answer and in batches keeps the user from waiting.

**Trade-off:** The summary is lossy and costs one fast-model call every two turns. It is per conversation – long-term memory across threads (a bonus item) is not built. It is labelled as context, never as a source, so a summarisation mistake cannot become a cited fact.

## A28. Document ids are indexed and named documents are searched directly

**Decision:** Each chunk's indexed text starts with its document id, and when a question names ids the retrieval agent also searches inside those documents (metadata filter `doc_id`) and ranks them first.

**Why:** Found while testing memory: *"What happened in incident INC-2026-021?"* returned nothing relevant, because the id was only in metadata. Asking by id is common for staff; a metadata filter is exact and explainable, where ranking alone is not.

**Trade-off:** One extra (parallel) search when ids are named. Ids are recognised by pattern (`INC-YYYY-NNN`, `RB-`, `ARCH-`, `POL-`, `SPEC-`, `MTG-`); a new document family needs the pattern extended.

## A29. Fail open or fail closed – decided per check

**Decision:** The input guard and the validator fail **open** (the answer proceeds and the failure is recorded); the output guard fails **closed** (the answer is withheld). Every other node fails to a useful fallback (D10 table, `docs/agents.md`).

**Why:** A check should fail closed when its failure would expose something. The input guard and validator sit on top of controls that work without them – RBAC, clearance filters, untrusted-content wrapping – so blocking every answer when they fail would cost availability for little security. The output guard is the last barrier against leaking secrets, prompts or restricted content. Found in the 13 Nov review: it originally passed answers through unchecked on an unexpected error.

**Trade-off:** A bug in the output guard now hides answers until it is fixed – visible immediately in the panel and logs, which is the preferred way to fail for a bank.

## A30. Scope: bonus items deferred

**Decision:** Built from the bonus list: multi-agent collaboration with failure isolation, human-in-the-loop approval, answer-quality feedback (A31). Deferred: Docker Compose, reranking layer, long-term (cross-conversation) memory. Also deferred from the core design: a size cap for long sections when chunking (`docs/retrieval.md`) and a conversation history list in the UI.

**Why:** The brief says to cover what matters for a POC first. Time went to the weighted areas (agent architecture, RAG, LangGraph, RLM, security) and to verifying each feature live. Each deferred item is designed, so the demo can explain how it would be added:
- **Docker Compose:** three services (api, ui, mcp) and a volume for SQLite / BM25 – diagram D14.
- **Reranker:** Pinecone-hosted `bge-reranker-v2-m3` over the top ~20 hybrid hits → 8, keeping the hybrid order if it fails; the 0.25 threshold would be re-tuned.
- **Long-term memory:** a per-user LangGraph `Store` of preferences and frequent topics, read by the supervisor.
- **Conversation history after login:** conversations are already saved per user in the checkpointer and survive logout and restarts, but the UI does not reopen them – log out, or a browser refresh, starts an empty chat. Next: `GET /threads` lists the logged-in user's conversations (first question as title, last update, message count; only keys starting with that user's id), the sidebar shows them, a click loads one through the existing `GET /threads/{id}` and continues it with its memory. Reopened conversations would show questions and answers; the activity panel and trace links are streamed live and not stored.

**Trade-off:** Fewer bonus points, in exchange for core features that were tested end to end.

## A31. Answer feedback stored on the LangSmith run

**Decision:** Every finished answer has 👍 / 👎 in the UI; after 👎 the user can add a short comment ("what was wrong?"). `POST /feedback` stores it in LangSmith as feedback `user_score` (1 / 0) on the run of that turn. Only the user who received the answer can rate it (run ownership checked – in memory, or from the run's metadata after a restart); one rating per user and answer, so changing the thumb updates it; comments are capped at 500 characters and never logged (only their length).

**Why:** The brief's objective names observability and production practice, and a "feedback loop for answer quality" is a bonus item. Storing the rating on the run puts it next to the full trace of how the answer was produced, so a 👎 can be investigated directly: filter runs by `user_score = 0` in LangSmith, open the trace, see whether retrieval, the plan or the answer went wrong. Those runs are also the natural seed of an evaluation dataset.

**Trade-off:** Collecting feedback is the first half of the loop; acting on it (an evaluation dataset built from rated runs, regression checks on prompt changes) is the next step and not automated here. If LangSmith is unavailable the rating is only in the JSON log, and the UI says so.

## A32. Deployment and data-handling assumptions

**Assumptions:**

1. **Sending data to cloud services is acceptable because all data is fictional.** Pinecone stores every chunk's text as metadata, and LangSmith traces contain the questions, the prompts, the retrieved document text and the answers. With real bank documents – confidential and restricted ones especially – this would need the bank's approval for data residency and confidentiality. The options, in order of preference: a self-hosted or region-pinned deployment (LangSmith self-hosted, a vector store inside the bank's network), keeping only ids in the vector index and fetching text from the document system, and masking or sampling what is sent to traces.
2. **The POC runs on one machine, bound to 127.0.0.1, without HTTPS.** The API, UI and MCP server listen on localhost only. Production would put them behind TLS, the bank's network controls and an API gateway, with secrets in a vault instead of `.env`.
3. **English only.** Questions, documents and answers are in English; the prompts, BM25 vocabulary and injection rules are English. Sinhala and Tamil would need multilingual embeddings, language-aware keyword search and translated guard rules.

**Why:** The brief asks for a POC with mock data and lets candidates choose a model and services. Free cloud tiers (Pinecone, LangSmith, an LLM gateway) made the full architecture buildable in three days; the data is written to be fictional (A2, A3) precisely so that using them is safe.

**Trade-off:** These are the first things that change before real documents are used. They are stated here – and in the demo – rather than hidden, because for a bank where the data goes matters as much as what the assistant can do.

<!-- Add further decisions as we build: retrieval, memory, RBAC, security, rate limiting -->

## A33. Typed spans for tools, MCP calls and the RLM sandbox

**Decision:** Besides the spans LangGraph creates (nodes, routing functions, LLM calls) and the `hybrid_search` retriever span, the code adds three child spans (`app/tracing.py`): `tool:<name>` (run type `tool`) around every tool execution, `mcp:<name>` (run type `tool`) around each MCP server call, and `rlm_sandbox` around each sandboxed plan run. A resumed approval run records the run that paused (`resumes_run_id`, tag `approval_resume`).

**Why:** The brief requires tracing of tool calls and lets the evaluator inspect execution traces. Before this, a tool call was visible only as the `tool_executor` node's state – the arguments, the RBAC decision and the result had to be dug out of a state diff, and the generated RLM code only appeared in the activity panel. Run type `tool` makes them first-class in LangSmith (filterable, with their own latency and error status). Found while checking the brief's observability list against real traces on 14 Nov.

**Trade-off:** Span data is truncated to 4,000 characters – enough to inspect, not a full copy of large results. The approval turn is still two traces (the run really stops while waiting for a human); they are linked by metadata, not merged. Traces stay in a private project: the evaluators see them in the demo video, not through public share links.
