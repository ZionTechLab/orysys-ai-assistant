# Architecture

Internal staff knowledge assistant for Commercial Bank of Ceylon PLC.
Diagrams are Mermaid (rendered by GitHub). Components drawn with **dashed borders are planned** and not built yet; this file is updated as each part is built.

- [D1 – System architecture](#d1--system-architecture)
- [D3 – Request lifecycle](#d3--request-lifecycle)
- [D12 – Observability and correlation](#d12--observability-and-correlation)
- [D14 – Deployment (Docker Compose)](#d14--deployment-docker-compose)

Related: agent graph and state (D2, D4, D10, D11) in `agents.md`, retrieval (D5, D6) in `retrieval.md`, security (D7, D8) in `security.md`, failure handling (D9) in `resilience.md`.

---

## D1 – System architecture

```mermaid
flowchart TB
    user(["Bank staff<br/>viewer · analyst · admin"])

    subgraph L1["1 · Presentation – Streamlit"]
        direction TB
        chat["Chat window<br/>streamed answer + citations"]
        panel["Agent activity panel<br/>every step, live"]
    end

    subgraph L2["2 · API – FastAPI, async"]
        direction LR
        mw["Request id<br/>JSON logs"] --> auth["JWT auth<br/>role + clearance"] --> rl["Rate limiter<br/>token bucket"] --> sse["/chat<br/>SSE stream"]
    end

    subgraph L3["3 · Agents – LangGraph"]
        direction TB
        subgraph ENTRY["Understand"]
            direction LR
            gin["Input guard<br/>injection, abuse"] --> sup["Supervisor<br/>intent · route · filters"]
        end
        subgraph SPEC["Specialists – one per route"]
            direction TB
            ret["Retrieval agent<br/>hybrid search"]
            res["Research agent – RLM<br/>Python plan · parallel workers"]
            tool["Tool agent<br/>RBAC · human approval"]
        end
        subgraph CHECK["Answer and check"]
            direction LR
            resp["Response agent<br/>cited answer"] --> val["Validator<br/>citations · grounding"] --> gout["Output guard<br/>leakage · brand"] --> memn["Memory<br/>rolling summary"]
        end
        ENTRY --> SPEC --> CHECK
    end

    subgraph L4["4 · Data and services"]
        direction TB
        pc[("Pinecone<br/>hybrid index · 45 docs<br/>namespace per dept")]
        bm25[("Local BM25<br/>fallback")]
        sbx["Python sandbox<br/>RLM plans, analysis"]
        mcp["MCP server<br/>live enterprise data"]
        ck[("SQLite checkpointer<br/>memory, paused approvals")]
        llm["LLM gateway<br/>+ fallback LLM"]
    end

    subgraph OBS["5 · Observability"]
        direction TB
        ls["LangSmith<br/>one trace per turn<br/>+ answer feedback"]
        logs["JSON logs<br/>request id · no content"]
    end

    user --> L1
    L1 <-->|"HTTPS · SSE events"| L2
    L2 -->|"question + user context<br/>(role, clearance from the token)"| L3
    L3 -->|"filtered by the user's clearance"| L4
    L3 -.->|"every node, LLM call,<br/>search and tool"| OBS

    classDef agent fill:#e8f0fe,stroke:#1a56db,color:#0b1f44
    classDef guard fill:#fdecea,stroke:#c0392b,color:#4a0e08
    classDef store fill:#eef2f5,stroke:#5f6b76,color:#1d252c
    classDef svc fill:#f3ecfb,stroke:#7b3fb6,color:#2b0f45
    classDef io fill:#fff7e0,stroke:#b7791f,color:#3d2a05
    class sup,ret,res,tool,resp,memn agent
    class gin,val,gout,auth,rl guard
    class pc,bm25,ck store
    class sbx,mcp,llm,ls,logs svc
    class chat,panel,mw,sse io
    style L1 fill:#fafafa,stroke:#9aa5b1,color:#1d252c
    style L2 fill:#fafafa,stroke:#9aa5b1,color:#1d252c
    style L3 fill:#fafafa,stroke:#9aa5b1,color:#1d252c
    style L4 fill:#fafafa,stroke:#9aa5b1,color:#1d252c
    style OBS fill:#fafafa,stroke:#9aa5b1,color:#1d252c
    style ENTRY fill:#ffffff,stroke:#c8d0d8,color:#1d252c
    style SPEC fill:#ffffff,stroke:#c8d0d8,color:#1d252c
    style CHECK fill:#ffffff,stroke:#c8d0d8,color:#1d252c
    linkStyle default stroke:#6b7785,stroke-width:1.5px
```

*Colours:* blue = agents · red = guards and access control · grey = data stores · purple = services and telemetry · amber = UI and API entry points.

**How to read it**

- **Five tiers, top to bottom:** presentation, API, agents, data and services, observability. The agents tier reads in three steps: understand (guard + supervisor), one specialist per route, then answer and check.
- **Identity and limits first.** Every request passes the middleware (request id, logging), then authentication and the per-user rate limiter before reaching the agents.
- **One graph, many agents.** The supervisor routes to specialist agents; every answer is checked by the validator and the output guard (which can replace the streamed text), then the memory node updates the conversation summary.
- **Data access is filtered by the server.** The retrieval filter on `access_level` comes from the user's token, never from the LLM.
- **Every dependency has a fallback.** Pinecone → local BM25, LLM gateway → direct provider, MCP or tools down → answer from documents with a note (see D9).
- **Two kinds of telemetry.** JSON logs for operations, LangSmith for full agent traces (see D12).

**Built so far:** middleware (request id, JSON logs), login, `/health`, LLM factory, Pinecone hybrid retrieval with sync, LangGraph graph (supervisor → retrieval or research (RLM) → response) with SQLite session memory, `/chat` SSE streaming, Streamlit chat + activity panel, LangSmith traces. Tool agent with RBAC and approval. Input guard, validator and output guard. Rate limiter. Python analysis tool. Local BM25 fallback and fallback LLM (optional, configured via `FALLBACK_LLM_*`); fault injection for demos (`docs/resilience.md`). MCP server (`mcp_server/`, separate process) with three tools for analysts. Rolling conversation summary.

---

## D3 – Request lifecycle

One question from a logged-in analyst, end to end.

```mermaid
sequenceDiagram
    autonumber
    actor U as Analyst
    participant UI as Streamlit
    participant API as FastAPI
    participant RL as Rate limiter
    participant G as LangGraph
    participant P as Pinecone
    participant L as 9router LLM
    participant LS as LangSmith

    U->>UI: Log in
    UI->>API: POST /auth/login
    API-->>UI: JWT (role, clearance)

    U->>UI: Ask a question
    UI->>API: POST /chat (JWT, thread_id, message)
    API->>API: Verify JWT, bind request_id + user to logs
    API->>RL: Take 1 token
    alt Bucket empty
        RL-->>API: Denied
        API-->>UI: 429 + Retry-After
    else Token available
        API->>G: Run graph (user, clearance, thread_id)
        G-->>UI: SSE node_start: input_guard
        G->>L: Supervisor - route as JSON
        G-->>UI: SSE node_start: retrieval_agent
        G->>P: Hybrid query, filter access_level in clearance
        P-->>G: Chunks with doc_id + section
        G-->>UI: SSE retrieval: hits, namespaces
        G->>L: Response agent (stream)
        loop Tokens
            G-->>UI: SSE token
        end
        G->>L: Validator – grounding check
        G-->>UI: SSE validation: citations exist, grounded
        G-->>UI: SSE output_guard: no issues (or replaced text)
        G->>G: Memory node – fold older turns into the summary
        API-->>UI: SSE memory_update, done (run id → trace link)
        G--)LS: Trace: nodes, tools, retrieval, LLM calls
    end
```

**Points shown**

- Every step is async; the UI sees progress events while the graph runs, not only the final answer.
- The rate limit is checked before any LLM cost is spent.
- The trace is sent to LangSmith in the background and linked in the final event.

---

## D12 – Observability and correlation

How one `request_id` connects what the user sees, the logs and the trace. Details: `logging.md`.

```mermaid
flowchart LR
    req["Incoming request"] --> mw["Middleware<br/>creates request_id"]
    mw --> ctx["Bound to log context<br/>request_id, user_id, role, thread_id"]
    ctx --> logs[("JSON logs - stdout<br/>metadata only")]
    mw --> hdr["X-Request-ID header<br/>+ error body"]
    hdr --> userview["User / support<br/>quotes the id"]
    mw --> trace["LangSmith run metadata"]
    mw --> mcpa["MCP server audit line<br/>X-Request-ID + caller"]
    trace --> ls[("LangSmith<br/>full prompts, outputs,<br/>agent steps")]
    userview -. "search by id" .-> logs
    userview -. "search by id" .-> ls

    classDef planned stroke-dasharray: 5 5
```

- **Logs** answer "what happened and how long did it take" without storing message content.
- **LangSmith** answers "why did the agent do that" with the full trace.

---

## D14 – Deployment (Docker Compose)

Designed, not built (bonus item deferred – decision A9). Today the three processes are started separately (README).

```mermaid
flowchart TB
    subgraph host["Docker host"]
        subgraph net["compose network"]
            ui["ui<br/>Streamlit :8501"]
            api["api<br/>FastAPI + LangGraph :8000"]
            mcp["mcp<br/>MCP server :8765"]
        end
        vol[("volume: data<br/>SQLite memory, BM25 params")]
    end
    browser(["Browser"]) -->|":8501"| ui
    ui -->|"http://api:8000"| api
    api -->|"http://mcp:8765"| mcp
    api --- vol
    api -->|"HTTPS"| router["9router LLM gateway"]
    api -->|"HTTPS"| pc["Pinecone"]
    api -->|"HTTPS"| ls["LangSmith"]
    env[/".env - secrets, not in image"/] -.-> api

    classDef planned stroke-dasharray: 5 5
    class ui,api,mcp,vol planned
```
