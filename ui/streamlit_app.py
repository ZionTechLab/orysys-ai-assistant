"""Streamlit UI: chat window + live Agent Activity Panel.

Run: .venv/Scripts/streamlit run ui/streamlit_app.py
Talks only to the FastAPI backend (API_URL); it holds no business logic.
"""

import html
import json
import os
import time
import uuid
from datetime import datetime

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

st.set_page_config(page_title="ComBank Knowledge Assistant", layout="wide")

# Brand styling on top of .streamlit/config.toml: colours of the public combank.lk site, no logo or imagery.
st.markdown("""
<style>
.cb-hero {background: linear-gradient(to right, #0949A2, #095BCC); border-radius: 14px;
          padding: 1.1rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 14px rgba(14, 46, 90, .18);}
.cb-hero .cb-title {color: #F7D700; font-weight: 800; font-size: 1.7rem; line-height: 1.2;}
.cb-hero .cb-sub {color: #FFFFFF; opacity: .92; font-size: .95rem; margin-top: .2rem;}
.cb-hero .cb-note {color: #D2D8E2; font-size: .75rem; margin-top: .45rem;}
[data-testid="stMain"] h3 {color: #0E2E5A !important; font-weight: 800;}
[data-testid="stSidebar"] h2 {color: #F7D700;}
.cb-user {background: #0949A2; border-left: 4px solid #F7D700; border-radius: 10px; padding: .6rem .8rem;
          margin: .4rem 0 .6rem 0; color: #FFFFFF;}
.cb-user .cb-name {color: #F7D700; font-weight: 700;}
/* the site's red call-to-action gradient, for the sidebar's primary button (Log in) */
[data-testid="stSidebar"] button[kind="primaryFormSubmit"], [data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(#F13444, #DF3020); border: none; color: #FFFFFF; font-weight: 700;}
/* chat avatars in the bank's blues instead of Streamlit's default red / orange */
[data-testid="stChatMessageAvatarUser"] {background-color: #095BCC !important;}
[data-testid="stChatMessageAvatarAssistant"] {background-color: #0E2E5A !important;}
[data-testid="stChatMessageAvatarAssistant"] svg {color: #F7D700;}
</style>
""", unsafe_allow_html=True)

# Header banner – before the sidebar, so the login screen shows it too (the login form stops the script).
st.markdown(
    '<div class="cb-hero"><div class="cb-title">ComBank Knowledge Assistant</div>'
    '<div class="cb-sub">Answers from the bank\'s policies, runbooks, incident reports and specifications – '
    'with citations, and every agent step shown live.</div>'
    '<div class="cb-note">Internal staff assistant · all sample documents are fictional – not real Commercial Bank '
    'of Ceylon information</div></div>', unsafe_allow_html=True)

# ---------- session state ----------
ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("user", None)
ss.setdefault("thread_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])  # [{"role", "content", "citations", "trace"}]
# One entry per question: number, clock time, label, route, duration and its events with the time since
# the question was sent – so the activity panel can be grouped by message.
ss.setdefault("turns", [])
ss.setdefault("job", None)  # the request being answered: {"path", "payload", "turn", "started"}
ss.setdefault("pending", None)  # tool call waiting for human approval

NODE_LABELS = {
    "input_guard": "Input guard – checking for prompt injection",
    "validator": "Validator – checking citations and grounding",
    "output_guard": "Output guard – secrets, leakage and brand rules",
    "supervisor": "Supervisor – understanding the question",
    "retrieval_agent": "Retrieval agent – searching documents",
    "research_planner": "Research planner – writing a Python search plan",
    "research_worker": "Research worker – analysing a batch",
    "research_aggregator": "Research aggregator – combining findings",
    "tool_agent": "Tool agent – choosing a tool",
    "approval_gate": "Approval gate – checking if a human must approve",
    "tool_executor": "Tool executor – running the tool (role re-checked)",
    "response_agent": "Response agent – writing the answer",
    "memory": "Memory – updating the conversation summary",
}
QUIET_NODES = {"research_worker", "research_aggregator"}  # their own events are more informative


# ---------- API helpers ----------
def api_login(username: str, password: str) -> tuple[bool, str]:
    try:
        r = httpx.post(f"{API_URL}/auth/login", json={"username": username, "password": password}, timeout=TIMEOUT)
    except httpx.HTTPError:
        return False, "The assistant service is not reachable."
    if r.status_code != 200:
        return False, r.json().get("error", {}).get("message", "Login failed.")
    body = r.json()
    ss.token, ss.user = body["access_token"], body["user"]
    return True, ""


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {ss.token}"}


def stream_events(path: str, payload: dict):
    """Yield (event, data) pairs from a Server-Sent Events endpoint (/chat or /chat/resume)."""
    with httpx.stream("POST", f"{API_URL}{path}", json=payload, headers=auth_headers(), timeout=TIMEOUT) as r:
        if r.status_code != 200:
            r.read()
            try:
                msg = r.json().get("error", {}).get("message", "Request failed.")
            except json.JSONDecodeError:
                msg = "Request failed."
            if r.status_code == 401:
                ss.token = None
            yield "error", {"message": msg, "status": r.status_code}
            return
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                yield event, json.loads(line[5:].strip())


def trace_link(run_id: str) -> str | None:
    """LangSmith needs a moment to ingest the trace; try a few times."""
    for _ in range(3):
        try:
            r = httpx.get(f"{API_URL}/traces/{run_id}", headers=auth_headers(), timeout=TIMEOUT)
            if r.status_code == 200 and r.json().get("url"):
                return r.json()["url"]
        except httpx.HTTPError:
            pass
        time.sleep(1.5)
    return None


# ---------- activity panel rendering ----------
def describe(event: str, d: dict) -> str | None:
    if event == "node_start":
        return None if d["node"] in QUIET_NODES else f"▶️ **{NODE_LABELS.get(d['node'], d['node'])}**"
    if event == "node_end":
        return None if d["node"] in QUIET_NODES else f"✅ {d['node']} done ({d.get('latency_ms', '?')} ms)"
    if event == "rlm_plan":
        status = (f"✅ valid – {d['documents']} documents in {d['batches']} batches" if d["ok"]
                  else f"❌ {d.get('error')} – asking the model to repair it")
        calls = ", ".join(f"{k}×{v}" for k, v in (d.get("catalog_calls") or {}).items() if k != "chars_read")
        code = f"\n```python\n{d['code']}\n```" if d.get("code") else ""
        return f"🧪 Plan attempt {d['attempt']}: {status}  \nSandbox: {d.get('lines_executed', 0)} lines, {calls or '-'}{code}"
    if event == "input_guard":
        if d.get("blocked"):
            return (f"🛡️ **Blocked** – {d.get('category')} (rule: {d.get('rule')}, checked by {d.get('checked_by')}) "
                    f"– no agent will run")
        why = f", second look requested by rule {d['referred_by']}" if d.get("referred_by") else (
            ", suspicious wording" if d.get("suspicious") else "")
        return f"🛡️ Input allowed (checked by {d.get('checked_by')}{why})"
    if event == "output_guard":
        if d.get("actions"):
            return "🧯 Output guard changed the answer:  \n" + "  \n".join(f"– {a}" for a in d["actions"])
        markers = f" (markers: {', '.join(d['markers'])})" if d.get("markers") else ""
        return f"🧯 Output guard: no issues – brand check {d.get('brand')}{markers}"
    if event == "retrieval_fallback":
        return "🛟 Vector search unavailable – **fallback to local keyword search** (answer marked as degraded)"
    if event == "retrieval_named_docs":
        return (f"📌 Question names {', '.join(d['doc_ids'])} – {d['sections']} sections from "
                f"{'it' if len(d['doc_ids']) == 1 else 'them'} put first (doc_id metadata filter)")
    if event == "content_flagged":
        return f"🧹 Retrieved text cleaned: `{d['chunk_id']}` – {', '.join(d['flags'])}"
    if event == "validation":
        icon = {"accepted": "✅", "retry": "🔄", "repaired": "🩹"}.get(d.get("action"), "•")
        text = {"accepted": "Answer validated", "retry": "Answer rejected – rewriting once",
                "repaired": "Still failing – invalid citations removed"}.get(d.get("action"), d.get("action"))
        problems = "  \n".join(f"– {p}" for p in d.get("problems", []))
        return f"{icon} {text} (grounding: {d.get('grounding')})" + (f"  \n{problems}" if problems else "")
    if event == "tool_selected":
        offered = ", ".join(d.get("offered", [])) or "none"
        return f"🧰 Tools offered to this role: {offered}  \nChosen: **{d.get('tool') or 'none'}** {d.get('args') or ''}"
    if event == "tool_call":
        return f"⚙️ Running **{d['tool']}** {d.get('args') or ''}"
    if event == "rbac_denied":
        return f"⛔ RBAC: role **{d['role']}** may not use **{d['tool']}** – denied at execution"
    if event == "tool_fallback":
        return (f"🛟 **{d['tool']}** unavailable ({d.get('reason') or 'error'}) – answering from "
                f"{d.get('source')} instead ({d.get('evidence', 0)} relevant sections)")
    if event == "tool_result":
        return f"📤 {d['tool']}: **{d['status']}** ({d.get('latency_ms', 0)} ms) {d.get('message') or ''}"
    if event == "approval_required":
        return f"✋ Waiting for approval to run **{d.get('tool')}** {d.get('args') or ''}"
    if event == "approval_decision":
        return f"{'👍 Approved' if d.get('approved') else '👎 Rejected'} by {d.get('by')}"
    if event == "rlm_fallback":
        return f"↩️ No valid plan – {d.get('documents', 0)} documents selected by search instead"
    if event == "rlm_batches":
        batches = " | ".join(", ".join(b) for b in d.get("batches", []))
        return f"📦 {d['documents']} documents → {len(d.get('batches', []))} parallel workers: {batches}"
    if event == "rlm_recurse":
        return f"🔁 Batch {d['batch_no']} too large – split into {d['split']} for sub-workers (depth {d['depth'] + 1})"
    if event == "rlm_worker":
        extra = ""
        if d.get("not_relevant"):
            extra += f" · not relevant: {', '.join(d['not_relevant'])}"
        if d.get("failed"):
            extra += f" · ⚠️ failed: {', '.join(d['failed'])}"
        return (f"👷 Worker {d['batch_no']}: {', '.join(d['documents'])} – read {d['sections_read']} sections "
                f"({d['chars_read']} chars), {d['findings']} findings{extra}")
    if event == "rlm_waiting":
        return f"⏳ Aggregator waiting: {d['covered']}/{d['expected']} documents done"
    if event == "rlm_aggregate":
        themes = "  \n".join(f"• **{t['theme']}** × {t['count']} ({', '.join(t['doc_ids'])})" for t in d["themes"])
        skipped = f"  \nNot relevant: {', '.join(d['not_relevant'])}" if d.get("not_relevant") else ""
        failed = f"  \n⚠️ Not analysed: {', '.join(d['failed'])}" if d.get("failed") else ""
        return f"🧩 Themes from {d['analysed']} documents (counted in code):  \n{themes}{skipped}{failed}"
    if event == "node_error":
        return f"⚠️ {d['node']} failed ({d.get('error_type')}) – continuing with fallback"
    if event == "route_selected":
        filters = [f"departments={d['departments']}" if d.get("departments") else "",
                   f"types={d['document_types']}" if d.get("document_types") else "",
                   f"from={d['created_from']}" if d.get("created_from") else "",
                   f"to={d['created_to']}" if d.get("created_to") else ""]
        f = ", ".join(x for x in filters if x) or "none"
        return f"🧭 Route: **{d['route']}** – {d.get('intent', '')}  \nFilters: {f}"
    if event == "retrieval":
        top = ", ".join(f"`{t['chunk_id']}` ({t['score']:.2f})" for t in d.get("top", [])[:4]) or "none"
        return (f"🔎 Searched: _{d.get('query', '')}_  \n{d['hits']} hits, **{d['relevant']} relevant** "
                f"(min score {d.get('min_score')})  \nTop: {top}")
    if event == "response_generation":
        return f"✍️ Generating answer from {d.get('evidence', 0)} evidence chunks"
    if event == "response_done":
        return f"📎 Citations: {', '.join(d.get('citations', [])) or 'none'}"
    if event == "memory_summary":
        if d.get("action") == "summarized":
            return (f"🧠 Older turns summarised: {d['added']} messages folded into the summary "
                    f"({d['chars']} chars); agents now see summary + recent messages")
        return None  # nothing aged out yet – the memory_update line below says enough
    if event == "memory_update":
        summarized = (f", {d['summarized']} older ones in the summary ({d.get('summary_chars', 0)} chars)"
                      if d.get("summarized") else "")
        return f"🧠 Memory saved – conversation has {d.get('messages')} messages{summarized}"
    if event == "error":
        return f"❌ {d.get('message')}"
    return None


def start_turn(label: str) -> dict:
    turn = {"no": len(ss.turns) + 1, "label": label, "time": datetime.now().strftime("%H:%M:%S"),
            "start": time.monotonic(), "events": [], "route": None, "seconds": None}
    ss.turns.append(turn)
    return turn


def turn_title(t: dict) -> str:
    label = t["label"] if len(t["label"]) <= 60 else t["label"][:57] + "…"
    route = f" · {t['route']}" if t.get("route") else ""
    if t.get("interrupted"):
        took = " · interrupted"
    else:
        took = f" · {t['seconds']:.1f} s" if t.get("seconds") is not None else " · running…"
    return f"#{t['no']} · {t['time']} · {label}{route}{took}"


def render_turn_events(t: dict) -> None:
    for event, data, at in t["events"]:
        text = describe(event, data)
        if text:
            st.markdown(f"`+{at:.1f}s` {text}")  # time since the question was sent


def render_activity(container) -> None:
    """Latest question open at the top; earlier ones collapsed below, each labelled with its number."""
    with container.container():
        if not ss.turns:
            st.caption("Ask a question – every step the agents take will appear here, grouped by question.")
            return
        latest, *older = reversed(ss.turns)
        st.markdown(f"**{turn_title(latest)}**")
        render_turn_events(latest)
        if older:
            st.divider()
            st.caption("Earlier questions")
            for t in older:
                with st.expander(turn_title(t)):
                    render_turn_events(t)


# ---------- sidebar: login ----------
with st.sidebar:
    st.header("ComBank Knowledge Assistant")
    st.caption("Internal staff assistant · sample data is fictional")
    if not ss.token:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", use_container_width=True):
                ok, err = api_login(username, password)
                if ok:
                    st.rerun()
                st.error(err)
        st.stop()

    u = ss.user
    st.markdown(f'<div class="cb-user"><div class="cb-name">{html.escape(u["display_name"])}</div>'
                f'<div>Role: <b>{html.escape(u["role"])}</b></div></div>', unsafe_allow_html=True)
    st.caption(f"Can read: {', '.join(u['clearance'])}")
    st.caption(f"Conversation: `{ss.thread_id}`")

    if u["role"] == "admin":
        # Demo of graceful degradation: simulate dependency outages (admin only, off by default).
        with st.expander("🧪 Simulate outages (demo)"):
            try:
                current = httpx.get(f"{API_URL}/admin/faults", headers=auth_headers(), timeout=TIMEOUT).json()
            except httpx.HTTPError:
                current = {}
            labels = {"pinecone": "Pinecone (vector DB) down", "llm": "LLM gateway down", "mcp": "MCP server down"}
            wanted = {k: st.checkbox(v, value=bool(current.get(k)), key=f"fault_{k}") for k, v in labels.items()}
            if wanted != {k: bool(current.get(k)) for k in labels}:
                httpx.post(f"{API_URL}/admin/faults", json=wanted, headers=auth_headers(), timeout=TIMEOUT)
                st.rerun()
            if any(wanted.values()):
                st.warning("Simulated outage active")
    c1, c2 = st.columns(2)
    if c1.button("New chat"):
        ss.thread_id, ss.messages, ss.turns, ss.pending, ss.job = uuid.uuid4().hex[:12], [], [], None, None
        st.rerun()
    if c2.button("Log out"):
        for k in ("token", "user"):
            ss[k] = None
        ss.messages, ss.turns, ss.job = [], [], None
        st.rerun()

# ---------- main layout ----------
# The chat input is created at page level (not inside the chat column): there Streamlit pins it to the
# bottom of the window. Inside a column it is drawn inline where the script calls it, so new messages
# appeared below it and the box ended up between older and newer messages.
# While an answer is being generated the input is disabled: Streamlit restarts the script on any input,
# which used to cut off the answer in progress (it was lost and its activity stayed "running…").
# (Fixed key and placeholder: changing the placeholder made Streamlit draw a second input box.)
prompt = st.chat_input("Ask about policies, incidents, runbooks…", key="question",
                       disabled=bool(ss.job) or bool(ss.pending))

chat_col, panel_col = st.columns([3, 2], gap="large")

with panel_col:
    st.subheader("Agent activity")
    panel = st.empty()
    render_activity(panel)


def turn_caption(m: dict) -> None:
    """Small line under a message: the turn number that matches the activity panel, and the trace link."""
    parts = [f"#{m['turn']}"] if m.get("turn") else []
    if m.get("time"):
        parts.append(m["time"])
    if m.get("seconds") is not None:
        parts.append(f"{m['seconds']:.1f} s")
    if m.get("trace"):
        parts.append(f"[LangSmith trace]({m['trace']})")
    if parts:
        st.caption(" · ".join(parts))


with chat_col:
    st.subheader("Chat")
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            turn_caption(m)

    def run_turn(path: str, payload: dict, turn: dict) -> None:
        """Stream one turn (new question or resumed run) into the chat and the activity panel."""
        answer, run_id = "", None
        with st.chat_message("assistant"):
            box = st.empty()
            box.markdown("_Thinking…_")
            for event, data in stream_events(path, payload):
                if event == "token":
                    answer += data["text"]
                    box.markdown(answer + "▌")
                    continue
                turn["events"].append((event, data, time.monotonic() - turn["start"]))
                if event == "route_selected":
                    turn["route"] = data.get("route")
                if event == "done":
                    turn["seconds"] = time.monotonic() - turn["start"]
                render_activity(panel)
                if event == "validation" and data.get("action") == "retry":
                    answer = ""  # the rewritten answer streams in next and replaces this one
                    box.markdown("_Answer failed validation – revising…_")
                if event == "output_guard" and data.get("answer"):
                    answer = data["answer"]
                    box.markdown(answer)
                if event == "validation" and data.get("action") == "repaired" and data.get("answer"):
                    answer = data["answer"]
                    box.markdown(answer)
                if event == "approval_required":
                    ss.pending = data
                if event == "error":
                    answer = answer or f"⚠️ {data.get('message')}"
                if event == "done":
                    run_id = data.get("run_id")
            if ss.pending and not answer:
                answer = f"✋ This action needs approval: **{ss.pending.get('tool')}** {ss.pending.get('args') or ''}"
            box.markdown(answer)
            if turn["seconds"] is None:
                turn["seconds"] = time.monotonic() - turn["start"]
                render_activity(panel)
            url = trace_link(run_id) if run_id else None
            message = {"role": "assistant", "content": answer, "trace": url, "turn": turn["no"],
                       "seconds": turn["seconds"]}
            turn_caption(message)
        ss.messages.append(message)

    # Work is queued in session state and run on the next script run, with the input disabled.
    # 1) a new question: record it, then rerun so the input shows as disabled while the answer streams
    if prompt:
        turn = start_turn(prompt)
        ss.messages.append({"role": "user", "content": prompt, "turn": turn["no"], "time": turn["time"]})
        ss.job = {"path": "/chat", "payload": {"thread_id": ss.thread_id, "message": prompt},
                  "turn": turn["no"], "started": False}
        st.rerun()

    # 2) run the queued job; if it had already started, the previous run was cut off (e.g. a sidebar click)
    job = ss.job
    if job:
        turn = ss.turns[job["turn"] - 1]
        if job["started"]:
            turn["interrupted"], turn["seconds"] = True, time.monotonic() - turn["start"]
            ss.messages.append({"role": "assistant", "turn": turn["no"],
                                "content": "⚠️ This answer was interrupted before it finished. Please ask again."})
        else:
            job["started"] = True
            run_turn(job["path"], job["payload"], turn)
        ss.job = None
        st.rerun()  # re-enable the input (or show the approval buttons)

    # Human-in-the-loop: a paused run waits here until the user decides.
    if ss.pending:
        p = ss.pending
        st.warning(f"**Approval required** – run `{p.get('tool')}` with {p.get('args') or 'no arguments'}?  \n"
                   f"Reason: {p.get('reason') or '-'}")
        a, r = st.columns(2)
        decision = True if a.button("✅ Approve") else False if r.button("❌ Reject") else None
        if decision is not None:
            ss.pending = None
            turn = start_turn(f"{'Approved' if decision else 'Rejected'}: {p.get('tool')}")
            ss.job = {"path": "/chat/resume", "payload": {"thread_id": ss.thread_id, "approved": decision},
                      "turn": turn["no"], "started": False}
            st.rerun()
