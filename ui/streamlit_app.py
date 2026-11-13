"""Streamlit UI: chat window + live Agent Activity Panel.

Run: .venv/Scripts/streamlit run ui/streamlit_app.py
Talks only to the FastAPI backend (API_URL); it holds no business logic.
"""

import json
import os
import time
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

st.set_page_config(page_title="ComBank Knowledge Assistant", layout="wide")

# ---------- session state ----------
ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("user", None)
ss.setdefault("thread_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])  # [{"role", "content", "citations", "trace"}]
ss.setdefault("activity", [])  # events of the latest turn

NODE_LABELS = {
    "supervisor": "Supervisor – understanding the question",
    "retrieval_agent": "Retrieval agent – searching documents",
    "research_planner": "Research planner – writing a Python search plan",
    "research_worker": "Research worker – analysing a batch",
    "research_aggregator": "Research aggregator – combining findings",
    "response_agent": "Response agent – writing the answer",
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


def stream_chat(message: str):
    """Yield (event, data) pairs from the /chat Server-Sent Events stream."""
    with httpx.stream("POST", f"{API_URL}/chat", json={"thread_id": ss.thread_id, "message": message},
                      headers=auth_headers(), timeout=TIMEOUT) as r:
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
    if event == "memory_update":
        return f"🧠 Memory saved – conversation has {d.get('messages')} messages"
    if event == "error":
        return f"❌ {d.get('message')}"
    return None


def render_activity(container, events: list[tuple[str, dict]]) -> None:
    with container.container():
        for event, data in events:
            text = describe(event, data)
            if text:
                st.markdown(text)


# ---------- sidebar: login ----------
with st.sidebar:
    st.header("ComBank Knowledge Assistant")
    st.caption("Internal staff assistant · sample data is fictional")
    if not ss.token:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                ok, err = api_login(username, password)
                if ok:
                    st.rerun()
                st.error(err)
        st.stop()

    u = ss.user
    st.success(f"{u['display_name']}  \nRole: **{u['role']}**")
    st.caption(f"Can read: {', '.join(u['clearance'])}")
    st.caption(f"Conversation: `{ss.thread_id}`")
    c1, c2 = st.columns(2)
    if c1.button("New chat"):
        ss.thread_id, ss.messages, ss.activity = uuid.uuid4().hex[:12], [], []
        st.rerun()
    if c2.button("Log out"):
        for k in ("token", "user"):
            ss[k] = None
        ss.messages, ss.activity = [], []
        st.rerun()

# ---------- main layout ----------
chat_col, panel_col = st.columns([3, 2], gap="large")

with panel_col:
    st.subheader("Agent activity")
    panel = st.empty()
    render_activity(panel, ss.activity)

with chat_col:
    st.subheader("Chat")
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                st.caption(f"[LangSmith trace]({m['trace']})")

    prompt = st.chat_input("Ask about policies, incidents, runbooks…")
    if prompt:
        ss.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ss.activity = []
        answer, run_id = "", None
        with st.chat_message("assistant"):
            box = st.empty()
            box.markdown("_Thinking…_")
            for event, data in stream_chat(prompt):
                if event == "token":
                    answer += data["text"]
                    box.markdown(answer + "▌")
                    continue
                ss.activity.append((event, data))
                render_activity(panel, ss.activity)
                if event == "error":
                    answer = answer or f"⚠️ {data.get('message')}"
                if event == "done":
                    run_id = data.get("run_id")
            box.markdown(answer)
            url = trace_link(run_id) if run_id else None
            if url:
                st.caption(f"[LangSmith trace]({url})")
        ss.messages.append({"role": "assistant", "content": answer, "trace": url})
