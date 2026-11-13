"""Conversation memory: recent messages verbatim + a rolling summary of older turns.

    messages:  [ m0 m1 m2 m3 m4 m5 | m6 m7 m8 m9 m10 m11 m12 ]
                 └─ summary.covered ─┘ └── shown to agents verbatim ──┘

- After each turn, the memory node folds messages that fell out of the recent window into
  `summary.text` – only when at least `memory_summarize_batch` have aged (fewer LLM calls).
- Agents see the summary + every message after `summary.covered`, so nothing falls into a gap
  between the summary and the window.
- All messages stay in the checkpoint (the UI can show the full history); only prompts are trimmed.
"""

from langchain_core.messages import AnyMessage

from app.config import get_settings

MAX_UNSUMMARISED = 16  # hard cap if summarising keeps failing: prompts never grow without limit
MAX_MESSAGE_CHARS = 1200  # per message, in the text given to the summariser


def conversation_context(messages: list[AnyMessage], summary: dict | None) -> tuple[str, list[AnyMessage]]:
    """(summary text, messages to show verbatim) for an agent prompt."""
    covered = min(int((summary or {}).get("covered", 0)), max(len(messages) - 1, 0))
    recent = messages[covered:][-MAX_UNSUMMARISED:]
    return (summary or {}).get("text", ""), recent


def messages_to_summarise(messages: list[AnyMessage], summary: dict | None) -> tuple[list[AnyMessage], int]:
    """(aged messages not yet in the summary, new covered index) – empty if not worth an LLM call yet."""
    s = get_settings()
    covered = int((summary or {}).get("covered", 0))
    boundary = len(messages) - s.memory_window_messages
    aged = messages[covered:boundary] if boundary > covered else []
    if len(aged) < s.memory_summarize_batch:
        return [], covered
    return aged, boundary


def transcript(messages: list[AnyMessage]) -> str:
    def role(m: AnyMessage) -> str:
        return "User" if m.type == "human" else "Assistant"

    return "\n".join(f"{role(m)}: {str(m.content)[:MAX_MESSAGE_CHARS]}" for m in messages)
