"""Output checks – the last line before an answer reaches the user (rules: docs/brand_guardrails.md).

Deterministic (always, no model call):
  - secrets: key formats and the actual secret values configured for this deployment -> redacted
  - system-prompt leakage: distinctive sentences from our own prompts -> removed
  - clearance leakage: sentences naming a document above the user's clearance -> removed
Brand (only when risk markers appear, so normal answers pay nothing):
  - markers decide whether the fast-model compliance check runs (graph node does the LLM part)
"""

import re
from dataclasses import dataclass, field

from app.config import get_settings

DOC_ID = re.compile(r"\b(?:INC-\d{4}-\d{3}|RB-\d{3}|ARCH-\d{3}|POL-\d{3}|SPEC-\d{3}|MTG-\d{3})\b")

_SECRET_FORMATS = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{16,}"                              # OpenAI-style keys
    r"|pcsk_[A-Za-z0-9_]{20,}"                               # Pinecone
    r"|lsv2_[A-Za-z0-9_]{20,}"                               # LangSmith
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"  # JWT
    r"|(?:LLM|PINECONE|LANGSMITH|JWT|FALLBACK_LLM)_[A-Z_]*(?:KEY|SECRET)\s*[=:]\s*\S+)")

# Risk markers for brand rules. Matching only triggers a compliance check – it never blocks by itself.
_BRAND_MARKERS = re.compile(
    r"\b(invest(ment|ing)?|buy (shares|stock)|share price|stock price|dividend|financial advice|tax advice|"
    r"legal advice|guarantee[ds]?|we promise|promise you|compensation|refund you|interest rates? will|"
    r"HNB|Hatton National|Sampath|(?<!Commercial )Bank of Ceylon|People'?s Bank|NDB|Nations Trust|DFCC|Seylan|"
    r"Pan Asia|"
    r"better than|worse than|competitor|election|government|minister|political|religio\w*|"
    r"lol|haha|lmao|stupid|idiot|incompetent|careless|fault of|blame[ds]?|disaster|catastroph\w*|"
    r"press release|official statement|on behalf of the bank)\b", re.IGNORECASE)

PROMPT_LEAK_NOTE = "[content removed]"
# Neutral wording: saying *why* would confirm that a restricted document exists.
CLEARANCE_NOTE = "\n\n_Part of this answer was removed by the assistant's safety checks._"


@dataclass
class OutputCheck:
    text: str
    actions: list[str] = field(default_factory=list)
    brand_markers: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.actions)


def _prompt_fragments() -> list[str]:
    """Distinctive sentences from our prompts (long enough not to occur by chance)."""
    from app.llm import prompts

    fragments = set()
    user_facing = {prompts.REFUSAL}  # texts we show to users on purpose are not "leaks"
    for value in vars(prompts).values():
        if isinstance(value, str) and len(value) > 200 and value not in user_facing:
            for sentence in re.split(r"(?<=[.:!?])\s+|\n", value):
                sentence = " ".join(sentence.split())
                if len(sentence) >= 45 and "{" not in sentence:
                    fragments.add(sentence.lower())
    return sorted(fragments)


_FRAGMENTS: list[str] | None = None


def _secret_values() -> list[str]:
    s = get_settings()
    values = [v.get_secret_value() for v in (s.llm_api_key, s.pinecone_api_key, s.langsmith_api_key, s.jwt_secret,
                                             s.fallback_llm_api_key)]
    return [v for v in values if v and len(v) >= 8 and v != "change-me"]


def hidden_doc_ids(text: str, clearance: tuple[str, ...]) -> list[str]:
    from app.rlm.catalog import get_corpus

    docs = get_corpus().docs
    return sorted({d for d in DOC_ID.findall(text) if d in docs and docs[d].access_level not in clearance})


def check_output(text: str, clearance: tuple[str, ...]) -> OutputCheck:
    global _FRAGMENTS
    out = OutputCheck(text)

    # 1. secrets – formats and the real configured values
    redacted = _SECRET_FORMATS.sub("[redacted]", out.text)
    for value in _secret_values():
        redacted = redacted.replace(value, "[redacted]")
    if redacted != out.text:
        out.text, out.actions = redacted, out.actions + ["secret_redacted"]

    # 2. system-prompt leakage
    if _FRAGMENTS is None:
        _FRAGMENTS = _prompt_fragments()
    flat = " ".join(out.text.lower().split())
    leaked = [f for f in _FRAGMENTS if f in flat]
    if leaked:
        cleaned = out.text
        for f in leaked:
            cleaned = re.sub(r"\s+".join(map(re.escape, f.split())), PROMPT_LEAK_NOTE, cleaned, flags=re.IGNORECASE)
        out.text, out.actions = cleaned, out.actions + ["prompt_leak_removed"]

    # 3. clearance – drop sentences naming documents the user may not see
    hidden = hidden_doc_ids(out.text, clearance)
    if hidden:
        parts = re.split(r"(?<=[.!?])\s+|\n", out.text)
        kept = [p for p in parts if not any(h in p for h in hidden)]
        out.text = "\n".join(kept).strip() + CLEARANCE_NOTE
        out.actions = out.actions + [f"clearance_leak_removed:{','.join(hidden)}"]

    # 4. brand – markers only; the LLM check runs in the graph node
    out.brand_markers = sorted({m.group(0).lower() for m in _BRAND_MARKERS.finditer(out.text)})
    return out
