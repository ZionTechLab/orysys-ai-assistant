"""Prompt-injection protection: user input and retrieved content.

Three categories from the brief:
  instruction_override – "ignore previous instructions", "you are now ...", "developer mode"
  data_exfiltration    – "reveal your system prompt", "list all restricted documents", "send ... to http"
  tool_abuse           – "bypass approval", "make me admin", "run admin_reindex without approval"

User input:   normalise (remove zero-width / control characters) -> high-confidence rules block ->
              softer "suspicious" markers send the message to an LLM classifier (graph node).
Retrieved content: remove hidden HTML comments, neutralise tags that could break out of the
              <document> wrapper, and flag injection-like text so it is visible in traces.

Rules are a first, cheap, explainable layer – not a complete defence. The other layers (RBAC
at execution, clearance filters, untrusted-data wrapping, the output guard) do not depend on
an attack being recognised.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Zero-width and bidi control characters are used to hide instructions from humans and filters.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")

_RULES: dict[str, list[tuple[str, str]]] = {
    "instruction_override": [
        ("ignore_previous", r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|the)"
                            r"\b.{0,20}\b(instructions?|rules|prompts?|directions|guidelines|guardrails)\b"),
        ("new_persona", r"\b(you are now|from now on,? you are|pretend (to be|you are)|act as) (an? )?"
                        r"(unrestricted|unfiltered|jailbroken|evil|dan\b|admin|administrator|system|developer)"),
        ("special_mode", r"\b(developer|maintenance|debug|god|jailbreak|unrestricted) mode\b"),
        ("system_role_spoof", r"(^|\n)\s*(system|assistant)\s*:\s"),
        ("new_instructions", r"\b(new|updated|real) (system )?instructions\s*:"),
    ],
    "data_exfiltration": [
        ("prompt_leak", r"\b(reveal|show|print|repeat|output|tell me|give me|what (is|are))\b.{0,25}"
                        r"\b(system prompt|your (instructions|prompt|rules)|hidden (instructions|prompt)|"
                        r"initial prompt)\b"),
        ("secret_leak", r"\b(reveal|show|print|list|give me|tell me|dump)\b.{0,30}"
                        r"\b(api[ _-]?keys?|secret keys?|access tokens?|jwt|credentials|env(ironment)? variables|"
                        r"\.env|connection strings?)\b"),
        ("bulk_restricted", r"\b(list|show|dump|give me|print)\b.{0,20}\b(all|every)\b.{0,20}"
                            r"\b(restricted|confidential|classified)\b"),
        ("send_out", r"\b(send|post|upload|forward|email|exfiltrate)\b.{0,40}\b(https?://|to (this|my|an?) "
                     r"(url|server|webhook|email))"),
        ("encode_out", r"\b(base64|hex|rot13)[- ]?(encode|encoded)\b.{0,40}\b(prompt|document|secret|key)"),
    ],
    "tool_abuse": [
        ("bypass_control", r"\b(bypass|skip|disable|turn off|ignore|override|circumvent)\b.{0,25}"
                           r"\b(approval|rbac|permissions?|authori[sz]ation|access control|guard(rail)?s?|"
                           r"security|audit)\b"),
        ("privilege_escalation", r"\b(make me|grant me|give me|set my role to|change my role|elevate)\b.{0,20}"
                                 r"\b(admin|administrator|root|superuser|full access|analyst)\b"),
        ("unapproved_admin_tool", r"\b(run|call|execute|invoke|use)\b.{0,30}\badmin_\w+\b.{0,40}"
                                  r"\b(without|no) (approval|asking|confirmation)\b"),
    ],
}
_COMPILED = {cat: [(name, re.compile(rx, re.IGNORECASE)) for name, rx in rules] for cat, rules in _RULES.items()}

# Softer markers: not blocked, but worth a second look by the LLM classifier.
_SUSPICIOUS = re.compile(
    r"\b(instructions?|prompt|pretend|role[- ]?play|persona|jailbreak|admin(istrator)?|bypass|override|"
    r"secret|token|restricted|confidential|unrestricted|hidden|policy says you|as an ai)\b", re.IGNORECASE)

MAX_INPUT_CHARS = 4000


@dataclass
class GuardResult:
    blocked: bool
    category: str | None = None
    rule: str | None = None
    suspicious: bool = False
    normalized: str = ""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # fold look-alike characters (full-width letters etc.)
    text = _INVISIBLE.sub("", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()


def detect_injection(text: str) -> GuardResult:
    clean = normalize(text)[:MAX_INPUT_CHARS]
    for category, rules in _COMPILED.items():
        for name, rx in rules:
            if rx.search(clean):
                return GuardResult(True, category, name, True, clean)
    return GuardResult(False, None, None, bool(_SUSPICIOUS.search(clean)), clean)


# ---------- retrieved content ----------

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_BREAKOUT = re.compile(r"</?\s*(document|system|instructions?|assistant|user)\b[^>]*>", re.IGNORECASE)


@dataclass
class ContentCheck:
    text: str
    flags: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


def sanitize_retrieved(text: str) -> ContentCheck:
    """Clean document text before any model sees it. Never raises; returns flags for visibility."""
    flags: list[str] = []
    clean = normalize(text)
    if _HTML_COMMENT.search(clean):
        # Hidden comments are invisible to people reading the rendered document – exactly where
        # injected instructions hide (see MTG-005). They carry no content staff rely on.
        clean = _HTML_COMMENT.sub(" ", clean)
        flags.append("hidden_comment_removed")
    if _TAG_BREAKOUT.search(clean):
        clean = _TAG_BREAKOUT.sub("[tag removed]", clean)
        flags.append("tag_breakout_removed")
    remaining = detect_injection(clean)
    if remaining.blocked:
        # Visible injection text is kept (it may be legitimate content, e.g. a security training doc)
        # but flagged; the model is instructed to treat document text as data only.
        flags.append(f"injection_pattern:{remaining.rule}")
    return ContentCheck(re.sub(r"[ \t]{2,}", " ", clean).strip(), flags)
