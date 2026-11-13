"""Prompt-injection protection: user input and retrieved content.

Three categories from the brief:
  instruction_override – "ignore previous instructions", "you are now ...", "developer mode"
  data_exfiltration    – "reveal your system prompt", "list all restricted documents", "send ... to http"
  tool_abuse           – "bypass approval", "make me admin", "run admin_reindex without approval"

User input:   normalise (remove zero-width / control characters), then two rule tiers:
              BLOCK – wording aimed at the assistant itself ("ignore your rules", "reveal your system
                      prompt", "make me admin"): fixed refusal, no LLM.
              REFER – the same topics in wording that staff also use in normal questions ("can I skip
                      approval for small expenses?"): not blocked by rule; the LLM classifier (graph
                      node) decides. Softer "suspicious" markers go to the classifier too.
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

_VERBS_OVERRIDE = r"\b(ignore|disregard|forget|override)\b"
_VERBS_BYPASS = r"\b(bypass|skip|disable|turn off|ignore|override|circumvent)\b"
_SENTENCE_START = r"(?:^|[.!?]\s+|\n\s*)\W*(?:(?:please|now|just|ok|okay|hey)[,!]?\s+)*"

# Tier 1 – BLOCK. Only wording aimed at the assistant itself ("your rules", "previous instructions",
# "your system prompt", "make me admin"). A match ends the turn with a fixed refusal, no LLM call.
_BLOCK: dict[str, list[tuple[str, str]]] = {
    "instruction_override": [
        ("ignore_previous", _VERBS_OVERRIDE + r".{0,30}\b(previous|prior|above|preceding|earlier|system)\s+"
                            r"(instructions?|prompts?|directions|messages?)\b"),
        ("ignore_your_rules", _VERBS_OVERRIDE + r".{0,30}\byour\s+(own\s+)?(instructions?|rules|prompts?|directions|"
                              r"guidelines|guardrails|restrictions|safety)\b"),
        ("new_persona", r"\b(you are now|from now on,? you are|pretend (to be|you are)|act as) (an? )?"
                        r"(unrestricted|unfiltered|jailbroken|evil|dan\b)"),
        ("assume_privileged_role", r"\b(you are now|from now on,? you are|pretend (to be|you are)) (an? )?"
                                   r"(admin|administrator|system|developer)\b"),
        ("special_mode", r"\b(jailbreak|god|unrestricted|dan) mode\b|"
                         r"\b(enable|enter|activate|switch (in)?to|turn on|go into)\s+(your\s+)?developer mode\b"
                         r"(?!\s+(on|for|in|of)\b)"),
        ("system_role_spoof", r"(^|\n)\s*(system|assistant)\s*:\s"),
        ("new_instructions", r"\b(new|updated|real) (system )?instructions\s*:"),
    ],
    "data_exfiltration": [
        ("prompt_leak", r"\b(reveal|show|print|repeat|output|display|dump|leak|tell me|give me)\b.{0,25}"
                        r"\b(your|the) (system|hidden|initial|original) (prompt|instructions)\b"
                        r"(?!\s+(injection|polic|guideline|standard|template))|"
                        r"\b(reveal|show|print|repeat|output|display|dump|leak)\b.{0,15}\byour (instructions|prompt|rules)\b|"
                        r"\bwhat (is|are) your (system |hidden |initial )?(instructions|prompt)\b"),
        ("secret_leak", r"\b(reveal|show|print|list|give me|tell me|dump|leak|read|open)\b.{0,30}"
                        r"(\.env\b|\byour (api[ _-]?keys?|secret keys?|access tokens?|credentials|jwt secret)\b)"),
        ("send_out", r"\b(send|post|upload|forward|email|exfiltrate)\b.{0,40}https?://"),
        ("encode_out", r"\b(base64|hex|rot13)[- ]?(encode|encoded)\b.{0,40}\b(prompt|document|secret|key)"),
    ],
    "tool_abuse": [
        # A command to the assistant (starts a sentence), or aimed at the assistant's own controls.
        ("bypass_control", _SENTENCE_START + _VERBS_BYPASS + r".{0,25}\b(approval|rbac|permissions?|authori[sz]ation|"
                           r"access control|guard(rail)?s?|audit)\b|"
                           + _VERBS_BYPASS + r".{0,25}\b(rbac|guardrails?|your (rules|restrictions|safety|filters?|"
                           r"permission checks?))\b"),
        ("privilege_escalation", r"\b(make me|set my role to|change my role to|elevate (me|my role)|promote me)\b.{0,15}"
                                 r"\b(admin|administrator|root|superuser)\b"),
        ("unapproved_admin_tool", r"\b(run|call|execute|invoke|use)\b.{0,30}\badmin_\w+\b.{0,40}"
                                  r"\b(without|no) (approval|asking|confirmation)\b"),
    ],
}

# Tier 2 – REFER. The same topics, but the wording is also normal for staff questions
# ("can I skip approval for small expenses?", "what does the runbook say about debug mode?").
# Not blocked by rule: the LLM classifier decides, with the matched rule shown in the panel.
_REFER: dict[str, list[tuple[str, str]]] = {
    "instruction_override": [
        ("ignore_rules", _VERBS_OVERRIDE + r".{0,30}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}"
                         r"\b(instructions?|rules|prompts?|directions|guidelines|guardrails)\b"),
        ("persona", r"\b(you are now|pretend (to be|you are)|act as)\b.{0,20}\b(admin|administrator|system|developer)\b"),
        ("special_mode", r"\b(developer|maintenance|debug) mode\b"),
    ],
    "data_exfiltration": [
        ("prompt_leak", r"\b(reveal|show|print|repeat|output|tell me|give me|what (is|are))\b.{0,25}"
                        r"\b(system prompt|your (instructions|prompt|rules)|hidden (instructions|prompt)|"
                        r"initial prompt)\b"),
        ("secret_leak", r"\b(reveal|show|print|list|give me|tell me|dump)\b.{0,30}"
                        r"\b(api[ _-]?keys?|secret keys?|access tokens?|jwt|credentials|env(ironment)? variables|"
                        r"connection strings?)\b"),
        ("bulk_restricted", r"\b(list|show|dump|give me|print)\b.{0,20}\b(all|every)\b.{0,20}"
                            r"\b(restricted|confidential|classified)\b"),
        ("send_out", r"\b(send|post|upload|forward|email|exfiltrate)\b.{0,40}\bto (this|my|an?) "
                     r"(url|server|webhook|email)"),
    ],
    "tool_abuse": [
        ("bypass_control", _VERBS_BYPASS + r".{0,25}\b(approval|rbac|permissions?|authori[sz]ation|access control|"
                           r"guard(rail)?s?|security|audit)\b"),
        ("privilege_escalation", r"\b(make me|grant me|give me|change my role|elevate)\b.{0,20}"
                                 r"\b(admin|administrator|superuser|full access|root access|analyst)\b"),
    ],
}


def _compile(tier: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, re.Pattern]]]:
    return {cat: [(name, re.compile(rx, re.IGNORECASE)) for name, rx in rules] for cat, rules in tier.items()}


_BLOCK_RX, _REFER_RX = _compile(_BLOCK), _compile(_REFER)

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
    suspicious: bool = False        # needs a second look by the LLM classifier
    normalized: str = ""
    referred_by: str | None = None  # "category/rule" of the tier-2 rule that asked for the second look


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # fold look-alike characters (full-width letters etc.)
    text = _INVISIBLE.sub("", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()


def detect_injection(text: str) -> GuardResult:
    """Rules only (no LLM). blocked = tier 1 matched; suspicious = tier 2 or a soft marker matched."""
    clean = normalize(text)[:MAX_INPUT_CHARS]
    for category, rules in _BLOCK_RX.items():
        for name, rx in rules:
            if rx.search(clean):
                return GuardResult(True, category, name, True, clean)
    for category, rules in _REFER_RX.items():
        for name, rx in rules:
            if rx.search(clean):
                return GuardResult(False, None, None, True, clean, referred_by=f"{category}/{name}")
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
