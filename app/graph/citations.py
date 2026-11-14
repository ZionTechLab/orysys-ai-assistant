"""Citation parsing, shared by the response agent (citation list) and the validator (checks + repair).

Answers cite chunk ids in square brackets. Models write them two ways, and both must be checked:

    one per bracket   [INC-2026-034#summary] [INC-2026-034#root-cause]
    grouped           [INC-2026-034#summary, INC-2026-034#root-cause]   (also "; " or " and ")

Seen live (14 Nov, research answer): most citations were grouped, and the first parser only read
single-id brackets – so an invented id inside a group passed the hallucinated-citation check.
A bracket counts as a citation group when at least one of its parts is a chunk id; other brackets
(markdown links, "[sic]", "[1]") are left alone.
"""

import re
from collections.abc import Iterable

CHUNK_ID = re.compile(r"[A-Z]+-[\w-]+#[\w-]+")
_BRACKET = re.compile(r"\[([^\[\]\n]{1,500})\]")
_SEPARATOR = re.compile(r"\s*[,;]\s*|\s+and\s+")


def _ids(inner: str) -> list[str]:
    parts = (p.strip(" `*_") for p in _SEPARATOR.split(inner))
    return [p for p in parts if CHUNK_ID.fullmatch(p)]


def find_citations(text: str) -> list[str]:
    """Every chunk id cited in the text, in order of first appearance, without duplicates."""
    seen: dict[str, None] = {}
    for match in _BRACKET.finditer(text):
        for cid in _ids(match.group(1)):
            seen.setdefault(cid, None)
    return list(seen)


def remove_citations(text: str, allowed: Iterable[str]) -> str:
    """Drop cited ids that are not in `allowed`; keep the valid ids of a group, remove a group left empty."""
    allowed = set(allowed)

    def fix(match: re.Match) -> str:
        ids = _ids(match.group(1))
        if not ids:
            return match.group(0)  # not a citation
        kept = [cid for cid in ids if cid in allowed]
        return f"[{', '.join(kept)}]" if kept else ""

    return _BRACKET.sub(fix, text)
