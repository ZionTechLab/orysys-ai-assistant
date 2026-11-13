"""Smoke test for the LLM gateway: plain call, JSON (structured) output, streaming.

Run: .venv/Scripts/python scripts/smoke_llm.py
Checks the things the graph depends on before we build on them.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm.factory import get_llm  # noqa: E402


class Route(BaseModel):
    intent: str
    route: str
    sub_tasks: list[str]


ROUTE_PROMPT = """You are a router. Reply with JSON only, no prose, matching:
{"intent": string, "route": one of "retrieve" | "research" | "tool" | "chitchat", "sub_tasks": [string]}"""


def _parse_json(text: str) -> dict:
    """Open models sometimes wrap JSON in ``` fences; strip them before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


async def main() -> int:
    s = get_settings()
    print(f"Gateway: {s.llm_base_url} | strong={s.llm_model_strong} | fast={s.llm_model_fast}")
    ok = True

    # 1. Plain call
    t = time.perf_counter()
    reply = await get_llm("fast").ainvoke([HumanMessage("Reply with the single word: ready")])
    print(f"[1] plain call      : {reply.content.strip()[:40]!r} ({time.perf_counter() - t:.1f}s)")

    # 2. JSON routing output validated by pydantic
    t = time.perf_counter()
    reply = await get_llm("strong").ainvoke([
        SystemMessage(ROUTE_PROMPT),
        HumanMessage("Summarise all payment outages last year and find recurring root causes."),
    ])
    try:
        route = Route.model_validate(_parse_json(reply.content))
        print(f"[2] JSON route      : route={route.route}, sub_tasks={len(route.sub_tasks)} ({time.perf_counter() - t:.1f}s)")
    except (json.JSONDecodeError, ValidationError) as exc:
        ok = False
        print(f"[2] JSON route      : FAILED {type(exc).__name__}: {reply.content[:200]!r}")

    # 3. Streaming
    t = time.perf_counter()
    chunks, first = 0, None
    async for chunk in get_llm("strong", streaming=True).astream([HumanMessage("Count from 1 to 10, comma separated.")]):
        if chunk.content:
            chunks += 1
            first = first or time.perf_counter() - t
    streamed = chunks > 1
    ok &= streamed
    print(f"[3] streaming       : {chunks} chunks, first token {first or 0:.1f}s ({'ok' if streamed else 'NOT streaming'})")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
