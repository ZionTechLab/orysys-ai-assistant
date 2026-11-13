"""Health endpoint: reports each dependency so the UI and evaluator can see degraded mode."""

import asyncio
import time
from typing import Awaitable, Callable

import httpx
from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


async def _timed(name: str, check: Callable[[], Awaitable[str]], timeout: float) -> dict:
    """Run one check with a timeout. A failing check never fails the endpoint."""
    start = time.perf_counter()
    try:
        detail = await asyncio.wait_for(check(), timeout=timeout)
        status = "ok"
    except asyncio.TimeoutError:
        status, detail = "down", "timeout"
    except Exception as exc:  # noqa: BLE001 - report any failure as "down"
        status, detail = "down", type(exc).__name__
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


async def _check_llm_gateway() -> str:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.llm_api_key.get_secret_value()}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{s.llm_base_url.rstrip('/')}/models", headers=headers)
        resp.raise_for_status()
    return "reachable"


async def _check_pinecone() -> str:
    s = get_settings()
    if not s.pinecone_api_key.get_secret_value():
        raise RuntimeError("not_configured")
    headers = {"Api-Key": s.pinecone_api_key.get_secret_value(), "X-Pinecone-API-Version": "2025-01"}
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.pinecone.io/indexes", headers=headers)
        resp.raise_for_status()
    names = [i["name"] for i in resp.json().get("indexes", [])]
    return "index_ready" if s.pinecone_index in names else "index_missing"


async def _check_mcp() -> str:
    s = get_settings()
    async with httpx.AsyncClient() as client:
        await client.get(s.mcp_server_url)
    return "reachable"


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    checks = await asyncio.gather(
        _timed("llm_gateway", _check_llm_gateway, s.health_check_timeout_seconds),
        _timed("pinecone", _check_pinecone, s.health_check_timeout_seconds),
        _timed("mcp", _check_mcp, s.health_check_timeout_seconds),
    )
    overall = "ok" if all(c["status"] == "ok" for c in checks) else "degraded"
    return {"status": overall, "app": s.app_name, "env": s.app_env, "checks": checks}
