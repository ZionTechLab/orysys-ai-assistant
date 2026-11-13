"""Token-bucket rate limiting.

Each key (user id, or username for login) has a bucket holding up to `capacity` tokens that
refills continuously at `refill_per_sec`. A request takes one token; an empty bucket means
"retry after N seconds". Bursts up to `capacity` are allowed, the long-run rate is the refill rate.

Buckets live in process memory – right for one API instance. With several instances the same
interface would be backed by Redis (atomic Lua script) so all instances share the counts.
"""

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.config import get_settings
from app.errors import RateLimitedError
from app.logging import get_logger

log = get_logger(__name__)

IDLE_EVICT_SECONDS = 3600


@dataclass(frozen=True)
class Limit:
    capacity: float
    refill_per_sec: float


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: int
    retry_after: int  # whole seconds, >= 1 when not allowed


class TokenBucketLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def take(self, key: str, limit: Limit, cost: float = 1.0) -> Decision:
        async with self._lock:
            now = self._clock()
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(limit.capacity, now)
            # refill for the time since the last request, never above capacity
            b.tokens = min(limit.capacity, b.tokens + (now - b.updated) * limit.refill_per_sec)
            b.updated = now
            if b.tokens >= cost:
                b.tokens -= cost
                return Decision(True, math.floor(b.tokens), 0)
            wait = (cost - b.tokens) / limit.refill_per_sec if limit.refill_per_sec > 0 else 3600
            return Decision(False, 0, max(1, math.ceil(wait)))

    def evict_idle(self, max_idle: float = IDLE_EVICT_SECONDS) -> int:
        """Drop buckets not used for a while (they would be full again anyway). Keeps memory bounded."""
        now = self._clock()
        stale = [k for k, b in self._buckets.items() if now - b.updated > max_idle]
        for k in stale:
            del self._buckets[k]
        return len(stale)


def limit_for_role(role: str) -> Limit:
    s = get_settings()
    override = (s.rate_limit_role_overrides or {}).get(role, {})
    return Limit(float(override.get("capacity", s.rate_limit_capacity)),
                 float(override.get("refill_per_sec", s.rate_limit_refill_per_sec)))


def login_limit() -> Limit:
    s = get_settings()
    return Limit(float(s.login_rate_capacity), float(s.login_rate_refill_per_sec))


CHAT_LIMITER = TokenBucketLimiter()
LOGIN_LIMITER = TokenBucketLimiter()


async def enforce(limiter: TokenBucketLimiter, key: str, limit: Limit, scope: str) -> Decision:
    decision = await limiter.take(key, limit)
    if not decision.allowed:
        log.warning("rate_limited", scope=scope, key=key, retry_after=decision.retry_after)
        raise RateLimitedError(decision.retry_after,
                               f"Too many requests. Please wait {decision.retry_after} seconds and try again.")
    if len(limiter._buckets) > 10_000:  # noqa: SLF001 – cheap periodic housekeeping
        limiter.evict_idle()
    return decision
