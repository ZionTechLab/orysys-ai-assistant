"""Fault injection – simulate dependency outages to demonstrate graceful degradation.

Toggled at runtime by an administrator (POST /admin/faults or the UI sidebar). Off by default.
Each check sits at the boundary to the real dependency, so everything above it – retries,
fallbacks, degraded answers – runs exactly as it would in a real outage.
"""

from app.logging import get_logger

log = get_logger(__name__)

KNOWN = ("pinecone", "llm", "mcp")
_ACTIVE: dict[str, bool] = {name: False for name in KNOWN}


class SimulatedOutage(ConnectionError):
    """Raised instead of calling a dependency while its fault is switched on."""


def active(name: str) -> bool:
    return _ACTIVE.get(name, False)


def check(name: str) -> None:
    if active(name):
        raise SimulatedOutage(f"simulated {name} outage (fault injection)")


def set_faults(changes: dict[str, bool]) -> dict[str, bool]:
    for name, on in changes.items():
        if name not in KNOWN:
            raise ValueError(f"unknown fault {name!r}; known: {', '.join(KNOWN)}")
        _ACTIVE[name] = bool(on)
    log.warning("faults_changed", **_ACTIVE)
    return dict(_ACTIVE)


def current() -> dict[str, bool]:
    return dict(_ACTIVE)
