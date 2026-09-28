"""Tiny thread-safe TTL cache for expensive control-room aggregations.

The Render free tier talks to a pooled Neon Postgres that can take 10-30s to
answer the heavy count/aggregation queries powering the Control Room
(``/api/data-quality``, ``/api/atmosphere/current``, ``/api/transport-risk/current``,
``/api/summary``). Those results are recomputed on every request, so a dashboard
mount pays ~11s per panel and, after a cold start, requests blow past the
client's timeout and the panels render ``—``.

The underlying observations only change on the 3-hour live-refresh cadence (or
at a demo re-hydration on boot), so caching the serialisable results for a few
minutes is safe and turns every dashboard load after the first into sub-second
responses.

Determinism: the cache is bypassed entirely when the backend runs on SQLite
(local dev / pytest), where queries are already fast and test sessions mutate
the same database across tests.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

_lock = threading.Lock()
_store: dict[str, tuple[float, Any]] = {}


def _running_under_pytest() -> bool:
    """True while pytest is executing a test.

    Both conditions are required. ``pytest in sys.modules`` alone is true for
    any process that merely imported pytest, and ``PYTEST_CURRENT_TEST`` alone
    is only set *while* a test runs, so importing the module at collection
    time would not flip caching on.
    """
    return "pytest" in sys.modules and "PYTEST_CURRENT_TEST" in os.environ


def _cache_enabled() -> bool:
    # Tests mutate the database between cases while the cache is process-global,
    # so a cached value computed for one test is served to the next. Gating on
    # the dialect used to hide this: SQLite disabled the cache, so the full local
    # suite passed, while the PostgreSQL CI job - which enables the cache - fed
    # stale payloads to tests asserting an empty result.
    if _running_under_pytest():
        return False
    from ..config import get_settings

    try:
        url = get_settings().database_url
    except Exception:
        return False
    return not str(url).startswith("sqlite")


def cached(key: str, ttl_seconds: float, builder: Callable[[], Any]) -> Any:
    """Return ``builder()``'s result, reusing it until ``ttl_seconds`` expire.

    When the cache is disabled (SQLite backend) ``builder`` always runs, so
    tests keep full isolation between cases.
    """
    if not _cache_enabled():
        return builder()
    now = time.monotonic()
    with _lock:
        hit = _store.get(key)
        if hit is not None and now - hit[0] < ttl_seconds:
            return hit[1]
    value = builder()
    with _lock:
        _store[key] = (time.monotonic(), value)
    return value


def invalidate_all() -> None:
    """Drop every cached entry (e.g. after a live refresh / demo re-seed)."""
    with _lock:
        _store.clear()


def cache_info() -> dict[str, int]:
    with _lock:
        return {"entries": len(_store), "enabled": int(_cache_enabled())}
