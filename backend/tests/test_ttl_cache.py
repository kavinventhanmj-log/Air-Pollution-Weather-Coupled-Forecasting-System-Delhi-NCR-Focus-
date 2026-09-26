"""Tests for the TTL cache used to speed up control-room aggregations.

The cache is intentionally bypassed on SQLite (tests/local), so these cases
force ``_cache_enabled`` on and exercise the TTL + invalidation behaviour
directly.
"""

from app.services import ttl_cache


def test_cached_builds_once_until_ttl(monkeypatch):
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    builds = {"n": 0}

    def builder():
        builds["n"] += 1
        return {"value": builds["n"]}

    first = ttl_cache.cached("k1", 60, builder)
    second = ttl_cache.cached("k1", 60, builder)
    assert first == {"value": 1}
    assert second == {"value": 1}
    assert builds["n"] == 1


def test_cached_rebuilds_after_ttl_expiry(monkeypatch):
    """Expiry is driven by an injected clock, not a real sleep.

    A previous version slept 60 ms against a 50 ms TTL. That 10 ms margin is
    inside the Windows ~15.6 ms timer granularity, so the sleep could return
    before the TTL had actually elapsed and the case failed roughly 1 run in 6.
    """
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    clock = {"now": 1000.0}
    monkeypatch.setattr(ttl_cache.time, "monotonic", lambda: clock["now"])
    builds = {"n": 0}

    def builder():
        builds["n"] += 1
        return builds["n"]

    assert ttl_cache.cached("k2", 60, builder) == 1

    # Just inside the TTL: still cached.
    clock["now"] += 59.0
    assert ttl_cache.cached("k2", 60, builder) == 1
    assert builds["n"] == 1

    # Past the TTL: rebuilt.
    clock["now"] += 2.0
    assert ttl_cache.cached("k2", 60, builder) == 2
    assert builds["n"] == 2


def test_cached_rebuilds_after_zero_ttl(monkeypatch):
    """A zero TTL never serves a stale value, with no timing dependency."""
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    builds = {"n": 0}

    def builder():
        builds["n"] += 1
        return builds["n"]

    assert ttl_cache.cached("k-zero", 0, builder) == 1
    assert ttl_cache.cached("k-zero", 0, builder) == 2
    assert builds["n"] == 2


def test_cached_distinguishes_keys(monkeypatch):
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    assert ttl_cache.cached("a", 60, lambda: "A") == "A"
    assert ttl_cache.cached("b", 60, lambda: "B") == "B"
    assert ttl_cache.cached("a", 60, lambda: "A2") == "A"


def test_invalidate_all_clears_every_key(monkeypatch):
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    ttl_cache.cached("x", 60, lambda: 1)
    ttl_cache.cached("y", 60, lambda: 2)
    assert ttl_cache.cache_info()["entries"] >= 2
    ttl_cache.invalidate_all()
    assert ttl_cache.cache_info()["entries"] == 0
    assert ttl_cache.cached("x", 60, lambda: 100) == 100


def test_cached_bypasses_when_sqlite_backend(monkeypatch):
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: False)
    builds = {"n": 0}

    def builder():
        builds["n"] += 1
        return builds["n"]

    assert ttl_cache.cached("z", 60, builder) == 1
    assert ttl_cache.cached("z", 60, builder) == 2
    assert builds["n"] == 2


def teardown_module(_module):
    ttl_cache.invalidate_all()
