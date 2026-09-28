"""Regression tests for TTL-cache isolation between tests.

The process-global TTL cache is enabled on PostgreSQL and disabled on SQLite, so
gating it on the database dialect let a payload computed for one test be served
to the next whenever CI ran against Postgres. Two tests asserting an empty
result failed there while passing on SQLite:

* ``test_full_pipeline.py::TestDispersionForecast::test_dispersion_forecast_with_no_surface``
  expected ``frames == []`` and received 24 cached frames.
* ``test_api.py::test_get_alerts_empty`` expected ``[]`` and received 2 alerts.

These tests pin the two halves of the fix: caching is refused while pytest runs
regardless of dialect, and a value cached for one request is never served to a
later one.
"""

from __future__ import annotations

import pytest
from app.services import ttl_cache
from app.services.ttl_cache import _running_under_pytest, cache_info, cached, invalidate_all


@pytest.fixture(autouse=True)
def _clean_cache():
    invalidate_all()
    yield
    invalidate_all()


def test_cache_is_disabled_while_pytest_runs():
    """Caching must be off during tests no matter which database is configured.

    This is the regression that matters. The cache is on for PostgreSQL in
    production, so any test asserting an empty or freshly-changed result would
    otherwise be answered from a previous test's payload.
    """
    assert _running_under_pytest() is True
    assert ttl_cache._cache_enabled() is False


def test_cache_info_reports_disabled():
    assert cache_info()["enabled"] == 0


def test_builder_runs_every_time_under_pytest():
    """A repeated call must re-run the builder, not replay the first result."""
    calls = []

    def builder():
        calls.append(len(calls))
        return calls[-1]

    first = cached("k", 300, builder)
    second = cached("k", 300, builder)

    assert first == 0
    assert second == 1, "second call served a cached value from the first"
    assert len(calls) == 2
    assert cache_info()["entries"] == 0, "nothing should be stored under pytest"


def test_mutated_database_is_observed_immediately():
    """The shape of the two real failures: value changes, cache must not mask it.

    ``frames`` goes empty, the cache would still hold the earlier 24 frames.
    With caching disabled the caller sees the new value on the very next call.
    """
    state = {"frames": [1] * 24}
    assert len(cached("dispersion", 300, lambda: state["frames"])) == 24

    state["frames"] = []
    assert cached("dispersion", 300, lambda: state["frames"]) == []


def test_invalidate_all_is_safe_when_empty():
    """The autouse fixture calls this on every test, including before any write."""
    invalidate_all()
    invalidate_all()
    assert cache_info()["entries"] == 0


def test_cache_enabled_outside_pytest(monkeypatch):
    """A PostgreSQL URL must still enable caching in a real request path.

    Disabling the cache under pytest must not disable it in production, which
    is the whole reason the cache exists: the Render free tier pays 10-30 s per
    heavy aggregation and a dashboard mount fans out ~14 of them.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "pytest", None)

    import app.config as config

    monkeypatch.setattr(
        config, "get_settings", lambda: type("S", (), {"database_url": "postgresql://u@h/db"})()
    )
    assert ttl_cache._cache_enabled() is True


def test_cache_stays_disabled_for_sqlite_outside_pytest(monkeypatch):
    """SQLite keeps caching off even outside pytest, as before."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "pytest", None)

    import app.config as config

    monkeypatch.setattr(
        config, "get_settings", lambda: type("S", (), {"database_url": "sqlite:///./x.db"})()
    )
    assert ttl_cache._cache_enabled() is False


def test_postgres_path_would_cache_without_the_pytest_gate(monkeypatch):
    """Demonstrate the bug the gate prevents.

    With a PostgreSQL URL and no pytest detection, the cache engages and the
    second identical call is served from the store. This is exactly what
    happened in CI, where the tests ran under a Postgres ``DATABASE_URL``.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "pytest", None)

    import app.config as config

    monkeypatch.setattr(
        config, "get_settings", lambda: type("S", (), {"database_url": "postgresql://u@h/db"})()
    )
    assert ttl_cache._cache_enabled() is True

    calls = []
    a = cached("pg", 300, lambda: calls.append(1) or "value")
    b = cached("pg", 300, lambda: calls.append(1) or "value")
    assert a == b == "value"
    assert len(calls) == 1, "under a Postgres URL the value is cached - hence the bug"
    assert cache_info()["entries"] == 1
