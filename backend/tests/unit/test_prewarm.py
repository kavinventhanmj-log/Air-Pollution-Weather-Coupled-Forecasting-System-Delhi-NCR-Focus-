"""Unit tests for the cold-start control-room cache pre-warm."""

import threading

import pytest


def test_entries_use_the_same_keys_as_the_http_layer():
    """A pre-warm under the wrong key fills a cache nobody reads.

    The keys are duplicated (not imported) so a rename on either side fails
    loudly here instead of silently costing a cold start.
    """
    from app.api import dispersion
    from app.services import prewarm

    entries = {key: (label, ttl) for label, key, ttl, _builder in prewarm._entries()}

    for key in (
        "atmosphere:current",
        "summary",
        "data-quality",
        "grap:current",
        "fire-activity",
        "fire-hotspots",
        "plume-risk",
        "grid:forecast:24",
        "transport-risk:current:72",
        "alerts:all_stations",
    ):
        assert key in entries, key

    # The dispersion entries must be the exact keys the endpoint derives from
    # the query the Spatial Forecast page sends. The frame set is part of the
    # key, so an unfiltered pre-warm is unreadable garbage the moment the client
    # filters — the bug this assertion was written for.
    for horizon in (24, 48, 72):
        hours = prewarm._dispersion_frame_hours(horizon)
        parsed = dispersion._parse_frame_hours(",".join(str(h) for h in hours))
        assert parsed == hours
        assert dispersion.dispersion_cache_key(horizon, 8, parsed) in entries

    assert "dispersion:72:8:all" not in entries
    assert [hours for hours in (prewarm._dispersion_frame_hours(h) for h in (24, 48, 72))] == [
        [6, 12, 18, 24],
        [6, 12, 18, 24, 30, 36, 42, 48],
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],
    ]

    # The expensive solver is the one panel a retry cannot rescue (a cold 12 s
    # serialisation either clears or blows the client timeout), and the sweep
    # shares one CPU with the inbound request, so it has to go first. Measured
    # against production: the three dispersion solves are ~33 s of the 76 s wake.
    labels = [label for label, _key, _ttl, _builder in prewarm._entries()]
    assert labels[0] == "dispersion:24:8:6-12-18-24"
    assert labels[1] == "dispersion:48:8:6-12-18-24-30-36-42-48"
    assert labels[2] == "dispersion:72:8:6-12-18-24-30-36-42-48-54-60-66-72"
    assert "summary" in labels
    assert labels.index("dispersion:72:8:6-12-18-24-30-36-42-48-54-60-66-72") < labels.index("summary")


def test_prewarm_populates_cache_and_survives_a_failing_entry(monkeypatch):
    from app.services import prewarm, ttl_cache

    calls: list[str] = []

    def boom(_db):
        raise RuntimeError("boom")

    entries = [
        ("one", "one", 60, lambda _db: {"label": "one"}),
        ("broken", "broken", 60, boom),
        ("two", "two", 60, lambda _db: {"label": "two"}),
    ]

    def fake_entries():
        calls.append("built")
        return entries

    monkeypatch.setattr(prewarm, "_entries", fake_entries)
    monkeypatch.setattr(prewarm, "_ENTRY_GAP_S", 0)
    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    ttl_cache.invalidate_all()

    prewarm.prewarm_control_room()

    # A failing entry is logged and skipped; the sweep continues and the good
    # entries still land in the cache.
    assert calls == ["built"]
    assert ttl_cache.cache_info()["entries"] == 2


def test_prewarm_honours_the_stop_event(monkeypatch):
    from app.services import prewarm

    ran: list[str] = []

    def builder(_db):
        ran.append("built")
        return {}

    monkeypatch.setattr(
        prewarm, "_entries", lambda: [(f"e{i}", f"e{i}", 60, builder) for i in range(3)]
    )
    monkeypatch.setattr(prewarm, "_ENTRY_GAP_S", 0)

    stop = threading.Event()
    stop.set()
    prewarm.prewarm_control_room(stop=stop)
    assert ran == []


def test_status_reports_progress_and_completion(monkeypatch):
    """A cache warm-up is invisible unless it says what it did.

    Verified live: a sweep that silently failed looked exactly like a sweep that
    was working, and the only way to tell was to time a request by hand.
    """
    from app.services import prewarm

    # The status is a module global, so isolate it from any earlier sweep in the
    # session rather than asserting on whatever the previous test left behind.
    monkeypatch.setattr(prewarm, "_STATUS", {"enabled": False, "state": "disabled"})
    monkeypatch.setattr(prewarm, "_ENTRY_GAP_S", 0)
    monkeypatch.setattr(
        prewarm, "_entries", lambda: [("one", "one", 60, lambda _db: {}), ("two", "two", 60, lambda _db: {})]
    )
    monkeypatch.setattr(prewarm, "cached", lambda *a, **k: None, raising=False)

    prewarm.note_scheduled()
    scheduled = prewarm.prewarm_status()
    assert scheduled["enabled"] is True
    assert scheduled["state"] == "scheduled"

    prewarm.prewarm_control_room()

    status = prewarm.prewarm_status()
    assert status["enabled"] is True
    assert status["state"] == "complete"
    assert status["entries_warmed"] == 2
    assert status["entries_failed"] == 0
    assert status["last_entry"] == "two"
    assert status["seconds"] >= 0
    # A snapshot, not the live dict: callers must not be able to mutate it.
    status["state"] = "tampered"
    assert prewarm.prewarm_status()["state"] == "complete"


def test_status_records_failures_instead_of_raising(monkeypatch):
    from app.services import prewarm

    def boom(_db):
        raise RuntimeError("boom")

    monkeypatch.setattr(prewarm, "_STATUS", {"enabled": False, "state": "disabled"})
    monkeypatch.setattr(prewarm, "_ENTRY_GAP_S", 0)
    monkeypatch.setattr(prewarm, "_entries", lambda: [("broken", "broken", 60, boom)])

    prewarm.prewarm_control_room()

    status = prewarm.prewarm_status()
    assert status["state"] == "complete"
    assert status["entries_warmed"] == 0
    assert status["entries_failed"] == 1


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_prewarm_setting_parses_from_the_environment(value, expected):
    from app.config import Settings

    assert Settings(control_room_prewarm=value).control_room_prewarm is expected


def test_prewarm_defaults_to_on_in_production_and_off_elsewhere():
    """A cold-start fix must not depend on a manual dashboard toggle.

    The first cut of this feature defaulted to off everywhere and relied on
    `render.yaml`. Render does not push newly added env vars to an
    already-created service, so the sweep never ran on the live deployment: the
    whole cold-start fix was silently inert in production, verifiable only by
    hand-timing requests. Production is where the ~76 s cold wake makes ~75 s of
    off-request-path CPU worth paying, so it is the default there.

    A production ``Settings`` now has to satisfy the release's fail-closed
    validator, so the secret and database URL are supplied explicitly here. That
    is itself part of the contract being tested: a production configuration that
    cannot be built is not a configuration, and the pre-warm default is only
    meaningful for one that can.
    """
    from app.config import Settings

    production = {
        "environment": "production",
        "secret_key": "k" * 48,
        "database_url": "postgresql://user:pass@db.example.com:5432/aerocast_ncr",
        "enable_demo_user": False,
        "demo_hydrate_empty_db": False,
    }

    assert Settings(**production).prewarm_enabled is True
    assert Settings(environment="development").prewarm_enabled is False
    assert Settings(environment="test").prewarm_enabled is False
    # An explicit setting always wins, in either direction.
    assert Settings(**production, control_room_prewarm=False).prewarm_enabled is False
    assert Settings(environment="development", control_room_prewarm=True).prewarm_enabled is True


def test_prewarm_is_off_by_default_outside_production():
    """Local dev, pytest and CI must never pay for the sweep unasked."""
    from app.config import Settings

    assert Settings().prewarm_enabled is False
    assert Settings().control_room_prewarm is None
