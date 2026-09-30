"""Provenance and freshness behaviour of ``GET /api/summary``.

Regression cover for the "NCR Average AQI — No data" defect. The summary used to
select each station's latest pollution reading inside a hard 24-hour window. When
the upstream CPCB/CKAN archive stopped publishing (newest record 2025-12-31), that
window matched nothing for every station, so ``ncr_avg_aqi`` went null and the
Overview card rendered "--" even though ~70k historical rows per station were
still on disk.

The fix averages each station's most recent *stored* reading regardless of age
and reports that reading's age so the client can label it. These tests pin that
behaviour, and pin the related honesty rule: a running refresh scheduler must not
be reported as "live" when the data it produces is outside the window.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.api import summary as summary_module
from app.models.db_models import FireReading, PollutionReading


def _age_all_readings(session, hours: float) -> None:
    """Backdate every pollution reading by `hours`, as a dead feed would leave them.

    `pollution_observations` is unique on (station_id, timestamp), so each row is
    offset by its own index to keep the shifted timestamps distinct.
    """
    anchor = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)
    for i, row in enumerate(session.query(PollutionReading).all()):
        row.timestamp = anchor - timedelta(minutes=i)
    session.commit()


def _summary(client):
    resp = client.get("/api/summary")
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture()
def live_refresh(monkeypatch):
    """Scheduler enabled, i.e. the configuration the hosted deployment runs."""
    monkeypatch.setattr(summary_module.settings, "live_refresh_enabled", True, raising=False)
    monkeypatch.setattr(summary_module.settings, "demo_hydrate_empty_db", False, raising=False)


@pytest.fixture()
def no_demo_hydration(monkeypatch):
    """Isolate the refresh flags from any ambient/demo configuration."""
    monkeypatch.setattr(summary_module.settings, "demo_hydrate_empty_db", False, raising=False)


class TestSummaryUsesNewestStoredReading:
    def test_aqi_is_reported_even_when_every_reading_is_stale(self, client, db_session, no_demo_hydration):
        """The core regression: months-old data must still produce an average."""
        _age_all_readings(db_session, hours=24 * 270)

        body = _summary(client)

        assert body["stations_with_readings"] >= 1
        assert body["ncr_avg_aqi"] is not None, "summary must average the newest stored readings even when they are old"
        assert body["worst_station"] is not None
        assert body["best_station"] is not None

    def test_observation_age_is_reported(self, client, db_session, no_demo_hydration):
        _age_all_readings(db_session, hours=24 * 270)

        body = _summary(client)

        assert body["observation_age_hours"] is not None
        # ~270 days back, allowing a generous margin for test runtime.
        assert body["observation_age_hours"] > 24 * 200
        assert body["latest_observation_at"] is not None

    def test_fresh_readings_report_a_small_age(self, client, db_session, no_demo_hydration):
        body = _summary(client)

        assert body["observation_age_hours"] is not None
        assert 0 <= body["observation_age_hours"] < 24


class TestSummaryDataMode:
    def test_stale_when_scheduler_enabled_but_feed_is_not_publishing(self, client, db_session, live_refresh):
        """A running scheduler is not evidence of current data."""
        _age_all_readings(db_session, hours=24 * 270)

        body = _summary(client)

        assert body["data_mode"] == "stale"
        assert "not" in body["data_mode_note"].lower()
        # The mode is only trustworthy because the AQI is still present.
        assert body["ncr_avg_aqi"] is not None

    def test_live_when_scheduler_enabled_and_data_is_current(self, client, db_session, live_refresh):
        body = _summary(client)

        assert body["data_mode"] == "live"
        assert body["observation_age_hours"] < 24

    def test_empty_when_no_observations_exist(self, client, db_session, live_refresh):
        db_session.query(PollutionReading).delete()
        db_session.commit()

        body = _summary(client)

        assert body["data_mode"] == "empty"
        assert body["ncr_avg_aqi"] is None
        assert body["observation_age_hours"] is None
        assert body["latest_observation_at"] is None

    def test_static_archive_when_refresh_disabled_and_data_old(self, client, db_session, monkeypatch):
        monkeypatch.setattr(summary_module.settings, "live_refresh_enabled", False, raising=False)
        monkeypatch.setattr(summary_module.settings, "demo_hydrate_empty_db", False, raising=False)
        _age_all_readings(db_session, hours=24 * 270)

        body = _summary(client)

        assert body["data_mode"] == "static_archive"
        assert body["ncr_avg_aqi"] is not None

    def test_demo_hydration_takes_precedence_over_live(self, client, db_session, monkeypatch):
        """Demo mode must not be reported as live, even with the scheduler on."""
        monkeypatch.setattr(summary_module.settings, "live_refresh_enabled", True, raising=False)
        monkeypatch.setattr(summary_module.settings, "demo_hydrate_empty_db", True, raising=False)

        body = _summary(client)

        assert body["data_mode"] == "demo_seeded"


class TestFiresKeepTheirOwnWindow:
    def test_fires_outside_24h_are_excluded(self, client, db_session, no_demo_hydration):
        """`active_fires_24h` is genuinely a 24 h metric and must not inherit the
        now-unwindowed pollution query."""
        old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=72)
        for row in db_session.query(FireReading).all():
            row.acq_date = old
        db_session.commit()

        body = _summary(client)

        assert body["active_fires_24h"] == 0
        # Polluting the fire timestamps must not disturb the AQI computation.
        assert body["ncr_avg_aqi"] is not None

    def test_fires_inside_24h_are_counted(self, client, db_session, no_demo_hydration):
        body = _summary(client)

        assert body["active_fires_24h"] >= 1
