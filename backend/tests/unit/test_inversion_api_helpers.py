"""Unit tests for the inversion API helper functions (SIH26082).

Covers the DB-row -> geometry path and the episode statistics over a history
window, using lightweight stand-ins for ORM rows so the shared seeded test
database is never mutated.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.api.inversion import (
    EPISODE_HISTORY_HOURS,
    _compute_inversion,
    _episode_stats,
    _geopotential_by_level,
    _temp_by_level,
)


class _Reading:
    """Minimal stand-in for a WeatherReading row."""

    def __init__(self, ts, **kw):
        self.timestamp = ts
        self.pbl_height = kw.pop("pbl_height", None)
        for name in ("temperature_1000hPa", "temperature_925hPa", "temperature_850hPa",
                     "temperature_700hPa", "geopotential_height_925hPa",
                     "geopotential_height_850hPa"):
            setattr(self, name, kw.get(name))


INVERTED_PROFILE = {
    "temperature_1000hPa": 12.0,
    "temperature_925hPa": 10.0,
    "temperature_850hPa": 14.0,
    "temperature_700hPa": 8.0,
}


class TestLevelExtraction:
    def test_temperatures_pulled(self):
        reading = _Reading(datetime.now(UTC), **INVERTED_PROFILE)
        assert _temp_by_level(reading) == {1000: 12.0, 925: 10.0, 850: 14.0, 700: 8.0}

    def test_missing_temperatures_are_omitted(self):
        assert _temp_by_level(_Reading(datetime.now(UTC), temperature_925hPa=10.0)) == {925: 10.0}

    def test_none_levels_omitted(self):
        assert _temp_by_level(_Reading(datetime.now(UTC))) == {}

    def test_geopotential_pulled(self):
        reading = _Reading(datetime.now(UTC), geopotential_height_925hPa=810.0,
                           geopotential_height_850hPa=1500.0)
        assert _geopotential_by_level(reading) == {925: 810.0, 850: 1500.0}

    def test_geopotential_absent(self):
        assert _geopotential_by_level(_Reading(datetime.now(UTC))) == {}


class TestComputeInversion:
    def test_profile_uses_lapse_rate_and_geometry(self):
        reading = _Reading(datetime.now(UTC), pbl_height=600.0, **INVERTED_PROFILE)
        out = _compute_inversion(reading)
        assert out["inversion_source"] == "lapse_rate"
        assert out["inversion_detected"] is True
        assert out["inversion_category"] == "strong"
        assert out["inversion_base_pressure"] == 925
        assert out["inversion_top_pressure"] == 850
        assert out["inversion_thickness_hpa"] == 75.0
        assert out["inversion_thickness_m"] > 0
        assert out["inversion_base_height_m"] is not None

    def test_geopotential_heights_used_when_present(self):
        reading = _Reading(
            datetime.now(UTC), pbl_height=600.0,
            geopotential_height_925hPa=800.0,
            geopotential_height_850hPa=1500.0,
            **INVERTED_PROFILE,
        )
        out = _compute_inversion(reading)
        assert out["inversion_base_height_m"] == pytest.approx(800.0, abs=0.01)
        assert out["inversion_thickness_m"] == pytest.approx(700.0, abs=0.01)

    def test_single_level_falls_back_to_proxy(self):
        reading = _Reading(datetime.now(UTC), pbl_height=200.0, temperature_925hPa=10.0)
        out = _compute_inversion(reading)
        assert out["inversion_source"] == "pbl_proxy"
        assert out["inversion_thickness_m"] is None

    def test_pbl_labels_preserved_for_proxy(self):
        reading = _Reading(datetime.now(UTC), pbl_height=180.0)
        out = _compute_inversion(reading)
        assert out["inversion_strength"] == "Moderate"
        assert out["trapping_risk"] == "MEDIUM"
        assert out["inversion_detected"] is True

    def test_lapse_labels_use_lapse_risk_scale(self):
        reading = _Reading(datetime.now(UTC), pbl_height=180.0, **INVERTED_PROFILE)
        out = _compute_inversion(reading)
        assert out["inversion_strength"] == "Strong"
        assert out["trapping_risk"] == "HIGH"

    def test_dispersion_condition_present(self):
        out = _compute_inversion(_Reading(datetime.now(UTC), pbl_height=180.0, **INVERTED_PROFILE))
        assert out["dispersion_condition"] in {"TRAPPED", "LIMITED", "MODERATE", "GOOD", "UNKNOWN"}


class TestEpisodeStats:
    def _history(self, n, profile=None, start="2026-09-26T00:00:00Z"):
        base = datetime.fromisoformat(start).replace(tzinfo=UTC)
        kw = profile if profile is not None else {"pbl_height": 180.0}
        return [_Reading(base - timedelta(hours=n - 1 - i), **kw) for i in range(n)]

    def test_no_readings(self):
        stats = _episode_stats([])
        assert stats["samples"] == 0
        assert stats["current_duration_h"] == 0.0
        assert stats["persistence_fraction"] is None

    def test_persistent_inversion_duration(self):
        stats = _episode_stats(self._history(12))
        assert stats["samples"] == 12
        assert stats["current_duration_h"] == pytest.approx(12.0, abs=0.01)
        assert stats["persistence_fraction"] == pytest.approx(1.0)
        assert stats["measured_window_h"] == pytest.approx(11.0, abs=0.01)
        assert stats["sufficient_history"] is False

    def test_history_sufficient_past_window(self):
        stats = _episode_stats(self._history(30))
        assert stats["sufficient_history"] is True
        assert stats["complete"] is True

    def test_clearing_episode(self):
        # Oldest 12 hours inverted, newest 6 clear: no current episode.
        base = datetime.fromisoformat("2026-09-26T00:00:00Z").replace(tzinfo=UTC)
        inverted = [_Reading(base - timedelta(hours=i), pbl_height=180.0) for i in range(12, 0, -1)]
        clear = [_Reading(base - timedelta(hours=i), pbl_height=900.0) for i in range(6)]
        stats = _episode_stats(sorted(inverted + clear, key=lambda r: r.timestamp))
        assert stats["current_duration_h"] == 0.0
        assert stats["onset"] is None
        assert 0.0 < stats["persistence_fraction"] < 1.0

    def test_stats_use_lapse_rate_when_available(self):
        stats = _episode_stats(self._history(6, profile=INVERTED_PROFILE))
        assert stats["persistence_fraction"] == pytest.approx(1.0)
        assert stats["current_duration_h"] > 0

    def test_history_bound_is_bounded(self):
        # The endpoint query is capped, so a multi-year archive cannot be walked
        # in a single request.
        assert EPISODE_HISTORY_HOURS == 48
        assert EPISODE_HISTORY_HOURS >= 24
