"""Provenance of the dispersion forecast's driving meteorology.

``_aggregate_weather`` computes ``wind_observed`` / ``pbl_observed`` /
``synthetic_meteorology`` specifically so a placeholder can be told apart from an
observation, and the comment there says so. Those flags were then dropped when the
two response dicts were assembled: the API returned only the four numeric fields,
so a client had no way to tell a synthesised 4 m/s wind from a measured one, and
the page rendered the placeholder as if it were a reading.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from app.services.dispersion_service import _latest_weather, _wx_payload


def _reading(**kw):
    base = {
        "wind_speed": None,
        "wind_direction": None,
        "pbl_height": None,
        "precipitation": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeQuery:
    """Minimal stand-in for a SQLAlchemy query over Station / WeatherReading."""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, weather_by_station):
        self._weather_by_station = weather_by_station

    def query(self, model):
        name = getattr(model, "__name__", "")
        if name == "Station":
            return _FakeQuery([SimpleNamespace(id=i) for i in self._weather_by_station])
        return _FakeQuery(self._weather_by_station.get("rows", []))


def _aggregate(rows):
    """Run `_latest_weather` over a fake session holding `rows`."""
    session = _FakeSession({"rows": rows})
    return _latest_weather(session)


class TestAggregateWeatherFlags:
    def test_no_observations_is_flagged_synthetic(self):
        wx = _aggregate([])
        assert wx["wind_observed"] is False
        assert wx["pbl_observed"] is False
        assert wx["synthetic_meteorology"] is True
        # Placeholder values are still returned so the solver can run.
        assert wx["wind_speed"] == pytest.approx(4.0)
        assert wx["pbl_height"] == pytest.approx(600.0)

    def test_observed_values_are_not_synthetic(self):
        wx = _aggregate(
            [_reading(wind_speed=3.0, wind_direction=200.0, pbl_height=900.0, precipitation=0.0)]
        )
        assert wx["wind_observed"] is True
        assert wx["pbl_observed"] is True
        assert wx["synthetic_meteorology"] is False
        assert wx["wind_speed"] == pytest.approx(3.0)
        assert wx["pbl_height"] == pytest.approx(900.0)

    def test_calm_zero_wind_counts_as_observed(self):
        """A genuinely calm network is a measurement, not missing data."""
        wx = _aggregate(
            [_reading(wind_speed=0.0, wind_direction=0.0, pbl_height=800.0, precipitation=0.0)]
        )
        assert wx["wind_observed"] is True
        assert wx["wind_speed"] == pytest.approx(0.0)
        assert wx["synthetic_meteorology"] is False

    def test_wind_without_pbl_is_still_flagged_synthetic(self):
        wx = _aggregate([_reading(wind_speed=5.0, wind_direction=90.0)])
        assert wx["wind_observed"] is True
        assert wx["pbl_observed"] is False
        assert wx["synthetic_meteorology"] is True


class TestWxPayloadSerialisesFlags:
    """The response contract: flags must survive into the API payload."""

    def test_preserves_rounded_values_and_flags(self):
        wx = {
            "wind_speed": 3.14159,
            "wind_direction": 200.04,
            "pbl_height": 900.123,
            "precipitation": 0.005,
            "wind_observed": True,
            "pbl_observed": True,
            "synthetic_meteorology": False,
        }
        payload = _wx_payload(wx)
        assert payload["wind_speed"] == 3.14
        assert payload["wind_direction"] == 200.0
        assert payload["pbl_height"] == 900.1
        assert payload["precipitation"] == 0.01
        assert payload["wind_observed"] is True
        assert payload["pbl_observed"] is True
        assert payload["synthetic_meteorology"] is False

    def test_synthetic_case_is_visible_to_the_client(self):
        wx = {
            "wind_speed": 4.0,
            "wind_direction": 90.0,
            "pbl_height": 600.0,
            "precipitation": 0.0,
            "wind_observed": False,
            "pbl_observed": False,
            "synthetic_meteorology": True,
        }
        payload = _wx_payload(wx)
        assert payload["synthetic_meteorology"] is True
        assert payload["wind_observed"] is False
        assert payload["pbl_observed"] is False

    def test_missing_flags_default_to_unobserved_not_to_observed(self):
        """An old caller without the keys must not be reported as measured."""
        payload = _wx_payload(
            {"wind_speed": 1.0, "wind_direction": 10.0, "pbl_height": 100.0, "precipitation": 0.0}
        )
        assert payload["wind_observed"] is False
        assert payload["pbl_observed"] is False
        assert payload["synthetic_meteorology"] is False

    def test_aggregate_output_round_trips_through_payload(self):
        observed = _aggregate(
            [_reading(wind_speed=6.0, wind_direction=270.0, pbl_height=1100.0, precipitation=0.0)]
        )
        assert _wx_payload(observed)["synthetic_meteorology"] is False

        missing = _aggregate([])
        assert _wx_payload(missing)["synthetic_meteorology"] is True


class TestPlaceholderDoesNotCorruptResults:
    def test_no_nan_reaches_the_payload(self):
        for readings in ([], [_reading(wind_speed=2.0)]):
            wx = _aggregate(readings)
            for key, value in _wx_payload(wx).items():
                if isinstance(value, float):
                    assert not np.isnan(value), key
