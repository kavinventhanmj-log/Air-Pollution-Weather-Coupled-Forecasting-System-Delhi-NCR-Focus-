"""Unit tests for the atmospheric-context future-weather path (SIH26082 PS).

The context builder fetches an Open-Meteo *forecast* once per build and matches
each horizon target (``now + h``) to the nearest forecast hour, falling back to
stored ``weather_observations`` only when the provider is unavailable. These
tests pin that behaviour with a mocked provider so the suite stays hermetic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest import mock

import pandas as pd
import pytest
from app.models.db_models import Station, WeatherReading
from app.services import coupling_service, refresh_service
from app.services.coupling_service import _nearest_weather_row, _utc_naive

HORIZON_KEYS = {
    "horizon_hours",
    "target_timestamp",
    "weather_match_timestamp",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_mps",
    "wind_direction_deg",
    "pbl_height_m",
    "inversion_detected",
    "inversion_strength",
    "inversion_category",
    "inversion_source",
    "dispersion_potential",
    "accumulation_potential",
    "inversion_trapping_potential",
    "pollution_stagnation_index",
    "fire_transport_influence",
    "regional_transport_potential",
    "ozone_photochemical_potential",
    "meteorology_pollution_interaction",
}


def _station(db_session) -> Station:
    return db_session.query(Station).filter(Station.name == "Anand Vihar").first()


def _make_forecast_frame() -> pd.DataFrame:
    """72 hourly forecast rows starting at the top of the next hour.

    The temperature / PBL signatures (18..36 degC, 405..760 m) deliberately sit
    outside the stored seed weather (22.4..22.5 degC, 180 m) so a test can prove
    which source won the match.
    """
    now = datetime.now(UTC).replace(tzinfo=None)
    base = now.replace(minute=0, second=0, microsecond=0)
    rows = []
    for h in range(1, 73):
        rows.append(
            {
                "time": base + timedelta(hours=h),
                "temperature_2m": round(18.0 + h * 0.25, 2),
                "relative_humidity_2m": 61.0,
                "pressure_msl": 1014.0,
                "surface_pressure": 994.0,
                "wind_speed_10m": 4.0,
                "wind_direction_10m": 280.0,
                "precipitation": 0.0,
                "cloud_cover": 42.0,
                "boundary_layer_height": 400.0 + h * 5.0,
                "temperature_1000hPa": 19.0,
                "temperature_925hPa": 16.0,
                "temperature_850hPa": 11.0,
                "temperature_700hPa": 4.0,
            }
        )
    return pd.DataFrame(rows)


def _stored_stamps(db_session, station) -> set[datetime]:
    return {
        _utc_naive(w.timestamp)
        for w in db_session.query(WeatherReading)
        .filter(WeatherReading.station_id == station.id)
        .all()
    }


def _naive(iso: str) -> datetime:
    return datetime.fromisoformat(iso[:-1])


def _provider_frame(start: datetime, hours: int) -> pd.DataFrame:
    """A provider-shaped hourly frame starting at *start*."""
    return pd.DataFrame(
        [
            {
                "time": start + timedelta(hours=h),
                "temperature_2m": 20.0 + h * 0.1,
                "relative_humidity_2m": 60.0,
                "pressure_msl": 1012.0,
                "surface_pressure": 992.0,
                "wind_speed_10m": 3.0,
                "wind_direction_10m": 270.0,
                "precipitation": 0.0,
                "cloud_cover": 40.0,
                "boundary_layer_height": 500.0,
                "temperature_1000hPa": 19.0,
                "temperature_925hPa": 16.0,
                "temperature_850hPa": 11.0,
                "temperature_700hPa": 4.0,
            }
            for h in range(hours)
        ]
    )


def test_future_targets_use_forecast_weather_aligned_to_target(db_session):
    station = _station(db_session)
    frame = _make_forecast_frame()
    val_by_time = {
        pd.Timestamp(r["time"]).to_pydatetime(): r["temperature_2m"]
        for _, r in frame.iterrows()
    }
    now = datetime.now(UTC).replace(tzinfo=None)
    with mock.patch.object(coupling_service, "fetch_forecast_hours", return_value=frame):
        body = coupling_service.get_forecast_context(db_session, station)

    assert len(body["horizons"]) == 72
    for h, horizon in enumerate(body["horizons"], start=1):
        matched = _naive(horizon["weather_match_timestamp"])
        target = _naive(horizon["target_timestamp"])
        assert matched in val_by_time, f"h{h} matched a non-forecast hour"
        # future forecast, never a past stored row
        assert matched >= now
        # aligned: matched hour within ~1h of the horizon target
        assert abs((matched - target).total_seconds()) / 3600 <= 1
        # value comes straight from the forecast row for that hour
        assert horizon["temperature_c"] == pytest.approx(val_by_time[matched])


def test_future_forecast_is_not_presented_as_stored_history(db_session):
    station = _station(db_session)
    stored = _stored_stamps(db_session, station)
    assert stored, "seed should provide stored observations"
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=_make_forecast_frame()
    ):
        body = coupling_service.get_forecast_context(db_session, station)
    for horizon in body["horizons"]:
        matched = _naive(horizon["weather_match_timestamp"])
        assert matched not in stored, "forecast hour leaked into historical matching"


def test_stored_observations_used_when_forecast_unavailable(db_session):
    station = _station(db_session)
    stored = _stored_stamps(db_session, station)
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=pd.DataFrame()
    ):
        body = coupling_service.get_forecast_context(db_session, station)

    assert len(body["horizons"]) == 72
    h1 = body["horizons"][0]
    # early horizons still align to real stored hours (seed temp ~22.5 degC)
    assert _naive(h1["weather_match_timestamp"]) in stored
    assert h1["temperature_c"] == pytest.approx(22.5, abs=1.0)
    # beyond stored coverage the fields are None, never invented
    h72 = body["horizons"][-1]
    assert h72["temperature_c"] is None
    assert h72["pbl_height_m"] is None


def test_complete_72h_context_from_forecast(db_session):
    station = _station(db_session)
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=_make_forecast_frame()
    ):
        body = coupling_service.get_forecast_context(db_session, station)

    horizons = body["horizons"]
    assert [h["horizon_hours"] for h in horizons] == list(range(1, 73))
    for horizon in horizons:
        assert horizon["temperature_c"] is not None
        assert horizon["humidity_pct"] is not None
        assert horizon["pressure_hpa"] is not None
        assert horizon["wind_speed_mps"] is not None
        assert horizon["pbl_height_m"] is not None


def test_provider_called_once_per_context_build(db_session):
    station = _station(db_session)
    frame = _make_forecast_frame()
    calls = 0

    def _provider(lat, lon):
        nonlocal calls
        calls += 1
        return frame

    with mock.patch.object(coupling_service, "fetch_forecast_hours", side_effect=_provider):
        body = coupling_service.get_forecast_context(db_session, station)

    assert calls == 1, "one forecast fetch must cover all 72 horizons"
    assert len(body["horizons"]) == 72


@pytest.mark.parametrize("fail_as_error", [False, True])
def test_missing_provider_data_handled_safely(db_session, fail_as_error):
    station = _station(db_session)
    if fail_as_error:
        patcher = mock.patch.object(
            coupling_service,
            "fetch_forecast_hours",
            side_effect=RuntimeError("provider down"),
        )
    else:
        patcher = mock.patch.object(
            coupling_service, "fetch_forecast_hours", return_value=pd.DataFrame()
        )
    with patcher:
        body = coupling_service.get_forecast_context(db_session, station)

    assert len(body["horizons"]) == 72
    assert body["horizons"][0]["temperature_c"] is not None  # stored fallback
    assert body["horizons"][-1]["temperature_c"] is None


def test_forecast_never_written_to_observations(db_session):
    station = _station(db_session)
    before = db_session.query(WeatherReading).count()
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=_make_forecast_frame()
    ):
        coupling_service.get_forecast_context(db_session, station)
    assert db_session.query(WeatherReading).count() == before


def test_coupling_fields_computed_for_future_horizons(db_session):
    station = _station(db_session)
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=_make_forecast_frame()
    ):
        body = coupling_service.get_forecast_context(db_session, station)

    coupling_keys = {
        "dispersion_potential",
        "accumulation_potential",
        "inversion_trapping_potential",
        "pollution_stagnation_index",
        "ozone_photochemical_potential",
        "fire_transport_influence",
        "regional_transport_potential",
        "meteorology_pollution_interaction",
    }
    for horizon in body["horizons"]:
        for key in coupling_keys:
            val = horizon[key]
            assert val is None or 0.0 <= val <= 1.0, f"h{horizon['horizon_hours']} {key} OOR"
        # core dispersion/accumulation/stagnation must be computed with full weather
        assert horizon["dispersion_potential"] is not None
        assert horizon["accumulation_potential"] is not None
        assert horizon["pollution_stagnation_index"] is not None


def test_response_and_horizon_contract_unchanged(db_session):
    station = _station(db_session)
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=pd.DataFrame()
    ):
        body = coupling_service.get_forecast_context(db_session, station)

    assert set(body) == {"station", "generated_at", "units", "regional_note", "horizons"}
    assert body["station"] == station.name
    assert set(body["units"]) == {
        "wind_speed_mps",
        "pbl_height_m",
        "temperature_c",
        "pressure_hpa",
        "features",
    }
    assert body["horizons"][0]["horizon_hours"] == 1
    assert body["horizons"][-1]["horizon_hours"] == 72
    for horizon in body["horizons"]:
        assert set(horizon) == HORIZON_KEYS


# --------------------------------------------------------------------------- #
# Provider fetch: request shape, coverage and failure signalling
# --------------------------------------------------------------------------- #


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _patched_provider(payload):
    """Run the real ``fetch_forecast_hours`` with the pytest network guard off."""
    return (
        mock.patch.object(refresh_service, "_running_under_pytest", return_value=False),
        mock.patch.object(refresh_service.requests, "get", return_value=_Resp(payload)),
    )


def test_forecast_request_asks_for_five_forecast_days():
    now = datetime.now(UTC).replace(tzinfo=None)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    payload = {"hourly": _provider_frame(start, 24 * 5).to_dict(orient="list")}
    guard, get = _patched_provider(payload)
    with guard, get as mocked_get:
        refresh_service.fetch_forecast_hours(28.6469, 77.3152)

    params = mocked_get.call_args.kwargs["params"]
    assert params["forecast_days"] == 5
    assert refresh_service.FORECAST_COVERAGE_DAYS == 5
    # A pure forward-looking frame: no archive-style window, no past data, so
    # the rows can never be mistaken for stored observations.
    assert "start_date" not in params
    assert "end_date" not in params
    assert "past_days" not in params
    assert params["timezone"] == "UTC"
    assert params["latitude"] == 28.6469 and params["longitude"] == 77.3152
    # Surface + pressure-level variables, matching the stored WeatherReading shape.
    hourly = params["hourly"]
    assert "temperature_2m" in hourly and "boundary_layer_height" in hourly
    assert "temperature_925hPa" in hourly and "temperature_700hPa" in hourly


def test_forecast_frame_times_are_naive_utc():
    now = datetime.now(UTC).replace(tzinfo=None)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    payload = {"hourly": _provider_frame(start, 5).to_dict(orient="list")}
    guard, get = _patched_provider(payload)
    with guard, get:
        df = refresh_service.fetch_forecast_hours(28.6, 77.3)

    assert len(df) == 5
    assert df["time"].dt.tz is None, "timestamps must be naive UTC to compare with targets"
    assert df["time"].iloc[0] == pd.Timestamp(start)


def test_forecast_request_raises_on_http_error_so_caller_falls_back(db_session):
    guard = mock.patch.object(refresh_service, "_running_under_pytest", return_value=False)
    with guard, mock.patch.object(
        refresh_service.requests, "get", side_effect=RuntimeError("502 from provider")
    ):
        with pytest.raises(RuntimeError):
            refresh_service.fetch_forecast_hours(28.6, 77.3)

    # ...and the context builder swallows it, serving stored observations instead.
    station = _station(db_session)
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", side_effect=RuntimeError("502")
    ):
        body = coupling_service.get_forecast_context(db_session, station)
    assert len(body["horizons"]) == 72
    assert body["horizons"][0]["temperature_c"] is not None


def test_forecast_request_returns_empty_frame_when_provider_has_no_hourly_block():
    guard, get = _patched_provider({})
    with guard, get:
        assert refresh_service.fetch_forecast_hours(28.6, 77.3).empty


def test_five_day_window_covers_every_horizon_from_midnight_start(db_session):
    """5 days measured from 00:00 today must span ``now + 72h`` at any hour."""
    now = datetime.now(UTC).replace(tzinfo=None)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    frame = _provider_frame(start, 24 * refresh_service.FORECAST_COVERAGE_DAYS)
    assert frame["time"].max() >= now + timedelta(hours=72)

    station = _station(db_session)
    with mock.patch.object(coupling_service, "fetch_forecast_hours", return_value=frame):
        body = coupling_service.get_forecast_context(db_session, station)

    for horizon in body["horizons"]:
        assert horizon["weather_match_timestamp"] is not None
        assert horizon["temperature_c"] is not None


# --------------------------------------------------------------------------- #
# Nearest-weather tolerance must actually be enforced
# --------------------------------------------------------------------------- #


class _Row:
    def __init__(self, ts):
        self.timestamp = ts


def test_row_inside_ceiling_but_outside_tolerance_is_rejected():
    """The bug: ``and`` let a 3h-old row satisfy a 2h tolerance via the 6h ceiling."""
    target = datetime(2026, 9, 12, 12, 0)
    rows = [_Row(target - timedelta(hours=3)), _Row(target - timedelta(hours=9))]

    assert _nearest_weather_row(rows, target, tolerance=timedelta(hours=2)) is None


def test_tighter_tolerance_of_the_two_bounds_wins():
    target = datetime(2026, 9, 12, 12, 0)

    # 1h30m away: inside both the 2h tolerance and the 6h ceiling -> kept.
    near = [_Row(target - timedelta(minutes=90))]
    assert _nearest_weather_row(near, target, tolerance=timedelta(hours=2)) is not None

    # Same 1h30m row against a 1h tolerance -> rejected, tolerance is honoured.
    assert _nearest_weather_row(near, target, tolerance=timedelta(hours=1)) is None

    # A ceiling tighter than the tolerance is honoured the other way round too.
    far = [_Row(target - timedelta(hours=4))]
    assert _nearest_weather_row(
        far, target, tolerance=timedelta(hours=6), within_max_hours=3
    ) is None


def test_horizon_beyond_gap_is_null_rather_than_pulled_from_far_away(db_session):
    """With no forecast and a 12h stored archive, only h1 stays within tolerance."""
    station = _station(db_session)
    with mock.patch.object(
        coupling_service, "fetch_forecast_hours", return_value=pd.DataFrame()
    ):
        body = coupling_service.get_forecast_context(db_session, station)

    horizons = body["horizons"]
    assert horizons[0]["temperature_c"] is not None
    # The newest stored row is the top of the current hour, so h2's target is
    # >2h from every stored observation: it must be null, not a 2h-stale match.
    assert horizons[1]["temperature_c"] is None
    assert horizons[1]["weather_match_timestamp"] is None

