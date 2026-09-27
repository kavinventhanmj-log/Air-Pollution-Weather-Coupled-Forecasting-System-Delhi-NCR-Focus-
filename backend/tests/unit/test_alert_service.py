"""Unit tests for the alert generation engine (SIH26082 Phase 38 #16).

Deterministic inputs exercise every alert trigger in
``backend/app/services/alert_service.py``: AQI thresholds, trend direction,
PM2.5 dominance, low/high wind, inversion/PBL depth, humidity, precipitation
scavenging, regional fires, plus combined severity ordering and resilience
to missing/None inputs. No randomness, no DB, no HTTP — the service is pure.
"""

from backend.app.services.alert_service import (
    ALERT_LEVEL_RANK,
    generate_alerts,
)

BASE_FORECAST = {"aqi_pred": 100, "dominant_pollutant": "pm25", "trend": "stable", "horizon_hours": 24}
BASE_WEATHER = {"wind_speed": 5.0, "pbl_height": 600.0, "humidity": 50.0, "precipitation": 3.0}
BASE_FIRE = {"fire_count": 0, "distance_nearest_fire": 400.0}


def _generate(forecast=None, weather=None, fire=None):
    return generate_alerts(
        forecast if forecast is not None else BASE_FORECAST,
        weather if weather is not None else BASE_WEATHER,
        fire if fire is not None else BASE_FIRE,
    )


def _titles(alerts):
    return [a["title"] for a in alerts]


def _levels(alerts):
    return [a["alert_level"] for a in alerts]


# --- AQI thresholds ---------------------------------------------------------
def test_low_aqi_yields_no_alerts():
    assert _generate(forecast={"aqi_pred": 120, "dominant_pollutant": "pm10", "trend": "stable"}) == []


def test_aqi_201_triggers_advisory():
    alerts = _generate(forecast={"aqi_pred": 250, "dominant_pollutant": "o3", "trend": "stable"})
    assert _levels(alerts) == ["ADVISORY"]
    assert "Poor AQI Advisory" in _titles(alerts)


def test_aqi_301_triggers_warning():
    alerts = _generate(forecast={"aqi_pred": 350, "dominant_pollutant": "no2", "trend": "stable"})
    assert _levels(alerts) == ["WARNING"]
    assert "Very Poor AQI Warning" in _titles(alerts)


def test_aqi_401_triggers_severe():
    alerts = _generate(forecast={"aqi_pred": 450, "dominant_pollutant": "o3", "trend": "stable"})
    assert _levels(alerts) == ["SEVERE"]
    assert "Severe Pollution Alert" in _titles(alerts)


def test_high_aqi_fires_exactly_one_threshold_alert():
    alerts = _generate(forecast={"aqi_pred": 450, "dominant_pollutant": "pm25", "trend": "stable"})
    threshold = [t for t in _titles(alerts) if "AQI" in t or "Pollution Alert" in t]
    assert len(threshold) == 1


def test_aqi_boundary_just_below_threshold_is_silent():
    assert _generate(forecast={"aqi_pred": 200, "trend": "stable"}) == []


# --- Trend direction --------------------------------------------------------
def test_rising_trend_watch():
    alerts = _generate(forecast={"aqi_pred": 100, "trend": "rising"})
    assert "Rising Pollution Trend" in _titles(alerts)
    assert all(level == "WATCH" for level in _levels(alerts))


def test_falling_trend_watch():
    alerts = _generate(forecast={"aqi_pred": 100, "trend": "falling"})
    assert "Improving Air Quality" in _titles(alerts)


def test_stable_trend_adds_no_trend_alert():
    alerts = _generate(forecast={"aqi_pred": 100, "trend": "stable"})
    assert not any("Trend" in t or "Improving" in t for t in _titles(alerts))


# --- PM2.5 dominance --------------------------------------------------------
def test_pm25_dominance_with_high_aqi_adds_watch():
    alerts = _generate(forecast={"aqi_pred": 350, "dominant_pollutant": "pm25", "trend": "stable"})
    assert "PM2.5 Dominated Pollution" in _titles(alerts)
    assert "WARNING" in _levels(alerts) and "WATCH" in _levels(alerts)


def test_pm25_dominance_needs_high_aqi():
    alerts = _generate(forecast={"aqi_pred": 100, "dominant_pollutant": "pm25", "trend": "stable"})
    assert "PM2.5 Dominated Pollution" not in _titles(alerts)


def test_non_pm25_dominance_skips_pm25_watch():
    alerts = _generate(forecast={"aqi_pred": 300, "dominant_pollutant": "o3", "trend": "stable"})
    assert "PM2.5 Dominated Pollution" not in _titles(alerts)


# --- Wind -------------------------------------------------------------------
def test_low_wind_watch():
    alerts = _generate(weather={**BASE_WEATHER, "wind_speed": 1.0})
    assert "Low Wind Speed Watch" in _titles(alerts)
    assert all(level == "WATCH" for level in _levels(alerts))


def test_high_wind_dust_watch():
    alerts = _generate(weather={**BASE_WEATHER, "wind_speed": 18.0})
    assert "High Wind / Dust Watch" in _titles(alerts)


def test_moderate_wind_no_wind_alert():
    alerts = _generate(weather={**BASE_WEATHER, "wind_speed": 5.0})
    assert not any("Wind" in t for t in _titles(alerts))


# --- Inversion / PBL --------------------------------------------------------
def test_strong_inversion_warning():
    alerts = _generate(weather={**BASE_WEATHER, "pbl_height": 100.0})
    assert "Strong Inversion Trapping" in _titles(alerts)
    assert "WARNING" in _levels(alerts)


def test_weak_inversion_watch():
    alerts = _generate(weather={**BASE_WEATHER, "pbl_height": 250.0})
    assert "Weak Inversion / Low PBL" in _titles(alerts)
    assert all(level == "WATCH" for level in _levels(alerts))


def test_deep_pbl_no_inversion_alert():
    alerts = _generate(weather={**BASE_WEATHER, "pbl_height": 1200.0})
    assert not any("Inversion" in t or "PBL" in t for t in _titles(alerts))


# --- Humidity / precipitation ----------------------------------------------
def test_high_humidity_advisory():
    alerts = _generate(weather={**BASE_WEATHER, "humidity": 90.0})
    assert "High Humidity / Secondary Aerosol" in _titles(alerts)
    assert _levels(alerts) == ["ADVISORY"]


def test_dry_conditions_no_humidity_alert():
    alerts = _generate(weather={**BASE_WEATHER, "humidity": 45.0})
    assert not any("Humidity" in t for t in _titles(alerts))


def test_no_rain_with_high_aqi_watch():
    alerts = _generate(
        forecast={"aqi_pred": 250, "trend": "stable"},
        weather={**BASE_WEATHER, "precipitation": 0.0},
    )
    assert "No Rain Scavenging" in _titles(alerts)


def test_rain_scavenging_suppresses_watch():
    alerts = _generate(
        forecast={"aqi_pred": 250, "trend": "stable"},
        weather={**BASE_WEATHER, "precipitation": 3.0},
    )
    assert "No Rain Scavenging" not in _titles(alerts)


# --- Regional fires ---------------------------------------------------------
def test_approaching_smoke_plume_warning():
    alerts = _generate(fire={"fire_count": 80, "distance_nearest_fire": 200.0})
    assert "Approaching Smoke Plume" in _titles(alerts)
    assert "WARNING" in _levels(alerts)


def test_elevated_burning_watch():
    alerts = _generate(fire={"fire_count": 30, "distance_nearest_fire": 400.0})
    assert "Elevated Regional Burning" in _titles(alerts)
    assert all(level == "WATCH" for level in _levels(alerts))


def test_many_fires_with_unknown_distance_still_watch():
    alerts = _generate(fire={"fire_count": 80, "distance_nearest_fire": None})
    assert "Elevated Regional Burning" in _titles(alerts)
    assert "Approaching Smoke Plume" not in _titles(alerts)


def test_few_fires_no_fire_alert():
    assert "Elevated Regional Burning" not in _titles(_generate(fire={"fire_count": 5}))


# --- Ordering, keys, resilience --------------------------------------------
def test_combined_alerts_sorted_by_severity():
    alerts = _generate(
        forecast={"aqi_pred": 450, "dominant_pollutant": "pm25", "trend": "rising"},
        weather={**BASE_WEATHER, "wind_speed": 1.0, "pbl_height": 100.0},
        fire={"fire_count": 80, "distance_nearest_fire": 200.0},
    )
    ranks = [ALERT_LEVEL_RANK[level] for level in _levels(alerts)]
    assert ranks == sorted(ranks, reverse=True)
    assert _levels(alerts)[0] == "SEVERE"


def test_every_alert_has_required_keys():
    alerts = _generate(
        forecast={"aqi_pred": 450, "dominant_pollutant": "pm25", "trend": "rising"},
        weather={**BASE_WEATHER, "wind_speed": 1.0, "pbl_height": 100.0, "humidity": 95.0},
        fire={"fire_count": 80, "distance_nearest_fire": 200.0},
    )
    for alert in alerts:
        assert set(alert) >= {"alert_level", "title", "description", "factors", "recommendation"}
        assert alert["alert_level"] in ALERT_LEVEL_RANK


def test_all_none_inputs_yield_no_alerts():
    assert generate_alerts({}, {}, {}) == []
    assert generate_alerts(None, None, None) == []  # type: ignore[arg-type]


def test_missing_weather_and_fire_keys_do_not_crash():
    alerts = generate_alerts({"aqi_pred": 350}, {}, {})
    assert _levels(alerts) == ["WARNING"]


def test_all_station_alerts_filter_is_a_subset_of_the_full_sweep():
    """``all_station_alerts`` caches the unfiltered 17-station sweep and applies
    the station name afterwards. That refactor moved filtering out of the SQL
    query, so lock in the invariant: a filtered call must be exactly the subset
    of the full call for that station, not an independently computed answer.
    """
    from backend.app.database import SessionLocal
    from backend.app.services.alert_service import all_station_alerts

    with SessionLocal() as db:
        every = all_station_alerts(db)
        if not every:
            return  # empty test database: nothing to relate a subset to
        for station in {row["station"] for row in every}:
            filtered = all_station_alerts(db, station)
            assert filtered, f"expected alerts for {station}"
            assert {row["station"] for row in filtered} == {station}
            assert filtered == [row for row in every if row["station"] == station]


def test_all_station_alerts_unknown_station_is_empty():
    from backend.app.database import SessionLocal
    from backend.app.services.alert_service import all_station_alerts

    with SessionLocal() as db:
        assert all_station_alerts(db, "Definitely Not A Station") == []


def test_all_station_alerts_never_mutates_the_cached_sweep():
    """The endpoint hands the cached list straight to callers, so a filter must
    not be able to corrupt it for the next request."""
    from backend.app.database import SessionLocal
    from backend.app.services.alert_service import all_station_alerts

    with SessionLocal() as db:
        every = all_station_alerts(db)
        if not every:
            return
        before = list(every)
        all_station_alerts(db, every[0]["station"]).clear()
        assert all_station_alerts(db) == before


# --- Observation fallback: AQI 0 is a score, not missing data -----------------
# When a station has no forecast run, ``_forecast_inputs`` scores its latest
# observation with ``calculate_aqi``. The guard used to be ``if not aqi``,
# which silently discarded a legitimate AQI of 0 because 0 is falsy in Python.
# AQI 0 is the floor of the CPCB "Good" band and is a real measurement, so the
# guard now asks the scorer whether the reading was scorable instead. The alert
# thresholds are untouched (the first one starts at 201), so a carried-through
# 0 still raises no alert -- the point is that the station is now evaluated on
# its real score rather than being dropped as though it had no data at all.


def _observation_only_station(db, name="RK Puram", **concentrations):
    """Return a seeded station carrying exactly one observation and no forecast,
    so ``_forecast_inputs`` is forced down the observation-fallback branch."""
    from datetime import UTC, datetime

    from backend.app.models.db_models import Forecast, PollutionReading, Station

    station = db.query(Station).filter(Station.name == name).first()
    assert station is not None, f"expected seeded station {name!r}"
    db.query(Forecast).filter(Forecast.station_id == station.id).delete()
    db.query(PollutionReading).filter(PollutionReading.station_id == station.id).delete()
    db.add(
        PollutionReading(
            station_id=station.id,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            aqi=500,  # stale legacy sentinel: the fallback must ignore this
            **concentrations,
        )
    )
    db.commit()
    return station


def test_forecast_inputs_carries_through_a_legitimate_aqi_of_zero(db_session):
    from backend.app.services.alert_service import _forecast_inputs

    station = _observation_only_station(
        db_session, pm25=0.0, pm10=0.0, o3=0.0, no2=0.0, so2=0.0, co=0.0
    )

    inputs = _forecast_inputs(db_session, station.id)

    assert inputs is not None, "AQI 0 is a real score and must not be treated as missing"
    assert inputs["aqi_pred"] == 0


def test_forecast_inputs_still_rejects_a_fully_unscorable_observation(db_session):
    from backend.app.services.alert_service import _forecast_inputs

    station = _observation_only_station(
        db_session, pm25=None, pm10=None, o3=None, no2=None, so2=None, co=None
    )

    assert _forecast_inputs(db_session, station.id) is None


def test_forecast_inputs_still_rejects_a_non_scored_pollutant_only_observation(db_session):
    """NH3 and Pb have no CPCB sub-index table in this repo, so a reading that
    carries only those cannot be scored and must stay missing rather than
    being reported as an AQI of 0."""
    from backend.app.services.alert_service import _forecast_inputs

    station = _observation_only_station(db_session, nh3=40.0, pb=2.0)

    assert _forecast_inputs(db_session, station.id) is None


def test_forecast_inputs_returns_none_when_the_station_has_no_observation(db_session):
    from backend.app.models.db_models import Forecast, PollutionReading, Station
    from backend.app.services.alert_service import _forecast_inputs

    station = (
        db_session.query(Station).filter(Station.name == "RK Puram").first()
    )
    assert station is not None
    db_session.query(Forecast).filter(Forecast.station_id == station.id).delete()
    db_session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    db_session.commit()

    assert _forecast_inputs(db_session, station.id) is None


def test_carried_through_aqi_zero_stays_below_the_first_alert_threshold(db_session):
    """Threshold behaviour is unchanged: the first trigger is AQI 201, so a
    valid 0 must produce no alert at all."""
    from backend.app.services.alert_service import _forecast_inputs

    station = _observation_only_station(
        db_session, pm25=0.0, pm10=0.0, o3=0.0, no2=0.0, so2=0.0, co=0.0
    )

    inputs = _forecast_inputs(db_session, station.id)

    assert inputs is not None
    assert generate_alerts(inputs, BASE_WEATHER, BASE_FIRE) == []
