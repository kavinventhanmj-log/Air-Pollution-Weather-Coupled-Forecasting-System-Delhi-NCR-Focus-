"""P0: the forecast path must refuse to invent values (SIH honesty requirement).

Before this work, ``build_features_from_db_with_meta`` returned
``{name: 0.0 for name in FEATURE_NAMES}`` whenever a station had no usable
history. Those zeros were consumed by the models and by the ``or``-defaulted
fallbacks, so an empty database produced confident, plausible, entirely
fabricated PM2.5/AQI numbers that were then persisted as real forecasts.

These tests lock in the corrected contract:
  * no feature vector is produced without real observations,
  * a genuine ``0.0`` measurement survives (it is data, not absence),
  * the API answers ``503 insufficient_data`` and persists nothing.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.models.db_models import Forecast, PollutionReading, Station, WeatherReading
from app.services import forecast_service as fs


def _station(db, name: str = "Anand Vihar") -> Station:
    return db.query(Station).filter(Station.name == name).first()


def _wipe_station_data(db, station: Station) -> None:
    db.query(PollutionReading).filter(PollutionReading.station_id == station.id).delete()
    db.query(WeatherReading).filter(WeatherReading.station_id == station.id).delete()
    db.commit()


# ---------------------------------------------------------------------------
# _feature: missing-ness vs. a real zero
# ---------------------------------------------------------------------------

class TestFeatureAccessor:
    def test_genuine_zero_is_preserved(self):
        assert fs._feature({"pm25_lag1": 0.0}, "pm25_lag1", 50.0) == 0.0

    def test_none_nan_and_absent_use_the_default(self):
        assert fs._feature({"pm25_lag1": None}, "pm25_lag1", 50.0) == 50.0
        assert fs._feature({"pm25_lag1": float("nan")}, "pm25_lag1", 50.0) == 50.0
        assert fs._feature({}, "pm25_lag1", 50.0) == 50.0

    def test_real_values_and_numeric_strings_survive(self):
        assert fs._feature({"pm25_lag1": 137.5}, "pm25_lag1", 50.0) == 137.5
        assert fs._feature({"pm25_lag1": "88.0"}, "pm25_lag1", 50.0) == 88.0

    def test_non_dict_input_is_tolerated(self):
        assert fs._feature(None, "pm25_lag1", 7.0) == 7.0


class TestFallbacksRespectZero:
    """The old ``features.get(k, d) or d`` idiom erased a real ``0.0``."""

    def test_zero_pm25_stays_zero(self):
        zero_vec = {name: 0.0 for name in fs.FEATURE_NAMES}
        assert fs._fallback_pm25(zero_vec, 6) == 0.0

    def test_zero_pm10_uses_the_pm25_ratio_not_a_default(self):
        zero_vec = {name: 0.0 for name in fs.FEATURE_NAMES}
        assert fs._fallback_pm10(20.0, zero_vec, 6) == pytest.approx(30.0)

    def test_real_pm25_lag_is_used(self):
        assert fs._fallback_pm25({"pm25_lag1": 100.0}, 1) > 0


# ---------------------------------------------------------------------------
# build_features_from_db_with_meta refuses instead of zero-filling
# ---------------------------------------------------------------------------

class TestInsufficientDataRefusal:
    def test_empty_station_raises(self, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        with pytest.raises(fs.InsufficientDataError) as excinfo:
            fs.build_features_from_db_with_meta(db_session, station.id)
        assert excinfo.value.available_hours == 0
        assert excinfo.value.required_hours == fs.MIN_FEATURE_ROWS

    def test_error_payload_is_machine_readable(self, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        with pytest.raises(fs.InsufficientDataError) as excinfo:
            fs.build_features_from_db_with_meta(db_session, station.id)
        payload = excinfo.value.to_payload()
        assert payload["code"] == "insufficient_data"
        assert payload["station"] == station.name
        assert payload["available_hours"] == 0
        assert payload["required_hours"] == fs.MIN_FEATURE_ROWS
        assert payload["reason"]

    def test_single_row_cannot_form_a_lag(self, db_session):
        """One row means every lag feature is NaN, which is why 2 is the floor."""
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        base = datetime.now(UTC).replace(tzinfo=None)
        db_session.add(
            PollutionReading(station_id=station.id, timestamp=base, pm25=90.0, pm10=180.0)
        )
        db_session.commit()
        with pytest.raises(fs.InsufficientDataError) as excinfo:
            fs.build_features_from_db_with_meta(db_session, station.id)
        assert excinfo.value.available_hours < fs.MIN_FEATURE_ROWS

    def test_two_rows_are_accepted(self, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        base = datetime.now(UTC).replace(tzinfo=None)
        for i in range(2):
            db_session.add(
                PollutionReading(
                    station_id=station.id,
                    timestamp=base - timedelta(hours=i),
                    pm25=100.0 + i,
                    pm10=200.0 + i,
                    o3=50.0,
                    no2=40.0,
                    so2=15.0,
                    co=1.5,
                )
            )
        db_session.commit()
        features, meta = fs.build_features_from_db_with_meta(db_session, station.id)
        # Rows land newest-last, so lag-1 of the newest row is the older one (101.0).
        assert features["pm25_lag1"] == 101.0
        assert meta["available_rows"] >= fs.MIN_FEATURE_ROWS

    def test_has_sufficient_history_helper(self, db_session):
        station = _station(db_session)
        assert fs.has_sufficient_history(db_session, station.id) is True
        _wipe_station_data(db_session, station)
        assert fs.has_sufficient_history(db_session, station.id) is False

    def test_meta_reports_the_real_window(self, db_session):
        features, meta = fs.build_features_from_db_with_meta(db_session, _station(db_session).id)
        assert meta["window_start"] and meta["window_end"]
        assert meta["available_rows"] > 0


# ---------------------------------------------------------------------------
# API contract: 503 insufficient_data, and nothing persisted
# ---------------------------------------------------------------------------

class TestApiRefusalContract:
    def test_generate_forecast_returns_503_for_an_empty_station(self, client, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        before = db_session.query(Forecast).filter(Forecast.station_id == station.id).count()

        resp = client.post(
            "/api/forecast/generate", json={"station_name": station.name, "horizons": [1, 24]}
        )
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["code"] == "insufficient_data"
        assert detail["station"] == station.name
        assert detail["available_hours"] == 0

        db_session.expire_all()
        after = db_session.query(Forecast).filter(Forecast.station_id == station.id).count()
        assert after == before, "a refused forecast must never persist a row"

    def test_coupled_endpoint_returns_503_for_an_empty_station(self, client, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        before = db_session.query(Forecast).filter(Forecast.station_id == station.id).count()

        resp = client.post(
            "/api/forecast/coupled", json={"station_name": station.name, "horizons": [1, 24]}
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "insufficient_data"

        db_session.expire_all()
        after = db_session.query(Forecast).filter(Forecast.station_id == station.id).count()
        assert after == before

    def test_explanation_endpoint_surfaces_insufficient_data(self, client, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        resp = client.get(f"/api/explanation/{station.name}")
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_present_on_a_successful_forecast(self, client, db_session):
        resp = client.post(
            "/api/forecast/generate", json={"station_name": "Anand Vihar", "horizons": [1, 24]}
        )
        assert resp.status_code == 200, resp.text
        prov = resp.json()["provenance"]
        assert prov["model"] in {"coupled-two-way", "direct-ml"}
        assert prov["pollution_rows"] > 0
        assert prov["is_stale"] is False
        assert prov["observation_age_hours"] is not None
        assert prov["generated_at"]

    def test_coupled_provenance_present(self, client, db_session):
        resp = client.post(
            "/api/forecast/coupled", json={"station_name": "Anand Vihar", "horizons": [1, 6]}
        )
        assert resp.status_code == 200, resp.text
        prov = resp.json()["provenance"]
        assert prov["model"] == "coupled-two-way"
        assert prov["history_rows"] > 0

    def test_stale_data_is_flagged_not_hidden(self, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        base = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=48)
        for i in range(4):
            db_session.add(
                PollutionReading(
                    station_id=station.id,
                    timestamp=base - timedelta(hours=i),
                    pm25=110.0,
                    pm10=210.0,
                )
            )
        db_session.commit()
        prov = fs.build_provenance(
            db_session, station_id=station.id, station_name=station.name, coverage={}
        )
        assert prov["is_stale"] is True
        assert prov["observation_age_hours"] > fs.STALE_AFTER_HOURS

    def test_demo_data_is_labelled(self, db_session):
        station = _station(db_session)
        _wipe_station_data(db_session, station)
        base = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
        db_session.add(
            PollutionReading(
                station_id=station.id,
                timestamp=base,
                pm25=120.0,
                pm10=220.0,
                data_source="demo-bundled-csv",
            )
        )
        db_session.commit()
        prov = fs.build_provenance(
            db_session, station_id=station.id, station_name=station.name, coverage={}
        )
        assert prov["is_demo"] is True
        assert prov["data_source"] == "demo-bundled-csv"
