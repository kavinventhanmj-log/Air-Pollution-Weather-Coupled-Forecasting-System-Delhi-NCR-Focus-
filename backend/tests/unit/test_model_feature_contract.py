"""Model feature-contract enforcement (C2).

Reproduces the defect these tests exist to prevent: the flat artifacts declare a
114-name contract, and the serving path used to satisfy it with
``features.get(name, 0.0)`` -- so a feature the builder never produced became a
confident ``0.0`` and the honest heuristic fallback was silently skipped.

Each test pins one property of the fix.
"""

import numpy as np
import pandas as pd
import pytest
from app.services import forecast_service as fs


class _StubModel:
    """Minimal stand-in for the artifact wrapper.

    Records the exact vector it was handed so a test can assert what the model
    was actually told, not merely what it returned.
    """

    def __init__(self, feature_names, value=42.0):
        self.feature_names_ = list(feature_names)
        self._value = value
        self.seen = None

    def predict(self, arr):
        self.seen = np.array(arr, dtype=float)
        return np.array([self._value])


CONTRACT = ["latitude", "longitude", "mean_frp", "max_bright", "pm25_lag1", "wind_speed"]


# --- contract introspection -------------------------------------------------


def test_columns_come_from_the_artifact_not_a_hardcoded_list():
    model = _StubModel(CONTRACT)
    assert fs._model_feature_columns(model) == CONTRACT


def test_columns_fall_through_to_inner_estimator():
    class Inner:
        feature_names_in_ = ["a", "b"]

    class Wrapper:
        model = Inner()

    assert fs._model_feature_columns(Wrapper()) == ["a", "b"]


def test_artifact_without_contract_reports_none():
    class Bare:
        pass

    assert fs._model_feature_columns(Bare()) is None


# --- missing-value detection ------------------------------------------------


def test_absent_key_counts_as_missing():
    assert "latitude" in fs.missing_required_features(CONTRACT, {})


def test_explicit_none_counts_as_missing():
    assert "longitude" in fs.missing_required_features(CONTRACT, {"latitude": 28.6, "longitude": None})


def test_nan_counts_as_missing():
    assert "mean_frp" in fs.missing_required_features(CONTRACT, {"mean_frp": float("nan")})


def test_real_zero_is_preserved_not_treated_as_missing():
    """A measured 0.0 is a value, not a gap.

    Rewriting it would turn a genuinely calm, fire-free hour into "unknown".
    """
    features = {
        "latitude": 28.6,
        "longitude": 77.2,
        "mean_frp": 0.0,
        "max_bright": 0.0,
        "pm25_lag1": 0.0,
        "wind_speed": 0.0,
    }
    assert fs.missing_required_features(CONTRACT, features) == []


# --- the core defect: coordinates must not be imputed -----------------------


def test_missing_coordinates_abort_the_prediction_instead_of_zero_filling():
    """The regression that matters.

    ``latitude``/``longitude`` are real measurements. Imputing 0.0 asserts a
    latitude of zero degrees, which is not a plausible Delhi-NCR coordinate, and
    previously suppressed the honest fallback downstream.
    """
    model = _StubModel(CONTRACT)
    features = {"pm25_lag1": 120.0, "wind_speed": 3.0}  # no coordinates

    assert fs._model_predict(model, features, "pm25") is None
    assert model.seen is None, "the model must not be called with an invalid contract"


def test_coordinates_present_permits_prediction():
    model = _StubModel(CONTRACT)
    features = {
        "latitude": 28.6492,
        "longitude": 77.2918,
        "mean_frp": 0.0,
        "max_bright": 0.0,
        "pm25_lag1": 120.0,
        "wind_speed": 3.0,
    }
    assert fs._model_predict(model, features, "pm25") == 42.0


def test_fire_features_absent_is_imputed_not_fatal():
    """``0.0`` is the validated encoding of "no fire that hour" (trainer fillna(0))."""
    model = _StubModel(CONTRACT)
    features = {
        "latitude": 28.6492,
        "longitude": 77.2918,
        "pm25_lag1": 120.0,
        "wind_speed": 3.0,
    }
    assert fs._model_predict(model, features, "pm25") == 42.0
    assert model.seen is not None
    # Imputed to 0.0, and the vector is still the full declared width.
    assert model.seen.shape == (1, len(CONTRACT))
    idx = CONTRACT.index("mean_frp")
    assert model.seen[0][idx] == 0.0


# --- ordering and artifact integrity ---------------------------------------


def test_vector_is_ordered_by_the_artifact_contract():
    """A builder that emits features in a different order must not matter.

    Projection is by name, so inserting a feature into the dict cannot shift a
    neighbouring column.
    """
    contract = ["z_feature", "a_feature", "m_feature"]
    model = _StubModel(contract)
    shuffled = {"m_feature": 3.0, "a_feature": 2.0, "z_feature": 1.0, "unused_extra": 99.0}

    fs._model_predict(model, shuffled, "pm25")

    assert model.seen[0].tolist() == [1.0, 2.0, 3.0], "columns must follow the artifact order"


def test_extra_features_are_ignored():
    model = _StubModel(["a"])
    fs._model_predict(model, {"a": 1.0, "b": 2.0, "c": 3.0}, "pm25")
    assert model.seen[0].tolist() == [1.0]


def test_artifact_without_contract_refuses_to_predict():
    class Bare:
        def predict(self, arr):
            raise AssertionError("must not be called")

    assert fs._model_predict(Bare(), {"a": 1.0}, "pm25") is None


def test_model_prediction_failure_returns_none():
    class Exploding:
        feature_names_ = ["a"]

        def predict(self, arr):
            raise ValueError("boom")

    assert fs._model_predict(Exploding(), {"a": 1.0}, "pm25") is None


def test_none_model_returns_none():
    assert fs._model_predict(None, {"a": 1.0}, "pm25") is None


def test_negative_prediction_is_clamped_to_zero():
    assert fs._model_predict(_StubModel(["a"], value=-5.0), {"a": 1.0}, "pm25") == 0.0


def test_nan_prediction_is_rejected():
    assert fs._model_predict(_StubModel(["a"], value=float("nan")), {"a": 1.0}, "pm25") is None


# --- FIRMS aggregate features ----------------------------------------------
#
# These pin the serving aggregation to the *training* aggregation in
# ``scripts/build_dataset.py::load_fires``: nearest-station attribution plus an
# exact clock-hour bucket. A window around the origin, or counting every
# nearby fire regardless of which station owns it, both invent values.

ANAND = (28.6492, 77.2918)   # Anand Vihar, in the training station set
ITO = (28.6290, 77.2410)     # ITO, also in the training station set
ORIGIN = pd.Timestamp("2026-09-28T12:00:00Z")


class _Fire:
    def __init__(self, frp=None, brightness=None, acq_date=None,
                 lat=ANAND[0], lon=ANAND[1]):
        self.frp = frp
        self.brightness = brightness
        self.acq_date = acq_date
        self.latitude = lat
        self.longitude = lon


def test_fire_aggregate_uses_detections_in_the_origin_hour():
    fires = [
        _Fire(frp=10.0, brightness=340.0, acq_date=ORIGIN + pd.Timedelta(minutes=10)),
        _Fire(frp=20.0, brightness=350.0, acq_date=ORIGIN + pd.Timedelta(minutes=50)),
    ]
    out = fs._firms_aggregate_features(fires, ORIGIN, *ANAND)
    assert out["mean_frp"] == 15.0
    assert out["max_bright"] == 350.0


def test_fire_aggregate_excludes_other_hours():
    """The training bucket is one clock hour, not a window around the origin.

    A detection in the previous hour belongs to the previous station-hour and
    must not leak into this one.
    """
    fires = [
        _Fire(frp=10.0, brightness=340.0, acq_date=ORIGIN + pd.Timedelta(minutes=5)),
        _Fire(frp=999.0, brightness=999.0, acq_date=ORIGIN - pd.Timedelta(minutes=5)),
    ]
    out = fs._firms_aggregate_features(fires, ORIGIN, *ANAND)
    assert out["mean_frp"] == 10.0
    assert out["max_bright"] == 340.0


def test_fire_aggregate_excludes_detections_belonging_to_another_station():
    """Nearest-station attribution, as the offline builder did.

    A fire nearest ITO must not raise Anand Vihar's ``mean_frp``: at training
    time it was attributed to ITO and never reached this station's row.
    """
    fires = [_Fire(frp=500.0, brightness=360.0, acq_date=ORIGIN, lat=ITO[0], lon=ITO[1])]
    assert fs._firms_aggregate_features(fires, ORIGIN, *ANAND) == {
        "mean_frp": 0.0,
        "max_bright": 0.0,
    }


def test_fire_aggregate_follows_a_detection_to_its_nearest_station():
    """A detection nearest ITO counts for ITO, so both can be correct at once."""
    near_ito = _Fire(frp=8.0, brightness=300.0, acq_date=ORIGIN, lat=ITO[0], lon=ITO[1])
    assert fs._firms_aggregate_features([near_ito], ORIGIN, *ITO)["mean_frp"] == 8.0
    assert fs._firms_aggregate_features([near_ito], ORIGIN, *ANAND)["mean_frp"] == 0.0


def test_fire_aggregate_returns_zero_when_no_detections():
    assert fs._firms_aggregate_features([], ORIGIN, *ANAND) == {"mean_frp": 0.0, "max_bright": 0.0}


def test_fire_aggregate_tolerates_missing_origin():
    assert fs._firms_aggregate_features([_Fire(frp=5.0)], None, *ANAND) == {
        "mean_frp": 0.0,
        "max_bright": 0.0,
    }


def test_fire_aggregate_tolerates_missing_station_coordinates():
    """No coordinates means no nearest-station attribution is possible."""
    fires = [_Fire(frp=5.0, acq_date=ORIGIN)]
    assert fs._firms_aggregate_features(fires, ORIGIN, None, None)["mean_frp"] == 0.0


def test_fire_aggregate_handles_naive_timestamps():
    """``acq_date`` is stored naive-UTC; the comparison must still work."""
    origin = pd.Timestamp("2026-09-28T12:00:00")
    fires = [_Fire(frp=7.0, brightness=300.0, acq_date=pd.Timestamp("2026-09-28T12:30:00"))]
    assert fs._firms_aggregate_features(fires, origin, *ANAND)["mean_frp"] == 7.0


def test_fire_aggregate_skips_detections_without_coordinates():
    """Cannot be attributed to a station, so cannot be counted."""
    fires = [_Fire(frp=5.0, acq_date=ORIGIN, lat=None, lon=None)]
    assert fs._firms_aggregate_features(fires, ORIGIN, *ANAND)["mean_frp"] == 0.0


# --- provenance audit -------------------------------------------------------


def test_audit_reports_unusable_contract_when_coordinates_absent(monkeypatch):
    monkeypatch.setattr(fs, "load_pollutant_model", lambda *a, **k: _StubModel(CONTRACT))
    audit = fs.audit_feature_contract({"pm25_lag1": 100.0}, [24])
    assert audit["usable"] is False
    assert audit["satisfied"] is False
    assert set(audit["missing_required"]) == {"latitude", "longitude"}
    assert "unavailable" in audit["reason"]


def test_audit_reports_satisfied_contract_with_coordinates(monkeypatch):
    monkeypatch.setattr(fs, "load_pollutant_model", lambda *a, **k: _StubModel(CONTRACT))
    audit = fs.audit_feature_contract(
        {
            "latitude": 28.6492,
            "longitude": 77.2918,
            "mean_frp": 0.0,
            "max_bright": 0.0,
            "pm25_lag1": 100.0,
            "wind_speed": 3.0,
        },
        [24],
    )
    assert audit["usable"] is True
    assert audit["declared_features"] == len(CONTRACT)
    assert audit["missing_required"] == []


def test_audit_lists_imputed_features(monkeypatch):
    monkeypatch.setattr(fs, "load_pollutant_model", lambda *a, **k: _StubModel(CONTRACT))
    audit = fs.audit_feature_contract(
        {"latitude": 28.6, "longitude": 77.2, "pm25_lag1": 1.0, "wind_speed": 1.0}, [24]
    )
    assert set(audit["imputed"]) == {"mean_frp", "max_bright"}
    assert audit["usable"] is True


def test_audit_is_honest_when_artifact_missing(monkeypatch):
    monkeypatch.setattr(fs, "load_pollutant_model", lambda *a, **k: None)
    audit = fs.audit_feature_contract({"latitude": 1.0}, [24])
    assert audit["usable"] is False
    assert "could not be loaded" in audit["reason"]


# --- fallback still works when the model is bypassed -----------------------


def test_fallback_used_when_contract_unsatisfiable(monkeypatch):
    """The end-to-end guarantee: a bad contract yields the heuristic, not a lie."""
    monkeypatch.setattr(fs, "load_pollutant_model", lambda *a, **k: _StubModel(CONTRACT))
    features = {"pm25_lag1": 200.0, "wind_speed": 2.0, "pbl_height": 300.0}
    preds = fs.predict_pollutants(features, [1])
    # The stub would return 42.0; with no coordinates the fallback path runs and
    # produces a value derived from the observed pm25, not the constant 42.0.
    assert len(preds) == 1
    assert preds[0]["pm25_pred"] != 42.0
    assert preds[0]["pm25_pred"] > 0


# --- season one-hot ---------------------------------------------------------


def test_season_dummies_always_emit_all_four_names():
    """``pd.get_dummies`` only emits the categories present in the frame.

    A short serving history produced one dummy and three absent names, which the
    model layer then received as 0.0. Every name must be present regardless.
    """
    out = fs._season_dummies(pd.Timestamp("2026-09-28T12:00:00Z"))  # autumn
    assert set(out) == {f"season_{s}" for s in fs.SEASON_NAMES}


def test_season_dummies_encode_the_row_own_season():
    for month, expected in ((1, "winter"), (4, "spring"), (7, "summer"), (10, "autumn")):
        ts = pd.Timestamp(f"2025-{month:02d}-15T00:00:00Z")
        out = fs._season_dummies(ts)
        assert out[f"season_{expected}"] == 1.0, f"month {month} should be {expected}"
        assert sum(out.values()) == 1.0, "exactly one season must be hot"


def test_season_dummies_agree_with_the_training_map():
    """Derived from the training SEASON_MAP, so the two cannot drift apart."""
    from ml.features.feature_engineering import SEASON_MAP

    for month, season in SEASON_MAP.items():
        ts = pd.Timestamp(f"2025-{int(month):02d}-15T00:00:00Z")
        out = fs._season_dummies(ts)
        assert out[f"season_{season}"] == 1.0


def test_season_dummies_of_unparseable_timestamp_are_all_zero():
    """Unknown season stays unknown; an arbitrary one would be a fabrication."""
    out = fs._season_dummies(None)
    assert set(out.values()) == {0.0}


def test_season_dummies_of_naive_timestamp_are_derived_not_zeroed():
    out = fs._season_dummies(pd.Timestamp("2026-01-15T00:00:00"))
    assert out["season_winter"] == 1.0


# --- the whole 114-name contract, end to end -------------------------------
#
# The decisive test. It runs the real feature builder against a real (temporary
# SQLite) database and asks the real artifact for its contract, so a regression
# anywhere in the builder is caught rather than papered over by a stub.


def _populated_db(tmp_path, station_name="Anand Vihar", lat=28.6492, lon=77.2918,
                  fire_lat=28.6492, fire_lon=77.2918):
    """A station with 30 hours of real-shaped observations and one FIRMS row."""
    from datetime import UTC, datetime, timedelta

    from app.database import Base
    from app.models.db_models import (
        FireReading,
        PollutionReading,
        Station,
        WeatherReading,
    )
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'contract.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    station = Station(name=station_name, latitude=lat, longitude=lon, city="Delhi NCR")
    session.add(station)
    session.commit()

    origin = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    for h in range(30):
        ts = origin - timedelta(hours=h)
        session.add(PollutionReading(
            station_id=station.id, timestamp=ts, pm25=100.0 + h, pm10=150.0 + h,
            no2=30.0, o3=40.0, so2=10.0, co=1.0, aqi=120.0, data_source="test",
        ))
        session.add(WeatherReading(
            station_id=station.id, timestamp=ts, temperature=30.0, humidity=60.0,
            pressure_msl=1000.0, surface_pressure=1005.0, wind_speed=3.0,
            wind_direction=270.0, precipitation=0.0, cloud_cover=2.0, pbl_height=800.0,
        ))
    session.add(FireReading(
        satellite="VIIRS", instrument="SNPP", latitude=fire_lat, longitude=fire_lon,
        acq_date=origin, frp=12.0, brightness=340.0, confidence="nominal",
        daynight="day",
    ))
    session.commit()
    return session, station.id


def _real_artifact_contract():
    """The contract the shipped flat artifact actually declares."""
    model = fs.load_pollutant_model("pm25", 24)
    assert model is not None, "the pm25 24h artifact must be loadable for this test to mean anything"
    return fs._model_feature_columns(model)


def test_real_builder_satisfies_the_real_artifact_contract(tmp_path):
    """The regression this whole exercise exists to prevent.

    Before the fix this reported seven absent contract names: the two
    coordinates, ``mean_frp``/``max_bright``, and three season dummies, all of
    which reached the model as a confident ``0.0``.
    """
    session, station_id = _populated_db(tmp_path)
    try:
        features, _ = fs.build_features_from_db_with_meta(session, station_id)
        contract = _real_artifact_contract()

        assert len(contract) == 114, "the flat artifact contract changed; update this test"
        assert fs.missing_required_features(contract, features) == []
    finally:
        session.close()


def test_real_builder_audit_reports_usable_with_nothing_imputed(tmp_path):
    """Coordinates and fires are now supplied, so nothing should be imputed."""
    session, station_id = _populated_db(tmp_path)
    try:
        features, _ = fs.build_features_from_db_with_meta(session, station_id)
        audit = fs.audit_feature_contract(features, [1, 6, 12, 24, 48, 72])
        assert audit["declared_features"] == 114
        assert audit["missing_required"] == []
        assert audit["imputed"] == []
        assert audit["usable"] is True
        assert audit["reason"] is None
    finally:
        session.close()


def test_real_builder_carries_the_coordinates_it_owns(tmp_path):
    """A real station coordinate, not the 0.0 that is not a Delhi-NCR point."""
    session, station_id = _populated_db(tmp_path)
    try:
        features, _ = fs.build_features_from_db_with_meta(session, station_id)
        assert features["latitude"] == 28.6492
        assert features["longitude"] == 77.2918
    finally:
        session.close()


def test_real_builder_aggregates_the_station_own_fire(tmp_path):
    session, station_id = _populated_db(tmp_path)
    try:
        features, _ = fs.build_features_from_db_with_meta(session, station_id)
        assert features["mean_frp"] == 12.0
        assert features["max_bright"] == 340.0
    finally:
        session.close()


def test_real_builder_reports_zero_fires_when_the_station_hour_is_clear(tmp_path):
    """No detection for this station-hour: 0.0 is the trained encoding."""
    session, station_id = _populated_db(tmp_path, fire_lat=0.0, fire_lon=0.0)
    try:
        features, _ = fs.build_features_from_db_with_meta(session, station_id)
        assert features["mean_frp"] == 0.0
        assert features["max_bright"] == 0.0
    finally:
        session.close()


def test_real_builder_fires_never_leak_between_stations(tmp_path):
    """A detection nearest another station must not raise this station's value."""
    ito = fs.TRAINING_FIRE_STATIONS["ITO"]
    session, station_id = _populated_db(tmp_path, fire_lat=ito[0], fire_lon=ito[1])
    try:
        features, _ = fs.build_features_from_db_with_meta(session, station_id)
        assert features["mean_frp"] == 0.0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
