
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "pm25_lag1", "wind_speed", "pbl_height", "temperature",
    "humidity", "fire_impact_score", "hour", "is_winter",
]


@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["pm25_lag1"] = np.clip(X["pm25_lag1"] * 20 + 120, 0, 500)
    X["wind_speed"] = np.clip(np.abs(X["wind_speed"]) * 2, 0, 15)
    X["pbl_height"] = np.clip(np.abs(X["pbl_height"]) * 100 + 200, 50, 1500)
    X["humidity"] = np.clip(X["humidity"] * 10 + 60, 10, 99)
    y = (
        X["pm25_lag1"] * 0.7
        + X["fire_impact_score"] * 30
        - X["wind_speed"] * 8
        - X["pbl_height"] * 0.02
        + rng.normal(0, 5, n)
    )
    y = np.clip(y, 0, 500)
    return X, y


def test_db_models_import():
    from app.models.db_models import (
        Alert,
        FireReading,
        Forecast,
        ModelMetrics,
        PollutionReading,
        Station,
        WeatherReading,
    )
    assert Station.__tablename__ == "stations"
    assert PollutionReading.__tablename__ == "pollution_observations"
    assert WeatherReading.__tablename__ == "weather_observations"
    assert FireReading.__tablename__ == "fire_readings"
    assert Forecast.__tablename__ == "forecasts"
    assert Alert.__tablename__ == "alerts"
    assert ModelMetrics.__tablename__ == "model_metrics"


def test_schemas_import():
    from app.schemas.schemas import ForecastPoint, StationResponse
    assert StationResponse is not None
    assert ForecastPoint is not None


def test_database_import():
    from app.database import Base, SessionLocal, engine
    assert engine is not None
    assert Base is not None
    assert SessionLocal is not None


def test_seed_data_fixture_available(client, db_session):
    from app.database import DEFAULT_STATIONS, SessionLocal
    with SessionLocal() as session:
        from app.models.db_models import Station
        assert session.query(Station).count() == len(DEFAULT_STATIONS)


def test_rf_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert mean_absolute_error(y, predictions) < 20
    assert r2_score(y, predictions) > 0.7


def test_xgboost_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = XGBRegressor(n_estimators=50, max_depth=4, random_state=42).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert mean_absolute_error(y, predictions) < 25
    assert r2_score(y, predictions) > 0.6


def test_model_metrics_from_training(synthetic_data):
    X, y = synthetic_data
    split = int(len(X) * 0.8)
    model = RandomForestRegressor(n_estimators=50, random_state=7)
    model.fit(X.iloc[:split], y[:split])
    predictions = model.predict(X.iloc[split:])
    true = y[split:]
    mae = mean_absolute_error(true, predictions)
    rmse = mean_squared_error(true, predictions) ** 0.5
    r2 = r2_score(true, predictions)
    assert mae < 30
    assert rmse < 45
    assert r2 > 0.5


def test_model_persistence_roundtrip(synthetic_data, tmp_path):
    X, y = synthetic_data
    model = RandomForestRegressor(n_estimators=30, random_state=3).fit(X, y)
    path = tmp_path / "rf_pm25_24h.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    pred_orig = model.predict(X.iloc[:5])
    pred_loaded = loaded.predict(X.iloc[:5])
    np.testing.assert_allclose(pred_orig, pred_loaded)
    assert loaded.n_estimators == model.n_estimators


def test_load_model_returns_none_when_missing(monkeypatch, tmp_path):
    from app.services import forecast_service as fs
    monkeypatch.setattr(fs, "MODEL_DIR", str(tmp_path))
    assert fs.load_model("xgboost_pm25_1h") is None
    assert fs.available_models() == []


def test_predict_pollutants_fallback_without_models(monkeypatch, tmp_path):
    from app.services import forecast_service as fs
    monkeypatch.setattr(fs, "MODEL_DIR", str(tmp_path))
    features = {
        "pm25_lag1": 140.0, "pm10_lag1": 260.0, "o3_lag1": 60.0, "no2_lag1": 80.0,
        "so2_lag1": 16.0, "co_lag1": 2.4, "temperature": 22.0, "humidity": 62.0,
        "pressure_msl": 1013.0, "wind_speed": 1.5, "wind_direction": 315.0,
        "precipitation": 0.0, "cloud_cover": 45.0, "pbl_height": 180.0,
        "fire_impact_score": 0.5, "fire_count_100km": 12, "nearest_fire_km": 80.0,
        "hour": 9, "is_winter": 1, "day_of_year": 320, "aqi_lag1": 210,
    }
    predictions = fs.predict_pollutants(features, horizons=[1, 6, 24, 72])
    assert len(predictions) == 4
    expected_horizons = [1, 6, 24, 72]
    for p, h in zip(predictions, expected_horizons, strict=True):
        assert p["horizon_hours"] == h
        assert p["pm25_pred"] > 0
        assert p["pm10_pred"] > p["pm25_pred"]
        assert p["aqi_pred"] > 0
        assert p["aqi_category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}


def test_predict_pollutants_uses_saved_model(monkeypatch, tmp_path, synthetic_data):
    from app.services import forecast_service as fs
    X, y = synthetic_data
    model = RandomForestRegressor(n_estimators=20, random_state=1).fit(X, y)
    fp = tmp_path / "xgboost_pm25_1h.joblib"
    fp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, fp)
    monkeypatch.setattr(fs, "MODEL_DIR", str(tmp_path))

    features = dict(zip(FEATURE_COLUMNS, X.iloc[0].tolist(), strict=True))
    features.update({"pm10_lag1": 200.0, "temperature": 20.0, "pressure_msl": 1012.0,
                     "wind_direction": 315.0, "precipitation": 0.0, "cloud_cover": 40.0,
                     "fire_impact_score": 0.3, "fire_count_100km": 5, "nearest_fire_km": 100.0,
                     "day_of_year": 320, "aqi_lag1": 180, "season": "winter", "inversion_strength": 0.5})
    predictions = fs.predict_pollutants(features, horizons=[1])
    feat_row = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)
    expected = float(model.predict(feat_row)[0])
    assert predictions[0]["pm25_pred"] == pytest.approx(round(expected, 1), abs=0.5)
    assert fs.available_models() == ["xgboost_pm25_1h.joblib"]


def test_generate_forecast_uses_db_readings(db_session):
    from app.database import SessionLocal
    from app.models.db_models import Forecast
    from app.services import forecast_service as fs
    with SessionLocal() as session:
        from app.models.db_models import Station
        station = session.query(Station).filter(Station.name == "Anand Vihar").first()
        forecasts, predictions = fs.generate_forecast(session, station.id, horizons=[1, 24])
        assert len(forecasts) == 2
        assert len(predictions) == 2
        assert forecasts[0].station_id == station.id
        # The seed already wrote horizons 1..12, so the generated t+1h row
        # replaces the seeded one and only t+24h is new: 12 + 1, not 12 + 2.
        rows = (
            session.query(Forecast)
            .filter(Forecast.station_id == station.id)
            .all()
        )
        assert len(rows) == 12 + 1
        horizons = [r.horizon_hours for r in rows]
        assert len(horizons) == len(set(horizons)), "duplicate horizon rows persisted"
    session.close()


def test_build_features_from_db(db_session):
    from app.database import SessionLocal
    from app.models.db_models import Station
    from app.services import forecast_service as fs
    with SessionLocal() as session:
        station = session.query(Station).filter(Station.name == "Anand Vihar").first()
        features = fs.build_features_from_db(session, station.id)
        # pm25_lag1 is the previous hour's reading (95+6 from the seeded trend)
        assert features["pm25_lag1"] == pytest.approx(101.0)
        assert features["pbl_height"] == 180.0
        assert 0.0 <= features["inversion_strength"] <= 1.0
        assert features["fire_count"] == 0
        assert features["is_winter"] in (0, 1)
    session.close()
