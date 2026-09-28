import os
import pathlib
import sys
import tempfile
from datetime import UTC, datetime, timedelta

import pytest

TEST_DB_PATH = pathlib.Path(tempfile.gettempdir()) / "aerocast_ncr_test.db"
if TEST_DB_PATH.exists():
    try:
        TEST_DB_PATH.unlink()
    except PermissionError:
        pass
# Default to an ephemeral local SQLite DB, unless CI already pointed us at a
# real database (e.g. PostgreSQL) via DATABASE_URL.
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

# make the repository root importable so tests can reach `ml.*` directly
REDIRECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REDIRECT_ROOT) not in sys.path:
    sys.path.insert(0, str(REDIRECT_ROOT))
# Prefer the real `app` package under backend/ over the repo-root Render shim
# (app/ re-export), which shadows it and breaks `from app.database import ...`.
BACKEND_ROOT = pathlib.Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import Base, SessionLocal, engine, seed_data
from app.main import app
from app.models.db_models import (
    Alert,
    FireReading,
    Forecast,
    ModelMetrics,
    PollutionReading,
    Station,
    WeatherReading,
)
from app.services.aqi_calculator import calculate_aqi


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

def _calculate_current_aqi(pm25, pm10, o3, no2, so2, co):
    aqi, _, _ = calculate_aqi(pm25, pm10, o3, no2, so2, co)
    return aqi

def _seed_test_data(session):
    seed_data(session)
    station = session.query(Station).filter(Station.name == "Anand Vihar").first()

    base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)

    readings = []
    weather = []
    for i in range(12):
        ts = base - timedelta(hours=i)
        pm25 = round(95 + 6 * i, 1)
        pm10 = round(180 + 10 * i, 1)
        o3 = round(55 + i, 1)
        no2 = round(80 + 3 * i, 1)
        so2 = round(16.0, 1)
        co = round(2.2 + 0.1 * i, 2)
        aqi = _calculate_current_aqi(pm25, pm10, o3, no2, so2, co)
        readings.append(PollutionReading(
            station_id=station.id, timestamp=ts,
            pm25=pm25, pm10=pm10, o3=o3, no2=no2, so2=so2, co=co, aqi=aqi,
        ))
        weather.append(WeatherReading(
            station_id=station.id, timestamp=ts,
            temperature=round(22.5 - 0.1 * i, 1), humidity=62.0,
            pressure_msl=1013.0, surface_pressure=993.0,
            wind_speed=1.6, wind_direction=315.0,
            precipitation=0.0, cloud_cover=45.0, pbl_height=180.0,
        ))
    session.add_all(readings)
    session.add_all(weather)

    fire_ts = base - timedelta(hours=2)
    session.add_all([
        FireReading(latitude=30.5, longitude=76.1, acq_date=fire_ts, confidence="high", frp=90.0, satellite="SNPP", daynight="D"),
        FireReading(latitude=30.9, longitude=76.5, acq_date=fire_ts, confidence="high", frp=120.0, satellite="S-NPP", daynight="D"),
        FireReading(latitude=28.0, longitude=74.5, acq_date=fire_ts, confidence="nominal", frp=40.0, satellite="SNPP", daynight="N"),
        FireReading(latitude=31.2, longitude=75.9, acq_date=fire_ts, confidence="low", frp=18.0, satellite="VIIRS", daynight="D"),
        FireReading(latitude=29.5, longitude=76.0, acq_date=fire_ts, confidence="high", frp=65.0, satellite="SNPP", daynight="D"),
    ])

    # Distinct horizon_hours per row: ``forecasts`` is unique on
    # (station_id, horizon_hours) so a regeneration replaces the horizon
    # instead of appending a duplicate.
    forecasts = []
    for i in range(12):
        ts = base - timedelta(hours=i)
        pm25 = round(100 + 6 * i, 1)
        pm10 = round(190 + 10 * i, 1)
        o3 = round(60 + i, 1)
        no2 = round(85 + 3 * i, 1)
        aqi = _calculate_current_aqi(pm25, pm10, o3, no2, 16.0, 2.5)
        forecasts.append(Forecast(
            station_id=station.id, forecast_timestamp=ts, horizon_hours=i + 1,
            pm25_pred=pm25, pm10_pred=pm10, o3_pred=o3, no2_pred=no2,
            aqi_pred=aqi, aqi_category="Moderate", dominant_pollutant="pm25",
            inversion_detected=1, inversion_strength=0.64, pbl_height=180.0,
        ))
    session.add_all(forecasts)

    session.add_all([
        ModelMetrics(
            model_name="xgboost", pollutant="pm25", horizon_hours=24,
            mae=12.4, rmse=19.3, r2=0.83, mape=14.2,
            test_period_start=base - timedelta(days=30), test_period_end=base,
        ),
        ModelMetrics(
            model_name="random_forest", pollutant="pm10", horizon_hours=24,
            mae=21.0, rmse=31.5, r2=0.76, mape=18.0,
            test_period_start=base - timedelta(days=30), test_period_end=base,
        ),
    ])

    session.add(Alert(
        station_id=station.id, alert_level="WARNING", title="Severe Pollution Alert",
        description="Test alert description", forecast_horizon_hours=24,
        factors="High AQI", recommendation="Avoid outdoor activity",
    ))
    session.commit()

@pytest.fixture()
def db_session():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        _seed_test_data(session)
        yield session
