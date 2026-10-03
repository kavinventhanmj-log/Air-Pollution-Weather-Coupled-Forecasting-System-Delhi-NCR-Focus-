import atexit
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

# --------------------------------------------------------------------------- #
# Disposable-database guard
#
# ``db_session`` drops and recreates every table, so a stray non-SQLite
# ``DATABASE_URL`` (the app reads the same variable) would let a local test run
# destroy a live schema. Refuse that unless the operator explicitly opts in.
# --------------------------------------------------------------------------- #

#: Opt-in required before pytest will touch a non-SQLite database.
ALLOW_LIVE_TEST_DB_ENV = "AEROCAST_ALLOW_LIVE_TEST_DB"

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}


class LiveTestDatabaseRefused(RuntimeError):
    """A non-SQLite test target was refused because it is not disposable."""


def _opt_in_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def ensure_disposable_test_database(environ) -> None:
    """Refuse to run when ``DATABASE_URL`` would target a non-throwaway database.

    The message never echoes the URL, so no credential can leak through it.
    """
    url = (environ.get("DATABASE_URL") or "").strip()
    if not url:
        return
    backend = url.split(":", 1)[0].split("+", 1)[0].strip().lower()
    if backend == "sqlite":
        return
    if _opt_in_enabled(environ.get(ALLOW_LIVE_TEST_DB_ENV)):
        return
    raise LiveTestDatabaseRefused(
        "Refusing to run pytest: DATABASE_URL points at a non-SQLite database "
        "and the test suite drops and recreates every table. This would destroy "
        "that database. Point DATABASE_URL at a throwaway SQLite file, or set "
        f"{ALLOW_LIVE_TEST_DB_ENV}=1 to explicitly allow a dedicated test "
        "database."
    )


def ensure_disposable_backend(backend_name: str, environ) -> None:
    """Belt-and-braces check run immediately before the schema is dropped."""
    if (backend_name or "").strip().lower() == "sqlite":
        return
    if _opt_in_enabled(environ.get(ALLOW_LIVE_TEST_DB_ENV)):
        return
    raise LiveTestDatabaseRefused(
        "Refusing to drop tables on a non-SQLite test database. Set "
        f"{ALLOW_LIVE_TEST_DB_ENV}=1 to explicitly allow a dedicated test "
        "database."
    )


def pytest_configure(config) -> None:
    """Abort before any test runs if the configured target is not disposable."""
    ensure_disposable_test_database(os.environ)


# Each pytest run gets its own database directory.
#
# This used to be a single fixed file, `gettempdir()/aerocast_ncr_test.db`, deleted
# once at import. That made the suite collide with itself: a run killed mid-write
# (timeout, Ctrl-C, crashed worker) left a truncated file behind, and the *next*
# run - or a concurrently running one - then died in `db_session` with
# "database disk image is malformed" while dropping tables. It also meant two
# pytest processes sharing a temp dir fought over the same file.
#
# A unique per-run directory removes both failure modes, and the directory is
# removed when the session finishes.
_RUN_DIR = pathlib.Path(tempfile.mkdtemp(prefix="aerocast_ncr_test_"))
TEST_DB_PATH = _RUN_DIR / "test.db"

#: Every naive timestamp this project persists is UTC. The serving contract is
#: spelled out in ``app.api.summary`` / ``app.services.forecast_service`` /
#: ``app.services.refresh_service``, all of which read a naive stored value as
#: UTC wall-clock, so the seed below must match it exactly. Storing station
#: observations as naive IST here placed seeded rows 5 h 30 m in the *future*
#: relative to those consumers, which is what broke the lag-1 and observation-age
#: assertions.
TEST_TZ = UTC

# Drop any stale fixed-path database left by the old scheme so it cannot be
# picked up or mistaken for a valid fixture.
_LEGACY_TEST_DB_PATH = pathlib.Path(tempfile.gettempdir()) / "aerocast_ncr_test.db"
if _LEGACY_TEST_DB_PATH.exists():
    try:
        _LEGACY_TEST_DB_PATH.unlink()
    except OSError:
        pass

# Default to an ephemeral local SQLite DB, unless CI already pointed us at a
# real database (e.g. PostgreSQL) via DATABASE_URL.
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

    def _remove_run_dir() -> None:
        # Release pooled connections first: on Windows the SQLite file cannot be
        # deleted while any connection still holds it open, and rmtree would
        # then silently leave the directory behind.
        try:
            from app.database import engine

            engine.dispose()
        except Exception:
            pass
        shutil.rmtree(_RUN_DIR, ignore_errors=True)

    atexit.register(_remove_run_dir)

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

    # Every naive timestamp this fixture writes is UTC, matching the application
    # serving contract (`summary._naive_utc`, `forecast_service._as_naive_utc`,
    # `refresh_service._existing_timestamps`), which read a naive stored value as
    # UTC wall-clock.
    #
    #   * station observations (pollution + weather) and forecast origin times are
    #     naive **UTC**, and
    #   * fire acquisition times are naive **UTC** as well, because FIRMS
    #     `acq_time` is UTC by definition.
    #
    # These used to differ (station observations naive IST, fires naive UTC). The
    # IST seeding put every station row 5 h 30 m in the future relative to the
    # serving code's "now", so the lag-1 feature window, the forecast-horizon
    # targets, the refresh upsert and the summary observation age all resolved to
    # the wrong row - or, for the age, to a negative number.
    base_utc = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)

    readings = []
    weather = []
    for i in range(12):
        ts = base_utc - timedelta(hours=i)
        pm25 = round(95 + 6 * i, 1)
        pm10 = round(180 + 10 * i, 1)
        o3 = round(55 + i, 1)
        no2 = round(80 + 3 * i, 1)
        so2 = round(16.0, 1)
        co = round(2.2 + 0.1 * i, 2)
        aqi = _calculate_current_aqi(pm25, pm10, o3, no2, so2, co)
        readings.append(
            PollutionReading(
                station_id=station.id,
                timestamp=ts,
                pm25=pm25,
                pm10=pm10,
                o3=o3,
                no2=no2,
                so2=so2,
                co=co,
                aqi=aqi,
            )
        )
        weather.append(
            WeatherReading(
                station_id=station.id,
                timestamp=ts,
                temperature=round(22.5 - 0.1 * i, 1),
                humidity=62.0,
                pressure_msl=1013.0,
                surface_pressure=993.0,
                wind_speed=1.6,
                wind_direction=315.0,
                precipitation=0.0,
                cloud_cover=45.0,
                pbl_height=180.0,
            )
        )
    session.add_all(readings)
    session.add_all(weather)

    # Fires are naive UTC, because FIRMS `acq_time` is UTC by definition. They are
    # anchored two hours before the release hour so they sit inside the trailing
    # window the feature builder derives and actually contribute to the fire
    # features - a fire stamped relative to the raw wall clock rather than to the
    # release hour would fall outside that window and be excluded from every one.
    fire_ts = base_utc - timedelta(hours=2)
    session.add_all(
        [
            FireReading(
                latitude=30.5,
                longitude=76.1,
                acq_date=fire_ts,
                confidence="high",
                frp=90.0,
                satellite="SNPP",
                daynight="D",
            ),
            FireReading(
                latitude=30.9,
                longitude=76.5,
                acq_date=fire_ts,
                confidence="high",
                frp=120.0,
                satellite="S-NPP",
                daynight="D",
            ),
            FireReading(
                latitude=28.0,
                longitude=74.5,
                acq_date=fire_ts,
                confidence="nominal",
                frp=40.0,
                satellite="SNPP",
                daynight="N",
            ),
            FireReading(
                latitude=31.2,
                longitude=75.9,
                acq_date=fire_ts,
                confidence="low",
                frp=18.0,
                satellite="VIIRS",
                daynight="D",
            ),
            FireReading(
                latitude=29.5,
                longitude=76.0,
                acq_date=fire_ts,
                confidence="high",
                frp=65.0,
                satellite="SNPP",
                daynight="D",
            ),
        ]
    )

    # Distinct horizon_hours per row: ``forecasts`` is unique on
    # (station_id, horizon_hours) so a regeneration replaces the horizon
    # instead of appending a duplicate.
    forecasts = []
    for i in range(12):
        # Forecast origin times sit on the same axis as the weather they extend,
        # i.e. the naive-UTC station timeline.
        ts = base_utc - timedelta(hours=i)
        pm25 = round(100 + 6 * i, 1)
        pm10 = round(190 + 10 * i, 1)
        o3 = round(60 + i, 1)
        no2 = round(85 + 3 * i, 1)
        aqi = _calculate_current_aqi(pm25, pm10, o3, no2, 16.0, 2.5)
        forecasts.append(
            Forecast(
                station_id=station.id,
                forecast_timestamp=ts,
                horizon_hours=i + 1,
                pm25_pred=pm25,
                pm10_pred=pm10,
                o3_pred=o3,
                no2_pred=no2,
                aqi_pred=aqi,
                aqi_category="Moderate",
                dominant_pollutant="pm25",
                inversion_detected=1,
                inversion_strength=0.64,
                pbl_height=180.0,
            )
        )
    session.add_all(forecasts)

    session.add_all(
        [
            ModelMetrics(
                model_name="xgboost",
                pollutant="pm25",
                horizon_hours=24,
                mae=12.4,
                rmse=19.3,
                r2=0.83,
                mape=14.2,
                test_period_start=base_utc - timedelta(days=30),
                test_period_end=base_utc,
            ),
            ModelMetrics(
                model_name="random_forest",
                pollutant="pm10",
                horizon_hours=24,
                mae=21.0,
                rmse=31.5,
                r2=0.76,
                mape=18.0,
                test_period_start=base_utc - timedelta(days=30),
                test_period_end=base_utc,
            ),
        ]
    )

    session.add(
        Alert(
            station_id=station.id,
            alert_level="WARNING",
            title="Severe Pollution Alert",
            description="Test alert description",
            forecast_horizon_hours=24,
            factors="High AQI",
            recommendation="Avoid outdoor activity",
        )
    )
    session.commit()


@pytest.fixture()
def db_session():
    # Belt-and-braces: refuse to drop tables on anything but a disposable
    # SQLite target, even if the import-time guard was bypassed somehow.
    ensure_disposable_backend(engine.url.get_backend_name(), os.environ)
    # Release every pooled connection before dropping the schema. On PostgreSQL
    # ``DROP TABLE`` needs an AccessExclusiveLock, which deadlocks against the
    # AccessShareLock still held by the session-scoped ``client`` fixture's
    # connection. SQLite has no such lock, so this only ever bit the CI job that
    # runs the suite against PostgreSQL.
    engine.dispose()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        _seed_test_data(session)
        yield session


@pytest.fixture()
def db_guard():
    """Expose the disposable-database guards for direct unit testing."""
    return SimpleNamespace(
        ensure_disposable_test_database=ensure_disposable_test_database,
        ensure_disposable_backend=ensure_disposable_backend,
        live_test_database_refused=LiveTestDatabaseRefused,
        allow_env=ALLOW_LIVE_TEST_DB_ENV,
    )


@pytest.fixture(autouse=True)
def _clear_ttl_cache():
    """Drop the process-global TTL cache around every test.

    The cache is enabled on PostgreSQL and disabled on SQLite, so gating it on
    the database dialect let cached payloads computed for one test be served to
    the next whenever CI ran against Postgres. Two tests asserting an empty
    result failed there for exactly this reason while passing locally.

    ``ttl_cache._cache_enabled`` now also refuses to cache under pytest, which
    is the actual fix. This fixture is the belt-and-braces half: it guarantees
    isolation even if a future change re-enables caching, and it keeps a
    non-cached store from holding references to objects from a dropped schema.
    """
    from app.services.ttl_cache import invalidate_all

    invalidate_all()
    yield
    invalidate_all()
