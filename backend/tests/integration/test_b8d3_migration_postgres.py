"""PostgreSQL verification of the head migration ``b8d3f1a9c4e2``.

Why this needs a real PostgreSQL and not SQLite:

* the revision's ``DELETE`` deduplication and orphan cleanup are the operations
  that will actually run against the live database, and they are irreversible;
* the revision adds foreign keys, which SQLite will happily accept as no-ops
  unless ``PRAGMA foreign_keys=ON`` -- a SQLite-only test would report the
  migration as verified while the constraint does nothing;
* the production duplicate rows are 9-deep for one horizon, so the collapse rule
  (keep max id) has to be checked against a real multi-row case.

These tests are skipped unless ``DATABASE_URL`` points at PostgreSQL, so the
default local run stays SQLite-only. CI's ``migrations`` job supplies one. Each
test builds its own throwaway schema and drops it afterwards; nothing here
touches a database that is not explicitly a test database.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest
import sqlalchemy as sa

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for _p in (str(BACKEND_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
PG_AVAILABLE = DATABASE_URL.startswith("postgresql")

pytestmark = pytest.mark.skipif(
    not PG_AVAILABLE,
    reason="needs a PostgreSQL DATABASE_URL; CI's migrations job provides one",
)

HEAD = "b8d3f1a9c4e2"


@pytest.fixture(scope="session")
def migration_db_url():
    """A dedicated throwaway database, created for this file alone.

    These tests drop and recreate every application table. The other
    integration tests reach the same server through ``app.database``, which
    resolves ``DATABASE_URL`` -- so sharing one database meant this file's
    ``DROP TABLE ... CASCADE`` tore down tables under their open connections and
    deadlocked the run. A separate database removes the interaction entirely.
    """
    if not PG_AVAILABLE:
        pytest.skip("no PostgreSQL DATABASE_URL")

    base = sa.make_url(DATABASE_URL)
    name = f"{base.database}_b8d3_migration"
    admin_url = base.set(database="postgres")
    admin = sa.create_engine(admin_url, poolclass=sa.pool.NullPool, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # pragma: no cover - environment problem
        pytest.skip(f"cannot create an isolated PostgreSQL database: {exc}")
    finally:
        admin.dispose()

    yield base.set(database=name).render_as_string(hide_password=False)

    cleanup = sa.create_engine(admin_url, poolclass=sa.pool.NullPool, isolation_level="AUTOCOMMIT")
    try:
        with cleanup.connect() as conn:
            # WITH (FORCE) terminates leftover backends, so a failed test cannot
            # leave the next run unable to create the database.
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        cleanup.dispose()


@pytest.fixture
def pg_engine(migration_db_url):
    engine = sa.create_engine(migration_db_url, poolclass=sa.pool.NullPool)
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment problem
        pytest.skip(f"PostgreSQL is not reachable: {exc}")
    yield engine
    engine.dispose()


def _config(engine):
    """Point Alembic at the engine's own database.

    Derived from the engine rather than read from ``DATABASE_URL`` so the
    migration can only ever run against the isolated database this file created,
    never the one the rest of the integration suite shares.
    """
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    # env.py reads `db_url` first and only falls back to the app settings -- the
    # developer's real .env database -- when it is absent.
    cfg.set_main_option("db_url", engine.url.render_as_string(hide_password=False))
    return cfg


def _reset(engine, *, duplicates: int = 3, orphans: bool = True, re_stamped: bool = True):
    """Build the production defect's shape, then stamp it at the prior revision.

    This is what the live database looked like: every table present, no
    uniqueness constraint, and repeated forecast rows per horizon because nothing
    stopped them accumulating.
    """
    with engine.begin() as conn:
        for table in (
            "forecasts", "forecast_runs", "pollution_observations",
            "weather_observations", "fire_readings", "users",
            "model_metrics", "coupling_states", "alerts", "stations",
        ):
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.execute(sa.text("DROP TABLE IF EXISTS alembic_version"))

        conn.execute(sa.text(
            "CREATE TABLE stations (id SERIAL PRIMARY KEY, name VARCHAR(100) NOT NULL UNIQUE, "
            "latitude DOUBLE PRECISION NOT NULL, longitude DOUBLE PRECISION NOT NULL, "
            "city VARCHAR(100), state VARCHAR(100))"
        ))
        conn.execute(sa.text(
            "CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(254) NOT NULL UNIQUE, "
            "name VARCHAR(120), role VARCHAR(40), password_hash VARCHAR(256), password_salt VARCHAR(64))"
        ))
        # pollution_observations WITHOUT re_stamped: the column the revision adds.
        conn.execute(sa.text(
            "CREATE TABLE pollution_observations (id SERIAL PRIMARY KEY, "
            "station_id INTEGER, timestamp TIMESTAMPTZ NOT NULL, pm25 DOUBLE PRECISION, "
            "UNIQUE (station_id, timestamp))"
        ))
        conn.execute(sa.text(
            "CREATE TABLE weather_observations (id SERIAL PRIMARY KEY, station_id INTEGER, "
            "timestamp TIMESTAMPTZ NOT NULL, temperature DOUBLE PRECISION, humidity DOUBLE PRECISION, "
            "UNIQUE (station_id, timestamp))"
        ))
        conn.execute(sa.text(
            "CREATE TABLE fire_readings (id SERIAL PRIMARY KEY, station_id INTEGER, "
            "satellite VARCHAR(40), latitude DOUBLE PRECISION NOT NULL, "
            "longitude DOUBLE PRECISION NOT NULL, acq_date TIMESTAMPTZ NOT NULL, "
            "UNIQUE (satellite, latitude, longitude, acq_date))"
        ))
        # forecasts WITHOUT UNIQUE(station_id, horizon_hours) -- the defect.
        conn.execute(sa.text(
            "CREATE TABLE forecasts (id SERIAL PRIMARY KEY, station_id INTEGER, "
            "horizon_hours INTEGER NOT NULL, generated_at TIMESTAMPTZ, "
            "pm25_pred DOUBLE PRECISION, pm10_pred DOUBLE PRECISION, aqi_pred INTEGER, "
            "aqi_category VARCHAR(40), model_used VARCHAR(60), source VARCHAR(60))"
        ))
        # Every sequence starts at 1 so "the row with the highest id" is
        # deterministic rather than dependent on what earlier tests left behind.
        for table in ("stations", "users", "pollution_observations", "weather_observations",
                      "fire_readings", "forecasts"):
            conn.execute(sa.text(f"ALTER SEQUENCE {table}_id_seq RESTART WITH 1"))

        conn.execute(sa.text("INSERT INTO stations (name, latitude, longitude, city) "
                             "VALUES ('Anand Vihar', 28.6492, 77.2918, 'Delhi NCR')"))
        if orphans:
            # A weather row pointing at a station that does not exist. The
            # revision must delete these before adding the foreign key.
            conn.execute(sa.text(
                "INSERT INTO weather_observations (station_id, timestamp, temperature) "
                "VALUES (999, now(), 20.0)"
            ))

        # `duplicates` rows for horizon 24, 3 rows for horizon 6, 1 for horizon 1.
        for horizon, count in ((24, duplicates), (6, 3), (1, 1)):
            for i in range(count):
                conn.execute(sa.text(
                    "INSERT INTO forecasts (station_id, horizon_hours, generated_at, pm25_pred) "
                    "VALUES (1, :h, now() - (INTERVAL '1 hour' * :i), :v)"
                ), {"h": horizon, "i": i, "v": 100.0 + i})

    if re_stamped:
        from alembic import command

        # Stamp at the revision immediately before the one under test, so
        # `upgrade` runs exactly this revision and nothing else.
        command.stamp(_config(engine), "a7c3e91b5d24")


def _rows(engine, sql, **params):
    with engine.connect() as conn:
        return list(conn.execute(sa.text(sql), params))


def _count(engine, table: str, where: str = "") -> int:
    """Row count as a plain int.

    Deliberately not `assert not _rows(...)`: a COUNT query always returns a row,
    so a `count(*) = 0` result would make a truthiness assertion fail while
    reporting a count of zero -- the inverse of the behaviour under test.
    """
    clause = f" WHERE {where}" if where else ""
    with engine.connect() as conn:
        return conn.execute(sa.text(f"SELECT count(*) FROM {table}{clause}")).scalar()


def _unique_forecast_keys(engine):
    return _rows(engine, "SELECT station_id, horizon_hours FROM forecasts")


# --- the migration itself ----------------------------------------------------


def test_migration_runs_and_reaches_head(pg_engine):
    """The headline check: the destructive revision applies cleanly at all.

    If this fails, nothing else in the file matters -- the deploy would abort in
    preDeploy, which is the safe outcome but a broken release.
    """
    from alembic import command

    _reset(pg_engine)
    command.upgrade(_config(pg_engine), "head")

    assert _rows(pg_engine, "SELECT version_num FROM alembic_version")[0][0] == HEAD


def test_duplicate_forecasts_are_collapsed_to_one_row_per_horizon(pg_engine):
    """The fix for the live 9-rows-at-horizon-24 accumulation.

    Every (station_id, horizon_hours) pair must end up with exactly one row, or
    the uniqueness constraint could not have been created.
    """
    from alembic import command

    _reset(pg_engine, duplicates=9)
    command.upgrade(_config(pg_engine), "head")

    rows = _unique_forecast_keys(pg_engine)
    assert len(rows) == len({tuple(r) for r in rows}), "duplicate horizons survived the migration"
    assert {r[1] for r in rows} == {1, 6, 24}, "a horizon was lost during deduplication"


def test_the_newest_duplicate_is_the_one_kept(pg_engine):
    """Keep max(id), so the freshest forecast survives and the stale one goes.

    The live database served 7-day-old rows alongside current ones; collapsing to
    the newest id is what makes the endpoint truthful after the migration.
    """
    from alembic import command

    _reset(pg_engine, duplicates=5)
    # The newest row *for horizon 24 specifically*. Taking a global max over the
    # table would compare against rows from other horizons, which are a
    # different (and wrong) question.
    newest_before = max(
        r[0] for r in _rows(
            pg_engine,
            "SELECT id FROM forecasts WHERE horizon_hours = 24 ORDER BY id",
        )
    )
    assert newest_before == 5, "fixture should create ids 1..5 for horizon 24"

    command.upgrade(_config(pg_engine), "head")

    after = _rows(pg_engine, "SELECT id FROM forecasts WHERE horizon_hours = 24")
    assert [r[0] for r in after] == [newest_before], "the wrong row was kept"
    # And it is the row with the newest generated_at, not merely the largest id.
    kept_ts = _rows(
        pg_engine, "SELECT generated_at FROM forecasts WHERE horizon_hours = 24"
    )[0][0]
    assert kept_ts is not None


def test_orphaned_observations_are_deleted_so_the_foreign_key_can_be_added(pg_engine):
    """The delete is what makes the FK possible; the FK is the real guarantee."""
    from sqlalchemy.exc import IntegrityError

    from alembic import command

    _reset(pg_engine, orphans=True)
    assert _count(pg_engine, "weather_observations", "station_id = 999") == 1

    command.upgrade(_config(pg_engine), "head")

    assert _count(pg_engine, "weather_observations", "station_id = 999") == 0
    # And the constraint is now real: PostgreSQL enforces it. This is the part a
    # SQLite-only test would have missed, since SQLite ignores FK declarations
    # unless `PRAGMA foreign_keys=ON`.
    with pytest.raises(IntegrityError):
        with pg_engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO weather_observations (station_id, timestamp, temperature) "
                "VALUES (999, now() + INTERVAL '1 day', 1.0)"
            ))


def test_migration_skips_the_alert_fk_when_that_table_is_absent(pg_engine):
    """A pre-Alembic database predating `alerts` must still be migrated.

    The revision used to DELETE FROM alerts unconditionally, so any database
    without that table aborted in preDeploy and the release could not ship at
    all -- over a table with no rows to constrain.
    """
    from alembic import command

    _reset(pg_engine)  # deliberately does not create `alerts`
    assert "alerts" not in {
        r[0] for r in _rows(pg_engine, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    }

    command.upgrade(_config(pg_engine), "head")

    assert _rows(pg_engine, "SELECT version_num FROM alembic_version")[0][0] == HEAD
    # The tables that do exist still got their constraints.
    rows = _unique_forecast_keys(pg_engine)
    assert len(rows) == len({tuple(r) for r in rows})


def test_migration_adds_the_alert_fk_when_the_table_exists(pg_engine):
    """The skip above must not become a blanket omission."""
    from sqlalchemy.exc import IntegrityError

    from alembic import command

    _reset(pg_engine)
    with pg_engine.begin() as conn:
        conn.execute(sa.text(
            "CREATE TABLE alerts (id SERIAL PRIMARY KEY, station_id INTEGER NOT NULL, "
            "created_at TIMESTAMPTZ DEFAULT now(), alert_level VARCHAR(40) NOT NULL, "
            "title VARCHAR(200) NOT NULL)"
        ))

    command.upgrade(_config(pg_engine), "head")

    with pytest.raises(IntegrityError):
        with pg_engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO alerts (station_id, alert_level, title) "
                "VALUES (999, 'warn', 'orphan')"
            ))


def test_forecast_uniqueness_is_enforced_afterwards(pg_engine):
    """The constraint has to actually reject a duplicate, not merely exist."""
    from sqlalchemy.exc import IntegrityError

    from alembic import command

    _reset(pg_engine)
    command.upgrade(_config(pg_engine), "head")

    # A horizon the fixture left free, so the failure is the constraint's doing
    # and not a collision with pre-existing seed rows.
    with pg_engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO forecasts (station_id, horizon_hours, generated_at, pm25_pred) "
            "VALUES (1, 48, now(), 1.0)"
        ))
    with pytest.raises(IntegrityError):
        with pg_engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO forecasts (station_id, horizon_hours, generated_at, pm25_pred) "
                "VALUES (1, 48, now(), 2.0)"
            ))


def test_re_stamped_column_exists_and_defaults_to_false(pg_engine):
    """Existing rows stay 0 (real ingest) and must not be backfilled as synthetic."""
    from alembic import command

    _reset(pg_engine)
    with pg_engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO pollution_observations (station_id, timestamp, pm25) "
            "VALUES (1, now(), 50.0)"
        ))

    command.upgrade(_config(pg_engine), "head")

    assert not _rows(pg_engine, "SELECT re_stamped FROM pollution_observations WHERE pm25 = 50.0")[0][0]


def test_forecast_runs_table_is_created(pg_engine):
    from alembic import command

    _reset(pg_engine)
    command.upgrade(_config(pg_engine), "head")

    names = {r[0] for r in _rows(
        pg_engine,
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
    )}
    assert "forecast_runs" in names


def test_unrelated_station_observations_survive(pg_engine):
    """Deduplication must not touch rows that are not duplicates.

    Guards against a too-broad DELETE that would quietly discard real data from
    the live database.
    """
    from alembic import command

    _reset(pg_engine, duplicates=4)
    before = _rows(pg_engine, "SELECT id, horizon_hours, pm25_pred FROM forecasts ORDER BY id")
    survivors = {tuple(r) for r in before if r[1] == 1}

    command.upgrade(_config(pg_engine), "head")

    after = {tuple(r) for r in _rows(
        pg_engine, "SELECT id, horizon_hours, pm25_pred FROM forecasts WHERE horizon_hours = 1"
    )}
    assert after == survivors


# --- migrate_safely against the same shape ----------------------------------


def _load_migrate_safely(name: str):
    """Load ``migrate_safely`` as a standalone module.

    Callers must repoint ``app.database.engine`` at the isolated database
    themselves: ``main()`` does ``from app.database import engine`` internally,
    which resolves ``DATABASE_URL`` -- the database the rest of the integration
    suite shares. Without that, assertions would inspect the isolated database
    while the script mutated the shared one.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, BACKEND_ROOT / "scripts" / "migrate_safely.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_safely_refuses_to_stamp_a_database_missing_the_constraint(pg_engine):
    """The production database's exact state must not be stamped as current.

    ``migrate_safely`` used to decide from table names alone, so this database
    was recorded as being at head while the constraint the code relies on did
    not exist.
    """
    import app.database as appdb

    module = _load_migrate_safely("migrate_safely_pg")
    original = appdb.engine
    appdb.engine = pg_engine
    try:
        _reset(pg_engine)
        # No stamp: the un-versioned production case.
        with pg_engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM alembic_version"))

        rc = module.main(["--status"])
        assert rc == 1, "an unverifiable schema must be refused, not stamped"
        # The stamp table stays empty: refusing must not half-write the database
        # it is judging.
        assert _rows(pg_engine, "SELECT count(*) FROM alembic_version")[0][0] == 0
    finally:
        appdb.engine = original


def test_migrate_safely_upgrades_a_versioned_database(pg_engine, monkeypatch):
    """A stamped database below head must be upgraded, not re-stamped."""
    import app.database as appdb

    module = _load_migrate_safely("migrate_safely_pg2")
    original = appdb.engine
    appdb.engine = pg_engine
    # ``_alembic`` shells out to `python -m alembic`, and env.py reads
    # DATABASE_URL, so the child process would otherwise migrate the shared
    # database while the assertions below read the isolated one.
    monkeypatch.setenv("DATABASE_URL", pg_engine.url.render_as_string(hide_password=False))
    try:
        _reset(pg_engine)  # stamps at a7c3e91b5d24
        assert module.main([]) == 0

        assert _rows(pg_engine, "SELECT version_num FROM alembic_version")[0][0] == HEAD
        rows = _unique_forecast_keys(pg_engine)
        assert len(rows) == len({tuple(r) for r in rows})
    finally:
        appdb.engine = original
