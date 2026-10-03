"""PostgreSQL verification of the head migration ``c5d7e9f1a3b0``.

Adds ``fire_readings.synthetic`` (NOT NULL, default false) and
``fire_readings.source`` so simulated 2023-24 stubble-fire history can never be
presented as live FIRMS data (SIH26082). Like the b8d3 suite, these tests need
a real PostgreSQL:

* NOT NULL + server_default must actually be enforced by the engine;
* the migration must leave pre-existing (legacy) rows as real — nothing here
  is allowed to guess provenance;
* downgrade must be able to remove the columns again.

Skipped unless ``DATABASE_URL`` points at PostgreSQL; CI's migrations job
provides one. Each test builds its own throwaway schema and drops it after.
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

HEAD = "c5d7e9f1a3b0"
PRIOR = "b8d3f1a9c4e2"


@pytest.fixture(scope="session")
def migration_db_url():
    if not PG_AVAILABLE:
        pytest.skip("no PostgreSQL DATABASE_URL")

    base = sa.make_url(DATABASE_URL)
    name = f"{base.database}_c5d7_provenance"
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
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("db_url", engine.url.render_as_string(hide_password=False))
    return cfg


def _reset(engine, *, legacy_rows: list[dict] | None = None):
    """A pre-c5d7 `fire_readings` table (no provenance columns), stamped below head."""
    from alembic import command

    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE IF EXISTS fire_readings CASCADE"))
        conn.execute(sa.text("DROP TABLE IF EXISTS alembic_version"))
        conn.execute(sa.text(
            "CREATE TABLE fire_readings ("
            "id SERIAL PRIMARY KEY, station_id INTEGER, satellite VARCHAR(40), "
            "latitude DOUBLE PRECISION NOT NULL, longitude DOUBLE PRECISION NOT NULL, "
            "acq_date TIMESTAMPTZ NOT NULL, confidence VARCHAR(40), frp DOUBLE PRECISION, "
            "brightness DOUBLE PRECISION, instrument VARCHAR(40), daynight VARCHAR(4))"
        ))
        for row in legacy_rows or []:
            conn.execute(sa.text(
                "INSERT INTO fire_readings "
                "(satellite, latitude, longitude, acq_date, confidence, frp) "
                "VALUES (:sat, :lat, :lon, :ts, :conf, :frp)"
            ), row)

    command.stamp(_config(engine), PRIOR)


def _rows(engine, sql, **params):
    with engine.connect() as conn:
        return list(conn.execute(sa.text(sql), params))


def _columns(engine):
    return {c["name"] for c in sa.inspect(engine).get_columns("fire_readings")}


def test_revision_chain_is_linear_with_single_head(pg_engine):
    from alembic.script import ScriptDirectory

    heads = ScriptDirectory(str(REPO_ROOT / "alembic")).get_heads()
    assert heads == [HEAD], "expected exactly one head revision, fire provenance"

    # And it really does extend the chain from the b8d3 revision.
    rev = ScriptDirectory(str(REPO_ROOT / "alembic")).get_revision(HEAD)
    assert rev.down_revision == PRIOR


def test_upgrade_adds_provenance_columns_and_defends_legacy_rows(pg_engine):
    """The irreversible part for the live DB: existing rows stay real."""
    from alembic import command

    _reset(pg_engine, legacy_rows=[{
        "sat": "SNPP", "lat": 30.5, "lon": 76.1,
        "ts": sa.text("now() - interval '1 day'"), "conf": "high", "frp": 90.0,
    }])

    command.upgrade(_config(pg_engine), "head")

    cols = _columns(pg_engine)
    assert "synthetic" in cols and "source" in cols
    assert _rows(pg_engine, "SELECT version_num FROM alembic_version")[0][0] == HEAD
    row = _rows(pg_engine, "SELECT synthetic, source FROM fire_readings WHERE latitude = 30.5")[0]
    assert row[0] is False, "legacy rows must not be invented synthetic"
    assert row[1] is None


def test_synthetic_is_not_null_and_defaults_false(pg_engine):
    """The server_default is what keeps inserts without the flag honest."""
    from sqlalchemy.exc import IntegrityError

    from alembic import command

    _reset(pg_engine)
    command.upgrade(_config(pg_engine), "head")

    with pg_engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO fire_readings (latitude, longitude, acq_date) "
            "VALUES (30.5, 76.1, now())"
        ))
    inserted = _rows(pg_engine, "SELECT synthetic, source FROM fire_readings")[0]
    assert inserted[0] is False
    assert inserted[1] is None

    # Explicit synthetic rows are allowed and stored distinctly.
    with pg_engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO fire_readings (latitude, longitude, acq_date, synthetic, source) "
            "VALUES (30.6, 76.2, now(), true, 'synthetic_sim')"
        ))
    assert _rows(pg_engine, "SELECT synthetic, source FROM fire_readings WHERE latitude = 30.6")[0] == (
        True, "synthetic_sim",
    )

    with pytest.raises(IntegrityError):
        with pg_engine.begin() as conn:
            conn.execute(sa.text(
                "INSERT INTO fire_readings (latitude, longitude, acq_date, synthetic) "
                "VALUES (30.7, 76.3, now(), NULL)"
            ))


def test_downgrade_removes_the_columns(pg_engine):
    from alembic import command

    _reset(pg_engine)
    command.upgrade(_config(pg_engine), "head")
    command.downgrade(_config(pg_engine), PRIOR)

    assert _rows(pg_engine, "SELECT version_num FROM alembic_version")[0][0] == PRIOR
    assert not {"synthetic", "source"} & _columns(pg_engine)


def test_reupgrade_after_downgrade_is_idempotent(pg_engine):
    from alembic import command

    _reset(pg_engine)
    command.upgrade(_config(pg_engine), "head")
    command.downgrade(_config(pg_engine), PRIOR)
    command.upgrade(_config(pg_engine), "head")
    assert _rows(pg_engine, "SELECT version_num FROM alembic_version")[0][0] == HEAD
