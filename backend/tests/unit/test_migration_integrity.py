"""Migration integrity: the C1 defect.

Reproduces the failure mode found in production:

* startup caught a migration exception, logged it, then ran ``create_all()``,
  which recreates tables but never writes ``alembic_version``. The service came
  up "healthy" against a schema missing the constraints the code relies on;
* ``run_migrations()`` returned successfully without checking the result, so a
  database stamped below head passed;
* ``migrate_safely.py`` decided to ``stamp head`` on the strength of a couple of
  table names, so a database holding every table but missing the uniqueness
  constraint was marked as current.

The tests below pin the corrected behaviour. Nothing here touches a production
database: every case builds a throwaway SQLite file in a temp directory.
"""

import pathlib
import sys
import tempfile

import pytest
import sqlalchemy as sa

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for _p in (str(BACKEND_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

import migrate_safely as ms  # noqa: E402


@pytest.fixture
def fresh_db():
    """An empty SQLite database, disposed and removed after the test."""
    path = pathlib.Path(tempfile.mkdtemp()) / "mig.db"
    engine = sa.create_engine(f"sqlite:///{path}")
    yield engine
    engine.dispose()


def _create_legacy_schema(engine, *, with_forecast_runs=True, with_unique=True,
                          with_re_stamped=True):
    """Build the shape a pre-Alembic ``create_all()`` database would have.

    ``with_unique=False`` models the production defect: every table present, the
    ``(station_id, horizon_hours)`` constraint absent.
    """
    with engine.begin() as c:
        c.execute(sa.text("CREATE TABLE stations (id INTEGER PRIMARY KEY, name VARCHAR)"))
        c.execute(
            sa.text(
                "CREATE TABLE weather_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
                "timestamp DATETIME, UNIQUE (station_id, timestamp))"
            )
        )
        c.execute(
            sa.text(
                "CREATE TABLE pollution_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
                "timestamp DATETIME, pm25 FLOAT"
                + (", re_stamped BOOLEAN NOT NULL DEFAULT 0" if with_re_stamped else "")
                + ", UNIQUE (station_id, timestamp))"
            )
        )
        unique = (
            ", UNIQUE (station_id, horizon_hours)"
            if with_unique
            else ""
        )
        c.execute(
            sa.text(
                "CREATE TABLE forecasts (id INTEGER PRIMARY KEY, station_id INTEGER, "
                f"horizon_hours INTEGER{unique})"
            )
        )
        if with_forecast_runs:
            c.execute(
                sa.text(
                    "CREATE TABLE forecast_runs (id INTEGER PRIMARY KEY, station_id INTEGER, "
                    "station_name VARCHAR, created_at DATETIME, status VARCHAR)"
                )
            )


# --- migrate_safely: classification ----------------------------------------


def test_empty_database_classifies_as_empty(fresh_db):
    state, tables = ms.classify(fresh_db)
    assert state == "empty"
    assert tables == set()


def test_legacy_database_without_alembic_version_classifies_unversioned(fresh_db):
    _create_legacy_schema(fresh_db)
    state, _ = ms.classify(fresh_db)
    assert state == "unversioned"


def test_stamped_database_classifies_versioned(fresh_db):
    _create_legacy_schema(fresh_db)
    with fresh_db.begin() as c:
        c.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR)"))
        c.execute(sa.text("INSERT INTO alembic_version VALUES ('b8d3f1a9c4e2')"))
    state, _ = ms.classify(fresh_db)
    assert state == "versioned"


def test_empty_alembic_version_table_is_not_treated_as_versioned(fresh_db):
    """An empty stamp table means "unknown", not "current".

    Reporting ``None`` here is what lets the caller decide; silently treating it
    as unversioned-and-therefore-stampable would hide the gap.
    """
    _create_legacy_schema(fresh_db)
    with fresh_db.begin() as c:
        c.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR)"))
    assert ms._current_revision(fresh_db) is None
    state, _ = ms.classify(fresh_db)
    assert state == "unversioned"


# --- migrate_safely: schema verification (the core fix) --------------------


def test_missing_forecast_runs_table_is_detected(fresh_db):
    """A missing recent table is caught by the RECENT_TABLES check."""
    _create_legacy_schema(fresh_db, with_forecast_runs=False)
    tables = set(sa.inspect(fresh_db).get_table_names())
    missing_tables = [t for t in ms.RECENT_TABLES if t not in tables]
    assert "forecast_runs" in missing_tables


def test_missing_forecast_uniqueness_is_detected(fresh_db):
    """The production defect: tables all present, constraint absent.

    Table-presence heuristics pass this database; the shape check must not.
    """
    _create_legacy_schema(fresh_db, with_unique=False)
    inspector = sa.inspect(fresh_db)
    missing = ms.missing_schema_objects(set(inspector.get_table_names()), inspector)
    assert any("uq_forecast_station_horizon" in m or "UNIQUE(station_id, horizon_hours)" in m for m in missing)


def test_missing_re_stamped_column_is_detected(fresh_db):
    _create_legacy_schema(fresh_db, with_re_stamped=False)
    inspector = sa.inspect(fresh_db)
    missing = ms.missing_schema_objects(set(inspector.get_table_names()), inspector)
    assert any("re_stamped" in m for m in missing)


def test_complete_legacy_schema_reports_nothing_missing(fresh_db):
    _create_legacy_schema(fresh_db)
    inspector = sa.inspect(fresh_db)
    assert ms.missing_schema_objects(set(inspector.get_table_names()), inspector) == []


def test_unique_index_satisfies_the_requirement(fresh_db):
    """A unique index enforces the same integrity as a named constraint."""
    _create_legacy_schema(fresh_db, with_unique=False)
    with fresh_db.begin() as c:
        c.execute(
            sa.text("CREATE UNIQUE INDEX uq_forecast_station_horizon ON forecasts (station_id, horizon_hours)")
        )
    inspector = sa.inspect(fresh_db)
    assert ms.missing_schema_objects(set(inspector.get_table_names()), inspector) == []


def test_schema_check_is_read_only(fresh_db):
    """The verification must not mutate the database it inspects.

    Notably it must not create ``alembic_version``: writing that table is the
    stamp, and a read-only probe that half-stamps the database it is judging
    would destroy the evidence the operator needs.
    """
    _create_legacy_schema(fresh_db)
    inspector = sa.inspect(fresh_db)
    before = set(inspector.get_table_names())
    ms.missing_schema_objects(before, inspector)
    after = set(sa.inspect(fresh_db).get_table_names())
    assert before == after
    assert "alembic_version" not in after
    with fresh_db.connect() as c:
        assert c.execute(sa.text("SELECT count(*) FROM forecasts")).scalar() == 0


# --- migrate_safely: refuses to stamp an incomplete schema ------------------


def test_incomplete_legacy_schema_is_refused_not_stamped(fresh_db, monkeypatch, capsys):
    """The end-to-end guard: no stamp, exit 1, explanation printed."""
    _create_legacy_schema(fresh_db, with_unique=False)
    monkeypatch.setattr(ms, "_alembic", lambda *a: pytest.fail("must not run alembic"))
    monkeypatch.setattr(ms, "main", ms.main)

    import app.database as appdb

    monkeypatch.setattr(appdb, "engine", fresh_db)
    monkeypatch.setitem(sys.modules, "app.database", appdb)

    exit_code = ms.main([])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "alembic_version" in out
    # And critically: nothing was written.
    assert "alembic_version" not in set(sa.inspect(fresh_db).get_table_names())


def test_status_mode_refuses_an_incomplete_legacy_database(fresh_db, monkeypatch, capsys):
    """``--status`` must not become a back door for an unsafe stamp.

    The refusal is reported even in status mode, because a status report that
    quietly claimed "nothing to do" on a schema missing its uniqueness
    constraint would tell the operator the wrong thing.
    """
    _create_legacy_schema(fresh_db, with_unique=False)
    monkeypatch.setattr(ms, "_alembic", lambda *a: pytest.fail("status must not run alembic"))

    import app.database as appdb

    monkeypatch.setattr(appdb, "engine", fresh_db)
    monkeypatch.setitem(sys.modules, "app.database", appdb)

    exit_code = ms.main(["--status"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "unique constraint absent" in out
    assert "alembic_version" not in set(sa.inspect(fresh_db).get_table_names())


def test_status_mode_reports_a_complete_legacy_database_without_writing(fresh_db, monkeypatch, capsys):
    """A genuinely complete legacy database is reported, and left alone."""
    _create_legacy_schema(fresh_db)
    monkeypatch.setattr(ms, "_alembic", lambda *a: pytest.fail("status must not run alembic"))

    import app.database as appdb

    monkeypatch.setattr(appdb, "engine", fresh_db)
    monkeypatch.setitem(sys.modules, "app.database", appdb)

    exit_code = ms.main(["--status"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no changes written" in out
    assert "alembic_version" not in set(sa.inspect(fresh_db).get_table_names())


# --- run_migrations: verifies head after upgrading ------------------------


def test_run_migrations_rejects_a_database_stamped_below_head(fresh_db, monkeypatch):
    """A clean upgrade that does not reach head must not be reported as success.

    This is the state the live deployment was found in: a stamp below head meant
    the uniqueness constraint the code depends on did not exist, while the
    service still started and served.
    """
    import app.database as appdb

    # A PostgreSQL URL so the SQLite short-circuit does not skip the
    # verification. `command.upgrade` is patched out below, so nothing here
    # connects anywhere; the SQLite engine is only the state being inspected.
    monkeypatch.setattr(
        appdb, "settings", type("S", (), {"database_url": "postgresql://u:p@localhost:5432/nodb"})()
    )
    monkeypatch.setattr(appdb, "engine", fresh_db)
    monkeypatch.setattr(appdb, "_alembic_heads", lambda: {"b8d3f1a9c4e2"})

    _create_legacy_schema(fresh_db)
    with fresh_db.begin() as c:
        c.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR)"))
        c.execute(sa.text("INSERT INTO alembic_version VALUES ('5c1b7d9a2f6e')"))

    # Simulate `command.upgrade` doing nothing (the real failure mode: the chain
    # is already "applied" from the stamp's point of view, or a no-op).
    import alembic.command

    monkeypatch.setattr(alembic.command, "upgrade", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="behind the code"):
        appdb.run_migrations()


def test_run_migrations_rejects_a_missing_alembic_version_table(fresh_db, monkeypatch):
    import app.database as appdb

    # A PostgreSQL URL so the SQLite short-circuit does not skip the
    # verification. `command.upgrade` is patched out below, so nothing here
    # connects anywhere; the SQLite engine is only the state being inspected.
    monkeypatch.setattr(
        appdb, "settings", type("S", (), {"database_url": "postgresql://u:p@localhost:5432/nodb"})()
    )
    monkeypatch.setattr(appdb, "engine", fresh_db)
    monkeypatch.setattr(appdb, "_alembic_heads", lambda: {"b8d3f1a9c4e2"})
    _create_legacy_schema(fresh_db)  # no alembic_version table

    import alembic.command

    monkeypatch.setattr(alembic.command, "upgrade", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="no alembic_version"):
        appdb.run_migrations()


def test_run_migrations_rejects_an_empty_alembic_version_table(fresh_db, monkeypatch):
    import app.database as appdb

    # A PostgreSQL URL so the SQLite short-circuit does not skip the
    # verification. `command.upgrade` is patched out below, so nothing here
    # connects anywhere; the SQLite engine is only the state being inspected.
    monkeypatch.setattr(
        appdb, "settings", type("S", (), {"database_url": "postgresql://u:p@localhost:5432/nodb"})()
    )
    monkeypatch.setattr(appdb, "engine", fresh_db)
    monkeypatch.setattr(appdb, "_alembic_heads", lambda: {"b8d3f1a9c4e2"})
    _create_legacy_schema(fresh_db)
    with fresh_db.begin() as c:
        c.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR)"))
        c.execute(sa.text("INSERT INTO alembic_version VALUES (NULL)"))

    import alembic.command

    monkeypatch.setattr(alembic.command, "upgrade", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="alembic_version is empty"):
        appdb.run_migrations()


def test_alembic_heads_helper_finds_the_real_head():
    """The head check must read the actual chain, not a hard-coded revision."""
    import app.database as appdb

    heads = appdb._alembic_heads()
    assert isinstance(heads, set)
    assert heads, "the repository must expose at least one migration head"
    assert all(isinstance(h, str) for h in heads)


# --- forecast uniqueness contract ------------------------------------------


def test_orm_declares_forecast_uniqueness():
    """The application contract the migration must enforce.

    Guards against the constraint being dropped from the ORM while the
    migration still claims to add it.
    """
    from app.models.db_models import Forecast

    constraints = {tuple(c.name for c in table.constraints) for table in [Forecast.__table__]}
    names = set()
    for table in [Forecast.__table__]:
        for c in table.constraints:
            if getattr(c, "columns", None) is not None:
                names.add(tuple(col.name for col in c.columns))
    assert ("station_id", "horizon_hours") in names
    assert constraints  # table has declared constraints


def test_duplicate_forecasts_would_violate_the_constraint(fresh_db):
    """Sanity check on the intent: the second row for a horizon must fail."""
    _create_legacy_schema(fresh_db, with_unique=True)
    with pytest.raises(sa.exc.IntegrityError):
        with fresh_db.begin() as c:
            c.execute(sa.text("INSERT INTO forecasts (station_id, horizon_hours) VALUES (1, 24)"))
            c.execute(sa.text("INSERT INTO forecasts (station_id, horizon_hours) VALUES (1, 24)"))


def test_duplicate_forecasts_are_allowed_without_the_constraint(fresh_db):
    """Confirms the defect is real: without it, duplicates accumulate silently.

    This is the shape the production database was serving.
    """
    _create_legacy_schema(fresh_db, with_unique=False)
    with fresh_db.begin() as c:
        c.execute(sa.text("INSERT INTO forecasts (station_id, horizon_hours) VALUES (1, 24)"))
        c.execute(sa.text("INSERT INTO forecasts (station_id, horizon_hours) VALUES (1, 24)"))
    with fresh_db.connect() as c:
        assert c.execute(sa.text("SELECT count(*) FROM forecasts")).scalar() == 2


# --- the head revision, actually executed -----------------------------------
#
# Everything above tests classification and verification. Nothing invoked
# `upgrade()` on the revision itself, so its SQLite branch -- which is a
# different code path from PostgreSQL's, using batch_alter_table table rebuilds
# instead of ALTER TABLE -- had never been run by any test. A break there would
# only surface in a developer's local environment.


def _alembic_config(db_url):
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    # env.py reads `db_url` first and only falls back to the app settings -- i.e.
    # the developer's real .env database -- when it is absent. Setting
    # `sqlalchemy.url` instead would silently target production.
    cfg.set_main_option("db_url", db_url)
    return cfg


def _run_revision(engine, *, with_alerts):
    """Stamp the prior revision and upgrade, on a throwaway SQLite file."""
    from alembic import command

    with engine.begin() as c:
        c.execute(sa.text("CREATE TABLE stations (id INTEGER PRIMARY KEY, name VARCHAR)"))
        c.execute(sa.text(
            "CREATE TABLE weather_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "timestamp DATETIME, UNIQUE (station_id, timestamp))"
        ))
        c.execute(sa.text(
            "CREATE TABLE pollution_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "timestamp DATETIME, pm25 FLOAT, UNIQUE (station_id, timestamp))"
        ))
        c.execute(sa.text(
            "CREATE TABLE forecasts (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "horizon_hours INTEGER NOT NULL, generated_at DATETIME, pm25_pred FLOAT)"
        ))
        if with_alerts:
            c.execute(sa.text(
                "CREATE TABLE alerts (id INTEGER PRIMARY KEY, station_id INTEGER NOT NULL, "
                "alert_level VARCHAR, title VARCHAR)"
            ))
        c.execute(sa.text("INSERT INTO stations (id, name) VALUES (1, 'Anand Vihar')"))
        c.execute(sa.text(
            "INSERT INTO weather_observations (station_id, timestamp) VALUES (999, CURRENT_TIMESTAMP)"
        ))
        for horizon, count in ((24, 5), (6, 3), (1, 1)):
            for i in range(count):
                c.execute(
                    sa.text(
                        "INSERT INTO forecasts (station_id, horizon_hours, generated_at, pm25_pred) "
                        "VALUES (1, :h, CURRENT_TIMESTAMP, :v)"
                    ),
                    {"h": horizon, "v": 100.0 + i},
                )

    cfg = _alembic_config(f"sqlite:///{engine.url.database}")
    command.stamp(cfg, "a7c3e91b5d24")
    command.upgrade(cfg, "head")
    return cfg


def _count(engine, table, where=""):
    clause = f" WHERE {where}" if where else ""
    with engine.connect() as c:
        return c.execute(sa.text(f"SELECT count(*) FROM {table}{clause}")).scalar()


def test_head_revision_upgrades_cleanly_on_sqlite(fresh_db):
    """The SQLite branch of b8d3f1a9c4e2 must work, not just PostgreSQL's."""
    _run_revision(fresh_db, with_alerts=True)

    with fresh_db.connect() as c:
        assert c.execute(sa.text("SELECT version_num FROM alembic_version")).scalar() == (
            "b8d3f1a9c4e2"
        )
        assert "re_stamped" in {
            r[1] for r in c.execute(sa.text("PRAGMA table_info(pollution_observations)"))
        }
    assert "forecast_runs" in sa.inspect(fresh_db).get_table_names()


def test_head_revision_deduplicates_and_drops_orphans_on_sqlite(fresh_db):
    """Same data repairs the PostgreSQL tests check must hold on SQLite."""
    _run_revision(fresh_db, with_alerts=True)

    assert _count(fresh_db, "weather_observations", "station_id = 999") == 0
    with fresh_db.connect() as c:
        rows = c.execute(
            sa.text("SELECT horizon_hours, count(*) FROM forecasts GROUP BY horizon_hours")
        ).fetchall()
    assert dict(rows) == {1: 1, 6: 1, 24: 1}, "duplicates survived the migration"


def test_head_revision_downgrades_on_sqlite(fresh_db):
    """Downgrade must invert the upgrade, or a rollback is a dead end."""
    from alembic import command

    cfg = _run_revision(fresh_db, with_alerts=True)
    command.downgrade(cfg, "a7c3e91b5d24")

    with fresh_db.connect() as c:
        assert c.execute(sa.text("SELECT version_num FROM alembic_version")).scalar() == (
            "a7c3e91b5d24"
        )
        assert "re_stamped" not in {
            r[1] for r in c.execute(sa.text("PRAGMA table_info(pollution_observations)"))
        }
    assert "forecast_runs" not in sa.inspect(fresh_db).get_table_names()


def test_head_revision_skips_a_missing_alerts_table_on_sqlite(fresh_db):
    """The pre-Alembic database shape that used to abort the whole deploy.

    `alerts` was added to the schema after some deployments were already
    serving, so a real database does not necessarily have it. The revision
    deleted from it unconditionally, which failed preDeploy and blocked the
    release entirely.
    """
    _run_revision(fresh_db, with_alerts=False)

    with fresh_db.connect() as c:
        assert c.execute(sa.text("SELECT version_num FROM alembic_version")).scalar() == (
            "b8d3f1a9c4e2"
        )
    assert "alerts" not in sa.inspect(fresh_db).get_table_names()
    # The tables that do exist were still repaired.
    assert _count(fresh_db, "weather_observations", "station_id = 999") == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
