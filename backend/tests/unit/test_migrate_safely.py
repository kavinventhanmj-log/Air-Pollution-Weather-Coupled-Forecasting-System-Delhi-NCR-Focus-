"""Regression tests for the guarded migration entrypoint.

``backend/scripts/migrate_safely.py`` replaces a bare ``alembic upgrade head``
in ``render.yaml`` and ``entrypoint.sh``. The distinction matters because a
production database created by the old ``Base.metadata.create_all()`` path has
application tables but **no ``alembic_version`` table**; replaying the chain
against it fails, while stamping succeeds. These tests pin the three-way
decision so a later refactor cannot silently regress into always-upgrade.

The script shells out to alembic and imports ``app.database.engine``, so the
tests exercise its ``classify()`` decision logic against real SQLite databases
and stub only the subprocess call. A separate end-to-end check that the real
chain actually runs is out of scope here; the CI migration job covers that.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "backend" / "scripts" / "migrate_safely.py"


def _load_script():
    """Import migrate_safely.py as a module.

    It is a script rather than an importable package module, so a normal import
    would not find it. ``main()`` imports ``app.database`` lazily so that
    ``--help`` works without a database, which is why module import is safe.
    """
    spec = importlib.util.spec_from_file_location("migrate_safely", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_safely"] = module
    spec.loader.exec_module(module)
    return module


migrate_safely = _load_script()


def _engine(tmp_path: Path, name: str):
    return create_engine(f"sqlite:///{(tmp_path / name).as_posix()}")


def _table_names(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _create_complete_legacy_schema(engine) -> None:
    """Build a pre-Alembic schema that genuinely matches what head describes.

    This is the only shape the stamp branch is allowed to accept. An earlier
    version of these tests used a database holding just ``stations`` and
    ``forecast_runs`` and asserted that it was stamped -- which is precisely the
    defect the SIH26082 audit found in production, where a database missing the
    forecast uniqueness constraint was stamped as current and the service went
    on appending duplicate rows per horizon.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY, name VARCHAR(100))")
        conn.exec_driver_sql("CREATE TABLE forecast_runs (id INTEGER PRIMARY KEY, status VARCHAR(20))")
        conn.exec_driver_sql(
            "CREATE TABLE pollution_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "timestamp DATETIME, re_stamped BOOLEAN NOT NULL DEFAULT 0, "
            "UNIQUE (station_id, timestamp))"
        )
        conn.exec_driver_sql(
            "CREATE TABLE weather_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "timestamp DATETIME, UNIQUE (station_id, timestamp))"
        )
        conn.exec_driver_sql(
            "CREATE TABLE forecasts (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "horizon_hours INTEGER, UNIQUE (station_id, horizon_hours))"
        )


def test_script_is_importable_without_a_database(monkeypatch):
    """Module import must not require a reachable database.

    ``render.yaml`` runs this on every deploy; an import-time connection would
    make ``--help`` and ``--status`` fail in exactly the situation where the
    operator needs them.
    """
    assert callable(migrate_safely.main)
    assert callable(migrate_safely.classify)


def test_empty_database_is_classified_empty(tmp_path):
    """No tables at all means the chain should be replayed normally."""
    engine = _engine(tmp_path, "empty.db")
    state, tables = migrate_safely.classify(engine)
    assert state == "empty"
    assert tables == set()


def test_versioned_database_is_classified_versioned(tmp_path):
    """An already-stamped database must upgrade, never re-stamp."""
    engine = _engine(tmp_path, "versioned.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.exec_driver_sql("INSERT INTO alembic_version VALUES ('b8d3f1a9c4e2')")
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    state, _ = migrate_safely.classify(engine)
    assert state == "versioned"


def test_legacy_database_without_alembic_version_is_classified_unversioned(tmp_path):
    """The production case this script exists for.

    ``create_all()`` produced the tables but no ``alembic_version``. Classifying
    this as ``versioned`` would be impossible, and treating it as ``empty``
    would replay the chain and fail.
    """
    engine = _engine(tmp_path, "legacy.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY, name VARCHAR(100))")
        conn.exec_driver_sql(
            "CREATE TABLE forecast_runs (id INTEGER PRIMARY KEY, status VARCHAR(20))"
        )

    state, tables = migrate_safely.classify(engine)
    assert state == "unversioned"
    assert "stations" in tables
    assert "alembic_version" not in tables


def test_alembic_version_row_missing_but_table_present_is_unversioned(tmp_path):
    """An empty ``alembic_version`` table must not be mistaken for a valid stamp.

    ``_current_revision`` returns ``None`` when the table exists but holds no
    row, and the classification must treat that as unversioned rather than
    upgrading a database whose real revision is unknown.
    """
    engine = _engine(tmp_path, "emptyversion.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")
        conn.exec_driver_sql("CREATE TABLE forecast_runs (id INTEGER PRIMARY KEY)")

    state, _ = migrate_safely.classify(engine)
    assert state == "unversioned"


def test_core_table_is_the_detection_signal(tmp_path):
    """A database with unrelated tables is still ``empty``.

    Only ``stations`` distinguishes "pre-Alembic app schema" from "nothing to
    migrate"; a stray table from another tool must not trigger the stamp path.
    """
    engine = _engine(tmp_path, "other.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE someone_elses_table (id INTEGER PRIMARY KEY)")

    state, _ = migrate_safely.classify(engine)
    assert state == "empty"


def test_recent_table_guard_detects_incomplete_legacy_schema(tmp_path):
    """A legacy database missing new tables must not be stamped at head.

    Stamping would claim ``forecast_runs`` exists when it does not, and the next
    code path touching it would fail at runtime instead of at deploy time. The
    script must refuse and point at ``--force-upgrade``.
    """
    engine = _engine(tmp_path, "incomplete.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    state, tables = migrate_safely.classify(engine)
    assert state == "unversioned"
    missing = [t for t in migrate_safely.RECENT_TABLES if t not in tables]
    assert missing == ["forecast_runs"]


def test_current_revision_reads_the_stamped_value(tmp_path):
    """Sanity-check the revision read used by ``classify``."""
    engine = _engine(tmp_path, "rev.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.exec_driver_sql("INSERT INTO alembic_version VALUES ('a7c3e91b5d24')")

    assert migrate_safely._current_revision(engine) == "a7c3e91b5d24"


def test_current_revision_is_none_for_unversioned_database(tmp_path):
    """The unversioned case must return ``None`` rather than raising."""
    engine = _engine(tmp_path, "norev.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    assert migrate_safely._current_revision(engine) is None


def test_status_mode_writes_nothing(tmp_path, monkeypatch):
    """``--status`` is documented as read-only; prove it makes no changes."""
    engine = _engine(tmp_path, "status.db")
    _create_complete_legacy_schema(engine)

    called = []
    monkeypatch.setattr(migrate_safely, "_alembic", lambda *a: called.append(a) or 0)
    monkeypatch.setattr("app.database.engine", engine)

    rc = migrate_safely.main(["--status"])

    assert rc == 0
    assert called == [], "--status must not run any alembic subcommand"

    con = sqlite3.connect(tmp_path / "status.db")
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    con.close()
    assert "alembic_version" not in tables


def test_legacy_database_takes_the_stamp_branch(tmp_path, monkeypatch):
    """Unversioned + genuinely complete schema must stamp, not upgrade."""
    engine = _engine(tmp_path, "stamp.db")
    _create_complete_legacy_schema(engine)

    called = []
    monkeypatch.setattr(migrate_safely, "_alembic", lambda *a: called.append(a) or 0)
    monkeypatch.setattr("app.database.engine", engine)

    rc = migrate_safely.main([])

    assert rc == 0
    assert called == [("stamp", "head")], "legacy database must be stamped, not upgraded"


def test_legacy_database_missing_the_uniqueness_constraint_is_refused(tmp_path, monkeypatch):
    """Tables present is not enough; the constraint is the thing that matters.

    This is the live production shape: every application table existed, so the
    old name-based check stamped it as current, and the service then wrote a new
    forecast row for every horizon on every run without the database object that
    would have stopped it.
    """
    engine = _engine(tmp_path, "nounique.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY, name VARCHAR(100))")
        conn.exec_driver_sql("CREATE TABLE forecast_runs (id INTEGER PRIMARY KEY, status VARCHAR(20))")
        conn.exec_driver_sql(
            "CREATE TABLE pollution_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "timestamp DATETIME, re_stamped BOOLEAN NOT NULL DEFAULT 0, "
            "UNIQUE (station_id, timestamp))"
        )
        conn.exec_driver_sql(
            "CREATE TABLE weather_observations (id INTEGER PRIMARY KEY, station_id INTEGER, "
            "timestamp DATETIME, UNIQUE (station_id, timestamp))"
        )
        # forecasts deliberately WITHOUT UNIQUE(station_id, horizon_hours)
        conn.exec_driver_sql(
            "CREATE TABLE forecasts (id INTEGER PRIMARY KEY, station_id INTEGER, horizon_hours INTEGER)"
        )

    called = []
    monkeypatch.setattr(migrate_safely, "_alembic", lambda *a: called.append(a) or 0)
    monkeypatch.setattr("app.database.engine", engine)

    rc = migrate_safely.main([])

    assert rc == 1
    assert called == [], "an unverifiable schema must not be stamped"
    con = sqlite3.connect(tmp_path / "nounique.db")
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    con.close()
    assert "alembic_version" not in tables


def test_versioned_database_takes_the_upgrade_branch(tmp_path, monkeypatch):
    """An already-versioned database must upgrade."""
    engine = _engine(tmp_path, "upgrade.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.exec_driver_sql("INSERT INTO alembic_version VALUES ('b8d3f1a9c4e2')")
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    called = []
    monkeypatch.setattr(migrate_safely, "_alembic", lambda *a: called.append(a) or 0)
    monkeypatch.setattr("app.database.engine", engine)

    rc = migrate_safely.main([])

    assert rc == 0
    assert called == [("upgrade", "head")]


def test_incomplete_legacy_database_refuses_and_explains(tmp_path, monkeypatch, capsys):
    """Missing new tables must abort with guidance, not stamp a false claim."""
    engine = _engine(tmp_path, "refuse.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    called = []
    monkeypatch.setattr(migrate_safely, "_alembic", lambda *a: called.append(a) or 0)
    monkeypatch.setattr("app.database.engine", engine)

    rc = migrate_safely.main([])

    assert rc == 1
    assert called == [], "must not run any alembic subcommand when refusing"
    out = capsys.readouterr().out
    assert "forecast_runs" in out
    assert "--force-upgrade" in out


def test_force_upgrade_overrides_the_refusal(tmp_path, monkeypatch):
    """``--force-upgrade`` is the documented escape hatch."""
    engine = _engine(tmp_path, "force.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    called = []
    monkeypatch.setattr(migrate_safely, "_alembic", lambda *a: called.append(a) or 0)
    monkeypatch.setattr("app.database.engine", engine)

    rc = migrate_safely.main(["--force-upgrade"])

    assert rc == 0
    assert called == [("upgrade", "head")]


def test_unreachable_database_fails_without_writing(monkeypatch, capsys):
    """A connectivity failure must abort, never attempt a migration."""
    class Boom:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("app.database.engine", Boom())

    rc = migrate_safely.main([])

    assert rc == 1
    out = capsys.readouterr().out
    assert "could not reach the database" in out


def test_render_and_docker_use_the_guarded_script():
    """The deploy configs must not regress to a bare ``alembic upgrade head``.

    A bare upgrade against the legacy production database is precisely the
    failure this script was added to prevent, and it is invisible in unit tests
    that never read the YAML.
    """
    render = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "preDeployCommand: python backend/scripts/migrate_safely.py" in render
    assert "preDeployCommand: python -m alembic upgrade head" not in render

    entrypoint = (REPO_ROOT / "backend" / "scripts" / "entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "backend/scripts/migrate_safely.py" in entrypoint
    assert "python -m alembic upgrade head" not in entrypoint


def test_dockerfile_ships_the_script():
    """The image only copies selected files; the script must be among them.

    ``entrypoint.sh`` invokes ``backend/scripts/migrate_safely.py`` but the
    Dockerfile copies ``backend/app`` and only ``entrypoint.sh`` from scripts.
    Without this COPY the container fails to start on migrations.
    """
    dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    assert "backend/scripts/migrate_safely.py" in dockerfile


@pytest.mark.parametrize("revision", ["0964e5b227e3", "b8d3f1a9c4e2"])
def test_classify_handles_arbitrary_revision_strings(tmp_path, revision):
    """Revision length varies across the chain; the read must not assume one.

    ``alembic_version.version_num`` is ``VARCHAR(32)`` but older rows are
    shorter, so the probe must not compare against a fixed-width value.
    """
    engine = _engine(tmp_path, f"rev-{revision}.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.exec_driver_sql(
            "INSERT INTO alembic_version VALUES (?)", (revision,)
        )
        conn.exec_driver_sql("CREATE TABLE stations (id INTEGER PRIMARY KEY)")

    assert migrate_safely._current_revision(engine) == revision
    state, _ = migrate_safely.classify(engine)
    assert state == "versioned"
