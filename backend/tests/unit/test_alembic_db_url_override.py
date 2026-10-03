"""Tests for the Alembic database-URL override (``-x db_url=...``).

Regression guard for a real incident: ``alembic/env.py`` read the override with
``config.get_main_option("db_url")``, but ``-x`` is exposed through
``Config.cmd_opts.x`` / ``EnvironmentContext.get_x_argument``. The override was
therefore ignored and ``alembic upgrade head`` silently ran against the
``DATABASE_URL`` fallback -- a live database at the time.

Two layers are covered:

* ``resolve_db_url`` (pure logic) -- precedence, validation, and the safety
  guarantee that an explicit-but-broken override raises instead of falling
  through;
* the Alembic CLI end to end, with ``DATABASE_URL`` pointed at an intentionally
  unreachable PostgreSQL host, so a fall-through would fail loudly. Every test
  uses a disposable SQLite file under pytest's ``tmp_path``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.scripts.alembic_db_url import (
    DbUrlResolutionError,
    parse_x_arguments,
    resolve_db_url,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: A syntactically valid but unreachable PostgreSQL URL. Port 1 on loopback is
#: refused immediately, so any accidental fall-through fails fast instead of
#: touching a real database. Never points at Neon.
UNREACHABLE_PG = "postgresql://nobody:nobody@127.0.0.1:1/never_used"

NEW_HEAD = "c5d7e9f1a3b0"


def _fail_settings() -> str:
    raise AssertionError("settings_loader must not be consulted when overridden")


# --------------------------------------------------------------------------- #
# Unit: parse_x_arguments
# --------------------------------------------------------------------------- #

def test_parse_x_arguments_matches_alembic_semantics():
    assert parse_x_arguments(["db_url=sqlite:///a.db", "other=1"]) == {
        "db_url": "sqlite:///a.db",
        "other": "1",
    }
    # No "=" -> empty string value (Alembic 1.13.1+ behaviour).
    assert parse_x_arguments(["db_url"]) == {"db_url": ""}


# --------------------------------------------------------------------------- #
# Unit: resolve_db_url precedence and safety
# --------------------------------------------------------------------------- #

def test_x_argument_wins_over_database_url_env(tmp_path):
    target = f"sqlite:///{tmp_path/'chosen.db'}"
    resolved = resolve_db_url(
        x_arguments=["db_url=" + target],
        main_option=None,
        environ={"DATABASE_URL": UNREACHABLE_PG},
        settings_loader=_fail_settings,
        log=lambda _: None,
    )
    assert resolved == target
    assert resolved != UNREACHABLE_PG


def test_resolver_receives_already_parsed_mapping(tmp_path):
    target = f"sqlite:///{tmp_path/'mapped.db'}"
    resolved = resolve_db_url(
        x_arguments={"db_url": target},
        main_option=None,
        environ={"DATABASE_URL": UNREACHABLE_PG},
        settings_loader=_fail_settings,
        log=lambda _: None,
    )
    assert resolved == target


def test_ini_main_option_used_when_no_x_argument(tmp_path):
    target = f"sqlite:///{tmp_path/'ini.db'}"
    resolved = resolve_db_url(
        x_arguments=None,
        main_option=target,
        environ={"DATABASE_URL": UNREACHABLE_PG},
        settings_loader=_fail_settings,
        log=lambda _: None,
    )
    assert resolved == target


def test_falls_back_to_env_then_settings():
    assert (
        resolve_db_url(
            x_arguments=None,
            main_option=None,
            environ={"DATABASE_URL": "sqlite:///env.db"},
            settings_loader=_fail_settings,
            log=lambda _: None,
        )
        == "sqlite:///env.db"
    )
    assert (
        resolve_db_url(
            x_arguments=None,
            main_option=None,
            environ={},
            settings_loader=lambda: "sqlite:///settings.db",
            log=lambda _: None,
        )
        == "sqlite:///settings.db"
    )


def test_blank_explicit_override_raises_instead_of_falling_back():
    """(Safety) ``-x db_url=`` must never silently migrate DATABASE_URL."""
    with pytest.raises(DbUrlResolutionError, match="explicit database URL"):
        resolve_db_url(
            x_arguments=["db_url="],
            main_option=None,
            environ={"DATABASE_URL": UNREACHABLE_PG},
            settings_loader=_fail_settings,
            log=lambda _: None,
        )


def test_invalid_explicit_override_raises_instead_of_falling_back():
    with pytest.raises(DbUrlResolutionError, match="not a valid SQLAlchemy URL"):
        resolve_db_url(
            x_arguments=["db_url=definitely not a url"],
            main_option=None,
            environ={"DATABASE_URL": UNREACHABLE_PG},
            settings_loader=_fail_settings,
            log=lambda _: None,
        )


def test_invalid_explicit_override_has_no_exception_cause():
    """The raw ArgumentError must not be chained onto the operator-facing error."""
    import traceback

    secret = "sup3rs3cret-do-not-log"
    with pytest.raises(DbUrlResolutionError) as exc:
        resolve_db_url(
            x_arguments=[f"db_url=not a real url {secret}"],
            main_option=None,
            environ={"DATABASE_URL": UNREACHABLE_PG},
            settings_loader=_fail_settings,
            log=lambda _: None,
        )
    assert exc.value.__cause__ is None
    rendered = "".join(
        traceback.format_exception(type(exc.value), exc.value, exc.value.__traceback__)
    )
    assert secret not in rendered


def test_password_is_masked_in_log(capsys):
    resolve_db_url(
        x_arguments=None,
        main_option=None,
        environ={"DATABASE_URL": "postgresql://user:supersecret@example.invalid/db"},
        settings_loader=_fail_settings,
    )
    out = capsys.readouterr().out
    assert "supersecret" not in out
    assert "example.invalid" in out


def test_logs_the_selected_target(tmp_path):
    lines: list[str] = []
    target = f"sqlite:///{tmp_path/'logged.db'}"
    resolve_db_url(
        x_arguments=["db_url=" + target],
        main_option=None,
        environ={"DATABASE_URL": UNREACHABLE_PG},
        settings_loader=_fail_settings,
        log=lines.append,
    )
    assert any(target in line for line in lines)


# --------------------------------------------------------------------------- #
# End to end: the actual Alembic CLI must honour -x db_url
# --------------------------------------------------------------------------- #

def _run_alembic(args: list[str], *, database_url: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": database_url}
    # Ensure the repo-root `app/` Render shim cannot shadow backend/app.
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _sqlite_target(tmp_path, name="override.db") -> tuple[str, Path]:
    path = tmp_path / name
    return f"sqlite:///{path.as_posix()}", path


def test_cli_x_override_selects_sqlite_not_database_url(tmp_path):
    """The CLI must connect to ``-x db_url`` even when DATABASE_URL is elsewhere.

    DATABASE_URL points at an unreachable PostgreSQL host: if the override were
    ignored (the original bug) this would fail to connect instead of succeeding.
    """
    target, path = _sqlite_target(tmp_path)
    result = _run_alembic(
        ["-x", f"db_url={target}", "current"], database_url=UNREACHABLE_PG
    )
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1:1" not in result.stderr
    assert path.exists(), "the override database was not the one actually opened"


def test_cli_explicit_override_migrates_only_the_named_sqlite(tmp_path):
    """Full upgrade/downgrade on a disposable SQLite file via the CLI override."""
    target, path = _sqlite_target(tmp_path, "chain.db")

    up = _run_alembic(
        ["-x", f"db_url={target}", "upgrade", "head"], database_url=UNREACHABLE_PG
    )
    assert up.returncode == 0, up.stderr
    # Our resolver logs the target to stdout; Alembic logs upgrades to stderr.
    assert target in up.stdout
    assert NEW_HEAD in (up.stdout + up.stderr)

    import sqlite3

    with sqlite3.connect(path) as conn:
        assert conn.execute("select version_num from alembic_version").fetchone()[0] == (
            NEW_HEAD
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fire_readings)")}
    assert {"synthetic", "source"} <= cols

    down = _run_alembic(
        ["-x", f"db_url={target}", "downgrade", "-1"], database_url=UNREACHABLE_PG
    )
    assert down.returncode == 0, down.stderr
    with sqlite3.connect(path) as conn:
        assert conn.execute("select version_num from alembic_version").fetchone()[0] == (
            "b8d3f1a9c4e2"
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fire_readings)")}
    assert not ({"synthetic", "source"} & cols)


def test_cli_blank_override_fails_without_touching_database_url(tmp_path):
    """(Safety) ``-x db_url=`` must abort, not fall through to DATABASE_URL."""
    result = _run_alembic(["-x", "db_url=", "upgrade", "head"], database_url=UNREACHABLE_PG)
    assert result.returncode != 0
    # No connection attempt to the fallback host may have been made.
    assert "127.0.0.1:1" not in result.stderr
    assert "refusing to fall back" in (result.stderr + result.stdout)
