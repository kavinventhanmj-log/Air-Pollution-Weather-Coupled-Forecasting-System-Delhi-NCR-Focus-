"""Bring a database to the Alembic head safely, including legacy un-versioned ones.

Why this exists
---------------
Historically this app built its schema with ``Base.metadata.create_all()`` and
``main.py`` *swallowed* migration exceptions. Deployments therefore ran against
a database with a full schema but **no ``alembic_version`` table at all**. For
such a database a plain ``alembic upgrade head`` is wrong twice over:

* it replays the entire chain from the baseline, so every ``CREATE TABLE`` fails
  against a table that already exists, and
* the resulting failure is non-idempotent, so the next deploy fails identically.

Blindly running ``alembic stamp head`` on such a database is *also* wrong: it
claims a schema shape the database may not actually have.

So this script inspects the database and picks one of three outcomes.

============  ==================================  ==============================
``alembic``   ``alembic_version`` table            Action
state         present
============  ==================================  ==============================
``versioned`` exists                              ``alembic upgrade head``
``unversioned`` absent, core tables present        ``alembic stamp head``
``empty``      no tables at all                    ``alembic upgrade head``
============  ==================================  ==============================

The ``unversioned`` branch is the one that matters in production. It stamps
rather than upgrades because the schema was created from the same ORM metadata
the migration chain is generated from, so the tables are already at (or close to)
head; replaying the chain against them can only fail. A mismatch introduced
between those two points would be missed, which is why the script prints a loud
verification notice and a follow-up ``alembic check`` hint rather than staying
silent.

Usage
-----
    python backend/scripts/migrate_safely.py              # upgrade or stamp
    python backend/scripts/migrate_safely.py --status     # report only, write nothing
    python backend/scripts/migrate_safely.py --force-upgrade
                                                           # never stamp; always
                                                           # replay the chain

``--status`` never writes and is safe to run against production.
``--force-upgrade`` exists for the rare case where a stamped database genuinely
needs the chain replayed, or for a database whose tables were created by hand
and do *not* match the ORM metadata.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# backend/ must precede any entry exposing a package named ``app``, otherwise
# ``import app`` resolves to the repo-root app/main.py Render shim instead of
# backend/app. alembic/env.py enforces the same ordering, and the two must agree
# or this script inspects a different schema than the one alembic migrates.
# Removing before inserting matters: a plain "if not in sys.path" guard skips the
# fix whenever the path is already present but shadowed.
BACKEND_ROOT = str(REPO_ROOT / "backend")
_repo_root_str = str(REPO_ROOT)
for _stale in (BACKEND_ROOT, _repo_root_str):
    while _stale in sys.path:
        sys.path.remove(_stale)
sys.path.insert(0, BACKEND_ROOT)
# The repo root is still needed to locate alembic.ini and models/, but it must
# come after backend/ so it cannot win the ``app`` lookup.
sys.path.append(_repo_root_str)

from sqlalchemy import inspect, text  # noqa: E402

#: Tables that only exist once the chain has run. ``stations`` is deliberately
#: the first migration in the history and is also the table create_all() would
#: have produced first, so its presence is the signal that this is a
#: pre-Alembic database rather than an empty one.
CORE_TABLES = ("stations",)

#: Tables created by the migrations added in this release. If an un-versioned
#: database is missing these, stamping at head would claim structure it does not
#: have, so the script refuses and explains instead.
RECENT_TABLES = ("forecast_runs",)

LOG_PREFIX = "[migrate_safely]"


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def _alembic(*args: str) -> int:
    """Run an alembic subcommand from the repo root and stream its exit code."""
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode


def _current_revision(engine) -> str | None:
    """Return the stamped revision, or ``None`` if the table is absent.

    ``alembic current`` shells out and prints prose; a direct read is both
    cheaper and unambiguous. A database with no ``alembic_version`` table yields
    ``None`` rather than raising, because that is the exact case being handled.
    """
    # to_regclass() is PostgreSQL-only, so the existence probe goes through the
    # inspector to stay portable to the SQLite development database.
    if "alembic_version" not in _table_names(engine):
        return None
    with engine.connect() as connection:
        row = connection.execute(text("SELECT version_num FROM alembic_version")).first()
    return row[0] if row else None


def _table_names(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def classify(engine) -> tuple[str, set[str]]:
    """Return ``(state, tables)`` where state is versioned|unversioned|empty."""
    tables = _table_names(engine)
    if _current_revision(engine) is not None:
        return "versioned", tables
    if not tables.intersection(CORE_TABLES):
        return "empty", tables
    return "unversioned", tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--status",
        action="store_true",
        help="report the detected state without writing anything",
    )
    parser.add_argument(
        "--force-upgrade",
        action="store_true",
        help="always replay the migration chain; never stamp a legacy database",
    )
    args = parser.parse_args(argv)

    # Imported after the path fix so `app` resolves to backend/app, matching
    # alembic/env.py. Importing at module scope would break `--help`.
    from app.database import engine  # noqa: PLC0415

    try:
        state, tables = classify(engine)
    except Exception as exc:  # pragma: no cover - connectivity failure path
        _log(f"could not reach the database: {exc}")
        _log("aborting without changes; the previous schema is untouched")
        return 1

    _log(f"detected state: {state}")

    if state == "empty":
        _log("no application tables found; the migration chain will create them")

    if state == "unversioned":
        missing = [t for t in RECENT_TABLES if t not in tables]
        if missing and not args.force_upgrade:
            _log(
                "this database has application tables but no alembic_version, "
                "and is missing the recently added table(s): "
                + ", ".join(missing)
            )
            _log("stamping at head would claim structure that does not exist")
            _log("re-run with --force-upgrade to replay the chain instead")
            return 1
        if args.force_upgrade:
            _log("--force-upgrade given: replaying the chain against existing tables")
        else:
            _log(
                "this is a pre-Alembic database created by Base.metadata.create_all(); "
                "its tables come from the same ORM metadata the chain is generated "
                "from, so it is already at head"
            )
            _log("stamping head instead of replaying the chain, which could only fail")
            _log(
                "verify with:  python -m alembic check   "
                "(reports any real drift between models and database)"
            )

    if args.status:
        _log("--status given; no changes written")
        return 0

    if state == "unversioned" and not args.force_upgrade:
        if _alembic("stamp", "head") != 0:
            _log("stamp failed; database left unchanged")
            return 1
        _log("stamped at head")
        return 0

    _log("running: alembic upgrade head")
    return _alembic("upgrade", "head")


if __name__ == "__main__":
    raise SystemExit(main())
