"""Restore fire provenance (synthetic/source) from a FIRMS CSV, exact-match only.

Deliberate, non-destructive one-off: marks ``fire_readings`` rows as
synthetic (or real) **only** where the row's full uniqueness signature
(satellite, latitude, longitude, acq_date) exactly matches a CSV row, using
the same acquisition-time parsing as ``backend.scripts.load_data.load_fire_data``.
Rows with no CSV match are left completely untouched - nothing is inferred,
guessed, or heuristically rewritten. No fuzzy matching, ever.

Safety model (all of it fail-safe; see docs/FIRE_PROVENANCE_FIX.md)
------------------------------------------------------------------
1. ``--db-url`` is **required**. There is no fallback to the application
   engine, ``DATABASE_URL``, or ``.env``. A mistargeted run must be impossible
   by omission, not merely discouraged. The URL is validated through
   ``alembic_db_url.resolve_db_url`` (the same validator Alembic uses) and the
   masked target is printed *before* any connection is attempted.
2. **Dry-run is the default.** Writing requires ``--execute``; writing to a
   non-loopback host additionally requires ``--allow-production``. Production
   approval is derived from the *URL*, never from an environment variable.
3. Contradictory or incomplete flag combinations are rejected before an engine
   is constructed, so an invalid invocation cannot open a socket.
4. Rows already carrying ``source='firms_live'`` are never rewritten.
5. Legacy ``source IS NULL`` rows *are* stamped - including real observations,
   which receive ``source='firms_csv'`` so no reconciled row is left with an
   ambiguous NULL provenance.
6. Duplicate CSV signature keys are a hard error rather than a silent
   last-one-wins.
7. Anomalous match counts (notably zero matches, which is indistinguishable
   from "already reconciled" and was the failure mode that made an earlier
   ``alembic`` invocation report success against the wrong database) are
   surfaced as investigation items, never as success. They are checked against
   the read phase *before* any write, so a failed expectation blocks the write
   rather than being discovered after it. Counts that only indicate normal
   operation - rows already carrying ``firms_live`` provenance or rows with no
   CSV match - are reported as warnings and do not fail the run. A non-zero
   exit status therefore always means nothing was written.

Usage::

    # 1. dry run (default) - SELECT only, no writes
    python -m backend.scripts.backfill_fire_provenance --db-url sqlite:///tmp.db

    # 2. write to a local/staging database
    python -m backend.scripts.backfill_fire_provenance --db-url sqlite:///tmp.db --execute

    # 3. write to production: both acknowledgements required
    python -m backend.scripts.backfill_fire_provenance --db-url "$URL" \\
        --execute --allow-production

The target must already carry the ``synthetic`` and ``source`` columns (alembic
revision ``c5d7e9f1a3b0`` or the ``apply_migrations()`` mirror). This script is
never run automatically.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

try:  # normal package layout: ``python -m backend.scripts.backfill_fire_provenance``
    from backend.scripts.alembic_db_url import (
        DB_URL_KEY,
        DbUrlResolutionError,
        resolve_db_url,
    )
except ImportError:  # pragma: no cover - ``backend/``-root sys.path layout
    from scripts.alembic_db_url import DB_URL_KEY, DbUrlResolutionError, resolve_db_url

DEFAULT_FIRE_CSV = (
    Path(__file__).resolve().parent.parent.parent / "data" / "fire" / "firms_fires.csv"
)

#: Same acquisition-time convention as load_data.load_fire_data / FIRMS
#: (acq_time 9999/2400 handled by the "%H%M" fallback).
_TRUTHY = {"1", "1.0", "true", "t", "yes", "y"}
_SOURCE_DEFAULTS = {True: "synthetic_sim", False: "firms_csv"}

#: Provenance written by ``firms_service`` from a live FIRMS download. A row
#: that already carries it is ground truth from the source system and is never
#: rewritten by this script.
LIVE_SOURCE = "firms_live"

#: Hosts that are, by construction, not the production deployment. A non-sqlite
#: URL pointing anywhere else is treated as production and demands an explicit
#: ``--allow-production``. Derived from the URL alone - never from the
#: environment - so an exported ``DATABASE_URL`` cannot imply approval.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}

#: Server session timezones under which naive timestamps are stored verbatim.
#: Anything else means the stored instants were interpreted in a local zone and
#: every exact-match key would be shifted (see ``probe_session_timezone``).
_UTC_ALIASES = {"utc", "etc/utc", "gmt", "z", "uct", "universal", "zulu"}

#: Rows per executemany batch. Bounds the size of any single driver call while
#: keeping the whole update set inside one transaction.
UPDATE_BATCH_SIZE = 5_000


@dataclass(frozen=True)
class TimezoneProbe:
    """Outcome of a session-timezone probe.

    ``applicable`` is whether the dialect has a session timezone to probe at all
    (PostgreSQL). ``value`` is the reported zone, or ``None`` when the probe is
    inconclusive - either not applicable, or the read failed/returned nothing.
    """

    applicable: bool
    value: str | None


@dataclass(frozen=True)
class Expectations:
    """Operator-supplied counts asserted before a write is allowed through."""

    min_matched: int = 1
    expect_db_rows: int | None = None
    expect_matched: int | None = None

SELECT_SQL = (
    "SELECT id, satellite, latitude, longitude, acq_date, synthetic, source "
    "FROM fire_readings"
)
UPDATE_SQL = (
    "UPDATE fire_readings SET synthetic = :synthetic, source = :source WHERE id = :id"
)


class BackfillError(RuntimeError):
    """Operator-facing refusal. Never carries a credential."""


class AmbiguousSignatureError(BackfillError):
    """The CSV maps one signature key to more than one classification."""


def _as_bool(v) -> bool:
    if v is None or (isinstance(v, float) and v != v):
        return False
    return str(v).strip().lower() in _TRUTHY


def _reconstruct_acq_time(acq_date, acq_time) -> datetime | None:
    """Reconstruct the naive-UTC datetime exactly as load_fire_data does."""
    if acq_date is None or (isinstance(acq_date, float) and acq_date != acq_date):
        return None
    try:
        hhmm = str(int(float(acq_time or 0))).zfill(4)
    except (TypeError, ValueError):
        hhmm = "0000"
    ts = pd.to_datetime(f"{acq_date} {hhmm}", format="%Y-%m-%d %H%M", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(acq_date, errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()


def _col(df, name, fallback=None):
    """Return a column, or a same-length Series so the strict zip() below aligns.

    A standard FIRMS CSV has no ``source`` column (and older exports may lack
    ``synthetic``), so ``df.get(name, empty)`` would hand the strict zip an
    empty Series and abort the whole backfill with
    "zip() argument N is shorter than arguments 1-N".
    """
    if name in df.columns:
        return df[name]
    if fallback is not None:
        return fallback
    return pd.Series(None, dtype="object", index=df.index)


def masked_target(db_url: str) -> str:
    """Render a URL with any password masked, for logs and operator output."""
    try:
        return make_url(db_url).render_as_string(hide_password=True)
    except ArgumentError:
        return "<unparseable>"


def resolve_target(db_url: str | None, *, log=None) -> str:
    """Validate an operator-supplied URL. Never falls back to any other target.

    Delegates to ``alembic_db_url.resolve_db_url`` so this script inherits the
    exact guarantee Alembic relies on: a supplied ``db_url`` is authoritative,
    and a blank or unparseable one raises instead of quietly selecting a
    different database. The resolver's own ``-x db_url`` origin label and log
    line are suppressed (they name Alembic); the masked target is announced
    here instead, before any connection exists.
    """
    if log is None:
        log = print

    if not db_url or not db_url.strip():
        raise BackfillError(
            "no database target supplied. --db-url is REQUIRED: this script never "
            "falls back to DATABASE_URL or the application engine, because a "
            "mistargeted reconciliation is unrecoverable. Pass an explicit "
            "SQLAlchemy URL, e.g. --db-url sqlite:///./scratch.db"
        )

    def _no_settings() -> str:
        raise BackfillError(
            "refusing to consult application settings for a database target"
        )

    try:
        resolved = resolve_db_url(
            x_arguments={DB_URL_KEY: db_url},
            main_option=None,
            environ={},
            settings_loader=_no_settings,
            log=lambda _msg: None,
        )
    except DbUrlResolutionError as exc:
        # The message is already password-free (alembic_db_url._safe_target), but
        # re-render the operator's own string defensively before echoing it.
        raise BackfillError(f"invalid --db-url: {masked_target(db_url)} ({exc})") from None
    log(f"[fire-backfill] database target (--db-url): {masked_target(resolved)}")
    return resolved


def is_production_target(db_url: str) -> bool:
    """True when the URL is not sqlite and not loopback.

    Deliberately URL-derived and deliberately conservative: anything that is
    not obviously a developer machine needs ``--allow-production``.
    """
    try:
        url = make_url(db_url)
    except ArgumentError:
        return True
    if (url.drivername or "").startswith("sqlite"):
        return False
    return (url.host or "") not in _LOOPBACK_HOSTS


def validate_invocation(
    *,
    db_url: str | None,
    execute: bool,
    dry_run: bool,
    allow_production: bool,
    allow_non_utc: bool,
) -> None:
    """Reject contradictory/incomplete flags. Runs before any engine exists."""
    if execute and dry_run:
        raise BackfillError(
            "--execute and --dry-run are mutually exclusive: --execute is the only "
            "way to write, and dry-run is the default. Pass exactly one."
        )
    if allow_production and not execute:
        raise BackfillError(
            "--allow-production requires --execute. Production acknowledgement "
            "without an execution request is a contradictory invocation and is "
            "refused before any connection is opened."
        )
    if allow_production and db_url and not is_production_target(db_url):
        raise BackfillError(
            "--allow-production was passed but the target is not a production "
            "host. Remove the flag: it exists so that writing to a live database "
            "is always a deliberate, two-flag decision."
        )
    if execute and db_url and is_production_target(db_url) and not allow_production:
        raise BackfillError(
            "refusing to write to a production database without explicit "
            "acknowledgement. Re-run with --execute --allow-production."
        )
    if allow_non_utc and execute and db_url and is_production_target(db_url) and not allow_production:
        raise BackfillError(
            "--allow-non-utc-timezone on a production write also requires "
            "--allow-production."
        )
    if allow_non_utc and not execute:
        raise BackfillError(
            "--allow-non-utc-timezone requires --execute. It waives a write-time "
            "guard, so on a dry run it would be silently inert and imply the "
            "timezone question had been settled when it had not."
        )


def load_csv_signatures(
    csv_path: Path,
    chunksize: int = 100_000,
    *,
    allow_duplicates: bool = False,
) -> dict[tuple, dict]:
    """Map (satellite, lat4, lon4, acq_date) -> {synthetic, source} from a FIRMS CSV.

    A signature key that appears twice is refused rather than resolved by
    last-one-wins, because "which row won" is unknowable afterwards and a
    rounded-coordinate collision could silently relabel a real observation as
    simulated. With ``allow_duplicates=True`` the *first* occurrence is kept
    (deterministic, never order-dependent on the final row) and keys whose
    classifications actually disagree are still refused.
    """
    signatures: dict[tuple, dict] = {}
    duplicate_examples: list[tuple] = []
    duplicate_count = 0
    conflicting: list[tuple] = []

    for df in pd.read_csv(csv_path, low_memory=False, chunksize=chunksize):
        for sat, lat, lon, adate, atime, syn, src in zip(
            _col(df, "satellite"),
            pd.to_numeric(_col(df, "latitude"), errors="coerce"),
            pd.to_numeric(_col(df, "longitude"), errors="coerce"),
            _col(df, "acq_date"),
            pd.to_numeric(_col(df, "acq_time", pd.Series(0, index=df.index)), errors="coerce"),
            _col(df, "synthetic"),
            _col(df, "source"),
            strict=True,
        ):
            if pd.isna(lat) or pd.isna(lon) or pd.isna(sat):
                continue
            ts = _reconstruct_acq_time(adate, atime)
            if ts is None:
                continue
            syn_flag = _as_bool(syn)
            src_v = None if pd.isna(src) else str(src).strip()
            key = (str(sat).strip(), round(float(lat), 4), round(float(lon), 4), ts)
            record = {
                "synthetic": syn_flag,
                "source": src_v or _SOURCE_DEFAULTS[syn_flag],
            }
            previous = signatures.get(key)
            if previous is None:
                signatures[key] = record
                continue
            duplicate_count += 1
            if len(duplicate_examples) < 3:
                duplicate_examples.append(key)
            if (previous["synthetic"], previous["source"]) != (
                record["synthetic"],
                record["source"],
            ):
                conflicting.append(key)
            if not allow_duplicates:
                continue
            # Keep the first occurrence: never the last.

    if conflicting:
        raise AmbiguousSignatureError(
            f"{len(set(conflicting)):,} signature key(s) in {csv_path} carry "
            "conflicting provenance (e.g. the same satellite/position/time is "
            "marked both synthetic and real). Refusing to reconcile: resolve the "
            "CSV first. First example: "
            f"{conflicting[0]!r}"
        )
    if duplicate_count and not allow_duplicates:
        raise AmbiguousSignatureError(
            f"{duplicate_count:,} duplicate signature key(s) in {csv_path} (e.g. "
            f"{duplicate_examples[0]!r}). Refusing to pick a winner silently. "
            "De-duplicate the CSV, or re-run with --allow-duplicate-signatures to "
            "keep the first occurrence of each key."
        )
    return signatures


def _to_naive_utc(dt) -> datetime:
    """Normalise a stored datetime to naive-UTC for key comparison.

    Most drivers return ``datetime``; some SQLite configurations hand back the
    raw ISO string, which would otherwise silently break every exact-match key.

    Timestamp semantics are deliberately UNCHANGED from the pre-hardening
    implementation: aware values are converted to UTC, naive values are
    compared as-is. The session-timezone assumption this rests on is reported
    by ``probe_session_timezone`` and must be acknowledged on a production
    write; it is not silently assumed here or patched here.
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        ts = pd.to_datetime(dt, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime().replace(tzinfo=None)
    if getattr(dt, "tzinfo", None) is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _row_signature(sat, lat, lon, acq_date) -> tuple:
    ts = _to_naive_utc(acq_date)
    return (str(sat or "").strip(), round(float(lat or 0), 4), round(float(lon or 0), 4), ts)


def _batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def probe_session_timezone(conn, dialect_name: str) -> TimezoneProbe:
    """Read the server session timezone with a read-only ``SELECT``.

    ``fire_readings.acq_date`` is TIMESTAMPTZ on PostgreSQL, and the exact-match
    keys are naive UTC. That round-trips only when the session timezone is UTC;
    otherwise naive wall-clocks were stored shifted and every key mismatches.
    Reported rather than assumed.

    Uses ``SELECT current_setting('TimeZone')`` rather than ``SHOW TimeZone``
    so the dry run issues ``SELECT`` statements exclusively - no ``SHOW``, no
    DDL, no DML. Dialect-gated because the setting is PostgreSQL-specific; on
    any other backend the probe is not applicable and no query is issued.

    Returns a :class:`TimezoneProbe`. An inconclusive PostgreSQL probe (the
    read failed or returned nothing) yields ``applicable=True, value=None`` so
    the caller can fail closed instead of silently assuming UTC.
    """
    if (dialect_name or "").split("+", 1)[0] != "postgresql":
        return TimezoneProbe(applicable=False, value=None)
    try:
        value = conn.execute(text("SELECT current_setting('TimeZone')")).scalar()
    except SQLAlchemyError:
        return TimezoneProbe(applicable=True, value=None)
    return TimezoneProbe(
        applicable=True,
        value=None if value is None or not str(value).strip() else str(value),
    )


def timezone_guard_error(
    *,
    timezone: str | None,
    timezone_known: bool,
    dry_run: bool,
    production: bool,
    allow_non_utc: bool,
) -> str | None:
    """Return a refusal message when a write must not proceed, else ``None``.

    Fail closed in the two cases that make exact-match keys untrustworthy: an
    inconclusive probe on a production target, and a positively non-UTC session
    on any target. ``--allow-non-utc-timezone`` waives either, because it is the
    operator explicitly accepting the comparison as computed.
    """
    if dry_run or allow_non_utc:
        return None
    if production and not timezone_known:
        return (
            "could not determine the database session timezone on a production "
            "target. Naive timestamps in fire_readings may have been interpreted "
            "in a local zone, so the exact-match keys cannot be trusted. "
            "Investigate, then re-run with --allow-non-utc-timezone to accept the "
            "comparison as computed."
        )
    if timezone and timezone.strip().lower() not in _UTC_ALIASES:
        return (
            f"session timezone is {timezone!r}, not UTC. Naive timestamps in "
            "fire_readings would have been interpreted in that zone, so the "
            "exact-match keys above may be shifted for every row. Investigate "
            "before writing; re-run with --allow-non-utc-timezone to accept the "
            "comparison as computed."
        )
    return None


def _new_report(dry_run: bool) -> dict:
    return {
        "db_rows_scanned": 0,
        "db_rows_matched": 0,
        # Rows the reconciliation *would* rewrite. Always computed, including
        # on a dry run, so a dry run states the blast radius rather than zeros.
        "db_rows_pending": 0,
        # Rows actually written. Zero on every dry run by construction.
        "db_rows_updated": 0,
        "db_rows_unchanged": 0,
        "db_rows_unmatched_left_untouched": 0,
        "db_rows_protected_live": 0,
        "db_rows_set_synthetic": 0,
        "db_rows_set_real": 0,
        "db_rows_legacy_null_source": 0,
        "db_session_timezone": None,
        # Whether the timezone comparison rests on a known value. Defaults true
        # for backends with no session timezone (SQLite): there is nothing to
        # be uncertain about.
        "db_session_timezone_known": True,
        "dry_run": bool(dry_run),
        # Populated by run(): errors that block success, and informational
        # warnings (protected live rows, unmatched rows) that never do.
        "preflight_anomalies": [],
        "postflight_errors": [],
        "postflight_warnings": [],
    }


def _apply_updates(engine, updates: list[dict], batch_size: int = UPDATE_BATCH_SIZE) -> int:
    """Write every pending update inside one transaction.

    All batches share a single ``engine.begin()`` block, so a failure part-way
    through rolls the whole reconciliation back: the table is never left half
    stamped. Parameter binding is used throughout (no string interpolation).
    """
    written = 0
    with engine.begin() as conn:
        for batch in _batched(updates, batch_size):
            conn.execute(text(UPDATE_SQL), batch)
            written += len(batch)
    return written


def run(
    db_url: str,
    signatures: dict[tuple, dict],
    dry_run: bool = True,
    *,
    allow_non_utc_timezone: bool = False,
    batch_size: int = UPDATE_BATCH_SIZE,
    expectations: Expectations | None = None,
) -> dict:
    """Reconcile provenance for an already-resolved target. Dry-run by default.

    When ``expectations`` is supplied, the read-phase counts are checked
    *before* any write: a failed expectation leaves the table untouched. Rows
    protected as ``firms_live`` and rows with no CSV match are reported as
    warnings only and never block the write.

    The engine is disposed on every path, including exceptions.
    """
    try:
        engine = create_engine(db_url)
    except (ArgumentError, SQLAlchemyError) as exc:
        raise BackfillError(
            f"could not build an engine for {masked_target(db_url)}: {type(exc).__name__}"
        ) from None

    try:
        try:
            cols = {c["name"] for c in inspect(engine).get_columns("fire_readings")}
        except SQLAlchemyError as exc:
            raise BackfillError(
                f"could not inspect fire_readings on {masked_target(db_url)}: "
                f"{type(exc).__name__}. Confirm the table exists on the named target."
            ) from None
        missing = {"synthetic", "source"} - cols
        if missing:
            raise BackfillError(
                f"fire_readings is missing provenance columns {sorted(missing)} on "
                f"{masked_target(db_url)} - apply revision c5d7e9f1a3b0 to that "
                "database first."
            )

        report = _new_report(dry_run)
        updates: list[dict] = []

        try:
            with engine.connect() as conn:
                probe = probe_session_timezone(conn, engine.dialect.name)
                report["db_session_timezone"] = probe.value
                report["db_session_timezone_known"] = (not probe.applicable) or bool(
                    probe.value
                )
                rows = conn.execute(text(SELECT_SQL))
                for row in rows:
                    report["db_rows_scanned"] += 1
                    rec = signatures.get(
                        _row_signature(
                            row.satellite, row.latitude, row.longitude, row.acq_date
                        )
                    )
                    if rec is None:
                        report["db_rows_unmatched_left_untouched"] += 1
                        continue
                    report["db_rows_matched"] += 1

                    current_source = (row.source or "").strip() or None
                    if current_source == LIVE_SOURCE:
                        # Ground truth from the live FIRMS feed. Reclassifying it
                        # would overwrite a fresher, more authoritative observation.
                        report["db_rows_protected_live"] += 1
                        continue
                    report["db_rows_set_synthetic" if rec["synthetic"] else "db_rows_set_real"] += 1
                    if current_source is None:
                        report["db_rows_legacy_null_source"] += 1

                    target_source = rec["source"]
                    # Strict equality: a NULL source is NOT treated as "already
                    # equal". Legacy rows are stamped so a reconciled database has
                    # no rows left with ambiguous provenance.
                    if (
                        bool(row.synthetic) == rec["synthetic"]
                        and current_source == target_source
                    ):
                        report["db_rows_unchanged"] += 1
                        continue
                    updates.append(
                        {"id": row.id, "synthetic": rec["synthetic"], "source": target_source}
                    )
        except BackfillError:
            raise
        except SQLAlchemyError as exc:
            # Never surface the raw driver message: psycopg2/SQLAlchemy errors can
            # embed the DSN it failed to connect with.
            raise BackfillError(
                f"read of fire_readings failed on {masked_target(db_url)}: "
                f"{type(exc).__name__}. No rows were written."
            ) from None

        report["db_rows_pending"] = len(updates)

        # Fail closed before any write if the timezone assumption cannot be
        # trusted (inconclusive probe on production, or a positively non-UTC
        # session). The acknowledgement flag waives either check.
        guard = timezone_guard_error(
            timezone=report["db_session_timezone"],
            timezone_known=report["db_session_timezone_known"],
            dry_run=dry_run,
            production=is_production_target(db_url),
            allow_non_utc=allow_non_utc_timezone,
        )
        if guard:
            raise BackfillError(guard)

        # Expectations are checked from the read phase, BEFORE the write. A
        # failed expectation therefore leaves the table completely untouched.
        preflight = (
            preflight_expectations(
                report,
                min_matched=expectations.min_matched,
                expect_db_rows=expectations.expect_db_rows,
                expect_matched=expectations.expect_matched,
            )
            if expectations is not None
            else []
        )
        report["preflight_anomalies"] = preflight

        if dry_run or preflight or not updates:
            report["db_rows_updated"] = 0
        else:
            try:
                report["db_rows_updated"] = _apply_updates(engine, updates, batch_size)
            except SQLAlchemyError as exc:
                # _apply_updates holds every batch in one transaction, so this
                # failure has already rolled the whole reconciliation back.
                raise BackfillError(
                    f"write to {masked_target(db_url)} failed: {type(exc).__name__}. "
                    f"All {len(updates)} row(s) were rolled back; the table is "
                    "unchanged."
                ) from None

        # A preflight failure means no write was attempted, so the "updated 0
        # rows" error would be misleading and is suppressed.
        report["postflight_errors"] = [] if preflight else postflight_errors(report)
        report["postflight_warnings"] = postflight_warnings(report)
        return report
    finally:
        engine.dispose()


def preflight_expectations(
    report: dict,
    *,
    min_matched: int = 1,
    expect_db_rows: int | None = None,
    expect_matched: int | None = None,
) -> list[str]:
    """Anomalies knowable from the read phase, before anything is written.

    A zero-match run is the dangerous one: it is what a mistargeted database
    (or a shifted timestamp convention) looks like, and it used to be reported
    as "Done. 0 row(s) updated." with exit status 0. Because these checks run
    before ``_apply_updates``, a failure here guarantees nothing was written.
    """
    anomalies: list[str] = []
    scanned = report["db_rows_scanned"]
    matched = report["db_rows_matched"]

    if scanned == 0:
        anomalies.append(
            "db_rows_scanned is 0: fire_readings is empty on this target, so the "
            "CSV could not be reconciled. Confirm the target."
        )
    if matched == 0:
        anomalies.append(
            "db_rows_matched is 0: not a single row matched the CSV signatures. "
            "Either this is the wrong database, or stored timestamps/coords do "
            "not line up. Re-check the target and the session timezone."
        )
    if matched < min_matched:
        anomalies.append(
            f"db_rows_matched ({matched:,}) is below the required minimum "
            f"({min_matched:,})."
        )
    if expect_db_rows is not None and scanned != expect_db_rows:
        anomalies.append(
            f"db_rows_scanned ({scanned:,}) != expected ({expect_db_rows:,})."
        )
    if expect_matched is not None and matched != expect_matched:
        anomalies.append(
            f"db_rows_matched ({matched:,}) != expected ({expect_matched:,})."
        )
    return anomalies


def postflight_errors(report: dict) -> list[str]:
    """Errors that only become meaningful once the write decision is known.

    The single entry here is an ``--execute`` run that updated nothing: it is a
    no-op, not a success. Nothing was written, so this never describes a
    committed change.

    The no-op is only an error when nothing explains it. Rows already carrying
    ``firms_live`` provenance are matched rows we deliberately decline to
    rewrite, so a run whose matched rows are all protected has correctly done
    nothing and exits 0 (see :func:`postflight_warnings`). An already-reconciled
    table with no such protection is still reported as a no-op.
    """
    if report["dry_run"] or report["db_rows_updated"] != 0:
        return []
    if report["db_rows_protected_live"] > 0:
        return []
    return [
        "an --execute run updated 0 rows. That is a no-op, not a success: the "
        "table may already be reconciled, or nothing matched."
    ]


def postflight_warnings(report: dict) -> list[str]:
    """Informational notes. Never change the exit code.

    Rows already carrying ``firms_live`` provenance and rows with no CSV match
    are expected on a real target. They are surfaced so the operator sees the
    whole picture, but a reconciliation that did exactly what it should is
    still a success.
    """
    warnings: list[str] = []
    if report["db_rows_protected_live"]:
        warnings.append(
            f"{report['db_rows_protected_live']:,} row(s) already carried "
            f"source='{LIVE_SOURCE}' and were deliberately left untouched."
        )
    if report["db_rows_unmatched_left_untouched"]:
        warnings.append(
            f"{report['db_rows_unmatched_left_untouched']:,} row(s) had no CSV "
            "match and were left unchanged (still carrying their previous "
            "provenance)."
        )
    return warnings


def check_expectations(
    report: dict,
    *,
    min_matched: int = 1,
    expect_db_rows: int | None = None,
    expect_matched: int | None = None,
) -> list[str]:
    """Union of preflight errors, postflight errors, and postflight warnings.

    Retained for callers that want every notable count in one list. The exit
    code is driven only by :func:`preflight_expectations` and
    :func:`postflight_errors`; :func:`postflight_warnings` is informational.
    """
    return (
        preflight_expectations(
            report,
            min_matched=min_matched,
            expect_db_rows=expect_db_rows,
            expect_matched=expect_matched,
        )
        + postflight_errors(report)
        + postflight_warnings(report)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore fire_readings provenance from a FIRMS CSV. Exact match only. "
            "Dry-run is the default; --db-url is mandatory and there is no "
            "DATABASE_URL fallback."
        )
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="REQUIRED. SQLAlchemy URL of the database to reconcile. Never inferred.",
    )
    parser.add_argument("--csv", default=str(DEFAULT_FIRE_CSV), help="Path to the FIRMS CSV")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write. Omit this (the default) for a SELECT-only dry run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry run (the default; provided for readability).",
    )
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help=(
            "Acknowledge that the target is a live/production database. Only "
            "valid together with --execute and only when the URL is not loopback."
        ),
    )
    parser.add_argument(
        "--allow-non-utc-timezone",
        action="store_true",
        help=(
            "Accept exact-match comparison on a target whose session timezone is "
            "not UTC. Investigate first: this is the documented way a whole-table "
            "match can silently fail."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-signatures",
        action="store_true",
        help=(
            "Proceed when the CSV repeats a signature key, keeping the first "
            "occurrence. Keys whose provenance conflicts are still refused."
        ),
    )
    parser.add_argument(
        "--min-matched",
        type=int,
        default=1,
        help="Refuse to call the run successful below this many matched rows (default 1).",
    )
    parser.add_argument(
        "--expect-db-rows",
        type=int,
        default=None,
        help="Assert the exact fire_readings row count; a mismatch is an anomaly.",
    )
    parser.add_argument(
        "--expect-matched",
        type=int,
        default=None,
        help="Assert the exact matched row count; a mismatch is an anomaly.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Flag validation happens first and touches nothing: no engine, no socket.
    try:
        validate_invocation(
            db_url=args.db_url,
            execute=args.execute,
            dry_run=args.dry_run,
            allow_production=args.allow_production,
            allow_non_utc=args.allow_non_utc_timezone,
        )
        db_url = resolve_target(args.db_url)
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise BackfillError(f"fire CSV not found: {csv_path}")
        signatures = load_csv_signatures(
            csv_path, allow_duplicates=args.allow_duplicate_signatures
        )
    except BackfillError as exc:
        print(f"[fire-backfill] REFUSED: {exc}", file=sys.stderr)
        return 2

    synthetic = sum(1 for v in signatures.values() if v["synthetic"])
    real = len(signatures) - synthetic
    print(f"CSV signatures  : {len(signatures):,} ({real:,} real / {synthetic:,} synthetic)")

    mode = "EXECUTE (will write)" if args.execute else "DRY RUN (read-only)"
    print(f"Mode            : {mode}")
    try:
        report = run(
            db_url,
            signatures,
            dry_run=not args.execute,
            allow_non_utc_timezone=args.allow_non_utc_timezone,
            expectations=Expectations(
                min_matched=args.min_matched,
                expect_db_rows=args.expect_db_rows,
                expect_matched=args.expect_matched,
            ),
        )
    except BackfillError as exc:
        print(f"[fire-backfill] REFUSED: {exc}", file=sys.stderr)
        return 2

    print("-" * 72)
    for key in (
        "db_rows_scanned",
        "db_rows_matched",
        "db_rows_unmatched_left_untouched",
        "db_rows_protected_live",
        "db_rows_legacy_null_source",
        "db_rows_unchanged",
        "db_rows_set_synthetic",
        "db_rows_set_real",
        "db_rows_pending",
        "db_rows_updated",
    ):
        print(f"  {key:<36} {report[key]:,}")
    if report["db_session_timezone"] is not None:
        print(f"  {'db_session_timezone':<36} {report['db_session_timezone']}")
    print("-" * 72)

    warnings = report["postflight_warnings"]
    if warnings:
        print("[fire-backfill] WARNING - review these counts:")
        for item in warnings:
            print(f"  ~ {item}")
        print("-" * 72)

    errors = report["preflight_anomalies"] + report["postflight_errors"]
    if errors:
        print("[fire-backfill] ANOMALY - no rows were written:")
        for item in errors:
            print(f"  ! {item}")
        print("-" * 72)
        print(
            "NOT reported as success. Re-run against a restored copy or reconcile the "
            "counts. This invocation committed nothing."
        )
        return 3

    if report["dry_run"]:
        print(
            f"DRY RUN - no rows were written. {report['db_rows_pending']:,} row(s) "
            "would change on --execute."
        )
    else:
        print(f"Done. {report['db_rows_updated']:,} row(s) updated in one transaction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
