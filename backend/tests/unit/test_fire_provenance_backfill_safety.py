"""Gate-0 hardening for the fire-provenance backfill.

Safety regressions covered here, all against disposable SQLite files or mocks:

* ``--db-url`` is mandatory and is the only target source; nothing falls back
  to ``DATABASE_URL``/``.env``/the application engine.
* Contradictory or incomplete flag combinations are refused *before* an engine
  is constructed, so an invalid invocation cannot open a socket.
* Dry-run is the default and performs no DML at all.
* Exact matching only: near-miss timestamps and coordinates stay unmatched.
* Duplicate / rounding-collision CSV signature keys are refused, not resolved
  by last-one-wins.
* Rows already stamped ``firms_live`` are never reclassified.
* Real FIRMS rows are given explicit provenance instead of being left NULL.
* A zero-match reconciliation is reported as an anomaly, never as success.
* The engine is disposed even when the run raises.
* Passwords never reach stdout, stderr, or an exception message.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.scripts import backfill_fire_provenance as bf

#: Syntactically valid but unreachable. Any accidental fall-through to an
#: environment-provided target would fail loudly instead of touching a real
#: database. Never points at Neon.
UNREACHABLE_PG = "postgresql://nobody:nobody@127.0.0.1:1/never_used"

SECRET = "sup3rs3cret-do-not-log"


def _no_engine(*args, **kwargs):  # pragma: no cover - only fires on a failure
    raise AssertionError("create_engine must not be reached for this invocation")


@pytest.fixture
def csv_factory(tmp_path):
    def _make(rows, name="fires.csv"):
        path = tmp_path / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    return _make


@pytest.fixture
def db_factory(tmp_path):
    """A minimal post-migration ``fire_readings`` table on a throwaway SQLite file."""

    def _make(row_specs, name="fire.db"):
        path = tmp_path / name
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE fire_readings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "satellite VARCHAR, latitude FLOAT NOT NULL, longitude FLOAT NOT NULL, "
            "acq_date DATETIME NOT NULL, synthetic BOOLEAN NOT NULL DEFAULT 0, "
            "source VARCHAR)"
        )
        for spec in row_specs:
            conn.execute(
                "INSERT INTO fire_readings "
                "(satellite, latitude, longitude, acq_date, synthetic, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                spec,
            )
        conn.commit()
        conn.close()
        return f"sqlite:///{path}"

    return _make


def _rows(url):
    engine = bf.create_engine(url)
    try:
        with engine.connect() as conn:
            return [
                (r[0], r[1], r[2])
                for r in conn.exec_driver_sql(
                    "SELECT latitude, synthetic, source FROM fire_readings"
                )
            ]
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #


class TestTargetSelection:
    def test_missing_db_url_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr(bf, "create_engine", _no_engine)
        monkeypatch.setenv("DATABASE_URL", UNREACHABLE_PG)

        assert bf.main(["--csv", "unused.csv"]) == 2

        err = capsys.readouterr().err
        assert "--db-url is REQUIRED" in err
        assert "127.0.0.1:1" not in err

    def test_blank_db_url_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr(bf, "create_engine", _no_engine)
        assert bf.main(["--db-url", "   "]) == 2
        assert "--db-url is REQUIRED" in capsys.readouterr().err

    def test_invalid_db_url_is_refused_without_connecting(self, monkeypatch, capsys):
        monkeypatch.setattr(bf, "create_engine", _no_engine)
        assert bf.main(["--db-url", "definitely not a url"]) == 2
        err = capsys.readouterr().err
        assert "invalid --db-url" in err

    def test_explicit_url_wins_over_environment(self, monkeypatch, csv_factory, db_factory):
        """--db-url is authoritative; DATABASE_URL is never consulted."""
        monkeypatch.setenv("DATABASE_URL", UNREACHABLE_PG)
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        # The dry run reached the tmp SQLite file (matched, would update) and
        # never attempted the unreachable DATABASE_URL host.
        assert bf.main(["--db-url", url, "--csv", str(csv)]) == 0

    def test_environment_database_url_is_never_read(self, monkeypatch, csv_factory, db_factory):
        """A populated DATABASE_URL must not become the implicit target."""
        opened: list[str] = []

        import sqlalchemy

        real_create_engine = sqlalchemy.create_engine

        def _spy(url, *a, **kw):
            opened.append(str(url))
            return real_create_engine(url, *a, **kw)

        monkeypatch.setattr(bf, "create_engine", _spy)
        monkeypatch.setenv("DATABASE_URL", UNREACHABLE_PG)
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        assert bf.main(["--csv", str(csv)]) == 2
        assert opened == []

    def test_masked_target_printed_before_connecting(self, monkeypatch, capsys, csv_factory, db_factory):
        printed: list[str] = []

        import sqlalchemy

        real_create_engine = sqlalchemy.create_engine

        def _spy(url, *a, **kw):
            printed.append(" ".join(capsys.readouterr().out.split()))
            return real_create_engine(url, *a, **kw)

        monkeypatch.setattr(bf, "create_engine", _spy)
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        bf.main(["--db-url", url, "--csv", str(csv)])
        assert printed, "no engine was created"
        assert "database target (--db-url)" in printed[0]


class TestUrlValidation:
    def test_masked_target_hides_password(self):
        rendered = bf.masked_target(f"postgresql://u:{SECRET}@db.example.invalid/app")
        assert SECRET not in rendered
        assert "db.example.invalid" in rendered

    def test_masked_target_survives_garbage(self):
        assert bf.masked_target("!!! not a url") == "<unparseable>"

    def test_resolve_target_rejects_blank(self):
        for bad in (None, "", "   "):
            with pytest.raises(bf.BackfillError, match="--db-url is REQUIRED"):
                bf.resolve_target(bad, log=lambda _m: None)

    def test_resolve_target_rejects_unparseable(self):
        with pytest.raises(bf.BackfillError):
            bf.resolve_target("not a url at all", log=lambda _m: None)

    def test_resolve_target_returns_valid_url_verbatim(self, tmp_path):
        target = f"sqlite:///{tmp_path/'x.db'}"
        assert bf.resolve_target(target, log=lambda _m: None) == target

    def test_is_production_target_classification(self):
        assert bf.is_production_target("postgresql://u:p@db.neon.tech/neondb") is True
        assert bf.is_production_target("postgresql://u:p@10.0.0.5:5432/app") is True
        assert bf.is_production_target("postgresql://u:p@localhost:5432/app") is False
        assert bf.is_production_target("postgresql://u:p@127.0.0.1:5432/app") is False
        assert bf.is_production_target("sqlite:///./scratch.db") is False


class TestSecretHygiene:
    def test_no_secret_in_refusal_output(self, monkeypatch, capsys):
        monkeypatch.setattr(bf, "create_engine", _no_engine)
        url = f"postgresql://user:{SECRET}@db.neon.tech/never_used"
        rc = bf.main(["--db-url", url, "--execute"])
        captured = capsys.readouterr()
        assert rc in (2, 3)
        assert SECRET not in captured.out
        assert SECRET not in captured.err

    def test_no_secret_in_schema_mismatch_error(self, tmp_path):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE fire_readings (id INTEGER PRIMARY KEY, "
            "satellite VARCHAR, latitude FLOAT, longitude FLOAT, acq_date DATETIME)"
        )
        conn.commit()
        conn.close()

        url = f"postgresql://user:{SECRET}@host.invalid/app"
        with pytest.raises(bf.BackfillError) as exc:
            bf.run(url, {})
        # Engine creation must fail first, without echoing the password.
        assert SECRET not in str(exc.value)

    def test_engine_creation_error_has_no_exception_cause(self):
        url = f"nonsense+dialect://user:{SECRET}@host.invalid/app"
        with pytest.raises(bf.BackfillError) as exc:
            bf.run(url, {})
        # Nothing chained: the raw SQLAlchemy error cannot be rendered.
        assert exc.value.__cause__ is None
        assert SECRET not in str(exc.value)

    def test_inspect_error_has_no_exception_cause(self, monkeypatch):
        import traceback

        from sqlalchemy.exc import SQLAlchemyError

        def _boom(_engine):
            raise SQLAlchemyError(f"driver exploded with {SECRET}")

        monkeypatch.setattr(bf, "inspect", _boom)
        with pytest.raises(bf.BackfillError) as exc:
            bf.run("sqlite:///plain.db", {})
        rendered = "".join(
            traceback.format_exception(
                type(exc.value), exc.value, exc.value.__traceback__
            )
        )
        assert exc.value.__cause__ is None
        assert SECRET not in rendered


# --------------------------------------------------------------------------- #
# Flag validation happens before any connection
# --------------------------------------------------------------------------- #


class TestFlagValidation:
    @pytest.fixture(autouse=True)
    def _no_engine(self, monkeypatch):
        monkeypatch.setattr(bf, "create_engine", _no_engine)

    def test_execute_and_dry_run_together_refused(self, tmp_path):
        rc = bf.main(
            ["--db-url", f"sqlite:///{tmp_path/'a.db'}", "--execute", "--dry-run"]
        )
        assert rc == 2

    def test_allow_production_without_execute_refused(self, tmp_path):
        rc = bf.main(
            ["--db-url", "postgresql://u:p@db.neon.tech/neondb", "--allow-production"]
        )
        assert rc == 2

    def test_production_write_without_acknowledgement_refused(self, tmp_path):
        rc = bf.main(["--db-url", "postgresql://u:p@db.neon.tech/neondb", "--execute"])
        assert rc == 2

    def test_allow_production_against_local_target_refused(self, tmp_path):
        """The flag exists to make production writes deliberate, not routine."""
        rc = bf.main(
            ["--db-url", f"sqlite:///{tmp_path/'a.db'}", "--execute", "--allow-production"]
        )
        assert rc == 2

    def test_non_utc_acknowledgement_on_production_write_requires_production_ack(self):
        rc = bf.main(
            [
                "--db-url",
                "postgresql://u:p@db.neon.tech/neondb",
                "--execute",
                "--allow-non-utc-timezone",
            ]
        )
        assert rc == 2

    def test_local_write_needs_only_execute(self, tmp_path):
        validate = lambda **kw: bf.validate_invocation(  # noqa: E731
            db_url=f"sqlite:///{tmp_path/'a.db'}", allow_non_utc=False, **kw
        )
        # A local write needs only --execute.
        validate(execute=True, dry_run=False, allow_production=False)
        # A local dry run (the default) needs nothing at all.
        validate(execute=False, dry_run=False, allow_production=False)
        # Production acknowledgement is still meaningless without a write.
        with pytest.raises(bf.BackfillError, match="requires --execute"):
            bf.validate_invocation(
                db_url="postgresql://u:p@db.neon.tech/neondb",
                execute=False,
                dry_run=False,
                allow_production=True,
                allow_non_utc=False,
            )

    def test_non_utc_acknowledgement_without_execute_refused(self):
        """It waives a write-time guard, so it is inert - and misleading - on a dry run."""
        with pytest.raises(bf.BackfillError, match="silently inert"):
            bf.validate_invocation(
                db_url="sqlite:///./scratch.db",
                execute=False,
                dry_run=False,
                allow_production=False,
                allow_non_utc=True,
            )


# --------------------------------------------------------------------------- #
# Dry run is the default and writes nothing
# --------------------------------------------------------------------------- #


class TestDryRunDefault:
    def test_default_invocation_is_read_only(self, csv_factory, db_factory, monkeypatch):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        called: list = []
        monkeypatch.setattr(
            bf, "_apply_updates",
            lambda *a, **k: called.append(a) or 0,
        )

        assert bf.main(["--db-url", url, "--csv", str(csv)]) == 0
        assert called == [], "dry run reached the write path"

    def test_dry_run_leaves_every_row_untouched(self, csv_factory, db_factory):
        csv = csv_factory([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"},
            {"latitude": 30.6, "longitude": 76.2, "acq_date": "2026-09-05",
             "acq_time": 400, "satellite": "SNPP", "synthetic": ""},
        ])
        url = db_factory([
            ("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None),
            ("SNPP", 30.6, 76.2, "2026-09-05 04:00:00", 0, None),
        ])

        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert report["db_rows_updated"] == 0
        assert report["db_rows_pending"] == 2
        assert _rows(url) == [(30.5, 0, None), (30.6, 0, None)]

    def test_run_defaults_to_dry_run(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv))
        assert report["dry_run"] is True
        assert report["db_rows_updated"] == 0
        assert report["db_rows_pending"] == 1
        assert _rows(url) == [(30.5, 0, None)]

    def test_execute_writes_only_matched_rows(self, csv_factory, db_factory):
        csv = csv_factory([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"},
            {"latitude": 30.6, "longitude": 76.2, "acq_date": "2026-09-05",
             "acq_time": 400, "satellite": "SNPP", "synthetic": ""},
        ])
        url = db_factory([
            ("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None),
            ("SNPP", 30.6, 76.2, "2026-09-05 04:00:00", 0, None),
            ("AQUA", 1.0, 2.0, "2026-09-05 03:00:00", 0, None),
        ])

        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert report["db_rows_updated"] == 2
        assert report["db_rows_unmatched_left_untouched"] == 1
        assert _rows(url) == [
            (30.5, 1, "synthetic_sim"),
            (30.6, 0, "firms_csv"),
            (1.0, 0, None),
        ]

    def test_dry_run_reports_blast_radius_separately_from_writes(
        self, csv_factory, db_factory, capsys
    ):
        """A dry run must state what WOULD change without claiming to have changed it."""
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        assert bf.main(["--db-url", url, "--csv", str(csv)]) == 0
        out = capsys.readouterr().out
        assert "db_rows_pending" in out
        assert "1 row(s) would change" in out
        assert "no rows were written" in out

    def test_writes_are_batched_but_atomic(self, csv_factory, db_factory, monkeypatch):
        rows = [
            {"latitude": 30.0 + i * 0.01, "longitude": 76.0, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}
            for i in range(7)
        ]
        csv = csv_factory(rows)
        specs = [
            ("SNPP", 30.0 + i * 0.01, 76.0, "2026-09-05 03:00:00", 0, None)
            for i in range(7)
        ]
        url = db_factory(specs)

        batches: list[int] = []
        real_apply = bf._apply_updates

        def _counting_apply(engine, updates, batch_size=bf.UPDATE_BATCH_SIZE):
            batches.append(-(-len(updates) // batch_size))
            return real_apply(engine, updates, batch_size)

        monkeypatch.setattr(bf, "_apply_updates", _counting_apply)
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False, batch_size=3)
        assert batches == [3]  # 7 rows -> 3 batches of 3/3/1
        assert report["db_rows_pending"] == 7
        assert report["db_rows_updated"] == 7


# --------------------------------------------------------------------------- #
# Exact matching only
# --------------------------------------------------------------------------- #


class TestExactMatchingOnly:
    def test_near_miss_timestamp_stays_unmatched(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        # One minute later: must not match.
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:01:00", 0, None)])

        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert report["db_rows_matched"] == 0
        assert report["db_rows_unmatched_left_untouched"] == 1
        assert _rows(url) == [(30.5, 0, None)]

    def test_near_miss_coordinates_stay_unmatched(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        # 0.0001 deg north, i.e. just outside the 4-dp rounding.
        url = db_factory([("SNPP", 30.5001, 76.1, "2026-09-05 03:00:00", 0, None)])

        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert report["db_rows_matched"] == 0
        assert _rows(url) == [(30.5001, 0, None)]

    def test_near_miss_satellite_stays_unmatched(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("AQUA", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert report["db_rows_matched"] == 0

    def test_tz_aware_db_value_is_normalised_to_utc(self, csv_factory):
        """An aware TIMESTAMPTZ read-back must still match the naive key."""
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        sig = bf.load_csv_signatures(csv)
        key = next(iter(sig))
        aware = datetime(2026, 9, 5, 3, 0, tzinfo=UTC)
        assert bf._row_signature(key[0], key[1], key[2], aware) == key
        # ...and an equivalent instant in another zone still normalises.
        plus2 = datetime(2026, 9, 5, 5, 0, tzinfo=timezone(timedelta(hours=2)))
        assert bf._row_signature(key[0], key[1], key[2], plus2) == key

    def test_to_naive_utc_semantics_unchanged(self):
        """Guard the documented convention: aware -> UTC, naive -> as-is."""
        naive = datetime(2026, 9, 5, 3, 0)
        assert bf._to_naive_utc(naive) == naive
        assert bf._to_naive_utc(naive.replace(tzinfo=UTC)) == naive
        assert bf._to_naive_utc(None) is None
        assert bf._to_naive_utc("2026-09-05 03:00:00") == naive
        assert bf._to_naive_utc("not a date") is None


# --------------------------------------------------------------------------- #
# Duplicate / ambiguous signatures
# --------------------------------------------------------------------------- #


class TestDuplicateSignatures:
    def test_duplicate_key_with_conflicting_provenance_refused(self, csv_factory):
        csv = csv_factory([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0", "source": "synthetic_sim"},
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "", "source": "firms_csv"},
        ])
        with pytest.raises(bf.AmbiguousSignatureError, match="conflicting provenance"):
            bf.load_csv_signatures(csv)

    def test_duplicate_key_with_same_provenance_still_refused(self, csv_factory):
        csv = csv_factory([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"},
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"},
        ])
        with pytest.raises(bf.AmbiguousSignatureError, match="duplicate signature"):
            bf.load_csv_signatures(csv)

    def test_rounding_collision_is_detected(self, csv_factory):
        """5-dp coordinates that collapse to one 4-dp key must not pass silently."""
        csv = csv_factory([
            {"latitude": 30.60174, "longitude": 73.64675, "acq_date": "2026-09-05",
             "acq_time": 820, "satellite": "N", "synthetic": ""},
            {"latitude": 30.60175, "longitude": 73.64675, "acq_date": "2026-09-05",
             "acq_time": 820, "satellite": "N", "synthetic": "1.0"},
        ])
        with pytest.raises(bf.AmbiguousSignatureError):
            bf.load_csv_signatures(csv)

    def test_allow_duplicates_keeps_first_and_still_refuses_conflicts(self, csv_factory):
        benign = csv_factory([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0", "source": "synthetic_sim"},
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0", "source": "synthetic_sim"},
        ], name="benign.csv")
        sig = bf.load_csv_signatures(benign, allow_duplicates=True)
        assert len(sig) == 1
        assert next(iter(sig.values()))["source"] == "synthetic_sim"

        conflicting = csv_factory([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0", "source": "synthetic_sim"},
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "", "source": "firms_csv"},
        ], name="conflict.csv")
        with pytest.raises(bf.AmbiguousSignatureError):
            bf.load_csv_signatures(conflicting, allow_duplicates=True)

    def test_shipped_csv_has_no_duplicate_signatures(self):
        """The bundled FIRMS export must remain unambiguous."""
        if not bf.DEFAULT_FIRE_CSV.exists():  # pragma: no cover - data not present
            pytest.skip("bundled FIRMS CSV not present")
        sig = bf.load_csv_signatures(bf.DEFAULT_FIRE_CSV)
        assert len(sig) == 383_103
        assert sum(1 for v in sig.values() if v["synthetic"]) == 383_093


# --------------------------------------------------------------------------- #
# Provenance protection
# --------------------------------------------------------------------------- #


class TestProvenanceProtection:
    def test_firms_live_row_is_never_reclassified(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        # Same signature, but the live feed already owns this observation.
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, "firms_live")])

        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert report["db_rows_protected_live"] == 1
        assert report["db_rows_pending"] == 0
        assert report["db_rows_updated"] == 0
        # The CSV calls this synthetic, but we never act on it: protected rows
        # must not inflate the synthetic/real write-plan counters.
        assert report["db_rows_matched"] == 1
        assert report["db_rows_set_synthetic"] == 0
        assert report["db_rows_set_real"] == 0
        assert _rows(url) == [(30.5, 0, "firms_live")]

    def test_firms_live_protection_survives_execute(self, csv_factory, db_factory, capsys):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, "firms_live")])
        # Protected live rows are expected on a real target: a warning, exit 0.
        assert bf.main(["--db-url", url, "--csv", str(csv), "--execute"]) == 0
        assert "WARNING" in capsys.readouterr().out
        assert _rows(url) == [(30.5, 0, "firms_live")]

    def test_real_firms_row_receives_explicit_provenance(self, csv_factory, db_factory):
        """A reconciled real row must not be left with a NULL source."""
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "N", "synthetic": ""}]
        )
        url = db_factory([("N", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert report["db_rows_set_real"] == 1
        assert report["db_rows_legacy_null_source"] == 1
        assert report["db_rows_pending"] == 1
        assert report["db_rows_updated"] == 1
        assert _rows(url) == [(30.5, 0, "firms_csv")]

    def test_already_correct_real_row_is_unchanged(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "N", "synthetic": ""}]
        )
        url = db_factory([("N", 30.5, 76.1, "2026-09-05 03:00:00", 0, "firms_csv")])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert report["db_rows_unchanged"] == 1
        assert report["db_rows_pending"] == 0
        assert report["db_rows_updated"] == 0

    def test_legacy_null_row_is_stamped(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert report["db_rows_legacy_null_source"] == 1
        assert _rows(url) == [(30.5, 1, "synthetic_sim")]

    def test_unmatched_rows_are_reported_and_untouched(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("AQUA", 1.0, 2.0, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert report["db_rows_unmatched_left_untouched"] == 1
        assert report["db_rows_matched"] == 0
        assert _rows(url) == [(1.0, 0, None)]

    def test_schema_mismatch_refused_before_writes(self, tmp_path):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE fire_readings (id INTEGER PRIMARY KEY, "
            "satellite VARCHAR, latitude FLOAT, longitude FLOAT, acq_date DATETIME)"
        )
        conn.commit()
        conn.close()
        with pytest.raises(bf.BackfillError, match="missing provenance columns"):
            bf.run(f"sqlite:///{path}", {}, dry_run=True)


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #


class TestAnomalyDetection:
    def test_zero_match_is_an_anomaly(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("AQUA", 1.0, 2.0, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        anomalies = bf.check_expectations(report)
        assert any("db_rows_matched is 0" in a for a in anomalies)

    def test_empty_table_is_an_anomaly(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        anomalies = bf.check_expectations(report)
        assert any("db_rows_scanned is 0" in a for a in anomalies)

    def test_execute_updating_nothing_is_an_anomaly(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 1, "synthetic_sim")])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert report["db_rows_updated"] == 0
        assert any("updated 0 rows" in a for a in bf.check_expectations(report))

    def test_expectation_mismatch_is_an_anomaly(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert bf.check_expectations(report, expect_db_rows=382_960)
        assert bf.check_expectations(report, expect_matched=99)
        assert bf.check_expectations(report, min_matched=1) == []

    def test_main_returns_anomaly_exit_code(self, csv_factory, db_factory, capsys):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("AQUA", 1.0, 2.0, "2026-09-05 03:00:00", 0, None)])
        assert bf.main(["--db-url", url, "--csv", str(csv)]) == 3
        assert "ANOMALY" in capsys.readouterr().out

    def test_live_protection_is_surfaced_as_a_warning_not_an_error(
        self, csv_factory, db_factory
    ):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, "firms_live")])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        warnings = bf.postflight_warnings(report)
        assert any("firms_live" in w for w in warnings)
        assert bf.postflight_errors(report) == []
        assert bf.preflight_expectations(report) == []

    def test_unmatched_rows_are_a_warning_not_an_error(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([
            ("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None),
            ("AQUA", 1.0, 2.0, "2026-09-05 03:00:00", 0, None),
        ])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert any(
            "no CSV match" in w for w in bf.postflight_warnings(report)
        )
        assert report["db_rows_updated"] == 0

    def test_clean_run_has_no_anomalies(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert bf.check_expectations(report) == []


# --------------------------------------------------------------------------- #
# Expectations gate the write
# --------------------------------------------------------------------------- #


class TestExpectationsGate:
    def test_expectation_mismatch_blocks_execute_write(self, csv_factory, db_factory, capsys):
        """A failed --expect-* must leave the table untouched, not write then complain."""
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        rc = bf.main(
            ["--db-url", url, "--csv", str(csv), "--execute", "--expect-matched", "99"]
        )
        assert rc == 3
        out = capsys.readouterr().out
        assert "ANOMALY" in out
        assert "no rows were written" in out
        # The write never happened.
        assert _rows(url) == [(30.5, 0, None)]

    def test_zero_match_execute_writes_nothing(self, csv_factory, db_factory, capsys):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("AQUA", 1.0, 2.0, "2026-09-05 03:00:00", 0, None)])
        assert bf.main(["--db-url", url, "--csv", str(csv), "--execute"]) == 3
        assert "ANOMALY" in capsys.readouterr().out
        assert _rows(url) == [(1.0, 0, None)]

    def test_expectation_matching_execute_writes(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        rc = bf.main(
            [
                "--db-url", url, "--csv", str(csv), "--execute",
                "--expect-db-rows", "1", "--expect-matched", "1",
            ]
        )
        assert rc == 0
        assert _rows(url) == [(30.5, 1, "synthetic_sim")]

    def test_execute_updating_nothing_is_an_error_and_writes_nothing(
        self, csv_factory, db_factory
    ):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 1, "synthetic_sim")])
        assert bf.main(["--db-url", url, "--csv", str(csv), "--execute"]) == 3
        assert bf.postflight_errors(
            bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        ) == [
            "an --execute run updated 0 rows. That is a no-op, not a success: the "
            "table may already be reconciled, or nothing matched."
        ]


# --------------------------------------------------------------------------- #
# Engine lifecycle
# --------------------------------------------------------------------------- #


class TestEngineLifecycle:
    def test_engine_disposed_on_success(self, monkeypatch, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        disposed: list[bool] = []
        import sqlalchemy

        real = sqlalchemy.create_engine

        def _spy(u, *a, **kw):
            eng = real(u, *a, **kw)
            orig = eng.dispose

            def _dispose(*x, **y):
                disposed.append(True)
                return orig(*x, **y)

            eng.dispose = _dispose
            return eng

        monkeypatch.setattr(bf, "create_engine", _spy)
        bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert disposed

    def test_engine_disposed_on_schema_error(self, monkeypatch, tmp_path):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE fire_readings (id INTEGER PRIMARY KEY, "
            "satellite VARCHAR, latitude FLOAT, longitude FLOAT, acq_date DATETIME)"
        )
        conn.commit()
        conn.close()

        disposed: list[bool] = []
        import sqlalchemy

        real = sqlalchemy.create_engine

        def _spy(u, *a, **kw):
            eng = real(u, *a, **kw)
            orig = eng.dispose
            eng.dispose = lambda *x, **y: (disposed.append(True), orig(*x, **y))[1]
            return eng

        monkeypatch.setattr(bf, "create_engine", _spy)
        with pytest.raises(bf.BackfillError):
            bf.run(f"sqlite:///{path}", {})
        assert disposed

    def test_engine_disposed_when_update_fails(self, monkeypatch, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])

        disposed: list[bool] = []
        import sqlalchemy

        real = sqlalchemy.create_engine

        def _spy(u, *a, **kw):
            eng = real(u, *a, **kw)
            orig = eng.dispose
            eng.dispose = lambda *x, **y: (disposed.append(True), orig(*x, **y))[1]
            return eng

        monkeypatch.setattr(bf, "create_engine", _spy)
        monkeypatch.setattr(
            bf, "_apply_updates",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError, match="boom"):
            bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        assert disposed
        # The failed run left the row untouched.
        assert _rows(url) == [(30.5, 0, None)]

    def test_mid_write_failure_rolls_back_every_batch(self, csv_factory, db_factory,
                                                      monkeypatch):
        """A failure on the *second* batch must undo the *first*."""
        rows = [
            ("SNPP", 30.0 + i / 100, 76.1, "2026-09-05 03:00:00", 0, None)
            for i in range(7)
        ]
        url = db_factory(rows)
        csv = csv_factory(
            [
                {"latitude": lat, "longitude": 76.1, "acq_date": "2026-09-05",
                 "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}
                for _, lat, _, _, _, _ in rows
            ]
        )

        import sqlalchemy
        from sqlalchemy.exc import OperationalError

        real_execute = sqlalchemy.Connection.execute
        state = {"n": 0}

        def _fail_on_second_batch(self, stmt, *a, **kw):
            text_ = str(getattr(stmt, "text", stmt)).strip().upper()
            if text_.startswith("UPDATE"):
                state["n"] += 1
                if state["n"] == 2:
                    raise OperationalError(stmt, {}, Exception("disk on fire"))
            return real_execute(self, stmt, *a, **kw)

        monkeypatch.setattr(sqlalchemy.Connection, "execute", _fail_on_second_batch)

        with pytest.raises(bf.BackfillError, match="rolled back"):
            bf.run(url, bf.load_csv_signatures(csv), dry_run=False, batch_size=3)

        assert state["n"] == 2
        # The first batch succeeded inside the transaction, then was undone.
        assert _rows(url) == [(lat, 0, None) for _, lat, _, _, _, _ in rows]

    def test_read_failure_message_carries_no_credentials(self, csv_factory, monkeypatch):
        import sqlalchemy
        from sqlalchemy.exc import OperationalError

        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        secret_url = "postgresql://user:hunter2@db.neon.tech/neondb"

        def _boom(*a, **kw):
            raise OperationalError("SELECT 1", {}, Exception("connection to server failed"))

        monkeypatch.setattr(sqlalchemy.Engine, "connect", _boom, raising=False)
        with pytest.raises(bf.BackfillError) as exc:
            bf.run(secret_url, bf.load_csv_signatures(csv), dry_run=True)
        assert "hunter2" not in str(exc.value)
        assert "OperationalError" in str(exc.value)
        assert "db.neon.tech" in str(exc.value)


# --------------------------------------------------------------------------- #
# Timezone probe (read-only, SQLite has none)
# --------------------------------------------------------------------------- #


class TestTimezoneProbe:
    def test_probe_returns_not_applicable_on_sqlite_without_querying(self, db_factory):
        url = db_factory([])
        engine = bf.create_engine(url)
        try:
            with engine.connect() as conn:
                probe = bf.probe_session_timezone(conn, engine.dialect.name)
        finally:
            engine.dispose()
        assert probe.applicable is False
        assert probe.value is None

    def test_probe_uses_select_and_never_show_or_dml(self, db_factory, monkeypatch):
        """The probe must issue a SELECT only - never SHOW, DDL or DML."""
        url = db_factory([])
        engine = bf.create_engine(url)
        statements: list[str] = []
        try:
            with engine.connect() as conn:
                real_execute = conn.execute

                def spy(stmt, *a, **kw):
                    statements.append(str(getattr(stmt, "text", stmt)).strip().upper())
                    return real_execute(stmt, *a, **kw)

                monkeypatch.setattr(conn, "execute", spy)
                # SQLite has no session timezone: must not query at all.
                assert bf.probe_session_timezone(conn, engine.dialect.name) == (
                    bf.TimezoneProbe(applicable=False, value=None)
                )
                assert statements == []
                # PostgreSQL: a read-only SELECT (fails on SQLite, which is fine -
                # what matters is the statement that was attempted), and the
                # inconclusive result is reported rather than assumed.
                pg_probe = bf.probe_session_timezone(conn, "postgresql")
        finally:
            engine.dispose()

        assert len(statements) == 1, statements
        assert statements[0].startswith("SELECT"), statements[0]
        assert not statements[0].startswith(("SHOW", "INSERT", "UPDATE", "DELETE"))
        assert pg_probe.applicable is True
        assert pg_probe.value is None

    def test_report_carries_timezone_field(self, csv_factory, db_factory):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        report = bf.run(url, bf.load_csv_signatures(csv), dry_run=True)
        assert "db_session_timezone" in report
        assert report["db_session_timezone_known"] is True


class TestTimezoneGuard:
    """The fail-open hole: an inconclusive probe must not silently allow a write."""

    def test_unknown_timezone_blocks_production_execute(self):
        message = bf.timezone_guard_error(
            timezone=None,
            timezone_known=False,
            dry_run=False,
            production=True,
            allow_non_utc=False,
        )
        assert message and "could not determine" in message

    def test_unknown_timezone_allowed_on_dry_run_local_and_with_ack(self):
        assert bf.timezone_guard_error(
            timezone=None, timezone_known=False, dry_run=True,
            production=True, allow_non_utc=False,
        ) is None
        assert bf.timezone_guard_error(
            timezone=None, timezone_known=False, dry_run=False,
            production=False, allow_non_utc=False,
        ) is None
        assert bf.timezone_guard_error(
            timezone=None, timezone_known=False, dry_run=False,
            production=True, allow_non_utc=True,
        ) is None

    def test_non_utc_blocks_execute_without_ack(self):
        message = bf.timezone_guard_error(
            timezone="Asia/Kolkata", timezone_known=True, dry_run=False,
            production=False, allow_non_utc=False,
        )
        assert message and "not UTC" in message
        assert bf.timezone_guard_error(
            timezone="Asia/Kolkata", timezone_known=True, dry_run=False,
            production=False, allow_non_utc=True,
        ) is None

    def test_utc_and_missing_timezone_allow_execute(self):
        for zone in ("UTC", "Etc/UTC", None):
            assert bf.timezone_guard_error(
                timezone=zone, timezone_known=True, dry_run=False,
                production=True, allow_non_utc=False,
            ) is None

    def test_run_fails_closed_on_inconclusive_production_probe(
        self, csv_factory, db_factory, monkeypatch
    ):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        monkeypatch.setattr(bf, "is_production_target", lambda _u: True)
        monkeypatch.setattr(
            bf, "probe_session_timezone",
            lambda *a, **k: bf.TimezoneProbe(applicable=True, value=None),
        )
        with pytest.raises(bf.BackfillError, match="could not determine"):
            bf.run(url, bf.load_csv_signatures(csv), dry_run=False)
        # The refusal happened before the write: the row is untouched.
        assert _rows(url) == [(30.5, 0, None)]

    def test_run_proceeds_on_inconclusive_probe_with_ack(
        self, csv_factory, db_factory, monkeypatch
    ):
        csv = csv_factory(
            [{"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
              "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0"}]
        )
        url = db_factory([("SNPP", 30.5, 76.1, "2026-09-05 03:00:00", 0, None)])
        monkeypatch.setattr(bf, "is_production_target", lambda _u: True)
        monkeypatch.setattr(
            bf, "probe_session_timezone",
            lambda *a, **k: bf.TimezoneProbe(applicable=True, value=None),
        )
        report = bf.run(
            url, bf.load_csv_signatures(csv), dry_run=False, allow_non_utc_timezone=True
        )
        assert report["db_rows_updated"] == 1
        assert _rows(url) == [(30.5, 1, "synthetic_sim")]
