"""Fire-data provenance: writers stamp synthetic/source, backfill is exact-match.

SIH26082: simulated 2023-24 stubble-fire history must be explicitly labelled
``synthetic`` with ``source="synthetic_sim"`` at every ingestion point, live
FIRMS records must be stamped real, and the backfill script must only rewrite
rows whose full uniqueness signature matches a CSV row exactly — never a guess.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from app.models.db_models import FireReading
from app.services import firms_service as fs

from backend.scripts import backfill_fire_provenance as bf
from backend.scripts import load_data as ld
from scripts.generate_fire_data import generate_fires_for_date


def _today():
    return datetime.now(UTC).replace(tzinfo=None)


def _fire_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _stamped_fire(synthetic=True, source="synthetic_sim", sat="SNPP", n=2):
    base = _today() - timedelta(hours=3)
    return [
        {
            "latitude": round(30.1 + i * 0.1, 4),
            "longitude": round(76.1 + i * 0.1, 4),
            "acq_date": (base + timedelta(hours=i)).strftime("%Y-%m-%d"),
            "acq_time": 1200 + i * 100,
            "confidence": "high",
            "frp": 90.0 + i,
            "bright_ti4": 311.0 + i,
            "satellite": sat,
            "daynight": "D",
            "synthetic": ["", "1.0"][synthetic],
            "source": source,
        }
        for i in range(n)
    ]


class TestGeneratorLabels:
    def test_generated_rows_are_explicitly_synthetic(self):
        records = generate_fires_for_date(datetime(2023, 10, 15).date(), total_budget=5)
        assert records
        for rec in records:
            assert rec["synthetic"] is True
            assert rec["source"] == "synthetic_sim"


class TestLoadDataStampsProvenance:
    def test_synthetic_rows_are_flagged(self, db_session, tmp_path):
        path = tmp_path / "fires.csv"
        _fire_csv(path, _stamped_fire(synthetic=True))
        ld.load_fire_data(db_session, path, chunksize=1)
        rows = db_session.query(FireReading).filter(
            FireReading.synthetic.is_(True)
        ).all()
        assert len(rows) == 2
        assert {r.source for r in rows} == {"synthetic_sim"}
        assert all(r.brightness is not None for r in rows)
        assert all(r.instrument is not None for r in rows)

    def test_real_rows_stamp_firms_csv_default(self, db_session, tmp_path):
        path = tmp_path / "fires.csv"
        # The CSV has an explicit `synthetic` column with empty values -> real.
        _fire_csv(path, _stamped_fire(synthetic=False, source=""))
        ld.load_fire_data(db_session, path, chunksize=1)
        rows = db_session.query(FireReading).filter(
            FireReading.latitude.in_([30.1, 30.2]), FireReading.synthetic.is_(False)
        ).all()
        assert len(rows) == 2
        assert {r.source for r in rows} == {"firms_csv"}

    def test_csv_without_synthetic_column_defaults_real(self, db_session, tmp_path):
        """Legacy CSV shape (no provenance columns) must not be misread as synthetic."""
        path = tmp_path / "fires.csv"
        rows = [
            {"latitude": 30.1, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "confidence": "high", "frp": 90.0,
             "satellite": "SNPP", "daynight": "D"}
        ]
        _fire_csv(path, rows)
        inserted = ld.load_fire_data(db_session, path, chunksize=2)
        assert inserted == 1
        row = db_session.query(FireReading).filter(
            FireReading.latitude == 30.1, FireReading.longitude == 76.1
        ).first()
        assert row is not None
        assert row.synthetic is False
        assert row.source == "firms_csv"


class TestFirmsServiceStampsProvenance:
    def test_normalise_marks_live_records_real(self):
        df = pd.DataFrame([{
            "latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-07",
            "acq_time": 1000, "confidence": "high", "frp": 90.0,
            "satellite": "SNPP", "instrument": "VIIRS", "daynight": "D",
        }])
        records = fs.normalise_fire_records(df)
        assert records[0]["synthetic"] is False
        assert records[0]["source"] == "firms_live"

    def test_upsert_writes_provenance(self, db_session):
        records = [{
            "satellite": "SNPP", "instrument": "VIIRS", "latitude": 29.0,
            "longitude": 75.0, "acq_date": datetime(2026, 9, 7, 10, 0, 0),
            "confidence": "nominal", "frp": 50.0, "brightness": None,
            "daynight": "D", "synthetic": False, "source": "firms_live",
        }]
        fs.upsert_fire_records(db_session, records, dry_run=False)
        row = db_session.query(FireReading).filter(
            FireReading.latitude == 29.0, FireReading.longitude == 75.0
        ).first()
        assert row.synthetic is False
        assert row.source == "firms_live"

    def test_upsert_defaults_unknown_provenance_to_real(self, db_session):
        """A record without provenance keys is treated as a live FIRMS row, not synthetic."""
        records = [{
            "satellite": "SNPP", "instrument": "VIIRS", "latitude": 28.5,
            "longitude": 76.5, "acq_date": datetime(2026, 9, 7, 11, 0, 0),
            "confidence": "high", "frp": 60.0, "brightness": None, "daynight": "D",
        }]
        fs.upsert_fire_records(db_session, records, dry_run=False)
        row = db_session.query(FireReading).filter(
            FireReading.latitude == 28.5, FireReading.longitude == 76.5
        ).first()
        assert row.synthetic is False
        assert row.source == "firms_live"


def _backfill_db(tmp_path, row_specs):
    """Minimal fire_readings table carrying the provenance columns."""
    from sqlalchemy import create_engine, text

    url = f"sqlite:///{tmp_path / 'backfill.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE fire_readings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "satellite VARCHAR, latitude FLOAT NOT NULL, longitude FLOAT NOT NULL, "
            "acq_date DATETIME NOT NULL, synthetic BOOLEAN NOT NULL DEFAULT 0, "
            "source VARCHAR)"
        ))
        for spec in row_specs:
            conn.execute(
                text(
                    "INSERT INTO fire_readings "
                    "(satellite, latitude, longitude, acq_date, synthetic, source) "
                    "VALUES (:sat, :lat, :lon, :ts, :syn, :src)"
                ),
                {
                    "sat": spec[0], "lat": spec[1], "lon": spec[2],
                    "ts": spec[3], "syn": spec[4], "src": spec[5],
                },
            )
    engine.dispose()
    return url


class TestBackfillExactMatch:
    def _csv_and_db(self, tmp_path):
        # Signature rows: one synthetic, one real.
        df = pd.DataFrame([
            {"latitude": 30.5, "longitude": 76.1, "acq_date": "2026-09-05",
             "acq_time": 300, "satellite": "SNPP", "synthetic": "1.0", "source": ""},
            {"latitude": 30.6, "longitude": 76.2, "acq_date": "2026-09-05",
             "acq_time": 400, "satellite": "SNPP", "synthetic": "", "source": ""},
        ])
        csv_path = tmp_path / "sources.csv"
        df.to_csv(csv_path, index=False)

        matched_ts = bf._reconstruct_acq_time("2026-09-05", 300)
        url = _backfill_db(tmp_path, [
            # matches CSV row 1: currently real -> should become synthetic
            ("SNPP", 30.5, 76.1, matched_ts, False, None),
            # matches CSV row 2: real, but legacy source IS NULL -> gets stamped
            # ``firms_csv`` so no reconciled row keeps an ambiguous provenance.
            ("SNPP", 30.6, 76.2, bf._reconstruct_acq_time("2026-09-05", 400), False, None),
            # no CSV match -> must be left untouched
            ("AQUA", 1.0, 2.0, matched_ts, False, None),
        ])
        return csv_path, url

    def test_dry_run_writes_nothing(self, tmp_path):
        csv_path, url = self._csv_and_db(tmp_path)
        sig = bf.load_csv_signatures(csv_path)
        report = bf.run(url, sig, dry_run=True)
        assert report["db_rows_scanned"] == 3
        assert report["db_rows_matched"] == 2
        assert report["db_rows_unmatched_left_untouched"] == 1
        # Both matched rows need a change: the synthetic one the flag, the real
        # one its NULL source.
        assert report["db_rows_pending"] == 2
        assert report["db_rows_updated"] == 0
        assert report["db_rows_set_synthetic"] == 1
        assert report["db_rows_set_real"] == 1
        assert report["db_rows_legacy_null_source"] == 2

        from sqlalchemy import create_engine, text
        engine = create_engine(url)
        with engine.connect() as conn:
            vals = conn.execute(text(
                "SELECT satellite, synthetic, source FROM fire_readings "
                "WHERE latitude = 30.5 AND longitude = 76.1"
            )).first()
        engine.dispose()
        assert vals is not None
        assert not vals[1]  # dry run left the flag untouched (0 = real)
        assert vals[2] is None

    def test_live_run_updates_exactly_the_matched_rows(self, tmp_path):
        csv_path, url = self._csv_and_db(tmp_path)
        sig = bf.load_csv_signatures(csv_path)
        report = bf.run(url, sig, dry_run=False)
        assert report["db_rows_pending"] == 2
        assert report["db_rows_updated"] == 2

        from sqlalchemy import create_engine, text
        engine = create_engine(url)
        with engine.connect() as conn:
            rows = {
                float(r[0]): (r[1], r[2])
                for r in conn.execute(text(
                    "SELECT latitude, synthetic, source FROM fire_readings"
                ))
            }
        engine.dispose()
        # matched synthetic row rewritten
        assert rows[30.5] == (1, "synthetic_sim")
        # matched real row: flag was already right, source was explicitly stamped
        assert rows[30.6] == (0, "firms_csv")
        # unmatched row completely untouched
        assert rows[1.0] == (0, None)

    def test_already_stamped_rows_are_unchanged(self, tmp_path):
        """A second run over a reconciled table writes nothing."""
        csv_path, url = self._csv_and_db(tmp_path)
        sig = bf.load_csv_signatures(csv_path)
        bf.run(url, sig, dry_run=False)
        report = bf.run(url, sig, dry_run=False)
        assert report["db_rows_updated"] == 0
        assert report["db_rows_pending"] == 0
        assert report["db_rows_unchanged"] == 2
        assert bf.check_expectations(report) != []
        assert "updated 0 rows" in " ".join(bf.check_expectations(report))

    def test_refuses_missing_provenance_columns(self, tmp_path):
        from sqlalchemy import create_engine, text

        url = f"sqlite:///{tmp_path / 'old.db'}"
        engine = create_engine(url)
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE fire_readings (id INTEGER PRIMARY KEY, "
                "satellite VARCHAR, latitude FLOAT, longitude FLOAT, acq_date DATETIME)"
            ))
        engine.dispose()

        with pytest.raises(RuntimeError, match="missing provenance columns"):
            bf.run(url, {}, dry_run=True)
