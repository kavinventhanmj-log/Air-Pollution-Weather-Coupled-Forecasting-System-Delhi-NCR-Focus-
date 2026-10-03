"""Backfill historical CPCB pollution for the 17-curated NCR stations.

Uses the real CPCB hourly-AQI dataset published by the community mirror
``Vonter/india-cpcb-aqi`` (ODbL, "sourced from the CPCB Data Repository and
CPCB AQI Repository"). The hourly AQI file is small and keyless.

- Idempotent: rows keyed on ``(station_id, timestamp)`` (the same unique
  constraint the live ingest uses); duplicates are skipped.
- Provenance: every inserted row is tagged ``data_source='cpcb_dataset'`` so
  the ``/api/pollution/coverage`` report can attribute it.
- Honest: this dataset carries hourly *AQI only*. Pollutant concentrations
  stay NULL; the per-station live ingest (data.gov.in with a personal key) is
  the concentration source. The script never invents values.

Usage:
    python -m scripts.backfill_pollution [--file data/pollution/cpcb-aqi.csv.gz]
        --file : use a locally-downloaded copy instead of fetching it.
        --no-download : never hit the network.
"""

import argparse
import gzip
import io
import os
import pathlib
import sys
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer a DATABASE_URL on the CLI/env; otherwise reuse the project .env.
_env_file = ROOT / ".env"
if not os.environ.get("DATABASE_URL") and _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL=") and not line.startswith("#"):
            os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

from backend.app.database import SessionLocal  # noqa: E402
from backend.app.models.db_models import PollutionReading, Station  # noqa: E402

DATASET_URL = (
    "https://raw.githubusercontent.com/Vonter/india-cpcb-aqi/main/data/cpcb-aqi.csv.gz"
)
SOURCE_TAG = "cpcb_dataset"

# The CPCB hourly-AQI mirror publishes its ``Date`` column and its ``HH:MM:SS``
# hour columns as India local wall-clock, with no offset. Every naive timestamp
# the application persists is UTC (``cpcb_service.upsert_ncr_data``,
# ``app.api.summary``), so these are interpreted as IST and converted rather than
# having their tzinfo stripped - storing the wall-clock unchanged would shift every
# backfilled reading 5 h 30 m into the future.
IST = ZoneInfo("Asia/Kolkata")

# Dataset "Station Name" -> canonical 17-station row. Common agency suffixes /
# city locators (", Delhi - DPCC" etc.) are stripped first by _normalise.
_HARD_ALIASES = {
    "sector 11": "Faridabad",
    "sector 62": "Noida Sector-62",
    "sector - 62": "Noida Sector-62",
    "teri gram": "Teri Gram",
}


def _normalise(station_name: str) -> str:
    """Strip CPCB 'X, City - Agency' suffixes and match a canonical name."""
    short = station_name
    for token in (" - DPCC", "- DPCC", " - CPCB", "- CPCB", " - HSPCB", "- HSPCB", " - UPPCB", "- UPPCB",
                  " - IMD", "- IMD"):
        if token in short:
            short = short.split(token)[0]
    short = short.split(",")[0].strip()
    return _HARD_ALIASES.get(short.lower(), short)


def download_dataset(output_dir: pathlib.Path) -> pathlib.Path | None:
    path = output_dir / "cpcb-aqi.csv.gz"
    print(f"  Downloading {DATASET_URL} ...")
    try:
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "AeroCast-NCR/1.0 (SIH26082)"})
        data = urllib.request.urlopen(req, timeout=120).read()
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return None
    path.write_bytes(data)
    return path


def load_frame(path: pathlib.Path) -> pd.DataFrame | None:
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as fh:
                df = pd.read_csv(io.BytesIO(fh.read()))
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        print(f"  Could not read {path}: {exc}")
        return None
    if "Station Name" not in df.columns or "Date" not in df.columns:
        print("  Dataset missing 'Station Name' / 'Date' columns (schema drift?)")
        return None
    return df


def hour_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if len(c) == 8 and c[2] == ":" and c[5] == ":"]


def join_records(df: pd.DataFrame, stations: dict[str, Station]):
    """Yield (station, y, m, d, h, aqi) INSERT-worthy tuples, IST wall-clock."""
    cols = hour_columns(df)
    seen = 0
    for _, row in df.iterrows():
        canonical = _normalise(str(row.get("Station Name") or ""))
        station = stations.get(canonical)
        if station is None:
            continue
        date_str = str(row.get("Date") or "").strip()
        if len(date_str) < 8:
            continue
        try:
            y, m, d = int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10])
        except (TypeError, ValueError):
            continue
        for col in cols:
            try:
                h = int(col[:2])
                aqi = int(float(row.get(col)))
            except (TypeError, ValueError):
                continue
            if aqi != aqi or aqi < 0:
                continue
            seen += 1
            yield station, y, m, d, h, aqi
        if seen > 0 and seen % 5000 == 0:
            print(f"    ...{seen} candidate hourly cells scanned")


def upsert(df: pd.DataFrame, db, stations: dict[str, Station]) -> dict:
    existing = {
        (sid, ts.replace(tzinfo=None) if ts.tzinfo else ts)
        for (sid, ts) in db.query(PollutionReading.station_id, PollutionReading.timestamp).all()
    }
    inserted = skipped = 0
    for station, y, m, d, h, aqi in join_records(df, stations):
        # The dataset's wall-clock is IST; the application's storage convention is
        # naive UTC, so localise as IST and convert rather than leaving the
        # wall-clock in place.
        ts = (
            pd.Timestamp(year=y, month=m, day=d, hour=h, tz=IST)
            .tz_convert("UTC")
            .tz_localize(None)
            .to_pydatetime()
        )
        key = (station.id, ts)
        if key in existing:
            skipped += 1
            continue
        db.add(PollutionReading(station_id=station.id, timestamp=ts, aqi=aqi, data_source=SOURCE_TAG))
        existing.add(key)
        inserted += 1
        if inserted % 2000 == 0:
            db.commit()
    db.commit()
    return {"inserted": inserted, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=pathlib.Path, default=None, help="Local cpcb-aqi.csv[.gz]")
    parser.add_argument("--no-download", action="store_true", help="Never hit the network")
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "data" / "pollution")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        path = args.file
        if path is None:
            if args.no_download:
                print("Nothing to do: no local file supplied and --no-download set.")
                return 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = download_dataset(args.output_dir)
        if path is None or not path.exists():
            print("No dataset available (network-gated). Run again when online, or pass --file.")
            return 1
        df = load_frame(path)
        if df is None:
            return 1
        stations = {s.name: s for s in db.query(Station).all()}
        print(f"Stations in DB: {len(stations)}")
        result = upsert(df, db, stations)
        print(f"Rows inserted: {result['inserted']:,}  skipped(dup): {result['skipped']:,}")
        covered = db.query(PollutionReading.station_id).filter(
            PollutionReading.data_source == SOURCE_TAG
        ).distinct().count()
        print(f"Stations now with a cpcb_dataset reading: {covered}/{len(stations)}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
