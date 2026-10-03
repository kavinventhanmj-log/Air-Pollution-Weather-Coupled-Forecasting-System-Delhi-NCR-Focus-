"""Load the real coupled dataset + trained metrics into the AeroCast-NCR backend DB.

Usage:
    python -m backend.scripts.load_data [--csv data/processed/coupled_dataset.csv]

Memory note: both CSV loaders stream in chunks and insert in small batches so
a 512 MB free-tier instance can hydrate the demo DB without being OOM-killed,
and so a single Neon-free-tier statement/connection hiccup skips only one
batch instead of aborting the whole hydration.
"""

import argparse
import gc
import json
import logging
from datetime import UTC
from pathlib import Path

import pandas as pd

import backend.app.models.db_models as dbm  # noqa: E402, F401  (registers tables)
from backend.app.database import Base, SessionLocal, engine  # noqa: E402

logger = logging.getLogger("aerocast.load_data")

# Natural unique keys per model; used for idempotent ``ON CONFLICT DO NOTHING``
# inserts so re-hydrating a partially-filled database never raises or drops data.
_CONFLICT_COLUMNS = {
    "PollutionReading": ["station_id", "timestamp"],
    "WeatherReading": ["station_id", "timestamp"],
    "FireReading": ["satellite", "latitude", "longitude", "acq_date"],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "coupled_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def load_coupled_data(db, csv_path: Path, chunksize: int = 25_000) -> dict:
    from backend.app.models.db_models import PollutionReading, Station, WeatherReading
    from backend.app.services.aqi_calculator import calculate_aqi

    stations = {s.name: s for s in db.query(Station).all()}
    station_coords = {
        "Anand Vihar": (28.6492, 77.2918),
        "RK Puram": (28.5601, 77.1835),
        "ITO": (28.6290, 77.2410),
        "Dwarka": (28.5921, 77.0460),
        "Punjabi Bagh": (28.6692, 77.1285),
        "Lodhi Road": (28.5866, 77.2268),
        "Sirifort": (28.5528, 77.2190),
        "Shadipur": (28.6542, 77.1489),
        "Okhla Phase-2": (28.5230, 77.2680),
        "Ashok Vihar": (28.6974, 77.1756),
        "Mundka": (28.6796, 77.0189),
        "Jahangirpuri": (28.7256, 77.1556),
        "Aya Nagar": (28.4771, 77.1148),
        "Vivek Vihar": (28.6727, 77.3169),
        "Teri Gram": (28.4422, 77.0115),
        "Noida Sector-62": (28.6227, 77.3615),
        "Faridabad": (28.4089, 77.3178),
    }
    raw_to_display = {
        "Anand_Vihar": "Anand Vihar",
        "RK_Puram": "RK Puram",
        "Punjabi_Bagh": "Punjabi Bagh",
    }

    for name, (lat, lon) in station_coords.items():
        if name not in stations:
            st = Station(name=name, latitude=lat, longitude=lon, city="Delhi NCR")
            db.add(st)
            db.flush()
            stations[name] = st

    existing_ts: dict[int, set] = {}
    for sid, ts in db.query(PollutionReading.station_id, PollutionReading.timestamp).all():
        existing_ts.setdefault(sid, set()).add(_as_utc(ts))

    total_poll = 0
    total_wx = 0
    for df in pd.read_csv(csv_path, parse_dates=["timestamp"], chunksize=chunksize):
        ts_col = df["timestamp"]
        if ts_col.dt.tz is None:
            df["timestamp"] = ts_col.dt.tz_localize("UTC")
        else:
            df["timestamp"] = ts_col.dt.tz_convert("UTC")
        df = df[df["timestamp"].notna()]
        df["station_name"] = df["station"].map(raw_to_display).fillna(df["station"])
        df["station_id"] = df["station_name"].map({s.name: s.id for s in stations.values()})
        df = df[df["station_id"].notna()]
        if df.empty:
            continue

        # Vectorised dedup vs. what is already persisted (kept light per chunk).
        keep = [
            ts not in existing_ts.get(int(sid), set())
            for sid, ts in zip(df["station_id"], df["timestamp"], strict=True)
        ]
        df = df[keep]
        if df.empty:
            continue

        for col in ("pm25", "pm10", "o3", "no2", "so2", "co"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        aqi = [calculate_aqi(head.pm25, head.pm10, head.o3, head.no2, head.so2, head.co)[0] for head in df.itertuples(index=False)]

        poll_payloads = []
        weather_payloads = []
        seen: set[tuple[int, object]] = set()
        for i, head in enumerate(df.itertuples(index=False)):
            sid = int(head.station_id)
            key = (sid, head.timestamp)
            if key in seen:
                # Coupled CSV carries a few duplicated station-hour rows; dropping
                # them here (instead of letting a later batch violate the Postgres
                # unique constraint) avoids losing the whole 2k batch.
                continue
            seen.add(key)
            poll_payloads.append({
                "station_id": sid,
                "timestamp": head.timestamp,
                "pm25": _clean(head.pm25),
                "pm10": _clean(head.pm10),
                "o3": _clean(head.o3),
                "no2": _clean(head.no2),
                "so2": _clean(head.so2),
                "co": _clean(head.co),
                "aqi": aqi[i],
            })
            weather_payloads.append({
                "station_id": sid,
                "timestamp": head.timestamp,
                "temperature": _clean(getattr(head, "temperature_2m", None)),
                "humidity": _clean(getattr(head, "relative_humidity_2m", None)),
                "pressure_msl": _clean(getattr(head, "pressure_msl", None)),
                "surface_pressure": _clean(getattr(head, "surface_pressure", None)),
                "wind_speed": _clean(getattr(head, "wind_speed_10m", None)),
                "wind_direction": _clean(getattr(head, "wind_direction_10m", None)),
                "precipitation": _clean(getattr(head, "precipitation", None)),
                "cloud_cover": _clean(getattr(head, "cloud_cover", None)),
                "pbl_height": _clean(getattr(head, "boundary_layer_height", None)),
            })

        total_poll += _insert_batches(db, PollutionReading, poll_payloads)
        total_wx += _insert_batches(db, WeatherReading, weather_payloads)

        for sid, ts in seen:
            existing_ts.setdefault(sid, set()).add(ts)

        del df, keep, poll_payloads, weather_payloads, aqi, seen
        if chunksize < 100_000:
            gc.collect()

    return {"pollution": total_poll, "weather": total_wx}


def load_fire_data(db, fire_csv: Path, chunksize: int = 50_000) -> int:
    """Load the fire CSV into ``FireReading``, preserving provenance.

    The CSV carries simulated stubble-fire history (``synthetic=1.0`` /
    ``source=synthetic_sim``) plus real FIRMS rows. Both are stored with their
    provenance so no simulated observation is ever presented as a live FIRMS
    detection: simulated rows are stamped ``synthetic=True`` and real rows
    ``synthetic=False`` (source ``firms_csv`` unless the CSV says otherwise).
    """
    if not fire_csv.exists():
        return 0
    from backend.app.models.db_models import FireReading

    existing = {
        (r.latitude, r.longitude)
        for r in db.query(FireReading.latitude, FireReading.longitude).all()
    }

    total = 0
    for df in pd.read_csv(fire_csv, low_memory=False, chunksize=chunksize):
        acq_time = (
            pd.to_numeric(df.get("acq_time"), errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        dt = pd.to_datetime(
            df["acq_date"].astype(str) + " " + acq_time,
            format="%Y-%m-%d %H%M",
            errors="coerce",
        )
        dt = dt.fillna(pd.to_datetime(df["acq_date"], errors="coerce"))

        lat = pd.to_numeric(df.get("latitude"), errors="coerce")
        lon = pd.to_numeric(df.get("longitude"), errors="coerce")

        synthetic = _col(df, "synthetic")
        source = _col(df, "source")
        bright = _col(df, "bright_ti4", _col(df, "bright_t31"))

        payloads = []
        for ax, ny, ts, conf, frp_v, sat, dn, syn_v, src_v, br in zip(
            lat,
            lon,
            dt,
            df.get("confidence", pd.Series(dtype="str")),
            df.get("frp"),
            df.get("satellite", pd.Series(dtype="str")),
            df.get("daynight", pd.Series(dtype="str")),
            synthetic,
            source,
            bright,
            strict=True,
        ):
            key = (_clean(ax), _clean(ny))
            if key[0] is None or key[1] is None or key in existing:
                continue
            ts_v = None if pd.isna(ts) else ts
            syn_flag = _as_bool(syn_v)
            payloads.append({
                "latitude": key[0],
                "longitude": key[1],
                "acq_date": ts_v,
                "confidence": "" if conf is None else str(conf),
                "frp": _clean(frp_v),
                "satellite": _clean(sat),
                "brightness": _clean(br),
                "instrument": _instrument_for_satellite(_clean(sat)),
                "daynight": _clean(dn),
                "synthetic": syn_flag,
                "source": _clean(src_v) or ("synthetic_sim" if syn_flag else "firms_csv"),
            })
            existing.add(key)

        if payloads:
            total += _insert_batches(db, FireReading, payloads)

        del df, payloads, dt
        if chunksize < 100_000:
            gc.collect()

    return total


def load_metrics(db, metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    from backend.app.models.db_models import ModelMetrics

    existing = {
        (m.model_name, m.pollutant, m.horizon_hours)
        for m in db.query(ModelMetrics.model_name, ModelMetrics.pollutant, ModelMetrics.horizon_hours).all()
    }

    with open(metrics_path, encoding="utf-8") as fh:
        entries = json.load(fh)

    count = 0
    for e in entries:
        key = (e["model"], e["target"], e["horizon"])
        if key in existing:
            continue
        existing.add(key)
        db.add(ModelMetrics(
            model_name=e["model"],
            pollutant=e["target"],
            horizon_hours=e["horizon"],
            mae=e["test_mae"],
            rmse=e["test_rmse"],
            r2=e["test_r2"],
            mape=e["test_mape"],
        ))
        count += 1
    db.commit()
    return count


def load_alerts_seed(db) -> int:
    from backend.app.models.db_models import Alert, Station

    stations = {s.name: s.id for s in db.query(Station).all()}
    if not stations:
        return 0
    existing = db.query(Alert).count()
    if existing > 0:
        return 0

    cozy = stations.get("Anand Vihar", 1)
    alerts = [
        Alert(
            station_id=cozy,
            alert_level="INFO",
            title="Monitoring Active",
            description="Delhi NCR monitoring network reporting hourly conditions.",
            recommendation="Continue monitoring hourly forecasts.",
        ),
        Alert(
            station_id=cozy,
            alert_level="WATCH",
            title="Winter Regime Active",
            description="Cold-season conditions favour pollution accumulation; inversion and PBL trends are being tracked.",
            factors="Seasonal meteorological conditions",
            recommendation="Review inversion strength and fire activity panels.",
        ),
    ]
    db.add_all(alerts)
    db.commit()
    return len(alerts)


def main():
    parser = argparse.ArgumentParser(description="Load real CPCB/Open-Meteo/FIRMS data into backend DB.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--fire-csv", default=str(PROJECT_ROOT / "data" / "fire" / "firms_fires.csv"))
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables.")
    args = parser.parse_args()

    if args.reset:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        print("Loading coupled dataset ...")
        counts = load_coupled_data(db, Path(args.csv))
        print(f"  Pollution readings : {counts['pollution']:,}")
        print(f"  Weather readings   : {counts['weather']:,}")

        print("Loading fire data ...")
        n_fire = load_fire_data(db, Path(args.fire_csv))
        print(f"  Fire records       : {n_fire:,}")

        print("Loading model metrics ...")
        n_metrics = load_metrics(db, MODELS_DIR / "metrics.json")
        print(f"  Metric entries     : {n_metrics:,}")

        print("Seeding alerts ...")
        n_alerts = load_alerts_seed(db)
        print(f"  Seed alerts        : {n_alerts:,}")
    finally:
        db.close()

    print("\nBackend database load complete.")


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v


def _col(df, name, fallback=None):
    """Return a column, or a same-length single-value Series so zips align."""
    if name in df.columns:
        return df[name]
    if fallback is not None:
        return fallback
    return pd.Series(None, dtype="object", index=df.index)


def _as_bool(v) -> bool:
    """Parse a CSV provenance flag ('1.0', '', True, 'true', ...) to a bool."""
    if v is None or (isinstance(v, float) and v != v):
        return False
    return str(v).strip().lower() in {"1", "1.0", "true", "t", "yes", "y"}


def _instrument_for_satellite(satellite: str | None) -> str | None:
    """Infer the FIRMS instrument from the satellite when the CSV omits it."""
    if not satellite:
        return None
    if satellite.strip() in {"Aqua", "Terra", "MODIS"}:
        return "MODIS"
    return "VIIRS"


def _as_utc(ts):
    """Normalise a datetime to tz-aware UTC (naive -> UTC) for dedup keys.

    pandas parses `parse_dates` columns as naive, but Postgres `timestamptz`
    columns read back tz-aware UTC. `naive != aware` silently defeats the
    existing-row dedup, so every run resubmits already-persisted rows and the
    unique constraint turns each rerun into a losing batch of integrity errors.
    """
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _insert_batches(db, model, payloads, batch_size: int = 2_000) -> int:
    """Insert payload rows in small idempotent batches.

    Each batch is committed on its own and uses ``ON CONFLICT DO NOTHING`` on the
    model's natural unique key, so reruns against a partially-hydrated database
    (or rows with legacy timezone representations) never raise integrity errors
    and never lose a whole batch. A batch-level failure (Neon-free-tier limits,
    timeouts) still skips only that batch, logs it, and lets the rest finish.

    Batches are issued as a *single multi-VALUES statement* rather than an
    executemany loop: executemany sends one round-trip per row, which costs
    ~0.06 s/row against a remote Neon database (a 2,000-row batch took 129 s),
    while one ``INSERT ... VALUES (...),(...)...`` statement over the same rows
    takes ~0.5 s. This is what keeps demo hydration feasible on a free-tier
    Postgres.

    Returns the number of rows handed to the database (conflicts are skipped
    server-side, so this is a lower bound of distinct data actually stored).
    """
    from sqlalchemy.dialects import postgresql, sqlite

    conflict_cols = _CONFLICT_COLUMNS[model.__name__]
    if db.get_bind().dialect.name == "sqlite":
        upsert = sqlite.insert(model).on_conflict_do_nothing(index_elements=conflict_cols)
    else:
        upsert = postgresql.insert(model).on_conflict_do_nothing(index_elements=conflict_cols)
    inserted = 0
    for start in range(0, len(payloads), batch_size):
        batch = payloads[start : start + batch_size]
        try:
            db.execute(upsert.values(batch))
            db.commit()
            inserted += len(batch)
        except Exception as e:  # noqa: BLE001 - keep hydrating despite a bad batch
            db.rollback()
            logger.warning(
                "skipped %d rows for %s (batch %d..%d): %s",
                len(batch),
                model.__name__,
                start,
                start + len(batch),
                str(e)[:160],
            )
    return inserted


if __name__ == "__main__":
    main()
