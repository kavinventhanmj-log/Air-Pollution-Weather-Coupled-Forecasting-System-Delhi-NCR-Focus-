"""NASA FIRMS active-fire integration for AeroCast-NCR.

Fetches *real* NASA FIRMS fire hotspots covering Delhi NCR and the upwind
stubble-burning tract (Punjab / Haryana / northern Rajasthan), normalises the
raw FIRMS CSV columns, validates each record, and stores the observations.

Two fetch paths (both official NASA FIRMS data):

* Primary — the FIRMS area-bounding-box API (``/api/area/csv/{map_key}/...``)
  which requires a free ``NASA_FIRMS_MAP_KEY``; it returns all active satellites
  for up to the last 10 days.
* Fallback — the public FIRMS 24h global CSV contributions (VIIRS C2 375m and
  MODIS C6), which need no key and still return real, current observations.

No synthetic data is ever synthesised here: every stored row is a FIRMS
detection, stamped ``synthetic=False`` / ``source="firms_live"`` so
operational readers can always tell real observations from simulated
history. This module only describes *what was observed*; it makes no
assertion that any given fire caused or contributed to Delhi pollution.
"""

import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd
import requests

logger = logging.getLogger("aerocast.firms")

HTTP_TIMEOUT = 60

# Delhi NCR plus the upwind winter stubble tract (Punjab, Haryana, parts of
# Rajasthan and western UP) that typically sits upwind of Delhi in the late
# monsoon / stubble season. Coordinates are (min_lon, min_lat, max_lon, max_lat).
FIRMS_REGION = {"min_lon": 73.5, "min_lat": 27.5, "max_lon": 78.5, "max_lat": 33.0}

# Official FIRMS area API. `area` is west,south,east,north (decimal degrees);
# the current API expects {map_key}/{source}/{area}/{day_range}.
FIRMS_API_AREA_CSV = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{area}/{day}"

# All active satellites exposed by the FIRMS area API (valid source names as of
# the 2026 API docs — NRT flavours + MODIS_SP; bare-named/VIIRS_NRT and
# MODIS_Aqua are rejected with HTTP 400).
FIRMS_API_SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_SP",
    "MODIS_NRT",
]

# Public FIRMS daily-24h CSV contributions (no API key required).
PUBLIC_CSV_URLS = [
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv",
]

DEFAULT_API_DAYS = 5
MAX_API_DAYS = 5

_CONFIDENCE_VALUES = {"low", "nominal", "high"}


def resolve_api_key() -> str:
    """Return the NASA FIRMS MAP_KEY from env vars or config (.env).

    Env takes precedence: ``NASA_FIRMS_MAP_KEY`` then ``FIRMS_MAP_KEY``,
    falling back to the pydantic setting. Returns "" (never raises).
    """
    key = os.getenv("NASA_FIRMS_MAP_KEY") or os.getenv("FIRMS_MAP_KEY") or ""
    if key:
        return key.strip()
    try:
        from ..config import get_settings

        return (get_settings().nasa_firms_map_key or "").strip()
    except Exception:
        return ""


def region_area() -> str:
    """The ``{area}`` string for the API: west,south,east,north."""
    r = FIRMS_REGION
    return f"{r['min_lon']},{r['min_lat']},{r['max_lon']},{r['max_lat']}"


def _parse_acq_time(row) -> pd.Timestamp | None:
    """Parse FIRMS ``acq_date`` + ``acq_time`` (HHMM UTC) into naive-UTC.

    FIRMS conventions: acq_time 2400 == next-day 00:00; 9999/empty == time
    unknown (use date at 00:00). Returns naive-UTC (matching the app's
    weather convention) or None when undecodable.
    """
    date = str(row.get("acq_date") or "").strip()
    if not date:
        return None
    t_expr = str(row.get("acq_time") or "").strip().zfill(4)
    try:
        if t_expr == "2400":
            ts = pd.Timestamp(date) + pd.Timedelta(days=1)
        elif t_expr in ("9999", "0000"):
            ts = pd.Timestamp(date)
        else:
            ts = pd.to_datetime(f"{date} {t_expr}", format="%Y-%m-%d %H%M", errors="coerce")
    except (ValueError, TypeError):
        ts = None
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _normalise_brightness(row) -> float | None:
    """Brightness temperature in Kelvin (VIIRS ``bright_ti4`` / MODIS ``bright_t31``)."""
    val = row.get("bright_ti4")
    if val is None or (isinstance(val, float) and val != val):
        val = row.get("bright_t31")
    return _to_float(val)


def _to_float(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _valid_coordinate(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def in_region(lat: float, lon: float) -> bool:
    r = FIRMS_REGION
    return r["min_lat"] <= lat <= r["max_lat"] and r["min_lon"] <= lon <= r["max_lon"]


def normalise_fire_records(df: pd.DataFrame) -> list[dict]:
    """Validate + normalise raw FIRMS rows into insert-ready dicts.

    Returns records with keys: satellite, instrument, latitude, longitude,
    acq_date (naive-UTC datetime), confidence, frp, brightness, daynight,
    synthetic (False), source ("firms_live").
    Invalid rows (bad coordinates / undecodable time / outside region) are
    dropped and counted in the log, not silently stored.
    """
    if df is None or df.empty:
        return []
    records: list[dict] = []
    dropped = 0
    for _, row in df.iterrows():
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if not _valid_coordinate(lat, lon):
            dropped += 1
            continue
        assert lat is not None and lon is not None
        if not in_region(lat, lon):
            dropped += 1
            continue
        ts = _parse_acq_time(row)
        if ts is None:
            dropped += 1
            continue
        confidence = str(row.get("confidence") or "").strip().lower()
        if confidence not in _CONFIDENCE_VALUES:
            confidence = ""
        records.append(
            {
                "satellite": str(row.get("satellite") or "").strip(),
                "instrument": _instrument_for(row),
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "acq_date": ts.to_pydatetime(),
                "confidence": confidence or None,
                "frp": _to_float(row.get("frp")),
                "brightness": _normalise_brightness(row),
                "daynight": str(row.get("daynight") or "").strip(),
                "synthetic": False,
                "source": "firms_live",
            }
        )
    if dropped:
        logger.info("firms: dropped %d of %d raw rows (out of region/invalid)", dropped, len(df))
    return records


def _instrument_for(row) -> str:
    """FIRMS `instrument` is 'VIIRS'/'MODIS'; infer from the source when absent."""
    raw = str(row.get("instrument") or "").strip()
    if raw:
        return raw
    satellite = str(row.get("satellite") or "").strip()
    if satellite in {"Terra", "Aqua", "MODIS"}:
        return "MODIS"
    if satellite:
        return "VIIRS"
    return ""


def _fetch_area_api(api_key: str, days: int) -> Iterable[pd.DataFrame]:
    """Request per-satellite area CSVs from the official FIRMS API."""
    day = min(max(int(days), 1), MAX_API_DAYS)
    area = region_area()
    for source in FIRMS_API_SOURCES:
        url = FIRMS_API_AREA_CSV.format(map_key=api_key, source=source, day=day, area=area)
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(resp.text))
            if df is not None and not df.empty:
                logger.info("firms api %s: %d raw rows", source, len(df))
                yield df
        except Exception as exc:
            logger.warning("firms api %s failed (%s) — will fall back if needed", source, exc)


def _fetch_public_csv() -> Iterable[pd.DataFrame]:
    """Download the public FIRMS 24h CSV contributions (real data, no key)."""
    for url, instrument in zip(PUBLIC_CSV_URLS, ("VIIRS", "MODIS"), strict=False):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(resp.text))
            if df is not None and not df.empty:
                # The public CSVs do not carry an instrument name keyed to
                # their own satellite shorthand (N/A/T), so tag it per-source.
                if "instrument" not in df.columns:
                    df["instrument"] = instrument
                logger.info("firms public CSV: %d raw rows from %s", len(df), url.split("/")[-1])
                yield df
        except Exception as exc:
            logger.warning("firms public CSV %s failed: %s", url, exc)


def fetch_fire_records(api_key: str = "", days: int = DEFAULT_API_DAYS) -> tuple[list[dict], str]:
    """Fetch real FIRMS records for the NCR + upwind region.

    Returns ``(records, source_label)``. ``records`` is the deduplicated list
    of validated observation dicts (see :func:`normalise_fire_records`).
    """
    frames: list[pd.DataFrame] = []
    source_label = ""
    if api_key:
        source_label = "FIRMS area API (NASA FIRMS)"
        frames.extend(_fetch_area_api(api_key, days))
        if not frames:
            logger.warning("firms API returned no data; falling back to public 24h CSV")
    if not frames:
        source_label = "FIRMS public 24h CSV (NASA FIRMS)"
        frames.extend(_fetch_public_csv())
    if not frames:
        return [], source_label

    raw = pd.concat(frames, ignore_index=True)
    records = normalise_fire_records(raw)

    seen: set = set()
    unique: list[dict] = []
    for rec in records:
        key = (rec["satellite"], rec["latitude"], rec["longitude"], rec["acq_date"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique, source_label


def to_naive_utc(dt: datetime) -> datetime:
    """Normalise a stored datetime to naive-UTC for key comparison."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def upsert_fire_records(db, records: list[dict], dry_run: bool = False) -> dict:
    """Insert fire observations that are not already stored.

    Duplicate prevention is enforced *before* insert against both the current
    batch and the already-persisted rows (same satellite + location + acq time),
    mirroring the ``uq_fire_lat_lon_time`` unique constraint. Returns
    {"retrieved": n, "inserted": n, "duplicates_skipped": n}.
    """
    from ..models.db_models import FireReading

    if not records:
        return {"retrieved": 0, "inserted": 0, "duplicates_skipped": 0}

    # Load only the observation window we are about to touch.
    tmin = min(r["acq_date"] for r in records)
    tmax = max(r["acq_date"] for r in records)
    existing = {
        (sat, round(lat, 4), round(lon, 4), to_naive_utc(ts))
        for sat, lat, lon, ts in db.query(
            FireReading.satellite,
            FireReading.latitude,
            FireReading.longitude,
            FireReading.acq_date,
        )
        .filter(FireReading.acq_date.between(tmin, tmax))
        .all()
    }

    rows: list[FireReading] = []
    skipped = 0
    for rec in records:
        key = (rec["satellite"], round(rec["latitude"], 4), round(rec["longitude"], 4), rec["acq_date"])
        if key in existing:
            skipped += 1
            continue
        existing.add(key)
        rows.append(
            FireReading(
                satellite=rec["satellite"] or None,
                instrument=rec["instrument"] or None,
                latitude=rec["latitude"],
                longitude=rec["longitude"],
                acq_date=rec["acq_date"],
                confidence=rec["confidence"] or None,
                frp=rec["frp"],
                brightness=rec["brightness"],
                daynight=rec["daynight"] or None,
                synthetic=bool(rec.get("synthetic", False)),
                source=rec.get("source") or "firms_live",
            )
        )

    if rows:
        db.add_all(rows)
        if not dry_run:
            db.commit()

    result = {
        "retrieved": len(records),
        "inserted": len(rows),
        "duplicates_skipped": skipped,
    }
    logger.info("firms upsert: %s", result)
    return result
