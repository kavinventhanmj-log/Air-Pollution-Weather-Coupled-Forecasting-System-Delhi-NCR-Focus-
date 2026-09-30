"""Live data refresh service for AeroCast-NCR.

Fetches the latest weather (Open-Meteo), active fires (NASA FIRMS),
and CPCB pollution (data.gov.in / opencity CKAN) readings and upserts them
into the backend database so forecasts always run on the most recent
observations.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .firms_service import (
    FIRMS_REGION,
    PUBLIC_CSV_URLS,
    fetch_fire_records,
    resolve_api_key,
    upsert_fire_records,
)
from .ttl_cache import _running_under_pytest

logger = logging.getLogger("aerocast.refresh")

STATIONS = {
    "Anand_Vihar": (28.6492, 77.2918),
    "RK_Puram": (28.5601, 77.1835),
    "ITO": (28.6290, 77.2410),
    "Dwarka": (28.5921, 77.0460),
    "Punjabi_Bagh": (28.6692, 77.1285),
    "Lodhi_Road": (28.5866, 77.2268),
    "Sirifort": (28.5528, 77.2190),
    "Shadipur": (28.6542, 77.1489),
    "Okhla_Phase-2": (28.5230, 77.2680),
    "Ashok_Vihar": (28.6974, 77.1756),
    "Mundka": (28.6796, 77.0189),
    "Jahangirpuri": (28.7256, 77.1556),
    "Aya_Nagar": (28.4771, 77.1148),
    "Vivek_Vihar": (28.6727, 77.3169),
    "Teri_Gram": (28.4422, 77.0115),
    "Noida_Sector-62": (28.6227, 77.3615),
    "Faridabad": (28.4089, 77.3178),
}

# Legacy aliases kept for callers/tests that referenced the old constants.
REGION = dict(FIRMS_REGION)
FIRMS_CSV = PUBLIC_CSV_URLS[0]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CKAN_BASE = "https://data.opencity.in/api/3/action/datastore_search"
# Per-station datastore resources from the opencity.in package
# "Delhi Hourly Air Quality Reports", 15-minute series for 2024-25. Keyed by the
# station name as it appears in our own `stations` table, because that is what the
# rows are written against; the resource's own "Station Name" column is not used
# for matching.
#
# This previously carried only the five regional-composite stations, so every
# other station in the table had *zero* pollution rows and the PM2.5 forecast,
# its SHAP explanation and the observed-vs-forecast verification panel all
# refused with "No pollution readings for station". Faridabad, Noida Sector-62
# and Teri Gram have no resource in that package (they sit outside Delhi) and
# are reported as insufficient by /api/pollution/coverage rather than filled in.
CKAN_RESOURCES = {
    "Anand Vihar": "5ef3f66f-2bb0-4593-91db-ba6e693a77f3",
    "RK Puram": "d9dfd28d-038d-448f-8e33-5e6f6b32d15c",
    "ITO": "890f786d-fb9f-475e-8516-191bfa1b01ea",
    "Dwarka": "495db3d4-5683-4b1d-9b7d-34ecb887ca13",
    "Punjabi Bagh": "82080ddc-e094-4a3a-8421-242ec6bc8a45",
    "Ashok Vihar": "73703a83-f321-4b60-8748-33fb046273e0",
    "Aya Nagar": "b64a05dc-a327-4f10-9f77-685b1fd59f0f",
    "Jahangirpuri": "734c459f-3e47-44c7-b0b8-8270090cecf0",
    "Lodhi Road": "99a25262-def3-4898-9f63-a65c0bbeaecd",
    "Mundka": "3fe46cce-659f-49a7-a743-090c2557cbfd",
    "Okhla Phase-2": "7b03c3c3-c95e-4d09-b1a8-8951b96c05b6",
    "Shadipur": "e91c68ea-85ac-49f8-8015-dd8b04c32a71",
    "Sirifort": "e562d0c9-cc37-4d5a-a04e-fd95ee9f2f04",
    "Vivek Vihar": "19172e66-45d4-4d3c-8736-e573b160fe99",
}

HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,pressure_msl,surface_pressure,"
    "wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,"
    "boundary_layer_height"
)

# Vertical pressure-level variables (SIH26082 inversion analysis). Temperatures
# in degC at standard levels; geopotential height at 925/850 hPa for
# inversion-base reporting.
PRESSURE_LEVEL_VARS = (
    "temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,"
    "geopotential_height_925hPa,geopotential_height_850hPa"
)

DEFAULT_LOOKBACK_DAYS = 3
HTTP_TIMEOUT = 60

# A 5-day forecast starts at 00:00 of the current day and always extends past
# ``now + 72h``, the horizon the atmospheric-context builder needs. No
# ``past_days`` is requested: the frame is deliberately a *future* forecast, so
# it can never be mistaken for a stored observation.
FORECAST_COVERAGE_DAYS = 5


def _to_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _existing_timestamps(db, model_cls, station_id) -> set:
    """Return stored timestamps for a station as *naive UTC* datetimes.

    PostgreSQL ``DateTime(timezone=True)`` columns return timezone-aware
    datetimes (``+00:00``) while the refresh pipeline normalises incoming
    source timestamps to naive UTC (no tzinfo). Comparing aware vs naive
    datetimes in Python is always ``False``, which previously defeated the
    pre-insert dedup and caused duplicate weather rows (and UniqueViolation
    aborts for pollution). Stripping the timezone makes the comparison
    correct under the project's "naive-UTC" storage convention.
    """
    return {
        ts.replace(tzinfo=None) if ts.tzinfo else ts
        for (ts,) in db.query(model_cls.timestamp).filter(model_cls.station_id == station_id).all()
    }


def _get_weather_df(station_name: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": f"{HOURLY_VARS},{PRESSURE_LEVEL_VARS}",
        "timezone": "UTC",
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        # Age out gracefully: fall back to forecast API for the next ~48h
        logger.warning("archive fetch failed (%s) — trying forecast", exc)
        f_params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": 2,
            "hourly": f"{HOURLY_VARS},{PRESSURE_LEVEL_VARS}",
            "timezone": "UTC",
        }
        resp = requests.get(FORECAST_URL, params=f_params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    # Normalize timestamps to naive UTC: Open-Meteo returns ISO datetimes in the
    # requested timezone (we request UTC); a naive datetime has no offset, so all
    # wall-clock ambiguity is removed and stored values are UTC by convention.
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)

    # Archive pressure-level data for recent dates can be all-NULL (variable
    # latency). When that happens, supplement with the live forecast window so
    # lapse-rate analysis still has real vertical data (SIH26082).
    pl_cols = [f"temperature_{int(p)}hPa" for p in (1000, 925, 850, 700)]
    pl_present = [c for c in pl_cols if c in df.columns]
    if pl_present and not df[pl_present].notna().any().any():
        try:
            f_params = {
                "latitude": lat,
                "longitude": lon,
                "forecast_days": 3,
                "hourly": PRESSURE_LEVEL_VARS,
                "timezone": "UTC",
            }
            fresp = requests.get(FORECAST_URL, params=f_params, timeout=HTTP_TIMEOUT)
            fresp.raise_for_status()
            fdata = fresp.json().get("hourly", {})
            if fdata and "time" in fdata:
                fdf = pd.DataFrame(fdata)
                fdf["time"] = pd.to_datetime(fdf["time"], utc=True).dt.tz_localize(None)
                for col in pl_present:
                    if col not in fdf.columns:
                        fdf[col] = None
                fdf = fdf[["time"] + pl_present]
                df = df.set_index("time")
                fdf = fdf.set_index("time")
                df.update(fdf)  # fill vertical temps where archive is NULL
                df = df.reset_index()
        except Exception as ferr:
            logger.warning("supplemental forecast pressure-level fetch failed: %s", ferr)

    raw_to_display = {
        "Anand_Vihar": "Anand Vihar",
        "RK_Puram": "RK Puram",
        "Punjabi_Bagh": "Punjabi Bagh",
    }
    df["station"] = raw_to_display.get(station_name, station_name.replace("_", " "))
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def fetch_forecast_hours(lat: float, lon: float) -> pd.DataFrame:
    """Fetch a multi-day Open-Meteo *forecast* window for a station.

    Returns hourly rows (naive-UTC ``time``) in the same shape ``_get_weather_df``
    produces (surface variables + pressure-level temperatures), so callers can
    feed the rows through the same normalisation / nearest-match helpers. With
    ``forecast_days=FORECAST_COVERAGE_DAYS`` the frame always covers
    ``now + 72h``, which is the look-ahead the atmospheric-context builder needs.

    Returns an empty frame when the provider returns no hourly data, and raises
    on transport / HTTP errors so the caller chooses its own fallback. Nothing
    here writes to the database.
    """
    # Keep the suite hermetic: under pytest the forecast is supplied by a mock
    # (see tests/unit/test_coupling_context_forecast.py).
    if _running_under_pytest():
        return pd.DataFrame()
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": FORECAST_COVERAGE_DAYS,
        "hourly": f"{HOURLY_VARS},{PRESSURE_LEVEL_VARS}",
        "timezone": "UTC",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    return df


def refresh_weather(db, dry_run: bool = False) -> int:
    """Upsert recent Open-Meteo weather readings; returns inserted row count."""
    from ..models.db_models import Station, WeatherReading

    stations = {s.name: s for s in db.query(Station).all()}
    end = datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d")
    start = (datetime.now(UTC).replace(tzinfo=None) - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    inserted = 0
    for raw_name, (lat, lon) in STATIONS.items():
        display = raw_name.replace("_", " ")
        if display not in stations:
            continue
        df = _get_weather_df(raw_name, lat, lon, start, end)
        if df.empty:
            continue
        existing = _existing_timestamps(db, WeatherReading, stations[display].id)
        rows = []
        for _, r in df.iterrows():
            ts = r["time"]
            if ts in existing:
                continue
            rows.append(
                WeatherReading(
                    station_id=stations[display].id,
                    timestamp=ts,
                    temperature=_to_float(r.get("temperature_2m")),
                    humidity=_to_float(r.get("relative_humidity_2m")),
                    pressure_msl=_to_float(r.get("pressure_msl")),
                    surface_pressure=_to_float(r.get("surface_pressure")),
                    wind_speed=_to_float(r.get("wind_speed_10m")),
                    wind_direction=_to_float(r.get("wind_direction_10m")),
                    precipitation=_to_float(r.get("precipitation")),
                    cloud_cover=_to_float(r.get("cloud_cover")),
                    pbl_height=_to_float(r.get("boundary_layer_height")),
                    temperature_1000hPa=_to_float(r.get("temperature_1000hPa")),
                    temperature_925hPa=_to_float(r.get("temperature_925hPa")),
                    temperature_850hPa=_to_float(r.get("temperature_850hPa")),
                    temperature_700hPa=_to_float(r.get("temperature_700hPa")),
                    geopotential_height_925hPa=_to_float(r.get("geopotential_height_925hPa")),
                    geopotential_height_850hPa=_to_float(r.get("geopotential_height_850hPa")),
                )
            )
            existing.add(ts)
        if rows:
            db.add_all(rows)
            if not dry_run:
                db.commit()
            inserted += len(rows)

        # Backfill vertical profile onto existing rows in this window that are
        # missing pressure-level temperatures (e.g. rows created before the
        # supplemental forecast fetch existed, while their timestamp already
        # existed so the insert path above skipped them).
        try:
            stale = (
                db.query(WeatherReading)
                .filter(
                    WeatherReading.station_id == stations[display].id,
                    WeatherReading.temperature_925hPa.is_(None),
                    WeatherReading.timestamp >= pd.Timestamp(start).to_pydatetime(),
                    WeatherReading.timestamp < pd.Timestamp(end).to_pydatetime() + pd.Timedelta(days=1),
                )
                .all()
            )
            by_ts = {r["time"]: r for _, r in df.iterrows()}
            updated = 0
            for rec in stale:
                src = by_ts.get(pd.Timestamp(rec.timestamp))
                if src is None:
                    continue
                rec.temperature_1000hPa = _to_float(src.get("temperature_1000hPa"))
                rec.temperature_925hPa = _to_float(src.get("temperature_925hPa"))
                rec.temperature_850hPa = _to_float(src.get("temperature_850hPa"))
                rec.temperature_700hPa = _to_float(src.get("temperature_700hPa"))
                rec.geopotential_height_925hPa = _to_float(src.get("geopotential_height_925hPa"))
                rec.geopotential_height_850hPa = _to_float(src.get("geopotential_height_850hPa"))
                updated += 1
            if updated and not dry_run:
                db.commit()
        except Exception as exc:
            logger.warning("vertical backfill skipped: %s", exc)
    return inserted


def refresh_fire(db, dry_run: bool = False) -> int:
    """Fetch + upsert live NASA FIRMS hotspots; returns rows inserted.

    Uses the official FIRMS area API when ``NASA_FIRMS_MAP_KEY`` is set,
    otherwise falls back to the public FIRMS 24h CSVs (both are real NASA
    data). Observations are validated, normalised to naive-UTC, filtered to
    the NCR + upwind region and deduplicated against the DB before insert.
    """
    records, source = fetch_fire_records(api_key=resolve_api_key())
    if not records:
        logger.info("fire refresh: no records available (source=%s)", source)
        return 0
    result = upsert_fire_records(db, records, dry_run=dry_run)
    logger.info(
        "fire refresh source=%s retrieved=%d inserted=%d duplicates=%d",
        source,
        result["retrieved"],
        result["inserted"],
        result["duplicates_skipped"],
    )
    return result["inserted"]


def refresh_pollution(db, dry_run: bool = False) -> int:
    """Upsert recent CPCB pollution readings from opencity.in CKAN; returns row count."""
    from ..models.db_models import PollutionReading, Station
    from ..services.aqi_calculator import calculate_aqi

    stations = {s.name: s for s in db.query(Station).all()}
    pmap = {
        "PM2.5 (ug/m3)": "pm25",
        "PM10 (ug/m3)": "pm10",
        "NO2 (ug/m3)": "no2",
        "SO2 (ug/m3)": "so2",
        "CO (mg/m3)": "co",
        "Ozone (ug/m3)": "o3",
        # NH3 is published by this feed (present in data/pollution/*.csv as
        # "NH3 (ug/m3)"). Stored so the measurement is not lost. Not scored into
        # the AQI - no verified CPCB sub-index table for NH3 in this repository.
        # This feed carries no Pb column, so r.get("Pb (ug/m3)") would be absent;
        # .get() keeps that a safe None either way.
        "NH3 (ug/m3)": "nh3",
    }
    inserted = 0
    for display, resource_id in CKAN_RESOURCES.items():
        if display not in stations:
            continue
        try:
            ckan_params: dict[str, Any] = {
                "resource_id": resource_id,
                "limit": 500,
                "sort": "Timestamp desc",
            }
            resp = requests.get(
                CKAN_BASE,
                params=ckan_params,
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            records = resp.json().get("result", {}).get("records", [])
        except Exception as exc:
            logger.warning("CKAN fetch failed for %s: %s", display, exc)
            continue
        if not records:
            continue
        df = pd.DataFrame(records)
        if {"datetime", "site"} not in ({c for c in df.columns}, set()):
            pass
        ts_col = next((c for c in ["datetime", "time", "From Date", "timestamp", "Timestamp"] if c in df.columns), None)
        if not ts_col:
            continue
        existing = _existing_timestamps(db, PollutionReading, stations[display].id)
        rows = []
        for _, r in df.iterrows():
            ts = pd.to_datetime(r[ts_col], utc=False, errors="coerce")
            if ts is None or pd.isna(ts):
                continue
            ts = ts.tz_localize(None) if ts.tzinfo else ts
            vals = {}
            for src, dst in pmap.items():
                vals[dst] = _to_float(r.get(src))
            # Skip placeholder/junk rows that carry no pollutant measurements
            # (the CKAN feed occasionally returns rows with a valid timestamp
            # but every sensor value NULL).
            if all(v is None for v in vals.values()):
                continue
            if ts in existing:
                continue
            aqi_val, _, _ = calculate_aqi(**vals)
            rows.append(
                PollutionReading(
                    station_id=stations[display].id,
                    timestamp=ts,
                    pm25=vals.get("pm25"),
                    pm10=vals.get("pm10"),
                    o3=vals.get("o3"),
                    no2=vals.get("no2"),
                    so2=vals.get("so2"),
                    co=vals.get("co"),
                    nh3=vals.get("nh3"),
                    aqi=aqi_val,
                    data_source="opencity_ckan",
                )
            )
            existing.add(ts)
        if rows:
            db.add_all(rows)
            if not dry_run:
                db.commit()
            inserted += len(rows)
    return inserted


def run_refresh_once(db=None, dry_run: bool = False) -> dict:
    """Run one full refresh pass. Returns a summary dict.

    When `dry_run` is True the summaries reflect what *would* be inserted but
    nothing is committed to the database (the session is rolled back).
    """
    from ..database import SessionLocal

    close = db is None
    session = db if db is not None else SessionLocal()
    summary = {"weather": 0, "fire": 0, "pollution": 0}
    try:
        summary["weather"] = refresh_weather(session, dry_run=dry_run)
        summary["fire"] = refresh_fire(session, dry_run=dry_run)
        summary["pollution"] = refresh_pollution(session, dry_run=dry_run)
        if not dry_run:
            from .ttl_cache import invalidate_all

            invalidate_all()
        if dry_run:
            session.rollback()
    except Exception as exc:
        logger.warning("live refresh failed partway: %s", exc, exc_info=True)
    finally:
        if close:
            session.close()
    logger.info("live refresh summary: %s", summary)
    return summary


async def refresh_loop(interval_hours: float = 3.0, stop: asyncio.Event | None = None):
    """Background loop that periodically refreshes live data."""
    while True:
        try:
            await asyncio.to_thread(run_refresh_once)
        except Exception as exc:
            logger.warning("refresh loop iteration failed: %s", exc)
        if stop is not None and stop.is_set():
            break
        await asyncio.sleep(interval_hours * 3600)
