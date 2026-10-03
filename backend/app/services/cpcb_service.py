"""Official data.gov.in / CPCB pollution service (Phase 1).

Fetches and normalises the Government of India "Real time Air Quality Index from
various locations" dataset (Central Pollution Control Board) and upserts NCR
station observations into the backend database.

External-API logic (``fetch_ncr_records`` / ``normalize_records``) is kept
strictly separate from database logic (``upsert_ncr_data``). No fake data is
ever generated, no field names are assumed (they are read from the live
response), missing values stay ``None`` and the API key never appears in logs.

Dataset resource (verified live):
    https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69
Raw records are one row per (station x pollutant x timestamp):
    country, state, city, station, last_update, latitude, longitude,
    pollutant_id, min_value, max_value, avg_value
Numeric fields are strings ("7") and missing values are "NA".
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from ..config import get_settings

logger = logging.getLogger("aerocast.cpcb")

# Provenance tag stored on every row this service upserts (see
# ``pollution_observations.data_source``).
POLLUTION_SOURCE = "data_gov_in"

IST = ZoneInfo("Asia/Kolkata")
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"
HTTP_TIMEOUT = 30
PAGE_LIMIT = 100
# The live API throttles rapid page bursts by stalling connections (observed as
# read timeouts, not HTTP 429s), so requests are paced and retried with backoff.
PAGE_DELAY = 0.3
RETRY_DELAY = 3
RETRY_ATTEMPTS = 3

# Verified against the live API (2026-09): requests with urllib3's default
# User-Agent stall on a read timeout, while a custom User-Agent is served.
HEADERS = {"User-Agent": "AeroCast-NCR/1.0 (Delhi NCR air-pollution forecasting)"}

POLLUTANT_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "OZONE": "o3",
    # NH3 and Pb are CPCB criteria pollutants that the feed publishes. They
    # were previously dropped here, discarding real measurements. They are now
    # stored (ug/m3) so the data is not lost. They are deliberately NOT scored
    # into the AQI: this repository has no verified CPCB sub-index breakpoint
    # table for them, and aqi_calculator reports them as unavailable rather than
    # inventing bands. See IAQI_BREAKPOINTS / UNSCORED_POLLUTANTS.
    "NH3": "nh3",
    "PB": "pb",
    "LEAD": "pb",
}

# Value column names, newest-first; legacy data.gov.in responses used the
# "pollutant_*" prefix and older mirrors the suffix variants. The mapper tries
# them in order so both current and archived payloads normalise identically.
_VALUE_COLUMNS: tuple[tuple[str, ...], ...] = (
    ("avg_value", "pollutant_avg"),
    ("max_value", "pollutant_max"),
    ("min_value", "pollutant_min"),
)

# Monitor display names published by CPCB differ from the canonical station
# names in the database. Lower-cased source short name -> canonical DB name.
# Every one of the 17 curated stations is mapped so the official feed covering
# any NCR monitor lands on the canonical row (SIH26082 17/17 coverage).
_STATION_ALIASES = {
    "imd lodhi road": "Lodhi Road",
    "lodhi road": "Lodhi Road",
    "r k puram": "RK Puram",
    "rk puram": "RK Puram",
    "dwarka-sector 8": "Dwarka",
    "dwarka": "Dwarka",
    "punjabi bagh": "Punjabi Bagh",
    "anand vihar": "Anand Vihar",
    "ito": "ITO",
    "sirifort": "Sirifort",
    "shadipur": "Shadipur",
    "okhla phase-2": "Okhla Phase-2",
    "ashok vihar": "Ashok Vihar",
    "mundka": "Mundka",
    "jahangirpuri": "Jahangirpuri",
    "aya nagar": "Aya Nagar",
    "vivek vihar": "Vivek Vihar",
    # Teri Gram is the HSDP/DPCC monitor that covers the Teri Gram (Gurugram)
    # site; both the exact monitor name and "Vikas Sadan, Gurugram" map here.
    "teri gram": "Teri Gram",
    "vikas sadan": "Teri Gram",
    "sector - 62": "Noida Sector-62",
    "sector-62": "Noida Sector-62",
    "sector 62": "Noida Sector-62",
    "noida sector-62": "Noida Sector-62",
    "sector 11": "Faridabad",  # Sector 11 monitor represents the Faridabad city point
    "faridabad": "Faridabad",
}


def _canonical_station_name(short: str) -> str:
    """Map a CPCB monitor short name onto the canonical database station name."""
    return _STATION_ALIASES.get(short.strip().lower(), short)


class CpcbError(Exception):
    """Typed error raised by the CPCB service.

    ``kind`` is one of: ``missing_key | http | network | json | empty``.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class NormalizedObservation:
    station_name: str
    city: str
    state: str
    latitude: float | None
    longitude: float | None
    timestamp: datetime
    values: dict[str, float | None]


def _to_float(value: Any) -> float | None:
    """Convert numeric strings to float; missing/invalid -> None (never 0)."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _parse_timestamp(raw: str) -> datetime | None:
    if not raw or str(raw).strip().upper() == "NA":
        return None
    try:
        return datetime.strptime(str(raw).strip(), TIMESTAMP_FORMAT).replace(tzinfo=IST)
    except ValueError:
        return None


def _short_station_name(full_name: str, city: str = "") -> str:
    """Derive a stable short name from API text like 'Sector-62, Noida - UPPCB'.

    Drops the trailing '- <agency>' token, then a trailing ', <city>' locator so
    'Anand Vihar, Delhi - DPCC' maps to the existing 'Anand Vihar' station.
    """
    if " - " in full_name:
        short = full_name.rsplit(" - ", 1)[0].strip()
    else:
        short = full_name.strip()
    if city:
        suffix = f", {city.strip()}"
        if short.endswith(suffix):
            short = short[: -len(suffix)].strip()
    return short or full_name.strip()


def _extract_value(record: dict[str, Any]) -> float | None:
    for candidates in _VALUE_COLUMNS:
        for key in candidates:
            if key in record:
                parsed = _to_float(record.get(key))
                if parsed is not None:
                    return parsed
    return None


def _fetch_page(base: str, params: dict[str, Any], timeout: float) -> dict[str, Any] | str:
    """Fetch one paginated response; returns parsed payload or an error label.

    The live API intermittently stalls (read timeout) instead of returning a
    429, so transient HTTP/network/parse errors are retried with backoff before
    a page is declared failed. A permanent client error returns immediately.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(base, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            return payload if isinstance(payload, dict) else "malformed-payload"
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                return f"http={status}"
            if attempt + 1 == RETRY_ATTEMPTS:
                logger.warning("CPCB fetch HTTP error (status=%s)", status)
                return f"http={status}"
        except requests.exceptions.RequestException as exc:
            if attempt + 1 == RETRY_ATTEMPTS:
                logger.warning("CPCB fetch network error: %s", type(exc).__name__)
                return f"network={type(exc).__name__}"
        except ValueError as exc:
            if attempt + 1 == RETRY_ATTEMPTS:
                logger.warning("CPCB fetch invalid JSON for %s: %s", params.get("filters[city]"), exc)
                return "invalid-json"
        time.sleep(RETRY_DELAY * (attempt + 1))
    return "network=unknown"


def fetch_ncr_records(api_key: str | None = None, cities: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch current NCR pollution rows from the official data.gov.in API.

    Paginates per city (offset until a short page). Never logs the API key or
    the full request URL. Raises ``CpcbError`` on any failure.
    """
    settings = get_settings()
    key = api_key if api_key is not None else settings.data_gov_api_key
    if not key:
        raise CpcbError(
            "missing_key",
            "DATA_GOV_API_KEY is not configured. Register for a free key at "
            "https://data.gov.in (Dashboard -> MyAccount) and set it in .env.",
        )
    city_list = cities if cities is not None else _ncr_cities(settings.data_gov_ncr_cities)
    if not city_list:
        raise CpcbError("empty", "No NCR cities configured for ingestion.")

    base = f"{settings.data_gov_api_url.rstrip('/')}/resource/{settings.data_gov_resource_id}"
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for city in city_list:
        offset = 0
        prev_len = None
        while True:
            params = {
                "api-key": key,
                "format": "json",
                "limit": PAGE_LIMIT,
                "offset": offset,
                "filters[city]": city,
            }
            payload = _fetch_page(base, params, settings.data_gov_timeout)
            if isinstance(payload, str):
                errors.append(f"city={city} {payload}")
                break

            status = payload.get("status")
            if status and str(status).lower() != "ok":
                errors.append(f"city={city} status={status}")
                break
            page: Any = payload.get("records")
            if not isinstance(page, list) or not page:
                # The live API intermittently returns an empty page (status=ok)
                # while more records remain — its throttling signature. Retry the
                # same offset before accepting a truncated result.
                total = payload.get("total")
                more_expected = isinstance(total, int) and not isinstance(total, bool) and total > offset
                if more_expected:
                    for attempt in range(RETRY_ATTEMPTS):
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        retry_payload = _fetch_page(base, params, settings.data_gov_timeout)
                        if isinstance(retry_payload, str):
                            continue
                        retry_page = retry_payload.get("records")
                        if isinstance(retry_page, list) and retry_page:
                            payload = retry_payload
                            page = retry_page
                            break
                if not isinstance(page, list) or not page:
                    errors.append(f"city={city} empty-page-offset={offset}")
                    logger.warning(
                        "CPCB empty page for %s at offset=%s (total=%s); %d record(s) known",
                        city,
                        offset,
                        payload.get("total"),
                        len(records),
                    )
                break
            records.extend(page)
            fetched = len(page)
            offset += fetched
            # The live API caps its page size well below PAGE_LIMIT (~10 rows for
            # limit=100 in 2026-09) and reports a per-city ``total``, so pagination
            # ends when the per-city offset reaches ``total`` or a page comes back
            # shorter than its predecessor (defensive fallback when total is absent).
            total = payload.get("total")
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0 and offset >= total:
                break
            if prev_len is not None and fetched < prev_len:
                break
            prev_len = fetched
            time.sleep(PAGE_DELAY)

    if not records:
        raise CpcbError("empty", f"No CPCB records returned ({' | '.join(errors) or 'no errors'}).")
    if errors:
        logger.warning("CPCB partial fetch; %d record(s), issues: %s", len(records), errors)
    return records


def _ncr_cities(configured: str) -> list[str]:
    return [c.strip() for c in configured.split(",") if c.strip()]


def normalize_records(records: list[dict[str, Any]]) -> list[NormalizedObservation]:
    """Transform row-per-pollutant records into one observation per station/timestamp."""
    merged: dict[tuple[str, datetime], dict[str, Any]] = {}
    station_meta: dict[str, dict[str, Any]] = {}

    for record in records:
        full_station = str(record.get("station") or "").strip()
        if not full_station:
            continue
        city = str(record.get("city") or "").strip()
        short = _short_station_name(full_station, city)
        ts = _parse_timestamp(str(record.get("last_update") or ""))
        if ts is None:
            continue
        pollutant = str(record.get("pollutant_id") or "").strip().upper()
        target = POLLUTANT_MAP.get(pollutant)
        if target is None:
            # Genuinely non-criteria pollutants (Benzene, Toluene, ...) have no
            # column in pollution_observations and are skipped. NH3/Pb are no
            # longer dropped here - they are mapped above.
            continue

        meta = station_meta.setdefault(short, {"city": "", "state": "", "lat": None, "lon": None})
        state = str(record.get("state") or "").strip()
        if city:
            meta["city"] = city
        if state:
            meta["state"] = state
        lat = _to_float(record.get("latitude"))
        lon = _to_float(record.get("longitude"))
        if lat is not None:
            meta["lat"] = lat
        if lon is not None:
            meta["lon"] = lon

        key = (short, ts)
        entry = merged.setdefault(key, {"values": {}})
        value = _extract_value(record)
        if value is not None:
            entry["values"][target] = value

    observations: list[NormalizedObservation] = []
    for (short, ts), entry in merged.items():
        meta = station_meta.get(short, {})
        obs = NormalizedObservation(
            station_name=short,
            city=meta.get("city") or short,
            state=meta.get("state") or "",
            latitude=meta.get("lat"),
            longitude=meta.get("lon"),
            timestamp=ts,
            values=entry["values"],
        )
        observations.append(obs)
    return observations


def upsert_ncr_data(db, observations: list[NormalizedObservation]) -> dict[str, int]:
    """Upsert NCR stations + pollution observations. Returns usage counters.

    Station identity is the canonical short name, aliased onto the curated
    ``Station`` rows (see ``_STATION_ALIASES``). Observations for monitors that
    are neither an existing station nor an alias are skipped — the station set
    is curated, ingestion never creates ad-hoc stations. Observation identity is
    ``(station_id, timestamp)`` — protected by the unique constraint
    ``uq_pollution_station_ts``; duplicates are never created.
    """
    from ..models.db_models import PollutionReading, Station
    from ..services.aqi_calculator import calculate_aqi

    stations = {s.name: s for s in db.query(Station).all()}
    counters = {
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "station_created": 0,
        "station_updated": 0,
        "station_skipped": 0,
    }

    matched: list[tuple[Station, NormalizedObservation]] = []
    for obs in observations:
        name = _canonical_station_name(obs.station_name)
        station = stations.get(name)
        if station is None:
            counters["station_skipped"] += 1
            continue
        changed = False
        if obs.latitude is not None and station.latitude != obs.latitude:
            station.latitude = obs.latitude
            changed = True
        if obs.longitude is not None and station.longitude != obs.longitude:
            station.longitude = obs.longitude
            changed = True
        if obs.city and station.city != obs.city:
            station.city = obs.city
            changed = True
        if obs.state and station.state != obs.state:
            station.state = obs.state
            changed = True
        if changed:
            counters["station_updated"] += 1
        matched.append((station, obs))

    if observations:
        db.flush()

    # All eight CPCB criteria pollutants the schema stores. nh3/pb are nullable
    # and may be absent from a given feed response - that is fine, they stay
    # NULL and are excluded from the AQI rather than defaulted to zero.
    _STORED = ("pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb")
    source = POLLUTION_SOURCE
    for station, obs in matched:
        # CPCB reports ``last_update`` in IST, so ``_parse_timestamp`` returns an
        # IST-aware value. Convert it to UTC *before* dropping the tzinfo: every
        # naive timestamp this application persists is UTC (see
        # ``app.api.summary``, ``app.services.forecast_service`` and
        # ``app.services.refresh_service``, all of which read a naive stored value
        # as UTC wall-clock). Stripping the tzinfo instead would store IST
        # wall-clock under a UTC contract and shift every reading 5 h 30 m into
        # the future.
        ts = obs.timestamp.astimezone(UTC).replace(tzinfo=None)
        existing = (
            db.query(PollutionReading)
            .filter(PollutionReading.station_id == station.id, PollutionReading.timestamp == ts)
            .first()
        )
        values = {k: obs.values.get(k) for k in _STORED}
        aqi, _, _ = calculate_aqi(**values)

        if existing:
            if all(getattr(existing, k) == v for k, v in values.items()) and existing.aqi == aqi:
                counters["skipped"] += 1
                continue
            for k, v in values.items():
                setattr(existing, k, v)
            existing.aqi = aqi
            if not existing.data_source:
                existing.data_source = source
            counters["updated"] += 1
        else:
            db.add(
                PollutionReading(
                    station_id=station.id,
                    timestamp=ts,
                    **values,
                    aqi=aqi,
                    data_source=source,
                )
            )
            counters["inserted"] += 1

    db.commit()
    return counters


def run_ingestion(db, api_key: str | None = None) -> dict[str, Any]:
    """Full pipeline: fetch official data -> normalize -> persist. Returns a summary."""
    from ..models.db_models import Station

    records = fetch_ncr_records(api_key=api_key)
    observations = normalize_records(records)
    counters = upsert_ncr_data(db, observations)
    existing_names = {s.name for s in db.query(Station).all()}
    return {
        "records_fetched": len(records),
        "observations_normalized": len(observations),
        "stations_processed": sum(1 for o in observations if _canonical_station_name(o.station_name) in existing_names),
        "inserted": counters["inserted"],
        "updated": counters["updated"],
        "skipped": counters["skipped"],
        "station_created": counters["station_created"],
        "station_updated": counters["station_updated"],
        "station_skipped": counters["station_skipped"],
        "errors": [],
    }
