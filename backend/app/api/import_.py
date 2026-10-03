"""CSV import endpoints (weather + pollution observations).

Accepts ``text/csv`` bodies that match the corresponding ``/api/export/*.csv``
formats so database rows can be exported, edited offline, and re-imported.
Ingestion is idempotent: observations are keyed on ``(station, timestamp)``
(the unique constraints ``uq_weather_station_ts`` / ``uq_pollution_station_ts``)
and existing rows are updated only when their values actually changed.

Only rows for stations that already exist in the database are imported; unknown
monitors are collected and skipped, matching the curated-station policy used by
the live CPRCB ingestion. Timestamps are stored as naive UTC, consistent with the
rest of the backend.
"""

import csv
import io
import logging
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import PollutionReading, Station, WeatherReading
from ..services.aqi_calculator import calculate_aqi

logger = logging.getLogger("aerocast.import")

router = APIRouter()

MAX_BODY_BYTES = 5_000_000
MAX_ERRORS_REPORTED = 20

#: Station observations are stored as naive UTC throughout the app (see
#: ``cpcb_service.upsert_ncr_data`` and ``app.api.summary``), so an imported
#: timestamp carrying an explicit offset is converted to UTC before the offset is
#: dropped, and a naive one is read as UTC wall-clock. This also keeps
#: ``/api/export/*`` -> ``/api/import/*`` round-trips lossless: ``export.py``
#: serialises ``.isoformat()``, which on PostgreSQL carries a ``+00:00`` marker.

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
)


def _col(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _to_float(value: str | None) -> float | None:
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


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a CSV timestamp into a naive UTC datetime.

    The app-wide storage convention is naive UTC (see
    ``cpcb_service.upsert_ncr_data``), so:

    * a value carrying an explicit offset is **converted** to UTC before the
      offset is dropped, and
    * a naive value is assumed to already be UTC wall-clock.

    Simply stripping the marker, as this used to, discarded the offset instead of
    applying it: ``00:00:00Z`` and ``05:30:00+05:30`` are the same instant, but
    were stored a day apart in effect (as 00:00 and 05:30 respectively). That
    silently moved every such import by up to 5 h 30 m.

    The offset is applied rather than dropped *and* the result is normalised to
    UTC rather than to IST, so the two spellings above collapse to one stored
    value. Under a naive-IST target they would instead agree only by accident,
    while an export taken from PostgreSQL - which emits ``+00:00`` via
    ``.isoformat()`` - would have been shifted on the way back in.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() == "NA":
        return None

    # If the value carries a UTC marker or an explicit offset, let the ISO parser
    # handle it and convert to UTC. A trailing "Z" is normalised to "+00:00"
    # because datetime.fromisoformat on older Pythons rejects the military zone.
    candidate = text.replace(" ", "T", 1) if " " in text and "T" not in text else text
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    looks_offset = bool(re.search(r"[+-]\d{2}:?\d{2}$", candidate))
    if looks_offset:
        try:
            aware = datetime.fromisoformat(candidate)
        except ValueError:
            aware = None
        if aware is not None:
            if aware.tzinfo is None:
                aware = aware.replace(tzinfo=UTC)
            return aware.astimezone(UTC).replace(tzinfo=None)

    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_csv(body: str) -> list[dict[str, str]]:
    # strip the UTF-8 BOM some spreadsheet exports prepend to the header row
    reader = csv.DictReader(io.StringIO(body.lstrip("\ufeff")))
    rows: list[dict[str, str]] = []
    for raw in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
        if not row:
            continue
        rows.append(row)
    return rows


def _check_columns(rows: list[dict[str, str]]) -> None:
    temporal = ("time", "timestamp")
    first = rows[0]
    if not _col(first, "station"):
        raise HTTPException(status_code=400, detail="CSV is missing the required 'station' column.")
    if not any(_col(first, col) for col in temporal):
        raise HTTPException(
            status_code=400,
            detail="CSV is missing the required temporal column ('time' or 'timestamp').",
        )


def _station_map(db: Session) -> dict[str, Station]:
    return {s.name: s for s in db.query(Station).all()}


def _upsert_weather(db: Session, rows: list[dict[str, str]], stations: dict[str, Station]) -> dict[str, Any]:
    summary = {"inserted": 0, "updated": 0, "unchanged": 0, "unknown_stations": [], "errors": []}

    def _err(message: str) -> None:
        if len(summary["errors"]) < MAX_ERRORS_REPORTED:
            summary["errors"].append(message)

    for row in rows:
        name = _col(row, "station")
        if not name:
            continue
        station = stations.get(name)
        ts = _parse_timestamp(_col(row, "time", "timestamp"))
        if station is None:
            if name not in summary["unknown_stations"]:
                summary["unknown_stations"].append(name)
            continue
        if ts is None:
            _err(f"row ignored: unparsable timestamp {_col(row, 'time', 'timestamp')!r}")
            continue

        values = {
            "temperature": _to_float(_col(row, "temperature_2m", "temperature")),
            "humidity": _to_float(_col(row, "relative_humidity_2m", "humidity")),
            "pressure_msl": _to_float(_col(row, "pressure_msl")),
            "surface_pressure": _to_float(_col(row, "surface_pressure")),
            "wind_speed": _to_float(_col(row, "wind_speed_10m", "wind_speed")),
            "wind_direction": _to_float(_col(row, "wind_direction_10m", "wind_direction")),
            "precipitation": _to_float(_col(row, "precipitation")),
            "cloud_cover": _to_float(_col(row, "cloud_cover")),
            "pbl_height": _to_float(_col(row, "boundary_layer_height", "pbl_height")),
        }
        _upsert_row(
            db, WeatherReading, station.id, ts, values, summary,
            compare_fields=tuple(values),
        )
    return summary


def _upsert_pollution(db: Session, rows: list[dict[str, str]], stations: dict[str, Station]) -> dict[str, Any]:
    summary = {"inserted": 0, "updated": 0, "unchanged": 0, "unknown_stations": [], "errors": []}

    def _err(message: str) -> None:
        if len(summary["errors"]) < MAX_ERRORS_REPORTED:
            summary["errors"].append(message)

    _FIELDS = ("pm25", "pm10", "o3", "no2", "so2", "co")
    for row in rows:
        name = _col(row, "station")
        if not name:
            continue
        station = stations.get(name)
        ts = _parse_timestamp(_col(row, "timestamp", "time"))
        if station is None:
            if name not in summary["unknown_stations"]:
                summary["unknown_stations"].append(name)
            continue
        if ts is None:
            _err(f"row ignored: unparsable timestamp {_col(row, 'timestamp', 'time')!r}")
            continue

        values = {key: _to_float(_col(row, key)) for key in _FIELDS}
        aqi = _to_float(_col(row, "aqi"))
        if aqi is None:
            aqi, _, _ = calculate_aqi(
                values["pm25"], values["pm10"], values["o3"],
                values["no2"], values["so2"], values["co"],
            )
        values["aqi"] = aqi
        _upsert_row(
            db, PollutionReading, station.id, ts, values, summary,
            compare_fields=_FIELDS + ("aqi",),
        )
    return summary


def _upsert_row(db, model, station_id: int, ts: datetime, values: dict[str, Any], summary, compare_fields) -> None:
    existing = (
        db.query(model)
        .filter(model.station_id == station_id, model.timestamp == ts)
        .first()
    )
    if existing is None:
        db.add(model(station_id=station_id, timestamp=ts, **values))
        summary["inserted"] += 1
        return
    changed = any(getattr(existing, key) != value for key, value in values.items())
    if not changed:
        summary["unchanged"] += 1
        return
    for key, value in values.items():
        setattr(existing, key, value)
    summary["updated"] += 1


@router.post("/import/weather")
def import_weather(
    db: Session = Depends(get_db),
    body: str = Body(..., media_type="text/csv"),
):
    """Import weather observations from a CSV body. Columns follow the
    ``/api/export/weather.csv`` format (``station,time,temperature_2m,...``).
    """
    if len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="CSV body too large (max 5 MB).")
    rows = _parse_csv(body)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV contains no data rows.")
    _check_columns(rows)
    summary = _upsert_weather(db, rows, _station_map(db))
    db.commit()
    return {"dataset": "weather", "rows": len(rows), **summary}


@router.post("/import/pollution")
def import_pollution(
    db: Session = Depends(get_db),
    body: str = Body(..., media_type="text/csv"),
):
    """Import CPCB pollution observations from a CSV body. Columns follow the
    ``/api/export/pollution.csv`` format (``station,timestamp,pm25,...``). When
    ``aqi`` is blank it is recomputed from the six criteria pollutants.
    """
    if len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="CSV body too large (max 5 MB).")
    rows = _parse_csv(body)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV contains no data rows.")
    _check_columns(rows)
    summary = _upsert_pollution(db, rows, _station_map(db))
    db.commit()
    return {"dataset": "pollution", "rows": len(rows), **summary}
