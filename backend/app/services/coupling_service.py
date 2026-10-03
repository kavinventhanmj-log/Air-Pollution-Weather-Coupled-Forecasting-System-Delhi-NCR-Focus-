"""Coupling-service: assemble real stored observations -> coupling features.

Bridge between the database and :mod:`ml.features.coupling_engine`. It reads
the latest stored CPCB pollution, Open-Meteo weather (+ stored vertical
pressure-level temperatures), an Open-Meteo *forecast* for the 72h look-ahead
and NASA FIRMS fire observations for a station / region, then runs the
deterministic coupling-engine feature computation.

Nothing here fabricates values: when a field is missing the corresponding input
is ``None`` and the engine reports the feature as unavailable with an explicit
basis string. Future weather comes from the live forecast provider (kept
in-memory, never written to ``weather_observations``); stored observations are
used as the fallback when the provider is unavailable.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from ml.features.atmospheric_profile import combine_inversion
from ml.features.coupling_engine import CouplingInputs, compute_coupling_features
from ml.features.fire_impact import DEFAULT_MAX_DISTANCE_KM, compute_fire_impact

from .refresh_service import fetch_forecast_hours

logger = logging.getLogger("aerocast.coupling")


def _float(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _pressure_level_temps(reading) -> dict[float, float]:
    temps: dict[float, float] = {}
    for p in (1000, 925, 850, 700):
        val = _float(getattr(reading, f"temperature_{p}hPa", None))
        if val is not None:
            temps[p] = val
    return temps


def _inversion_from(reading) -> dict[str, Any]:
    temps = _pressure_level_temps(reading)
    pbl = _float(getattr(reading, "pbl_height", None))
    return combine_inversion(pbl, temps if len(temps) >= 2 else None)


def _fire_impact_from_rows(fire_rows: list, lat: float, lon: float, wind_dir: float | None, wind_speed: float | None) -> dict[str, Any]:
    """Compute fire impact from already-loaded FireReading rows (pure helper)."""
    empty = {
        "fire_count": 0,
        "fire_impact_score": 0.0,
        "nearest_fire_distance": DEFAULT_MAX_DISTANCE_KM + 1.0,
        "wind_aligned_fire_count": 0,
        "wind_alignment_pct": 0.0,
        "transport_time_hours": None,
        "transport_risk": 0.0,
        "transport_risk_level": "none",
        "stubble_impact_score": 0.0,
    }
    if lat is None or lon is None:
        # No station geometry means no fire geometry can be computed honestly.
        return empty
    if not fire_rows:
        return empty
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "lat": _float(f.latitude),
                "lon": _float(f.longitude),
                "frp": _float(f.frp) if f.frp is not None else 1.0,
            }
            for f in fire_rows
        ]
    )
    if df.empty:
        return empty
    impact = compute_fire_impact(
        df,
        lat,
        lon,
        wind_dir if wind_dir is not None else 0.0,
        wind_speed or 0.0,
        max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    )
    impact["nearest_fire_distance"] = round(float(impact["nearest_fire_distance"]), 1)
    return impact


def _load_recent_fires(db, limit: int = 500):
    from ..models.db_models import FireReading

    return (
        db.query(FireReading)
        .filter(FireReading.synthetic.is_(False))
        .order_by(FireReading.acq_date.desc())
        .limit(limit)
        .all()
    )


def get_coupling_inputs(db, station) -> tuple[CouplingInputs, dict[str, Any]]:
    """Build ``CouplingInputs`` + provenance from the station's stored data.

    Returns ``(inputs, provenance)`` where ``provenance`` maps the same inputs
    to the actual DB rows that produced them (honest traceability).
    """
    from ..models.db_models import PollutionReading, WeatherReading

    wx = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station.id)
        .order_by(WeatherReading.timestamp.desc())
        .first()
    )
    poll = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station.id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )

    inversion = _inversion_from(wx) if wx is not None else None

    fires = _load_recent_fires(db)
    lat = _float(getattr(station, "latitude", None))
    lon = _float(getattr(station, "longitude", None))
    wind_dir = _float(getattr(wx, "wind_direction", None)) if wx is not None else None
    wind_speed = _float(getattr(wx, "wind_speed", None)) if wx is not None else None
    fire = _fire_impact_from_rows(fires, lat, lon, wind_dir, wind_speed)

    inputs = CouplingInputs(
        temperature_c=_float(getattr(wx, "temperature", None)) if wx is not None else None,
        humidity_pct=_float(getattr(wx, "humidity", None)) if wx is not None else None,
        pressure_hpa=_float(getattr(wx, "pressure_msl", None)) if wx is not None else None,
        wind_speed_mps=wind_speed,
        wind_direction_deg=wind_dir,
        pbl_height_m=_float(getattr(wx, "pbl_height", None)) if wx is not None else None,
        pm25_ugm3=_float(getattr(poll, "pm25", None)) if poll is not None else None,
        pm10_ugm3=_float(getattr(poll, "pm10", None)) if poll is not None else None,
        no2_ugm3=_float(getattr(poll, "no2", None)) if poll is not None else None,
        o3_ugm3=_float(getattr(poll, "o3", None)) if poll is not None else None,
        inversion_detected=bool(inversion.get("inversion_detected", False)) if inversion else None,
        inversion_strength=_float(inversion.get("inversion_strength")) if inversion is not None else None,
        inversion_category=str(inversion.get("inversion_category", "unknown")) if inversion else None,
        fire_count=int(fire["fire_count"]) if fire["fire_count"] else None,
        upwind_fire_count=int(fire["wind_aligned_fire_count"]) if fire["wind_aligned_fire_count"] else None,
        nearest_fire_distance_km=fire["nearest_fire_distance"],
        fire_impact_score=fire["fire_impact_score"],
        wind_alignment_pct=fire["wind_alignment_pct"],
        transport_time_hours=fire["transport_time_hours"],
    )

    provenance = {
        "weather_reading_timestamp": wx.timestamp.isoformat() if wx and wx.timestamp else None,
        "pollution_reading_timestamp": poll.timestamp.isoformat() if poll and poll.timestamp else None,
        "inversion_source": inversion.get("inversion_source", "unavailable") if inversion else "unavailable",
        "inversion_profile_available": bool(inversion.get("profile_available", False)) if inversion else False,
        "inversion_category": str(inversion.get("inversion_category", "unknown")) if inversion else None,
        "fire_impact_basis": (
            f"{int(fire['fire_count'])} stored fire(s) within {DEFAULT_MAX_DISTANCE_KM:.0f} km; "
            f"{int(fire['wind_aligned_fire_count'])} upwind of the surface wind vector"
        ),
        "note": (
            "All inputs are the latest stored observations. Inversion uses the "
            "real vertical pressure-level lapse-rate when >=2 stored levels exist, "
            "otherwise the labelled PBL-height proxy."
        ),
    }
    return inputs, provenance


def get_coupling_features(db, station) -> dict[str, Any]:
    """Full coupling-features packet (inputs + features + methodology + provenance).

    Persists the snapshot to the ``coupling_states`` table (write-through,
    one latest row per station) and labels the state with ``coupling_state``
    (feedback-surrogate band), ``coupling_domains`` (which feature domains
    were present) and ``data_quality`` (how many of the nine features were
    computable).
    """
    inputs, provenance = get_coupling_inputs(db, station)
    result = compute_coupling_features(inputs)
    result["station"] = station.name
    result["timestamp"] = datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"
    result["provenance"] = provenance
    result["coupling_state"] = _coupling_state(result["features"])
    result["coupling_domains"] = _coupling_domains(result["features"])
    result["data_quality"] = _data_quality(result["features"])
    _persist_coupling_state(db, station, result)
    return result


_COUPLING_DOMAINS = [
    ("aerosol_accumulation_potential", "aerosol"),
    ("dispersion_potential", "atmospheric"),
    ("accumulation_potential", "atmospheric"),
    ("inversion_trapping_potential", "atmospheric"),
    ("pollution_stagnation_index", "atmospheric"),
    ("meteorology_pollution_interaction", "feedback"),
    ("fire_transport_influence", "fire"),
    ("regional_transport_potential", "fire"),
    ("ozone_photochemical_potential", "ozone"),
]


def _coupling_state(features: dict[str, Any]) -> str:
    """Band of the composite feedback-surrogate feature (NONE / LOW / MODERATE / HIGH)."""
    value = features.get("meteorology_pollution_interaction", {}).get("value")
    if value is None:
        return "NONE"
    if value >= 0.66:
        return "HIGH"
    if value >= 0.33:
        return "MODERATE"
    return "LOW"


def _coupling_domains(features: dict[str, Any]) -> str:
    """Which coupling domains contributed stored data (e.g. 'aerosol+atmospheric+fire')."""
    present: list[str] = []
    for feature_name, domain in _COUPLING_DOMAINS:
        if features.get(feature_name, {}).get("available"):
            if domain not in present:
                present.append(domain)
    return "+".join(present) if present else "none"


def _data_quality(features: dict[str, Any]) -> str:
    """How many of the nine coupling features were computable from stored data."""
    available = sum(1 for f in features.values() if f.get("available"))
    if available == len(features):
        return "GOOD"
    if available >= 6:
        return "PARTIAL"
    if available >= 1:
        return "SPARSE"
    return "UNAVAILABLE"


def _compass(direction: float | None) -> str | None:
    if direction is None:
        return None
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    deg = float(direction) % 360.0
    return pts[int((deg + 11.25) / 22.5) % 16]


def _persist_coupling_state(db, station, result: dict[str, Any]) -> None:
    """Upsert the latest coupling snapshot for a station (Phase 30 persistence)."""
    from ..models.db_models import CouplingState

    inputs = result["inputs"]
    feats = result["features"]
    prov = result.get("provenance", {})

    row = db.query(CouplingState).filter(CouplingState.station_id == station.id).first()
    if row is None:
        row = CouplingState(station_id=station.id)
        db.add(row)

    row.computed_at = datetime.now(UTC).replace(tzinfo=None)
    row.wind_speed_mps = inputs.get("wind_speed_mps")
    row.wind_direction_deg = inputs.get("wind_direction_deg")
    row.pbl_height_m = inputs.get("pbl_height_m")
    row.inversion_detected = inputs.get("inversion_detected")
    row.inversion_strength = inputs.get("inversion_strength")
    row.inversion_category = inputs.get("inversion_category")
    row.inversion_source = prov.get("inversion_source")
    row.fire_count = inputs.get("fire_count")
    row.upwind_fire_count = inputs.get("upwind_fire_count")
    row.nearest_fire_distance_km = inputs.get("nearest_fire_distance_km")
    row.fire_impact_score = inputs.get("fire_impact_score")
    row.wind_alignment_pct = inputs.get("wind_alignment_pct")
    row.fire_transport_direction = _compass(
        (inputs.get("wind_direction_deg") or 0.0) + 180.0
        if inputs.get("wind_direction_deg") is not None
        else None
    )
    row.fire_transport_time_hours = inputs.get("transport_time_hours")
    row.fire_transport_influence = feats["fire_transport_influence"]["value"]
    row.dispersion_potential = feats["dispersion_potential"]["value"]
    row.accumulation_potential = feats["accumulation_potential"]["value"]
    row.inversion_trapping_potential = feats["inversion_trapping_potential"]["value"]
    row.pollution_stagnation_index = feats["pollution_stagnation_index"]["value"]
    row.aerosol_accumulation_potential = feats["aerosol_accumulation_potential"]["value"]
    row.regional_transport_potential = feats["regional_transport_potential"]["value"]
    row.ozone_photochemical_potential = feats["ozone_photochemical_potential"]["value"]
    row.meteorology_pollution_interaction = feats["meteorology_pollution_interaction"]["value"]
    row.coupling_state = result["coupling_state"]
    row.coupling_domains = result["coupling_domains"]
    row.data_quality = result["data_quality"]
    row.weather_reading_timestamp = _parse_dt(prov.get("weather_reading_timestamp"))
    row.pollution_reading_timestamp = _parse_dt(prov.get("pollution_reading_timestamp"))
    db.commit()


def _parse_dt(value):
    """Parse an ISO-8601 string (or pass through a datetime) for DB storage."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _utc_naive(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except (TypeError, ValueError):
        return None


def _utc_naive(value: datetime) -> datetime:
    """Normalise a DB timestamp to naive-UTC (the repo-wide convention).

    SQLite stores naive datetimes while PostgreSQL ``timestamp with time zone``
    columns come back tz-aware. Subtracting one of each raises
    ``TypeError: can't subtract offset-naive and offset-aware datetimes``, so
    every comparison path normalises first (mirrors
    ``forecast_service._as_naive_utc`` / ``atmosphere_service._utc_naive``).
    A naive value is treated as UTC wall-clock and left unchanged.
    """
    if value is None:
        return None
    try:
        aware = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    except (TypeError, ValueError, OSError):
        return value
    return aware.replace(tzinfo=None)


_FORECAST_WINDOW_HOURS = 72


def _weather_rows_from_forecast_df(df) -> list[Any]:
    """Project an Open-Meteo forecast frame into ``WeatherReading``-shaped rows.

    Kept as plain in-memory objects (never written to ``weather_observations``)
    so the existing nearest-match and inversion helpers reuse the attribute
    names/units a stored ``WeatherReading`` exposes while the data provably
    remains detached from the historical table.
    """
    rows = []
    for _, r in df.iterrows():
        rows.append(
            SimpleNamespace(
                timestamp=r["time"].to_pydatetime(),
                temperature=_float(r.get("temperature_2m")),
                humidity=_float(r.get("relative_humidity_2m")),
                pressure_msl=_float(r.get("pressure_msl")),
                surface_pressure=_float(r.get("surface_pressure")),
                wind_speed=_float(r.get("wind_speed_10m")),
                wind_direction=_float(r.get("wind_direction_10m")),
                precipitation=_float(r.get("precipitation")),
                cloud_cover=_float(r.get("cloud_cover")),
                pbl_height=_float(r.get("boundary_layer_height")),
                temperature_1000hPa=_float(r.get("temperature_1000hPa")),
                temperature_925hPa=_float(r.get("temperature_925hPa")),
                temperature_850hPa=_float(r.get("temperature_850hPa")),
                temperature_700hPa=_float(r.get("temperature_700hPa")),
            )
        )
    return rows


def _load_future_weather_rows(lat: float | None, lon: float | None) -> list[Any]:
    """Best-effort Open-Meteo *forecast* for the 72h look-ahead (fetched once).

    One provider request per context build — never per horizon — and only when
    the build actually runs (the route wraps it in a 300s TTL cache). Returns
    ``[]`` on any provider failure so the caller falls back to stored
    observations.
    """
    if lat is None or lon is None:
        return []
    try:
        df = fetch_forecast_hours(lat, lon)
    except Exception as exc:  # noqa: BLE001 - provider outage must not break /context
        logger.warning("weather forecast fetch failed (%s) — using stored observations", exc)
        return []
    if df is None or df.empty or "time" not in df.columns:
        return []
    return _weather_rows_from_forecast_df(df)


def get_forecast_context(db, station) -> dict[str, Any]:
    """Per-horizon atmospheric + coupling context for the station's 72h window.

    For each forecast horizon (1..72 h) the nearest weather row for the target
    timestamp supplies the atmospheric state and a photochemical/inversion
    estimate; regional fire impact and observed pollution are shared across
    horizons (they are region-wide, not per-hour).

    Weather source: live Open-Meteo *forecast* hours (fetched once, aligned to
    each ``target = now + h``); the stored ``weather_observations`` archive is
    only the fallback when the provider is unavailable. Missing rows produce
    ``None`` fields — the UI renders them as unavailable.
    """
    from ..models.db_models import PollutionReading, WeatherReading

    now = datetime.now(UTC).replace(tzinfo=None)

    poll = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station.id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )
    pm25 = _float(getattr(poll, "pm25", None)) if poll is not None else None
    pm10 = _float(getattr(poll, "pm10", None)) if poll is not None else None
    no2 = _float(getattr(poll, "no2", None)) if poll is not None else None
    o3 = _float(getattr(poll, "o3", None)) if poll is not None else None

    fires = _load_recent_fires(db)
    lat = _float(getattr(station, "latitude", None))
    lon = _float(getattr(station, "longitude", None))
    latest_wx = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station.id)
        .order_by(WeatherReading.timestamp.desc())
        .first()
    )
    wind_dir = _float(getattr(latest_wx, "wind_direction", None)) if latest_wx is not None else None
    wind_speed = _float(getattr(latest_wx, "wind_speed", None)) if latest_wx is not None else None
    fire = _fire_impact_from_rows(fires, lat, lon, wind_dir, wind_speed)

    # Stored observations remain the fallback; the live Open-Meteo *forecast*
    # (fetched once as in-memory rows) supplies real future weather for the full
    # 72h look-ahead, so atmospheric context is not left null just because the
    # archive currently stops around today. Only one source feeds the matcher, so
    # forecast rows are never treated as (or merged into) historical observations.
    stored_wx_rows = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station.id)
        .order_by(WeatherReading.timestamp.asc())
        .all()
    )
    future_wx_rows = _load_future_weather_rows(lat, lon)
    wx_rows = future_wx_rows if future_wx_rows else stored_wx_rows

    contexts: list[dict[str, Any]] = []
    for h in range(1, _FORECAST_WINDOW_HOURS + 1):
        target = now + timedelta(hours=h)
        wx = _nearest_weather_row(wx_rows, target, tolerance=timedelta(hours=2))
        inversion = _inversion_from(wx) if wx is not None else None

        ctx_wind_dir = _float(getattr(wx, "wind_direction", None)) if wx is not None else None
        ctx_wind_speed = _float(getattr(wx, "wind_speed", None)) if wx is not None else None
        inputs = CouplingInputs(
            temperature_c=_float(getattr(wx, "temperature", None)) if wx is not None else None,
            humidity_pct=_float(getattr(wx, "humidity", None)) if wx is not None else None,
            pressure_hpa=_float(getattr(wx, "pressure_msl", None)) if wx is not None else None,
            wind_speed_mps=ctx_wind_speed,
            wind_direction_deg=ctx_wind_dir or wind_dir,
            pbl_height_m=_float(getattr(wx, "pbl_height", None)) if wx is not None else None,
            pm25_ugm3=pm25,
            pm10_ugm3=pm10,
            no2_ugm3=no2,
            o3_ugm3=o3,
            inversion_detected=bool(inversion.get("inversion_detected", False)) if inversion else None,
            inversion_strength=_float(inversion.get("inversion_strength")) if inversion is not None else None,
            inversion_category=str(inversion.get("inversion_category", "unknown")) if inversion else None,
            fire_count=int(fire["fire_count"]) if fire["fire_count"] else None,
            upwind_fire_count=int(fire["wind_aligned_fire_count"]) if fire["wind_aligned_fire_count"] else None,
            nearest_fire_distance_km=fire["nearest_fire_distance"],
            fire_impact_score=fire["fire_impact_score"],
            wind_alignment_pct=fire["wind_alignment_pct"],
            transport_time_hours=fire["transport_time_hours"],
        )
        feat = compute_coupling_features(inputs)["features"]

        contexts.append(
            {
                "horizon_hours": h,
                "target_timestamp": target.isoformat() + "Z",
                "weather_match_timestamp": _utc_naive(wx.timestamp).isoformat() + "Z" if wx and wx.timestamp else None,
                "temperature_c": inputs.temperature_c,
                "humidity_pct": inputs.humidity_pct,
                "pressure_hpa": inputs.pressure_hpa,
                "wind_speed_mps": inputs.wind_speed_mps,
                "wind_direction_deg": inputs.wind_direction_deg,
                "pbl_height_m": inputs.pbl_height_m,
                "inversion_detected": inputs.inversion_detected,
                "inversion_strength": inputs.inversion_strength,
                "inversion_category": inputs.inversion_category,
                "inversion_source": inversion.get("inversion_source") if inversion is not None else None,
                "dispersion_potential": feat["dispersion_potential"]["value"],
                "accumulation_potential": feat["accumulation_potential"]["value"],
                "inversion_trapping_potential": feat["inversion_trapping_potential"]["value"],
                "pollution_stagnation_index": feat["pollution_stagnation_index"]["value"],
                "fire_transport_influence": feat["fire_transport_influence"]["value"],
                "regional_transport_potential": feat["regional_transport_potential"]["value"],
                "ozone_photochemical_potential": feat["ozone_photochemical_potential"]["value"],
                "meteorology_pollution_interaction": feat["meteorology_pollution_interaction"]["value"],
            }
        )

    return {
        "station": station.name,
        "generated_at": datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z",
        "units": {
            "wind_speed_mps": "meters/second (meteorological FROM direction)",
            "pbl_height_m": "meters",
            "temperature_c": "degrees Celsius",
            "pressure_hpa": "hectopascals",
            "features": "normalized 0..1 (None = input data unavailable)",
        },
        "regional_note": (
            "Fire-influence and observed-pollution terms are regional and reuse the "
            "latest stored observations across all horizons. Atmospheric fields vary "
            "per horizon: live Open-Meteo forecast hours are matched to each target "
            "timestamp (stored observations are the fallback when the provider is "
            "unavailable)."
        ),
        "horizons": contexts,
    }


def _nearest_weather_row(rows, target, tolerance: timedelta, within_max_hours: int = 6):
    """Nearest weather row to *target* time, within *tolerance*.

    A row qualifies only when it satisfies **both** bounds, so the check rejects
    on ``or`` rather than ``and``: the old ``and`` skipped a row only when it
    exceeded *both*, which meant any row inside the 6h ceiling was accepted no
    matter how far it sat from the target, and the 2h ``tolerance`` was never
    actually enforced. With the fix the effective window is
    ``min(tolerance, within_max_hours)`` — hourly forecast/observation steps
    match within the 2h the context builder asks for, and a gap in the source
    yields ``None`` instead of a stale row pulled in from hours away.

    ``rows`` come from the DB (naive SQLite / tz-aware PostgreSQL timestamps) or
    from the in-memory forecast projection; ``target`` is naive-UTC. All are
    normalised to naive-UTC before ``abs(a - b)`` so the subtraction never raises.
    """
    best = None
    best_delta = None
    max_delta = min(tolerance, timedelta(hours=within_max_hours))
    for r in rows:
        ts = _utc_naive(r.timestamp)
        if ts is None:
            continue
        d = abs(ts - target)
        if d > max_delta:
            continue
        if best_delta is None or d < best_delta:
            best = r
            best_delta = d
    return best
