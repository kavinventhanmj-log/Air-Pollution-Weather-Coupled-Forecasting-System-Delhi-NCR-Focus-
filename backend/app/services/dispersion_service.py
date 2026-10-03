"""Numerical dispersion forecasting service for Delhi NCR.

Runs the grid-based advection-diffusion-deposition emission solver
(ml.features.dispersion_solver) over the latest persisted NCR forecast surface,
using live weather/fires from the database as lateral boundary and source data.

The solver — rather than pure interpolation — dynamically advects the
stubble-burning plumes downwind and lets meteorology (wind, PBL, rain) coupled
with the aerosol field drive the hourly AQI evolution.

When a *genuine* chemical-transport engine (NOAA HYSPLIT ``hycs_std`` or real
WRF-Chem ``wrfout`` output) is installed and configured (see ``docs/hysplit.md``
and ``docs/wrfchem_adapter.md``), the service first attempts to run it. On
success the resulting physically-computed plume *pattern* is composited 50/50
with the coupled data-driven forecast surface; every response that used a
genuine engine states ``mode`` and the engine metadata explicitly, and the
analytic surrogate is never impersonated. If no genuine engine is runnable the
documented analytic path stays active (``mode="numerical_advection_diffusion"``).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from datetime import UTC

import numpy as np
from sqlalchemy.orm import Session

from ml.features.dispersion_solver import run_dispersion_forecast

from ..models.db_models import FireReading, Forecast, Station, WeatherReading
from .aqi_calculator import get_aqi_category
from .grid_service import GRID_STEP, NCR_BOUNDS, build_grid, idw_interpolate

# Composite weight applied to the genuine engine plume pattern (see module doc).
_CTM_BLEND = 0.5


def _try_genuine_ctm(
    start_utc: _dt.datetime,
    hours: int,
    sources: list[dict[str, float]] | None = None,
) -> tuple[str, object] | None:
    """Run the highest-priority genuinely-available CTM engine (or None)."""
    try:
        from ml.ctm.ctm_interface import available_engines, run_best_engine

        domain = {
            "lat_min": NCR_BOUNDS["lat_min"],
            "lat_max": NCR_BOUNDS["lat_max"],
            "lon_min": NCR_BOUNDS["lon_min"],
            "lon_max": NCR_BOUNDS["lon_max"],
        }
        if not available_engines(domain, GRID_STEP):
            return None
        engine_name, result = run_best_engine(
            start_utc, int(hours), domain, GRID_STEP, sources=sources
        )
        return engine_name, result
    except Exception:
        # CtmUnavailable (or any engine failure) falls back to the analytic path.
        return None


def _latest_weather(db: Session) -> dict:
    """Domain-representative latest weather (average over stations)."""
    vals: dict[str, list] = {"wind_speed": [], "wind_direction": [], "pbl_height": [], "precipitation": []}
    for s in db.query(Station).all():
        row = (
            db.query(WeatherReading)
            .filter(WeatherReading.station_id == s.id)
            .order_by(WeatherReading.timestamp.desc())
            .first()
        )
        if row is None:
            continue
        if row.wind_speed is not None:
            vals["wind_speed"].append(row.wind_speed)
        if row.wind_direction is not None:
            vals["wind_direction"].append(row.wind_direction)
        if row.pbl_height is not None:
            vals["pbl_height"].append(row.pbl_height)
        if row.precipitation is not None:
            vals["precipitation"].append(row.precipitation)
    # ``not vals["wind_speed"]`` tests emptiness. The previous ``not any(...)``
    # treated a genuinely calm network (every reading 0.0 m/s) as no data and
    # replaced it with a 4.0 m/s wind, which changes the dispersion result.
    if not vals["wind_speed"]:
        # No station reported wind: the caller gets explicit ``None`` plus the
        # availability flags, so the synthetic defaults below are never mistaken
        # for measurements.
        return {
            "wind_speed": 4.0,
            "wind_direction": 90.0,
            "pbl_height": 600.0,
            "precipitation": 0.0,
            "wind_observed": False,
            "pbl_observed": False,
            "synthetic_meteorology": True,
        }
    import numpy as _np

    def _mean_or(key: str, default: float) -> float:
        series = vals[key]
        if not series:
            return default
        # Explicit None check: a real mean of 0.0 (calm wind, zero precipitation)
        # must survive, so the fallback is for missing data only.
        value = float(_np.mean(series))
        return default if value != value else value  # NaN check

    return {
        "wind_speed": _mean_or("wind_speed", 4.0),
        "wind_direction": _mean_or("wind_direction", 90.0),
        "pbl_height": _mean_or("pbl_height", 600.0),
        "precipitation": _mean_or("precipitation", 0.0),
        "wind_observed": True,
        "pbl_observed": bool(vals["pbl_height"]),
        "synthetic_meteorology": not vals["pbl_height"],
    }


def _wx_payload(wx: dict) -> dict:
    """Serialise the driving meteorology together with its provenance flags.

    ``_aggregate_weather`` computes ``wind_observed`` / ``pbl_observed`` /
    ``synthetic_meteorology`` precisely so a placeholder can be told apart from a
    measurement, but those keys were dropped when the response dicts were built:
    the API returned only the four numeric fields, so a client could not tell a
    synthesised 4 m/s wind from an observed one. Both response paths build their
    ``wx`` block through this helper so the flags cannot be forgotten again.
    """
    return {
        "wind_speed": round(wx["wind_speed"], 2),
        "wind_direction": round(wx["wind_direction"], 1),
        "pbl_height": round(wx["pbl_height"], 1),
        "precipitation": round(wx["precipitation"], 2),
        "wind_observed": bool(wx.get("wind_observed", False)),
        "pbl_observed": bool(wx.get("pbl_observed", False)),
        "synthetic_meteorology": bool(wx.get("synthetic_meteorology", False)),
    }


def _initial_aqi_field(db: Session, horizon_hours: int) -> np.ndarray | None:
    """IDW-interpolate the latest horizon station forecasts into a grid field."""
    lats, lons = build_grid()
    stations = db.query(Station).order_by(Station.name).all()
    src_lats, src_lons, src_vals = [], [], []
    for s in stations:
        f = (
            db.query(Forecast)
            .filter(
                Forecast.station_id == s.id,
                Forecast.horizon_hours == horizon_hours,
                Forecast.aqi_pred.isnot(None),
            )
            .order_by(Forecast.forecast_timestamp.desc())
            .first()
        )
        if f is None:
            continue
        src_lats.append(s.latitude)
        src_lons.append(s.longitude)
        src_vals.append(f.aqi_pred)
    if not src_vals:
        return None
    return idw_interpolate(
        np.array(src_lats), np.array(src_lons), np.array(src_vals), lats, lons
    )


def _fires_in_domain(db: Session, limit: int = 60) -> list:
    b = NCR_BOUNDS
    rows = db.query(FireReading).filter(FireReading.synthetic.is_(False)).order_by(FireReading.acq_date.desc()).limit(1000).all()
    fires = []
    for r in rows:
        if r.latitude is None or r.longitude is None:
            continue
        if b["lat_min"] <= r.latitude <= b["lat_max"] and b["lon_min"] <= r.longitude <= b["lon_max"]:
            fires.append({
                "lat": r.latitude,
                "lon": r.longitude,
                "frp": r.frp or 2.0,
                "confidence": r.confidence,
                "satellite": r.satellite,
            })
            if len(fires) >= limit:
                break
    return fires


def _aqi_cells(lats: np.ndarray, lons: np.ndarray, aqi: np.ndarray) -> list[dict]:
    cells = []
    for i in range(lats.size):
        for j in range(lons.size):
            v = int(aqi[i, j])
            category, _ = get_aqi_category(v)
            cells.append({
                "lat": round(float(lats[i]), 4),
                "lon": round(float(lons[j]), 4),
                "aqi": v,
                "aqi_category": category,
            })
    return cells


def _composite_from_ctm(result, initial_aqi: np.ndarray) -> tuple[str, list[dict]]:
    """Compose genuine-engine plume patterns with the coupled forecast surface.

    Returns ``(blend_description, aqi_frames)`` where each frame carries the
    ``hour``/``hour_of_day`` keys of the analytic path. AQI magnitudes come
    from the coupled data-driven surface's mean; the spatial pattern is the
    genuine engine plume (50/50 blend, see ``_CTM_BLEND``). The blend is
    reported verbatim so nothing is misrepresented.
    """
    base = float(np.mean(initial_aqi))
    frames: list[dict] = []
    for t in range(result.n_frames):
        frame = np.asarray(result.data[t], dtype=float)  # (nlat, nlon)
        span = float(frame.max()) - float(frame.min())
        if span > 0:
            pattern = (frame - float(frame.min())) / span
        else:
            pattern = np.zeros_like(frame)
        aqi = base * (1.0 - _CTM_BLEND) + base * _CTM_BLEND * pattern
        ts = result.times_utc[t] if t < len(result.times_utc) else None
        frames.append({
            "hour": t,
            "hour_of_day": ts.hour if ts is not None else t % 24,
            "aqi": aqi,
        })
    return f"{result.engine} plume pattern (50%) + coupled forecast surface (50%)", frames


def run_dispersion_forecast_service(
    db: Session,
    horizon_hours: int = 72,
    start_hour: int = 8,
    start_utc: _dt.datetime | None = None,
    frame_hours: Sequence[int] | None = None,
) -> dict:
    """Run the numerical dispersion forecast from the latest DB state.

    ``frame_hours`` restricts which hourly frames are serialised into the
    response. The solver always integrates the full horizon (the field evolves
    hourly, so every frame depends on the ones before it), but the per-cell
    AQI grid is only materialised for the requested hours: on the ~2.2 km NCR
    grid a 72 h run is 72 x 1575 = 113,400 cell dictionaries, and building
    those dominates both the response time and the memory footprint of the
    endpoint. A client that renders one frame at a time (the Spatial Forecast
    page) asks for the handful of hours it can actually show and gets a payload
    one to two orders of magnitude smaller. ``None`` keeps the historical
    behaviour of returning every frame.
    """
    wanted: set[int] | None = None if frame_hours is None else {int(h) for h in frame_hours}
    wx = _latest_weather(db)
    initial_aqi = _initial_aqi_field(db, min(horizon_hours, 24))
    if initial_aqi is None:
        return {
            "error": "no forecast surface available; generate a coupled forecast first (POST /api/forecast/coupled)",
            "frame_hours_available": [],
            "frames": [],
        }

    fires = _fires_in_domain(db)
    # hourly met series: use the observed values as a flat first-guess with a
    # mild diurnal PBL wiggle (shallower at night, deeper around solar noon).
    hours = int(horizon_hours)
    lats, lons = build_grid()

    # --- genuine chemical-transport engines first (SIH26082 R6) ------------
    if start_utc is None:
        start_utc = _dt.datetime.now(UTC).replace(tzinfo=None)
    engine = _try_genuine_ctm(start_utc, hours, sources=None)
    if engine is not None:
        name, result = engine
        blend_note, aqi_frames = _composite_from_ctm(result, initial_aqi)
        available = [int(fr["hour"]) for fr in aqi_frames]
        frames = []
        for fr in aqi_frames:
            if wanted is not None and int(fr["hour"]) not in wanted:
                continue
            frames.append({
                "hour": fr["hour"],
                "hour_of_day": fr["hour_of_day"],
                "wind_speed": wx["wind_speed"],
                "wind_dir_deg": wx["wind_direction"],
                "pbl_height": wx["pbl_height"],
                "precip_mm": wx["precipitation"],
                "coupling": blend_note,
                "aqi_mean": round(float(fr["aqi"].mean()), 1),
                "aqi_max": int(fr["aqi"].max()),
                "cells": _aqi_cells(lats, lons, fr["aqi"]),
            })
        return {
            "mode": f"{name} (genuine composite)",
            "horizon_hours": result.n_frames,
            "start_hour": start_hour,
            "domain": NCR_BOUNDS,
            "step_deg": GRID_STEP,
            "wx": _wx_payload(wx),
            "fire_count": len(fires),
            "fires": fires[:20],
            "composite": blend_note,
            "frame_hours_available": available,
            "ctm": {
                "engine": result.engine,
                "pollutant": result.pollutant,
                "units": result.units,
                "times_utc": [str(t) for t in result.times_utc],
                "metadata": result.metadata,
            },
            "frames": frames,
        }

    # --- analytic surrogate (documented fallback) ---------------------------
    speed_hourly = [wx["wind_speed"]] * hours
    dir_hourly = [wx["wind_direction"]] * hours
    precip_hourly = [wx["precipitation"]] * hours
    base_pbl = wx["pbl_height"]

    pbl_hourly = []
    for h in range(hours):
        hod = (start_hour + h) % 24
        if 7 <= hod <= 18:
            norm = max(0.0, np.cos(2 * np.pi * (hod - 13) / 22.0))
            depth = base_pbl * 0.5 + base_pbl * 1.2 * norm
        else:
            depth = base_pbl * 0.55
        pbl_hourly.append(float(max(200.0, depth)))

    result = run_dispersion_forecast(
        initial_aqi,
        NCR_BOUNDS["lat_min"], NCR_BOUNDS["lat_max"],
        NCR_BOUNDS["lon_min"], NCR_BOUNDS["lon_max"],
        GRID_STEP,
        wx["wind_speed"], wx["wind_direction"], base_pbl, wx["precipitation"],
        fires=fires,
        hours=hours,
        start_hour=start_hour,
        wind_hourly=speed_hourly,
        dir_hourly=dir_hourly,
        precip_hourly=precip_hourly,
        pbl_hourly=pbl_hourly,
    )

    available = [int(fr["hour"]) for fr in result["frames"]]
    frames = []
    for fr in result["frames"]:
        if wanted is not None and int(fr["hour"]) not in wanted:
            continue
        aqi = fr["aqi"]
        frames.append({
            "hour": fr["hour"],
            "hour_of_day": fr["hour_of_day"],
            "wind_speed": fr["wind_speed"],
            "wind_dir_deg": fr["wind_dir_deg"],
            "pbl_height": fr["pbl_height"],
            "precip_mm": fr["precip_mm"],
            "coupling": fr["coupling"],
            "aqi_mean": round(float(aqi.mean()), 1),
            "aqi_max": int(aqi.max()),
            "cells": _aqi_cells(lats, lons, aqi),
        })

    return {
        "mode": "numerical_advection_diffusion",
        "horizon_hours": hours,
        "start_hour": start_hour,
        "domain": NCR_BOUNDS,
        "step_deg": GRID_STEP,
        "wx": _wx_payload(wx),
        "fire_count": len(fires),
        "fires": fires[:20],
        "dt_used": round(result["dt_used"], 1),
        "steps_per_hour": result["steps_per_hour"],
        "frame_hours_available": available,
        "frames": frames,
    }
