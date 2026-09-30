"""High-resolution spatial forecasting layer for Delhi NCR.

Produces a gridded AQI / pollutant surface across the NCR domain by
interpolating the per-station coupled forecasts. Uses inverse-distance
weighting (IDW) informed by wind advection, so the map also reflects
prevailing transport direction.

The NCR domain is wrapped at ~0.02deg (~2.2km) resolution between
[28.2..28.9]N x [76.6..77.5]E, matching the Delhi NCR administrative extent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .aqi_calculator import get_aqi_category

NCR_BOUNDS = {
    "lat_min": 28.2,
    "lat_max": 28.9,
    "lon_min": 76.6,
    "lon_max": 77.5,
}
GRID_STEP = 0.02  # ~2.2 km cell
POWER = 2.0  # IDW power
ADVECTION_WEIGHT = 0.25  # how strongly wind shifts the field downwind


def build_grid() -> tuple[np.ndarray, np.ndarray]:
    lat0, lat1 = NCR_BOUNDS["lat_min"], NCR_BOUNDS["lat_max"]
    lon0, lon1 = NCR_BOUNDS["lon_min"], NCR_BOUNDS["lon_max"]
    nlat = int(round((lat1 - lat0) / GRID_STEP)) + 1
    nlon = int(round((lon1 - lon0) / GRID_STEP)) + 1
    lats = lat0 + np.arange(nlat) * GRID_STEP
    lons = lon0 + np.arange(nlon) * GRID_STEP
    return lats, lons


def idw_interpolate(
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    station_values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    power: float = POWER,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation of station values onto a grid."""
    station_lats = np.asarray(station_lats, dtype=float)
    station_lons = np.asarray(station_lons, dtype=float)
    station_values = np.asarray(station_values, dtype=float)

    out = np.full((len(grid_lats), len(grid_lons)), np.nan)
    for i, glat in enumerate(grid_lats):
        for j, glon in enumerate(grid_lons):
            dlat = (station_lats - glat) * 110.0
            dlon = (station_lons - glon) * 100.0 * np.cos(np.deg2rad(glat))
            dist = np.sqrt(dlat**2 + dlon**2)
            near = dist < 0.0001
            if near.any():
                out[i, j] = station_values[near][0]
                continue
            w = 1.0 / (dist**power + 1e-6)
            w = w / w.sum()
            valid = ~np.isnan(station_values)
            if valid.any():
                masked = station_values[valid]
                wm = w[valid]
                out[i, j] = float((wm * masked).sum() / wm.sum())
    return out


def advective_shift(
    lon_idx: np.ndarray, lat_idx: np.ndarray, wind_dir: float, wind_speed: float
) -> tuple[np.ndarray, np.ndarray]:
    """Displace grid indices in the downwind direction by a wind-dependent shift.

    Returns (shifted_j, shifted_i) index arrays applying advection so the
    pollutant field is biased toward the transport direction.
    """
    deg = float(wind_dir or 0.0)
    speed = float(wind_speed or 0.0)
    rad = np.deg2rad(deg)
    # Meteorological wind direction is the compass bearing the wind blows *from*,
    # so the transport vector is the negation of that bearing. This matches
    # ml/preprocessing/weather_processor.py::compute_wind_components, which is the
    # project's own reference for this conversion.
    #
    # Without the negation a 90 deg (east) wind - which travels west - shifted the
    # pollutant field *east*, i.e. upwind, putting the plume on the wrong side of
    # every station.
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    # convert m/s to grid cells over a nominal 3h transport window
    hours = 3.0
    dlon_cells = u * 3600 * hours / (GRID_STEP * 100000.0)
    dlat_cells = v * 3600 * hours / (GRID_STEP * 111000.0)
    return (lon_idx.astype(float) + dlon_cells, lat_idx.astype(float) + dlat_cells)


def _clip_grid_indices(inds: np.ndarray, limit: int) -> np.ndarray:
    inds = np.floor(inds).astype(int)
    return np.clip(inds, 0, limit - 1)


def compute_ncr_grid(
    stations: list,
    forecasts_by_station: dict,
    horizon_hours: int,
    wind_dir: float | None = None,
    wind_speed: float | None = None,
) -> dict[str, Any]:
    """Interpolate a selected horizon's AQI across the NCR domain.

    Args:
        stations: list of DB station objects (have .name/.latitude/.longitude).
        forecasts_by_station: {name: [ForecastPoint dicts]} for the horizon set.
        horizon_hours: which horizon to render.
        wind_dir / wind_speed: prevailing transport conditions.

    Returns a GeoJSON-style structure for the dashboard map.
    """
    lats, lons = build_grid()
    result: dict[str, Any] = {
        "horizon_hours": horizon_hours,
        "step_deg": GRID_STEP,
        "bounds": NCR_BOUNDS,
        "wind_dir": wind_dir,
        "wind_speed": wind_speed,
        "cells": [],
        "extent": {
            "lats_min": float(lats.min()),
            "lats_max": float(lats.max()),
            "lons_min": float(lons.min()),
            "lons_max": float(lons.max()),
        },
    }

    sit_lats, sit_lons, sit_vals = [], [], []
    for s in stations:
        fs = forecasts_by_station.get(s.name) or []
        point = next((f for f in fs if f.get("horizon_hours") == horizon_hours), None)
        if point is None or point.get("aqi_pred") is None:
            continue
        sit_lats.append(s.latitude)
        sit_lons.append(s.longitude)
        sit_vals.append(point["aqi_pred"])

    if len(sit_vals) < 1:
        return result

    field = idw_interpolate(
        np.array(sit_lats),
        np.array(sit_lons),
        np.array(sit_vals),
        lats,
        lons,
    )

    # optional advection bias
    if wind_dir is not None and wind_speed is not None and wind_speed > 1.0:
        jj, ii = np.meshgrid(np.arange(lons.size), np.arange(lats.size))
        sj, si = advective_shift(jj, ii, wind_dir, wind_speed)
        shifted = np.full(field.shape, np.nan)
        shifted[_clip_grid_indices(si, lats.size), _clip_grid_indices(sj, lons.size)] = field
        # Outside the shifted footprint `shifted` is NaN. That NaN used to
        # propagate through the blend, punching holes in the map wherever advection
        # pushed a cell past the domain edge. The fallback keeps the original field
        # in those cells, so the border degrades to "no advection bias" instead of
        # disappearing.
        blended = (1 - ADVECTION_WEIGHT) * field + ADVECTION_WEIGHT * shifted
        field = np.where(np.isnan(blended), field, blended)

    for i in range(lats.size):
        for j in range(lons.size):
            aqi = field[i, j]
            if np.isnan(aqi):
                continue
            aqi = int(round(float(aqi)))
            category, _ = get_aqi_category(aqi)
            result["cells"].append(
                {
                    "lat": round(float(lats[i]), 4),
                    "lon": round(float(lons[j]), 4),
                    "aqi": aqi,
                    "aqi_category": category,
                }
            )

    result["grid_size"] = [len(result["cells"]), len(lats), len(lons)]
    return result
