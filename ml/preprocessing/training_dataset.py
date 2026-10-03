"""Historical training-dataset builder for AeroCast-NCR (no model training).

Assembles time-aligned, chronologically-ordered observations for future ML
training on the **target PM2.5**. Everything in this module is deterministic
and leak-aware:

Key decisions (documented in code and re-printed in the summary report)
------------------------------------------------------------------------
* **Time alignment** — pollution and weather are floored to hourly buckets in
  naive-UTC (PostgreSQL stores aware timestamps; we normalise everything to
  naive-UTC so SQLite/PG behave identically). Weather is joined to the
  pollution hour anchor and forward-filled up to ``MAX_FFILL_HOURS`` (3 h) per
  station. Rows with no PM2.5 target are dropped (nothing to supervise on).
* **Duplicates** — (station, hour) pairs are de-duplicated keeping the *latest*
  raw observation in the bucket; the removed count is reported.
* **Outliers** — implausible target PM2.5 (<=0 or >1000 ug/m3) is dropped
  (reported); all numeric features get an ``outlier_<col>`` IQR*3 flag column
  (values left in place; documented counts are reported).
* **No future-data leakage**:
    - ``pm25`` at hour *t* is the TARGET and is never used as a feature.
    - Lag features ``pm25_lag{h}`` == ``pm25[t-h]`` (shift h, per station).
    - Rolling features are causal: ``pm25.shift(1).rolling(w)`` — the current
      hour's target is excluded from every rolling window.
    - Fire features use only fires with ``acq_time <= t`` within the trailing
      window (no future fires).
* **Splits** — chronological, contiguous, no shuffling:
  train 60% | validation 20% | test 20% by time order of the pooled panel.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("aerocast.training_dataset")

TARGET = "pm25"
LAG_HOURS = [1, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]
MAX_FFILL_HOURS = 3
FIRE_WINDOW_HOURS = 24
FIRE_MAX_DISTANCE_KM = 500.0
PM25_PLAUSIBLE = (0.0, 1000.0)
SPLIT_RATIOS = (0.60, 0.20, 0.20)
SPLIT_NAMES = ("train", "validation", "test")

# All pollutants the pipeline can supervise on (the DB stores all six; only the
# configured target anchors the panel and receives lag/rolling history).
ALL_POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

# Plausible target ranges per pollutant, used to reject implausible readings.
PLAUSIBLE_RANGES = {
    "pm25": (0.0, 1000.0),
    "pm10": (0.0, 1000.0),
    "o3": (0.0, 500.0),
    "no2": (0.0, 500.0),
    "so2": (0.0, 500.0),
    "co": (0.0, 20.0),
}

# Numeric columns considered for IQR-outlier flagging (extended beyond the
# explicitly required feature set to cover every engineered feature).
IQR_FLAG_COLS = [
    "temperature", "humidity", "pressure_msl", "surface_pressure",
    "wind_speed", "wind_direction", "pbl_height", "ventilation_coefficient",
    "inversion_strength", "strongest_layer_gradient",
    "fire_count", "fire_impact_score", "nearest_fire_distance",
    "wind_aligned_fire_count", "wind_alignment_pct", "transport_time_hours",
    "transport_risk", "stubble_impact_score",
    *ALL_POLLUTANTS,
]

# Documented exogenous features (not part of the target history).
EXOGENOUS_FEATURES = [
    "temperature", "humidity", "pressure_msl", "surface_pressure",
    "wind_speed", "wind_direction", "wind_dir_sin", "wind_dir_cos",
    "pbl_height", "ventilation_coefficient",
    "inversion_detected", "inversion_strength", "inversion_category",
    "inversion_source", "strongest_layer_gradient",
    "fire_count", "fire_impact_score", "nearest_fire_distance",
    "wind_aligned_fire_count", "wind_alignment_pct", "transport_time_hours",
    "transport_risk", "transport_risk_level", "stubble_impact_score",
]


#: Every naive timestamp the application persists is **UTC**. ``cpcb_service.
#: upsert_ncr_data`` parses CPCB ``last_update`` as IST but converts to UTC before
#: stripping the tzinfo, ``api.import_`` normalises to UTC, and the weather
#: importers follow suit. The serving code (``api.summary``, ``forecast_service``,
#: ``refresh_service``) reads a naive stored value as UTC wall-clock, so a naive
#: value may be assumed to be UTC.
#:
#: This constant is retained for the one case that is genuinely IST-native:
#: legacy naive-IST rows that have not yet been converted. Pass it explicitly as
#: ``assume_tz=IST`` rather than relying on it as a default.
IST = ZoneInfo("Asia/Kolkata")


def coerce_utc_naive(series: pd.Series, *, assume_tz: str | ZoneInfo = UTC) -> pd.Series:
    """Normalise a datetime series to naive-UTC.

    Aware inputs are converted from their own offset. **Naive** inputs are first
    interpreted in ``assume_tz`` before conversion.

    Getting the naive case wrong is not cosmetic, in either direction.

    ``pd.to_datetime(series, utc=True)`` silently treats a naive value as already
    being UTC. That was once the bug here: a reading recorded at 11:30 IST stayed
    at 11:30 instead of becoming 06:00 UTC, so every FIRMS window (genuine UTC)
    joined 5 h 30 m out of phase against the pollution and weather rows - long
    enough to drop real detections and to attach the wrong ones.

    The naive input convention has since been unified the *other* way: the
    application now stores naive **UTC** end to end (see ``IST`` above), so a
    naive value is interpreted as UTC by default. Defaulting to ``IST`` instead
    would re-introduce the same 5 h 30 m phase error against the serving code,
    which reads naive stored values as UTC: the fire window is queried on the raw
    UTC axis while the panel was shifted back 5 h 30 m, so no fire ever aligned.

    ``assume_tz`` remains a parameter for legacy naive-IST rows that have not been
    converted yet. Pass ``IST`` for those; do not change the default.
    """
    dt = pd.to_datetime(series, errors="coerce")
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        # Aware: pandas already knows the offset, so convert to UTC directly
        # rather than re-interpreting the wall-clock.
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)
    # Naive (or all-NaT, which is object dtype): attach the source timezone
    # first, otherwise the wall-clock would be taken as UTC.
    if not isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_localize(assume_tz)
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)


def _epoch_ns(series: pd.Series) -> np.ndarray:
    """Convert a datetime series to epoch-nanoseconds (int64).

    Unit-explicit conversion: ``Series.astype("int64")`` is timezone/resolution
    dependent (pandas may return microseconds), which silently breaks nanosecond
    windows. ``datetime64[ns]`` is always nanoseconds.
    """
    return series.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _dedupe_per_hour(df: pd.DataFrame, hour_col: str) -> pd.DataFrame:
    """De-duplicate (station, hour) pairs keeping the latest observation."""
    n_before = len(df)
    out = df.sort_values([hour_col]).drop_duplicates(
        subset=["station", hour_col], keep="last"
    ).reset_index(drop=True)
    return out, n_before - len(out)


def align_observations(
    poll_df: pd.DataFrame,
    wx_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    max_ffill_hours: int = MAX_FFILL_HOURS,
    drop_target_null: bool = True,
    target: str = TARGET,
    plausible: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hour-align pollution (target anchor) with weather (exogenous).

    Returns the aligned panel and a diagnostics dict with all handling counts.
    """
    report: dict[str, Any] = {}

    if target not in poll_df.columns:
        raise ValueError(
            f"Target {target!r} not present in pollution frame (columns: {list(poll_df.columns)})"
        )
    plausible = plausible or PM25_PLAUSIBLE

    poll = poll_df.copy()
    wx = wx_df.copy()

    # Station observations are stored naive-UTC, and fires naive-UTC, so both
    # coercions use the UTC default. They must share one axis: the fire window is
    # derived from the raw stored timestamps, so shifting only these would put the
    # panel 5 h 30 m out of phase with every FIRMS join.
    poll["timestamp"] = coerce_utc_naive(poll["timestamp"])
    wx["timestamp"] = coerce_utc_naive(wx["timestamp"])
    poll["hour"] = poll["timestamp"].dt.floor("h")
    wx["hour"] = wx["timestamp"].dt.floor("h")

    # Duplicate timestamps
    poll, d_poll = _dedupe_per_hour(poll, "hour")
    wx, d_wx = _dedupe_per_hour(wx, "hour")
    report["duplicate_hour_rows_removed"] = {"pollution": d_poll, "weather": d_wx}

    # Attach station coordinates once
    coords = stations_df[["station", "latitude", "longitude"]].drop_duplicates("station")
    poll = poll.merge(coords, on="station", how="left")

    # Time-aligned left join: pollution hour anchors, weather joins left.
    wx_feats = [c for c in ["temperature", "humidity", "pressure", "pressure_msl",
                            "surface_pressure", "wind_speed", "wind_direction",
                            "precipitation", "cloud_cover", "pbl_height",
                            "temperature_1000hPa", "temperature_925hPa",
                            "temperature_850hPa", "temperature_700hPa"]
                if c in wx.columns]
    wx_join = wx[["station", "hour"] + wx_feats].drop_duplicates(subset=["station", "hour"])
    df = poll.merge(wx_join, on=["station", "hour"], how="left")

    # Missing data: forward-fill weather exogenous features (<=max_ffill_hours)
    ff_cols = [c for c in wx_feats if c not in ("temperature_1000hPa", "temperature_925hPa",
                                                "temperature_850hPa", "temperature_700hPa")]
    ffilled_counts: dict[str, int] = {}
    for _station, group in df.groupby("station", sort=False):
        idx = group.index
        for col in ff_cols:
            filled = group[col].ffill(limit=max_ffill_hours)
            ffilled_counts[col] = ffilled_counts.get(col, 0) + int(filled.notna().sum() - group[col].notna().sum())
            df.loc[idx, col] = filled.values
    report["ffilled_cells"] = ffilled_counts
    report["weather_columns_ffilled"] = ff_cols
    report["ffill_limit_hours"] = max_ffill_hours

    # Target plausibility / outliers
    before_plaus = int(df[target].notna().sum())
    df.loc[(df[target] <= plausible[0]) | (df[target] > plausible[1]), target] = np.nan
    report["target_implausible_removed"] = before_plaus - int(df[target].notna().sum())
    report["target_plausible_range"] = list(plausible)

    if drop_target_null:
        n_null = int(df[target].isna().sum())
        df = df[df[target].notna()].reset_index(drop=True)
        report["rows_dropped_no_target"] = n_null

    df = df.sort_values(["station", "hour"]).reset_index(drop=True)
    report["rows_after_alignment"] = len(df)
    return df, report


def _temps_from_row(row: pd.Series) -> dict[float, float]:
    temps = {}
    for level in (1000, 925, 850, 700):
        v = row.get(f"temperature_{level}hPa")
        if pd.notna(v):
            temps[level] = float(v)
    return temps


def add_atmosphere_and_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive inversion, ventilation and cyclical time features per row.

    Inversion reuses ``ml.features.atmospheric_profile.combine_inversion``:
    lapse-rate from the stored pressure-level temperatures when 2+ levels exist,
    PBL-height proxy otherwise (source column records which basis was used).
    """
    from ..features.atmospheric_profile import combine_inversion

    out = df.copy()
    dt = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    vent = out["wind_speed"].to_numpy(dtype=float) * out["pbl_height"].to_numpy(dtype=float)
    out["ventilation_coefficient"] = np.where(np.isnan(vent), np.nan, vent)
    out["ventilation_norm"] = (out["ventilation_coefficient"] / 6000.0).clip(0.0, 1.0)

    inv_detected, inv_strength, inv_cat, inv_src, inv_grad, inv_profile = [], [], [], [], [], []
    for _, row in out.iterrows():
        pbl = row.get("pbl_height")
        pbl = None if pd.isna(pbl) else float(pbl)
        diag = combine_inversion(pbl, _temps_from_row(row))
        inv_detected.append(int(bool(diag["inversion_detected"])))
        inv_strength.append(float(diag["inversion_strength"]))
        inv_cat.append(str(diag["inversion_category"]))
        inv_src.append(str(diag["inversion_source"]))
        g = diag.get("strongest_layer_gradient")
        inv_grad.append(None if g is None or (isinstance(g, float) and np.isnan(g)) else float(g))
        inv_profile.append(int(bool(diag.get("profile_available", False))))
    out["inversion_detected"] = inv_detected
    out["inversion_strength"] = inv_strength
    out["inversion_category"] = inv_cat
    out["inversion_source"] = inv_src
    out["strongest_layer_gradient"] = inv_grad
    out["inversion_profile_available"] = inv_profile

    # Cyclical wind direction decomposition (0-360 deg, standard trig).
    wdir = out["wind_direction"].fillna(0.0).to_numpy(dtype=float)
    out["wind_dir_sin"] = np.sin(np.deg2rad(wdir))
    out["wind_dir_cos"] = np.cos(np.deg2rad(wdir))

    # Temporal features (cyclic encodings avoid discontinuity at 23->0h).
    # NOTE: the floored datetime bucket stays in column ``hour`` (used by the
    # alignment/split logic); the integer hour-of-day gets its own name.
    out["hour_of_day"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["day_of_year"] = dt.dt.dayofyear
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def _haversine_arrays(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance (km); mirrors the scalar helper in
    ``fire_impact`` so windowed fire features match it exactly."""
    from ..features.fire_impact import EARTH_RADIUS_KM

    lat1_r = math.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - float(lat1))
    dlon = np.radians(lon2 - float(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def _bearing_array(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised bearing (deg, 0-360) from (lat1,lon1) to each (lat2,lon2);
    mirrors ``fire_impact._angle_between``."""
    dlon = np.radians(lon2 - float(lon1))
    lat1_r = math.radians(lat1)
    lat2_r = np.radians(lat2)
    y = np.sin(dlon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def add_fire_features(
    df: pd.DataFrame,
    fires_df: pd.DataFrame,
    window_hours: int = FIRE_WINDOW_HOURS,
    max_distance_km: float = FIRE_MAX_DISTANCE_KM,
    fire_tz: str | ZoneInfo = UTC,
) -> pd.DataFrame:
    """Attach fire-transport features per row using fires in the trailing window.

    Fire window is [row_hour - window_hours, row_hour] — fires acquired after
    the row's hour are never used (no future leakage). Logically identical to
    ``ml.features.fire_impact.compute_fire_impact`` (FRP-weighted, wind-aligned,
    distance-weighted) but vectorised for large panels:

    * geometry (distance/bearing) is computed once per **station** against all
      stored fires (a handful of stations × ~100k fires), not per sample;
    * row windows are obtained with vectorised ``searchsorted`` per station;
    * per-sample results accumulate into arrays and are written back as whole
      columns (no row-by-row ``.loc`` writes).
    """
    from ..features.fire_impact import _normalize_impact

    out = df.copy()
    _init = {
        "fire_count": 0, "fire_impact_score": 0.0,
        "nearest_fire_distance": float(max_distance_km + 1.0),
        "wind_aligned_fire_count": 0, "wind_alignment_pct": 0.0,
        "transport_time_hours": None, "transport_risk": 0.0,
        "transport_risk_level": "none", "stubble_impact_score": 0.0,
    }
    for col in _init:
        out[col] = _init[col]

    if fires_df is None or fires_df.empty:
        return out

    fires = fires_df.copy()
    # FIRMS acquisition times are UTC by definition (the API documents
    # ``acq_time`` as HHMM UTC), and rows read back from ``fire_readings`` store
    # ``acq_date`` as naive-UTC, so UTC is correct for both and matches the
    # default. Station observations are naive-UTC too, so this now puts fires and
    # station rows on the same axis rather than 5 h 30 m apart.
    fires["acq_timestamp"] = coerce_utc_naive(fires["acq_date"], assume_tz=fire_tz)
    fires = fires.dropna(subset=["acq_timestamp", "lat", "lon"]).reset_index(drop=True)
    if fires.empty:
        return out

    f_lat = fires["lat"].to_numpy(dtype=float)
    f_lon = fires["lon"].to_numpy(dtype=float)
    f_frp = fires["frp"].to_numpy(dtype=float)
    f_frp = np.where(np.isnan(f_frp) | (f_frp <= 0), 1.0, f_frp)

    n_rows = len(out)
    fire_count = np.zeros(n_rows, dtype=np.int64)
    fire_impact = np.zeros(n_rows, dtype=np.float64)
    nearest_dist = np.full(n_rows, float(max_distance_km + 1.0), dtype=np.float64)
    aligned_count = np.zeros(n_rows, dtype=np.int64)
    alignment_pct = np.zeros(n_rows, dtype=np.float64)
    transport_time = np.full(n_rows, np.nan, dtype=np.float64)
    transport_risk = np.zeros(n_rows, dtype=np.float64)
    risk_level = np.full(n_rows, "none", dtype=object)
    stubble_impact = np.zeros(n_rows, dtype=np.float64)

    win_ns = int(window_hours * 3.6e12)

    row_ns = _epoch_ns(out["hour"])
    r_lat = out["latitude"].to_numpy(dtype=float)
    r_lon = out["longitude"].to_numpy(dtype=float)
    r_wind_dir = out["wind_direction"].to_numpy(dtype=float)
    r_wind_spd = out["wind_speed"].to_numpy(dtype=float)

    positions = out.groupby("station", sort=False).indices
    for _station, pos in positions.items():
        lat = float(np.unique(r_lat[pos])[0]) if len(pos) else r_lat[0]
        lon = float(np.unique(r_lon[pos])[0]) if len(pos) else r_lon[0]

        # Per-station geometry over the full fire archive, then distance filter.
        dist = _haversine_arrays(lat, lon, f_lat, f_lon)
        bearing = _bearing_array(lat, lon, f_lat, f_lon)
        keep = dist <= max_distance_km
        if not keep.any():
            continue

        f_ts = fires["acq_timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        keep_idx = np.flatnonzero(keep)
        order = keep_idx[np.argsort(f_ts[keep_idx], kind="stable")]
        f_ts_s = f_ts[order]
        f_d_s = dist[order]
        f_b_s = bearing[order]
        f_f_s = f_frp[order]

        rts = row_ns[pos]
        lo = np.searchsorted(f_ts_s, rts - win_ns, side="left")
        hi = np.searchsorted(f_ts_s, rts, side="right")

        for k in range(len(pos)):
            i = int(pos[k])
            lo_k, hi_k = int(lo[k]), int(hi[k])
            if hi_k <= lo_k:
                continue
            w_d = f_d_s[lo_k:hi_k]
            w_b = f_b_s[lo_k:hi_k]
            w_f = f_f_s[lo_k:hi_k]
            count = hi_k - lo_k
            min_dist = float(w_d.min())

            wind_dir = r_wind_dir[i]
            if not np.isfinite(wind_dir):
                wind_dir = 0.0
            angle_diff = np.abs(w_b - wind_dir)
            angle_diff = np.where(angle_diff > 180.0, 360.0 - angle_diff, angle_diff)
            alignment = np.cos(np.radians(angle_diff))

            weights = (w_f * np.maximum(alignment, 0.0)) / (w_d + 1.0)
            impact_score = _normalize_impact(float(weights.sum()))
            al = int(np.sum(angle_diff <= 90.0))
            pct = al / count * 100.0

            wind_spd = r_wind_spd[i]
            t_time = None
            if np.isfinite(wind_spd) and wind_spd > 0.5:
                t_time = min_dist / (wind_spd * 3.6)

            proximity = 1.0 - min(1.0, min_dist / max_distance_km)
            alignment_term = pct / 100.0
            time_term = 0.0 if t_time is None else 1.0 - min(1.0, t_time / 24.0)
            risk = 0.4 * impact_score + 0.3 * proximity + 0.2 * alignment_term + 0.1 * time_term
            risk = min(1.0, max(0.0, risk))
            level = ("severe" if risk >= 0.7
                     else "high" if risk >= 0.4
                     else "moderate" if risk >= 0.15 else "low")
            stubble = min(1.0, max(0.0, impact_score * (0.5 + 0.5 * alignment_term)))

            fire_count[i] = count
            fire_impact[i] = impact_score
            nearest_dist[i] = min_dist
            aligned_count[i] = al
            alignment_pct[i] = pct
            transport_time[i] = t_time if t_time is not None else np.nan
            transport_risk[i] = risk
            risk_level[i] = level
            stubble_impact[i] = stubble

    out["fire_count"] = fire_count
    out["fire_impact_score"] = np.round(fire_impact, 4)
    out["nearest_fire_distance"] = np.round(nearest_dist, 2)
    out["wind_aligned_fire_count"] = aligned_count
    out["wind_alignment_pct"] = np.round(alignment_pct, 1)
    out["transport_time_hours"] = pd.Series(
        np.where(np.isnan(transport_time), None, np.round(transport_time, 2)), index=out.index
    )
    out["transport_risk"] = np.round(transport_risk, 3)
    out["transport_risk_level"] = risk_level
    out["stubble_impact_score"] = np.round(stubble_impact, 3)

    return out


def add_target_lags_and_rolling(
    df: pd.DataFrame,
    target: str = TARGET,
    lags: list[int] = LAG_HOURS,
    rolling_windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Causal lag + rolling features per station. Never uses the current target.

    - ``{target}_lag{h}``        = target shifted h hours.
    - ``{target}_roll_mean_{w}h`` = mean of target over the *preceding* w hours
                                    (shift(1) then rolling).
    - ``{target}_roll_std_{w}h``  = same but standard deviation.
    """
    out = df.copy().sort_values(["station", "hour"]).reset_index(drop=True)
    for lag in lags:
        out[f"{target}_lag{lag}"] = out.groupby("station")[target].shift(lag)
    shifted1 = out.groupby("station")[target].shift(1)
    for w in rolling_windows:
        out[f"{target}_roll_mean_{w}h"] = shifted1.groupby(out["station"]).transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
        out[f"{target}_roll_std_{w}h"] = shifted1.groupby(out["station"]).transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()
        )
    return out


def flag_outliers(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Flag extreme outliers (outside Q1-3*IQR .. Q3+3*IQR) per numeric column.

    Values are kept in place; the flag column records ``True`` where the value
    is an extreme outlier. Missing (NaN) values are never flagged (handled
    separately by the missing-data policy).
    """
    out = df.copy()
    cols = cols or [c for c in IQR_FLAG_COLS if c in out.columns]
    counts: dict[str, int] = {}
    for col in cols:
        flag_col = f"outlier_{col}"
        out[flag_col] = False
        if col not in out.columns or out[col].notna().sum() == 0:
            continue
        try:
            vals = out[col].to_numpy(dtype=float)
        except (TypeError, ValueError):
            continue
        q1, q3 = np.nanpercentile(vals, [25, 75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        flag = ((vals < lo) | (vals > hi)) & np.isfinite(vals)
        out[flag_col] = flag
        counts[col] = int(flag.sum())
    return out, counts


def chronological_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> pd.DataFrame:
    """Chronological, contiguous, no-shuffle train/val/test split.

    Rows are ordered by (timestamp, station) and cut at the cumulative
    fractions. Splits are non-overlapping and in strict time order.
    """
    out = df.copy()
    out = out.sort_values(["hour", "station"]).reset_index(drop=True)
    n = len(out)
    r_train, r_val, r_test = ratios
    assert abs((r_train + r_val + r_test) - 1.0) < 1e-6
    cuts = [int(round(n * r_train)), int(round(n * (r_train + r_val)))]
    labels = np.concatenate([
        np.repeat(SPLIT_NAMES[0], cuts[0]),
        np.repeat(SPLIT_NAMES[1], cuts[1] - cuts[0]),
        np.repeat(SPLIT_NAMES[2], n - cuts[1]),
    ])
    out["split"] = labels
    return out


def build_summary(
    df: pd.DataFrame,
    alignment_report: dict[str, Any] | None = None,
    outlier_counts: dict[str, int] | None = None,
    target: str = TARGET,
) -> dict[str, Any]:
    """Generate the dataset summary report (rows/ranges/missing/target/splits)."""
    summary: dict[str, Any] = {
        "target": target,
        "rows": len(df),
        "time_range": {
            "min": str(df["timestamp"].min()) if len(df) else None,
            "max": str(df["timestamp"].max()) if len(df) else None,
        },
        "stations": sorted(df["station"].unique().tolist()) if len(df) else [],
        "chronological_ordering": True,
        "shuffled": False,
        "split_ratios": {"train": SPLIT_RATIOS[0], "validation": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "handling": alignment_report or {},
        "features": {
            "exogenous": EXOGENOUS_FEATURES,
            "target_history": [f"{target}_lag{h}" for h in LAG_HOURS]
                              + [f"{target}_roll_{kind}_{w}h" for kind in ("mean", "std") for w in ROLLING_WINDOWS],
            "temporal": ["hour_of_day", "day_of_week", "month", "day_of_year", "is_weekend",
                         "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
                         "month_sin", "month_cos", "ventilation_norm", "inversion_profile_available"],
        },
    }

    if len(df):
        missing = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                missing[col] = {"count": n, "pct": round(100.0 * n / len(df), 2)}
        summary["missing_values"] = missing

        t = df[target].dropna()
        summary["target_statistics"] = {
            "count": int(t.count()),
            "mean": round(float(t.mean()), 2),
            "std": round(float(t.std()), 2),
            "min": round(float(t.min()), 2),
            "q25": round(float(t.quantile(0.25)), 2),
            "median": round(float(t.median()), 2),
            "q75": round(float(t.quantile(0.75)), 2),
            "max": round(float(t.max()), 2),
        }

        splits = {}
        for name in SPLIT_NAMES:
            sub = df[df["split"] == name]
            splits[name] = {
                "rows": int(len(sub)),
                "time_range": {
                    "min": str(sub["timestamp"].min()) if len(sub) else None,
                    "max": str(sub["timestamp"].max()) if len(sub) else None,
                },
                "stations": sorted(sub["station"].unique().tolist()),
            }
        summary["splits"] = splits
        summary["outlier_flags"] = outlier_counts or {}
    return summary


def build_training_dataset_from_dataframes(
    poll_df: pd.DataFrame,
    wx_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    fires_df: pd.DataFrame,
    target: str = TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end pure pipeline from DataFrames (deterministic, DB-agnostic)."""
    plausible = PLAUSIBLE_RANGES.get(target, PM25_PLAUSIBLE)
    df, alignment = align_observations(poll_df, wx_df, stations_df, target=target, plausible=plausible)
    df = add_atmosphere_and_temporal_features(df)
    df = add_fire_features(df, fires_df)
    df = add_target_lags_and_rolling(df, target=target)
    df, outlier_counts = flag_outliers(df)
    df = chronological_split(df)
    summary = build_summary(df, alignment_report=alignment, outlier_counts=outlier_counts, target=target)
    return df, summary


def _df_from_model(db, model, cols: list[str]) -> pd.DataFrame:
    rows = db.query(model).all()
    data = {c: [getattr(r, c) for r in rows] for c in cols}
    return pd.DataFrame(data)


def build_training_dataset_from_db(
    db=None,
    *,
    station_names: list[str] | None = None,
    fire_window_hours: int = FIRE_WINDOW_HOURS,
    target: str = TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the configured backend database and build the training dataset.

    ``db`` may be injected (tests) or None to use the app SessionLocal.
    ``target`` selects the pollution column to anchor the panel on.
    """
    from backend.app.models.db_models import FireReading, PollutionReading, Station, WeatherReading

    if target not in ALL_POLLUTANTS:
        raise ValueError(
            f"target must be one of {ALL_POLLUTANTS}; got {target!r}"
        )

    if db is None:
        from backend.app.database import SessionLocal
        session = SessionLocal()
    else:
        session = db
    try:
        stations = _df_from_model(session, Station, ["id", "name", "latitude", "longitude"])
        stations["station"] = stations["name"]
        if station_names:
            stations = stations[stations["station"].isin(station_names)]

        poll = _df_from_model(session, PollutionReading,
                              ["station_id", "timestamp", *ALL_POLLUTANTS])
        wx = _df_from_model(session, WeatherReading,
                            ["station_id", "timestamp", "temperature", "humidity",
                             "pressure", "pressure_msl", "surface_pressure",
                             "wind_speed", "wind_direction", "precipitation",
                             "cloud_cover", "pbl_height",
                             "temperature_1000hPa", "temperature_925hPa",
                             "temperature_850hPa", "temperature_700hPa"])
        fires = _df_from_model(session, FireReading,
                               ["latitude", "longitude", "acq_date", "frp"])
        fires.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

        # Map station_id -> station name
        id_to_name = dict(zip(stations["id"], stations["station"], strict=True))
        poll["station"] = poll["station_id"].map(id_to_name)
        wx["station"] = wx["station_id"].map(id_to_name)
        poll = poll.dropna(subset=["station"])
        wx = wx.dropna(subset=["station"])

        return build_training_dataset_from_dataframes(
            poll, wx, stations[["station", "latitude", "longitude"]], fires, target=target
        )
    finally:
        if db is None:
            session.close()
