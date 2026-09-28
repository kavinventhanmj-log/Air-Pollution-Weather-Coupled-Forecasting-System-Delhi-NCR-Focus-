"""Temperature inversion detection.

Two complementary methods are provided:

* **Vertical lapse-rate inversion** (:mod:`ml.features.atmospheric_profile`) —
  the scientifically defensible method required by SIH26082: compute the
  vertical temperature gradient from standard pressure-level temperatures
  (1000/925/850/700 hPa) and detect layers where temperature *increases* with
  height. This is the authoritative method when vertical data is present.

* **PBL-height proxy** (legacy, kept backward compatible) — infers inversion
  from planetary boundary layer height (lower PBL => stronger trapping). This
  remains the documented fallback when no vertical profile is available.

Both are described in ``docs/SIH_GAP_AUDIT.md`` §E and used consistently at
training time and in the live API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_inversion(pbl_height: float, hour: int = 12) -> tuple:
    """Detect inversion conditions from PBL height and hour of day.

    Args:
        pbl_height: Planetary boundary layer height in meters.
        hour: Hour of day (0-23) for diurnal correction.

    Returns:
        Tuple of (is_inversion: bool, category: str, strength: float).
        - is_inversion: True if any inversion detected (PBL < 500m).
        - category: One of "none", "weak", "moderate", "strong".
        - strength: Normalized 0-1, where 1 = strongest.
    """
    if pbl_height is None or (isinstance(pbl_height, float) and np.isnan(pbl_height)):
        return False, "unknown", 0.0

    if pbl_height < 150:
        category = "strong"
        is_inversion = True
    elif pbl_height < 300:
        category = "moderate"
        is_inversion = True
    elif pbl_height < 500:
        category = "weak"
        is_inversion = True
    else:
        category = "none"
        is_inversion = False

    strength = max(0.0, min(1.0, (500.0 - pbl_height) / 500.0))

    diurnal_factor = _diurnal_inversion_factor(hour)
    strength = min(1.0, strength * diurnal_factor)

    return is_inversion, category, float(strength)


def _diurnal_inversion_factor(hour: int) -> float:
    """Compute diurnal multiplier for inversion strength.

    Inversions are naturally stronger during nighttime and early morning
    (stable boundary layer) and weaker during daytime (convective mixing).

    Returns a multiplier in [0.7, 1.3]:
      - Night/early morning (22-8):  factor ~1.1-1.3 (amplify)
      - Daytime (10-16):             factor ~0.7-0.9 (suppress)
      - Evening (17-21):             factor ~0.9-1.1 (transition)
    """
    if 0 <= hour <= 5:
        return 1.3
    elif 6 <= hour <= 8:
        return 1.2
    elif 9 <= hour <= 10:
        return 1.05
    elif 11 <= hour <= 15:
        return 0.75
    elif 16 <= hour <= 17:
        return 0.9
    elif 18 <= hour <= 20:
        return 1.05
    elif 21 <= hour <= 23:
        return 1.2
    return 1.0


def add_inversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add inversion detection features to the dataframe.

    Adds columns:
      - inversion_detected: Binary (1/0) indicating inversion.
      - inversion_strength: Normalized strength 0-1.
      - inversion_category: Categorical label (none/weak/moderate/strong).
    """
    df = df.copy()
    if "pbl_height" not in df.columns:
        df["inversion_detected"] = 0
        df["inversion_strength"] = 0.0
        df["inversion_category"] = "none"
        return df

    hour_col = "hour" if "hour" in df.columns else None
    if hour_col is None:
        if "timestamp" in df.columns:
            dt = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            hour_col = "_tmp_hour"
            df[hour_col] = dt.dt.hour

    results = df.apply(
        lambda row: detect_inversion(row["pbl_height"], row[hour_col] if hour_col else 12),
        axis=1,
    )

    df["inversion_detected"] = results.apply(lambda r: int(r[0]))
    df["inversion_category"] = results.apply(lambda r: r[1])
    df["inversion_strength"] = results.apply(lambda r: r[2])

    if hour_col == "_tmp_hour":
        df.drop(columns=[hour_col], inplace=True)

    return df


# --------------------------------------------------------------------------
# Lapse-rate + PBL-classification features (SIH26082 §4)
# --------------------------------------------------------------------------

#: Column names for the vertical temperature profile. When present (as e.g.
#: ``temperature_925hPa``, ``temperature_850hPa``), lapse-rate inversion is used.
PRESSURE_LEVEL_COL_PREFIX = "temperature_{}hPa"
PRESSURE_LEVELS = [1000, 925, 850, 700]

#: Column names for geopotential heights (used for inversion_base/top reporting).
GEOPOTENTIAL_COL_PREFIX = "geopotential_height_{}hPa"


def _extract_temp_by_level(df_row) -> dict:
    """Pull {pressure_hPa: temperature} from a DataFrame row when available."""
    temps = {}
    for p in PRESSURE_LEVELS:
        col = PRESSURE_LEVEL_COL_PREFIX.format(p)
        if col in df_row.index:
            val = df_row.get(col)
            if pd.notna(val):
                try:
                    temps[p] = float(val)
                except (TypeError, ValueError):
                    continue
    return temps


def _extract_geopotential_by_level(df_row) -> dict:
    """Pull {pressure_hPa: geopotential height} from a DataFrame row."""
    heights = {}
    for p in PRESSURE_LEVELS:
        col = GEOPOTENTIAL_COL_PREFIX.format(p)
        if col in df_row.index:
            val = df_row.get(col)
            if pd.notna(val):
                try:
                    heights[p] = float(val)
                except (TypeError, ValueError):
                    continue
    return heights


# --------------------------------------------------------------------------
# Episode statistics: duration and persistence over time (SIH26082 §4)
# --------------------------------------------------------------------------

#: Longest gap between two samples that still counts as "continuous" (hours).
#: Hourly sampling makes this generous enough to tolerate one dropped reading.
EPISODE_GAP_TOLERANCE_H = 3.0


def summarise_inversion_episode(
    timestamps,
    detected,
    *,
    window_h: int = 24,
    gap_tolerance_h: float = EPISODE_GAP_TOLERANCE_H,
) -> dict:
    """Duration and persistence of the current inversion episode.

    The audit listed inversion *duration* and *persistence* as unimplemented
    "requires a multi-day vertical archive". That is true of the depth of the
    archive, not of the computation: given whatever hourly history exists, an
    episode's duration is the length of the current unbroken run of inversion
    hours and its persistence is the fraction of the window it covers.

    Both are reported together with the window they were measured over, so a
    6-hour archive can never be mistaken for a 6-hour episode. Nothing is
    extrapolated beyond the samples supplied.

    Args:
        timestamps: time-ordered timestamps (oldest first). Anything else and
            the series is sorted first.
        detected: per-sample inversion flag, aligned with ``timestamps``.
        window_h: persistence window in hours (default 24).
        gap_tolerance_h: samples further apart than this break a run.

    Returns:
        dict with ``current_duration_h``, ``persistence_fraction``,
        ``persistence_window_h``, ``measured_window_h``, ``samples``,
        ``complete``, ``onset``, and ``sufficient_history``.
    """
    empty = {
        "current_duration_h": 0.0,
        "persistence_fraction": None,
        "persistence_window_h": window_h,
        "measured_window_h": 0.0,
        "samples": 0,
        "complete": False,
        "onset": None,
        "sufficient_history": False,
    }
    ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True, errors="coerce")
    flags = pd.Series([bool(d) for d in detected], dtype="boolean")
    valid = ts.notna().to_numpy()
    if len(ts) == 0 or not valid.any():
        return empty
    ts = ts[valid].reset_index(drop=True)
    flags = flags[valid].reset_index(drop=True)
    order = ts.sort_values(kind="stable").index
    ts, flags = ts[order].reset_index(drop=True), flags[order].reset_index(drop=True)
    if ts.empty:
        return empty

    # Measured window: newest - oldest sample, floored at one interval so a
    # single sample reports 1 h rather than 0 h of history.
    span_h = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0)
    measured_h = max(span_h, 1.0) if len(ts) > 1 else 1.0

    # Current episode: the unbroken run of inversion samples ending at the
    # newest reading. A sampling gap breaks the run -- and the gap itself is
    # NOT added to the duration, because nothing was observed during it and
    # claiming otherwise would invent persistence.
    first_idx = len(ts) - 1
    for i in range(len(ts) - 1, 0, -1):
        if not bool(flags.iloc[i]):
            break
        gap_h = float((ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 3600.0)
        if gap_h > gap_tolerance_h:
            break
        if not bool(flags.iloc[i - 1]):
            break
        first_idx = i - 1

    onset = None
    if bool(flags.iloc[-1]):
        onset = ts.iloc[first_idx]
        span_of_run = float((ts.iloc[-1] - ts.iloc[first_idx]).total_seconds() / 3600.0)
        # Each sample represents one sampling interval, so a run of N hourly
        # samples covers N hours even though its clock-time span is N-1.
        #
        # The typical interval is derived with `diff().dt.total_seconds()` rather
        # than `np.diff(series.astype("int64")) / 3.6e12`. Casting a datetime
        # Series to int64 is resolution-dependent, not nanosecond-guaranteed:
        # pandas 3 resolves datetime64 to microseconds, so the hard-coded
        # nanosecond divisor turned every 1 h gap into 0.001 h, `typical` came
        # out 0.001, and each episode was reported an hour short (19 h of
        # inversion -> 18.001 -> 18.0). `dt.total_seconds()` means the same thing
        # on every supported pandas. `ml/preprocessing/training_dataset.py`
        # documents the same hazard for its own epoch conversion.
        deltas_h = ts.diff().dt.total_seconds().dropna() / 3600.0
        typical = float(deltas_h.median()) if len(deltas_h) else 1.0
        duration_h = span_of_run + (typical if len(ts) > 1 else 0.0)
    else:
        duration_h = 0.0

    # Persistence over the trailing window.
    cutoff = ts.iloc[-1] - pd.Timedelta(hours=window_h)
    in_window = ts >= cutoff
    window_flags = flags[in_window]
    persistence = float(window_flags.mean()) if len(window_flags) else None

    return {
        "current_duration_h": round(float(duration_h), 1),
        "persistence_fraction": None if persistence is None else round(persistence, 3),
        "persistence_window_h": window_h,
        "measured_window_h": round(measured_h, 1),
        "samples": int(len(ts)),
        "complete": bool(len(ts) > 1 and span_h >= window_h),
        "onset": onset,
        # A window shorter than the requested one cannot support a persistence
        # claim, and the UI must say so instead of showing a confident fraction.
        "sufficient_history": bool(span_h >= window_h),
    }


def add_lapse_rate_inversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lapse-rate inversion features from vertical temperature columns.

    Expected optional input columns: ``temperature_{1000,925,850,700}hPa``
    (degC) from the Open-Meteo pressure-level feed. When none are present the
    existing PBL-proxy columns are returned unchanged.

    Adds:
      - inversion_source            : "lapse_rate" | "pbl_proxy"
      - inversion_base_pressure     : hPa (strongest layer base) or None
      - inversion_top_pressure      : hPa (strongest layer top) or None
      - inversion_base_height_m     : m (hypsometric / geopotential) or None
      - inversion_top_height_m      : m or None
      - inversion_thickness_m       : m (layer depth) or None
      - inversion_thickness_hpa     : hPa or None
      - strongest_layer_gradient    : K/100 hPa or None
      - low_pbl_flag                : bool
      - pbl_category                : strong_trapping/.../good_dispersion/unknown
      - dispersion_condition        : TRAPPED/LIMITED/MODERATE/GOOD/UNKNOWN
    """
    from .atmospheric_profile import combine_inversion

    df = df.copy()

    def _row_analysis(row):
        temps = _extract_temp_by_level(row)
        heights = _extract_geopotential_by_level(row)
        pbl = row.get("pbl_height")
        if pd.notna(pbl):
            try:
                pbl = float(pbl)
            except (TypeError, ValueError):
                pbl = None
        return combine_inversion(pbl, temps if temps else None, height_by_level=heights or None)

    analyses = df.apply(_row_analysis, axis=1)

    for key in (
        "inversion_source",
        "inversion_base_pressure",
        "inversion_top_pressure",
        "inversion_base_height_m",
        "inversion_top_height_m",
        "inversion_thickness_m",
        "inversion_thickness_hpa",
        "strongest_layer_gradient",
        "low_pbl_flag",
        "pbl_category",
        "dispersion_condition",
    ):
        df[key] = analyses.apply(lambda a, k=key: a.get(k))

    # Refresh the base inversion columns from the (possibly lapse-rate) analysis.
    df["inversion_detected"] = analyses.apply(lambda a: int(bool(a.get("inversion_detected"))))
    df["inversion_strength"] = analyses.apply(lambda a: float(a.get("inversion_strength", 0.0)))
    df["inversion_category"] = analyses.apply(lambda a: str(a.get("inversion_category", "none")))

    return df
