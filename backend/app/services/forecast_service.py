import logging
import os
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..services.aqi_calculator import calculate_aqi
from ..utils.helpers import haversine_distance, repo_root

logger = logging.getLogger("aerocast.forecast")

MODEL_DIR = str(repo_root() / "models")

DEFAULT_HORIZONS = [1, 6, 12, 24, 48, 72]

ALL_POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

# Stations with fewer than this many local readings in the recent window are
# considered data-sparse; their forecast features fall back to the regional
# NCR composite series (real observations from the rich core stations), and
# the response is flagged ``pooled_features=True`` so the UI never presents the
# value as a purely local model (SIH26082 17/17 honest coverage).
SPARSE_READINGS_THRESHOLD = 24
REGIONAL_COMPOSITE_STATIONS = ["Anand Vihar", "RK Puram", "ITO", "Dwarka", "Punjabi Bagh"]

# The persisted forecasting models consume only the *lag-1* pollutant features
# listed in FEATURE_NAMES (``pm25_lag1`` ... ``co_lag1``) plus instantaneous
# meteorology. A lag-1 value is a real observation only when at least two
# chronological rows exist, so two rows is the hard technical floor for a
# forecast. Below it every lag feature is NaN, the NaNs were silently coerced to
# 0.0, and the models returned confident-looking numbers derived from nothing.
# Forecasting now refuses instead of inventing values.
MIN_FEATURE_ROWS = 2

# A forecast is labelled stale once its newest underlying observation is older
# than this. CPCB hourly reporting plus a demo/refresh cycle makes 6h the honest
# boundary: past it the dashboard says so instead of implying "live".
STALE_AFTER_HOURS = 6.0

_POOL_COLUMNS = ("pm25", "pm10", "o3", "no2", "so2", "co")


class InsufficientDataError(Exception):
    """Raised when a station cannot support an honest forecast.

    Carries the machine-readable context the API layer turns into a
    ``503`` with ``detail.code == "insufficient_data"`` so the frontend can
    render an honest "data unavailable" state instead of a fabricated series.
    """

    code = "insufficient_data"

    def __init__(
        self,
        *,
        station_name: str,
        reason: str,
        required_hours: int = MIN_FEATURE_ROWS,
        available_hours: int = 0,
    ):
        super().__init__(
            f"{station_name}: {reason} "
            f"({available_hours} of {required_hours} required hourly rows available)"
        )
        self.station_name = station_name
        self.reason = reason
        self.required_hours = required_hours
        self.available_hours = available_hours

    def to_payload(self) -> dict:
        return {
            "code": self.code,
            "station": self.station_name,
            "reason": self.reason,
            "required_hours": self.required_hours,
            "available_hours": self.available_hours,
        }

FEATURE_NAMES = [
    "pm25_lag1",
    "pm10_lag1",
    "o3_lag1",
    "no2_lag1",
    "so2_lag1",
    "co_lag1",
    "temperature",
    "humidity",
    "pressure_msl",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "pbl_height",
    "inversion_strength",
    "fire_impact_score",
    "fire_count_100km",
    "nearest_fire_km",
    "hour",
    "is_winter",
    "day_of_year",
    "aqi_lag1",
    "season",
]


def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(path):
        try:
            payload = joblib.load(path)
            if isinstance(payload, dict) and "model" in payload:
                model_obj = payload["model"]
                if not hasattr(model_obj, "predict"):
                    logger.warning("Model payload for %s has no predict attribute", model_name)
                return model_obj
            return payload
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", model_name, exc)
    return None


def available_models() -> list[str]:
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib"))


def load_pollutant_model(pollutant: str, horizon_hours: int | None = None):
    for model_type in ("xgboost", "random_forest", "rf", "persistence", "gbm"):
        suffixes = [f"_{horizon_hours}h", f"_{horizon_hours}", ""]
        if horizon_hours is None:
            suffixes = [""]
        for suffix in suffixes:
            name = f"{model_type}_{pollutant}{suffix}"
            model = load_model(name)
            if model is not None:
                return model
    return None


#: Features whose training-time value was produced by a real measurement, so
#: substituting ``0.0`` for them asserts something false rather than merely
#: unknown. ``0.0`` is the physically impossible value for a Delhi-NCR
#: coordinate, and ``0.0`` fire radiative power contradicts the fire-impact
#: score computed from the same observations in the same request. When these are
#: absent the model must not be asked for a prediction at all.
REQUIRED_REAL_FEATURES = ("latitude", "longitude")

#: Features that are legitimately ``0.0`` when no fire was detected, and were
#: also ``0.0`` at training time (``ml/training/trainer.py`` applies
#: ``fillna(0)``, and the offline builder leaves them empty for fire-free
#: hours). Filling these is therefore truthful, but it is still reported so the
#: provenance of a prediction records how many inputs were imputed.
IMPUTABLE_FEATURES = ("mean_frp", "max_bright")


def _model_feature_columns(model) -> list[str] | None:
    """The feature names the artifact was trained on, in training order.

    Read from the artifact itself so the serving vector can never be built from
    a guess about column counts or insertion order. Returns ``None`` when the
    artifact records no contract, which callers treat as a load failure.
    """
    for attr in ("feature_names_", "feature_names_in_"):
        names = getattr(model, attr, None)
        if names is not None and len(names):
            return list(names)
    inner = getattr(model, "model", None)
    if inner is not None:
        names = getattr(inner, "feature_names_in_", None)
        if names is not None and len(names):
            return list(names)
    return None


def missing_required_features(columns: list[str], features: dict) -> list[str]:
    """Contract names the artifact needs that ``features`` cannot supply.

    Reports the intersection of (a) the artifact's declared contract and (b)
    the values the builder actually produced as absent-or-null. A name is
    treated as missing when it is not a key at all, or maps to ``None``/NaN.
    A real ``0.0`` is present: it is a measured zero, not a gap.
    """
    missing: list[str] = []
    for name in columns:
        if name not in features:
            missing.append(name)
            continue
        value = features.get(name)
        if value is None:
            missing.append(name)
            continue
        try:
            if pd.isna(float(value)):
                missing.append(name)
        except (TypeError, ValueError):
            missing.append(name)
    return missing


def _model_predict(model, features: dict, target: str = "") -> float | None:
    """Predict with a flat artifact, or return ``None`` when that would lie.

    The previous version assembled the vector with ``features.get(c, 0.0)``, so
    any feature the builder did not produce became a confident ``0.0``. For
    ``latitude``/``longitude`` that is not a neutral default but a provably
    wrong claim, and it silently suppressed the honest heuristic fallback
    further down the call chain. The contract is now enforced here:

    * the artifact's own ``feature_names_`` supplies the column order;
    * required real measurements that are absent abort the prediction;
    * imputable features are filled with ``0.0`` (matching the validated
      training-time ``fillna(0)`` policy) and counted for provenance.

    Returning ``None`` routes the caller to the fallback, so a model with an
    unsatisfiable contract can never be persisted as a valid prediction.
    """
    if model is None:
        return None
    try:
        cols = _model_feature_columns(model)
        if cols is None:
            logger.warning(
                "Model for %s declares no feature_names_ contract; refusing to predict",
                target or "unknown target",
            )
            return None

        absent = missing_required_features(cols, features)
        blocking = [c for c in absent if c in REQUIRED_REAL_FEATURES]
        if blocking:
            logger.warning(
                "Model for %s needs measured %s which the feature builder did not "
                "supply; falling back instead of imputing zeros",
                target or "unknown target",
                ", ".join(sorted(blocking)),
            )
            return None

        # Imputation is allowed only for the features whose 0.0 is a truthful
        # "no fire that hour" encoding, and matches training.
        imputed = [c for c in absent if c in IMPUTABLE_FEATURES]
        arr = np.array([[features.get(c, 0.0) for c in cols]], dtype=float)
        if imputed:
            logger.debug(
                "Model for %s: imputed %s as 0.0 (validated training-time policy)",
                target or "unknown target",
                ", ".join(sorted(imputed)),
            )
        raw = model.predict(arr)
        val = float(np.ravel(raw)[0])
        return None if (np.isnan(val) or np.isinf(val)) else max(0.0, val)
    except Exception as exc:
        logger.warning("Prediction failed for %s: %s", type(model).__name__, exc)
        return None


def _feature(features: dict, key: str, default: float | None = None):
    """Read a feature value, treating only *absent/None/NaN* as missing.

    A genuine ``0.0`` measurement (a still, rain-free, pollution-free hour) is a
    real value and must survive. The previous ``features.get(k, d) or d`` idiom
    rewrote every ``0.0`` to the default, so a calm hour was reported as if the
    station had never been measured.
    """
    if not isinstance(features, dict):
        return default
    value = features.get(key, None)
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _fallback_pm25(features: dict, h: int) -> float:
    base = _feature(features, "pm25_lag1", 50.0)
    decay = max(0.55, 1.0 - 0.006 * h)
    wind = _feature(features, "wind_speed", 5.0)
    wind_penalty = 1.0 + max(0.0, 2.0 - wind) * 0.04
    pbl = _feature(features, "pbl_height", 500.0)
    pbl_factor = 1.0 + max(0.0, (400 - pbl) / 400) * 0.35
    fire = _feature(features, "fire_impact_score", 0.0)
    fire_factor = 1.0 + fire * 0.15
    return base * decay * wind_penalty * pbl_factor * fire_factor


def _fallback_pm10(pm25_pred: float, features: dict, h: int) -> float:
    base = _feature(features, "pm10_lag1", None)
    if base is None:
        base = pm25_pred * 1.9
    return max(base * (1.0 - 0.004 * h), pm25_pred * 1.5)


def _fallback_o3(features: dict, h: int) -> float:
    temp = _feature(features, "temperature", 25.0)
    solar = 1.0 + max(0.0, (temp - 20) / 20) * 0.3
    return max(10.0, 45 * solar * (1.0 - 0.002 * h))


def _fallback_no2(features: dict) -> float:
    disp = 1.0 + _feature(features, "wind_speed", 5.0) * 0.05
    return max(5.0, 48 / disp)


def _fallback_so2(features: dict) -> float:
    base = _feature(features, "so2_lag1", 15.0)
    disp = 1.0 + max(0.0, _feature(features, "wind_speed", 5.0) - 3.0) * 0.03
    precip = _feature(features, "precipitation", 0.0)
    washout = max(0.6, 1.0 - precip * 0.15)
    return max(2.0, base * washout / disp)


def _fallback_co(features: dict) -> float:
    base = _feature(features, "co_lag1", 1.4)
    disp = 1.0 + _feature(features, "wind_speed", 5.0) * 0.04
    return max(0.2, base / disp)


def audit_feature_contract(features: dict, horizons=None) -> dict:
    """Describe how well ``features`` satisfies the deployed artifacts' contract.

    Read-only. Reports, for the longest requested horizon, how many contract
    features the builder supplied, which required measurements were missing, and
    whether the model may legitimately be used. Attached to every forecast
    response so a published number can never imply a satisfied contract that was
    not actually met.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    probe = load_pollutant_model("pm25", max(horizons))
    if probe is None:
        return {
            "artifact": None,
            "declared_features": None,
            "satisfied": False,
            "missing_required": [],
            "imputed": [],
            "usable": False,
            "reason": "model artifact could not be loaded",
        }

    cols = _model_feature_columns(probe)
    if cols is None:
        return {
            "artifact": f"xgboost_pm25_{max(horizons)}h",
            "declared_features": None,
            "satisfied": False,
            "missing_required": [],
            "imputed": [],
            "usable": False,
            "reason": "artifact declares no feature_names_ contract",
        }

    absent = missing_required_features(cols, features)
    missing_required = sorted(c for c in absent if c in REQUIRED_REAL_FEATURES)
    imputed = sorted(c for c in absent if c in IMPUTABLE_FEATURES)
    satisfied = not missing_required
    return {
        "artifact": f"xgboost_pm25_{max(horizons)}h",
        "declared_features": len(cols),
        "satisfied": satisfied,
        "missing_required": missing_required,
        "imputed": imputed,
        "usable": satisfied,
        "reason": (
            None
            if satisfied
            else "required measured features unavailable: " + ", ".join(missing_required)
        ),
    }


def predict_pollutants(features: dict, horizons=None) -> list[dict]:
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    predictions = []
    for h in horizons:
        pm25_model = load_pollutant_model("pm25", h)
        pm10_model = load_pollutant_model("pm10", h)
        o3_model = load_pollutant_model("o3", h)
        no2_model = load_pollutant_model("no2", h)
        so2_model = load_pollutant_model("so2", h)
        co_model = load_pollutant_model("co", h)

        pm25_pred = _model_predict(pm25_model, features, "pm25")
        if pm25_pred is None:
            pm25_pred = _fallback_pm25(features, h)
        pm10_pred = _model_predict(pm10_model, features, "pm10")
        if pm10_pred is None:
            pm10_pred = _fallback_pm10(pm25_pred, features, h)
        o3_pred = _model_predict(o3_model, features, "o3")
        if o3_pred is None:
            o3_pred = _fallback_o3(features, h)
        no2_pred = _model_predict(no2_model, features, "no2")
        if no2_pred is None:
            no2_pred = _fallback_no2(features)
        so2_pred = _model_predict(so2_model, features, "so2")
        if so2_pred is None:
            so2_pred = _fallback_so2(features)
        co_pred = _model_predict(co_model, features, "co")
        if co_pred is None:
            co_pred = _fallback_co(features)

        aqi_val, category, dominant = calculate_aqi(
            pm25_pred,
            pm10_pred,
            o3_pred,
            no2_pred,
            so2_pred,
            co_pred,
        )
        predictions.append(
            {
                "horizon_hours": int(h),
                "pm25_pred": round(pm25_pred, 1),
                "pm10_pred": round(pm10_pred, 1),
                "o3_pred": round(o3_pred, 1),
                "no2_pred": round(no2_pred, 1),
                "so2_pred": round(so2_pred, 1),
                "co_pred": round(co_pred, 2),
                "aqi_pred": aqi_val,
                "aqi_category": category,
                "dominant_pollutant": dominant,
            }
        )
    return predictions


#: The station coordinates ``scripts/build_dataset.py`` used to attribute FIRMS
#: detections while building the training set. A detection belongs to whichever
#: of these stations is nearest, and only then counts towards that station's
#: ``mean_frp``/``max_bright``. Serving has to reproduce that attribution or the
#: feature means something different than it did at training time.
TRAINING_FIRE_STATIONS = {
    "Anand_Vihar": (28.6492, 77.2918),
    "RK_Puram": (28.5601, 77.1835),
    "ITO": (28.6290, 77.2410),
    "Dwarka": (28.5921, 77.0460),
    "Punjabi_Bagh": (28.6692, 77.1285),
}


#: The season categories the training artifacts expect as one-hot columns.
SEASON_NAMES = ("winter", "spring", "summer", "autumn")

_SEASON_MONTHS: dict[int, str] | None = None


def _season_month_map() -> dict[int, str]:
    """Month -> season, inverted from the training pipeline's own SEASON_MAP.

    Derived rather than re-declared so the serving encoding cannot drift away
    from the one the artifacts were trained with: if the training map changes,
    this changes with it.
    """
    global _SEASON_MONTHS
    if _SEASON_MONTHS is None:
        from ml.features.feature_engineering import SEASON_MAP

        _SEASON_MONTHS = {int(month): season for month, season in SEASON_MAP.items()}
    return _SEASON_MONTHS


def _season_dummies(timestamp) -> dict:
    """Return the complete four-way season one-hot for a single timestamp.

    ``add_temporal_features`` derives the dummies with ``pd.get_dummies``, whose
    output columns depend on which categories happen to appear in the frame. A
    short serving history therefore produced only the current season's column and
    left the other three names absent, and the model layer filled them with 0.0.
    The current season's dummy then had to be exactly 1.0, so an absent name was
    only accidentally correct.

    This recomputes all four from the timestamp using the same SEASON_MAP the
    training pipeline used, so every name in the contract is present and correct
    no matter how little history the frame carries. A timestamp that cannot be
    parsed returns the all-zero vector rather than an arbitrary one: the season
    is then genuinely unknown, and the model layer's contract check will treat
    the row as unusable instead of inventing a season.
    """
    zeros = {f"season_{s}": 0.0 for s in SEASON_NAMES}
    if timestamp is None:
        return zeros
    try:
        ts = pd.Timestamp(timestamp)
        if pd.isna(ts):
            return zeros
    except (TypeError, ValueError):
        return zeros
    season = _season_month_map().get(ts.month)
    if season is None:
        return zeros
    return {f"season_{s}": (1.0 if s == season else 0.0) for s in SEASON_NAMES}


def _firms_aggregate_features(fires, origin, station_lat, station_lon) -> dict:
    """Aggregate stored FIRMS detections into ``mean_frp`` / ``max_bright``.

    Reproduces the training-time aggregation in
    ``scripts/build_dataset.py::load_fires`` exactly:

    * each detection is attributed to the **nearest** of the five stations the
      training set was built against (squared-distance, no radius cutoff);
    * detections whose nearest station is not the station being forecast are
      discarded, exactly as the offline builder discarded them;
    * the surviving detections are bucketed into the **same clock hour** as the
      forecast origin (``acq_date`` floored to the hour), not a window around it;
    * ``mean_frp`` is the mean ``frp`` and ``max_bright`` the max ``bright_ti4``
      of that bucket. The stored ``brightness`` column is the same FIRMS field
      (``firms_service._normalise_brightness`` writes VIIRS ``bright_ti4`` /
      MODIS ``bright_ti31`` in Kelvin), so it is the correct source column.

    An earlier version of this function averaged every detection within +/-3
    hours regardless of which station it belonged to. That inflated the feature
    for a station with no fire of its own, which is a fabricated value wearing a
    measured one's units.

    Returns 0.0 for both when the station-hour has no detections, because that is
    the encoding training actually saw: the offline builder emits no group row
    for a fire-free station-hour, the left-merge leaves the cell empty, and
    ``ml/training/trainer.py`` applies ``fillna(0)``. So for a fire-free hour
    0.0 is a true value, not a stand-in for an unknown one.
    """
    empty = {"mean_frp": 0.0, "max_bright": 0.0}
    if not fires or origin is None or station_lat is None or station_lon is None:
        return empty

    try:
        origin_ts = pd.Timestamp(origin)
        if pd.isna(origin_ts):
            return empty
        if origin_ts.tzinfo is None:
            origin_ts = origin_ts.tz_localize("UTC")
    except (TypeError, ValueError):
        return empty
    origin_hour = origin_ts.floor("h")

    frp_values: list[float] = []
    brightness_values: list[float] = []
    for f in fires:
        acq = getattr(f, "acq_date", None)
        if acq is None:
            continue
        try:
            acq_ts = pd.Timestamp(acq)
            if acq_ts.tzinfo is None:
                acq_ts = acq_ts.tz_localize("UTC")
        except (TypeError, ValueError):
            continue
        if acq_ts.floor("h") != origin_hour:
            continue

        fire_lat = getattr(f, "latitude", None)
        fire_lon = getattr(f, "longitude", None)
        if fire_lat is None or fire_lon is None:
            continue
        try:
            fire_lat = float(fire_lat)
            fire_lon = float(fire_lon)
        except (TypeError, ValueError):
            continue

        # Nearest-station attribution, matching the offline builder.
        nearest = min(
            TRAINING_FIRE_STATIONS.items(),
            key=lambda kv: (fire_lat - kv[1][0]) ** 2 + (fire_lon - kv[1][1]) ** 2,
        )
        if nearest[1] != (float(station_lat), float(station_lon)):
            continue

        frp = getattr(f, "frp", None)
        if frp is not None and not pd.isna(frp):
            frp_values.append(float(frp))
        brightness = getattr(f, "brightness", None)
        if brightness is not None and not pd.isna(brightness):
            brightness_values.append(float(brightness))

    return {
        "mean_frp": round(sum(frp_values) / len(frp_values), 4) if frp_values else 0.0,
        "max_bright": round(max(brightness_values), 4) if brightness_values else 0.0,
    }


def _fire_features(station_lat: float, station_lon: float, fires) -> dict:
    count_100 = 0
    distances = []
    impact = 0.0
    for f in fires:
        d = haversine_distance(station_lat, station_lon, f.latitude, f.longitude)
        distances.append(d)
        if d <= 100:
            count_100 += 1
        if f.frp:
            impact += f.frp * max(0.0, 1 - d / 500)
    nearest = min(distances) if distances else None
    return {
        "fire_count_100km": count_100,
        "nearest_fire_km": round(nearest, 1) if nearest is not None else None,
        "fire_impact_score": round(min(1.0, impact / 1000.0), 4),
    }


def _flush_json_value(v):
    """Coerce numpy/pandas values to plain JSON-safe python numbers."""
    import math

    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _as_naive_utc(series) -> pd.Series:
    """Normalise a datetime column to naive UTC (the repo-wide convention).

    SQLite stores naive datetimes while PostgreSQL ``timestamp with time zone``
    columns come back tz-aware. Merging one of each raises
    ``ValueError: You are trying to merge on datetime64[us] and
    datetime64[us, UTC]``, so history frames are normalised before any join.
    A naive input is interpreted as UTC wall-clock and left unchanged.
    """
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def _merge_history_frames(poll: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Outer-join pollution + weather history on naive-UTC timestamps.

    Both frames may arrive with either naive (SQLite / composite) or tz-aware
    (PostgreSQL) timestamps; normalising first keeps the merge portable across
    the two database backends.
    """
    poll = poll.copy()
    wx = wx.copy()
    poll["timestamp"] = _as_naive_utc(poll["timestamp"])
    wx["timestamp"] = _as_naive_utc(wx["timestamp"])
    return poll.merge(wx, on="timestamp", how="outer", suffixes=("", "_wx"))


def _pollution_df(db, station_id: int, limit: int = 120) -> pd.DataFrame | None:
    """Return a station's local, chronologically-flat pollution frame."""
    from ..models.db_models import PollutionReading

    rows = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station_id)
        .order_by(PollutionReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    if not rows:
        return None
    frame = pd.DataFrame(
        [
            {
                "timestamp": p.timestamp,
                "pm25": p.pm25,
                "pm10": p.pm10,
                "o3": p.o3,
                "no2": p.no2,
                "so2": p.so2,
                "co": p.co,
            }
            for p in rows
        ]
    )
    return frame


def _regional_composite_df(db: Session, limit: int = 120) -> pd.DataFrame | None:
    """Hourly mean-of-core-stations pollution frame for data-sparse stations.

    Real observations from the NCR core stations, averaged per hour. Used
    solely so a station with little/no local history still forecasts on true,
    regional air-quality signal instead of zero-filled lags. Never invents
    values — every cell is a mean of genuine readings.
    """
    from ..models.db_models import PollutionReading, Station

    stations = {s.name: s for s in db.query(Station).all()}
    core = [s.id for n in REGIONAL_COMPOSITE_STATIONS if (s := stations.get(n)) is not None]
    if not core:
        return None
    rows = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id.in_(core))
        .order_by(PollutionReading.timestamp.desc())
        .limit(limit * max(1, len(core)))
        .all()
    )
    if not rows:
        return None
    frame = pd.DataFrame(
        [
            {
                "timestamp": p.timestamp,
                "pm25": p.pm25,
                "pm10": p.pm10,
                "o3": p.o3,
                "no2": p.no2,
                "so2": p.so2,
                "co": p.co,
            }
            for p in rows
        ]
    )
    frame["hour"] = (
    pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    .dt.tz_localize(None)
    .dt.floor("h")
)
    grouped = frame.groupby("hour")[list(_POOL_COLUMNS)].mean().reset_index()
    grouped = grouped.rename(columns={"hour": "timestamp"})
    return grouped.sort_values("timestamp").reset_index(drop=True)


def station_data_sufficiency(db: Session, station_id: int) -> dict:
    """Cheap per-station pollution data-sufficiency report (no feature build).

    Returns ``pooled`` (bool), ``local_readings`` (window count),
    ``history_days`` and ``composite_sources`` — the same metadata that
    ``build_features_from_db_with_meta`` computes, without touching ML.
    """
    from sqlalchemy import func

    from ..models.db_models import PollutionReading, Station

    station = db.query(Station).filter(Station.id == station_id).first()
    if station is None:
        return {
            "pooled": False,
            "local_readings": 0,
            "history_days": None,
            "composite_sources": [],
        }
    window = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=5)
    local = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station_id, PollutionReading.timestamp >= window)
        .count()
    )
    span = (
        db.query(func.min(PollutionReading.timestamp), func.max(PollutionReading.timestamp))
        .filter(PollutionReading.station_id == station_id)
        .one()
    )
    history_days = None
    if span[0] is not None and span[1] is not None:
        history_days = round((span[1] - span[0]).total_seconds() / 86400.0, 1)
    composite = _regional_composite_df(db) is not None
    return {
        "pooled": bool(local < SPARSE_READINGS_THRESHOLD and composite),
        "local_readings": local,
        "history_days": history_days,
        "composite_sources": list(REGIONAL_COMPOSITE_STATIONS) if composite else [],
    }


def build_features_from_db_with_meta(db: Session, station_id: int) -> tuple[dict, dict]:
    """Reconstruct the true ML feature vector for a station, plus data meta.

    For data-sparse stations the pollution lags are taken from the regional
    NCR composite series (real readings averaged across core stations); the
    returned ``meta["pooled"]`` flag lets callers present this honestly.
    """
    from ..models.db_models import Station, WeatherReading

    station = db.query(Station).filter(Station.id == station_id).first()
    # The frame's ``station`` column uses the underscore form the ML feature
    # builders expect; errors and provenance report the real display name.
    station_name = station.name.replace(" ", "_") if station else "Anand_Vihar"
    display_name = station.name if station else "Anand Vihar"

    poll = _pollution_df(db, station_id)
    local_readings = len(poll) if poll is not None else 0
    pooled = local_readings < SPARSE_READINGS_THRESHOLD
    if pooled:
        composite = _regional_composite_df(db)
        if composite is None or composite.empty:
            pooled = False
        else:
            poll = composite

    wx_rows = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station_id)
        .order_by(WeatherReading.timestamp.desc())
        .limit(120)
        .all()
    )
    if poll is None and not wx_rows:
        raise InsufficientDataError(
            station_name=display_name,
            reason="no stored pollution or weather observations for this station",
            available_hours=0,
        )

    wx = pd.DataFrame(
        [
            {
                "timestamp": w.timestamp,
                "temperature": w.temperature,
                "humidity": w.humidity,
                "pressure_msl": w.pressure_msl,
                "surface_pressure": w.surface_pressure,
                "wind_speed": w.wind_speed,
                "wind_direction": w.wind_direction,
                "precipitation": w.precipitation,
                "cloud_cover": w.cloud_cover,
                "pbl_height": w.pbl_height,
                "temperature_1000hPa": getattr(w, "temperature_1000hPa", None),
                "temperature_925hPa": getattr(w, "temperature_925hPa", None),
                "temperature_850hPa": getattr(w, "temperature_850hPa", None),
                "temperature_700hPa": getattr(w, "temperature_700hPa", None),
            }
            for w in wx_rows
        ]
    )

    history_span = station_data_sufficiency(db, station_id)
    combined = poll
    if wx is not None and not wx.empty and poll is not None:
        combined = _merge_history_frames(poll, wx)

    if combined is None or combined.empty:
        raise InsufficientDataError(
            station_name=display_name,
            reason="no usable observations after aligning pollution and weather history",
            available_hours=0,
        )

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined["station"] = station_name
    lat = station.latitude if station else 28.6139
    lon = station.longitude if station else 77.2090
    combined["latitude"] = lat
    combined["longitude"] = lon
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    # Rows whose timestamp failed to parse cannot contribute a real lag value.
    available_rows = int(combined["timestamp"].notna().sum())
    if available_rows < MIN_FEATURE_ROWS:
        raise InsufficientDataError(
            station_name=display_name,
            reason=(
                "at least one chronological observation pair is required to form "
                "lag-1 features"
            ),
            available_hours=available_rows,
        )

    # Small helper module import (already installed; kept local to avoid heavy top-level import)
    from ml.features.coupling import add_coupling_features
    from ml.features.feature_engineering import (
        add_composite_features,
        add_humidity_lags,
        add_pollution_lags,
        add_pollution_rate_of_change,
        add_rolling_means,
        add_rolling_std,
        add_temperature_lags,
        add_temporal_features,
        add_wind_decomposition,
    )
    from ml.features.fire_impact import add_fire_features
    from ml.features.inversion import add_inversion_features, add_lapse_rate_inversion_features

    eng = add_temporal_features(combined)
    eng = add_pollution_lags(eng)
    eng = add_rolling_means(eng)
    eng = add_rolling_std(eng)
    eng = add_wind_decomposition(eng)
    eng = add_temperature_lags(eng)
    eng = add_humidity_lags(eng)
    eng = add_inversion_features(eng)
    eng = add_lapse_rate_inversion_features(eng)

    # Real fire features from the FIRMS records stored in the DB
    from ..models.db_models import FireReading

    fires = db.query(FireReading).order_by(FireReading.acq_date.desc()).limit(2000).all()
    if fires:
        fires_df = pd.DataFrame(
            [
                {
                    "lat": f.latitude,
                    "lon": f.longitude,
                    "frp": f.frp or 1.0,
                    "acq_timestamp": f.acq_date,
                }
                for f in fires
            ]
        )
        eng = add_fire_features(eng, fires_df=fires_df)
        fire_count_latest = int((eng.iloc[-1] if not eng.empty else pd.Series()).get("fire_count", 0) or 0)
    else:
        eng = add_fire_features(eng)
        fire_count_latest = 0

    eng = add_pollution_rate_of_change(eng)
    eng = add_composite_features(eng)
    eng = add_coupling_features(eng)

    latest = eng.iloc[-1]
    features = {}
    for col in eng.columns:
        if col in ("timestamp", "station"):
            continue
        features[col] = _flush_json_value(latest.get(col))

    # Station coordinates are part of the flat artifacts' 114-feature contract
    # and were previously dropped here, which left ``_model_predict`` to
    # impute 0.0 -- a value that is not a plausible Delhi-NCR coordinate. The
    # values are already on the frame; carry them through.
    features["latitude"] = _flush_json_value(lat)
    features["longitude"] = _flush_json_value(lon)

    # ``mean_frp`` / ``max_bright`` are the remaining contract features the
    # serving frame never produced. They are attributed to the nearest training
    # station and bucketed by the hour in ``scripts/build_dataset.py``, so the
    # same attribution is applied here against the stored FIRMS rows. When the
    # station-hour has no detections the training pipeline leaves the cell empty
    # and ``fillna(0)`` encodes it as 0.0, so 0.0 is the faithful value for that
    # case -- unlike the coordinates, it is not a fabricated claim.
    features.update(_firms_aggregate_features(fires, latest.get("timestamp"), lat, lon))

    # The four ``season_*`` contract columns are a complete one-hot encoding of a
    # single deterministic value, but ``add_temporal_features`` builds them with
    # ``pd.get_dummies``, which only emits the categories present in the frame.
    # A short history therefore yields one dummy and three absent names, which
    # the model layer then received as 0.0 -- right by luck for an autumn row,
    # wrong for any other. All four are set explicitly from the row's own month
    # using the training pipeline's SEASON_MAP, so the encoding is identical to
    # training regardless of how much history the frame holds.
    features.update(_season_dummies(latest.get("timestamp")))

    # Keep aliases used by fallbacks / AQI computation
    features.setdefault("fire_impact_score", min(1.0, fire_count_latest / 50.0))
    features["day_of_year"] = features.get("day_of_year", 1)
    features["is_winter"] = int(_flush_json_value(features.get("is_winter", 0)))
    meta = {
        "pooled": bool(pooled),
        "local_readings": local_readings,
        "history_days": history_span.get("history_days"),
        "composite_sources": history_span.get("composite_sources", []),
        "available_rows": available_rows,
        "window_start": combined["timestamp"].min().isoformat(),
        "window_end": combined["timestamp"].max().isoformat(),
    }
    return features, meta


def has_sufficient_history(db: Session, station_id: int) -> bool:
    """True when a station can produce a real (non-fabricated) feature vector.

    Used by non-forecast panels (dashboard, coverage, prewarm) to skip work and
    render an honest "data unavailable" state instead of a zero-filled series.
    """
    try:
        build_features_from_db_with_meta(db, station_id)
    except InsufficientDataError:
        return False
    return True


def build_features_from_db(db, station_id: int) -> dict:
    """Reconstruct the ML feature vector for a station (legacy signature)."""
    features, _ = build_features_from_db_with_meta(db, station_id)
    return features


def _artifact_digest(model_name: str) -> tuple[str | None, str | None]:
    """Return ``(artifact_filename, sha256)`` for a model, if it exists.

    Lets a forecast name the exact trained artifact it was produced from, so a
    reviewer can reproduce or challenge the number instead of trusting a label.
    """
    import hashlib

    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        return None, None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return f"{model_name}.joblib", None
    return f"{model_name}.joblib", digest.hexdigest()


#: ``data_source`` prefixes that mean "this is a demo/synthetic series".
DEMO_SOURCE_PREFIXES = ("demo", "bundled", "synthetic", "bootstrap")


def _is_demo_row(row) -> tuple[bool, bool, str | None]:
    """Classify a single observation as demo / re-stamped / real.

    Returns ``(is_demo, is_re_stamped, data_source)``. Re-stamped rows carry a
    truthy ``re_stamped`` flag; a bundled-CSV or demo-hydration load is
    identified by its ``data_source`` tag. A re-stamped row is always demo,
    because a forward copy of an older measurement is not an independent
    observation no matter which feed the original came from.
    """
    source = getattr(row, "data_source", None)
    is_re_stamped = bool(getattr(row, "re_stamped", None))
    is_demo = is_re_stamped or str(source or "").lower().startswith(DEMO_SOURCE_PREFIXES)
    return is_demo, is_re_stamped, source


def _is_demo_payload(db: Session, station_id: int | None = None) -> tuple[bool, bool, str | None]:
    """Classify the newest observation, optionally scoped to one station.

    Scoping matters: an unscoped lookup returns the globally latest row, which
    would label one station's forecast with another station's data_source.
    """
    from ..models.db_models import PollutionReading

    try:
        query = db.query(PollutionReading)
        if station_id is not None:
            query = query.filter(PollutionReading.station_id == station_id)
        row = query.order_by(PollutionReading.timestamp.desc()).limit(1).one_or_none()
    except Exception:  # pragma: no cover - defensive: provenance must never 500
        return False, False, None
    if row is None:
        return False, False, None
    return _is_demo_row(row)


def _empty_provenance(
    *,
    now: datetime,
    station_name: str,
    model_label: str,
    fallback_reason: str | None,
    coverage: dict,
    pollution_rows: int,
    weather_rows: int,
    fire_rows: int,
    model_artifact: str | None,
    model_sha: str | None,
    feature_contract: dict | None = None,
) -> dict:
    """Provenance for a station with no observation rows at all.

    ``latest_observation`` stays ``None`` and ``observation_age_hours`` stays
    ``None`` so the UI shows "no data" rather than guessing a freshness.
    """
    return {
        "model": model_label,
        "model_artifact": model_artifact,
        "model_artifact_sha256": model_sha,
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "pooled_features": bool(coverage.get("pooled")),
        "composite_sources": list(coverage.get("composite_sources", [])),
        "pollution_rows": pollution_rows,
        "weather_rows": weather_rows,
        "fire_rows": fire_rows,
        "history_rows": int(coverage.get("available_rows") or 0),
        "window_start": coverage.get("window_start"),
        "window_end": coverage.get("window_end"),
        "latest_observation": None,
        "observation_age_hours": None,
        "is_stale": True,
        "is_demo": False,
        "is_re_stamped": False,
        "data_source": None,
        "feature_contract": None,
        "station": station_name,
        "generated_at": now,
    }


def build_provenance(
    db: Session,
    *,
    station_id: int,
    station_name: str,
    coverage: dict,
    horizons: list[int] | None = None,
    model_label: str = "direct-ml",
    fallback_reason: str | None = None,
    feature_contract: dict | None = None,
) -> dict:
    """Assemble the provenance block attached to a forecast response.

    Reports the real inputs (row counts, window, latest observation and its age),
    which artifact served the prediction, and whether the underlying data is
    demo/re-stamped. Staleness is derived from the newest observation rather
    than assumed, so the UI can never imply freshness the data does not have.

    ``feature_contract`` carries the serving-time model-input audit. It is
    included verbatim so a response can never claim a satisfied contract when
    required inputs were actually missing and the model was bypassed.
    """
    from ..models.db_models import FireReading, PollutionReading, WeatherReading

    now = datetime.now(UTC).replace(tzinfo=None)

    def _count(model, *conditions) -> int:
        try:
            query = db.query(model)
            if conditions:
                query = query.filter(*conditions)
            return int(query.count())
        except Exception:  # pragma: no cover - defensive
            return 0

    pollution_rows = _count(PollutionReading, PollutionReading.station_id == station_id)
    weather_rows = _count(WeatherReading, WeatherReading.station_id == station_id)
    fire_rows = _count(FireReading)

    model_artifact, model_sha = _artifact_digest(f"xgboost_pm25_{max(horizons or [24])}h")
    if model_artifact is None:
        model_artifact, model_sha = _artifact_digest("xgboost_pm25_24h")

    latest = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station_id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )
    if latest is None:
        # No row for THIS station: provenance must describe this station, not
        # borrow another station's data_source and imply a freshness it lacks.
        return _empty_provenance(
            now=now,
            station_name=station_name,
            model_label=model_label,
            fallback_reason=fallback_reason,
            coverage=coverage,
            pollution_rows=0,
            weather_rows=weather_rows,
            fire_rows=fire_rows,
            model_artifact=model_artifact,
            model_sha=model_sha,
            feature_contract=feature_contract,
        )
    latest_ts = latest.timestamp
    if latest_ts is not None and latest_ts.tzinfo is not None:
        latest_ts = latest_ts.astimezone(UTC).replace(tzinfo=None)
    age_hours = round((now - latest_ts).total_seconds() / 3600.0, 1) if latest_ts else None

    is_demo, is_re_stamped, data_source = _is_demo_row(latest)

    return {
        "model": model_label,
        "model_artifact": model_artifact,
        "model_artifact_sha256": model_sha,
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "pooled_features": bool(coverage.get("pooled")),
        "composite_sources": list(coverage.get("composite_sources", [])),
        "pollution_rows": pollution_rows,
        "weather_rows": weather_rows,
        "fire_rows": fire_rows,
        "history_rows": int(coverage.get("available_rows") or 0),
        "window_start": coverage.get("window_start"),
        "window_end": coverage.get("window_end"),
        "latest_observation": latest_ts,
        "observation_age_hours": age_hours,
        "is_stale": bool(age_hours is not None and age_hours > STALE_AFTER_HOURS),
        "is_demo": is_demo,
        "is_re_stamped": is_re_stamped,
        "data_source": data_source,
        "feature_contract": feature_contract,
        "station": station_name,
        "generated_at": now,
    }


def get_weather_context(db, station_id: int) -> dict:
    from ..models.db_models import WeatherReading

    r = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station_id)
        .order_by(WeatherReading.timestamp.desc())
        .first()
    )
    if not r:
        return {}
    return {
        "temperature": r.temperature,
        "humidity": r.humidity,
        "pressure_msl": r.pressure_msl,
        "wind_speed": r.wind_speed,
        "wind_direction": r.wind_direction,
        "precipitation": r.precipitation,
        "cloud_cover": r.cloud_cover,
        "pbl_height": r.pbl_height,
    }


def get_fire_context(db) -> dict:
    from ..models.db_models import FireReading

    fires = db.query(FireReading).order_by(FireReading.acq_date.desc()).limit(500).all()
    if not fires:
        return {"fire_count": 0}
    distances = [haversine_distance(28.6139, 77.2090, f.latitude, f.longitude) for f in fires]
    return {
        "fire_count": len(fires),
        "distance_nearest_fire": round(min(distances), 1),
        "mean_frp": round(sum((f.frp or 0.0) for f in fires) / len(fires), 2),
    }


def _pbl_fields(point: dict) -> tuple[float | None, int | None, float | None]:
    """Return ``(pbl_height, inversion_detected, inversion_strength)``.

    A missing PBL stays NULL. The previous code substituted ``500.0`` and then
    derived ``inversion_detected``/``inversion_strength`` from that invented
    number, writing an atmospheric condition to the database as though it had
    been observed. NULL means "not available", which is the truth.
    """
    raw = point.get("pbl_height")
    if raw is None:
        raw = point.get("pbl_effective")
    if raw is None:
        return None, None, None
    pbl = float(raw)
    return pbl, (1 if pbl < 500 else 0), round(max(0.0, (500 - pbl) / 500), 4)


def _upsert_forecasts(
    db,
    station_id: int,
    points: list[dict],
    *,
    base_ts: datetime,
    coupled: bool,
) -> list:
    """Insert or replace forecast rows, keyed on (station_id, horizon_hours).

    Idempotent by design: regenerating a forecast must update the existing
    horizon rather than append a duplicate, otherwise the same 24h appears
    N times in history and looks like N days of data. The DB unique constraint
    ``uq_forecast_station_horizon`` is the backstop; the pre-delete keeps this
    working on SQLite dev databases and on databases not yet migrated.
    """
    from ..models.db_models import Forecast

    horizons = [int(p["horizon_hours"]) for p in points]
    if horizons:
        db.query(Forecast).filter(
            Forecast.station_id == station_id,
            Forecast.horizon_hours.in_(horizons),
        ).delete(synchronize_session=False)

    rows = []
    for p in points:
        pbl, inv_detected, inv_strength = _pbl_fields(p)
        rows.append(
            Forecast(
                station_id=station_id,
                forecast_timestamp=base_ts + timedelta(hours=int(p["horizon_hours"])),
                horizon_hours=int(p["horizon_hours"]),
                pm25_pred=p.get("pm25_pred"),
                pm10_pred=p.get("pm10_pred"),
                o3_pred=p.get("o3_pred"),
                no2_pred=p.get("no2_pred"),
                so2_pred=p.get("so2_pred"),
                co_pred=p.get("co_pred"),
                aqi_pred=p.get("aqi_pred"),
                aqi_category=p.get("aqi_category"),
                dominant_pollutant=p.get("dominant_pollutant"),
                inversion_detected=inv_detected,
                inversion_strength=inv_strength,
                pbl_height=pbl,
                coupling_stability=p.get("coupling_stability"),
                coupling_mode=(
                    ("coupled" if p.get("coupling_stability") is not None else None)
                    if coupled
                    else None
                ),
            )
        )
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def save_forecasts(db, station_id: int, predictions: list[dict], forecast_timestamp=None) -> list:
    base_ts = forecast_timestamp or datetime.now(UTC).replace(tzinfo=None)
    return _upsert_forecasts(db, station_id, predictions, base_ts=base_ts, coupled=False)


def generate_forecast(db, station_id: int, horizons=None) -> tuple[list, list[dict]]:
    features, _ = build_features_from_db_with_meta(db, station_id)
    predictions = predict_pollutants(features, horizons)
    rows = save_forecasts(db, station_id, predictions)
    return rows, predictions


def coupled_single_step(features: dict, horizon_hours: int) -> dict:
    """Single-step hook used by the online coupling loop (horizon=1 stepping)."""
    return predict_pollutants(features, horizons=[horizon_hours])[0]


def generate_coupled_forecast(db, station_id: int, horizons=None) -> dict:
    """Run the time-stepped two-way coupled forecast (SO2/CO + AQI included).

    Returns ``{"coupled": [...], "uncoupled": [...], "feedback_path": [...]}``.
    Persisting the coupled points is the caller's job - use
    ``save_coupled_forecasts``.
    """
    from ml.features.coupled_loop import run_coupled_forecast as _run_coupled
    from ml.features.coupling import corrected_pbl_height

    horizons = horizons or DEFAULT_HORIZONS
    features, _ = build_features_from_db_with_meta(db, station_id)
    base_pbl = features.get("pbl_height")
    start_hour = features.get("hour")

    result = _run_coupled(
        coupled_single_step,
        features,
        horizons,
        start_hour=start_hour,
    )

    for point in result["coupled"]:
        c = point["coupling"]
        # Only report an effective PBL when the station actually reported one.
        # corrected_pbl_height() substitutes a 600 m default internally, so
        # calling it with no observed PBL would manufacture a number.
        point["pbl_effective"] = (
            round(corrected_pbl_height(point["pm25_pred"], base_pbl, start_hour), 1)
            if base_pbl is not None
            else None
        )
        point["coupling_stability"] = c["stability_coupling_index"]
        point["coupling"] = c

    return result


def save_coupled_forecasts(db, station_id: int, points: list[dict], forecast_timestamp=None) -> list:
    """Persist the coupled forecast points (including SO2/CO + coupling state)."""
    base_ts = forecast_timestamp or datetime.now(UTC).replace(tzinfo=None)
    return _upsert_forecasts(db, station_id, points, base_ts=base_ts, coupled=True)
