"""PM2.5 forecasting service: builds a feature row from the live DB and runs
the trained direct multi-horizon XGBoost models.

Consistency with training is maintained by re-using the exact same feature
engineering from ``ml.preprocessing.training_dataset`` on a rolling window of
recent observations, then taking the last aligned row (the latest hour) as the
feature vector. The forecast is issued from that hour; no future information is
used (fire windows are strictly trailing; weather is joined at the same hour).
"""

from __future__ import annotations

import logging
import os
import pathlib
from datetime import timedelta
from typing import Any

import pandas as pd

from ..models.db_models import FireReading, PollutionReading, Station, WeatherReading
from ..utils.helpers import repo_root

logger = logging.getLogger("pm25_forecast_service")

DEFAULT_MODEL_DIR = repo_root() / "models" / "pm25"

WINDOW_DAYS = 5
MIN_POLLUTION_ROWS = 4
FIRE_BUFFER_HOURS = 26
WINDOW_SPANS_DAYS = (10, 30, 120)  # adaptive: try wider windows on sparse feeds

_weather_cols = [
    "station_id",
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "pressure_msl",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "pbl_height",
    "temperature_1000hPa",
    "temperature_925hPa",
    "temperature_850hPa",
    "temperature_700hPa",
]

_forecaster: Any | None = None


def reset_forecaster() -> None:
    """Drop the cached forecaster (used by tests to switch model dirs)."""
    global _forecaster
    _forecaster = None


def get_model_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AEROCAST_PM25_MODEL_DIR", DEFAULT_MODEL_DIR))


def get_pm25_forecaster():
    """Return the cached Pm25Forecaster bound to the configured model dir."""
    global _forecaster
    from ml.inference.pm25_forecaster import Pm25Forecaster

    model_dir = get_model_dir()
    if _forecaster is None or str(_forecaster.model_dir) != str(model_dir):
        _forecaster = Pm25Forecaster(model_dir)
    return _forecaster


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """A zero-row frame that still carries ``columns``.

    ``pd.DataFrame([])`` has *no* columns, and every downstream consumer indexes
    a column unconditionally (``align_observations`` does ``wx["timestamp"]``).
    A window with no rows must therefore still be shaped like a window, so the
    exogenous features arrive as NaN and are handled by the model rather than
    crashing the request.
    """
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _query_pollution(db, station_id: int, since, until=None) -> pd.DataFrame:
    query = db.query(PollutionReading).filter(
        PollutionReading.station_id == station_id,
        PollutionReading.timestamp >= since,
    )
    if until is not None:
        query = query.filter(PollutionReading.timestamp <= until)
    rows = query.order_by(PollutionReading.timestamp).all()
    if not rows:
        return _empty_frame(["station_id", "timestamp", "pm25"])
    return pd.DataFrame([{"station_id": r.station_id, "timestamp": r.timestamp, "pm25": r.pm25} for r in rows])


def _query_weather(db, station_id: int, since, until=None) -> pd.DataFrame:
    query = db.query(WeatherReading).filter(
        WeatherReading.station_id == station_id,
        WeatherReading.timestamp >= since,
    )
    if until is not None:
        query = query.filter(WeatherReading.timestamp <= until)
    rows = query.order_by(WeatherReading.timestamp).all()
    if not rows:
        return _empty_frame(_weather_cols)
    return pd.DataFrame([{c: getattr(r, c) for c in _weather_cols} for r in rows])


def _query_fires(db, since, until) -> pd.DataFrame:
    rows = db.query(FireReading).filter(FireReading.acq_date >= since, FireReading.acq_date <= until).all()
    return pd.DataFrame([{"lat": r.latitude, "lon": r.longitude, "acq_date": r.acq_date, "frp": r.frp} for r in rows])


def build_feature_row(
    db,
    station: Station,
    as_of_hour: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.Timestamp, dict[str, Any]]:
    """Build the feature vector for the latest hour of ``station``.

    Re-uses the training-dataset pipeline on a rolling window so feature
    semantics match training exactly. Returns (feature_row, release_time,
    context) where ``context`` records the data actually used (for audit).

    When ``as_of_hour`` is given, the window is anchored on that hour (data up
    to and including it) instead of the latest observation in the database —
    used to explain forecasts that were issued for a *past* release hour.
    """
    from ml.preprocessing.training_dataset import (
        add_atmosphere_and_temporal_features,
        add_fire_features,
        add_target_lags_and_rolling,
        align_observations,
    )

    # The model's "now" is the LATEST available pollution observation, not the
    # wall clock (the feed may lag). Anchor the feature window on it — or on an
    # explicit historical hour when re-building a stored forecast's feature row.
    if as_of_hour is not None:
        latest_poll_ts = pd.Timestamp(as_of_hour).floor("h")
    else:
        latest_poll = (
            db.query(PollutionReading)
            .filter(PollutionReading.station_id == station.id)
            .order_by(PollutionReading.timestamp.desc())
            .first()
        )
        if latest_poll is None:
            raise ValueError(f"No pollution readings for station '{station.name}'")
        latest_poll_ts = pd.Timestamp(latest_poll.timestamp)

    # Adaptive window: keep extending until we have enough history for the
    # 24h rolling features, or give up beyond the largest span.
    poll = None
    for span_days in WINDOW_SPANS_DAYS:
        since = latest_poll_ts - timedelta(days=span_days)
        poll = _query_pollution(db, station.id, since, until=latest_poll_ts)  # type: ignore[arg-type]
        if len(poll) >= MIN_POLLUTION_ROWS:
            break

    if poll is None or len(poll) < MIN_POLLUTION_ROWS:
        raise ValueError(
            f"Not enough recent pollution history for '{station.name}' "
            f"(have {0 if poll is None else len(poll)} rows, need >= {MIN_POLLUTION_ROWS})"
        )

    wx = _query_weather(
        db,
        station.id,  # type: ignore[arg-type]
        latest_poll_ts - timedelta(days=WINDOW_DAYS),
        until=latest_poll_ts,
    )

    release_hour = latest_poll_ts.floor("h")
    fires_until = release_hour
    fires_since = release_hour - timedelta(hours=FIRE_BUFFER_HOURS)
    fires = _query_fires(db, fires_since, fires_until)

    stations_df = pd.DataFrame(
        [{"station": station.name, "latitude": station.latitude, "longitude": station.longitude}]
    )
    poll["station"] = station.name
    wx["station"] = station.name
    poll = poll.dropna(subset=["station"])
    wx = wx.dropna(subset=["station"])

    df, _ = align_observations(poll, wx, stations_df, drop_target_null=False)
    df = add_atmosphere_and_temporal_features(df)
    df = add_fire_features(df, fires)
    df = add_target_lags_and_rolling(df)

    if df.empty:
        raise ValueError(f"No aligned observations for '{station.name}'")

    last = df.iloc[-1]
    release_time = pd.Timestamp(last["hour"])
    feature_row = {c: last.get(c) for c in df.columns}
    feature_row = {c: (None if pd.isna(v) else v) for c, v in feature_row.items()}

    context = {
        "pollution_rows_in_window": len(poll),
        "weather_rows_in_window": len(wx),
        "fires_in_window": len(fires),
        "last_observed_pm25": None if pd.isna(last.get("pm25")) else float(last["pm25"]),
        "pm25_lag1": None if pd.isna(last.get("pm25_lag1")) else float(last["pm25_lag1"]),
        "latest_weather_temperature": None if pd.isna(last.get("temperature")) else float(last["temperature"]),
        "latest_wind_speed": None if pd.isna(last.get("wind_speed")) else float(last["wind_speed"]),
        "latest_pbl_height": None if pd.isna(last.get("pbl_height")) else float(last["pbl_height"]),
    }
    return feature_row, release_time, context


def forecast_pm25(db, station_name: str, hours: int) -> dict[str, Any]:
    """Full PM2.5 forecast for one station covering 1..hours.

    Returns the response payload; raises ValueError/NotFound-style exceptions
    for the API layer to translate into HTTP errors.
    """
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise ValueError(f"station_not_found: {station_name}")

    forecaster = get_pm25_forecaster()
    if not forecaster.is_available:
        raise RuntimeError(
            "PM2.5 models are not trained yet; run `python -m ml.training.train_pm25 --horizons '1..72'` first"
        )

    feature_row, release_time, context = build_feature_row(db, station)

    desired = [h for h in range(1, hours + 1)]
    trained = forecaster.horizons
    if not trained or max(trained) < hours:
        max_h = forecaster.max_horizon
        raise RuntimeError(f"Requested {hours}h but trained PM2.5 models cover only up to {max_h}h")
    supported = desired  # all desired horizons are covered since hours <= max(trained)

    result = forecaster.forecast(feature_row, horizons=supported, base_time=release_time.to_pydatetime())

    forecasts = result["forecasts"]
    # Attach honest per-horizon held-out test metrics from training time
    for point in forecasts:
        met = forecaster.test_metrics(point["forecast_horizon"])
        point["test_mae"] = met.get("mae")
        point["test_rmse"] = met.get("rmse")
        point["test_r2"] = met.get("r2")
        point["test_n"] = met.get("n")

    return {
        "station": station.name,
        "station_id": station.id,
        "model": "xgboost",
        "forecast_strategy": result["strategy"],
        "uncertainty_method": result["uncertainty_method"],
        "coverage_target": result.get("coverage_target"),
        "feature_version": result.get("feature_version"),
        "release_time": release_time.to_pydatetime(),
        "data_as_of": release_time.to_pydatetime(),
        "generated_at": pd.Timestamp.now(tz=None).to_pydatetime(),
        "requested_hours": hours,
        "served_horizons": [p["forecast_horizon"] for p in forecasts],
        "context": context,
        "forecasts": forecasts,
    }
