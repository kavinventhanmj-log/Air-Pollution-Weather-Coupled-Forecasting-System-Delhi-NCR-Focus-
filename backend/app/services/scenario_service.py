"""What-If Scenario Analysis for AeroCast-NCR (PM2.5).

This module answers the question: *"if a small, controlled set of environmental
inputs were different, how would the *trained* forecasting system respond?"*

EXPLICIT NON-CLAIMS
-------------------
* This is **SCENARIO ANALYSIS**, not a causal experiment and not an emissions or
  dispersion simulation. The response is the black-box reaction of the deployed
  XGBoost PM2.5 models to a changed feature vector.
* No causal certainty is claimed. A "scenario" is an *input-level what-if*: only
  the requested features are perturbed, everything else is held fixed, and the
  resulting model output is compared with the baseline. It never estimates what
  "actually happens" in the atmosphere.
* **No values are fabricated.** Observations already in the database are read
  and reused; the scenario only applies the user's requested changes on copies
  in memory.

THE SAFETY CONTRACT
-------------------
* This module performs **no database writes at all**: no INSERT/UPDATE/DELETE,
  no ORM flush, no commit. ``build_feature_row`` and the fire window query are
  pure reads. Observational tables (pollution, weather, fire, forecast) are
  never touched.
* The ``data_integrity`` field in every response records this contract so it can
  be machine-checked, and the test-suite asserts the database is byte-for-byte
  unchanged after a scenario run.

How a scenario is applied (transparent, matches training semantics)
-------------------------------------------------------------------
1. Build the **baseline** feature row exactly as the forecast endpoint does
   (``pm25_forecast_service.build_feature_row``) — real stored observations.
2. Apply the requested raw input changes (wind speed, wind direction, PBL
   height) to a **copy** of that row.
3. Re-run the same feature functions used at training/build time
   (``add_atmosphere_and_temporal_features`` + ``add_fire_features``) so derived
   features (ventilation coefficient, wind-direction encoding, fire transport
   terms) are recomputed consistently from the changed inputs. Fire activity and
   inversion overrides are handled as documented first-order hooks (see
   ``docs/scenario_analysis.md``).
4. Run the **same trained models** on the baseline row and on the scenario row.
   The difference (scenario − baseline) per horizon is reported as the model's
   response, with the explicit disclaimer that it is not a causality claim.

Restrictions
------------
* Relative changes require a present baseline value (else 422).
* At least one change must be supplied (else 422).
* ``hours`` must be covered by the trained horizons (else 503).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..models.db_models import Station

logger = logging.getLogger("aerocast.scenario")

SCENARIO_LABEL = "SCENARIO ANALYSIS"

DISCLAIMER = (
    "This is a WHAT-IF (scenario) analysis: only the requested inputs were "
    "changed in an in-memory copy of the forecast feature row; everything else "
    "was held fixed and the trained XGBoost PM2.5 models were re-run on that "
    "changed row. The reported difference is the model's response to the input "
    "change and is NOT a measurement, a causal estimate, or an emissions/"
    "dispersion simulation. Observed data in the database was not modified."
)

# Same look-back constants the forecast pipeline uses (pm25_forecast_service).
FIRE_BUFFER_HOURS = 26
WINDOW_FIRE_HOURS = 24  # trailing window used by add_fire_features (training_dataset)

# Documented cap on fire-activity intensity scaling (keeps scores in [0,1]).
FIRE_ACTIVITY_MAX_MULTIPLIER = 100.0

_MODEL_FEATURE_SCENARIO_UNITS = {
    "wind_speed": "m/s",
    "wind_direction": "deg",
    "wind_dir_sin": "—",
    "wind_dir_cos": "—",
    "pbl_height": "m",
    "ventilation_coefficient": "m2/s",
    "inversion_detected": "0/1",
    "inversion_strength": "0-1",
    "fire_count": "fires",
    "fire_impact_score": "0-1",
    "nearest_fire_distance": "km",
    "wind_aligned_fire_count": "fires",
    "wind_alignment_pct": "%",
    "transport_time_hours": "h",
    "transport_risk": "0-1",
    "stubble_impact_score": "0-1",
}


def _num(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v


def _fmt(v, nd: int = 4) -> float | None:
    f = _num(v)
    return round(f, nd) if f is not None else None


def _resolve(base: float | None, change) -> float:
    """Apply one absolute/relative perturbation to a baseline value."""
    if change.mode == "absolute":
        return float(change.value)
    # relative
    if base is None:
        raise ValueError(
            "relative change requested but no baseline value is available; use the 'absolute' mode for this input"
        )
    return base * float(change.value)


def _query_fires(db, since, until) -> pd.DataFrame:
    """Read stored fires (trailing window) — SELECT only, never writes."""
    from ..models.db_models import FireReading

    rows = db.query(FireReading).filter(
        FireReading.acq_date >= since,
        FireReading.acq_date <= until,
        FireReading.synthetic.is_(False),
    ).all()
    return pd.DataFrame([{"lat": r.latitude, "lon": r.longitude, "acq_date": r.acq_date, "frp": r.frp} for r in rows])


def _one_row_df(feature_row: dict[str, Any], station: Station, release_hour: pd.Timestamp) -> pd.DataFrame:
    """Reconstruct the single (last) aligned observation row from the baseline
    feature row so the training-time feature functions can be re-run on it."""
    d = {
        "station": station.name,
        "timestamp": release_hour.to_pydatetime(),
        "hour": release_hour,
        "latitude": float(station.latitude),
        "longitude": float(station.longitude),
        "temperature": feature_row.get("temperature"),
        "humidity": feature_row.get("humidity"),
        "pressure_msl": feature_row.get("pressure_msl"),
        "surface_pressure": feature_row.get("surface_pressure"),
        "wind_speed": feature_row.get("wind_speed"),
        "wind_direction": feature_row.get("wind_direction"),
        "pbl_height": feature_row.get("pbl_height"),
    }
    for level in (1000, 925, 850, 700):
        d[f"temperature_{level}hPa"] = feature_row.get(f"temperature_{level}hPa")
    return pd.DataFrame([d])


def _effect(feature: str, baseline_row: dict, scenario_row: dict) -> dict[str, Any]:
    unit = _MODEL_FEATURE_SCENARIO_UNITS.get(feature)
    return {
        "feature": feature,
        "unit": unit,
        "baseline_value": _fmt(baseline_row.get(feature)),
        "scenario_value": _fmt(scenario_row.get(feature)),
    }


def _build_scenario_row(
    db,
    station: Station,
    baseline_row: dict[str, Any],
    release_hour: pd.Timestamp,
    changes,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Apply the requested changes to a copy of the baseline feature row.

    Returns ``(scenario_row, change_effects, notes)``. Pure reads from ``db``;
    nothing is ever written.
    """
    from ml.preprocessing.training_dataset import (
        add_atmosphere_and_temporal_features,
        add_fire_features,
    )

    baseline = dict(baseline_row)
    effects: list[dict[str, Any]] = []
    notes: list[str] = []

    df = _one_row_df(baseline, station, release_hour)

    # ---------------------------------------------------------------- wind speed
    if changes.wind_speed is not None:
        base_ws = _num(baseline.get("wind_speed"))
        new_ws = _resolve(base_ws, changes.wind_speed)
        df.loc[0, "wind_speed"] = new_ws
        notes.append(
            "wind_speed change: ventilation_coefficient, transport_time_hours and the "
            "wind-sensitive fire transport terms are recomputed from the scenario wind "
            "speed using the training-time feature functions."
        )
        logger.info("scenario %s wind_speed %s -> %s m/s", station.name, base_ws, new_ws)

    # ------------------------------------------------------------ wind direction
    if changes.wind_direction is not None:
        base_wdir = _num(baseline.get("wind_direction"))
        if changes.wind_direction.mode == "absolute":
            new_wdir = float(changes.wind_direction.value)
        else:
            if base_wdir is None:
                raise ValueError(
                    "wind_direction 'delta' change requested but baseline wind "
                    "direction is unavailable; use 'absolute' mode"
                )
            new_wdir = (base_wdir + float(changes.wind_direction.value)) % 360.0
        df.loc[0, "wind_direction"] = new_wdir
        notes.append(
            "wind_direction change: the wind-direction encoding (sin/cos) and the "
            "wind-alignment fire terms are recomputed from real stored fire geometry "
            "against the scenario direction."
        )
        logger.info("scenario %s wind_direction %s -> %s deg", station.name, base_wdir, new_wdir)

    # ---------------------------------------------------------------- pbl height
    if changes.pbl_height is not None:
        base_pbl = _num(baseline.get("pbl_height"))
        new_pbl = _resolve(base_pbl, changes.pbl_height)
        df.loc[0, "pbl_height"] = new_pbl
        notes.append(
            "pbl_height change: ventilation_coefficient is recomputed from the "
            "scenario PBL height; the inversion indicator is re-derived from PBL "
            "(lapse-rate proxy) unless explicitly overridden."
        )
        logger.info("scenario %s pbl_height %s -> %s m", station.name, base_pbl, new_pbl)

    # --------------------------------------------------------- fire activity hook
    fire_multiplier = 1.0
    if changes.fire_activity is not None:
        fire_multiplier = max(0.0, min(FIRE_ACTIVITY_MAX_MULTIPLIER, changes.fire_activity.multiplier))
        notes.append(
            f"fire_activity multiplier {fire_multiplier}x scales the FRP-weighted "
            "intensity terms (fire_impact_score, transport_risk, stubble_impact_score). "
            "fire_count / wind_aligned_fire_count reflect REAL detected fires and are "
            "not changed, and distance/time terms are unchanged."
        )

    # --------------------------------------------------- re-run training features
    df = add_atmosphere_and_temporal_features(df)

    fires_since = release_hour - timedelta(hours=FIRE_BUFFER_HOURS)
    fires = _query_fires(db, fires_since, release_hour)
    if changes.fire_activity is not None:
        if fires.empty:
            notes.append(
                "no stored fires in the trailing 24h window for this station; the "
                "fire_activity change had no effect on fire features."
            )
        else:
            fires = fires.copy()
            fires["frp"] = fires["frp"].to_numpy(dtype=float) * fire_multiplier
    df = add_fire_features(df, fires)

    scenario = dict(baseline)
    for col in df.columns:
        scenario[col] = _clean(df[col].iloc[0])

    # ------------------------------------------------- inversion indicator hook
    if changes.inversion is not None:
        scenario["inversion_detected"] = int(bool(changes.inversion.detected))
        if changes.inversion.strength is not None:
            scenario["inversion_strength"] = float(min(1.0, max(0.0, changes.inversion.strength)))
        notes.append(
            "inversion indicator overridden directly (the requested detected/strength "
            "values replace the re-derived indicator for this scenario run)."
        )
        logger.info(
            "scenario %s inversion_detected -> %s strength -> %s",
            station.name,
            scenario.get("inversion_detected"),
            scenario.get("inversion_strength"),
        )

    # ------------------------------------------------------------------- effects
    if changes.wind_speed is not None:
        effects.append(_effect("wind_speed", baseline, scenario))
        effects.append(_effect("ventilation_coefficient", baseline, scenario))
    if changes.wind_direction is not None:
        effects.append(_effect("wind_direction", baseline, scenario))
        effects.append(_effect("wind_dir_sin", baseline, scenario))
        effects.append(_effect("wind_dir_cos", baseline, scenario))
    if changes.pbl_height is not None:
        effects.append(_effect("pbl_height", baseline, scenario))
        effects.append(_effect("ventilation_coefficient", baseline, scenario))
        if changes.inversion is None:
            for feat in ("inversion_detected", "inversion_strength"):
                if (
                    feat in scenario
                    and _num(baseline.get(feat)) is not None
                    and _num(baseline.get(feat)) != _num(scenario.get(feat))
                ):
                    effects.append(_effect(feat, baseline, scenario))
    if changes.fire_activity is not None:
        for feat in ("fire_impact_score", "transport_risk", "stubble_impact_score"):
            effects.append(_effect(feat, baseline, scenario))
    if changes.inversion is not None:
        effects.append(_effect("inversion_detected", baseline, scenario))
        if changes.inversion.strength is not None:
            effects.append(_effect("inversion_strength", baseline, scenario))

    return scenario, effects, notes


def _annotate_test_metrics(forecaster, forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for point in forecasts:
        met = forecaster.test_metrics(point["forecast_horizon"])
        point["test_mae"] = met.get("mae")
        point["test_rmse"] = met.get("rmse")
        point["test_r2"] = met.get("r2")
        point["test_n"] = met.get("n")
    return forecasts


def run_scenario_analysis(db, station_name: str, hours: int, changes) -> dict[str, Any]:
    """Run a what-if scenario and return the labelled, read-only response.

    Raises ``ValueError`` for client errors (station not found, no changes,
    relative change without a baseline value) and ``RuntimeError`` when the
    trained models are unavailable or do not cover the requested horizon — the
    API layer translates these into HTTP codes. Never writes to the database.
    """
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise ValueError(f"station_not_found: {station_name}")

    has_change = any(
        c is not None
        for c in (
            getattr(changes, "wind_speed", None),
            getattr(changes, "wind_direction", None),
            getattr(changes, "pbl_height", None),
            getattr(changes, "fire_activity", None),
            getattr(changes, "inversion", None),
        )
    )
    if not has_change:
        raise ValueError("no_input_changes: at least one scenario change is required")

    from . import pm25_forecast_service as _p25

    forecaster = _p25.get_pm25_forecaster()
    if not forecaster.is_available:
        raise RuntimeError(
            "PM2.5 models are not trained yet; run `python -m ml.training.train_pm25 --horizons '1..72'` first"
        )

    baseline_row, release_time, context = _p25.build_feature_row(db, station)
    release_hour = pd.Timestamp(release_time)

    scenario_row, effects, notes = _build_scenario_row(db, station, baseline_row, release_hour, changes)

    trained = forecaster.horizons
    if not trained or max(trained) < hours:
        raise RuntimeError(f"Requested {hours}h but trained PM2.5 models cover only up to {max(trained)}h")
    desired = list(range(1, hours + 1))
    base_time = release_time.to_pydatetime()

    baseline_result = forecaster.forecast(baseline_row, horizons=desired, base_time=base_time)
    scenario_result = forecaster.forecast(scenario_row, horizons=desired, base_time=base_time)

    baseline_fc = _annotate_test_metrics(forecaster, baseline_result["forecasts"])
    scenario_fc = _annotate_test_metrics(forecaster, scenario_result["forecasts"])

    differences = []
    for b, s in zip(baseline_fc, scenario_fc, strict=True):
        differences.append(
            {
                "forecast_horizon": b["forecast_horizon"],
                "timestamp": b["timestamp"],
                "baseline_pm25": b["predicted_pm25"],
                "scenario_pm25": s["predicted_pm25"],
                "difference_pm25": round(float(s["predicted_pm25"]) - float(b["predicted_pm25"]), 2),
                "baseline_lower_bound": b.get("pm25_lower_bound"),
                "baseline_upper_bound": b.get("pm25_upper_bound"),
                "scenario_lower_bound": s.get("pm25_lower_bound"),
                "scenario_upper_bound": s.get("pm25_upper_bound"),
            }
        )

    def _summary(pts: list[dict[str, Any]]) -> dict[str, Any]:
        vals: list[float] = []
        for p in pts:
            v = _num(p["predicted_pm25"])
            if v is not None:
                vals.append(v)
        return {
            "peak_pm25": round(float(np.max(vals)), 2) if vals else None,
            "mean_pm25": round(float(np.mean(vals)), 2) if vals else None,
            "forecasts": pts,
        }

    obs_pm25 = _num(context.get("last_observed_pm25"))

    return {
        "label": SCENARIO_LABEL,
        "disclaimer": DISCLAIMER,
        "station": station.name,
        "station_id": station.id,
        "release_time": release_time.to_pydatetime(),
        "data_as_of": release_time.to_pydatetime(),
        "generated_at": pd.Timestamp.now(tz=None).to_pydatetime(),
        "model": "xgboost",
        "forecast_strategy": baseline_result.get("strategy"),
        "uncertainty_method": baseline_result.get("uncertainty_method"),
        "coverage_target": baseline_result.get("coverage_target"),
        "horizon_hours": hours,
        "served_horizons": desired,
        "value_kinds": {
            "observed": (
                "last stored CPCB PM2.5 observation used as the model's anchor (never modified by the scenario)."
            ),
            "forecast": "model output on the unchanged baseline feature row.",
            "scenario": "model output on the feature row after ONLY the requested input(s) changed.",
            "difference": "scenario − forecast per horizon. NOT a causal estimate — see disclaimer.",
        },
        "observed": {
            "pm25_last_observed_ugm3": obs_pm25,
            "pm25_lag1_anchor_ugm3": _num(context.get("pm25_lag1")),
            "timestamp": (release_time - timedelta(hours=1)).to_pydatetime(),
        },
        "baseline_forecast": _summary(baseline_fc),
        "scene_forecast": _summary(scenario_fc),
        "difference": {
            "peak_difference_pm25": (
                round(float(np.max([d["difference_pm25"] for d in differences])), 2) if differences else None
            ),
            "mean_difference_pm25": (
                round(float(np.mean([d["difference_pm25"] for d in differences])), 2) if differences else None
            ),
            "points": differences,
        },
        "input_changes": effects,
        "notes": notes,
        "data_integrity": {
            "mode": "read_only",
            "database_writes": 0,
            "database_verification": (
                "Scenarios run entirely in memory against copies of the forecast "
                "feature row. The module issues only SELECT queries and never "
                "flushes or commits; observational tables are untouched."
            ),
            "baseline_is_stored_observations": True,
            "scenario_overwrites_nothing": True,
        },
    }
