"""Online time-stepped two-way weather-chemistry coupled forecast loop.

This is the lightweight surrogate for a full WRF-Chem online coupling. Instead
of treating meteorology as fixed external input, pollutants and meteorology are
advanced *together* hour by hour:

  for each hour step:
      1. predict next-hour pollutant concentrations from current features
         (meteorology -> chemistry forward path)
      2. derive the aerosol radiative forcing from the freshly forecast PM2.5
         (AOD, radiation attenuation, PBL suppression, boundary-layer stability)
      3. correct the meteorological fields (PBL height, temperature, inversion
         strength, stability coupling index)
      4. advance the pollution lags and re-enter the loop with the corrected
         meteorology (chemistry -> meteorology feedback path)

The result is a true sequential two-way coupling simulation, not a one-pass
statistical forecast.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np

from .coupling import (
    boundary_stability_index,
    corrected_pbl_height,
    coupling_feedback_score,
    estimate_aod,
    pbl_suppression_factor,
    surface_radiation_attenuation,
    surface_temperature_damping,
)

# Predictor signature: predict(features: dict, horizon_hours: int) -> dict
# returning at minimum {"pm25_pred", "pm10_pred", "o3_pred", "no2_pred"} and
# optionally "so2_pred", "co_pred", "aqi_pred".

#: Nominal PBL (m) and surface wind (m/s) used when the station has no observed
#: value, so the loop can still advance. They are the standard Delhi-NCR
#: afternoon/mean values used elsewhere in this module and are reported as
#: ``pbl_height_source`` so a caller can tell an assumed PBL from an observed one.
DEFAULT_PBL_M = 600.0
DEFAULT_WIND_MPS = 4.0
#: Nominal surface temperature (degC) applied when no temperature is available.
DEFAULT_TEMPERATURE_C = 20.0


def _first_not_none(*values):
    """Return the first value that is not None, else None."""
    for value in values:
        if value is not None:
            return value
    return None


def _next_prediction(predict_func: Callable, features: dict) -> dict:
    """Get a single-hour-ahead prediction using the 1h forecasters."""
    return predict_func(features, horizon_hours=1)


def _apply_coupling_forcing(features: dict, pred: dict, hour: int) -> dict:
    """Advance + correct the feature vector using aerosol feedback physics."""
    f = deepcopy(features)

    # ``or`` is only correct here because a real measurement of 0.0 means
    # "pristine air" for pm25 and "dead calm" is not a valid PBL/wind reading;
    # the PM2.5 chain is checked with an explicit None test so a missing
    # prediction cannot silently become a clean-air 0.0.
    pm25 = _first_not_none(pred.get("pm25_pred"), pred.get("pm25"), f.get("pm25_lag1"))
    if pm25 is None:
        raise ValueError(
            "coupled loop requires a PM2.5 level (pm25_pred / pm25 / pm25_lag1); "
            "refusing to continue with a 0.0 stand-in"
        )
    pm25 = float(pm25)
    pbl = _first_not_none(f.get("pbl_height"), f.get("corrected_pbl_height"))
    wind = _first_not_none(f.get("wind_speed"))

    # chemistry -> meteorology: aerosol suppresses PBL, damps diurnal temp
    # An unobserved PBL/wind/temperature falls back to the documented nominal
    # value and is labelled, so nothing downstream can mistake an assumption
    # for a measurement.
    pbl_observed = pbl is not None
    wind_observed = wind is not None
    base_temp_raw = _first_not_none(f.get("temperature"))
    temp_observed = base_temp_raw is not None
    if pbl is None:
        pbl = DEFAULT_PBL_M
    if wind is None:
        wind = DEFAULT_WIND_MPS
    if base_temp_raw is None:
        base_temp_raw = DEFAULT_TEMPERATURE_C

    supp = pbl_suppression_factor(pm25, hour)
    corrected_pbl = corrected_pbl_height(pm25, pbl, hour)
    damping = surface_temperature_damping(pm25, hour)

    base_temp = float(base_temp_raw)
    damped_temp = 13.0 + (base_temp - 13.0) * damping

    stability = boundary_stability_index(pm25, pbl, wind, hour)
    aod = estimate_aod(pm25)
    transmittance = surface_radiation_attenuation(pm25)

    # update meteorological feature fields for the next step
    f["pbl_height"] = corrected_pbl
    f["corrected_pbl_height"] = corrected_pbl
    f["pbl_suppression_factor"] = supp
    f["aod_est"] = aod
    f["radiation_transmittance"] = transmittance
    f["temperature"] = damped_temp
    f["temperature_lag1"] = base_temp
    f["temperature_lag6"] = base_temp if "temperature_lag6" in f else f.get("temperature_lag6")
    f["inversion_strength"] = float(np.clip((500.0 - corrected_pbl) / 500.0, 0.0, 1.0))
    f["inversion_detected"] = int(corrected_pbl < 500)
    f["stability_coupling_index"] = stability
    f["feedback_multiplier"] = 1.0 + stability * 0.4
    f["pbl_height_source"] = "observed" if pbl_observed else "assumed_nominal"
    f["wind_speed_source"] = "observed" if wind_observed else "assumed_nominal"
    f["temperature_source"] = "observed" if temp_observed else "assumed_nominal"

    # advance pollution lags: shift current predictions into lag-1 slots
    for slug in ("pm25", "pm10", "o3", "no2", "so2", "co"):
        pred_key = f"{slug}_pred"
        lag_key = f"{slug}_lag1"
        if pred_key in pred and pred.get(pred_key) is not None:
            f[lag_key] = float(pred[pred_key])
            f[slug] = float(pred[pred_key])
            for extra in (3, 6, 12):
                key = f"{slug}_lag{extra}"
                if key in f:
                    f[key] = f[key]  # keep historical; simple persistence
    # rolling means approximate with new lag1 value
    for slug in ("pm25", "pm10", "o3", "no2"):
        lag_key = f"{slug}_lag1"
        if lag_key in f:
            for w in (3, 6, 12, 24):
                key = f"{slug}_roll_mean_{w}h"
                if key in f:
                    f[key] = (float(f[key]) * (w - 1) + float(f[lag_key])) / w

    return f


def run_coupled_forecast(
    predict_func: Callable,
    features: dict,
    horizons: list | None = None,
    start_hour: int = 12,
) -> dict:
    """Run the sequential two-way coupling forecast.

    Args:
        predict_func: callable(features, horizon_hours) -> dict of predictions.
            For the coupling loop it is invoked with horizon=1 at every step so
            the meteorology can be corrected in-between.
        features: starting feature vector (observed conditions).
        horizons: forecast horizons requested (hours after t0).
        start_hour: hour of day at t0 for the diurnal coupling factor.

    Returns:
        {
          "coupled": [ {horizon, predictions..., coupling: {...}} ... ],
          "uncoupled": [ dict of direct per-horizon predictions ],
          "feedback_path": [list of hourly (hour_index, pbl, pm25, stability)],
        }
    """
    horizons = sorted(horizons or [1, 6, 12, 24, 48, 72])
    hour = int(start_hour or 12) % 24

    coupled = []
    feedback_path = []
    state = deepcopy(features)

    # step continuously to the farthest horizon, record at requested horizons
    for h in range(1, max(horizons) + 1):
        pred = _next_prediction(predict_func, state)

        state = _apply_coupling_forcing(state, pred, hour)
        hour = (hour + 1) % 24

        diag_pm25 = _first_not_none(pred.get("pm25_pred"))
        if diag_pm25 is None:
            raise ValueError(
                f"coupled step h={h}: predictor returned no pm25_pred; refusing to "
                "report coupling diagnostics computed from a 0.0 stand-in"
            )
        diag_pbl = _first_not_none(state.get("pbl_height"))
        diag_wind = _first_not_none(state.get("wind_speed"))
        diag = coupling_feedback_score(
            float(diag_pm25),
            float(diag_pbl if diag_pbl is not None else DEFAULT_PBL_M),
            float(diag_wind if diag_wind is not None else DEFAULT_WIND_MPS),
            hour,
        )
        feedback_path.append({
            "t_plus": int(h),
            "pm25": round(float(diag_pm25), 1),
            "pbl_effective": round(diag["corrected_pbl_height"], 1),
            "stability": round(diag["stability_coupling_index"], 4),
            "feedback_multiplier": round(diag["feedback_multiplier"], 4),
            "pbl_source": state.get("pbl_height_source", "observed"),
            "wind_source": state.get("wind_speed_source", "observed"),
        })

        if h in horizons:
            coupled.append({
                "horizon_hours": h,
                "pm25_pred": float(pred.get("pm25_pred")),
                "pm10_pred": float(pred.get("pm10_pred")),
                "o3_pred": float(pred.get("o3_pred")),
                "no2_pred": float(pred.get("no2_pred")),
                "so2_pred": float(pred.get("so2_pred")) if pred.get("so2_pred") is not None else None,
                "co_pred": float(pred.get("co_pred")) if pred.get("co_pred") is not None else None,
                "aqi_pred": pred.get("aqi_pred"),
                "aqi_category": pred.get("aqi_category"),
                "dominant_pollutant": pred.get("dominant_pollutant"),
                "coupling": diag,
            })

    # direct (uncoupled) per-horizon predictions for skill comparison
    uncoupled = []
    for h in horizons:
        pred = predict_func(deepcopy(features), horizon_hours=h)
        uncoupled.append({
            "horizon_hours": h,
            "pm25_pred": float(pred.get("pm25_pred")),
            "pm10_pred": float(pred.get("pm10_pred")),
            "o3_pred": float(pred.get("o3_pred")),
            "no2_pred": float(pred.get("no2_pred")),
            "so2_pred": float(pred.get("so2_pred")) if pred.get("so2_pred") is not None else None,
            "co_pred": float(pred.get("co_pred")) if pred.get("co_pred") is not None else None,
            "aqi_pred": pred.get("aqi_pred"),
            "aqi_category": pred.get("aqi_category"),
            "dominant_pollutant": pred.get("dominant_pollutant"),
        })

    return {"coupled": coupled, "uncoupled": uncoupled, "feedback_path": feedback_path}
