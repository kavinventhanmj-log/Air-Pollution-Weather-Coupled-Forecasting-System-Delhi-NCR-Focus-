import numpy as np
import shap

FEATURE_DESCRIPTIONS = {
    "pm25_lag1": ("Previous-hour PM2.5 concentration", "High recent PM2.5 levels are carrying pollution forward"),
    "pm10_lag1": ("Previous-hour PM10 concentration", "Elevated coarse-particle levels add to the burden"),
    "o3_lag1": ("Previous-hour Ozone concentration", "Photochemical ozone contributes to the index"),
    "no2_lag1": ("Previous-hour NO2 concentration", "Traffic-derived NO2 is a key driver"),
    "so2_lag1": ("Previous-hour SO2 concentration", "Industrial SO2 emissions are present"),
    "co_lag1": ("Previous-hour CO concentration", "Combustion sources keep CO elevated"),
    "temperature": ("Temperature", "Temperature modulates photochemical production and mixing"),
    "humidity": ("Relative humidity", "High humidity promotes secondary aerosol formation"),
    "pressure_msl": ("MSL pressure", "Pressure regime influences ventilation"),
    "wind_speed": ("Wind speed", "Low wind speed is limiting pollutant dispersion"),
    "wind_direction": ("Wind direction", "Winds from the north-west advect stubble-burning smoke"),
    "precipitation": ("Precipitation", "Rain scavenges pollutants from the air"),
    "cloud_cover": ("Cloud cover", "Cloud cover reduces photochemical activity"),
    "pbl_height": (
        "Planetary boundary layer height",
        "A compressed boundary layer is trapping pollutants near the surface",
    ),
    "inversion_strength": ("Atmospheric inversion strength", "An inversion layer is inhibiting vertical mixing"),
    "fire_impact_score": ("Regional fire impact", "Stubble-burning smoke is being advected into the region"),
    "fire_count_100km": ("Fires within 100 km", "Nearby burning hotspots add fresh emissions"),
    "nearest_fire_km": ("Distance to nearest fire", "A close fire front signals imminent smoke"),
    "hour": ("Hour of day", "The diurnal cycle drives emission and chemistry patterns"),
    "is_winter": ("Winter season flag", "Winter inversions favour accumulation"),
    "day_of_year": ("Day of year", "Seasonal progression affects baseline pollution"),
    "aqi_lag1": ("Previous-hour AQI", "Persistent poor air continues into the forecast window"),
    "season": ("Season", "Season conditions the typical pollution regime"),
    "aod_est": ("Estimated aerosol optical depth", "Aerosol loading is attenuating solar radiation"),
    "radiation_transmittance": ("Surface radiation transmittance", "Aerosols are reducing surface solar heating"),
    "pbl_suppression_factor": (
        "PBL suppression by aerosols",
        "Aerosol radiative forcing is suppressing boundary-layer growth",
    ),
    "corrected_pbl_height": ("Aerosol-corrected PBL height", "Effective mixing layer is reduced by aerosol feedback"),
    "stability_coupling_index": (
        "Aerosol-PBL stability coupling",
        "Coupled aerosol-PBL feedback is stabilising the layer",
    ),
    "feedback_multiplier": (
        "Two-way feedback multiplier",
        "Stable coupled system is retaining pollutants near surface",
    ),
}

FALLBACK_WEIGHTS: list[tuple[str, float]] = []  # NOT USED: feature weights are never fabricated.


def _get_feature_names(model) -> list:
    for attr in ("feature_names_", "feature_names_in_"):
        if hasattr(model, attr):
            names = getattr(model, attr)
            if names is not None and len(names):
                return list(names)
    if hasattr(model, "model") and hasattr(model.model, "feature_names_in_"):
        names = model.model.feature_names_in_
        if names is not None and len(names):
            return list(names)
    return []


def _tree_estimator(model):
    """Return the underlying tree estimator for SHAP.

    AeroCast wrappers (RandomForestModel / XGBoostModel) expose the fitted
    scikit/xgboost estimator as ``.model``; shap.TreeExplainer needs the raw
    estimator, not the wrapper. Falls back to ``model`` itself when it is not
    a wrapper.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "predict"):
        return inner
    return model


def explain_prediction(model, features: dict) -> list[dict]:
    """Return real SHAP feature contributions for ``model`` on ``features``.

    Every value is computed with ``shap.TreeExplainer`` on the actual model —
    there are no hardcoded weights and no fabricated percentages. Raises
    ``RuntimeError`` (never fabricates) when the model cannot be explained.
    """
    cols = _get_feature_names(model)
    if not cols:
        raise RuntimeError(
            "no_tree_model: the loaded model exposes no feature names, so SHAP "
            "cannot attribute feature contributions. No fabricated weights are returned."
        )
    missing = [c for c in cols if features.get(c) is None]
    if missing:
        # SHAP on a zero-filled row produces genuine-looking attributions for
        # inputs that were never measured, which is worse than no explanation:
        # it reads as evidence. Refuse and name the gap instead.
        raise RuntimeError(
            f"missing_features: {len(missing)} of {len(cols)} model features are "
            f"unavailable for this row (e.g. {', '.join(missing[:5])}). SHAP "
            "attributions would be computed for values that were never measured, "
            "so no explanation is returned."
        )
    explainer = shap.TreeExplainer(_tree_estimator(model))
    arr = np.array([[float(features[k]) for k in cols]], dtype=float)
    shap_values = explainer.shap_values(arr)
    values = np.asarray(shap_values)
    if values.ndim >= 3:
        values = values.reshape(values.shape[0], -1)
    values = values[0]
    total = float(np.sum(np.abs(values)))
    if not total:
        raise RuntimeError(
            "no_variance: all SHAP contributions are zero for this feature row; "
            "the model attributes nothing. No fabricated weights are returned."
        )
    feature_importance = []
    for i, name in enumerate(cols):
        if i >= len(values):
            break
        v = float(values[i])
        desc, hint = FEATURE_DESCRIPTIONS.get(name, (name, ""))
        description = f"{desc}. {hint}" if hint else desc
        feature_importance.append(
            {
                "feature": name,
                "importance": round(abs(v), 4),
                "importance_pct": round(abs(v) / total * 100, 1),
                "direction": "positive" if v > 0 else "negative",
                "value": features.get(name),
                "description": description,
            }
        )
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    return feature_importance[:6]


def explain_fallback(features: dict) -> list[dict]:
    raise RuntimeError(
        "no_model_for_explanation: no tree-based model is available, so SHAP "
        "cannot compute feature contributions. No fabricated weights are returned."
    )


def generate_natural_language(features: dict, top_features: list[dict], prediction=None) -> list[str]:
    lines = []
    features = features or {}
    if not top_features:
        return [
            "Explanation unavailable: no tree-based model could be loaded for SHAP "
            "analysis, so no feature contributions are reported (no estimates are fabricated)."
        ]
    if top_features:
        first = top_features[0]
        value = features.get(first["feature"])
        value_txt = f" (value {value:g})" if isinstance(value, (int, float)) and value is not None else ""
        lines.append(
            f"The dominant driver of this forecast is {first['feature']}{value_txt}, "
            f"whose contribution is {first.get('description', 'important')}."
        )
    for name in ("wind_speed", "pbl_height", "fire_impact_score", "humidity", "inversion_strength"):
        val = features.get(name)
        if not isinstance(val, (int, float)) or val is None:
            continue
        if name == "wind_speed" and val < 2:
            lines.append(f"Low wind speed of {val:.1f} m/s is preventing dispersion of pollutants.")
        elif name == "pbl_height" and val is not None and val < 300:
            lines.append(f"The planetary boundary layer is compressed to about {val:.0f} m, limiting vertical mixing.")
        elif name == "inversion_strength" and val > 0.3:
            lines.append("A strong atmospheric inversion layer is trapping pollution near the ground.")
        elif name == "fire_impact_score" and val > 0.1:
            lines.append("Active stubble burning in Punjab/Haryana is contributing regional smoke.")
        elif name == "humidity" and val > 80:
            lines.append("High humidity is promoting secondary aerosol formation.")
    stability = features.get("stability_coupling_index")
    if isinstance(stability, (int, float)) and stability:
        if stability >= 0.6:
            lines.append(
                "Aerosol-PBL coupling is strong: the aerosol layer is suppressing the boundary layer, "
                "which traps more PM2.5 at the surface and feeds back into even higher aerosol loading."
            )
        elif stability >= 0.35:
            lines.append(
                "Aerosol-PBL coupling is active: reduced vertical mixing from the coupled feedback is "
                "moderately enhancing pollutant retention."
            )
    if prediction:
        pts = []
        for key in ("pm25_pred", "pm10_pred", "o3_pred", "no2_pred"):
            if prediction.get(key) is not None:
                pts.append(f"{key[:-5].upper()} {prediction[key]:g}")
        if pts:
            # Only cite the AQI fields when the caller actually supplied them.
            #
            # `api/explanation.py` passes only `{"pm25_pred": ...}`, because the
            # AQI there is a PM2.5-only sub-index and the remaining sub-indices are
            # deliberately left absent. Unguarded `.get()` calls rendered the
            # literal text "None" for AQI, category and dominant pollutant, which
            # reached the AI Explanation page as
            # "with AQI None (None), dominated by None."
            aqi = prediction.get("aqi_pred")
            if aqi is None:
                lines.append(f"Forecast concentrations: {', '.join(pts)}.")
            else:
                category = prediction.get("aqi_category")
                dominant = prediction.get("dominant_pollutant")
                category_text = f" ({category})" if category else ""
                dominant_text = f", dominated by {dominant}." if dominant else "."
                lines.append(
                    f"Forecast concentrations: {', '.join(pts)} with AQI {aqi:g}"
                    f"{category_text}{dominant_text}"
                )
    return lines
