"""Tests for the SHAP-based explanation dashboard path.

Covers ``app.services.explanation_service`` — the service behind the
"AI Explanation" card. The core contract is that explanations are computed
with real ``shap.TreeExplainer`` output and never fabricated.
"""

import numpy as np
import pytest
import shap
from app.services import explanation_service as svc
from sklearn.ensemble import RandomForestRegressor


class _WrapperWithInner:
    """Mimics AeroCast RandomForestModel/XGBoostModel wrappers."""

    def __init__(self, inner):
        self.model = inner


def _fit_tiny_forest() -> RandomForestRegressor:
    rng = np.random.default_rng(7)
    X = {"temperature": rng.normal(20.0, 4.0, 80), "wind_speed": rng.uniform(0.5, 6.0, 80)}
    y = 60.0 - 30.0 * X["wind_speed"] + 1.2 * X["temperature"] + rng.normal(0.0, 1.0, 80)
    X_df = np.column_stack([X["temperature"], X["wind_speed"]])
    m = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=0)
    m.fit(X_df, y)
    m.feature_names_in_ = np.array(["temperature", "wind_speed"])
    return m


# --- _get_feature_names / _tree_estimator ---


def test_get_feature_names_prefers_feature_names_attr():
    class M:
        feature_names_ = ["a", "b"]
        feature_names_in_ = ["z"]

    assert svc._get_feature_names(M()) == ["a", "b"]


def test_get_feature_names_unwraps_wrapper():
    inner = _fit_tiny_forest()
    wrapper = _WrapperWithInner(inner)
    assert svc._get_feature_names(wrapper) == ["temperature", "wind_speed"]


def test_get_feature_names_empty_when_unavailable():
    class M:
        pass

    assert svc._get_feature_names(M()) == []


def test_tree_estimator_unwraps_wrapper():
    inner = _fit_tiny_forest()
    assert svc._tree_estimator(_WrapperWithInner(inner)) is inner


def test_tree_estimator_returns_plain_model():
    inner = _fit_tiny_forest()
    assert svc._tree_estimator(inner) is inner


# --- explain_prediction ---


def test_explain_prediction_returns_real_shap_contributions():
    model = _fit_tiny_forest()
    features = {"temperature": 22.0, "wind_speed": 1.2}
    top = svc.explain_prediction(model, features)

    assert isinstance(top, list) and top
    assert len(top) <= 6
    first = top[0]
    assert set(first) == {
        "feature",
        "importance",
        "importance_pct",
        "direction",
        "value",
        "description",
    }
    total_pct = round(sum(t["importance_pct"] for t in top))
    assert total_pct > 0, "contributions must be real SHAP output"
    # wind_speed dominates a strongly wind-driven synthetic target
    assert first["feature"] == "wind_speed"
    assert first["direction"] in ("positive", "negative")
    assert first["value"] == features["wind_speed"]


def test_explain_prediction_raises_without_feature_names():
    class M:
        pass

    with pytest.raises(RuntimeError, match="no_tree_model"):
        svc.explain_prediction(M(), {"temperature": 1.0})


def test_explain_prediction_raises_when_shap_value_is_zero():
    M = _fit_tiny_forest()
    # Zero features column gives the tree no signal in either direction only
    # when the whole row is all-zeros and trees cannot move; simpler: monkeypatch.
    original = shap.TreeExplainer.shap_values
    try:
        shap.TreeExplainer.shap_values = lambda self, arr: np.zeros((1, 4))
        with pytest.raises(RuntimeError, match="no_variance"):
            svc.explain_prediction(M, {"temperature": 0.0, "wind_speed": 0.0})
    finally:
        shap.TreeExplainer.shap_values = original


def test_explain_fallback_never_fabricates():
    with pytest.raises(RuntimeError, match="no_model_for_explanation"):
        svc.explain_fallback({})


# --- generate_natural_language ---


def test_natural_language_unavailable_without_top_features():
    lines = svc.generate_natural_language({}, [])
    assert lines
    assert "no tree-based model could be loaded" in lines[0]


def test_natural_language_dominance_line():
    features = {"wind_speed": 1.2, "pbl_height": 250.0}
    top = [
        {
            "feature": "wind_speed",
            "description": "Wind speed. Low wind speed is limiting pollutant dispersion.",
        }
    ]
    lines = svc.generate_natural_language(features, top)
    assert lines
    assert lines[0].startswith("The dominant driver of this forecast is wind_speed (value 1.2)")
    assert any("Low wind speed of 1.2 m/s" in line for line in lines)
    assert any("compressed to about 250 m" in line for line in lines)


def test_natural_language_condition_lines():
    features = {
        "inversion_strength": 0.7,
        "fire_impact_score": 0.5,
        "humidity": 90.0,
        "stability_coupling_index": 0.7,
    }
    top = [{"feature": "humidity", "description": "Relative humidity"}]
    lines = svc.generate_natural_language(features, top)
    joined = "\n".join(lines)
    assert "strong atmospheric inversion layer" in joined
    assert "stubble burning" in joined
    assert "High humidity" in joined
    assert "Aerosol-PBL coupling is strong" in joined


def test_natural_language_skips_missing_conditions():
    features = {"wind_speed": 5.0, "pbl_height": 1200.0, "humidity": 50.0, "inversion_strength": 0.1}
    top = [{"feature": "temperature", "description": "Temperature"}]
    lines = svc.generate_natural_language(features, top)
    assert lines
    assert len(lines) == 1, f"expected only the dominance line, got: {lines}"


def test_natural_language_prediction_line():
    prediction = {
        "pm25_pred": 123.5,
        "pm10_pred": None,
        "o3_pred": 45.0,
        "no2_pred": 33.0,
        "aqi_pred": 168,
        "aqi_category": "Unhealthy",
        "dominant_pollutant": "pm25",
    }
    lines = svc.generate_natural_language({"humidity": 50.0}, [{"feature": "temperature"}], prediction)
    joined = "\n".join(lines)
    assert "PM25 123.5" in joined
    assert "AQI 168 (Unhealthy)" in joined
    assert "dominated by pm25" in joined


def test_natural_language_never_renders_literal_none():
    """No caller shape may surface the text "None" in the narrative.

    The real caller (``api/explanation.py``) passes only ``{"pm25_pred": ...}``,
    because the AQI there is a PM2.5-only sub-index and the other sub-indices are
    deliberately absent. Unguarded ``.get()`` calls turned those absences into the
    literal string "None", which the AI Explanation page rendered as
    "with AQI None (None), dominated by None.". The previous test only covered a
    fully populated prediction, so it could not catch this.
    """
    partial = {"pm25_pred": 87.4}
    lines = svc.generate_natural_language({"humidity": 50.0}, [{"feature": "temperature"}], partial)
    joined = "\n".join(lines)
    assert "PM25 87.4" in joined
    assert "None" not in joined
    assert "null" not in joined.lower()

    # Same guarantee for a prediction carrying an AQI but no category/dominant.
    mid = {"pm25_pred": 50.0, "aqi_pred": 120, "dominant_pollutant": "pm25"}
    joined_mid = "\n".join(
        svc.generate_natural_language({"humidity": 50.0}, [{"feature": "temperature"}], mid)
    )
    assert "None" not in joined_mid
    assert "AQI 120" in joined_mid

    # And when there is no scorable concentration at all, no line is invented.
    empty = {"aqi_pred": 100}
    joined_empty = "\n".join(
        svc.generate_natural_language({"humidity": 50.0}, [{"feature": "temperature"}], empty)
    )
    assert "None" not in joined_empty
    assert "Forecast concentrations" not in joined_empty
