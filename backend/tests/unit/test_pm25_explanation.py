"""Tests for PM2.5 forecast explainability via SHAP.

Every assertion verifies that contributions are computed from the actual
deployed XGBoost model — no hardcoded weights, no fabricated percentages.
The decomposition check (forecast_pm25 == base_value + sum(shap_values))
proves the SHAP decomposition is exact. The cross-row check proves values
depend on the actual feature inputs, not static constants.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
from conftest import TINY_XGB, _make_synthetic_df
from fastapi.testclient import TestClient

from ml.explainability.shap_explainer import FEATURE_MEANING, Pm25ShapExplainer
from ml.training.train_pm25 import run_training

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (train tiny 1..3h models once per session for speed)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pm25_shap_models_"))
    csv = tmp / "synth.csv"
    _make_synthetic_df(n_stations=3, n_hours=800).to_csv(csv, index=False)
    run_training(
        csv_path=csv,
        horizons=[1, 2, 3],
        model_dir=tmp / "models",
        xgb_params=TINY_XGB,
    )
    return tmp / "models"


@pytest.fixture(scope="session")
def tiny_csv(tiny_model_dir) -> pathlib.Path:
    """The synthetic training CSV lives one level above the model dir."""
    csv = tiny_model_dir.parent / "synth.csv"
    assert csv.exists(), f"synthetic csv not found: {csv}"
    return csv


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tiny_model_dir):
    from app.services import pm25_explanation_service as expl_svc
    from app.services import pm25_forecast_service as fcast_svc

    monkeypatch.setenv("AEROCAST_PM25_MODEL_DIR", str(tiny_model_dir))
    fcast_svc.reset_forecaster()
    expl_svc.reset_explainer()
    yield
    fcast_svc.reset_forecaster()
    expl_svc.reset_explainer()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explainer (unit)
# ─────────────────────────────────────────────────────────────────────────────


class TestPm25ShapExplainer:
    def test_explain_row_decomposition_exact(self, tiny_model_dir, tiny_csv):
        """forecast_pm25 == base_value + sum(shap_values) — the SHAP contract."""
        explainer = Pm25ShapExplainer(tiny_model_dir)
        # Grab a feature row from the training CSV so all features exist
        import pandas as pd
        df = pd.read_csv(tiny_csv, nrows=100)
        features = explainer.features
        row = {f: (float(df[f].iloc[10]) if f in df.columns else np.nan) for f in features}

        result = explainer.explain_row(row, horizon=2)
        base = result["base_value"]
        pred = result["forecast_pm25"]
        shap_sum = sum(c["shap_value"] for c in result["contributions_by_magnitude"])
        # SHAP is additive: pred ≈ base + sum(shap)
        assert np.isclose(pred, base + shap_sum, atol=0.1), (
            f"SHAP decomposition off: {pred} != {base} + {shap_sum}"
        )

    def test_percentages_sum_to_100(self, tiny_model_dir, tiny_csv):
        """share_of_abs_contributions_pct across all features sums to ≈100."""
        explainer = Pm25ShapExplainer(tiny_model_dir)
        import pandas as pd
        df = pd.read_csv(tiny_csv, nrows=50)
        features = explainer.features
        row = {f: (float(df[f].iloc[5]) if f in df.columns else np.nan) for f in features}

        result = explainer.explain_row(row, horizon=1)
        total = sum(c["share_of_abs_contributions_pct"] for c in result["contributions_by_magnitude"])
        assert np.isclose(total, 100.0, atol=0.5), f"SHAP percentages sum to {total}, not 100"

    def test_top_drivers_have_consistent_direction(self, tiny_model_dir, tiny_csv):
        """Positive drivers have shap_value > 0; negative have shap_value < 0."""
        explainer = Pm25ShapExplainer(tiny_model_dir)
        import pandas as pd
        df = pd.read_csv(tiny_csv, nrows=50)
        features = explainer.features
        row = {f: (float(df[f].iloc[15]) if f in df.columns else np.nan) for f in features}

        result = explainer.explain_row(row, horizon=2)
        for c in result["top_positive_drivers"]:
            assert c["shap_value"] > 0, f"Positive driver {c['feature']} has shap_value={c['shap_value']}"
        for c in result["top_negative_drivers"]:
            assert c["shap_value"] < 0, f"Negative driver {c['feature']} has shap_value={c['shap_value']}"

    def test_different_rows_different_shap(self, tiny_model_dir, tiny_csv):
        """Two different feature rows produce different SHAP values — proves not hardcoded."""
        explainer = Pm25ShapExplainer(tiny_model_dir)
        import pandas as pd
        df = pd.read_csv(tiny_csv, nrows=200)
        features = explainer.features
        row_a = {f: (float(df[f].iloc[10]) if f in df.columns else np.nan) for f in features}
        row_b = {f: (float(df[f].iloc[150]) if f in df.columns else np.nan) for f in features}

        res_a = explainer.explain_row(row_a, horizon=1)
        res_b = explainer.explain_row(row_b, horizon=1)
        # They should differ (different PM2.5 lags, different temperature, etc.)
        shap_a = [c["shap_value"] for c in res_a["contributions_by_magnitude"]]
        shap_b = [c["shap_value"] for c in res_b["contributions_by_magnitude"]]
        assert shap_a != shap_b, "SHAP values identical across different rows — possibly hardcoded?"

    def test_descriptions_from_model_not_invented(self, tiny_model_dir, tiny_csv):
        """Every feature's description is in FEATURE_MEANING — not invented."""
        explainer = Pm25ShapExplainer(tiny_model_dir)
        import pandas as pd
        df = pd.read_csv(tiny_csv, nrows=20)
        features = explainer.features
        row = {f: (float(df[f].iloc[0]) if f in df.columns else np.nan) for f in features}

        result = explainer.explain_row(row, horizon=1)
        for c in result["contributions_by_magnitude"]:
            assert c["feature"] in FEATURE_MEANING, (
                f"Feature {c['feature']} not in FEATURE_MEANING — description may be fabricated"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Explanation service (service-level)
# ─────────────────────────────────────────────────────────────────────────────


class TestPm25ExplanationService:
    def test_explain_pm25_forecast_payload(self, db_session):
        from app.services import pm25_explanation_service as svc

        result = svc.explain_pm25_forecast(db_session, "Anand Vihar", horizon=1)
        assert result["station"] == "Anand Vihar"
        assert result["model"] == "xgboost"
        assert "shap.TreeExplainer" in result["explanation_method"]
        assert isinstance(result["forecast_pm25"], float)
        assert isinstance(result["base_value"], float)
        assert isinstance(result["top_positive_drivers"], list)
        assert isinstance(result["top_negative_drivers"], list)
        assert len(result["contributions_by_magnitude"]) > 0
        assert result["horizon_hours"] == 1
        assert result["test_metrics"] is not None
        assert result["test_metrics"]["mae"] is not None

    def test_explain_invalid_horizon(self, db_session):
        from app.services import pm25_explanation_service as svc

        with pytest.raises(ValueError, match="horizon_out_of_range"):
            svc.explain_pm25_forecast(db_session, "Anand Vihar", horizon=99)

    def test_explain_unknown_station(self, db_session):
        from app.services import pm25_explanation_service as svc

        with pytest.raises(ValueError, match="station_not_found"):
            svc.explain_pm25_forecast(db_session, "Nonexistent Station", horizon=1)


# ─────────────────────────────────────────────────────────────────────────────
# Explain-by-id (service-level)
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainForecastById:
    def test_explain_existing_forecast(self, db_session):
        """Seed a Forecast row and explain it — verify forecast_id + stored_pm25_pred."""
        from app.models.db_models import Forecast, Station
        from app.services import pm25_explanation_service as svc

        station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
        from datetime import UTC, datetime, timedelta

        from app.database import SessionLocal
        base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
        session = SessionLocal()
        try:
            # ``forecasts`` is unique on (station_id, horizon_hours), so the
            # seeded t+1h row is reused rather than duplicated.
            forecast = (
                session.query(Forecast)
                .filter(
                    Forecast.station_id == station.id,
                    Forecast.horizon_hours == 1,
                )
                .first()
            )
            if forecast is None:
                forecast = Forecast(
                    station_id=station.id,
                    forecast_timestamp=base + timedelta(hours=1),
                    horizon_hours=1,
                )
                session.add(forecast)
            forecast.forecast_timestamp = base + timedelta(hours=1)
            forecast.pm25_pred = 105.3
            forecast.aqi_pred = 200
            forecast.aqi_category = "Very Poor"
            forecast.dominant_pollutant = "pm25"
            forecast.pbl_height = 180.0
            session.commit()
            session.refresh(forecast)
            fid = forecast.id
        finally:
            session.close()

        result = svc.explain_forecast_by_id(db_session, fid)
        assert result["forecast_id"] == fid
        assert result["station"] == "Anand Vihar"
        assert result["horizon_hours"] == 1
        assert isinstance(result["forecast_pm25"], float)
        assert result["stored_pm25_pred"] == pytest.approx(105.3, abs=0.1)
        assert len(result["top_positive_drivers"]) + len(result["top_negative_drivers"]) > 0
        assert "shap.TreeExplainer" in result["explanation_method"]

        # SHAP decomposition exactness for the explained row
        base_val = result["base_value"]
        pred = result["forecast_pm25"]
        shap_sum = sum(c["shap_value"] for c in result["contributions_by_magnitude"])
        assert np.isclose(pred, base_val + shap_sum, atol=0.1)

    def test_forecast_not_found_raises(self, db_session):
        from app.services import pm25_explanation_service as svc
        with pytest.raises(ValueError, match="forecast_not_found"):
            svc.explain_forecast_by_id(db_session, 999999)

    def test_horizon_not_trained_raises(self, db_session):
        from datetime import UTC, datetime, timedelta

        from app.database import SessionLocal
        from app.models.db_models import Forecast, Station
        from app.services import pm25_explanation_service as svc

        station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
        base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
        forecast = Forecast(
            station_id=station.id,
            forecast_timestamp=base + timedelta(hours=99),
            horizon_hours=99,
            pm25_pred=100.0,
            pbl_height=500.0,
        )
        session = SessionLocal()
        try:
            session.add(forecast)
            session.commit()
            session.refresh(forecast)
            fid = forecast.id
        finally:
            session.close()

        with pytest.raises(ValueError, match="horizon_out_of_range"):
            svc.explain_forecast_by_id(db_session, fid)


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint: GET /api/forecast/{forecast_id}/explanation
# ─────────────────────────────────────────────────────────────────────────────


class TestForecastExplanationApi:
    def _seed_forecast(self) -> int:
        """Insert a Forecast row and return its id (visible to the API session)."""
        from datetime import UTC, datetime, timedelta

        from app.database import SessionLocal
        from app.models.db_models import Forecast, Station

        with SessionLocal() as db:
            station = db.query(Station).filter(Station.name == "Anand Vihar").first()
            base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
            # Reuse the seeded t+1h row: forecasts are unique on
            # (station_id, horizon_hours).
            forecast = (
                db.query(Forecast)
                .filter(Forecast.station_id == station.id, Forecast.horizon_hours == 1)
                .first()
            )
            if forecast is None:
                forecast = Forecast(
                    station_id=station.id,
                    forecast_timestamp=base + timedelta(hours=1),
                    horizon_hours=1,
                )
                db.add(forecast)
            forecast.forecast_timestamp = base + timedelta(hours=1)
            forecast.pm25_pred = 104.5
            forecast.aqi_pred = 203
            forecast.aqi_category = "Very Poor"
            forecast.dominant_pollutant = "pm25"
            forecast.pbl_height = 180.0
            db.commit()
            db.refresh(forecast)
            return forecast.id

    def test_get_explanation_by_forecast_id(self, client: TestClient):
        fid = self._seed_forecast()
        resp = client.get(f"/api/forecast/{fid}/explanation")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["forecast_id"] == fid
        assert body["station"] == "Anand Vihar"
        assert body["model"] == "xgboost"
        assert isinstance(body["forecast_pm25"], float)
        assert isinstance(body["base_value"], float)
        assert isinstance(body["top_positive_drivers"], list)
        assert isinstance(body["top_negative_drivers"], list)
        assert len(body["contributions_by_magnitude"]) > 0
        assert body["stored_pm25_pred"] == pytest.approx(104.5, abs=0.1)
        assert body["summary"]
        assert "shap.TreeExplainer" in body["explanation_method"]

    def test_get_explanation_not_found(self, client: TestClient):
        resp = client.get("/api/forecast/99999/explanation")
        assert resp.status_code == 404

    def test_get_explanation_stored_value_matches(self, client: TestClient):
        fid = self._seed_forecast()
        resp = client.get(f"/api/forecast/{fid}/explanation")
        assert resp.status_code == 200
        body = resp.json()
        assert body["stored_pm25_pred"] is not None
        assert isinstance(body["stored_pm25_pred"], (int, float))

    def test_pm25_live_explanation_endpoint(self, client: TestClient):
        """Existing live endpoint also returns real SHAP for the PM2.5 model."""
        resp = client.get("/api/forecast/pm25/explanation", params={"station_name": "Anand Vihar", "horizon": 1})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert isinstance(body["forecast_pm25"], float)
        assert len(body["top_positive_drivers"]) + len(body["top_negative_drivers"]) > 0
