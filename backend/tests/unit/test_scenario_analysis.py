"""Unit + API tests for the what-if scenario-analysis engine and endpoint.

The scenario engine is read-only by design. The strongest test in this file
snapshots every table before and after a full scenario run and asserts the
database is unchanged. The trained XGBoost artifacts are replaced by a
deterministic fake forecaster (a known linear function of the feature row) so
baseline-vs-scenario behaviour is exact and independent of the model files.
"""

from __future__ import annotations

from datetime import timedelta

from app.models.db_models import (
    Alert,
    FireReading,
    Forecast,
    ModelMetrics,
    PollutionReading,
    Station,
    WeatherReading,
)
from app.schemas.schemas import (
    FireActivityChange,
    InversionChange,
    ScenarioChanges,
    ScenarioPerturbation,
)
from app.services import pm25_forecast_service as p25
from app.services import scenario_service as svc

ALL_TABLES = (Station, PollutionReading, WeatherReading, FireReading, Forecast, Alert, ModelMetrics)


class FakeForecaster:
    """Deterministic stand-in for the PM2.5 forecaster.

    PM2.5 is a known linear function of the feature row so every scenario test
    can assert exact deltas:
        pm25 = 20 + 8*wind_speed + 30*inversion_detected
                 + 120*fire_impact_score + 0.02*pbl_height
    """

    is_available = True
    horizons = list(range(1, 73))
    coverage_target = 0.85
    uncertainty_method = "split-conformal"

    def forecast(self, feature_row, horizons, base_time):
        ws = svc._num(feature_row.get("wind_speed")) or 0.0
        inv = svc._num(feature_row.get("inversion_detected")) or 0.0
        fire = svc._num(feature_row.get("fire_impact_score")) or 0.0
        pbl = svc._num(feature_row.get("pbl_height")) or 0.0
        forecasts = []
        for h in horizons:
            pm = 20.0 + 8.0 * ws + 30.0 * inv + 120.0 * fire + 0.02 * pbl
            forecasts.append({
                "forecast_horizon": h,
                "timestamp": (base_time + timedelta(hours=h)).isoformat() + "Z",
                "predicted_pm25": round(pm, 2),
                "pm25_lower_bound": round(pm - 5.0, 2),
                "pm25_upper_bound": round(pm + 5.0, 2),
            })
        return {
            "strategy": "direct",
            "uncertainty_method": self.uncertainty_method,
            "coverage_target": self.coverage_target,
            "forecasts": forecasts,
        }

    def test_metrics(self, _horizon):
        return {"mae": 1.0, "rmse": 2.0, "r2": 0.9, "n": 100}


def install_fake_forecaster(monkeypatch):
    monkeypatch.setattr(p25, "get_pm25_forecaster", lambda: FakeForecaster())


def run_scenario(db, changes, hours=12, station="Anand Vihar"):
    return svc.run_scenario_analysis(db, station, hours, changes)


def _snapshot(db):
    """Serialize every row of every table (excluding SQLAlchemy internals)."""
    rows = []
    for model in ALL_TABLES:
        for row in db.query(model).all():
            d = {c: v for c, v in row.__dict__.items() if not c.startswith("_")}
            d["__table"] = model.__tablename__
            rows.append(d)
    return sorted(rows, key=lambda r: (r["__table"], str(r)))


def _counts(db):
    return {m.__tablename__: db.query(m).count() for m in ALL_TABLES}


def _repr(inst):
    return inst


# ---------------------------------------------------------------------------
# Service-level tests (real pipeline, fake forecaster)
# ---------------------------------------------------------------------------

class TestScenarioService:
    def test_response_is_explicitly_labeled_with_disclaimer(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(wind_speed=ScenarioPerturbation(mode="relative", value=2.0)),
        )
        assert res["label"] == svc.SCENARIO_LABEL == "SCENARIO ANALYSIS"
        assert "WHAT-IF" in res["disclaimer"]
        assert "NOT a causal estimate" in res["disclaimer"] or "causal" in res["disclaimer"]
        assert res["value_kinds"]["observed"].startswith("last stored")
        assert res["value_kinds"]["scenario"].startswith("model output on the feature row after ONLY")
        assert res["data_integrity"]["mode"] == "read_only"
        assert res["data_integrity"]["database_writes"] == 0
        assert res["data_integrity"]["scenario_overwrites_nothing"] is True
        assert res["station"] == "Anand Vihar"
        assert res["horizon_hours"] == 12
        assert res["served_horizons"] == list(range(1, 13))
        assert res["observed"]["pm25_last_observed_ugm3"] is not None

    def test_scenario_run_leaves_database_unchanged(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        before_rows, before_counts = _snapshot(db_session), _counts(db_session)
        run_scenario(
            db_session,
            ScenarioChanges(
                wind_speed=ScenarioPerturbation(mode="relative", value=2.0),
                inversion=InversionChange(detected=False, strength=0.1),
            ),
        )
        db_session.expire_all()
        assert _counts(db_session) == before_counts
        assert _snapshot(db_session) == before_rows

    def test_wind_speed_relative_change_shifts_forecast_exactly(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(wind_speed=ScenarioPerturbation(mode="relative", value=2.0)),
        )
        effects = {e["feature"]: e for e in res["input_changes"]}
        assert effects["wind_speed"]["baseline_value"] == 1.6
        assert effects["wind_speed"]["scenario_value"] == 3.2
        assert res["difference"]["mean_difference_pm25"] == 12.8
        assert res["difference"]["peak_difference_pm25"] == 12.8
        assert all(p["difference_pm25"] == 12.8 for p in res["difference"]["points"])
        vent = _repr(effects["ventilation_coefficient"])
        assert vent["baseline_value"] == 288.0 and vent["scenario_value"] == 576.0

    def test_wind_speed_relative_0_5_halves_the_input(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(wind_speed=ScenarioPerturbation(mode="relative", value=0.5)),
        )
        effects = {e["feature"]: e for e in res["input_changes"]}
        assert effects["wind_speed"]["scenario_value"] == 0.8
        assert res["difference"]["mean_difference_pm25"] == -6.4

    def test_wind_direction_delta_recomputes_alignment_and_encoding(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(wind_direction=ScenarioPerturbation(mode="absolute", value=90.0)),
        )
        effects = {e["feature"]: e for e in res["input_changes"]}
        assert effects["wind_direction"]["scenario_value"] == 90.0
        assert effects["wind_dir_sin"] and effects["wind_dir_cos"]
        assert any("wind-alignment fire terms" in n for n in res["notes"])

    def test_pbl_height_absolute_change_recomputes_ventilation_and_inversion(
        self, db_session, monkeypatch
    ):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(pbl_height=ScenarioPerturbation(mode="absolute", value=500.0)),
        )
        effects = {e["feature"]: e for e in res["input_changes"]}
        assert effects["pbl_height"]["baseline_value"] == 180.0
        assert effects["pbl_height"]["scenario_value"] == 500.0
        assert effects["ventilation_coefficient"]["baseline_value"] == 288.0
        assert effects["ventilation_coefficient"]["scenario_value"] == 800.0
        # Raising the PBL also re-derives the inversion indicator (PBL proxy),
        # which flips it off here; the model then responds to BOTH inputs.
        assert effects["inversion_detected"]["baseline_value"] == 1.0
        assert effects["inversion_detected"]["scenario_value"] == 0.0
        assert res["difference"]["mean_difference_pm25"] == -23.6

    def test_inversion_override_sets_detected_and_strength(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(inversion=InversionChange(detected=False, strength=0.2)),
        )
        effects = {e["feature"]: e for e in res["input_changes"]}
        assert effects["inversion_detected"]["baseline_value"] == 1.0
        assert effects["inversion_detected"]["scenario_value"] == 0.0
        assert effects["inversion_strength"]["scenario_value"] == 0.2
        assert res["difference"]["mean_difference_pm25"] == -30.0

    def test_fire_activity_multiplier_scales_only_frp_intensity_terms(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(fire_activity=FireActivityChange(multiplier=2.0)),
        )
        effects = {e["feature"]: e for e in res["input_changes"]}
        base = effects["fire_impact_score"]["baseline_value"]
        assert base == 0.5747
        double = effects["fire_impact_score"]["scenario_value"]
        assert base < double < 1.0  # logistic-normalized: increases but does not 2x
        assert effects["transport_risk"]["scenario_value"] > effects["transport_risk"]["baseline_value"]
        assert effects["stubble_impact_score"]["scenario_value"] > effects["stubble_impact_score"]["baseline_value"]
        assert set(effects) == {"fire_impact_score", "transport_risk", "stubble_impact_score"}
        assert "reflect REAL detected fires" in res["notes"][0]
        assert res["difference"]["mean_difference_pm25"] > 0.0

    def test_fire_activity_multiplier_is_monotonic_and_1x_is_neutral(
        self, db_session, monkeypatch
    ):
        install_fake_forecaster(monkeypatch)
        def impact(mult):
            res = run_scenario(db_session, ScenarioChanges(
                fire_activity=FireActivityChange(multiplier=mult)))
            return {e["feature"]: e["scenario_value"] for e in res["input_changes"]}

        one = impact(1.0)
        two = impact(2.0)
        four = impact(4.0)
        half = impact(0.5)
        assert one["fire_impact_score"] == 0.5747  # 1x leaves the row unchanged
        assert half["fire_impact_score"] < one["fire_impact_score"]
        assert one["fire_impact_score"] < two["fire_impact_score"] < four["fire_impact_score"]
        for f in ("fire_impact_score", "transport_risk", "stubble_impact_score"):
            assert one[f] == 0.5747 if f == "fire_impact_score" else True

    def test_forecast_points_carry_bounds_and_test_metrics(self, db_session, monkeypatch):
        install_fake_forecaster(monkeypatch)
        res = run_scenario(
            db_session,
            ScenarioChanges(wind_speed=ScenarioPerturbation(mode="relative", value=2.0)),
        )
        for pt in res["baseline_forecast"]["forecasts"]:
            assert pt["pm25_lower_bound"] < pt["predicted_pm25"] < pt["pm25_upper_bound"]
            assert pt["test_mae"] == 1.0 and pt["test_r2"] == 0.9
        for b, s, d in zip(
            res["baseline_forecast"]["forecasts"],
            res["scene_forecast"]["forecasts"],
            res["difference"]["points"],
            strict=True,
        ):
            assert d["baseline_pm25"] == b["predicted_pm25"]
            assert d["scenario_pm25"] == s["predicted_pm25"]
            assert d["difference_pm25"] == round(s["predicted_pm25"] - b["predicted_pm25"], 2)

    def test_no_changes_raises_value_error(self, db_session):
        try:
            run_scenario(db_session, ScenarioChanges())
        except ValueError as exc:
            assert "no_input_changes" in str(exc)
        else:
            raise AssertionError("expected ValueError for empty changes")

    def test_unknown_station_raises_value_error(self, db_session):
        try:
            run_scenario(db_session, ScenarioChanges(), station="No Such Station")
        except ValueError as exc:
            assert str(exc).startswith("station_not_found")
        else:
            raise AssertionError("expected ValueError for unknown station")

    def test_relative_change_without_baseline_raises_value_error(self, db_session):
        try:
            svc._resolve(None, ScenarioPerturbation(mode="relative", value=2.0))
        except ValueError as exc:
            assert "relative" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError when relative change has no baseline")

    def test_unavailable_models_raise_runtime_error(self, db_session, monkeypatch):
        class Unavailable(FakeForecaster):
            is_available = False

        monkeypatch.setattr(p25, "get_pm25_forecaster", lambda: Unavailable())
        changes = ScenarioChanges(wind_speed=ScenarioPerturbation(mode="relative", value=2.0))
        try:
            run_scenario(db_session, changes)
        except RuntimeError as exc:
            assert "not trained" in str(exc)
        else:
            raise AssertionError("expected RuntimeError when models are unavailable")

    def test_requested_horizon_beyond_trained_models_raises(self, db_session, monkeypatch):
        class ShortHorizon(FakeForecaster):
            horizons = list(range(1, 25))

        monkeypatch.setattr(p25, "get_pm25_forecaster", lambda: ShortHorizon())
        try:
            run_scenario(
                db_session,
                ScenarioChanges(wind_speed=ScenarioPerturbation(mode="relative", value=2.0)),
                hours=48,
            )
        except RuntimeError as exc:
            assert "cover only up to" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for horizon beyond model coverage")


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

WIND_CHANGE = {"wind_speed": {"mode": "relative", "value": 2.0}}


class TestScenarioApi:
    def test_post_scenario_analysis_for_named_station(self, client, db_session, monkeypatch, auth_headers):
        install_fake_forecaster(monkeypatch)
        resp = client.post(
            "/api/scenario/analysis",
            json={"station_name": "Anand Vihar", "hours": 12, "changes": WIND_CHANGE},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["label"] == "SCENARIO ANALYSIS"
        assert body["station"] == "Anand Vihar"
        assert body["data_integrity"]["mode"] == "read_only"
        assert len(body["served_horizons"]) == 12
        assert len(body["difference"]["points"]) == 12
        assert body["difference"]["mean_difference_pm25"] == 12.8
        assert {e["feature"] for e in body["input_changes"]} == {"wind_speed", "ventilation_coefficient"}
        first = body["difference"]["points"][0]
        assert first["baseline_pm25"] and first["scenario_pm25"] and first["difference_pm25"]

    def test_post_scenario_defaults_to_first_station(self, client, db_session, monkeypatch, auth_headers):
        install_fake_forecaster(monkeypatch)
        resp = client.post(
            "/api/scenario/analysis",
            json={"hours": 12, "changes": {"pbl_height": {"mode": "absolute", "value": 400.0}}},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["station"] == "Anand Vihar"

    def test_unknown_station_returns_404(self, client, db_session, monkeypatch, auth_headers):
        install_fake_forecaster(monkeypatch)
        resp = client.post(
            "/api/scenario/analysis",
            json={"station_name": "No Such Station", "hours": 12, "changes": WIND_CHANGE},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_empty_changes_returns_422(self, client, db_session, monkeypatch, auth_headers):
        install_fake_forecaster(monkeypatch)
        resp = client.post(
            "/api/scenario/analysis",
            json={"station_name": "Anand Vihar", "hours": 12, "changes": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert "no_input_changes" in resp.json()["detail"]

    def test_invalid_change_mode_returns_422(self, client, db_session, auth_headers):
        resp = client.post(
            "/api/scenario/analysis",
            json={"changes": {"wind_speed": {"mode": "bogus", "value": 3.0}}},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_hours_out_of_range_returns_422(self, client, db_session, auth_headers):
        resp = client.post(
            "/api/scenario/analysis",
            json={"hours": 0, "changes": WIND_CHANGE},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_unavailable_models_return_503(self, client, db_session, monkeypatch, auth_headers):
        class Unavailable(FakeForecaster):
            is_available = False

        monkeypatch.setattr(p25, "get_pm25_forecaster", lambda: Unavailable())
        resp = client.post(
            "/api/scenario/analysis",
            json={"station_name": "Anand Vihar", "hours": 12, "changes": WIND_CHANGE},
            headers=auth_headers,
        )
        assert resp.status_code == 503
        assert "not trained" in resp.json()["detail"]
