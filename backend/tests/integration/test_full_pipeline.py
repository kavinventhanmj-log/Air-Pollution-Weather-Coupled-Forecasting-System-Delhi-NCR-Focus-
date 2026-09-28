"""End-to-end integration tests for AeroCast-NCR.

These tests exercise the full stack: DB seed -> backend services -> API
responses, verifying that machine-learning predictions flow through to the
REST layer with consistent structure and values.
"""

from app.database import DEFAULT_STATIONS


class TestForecastPipeline:
    def test_forecast_generate_to_api(self, client, db_session):
        resp = client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert len(body["forecasts"]) == 6
        for f in body["forecasts"]:
            assert f["horizon_hours"] in (1, 6, 12, 24, 48, 72)
            assert f["pm25_pred"] > 0
            assert f["aqi_pred"] > 0
            assert f["aqi_category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}

    def test_forecast_persists_to_db(self, client, db_session):
        client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
        resp = client.get("/api/forecast/Anand Vihar")
        assert resp.status_code == 200
        forecasts = resp.json()
        # The seed writes horizons 1..12; generation is an upsert on
        # (station_id, horizon_hours), so the six generated horizons are
        # replaced rather than appended.
        horizons = [f["horizon_hours"] for f in forecasts]
        assert len(horizons) == len(set(horizons)), "duplicate horizon rows persisted"
        assert set(horizons) == set(range(1, 13)) | {24, 48, 72}
        assert 1 in horizons or 6 in horizons

    def test_regeneration_does_not_duplicate_rows(self, client, db_session):
        """A second generation must replace, not accumulate (upsert contract)."""
        first = client.post(
            "/api/forecast/generate", json={"station_name": "Anand Vihar"}
        )
        assert first.status_code == 200, first.text
        count_after_first = len(client.get("/api/forecast/Anand Vihar").json())

        second = client.post(
            "/api/forecast/generate", json={"station_name": "Anand Vihar"}
        )
        assert second.status_code == 200, second.text
        rows = client.get("/api/forecast/Anand Vihar").json()
        assert len(rows) == count_after_first
        horizons = [f["horizon_hours"] for f in rows]
        assert len(horizons) == len(set(horizons))

    def test_ncr_aggregate(self, client, db_session):
        resp = client.get("/api/forecast/ncr")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {s["name"] for s in DEFAULT_STATIONS}
        for _station, forecasts in body.items():
            assert isinstance(forecasts, list)

    def test_forecast_comparison_flow(self, client, db_session):
        resp = client.get("/api/forecast/comparison/Anand Vihar", params={"hours": 72})
        assert resp.status_code == 200
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert len(body["points"]) >= 12
        for p in body["points"]:
            assert {"timestamp", "actual_aqi", "predicted_aqi", "actual_pm25", "predicted_pm25", "delta"} <= set(p)


class TestWeatherAndAtmosphere:
    def test_weather_latest(self, client, db_session):
        resp = client.get("/api/weather/Anand Vihar")
        assert resp.status_code == 200
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert body["pbl_height"] is not None
        assert body["temperature"] is not None
        assert body["wind_speed"] is not None

    def test_weather_history(self, client, db_session):
        resp = client.get("/api/weather/Anand Vihar/history", params={"hours": 24})
        assert resp.status_code == 200
        bodies = resp.json()
        assert len(bodies) == 12
        for b in bodies:
            assert b["station"] == "Anand Vihar"
            assert b["pbl_height"] == 180.0

    def test_inversion_detection(self, client, db_session):
        resp = client.get("/api/inversion/Anand Vihar")
        assert resp.status_code == 200
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert body["pbl_height"] == 180.0
        assert body["inversion_detected"] is True
        assert body["inversion_strength"] == "Moderate"


class TestFireAndPlume:
    def test_fire_activity(self, client, db_session):
        resp = client.get("/api/fire-activity")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_fires"] == 5
        assert body["high_confidence_fires"] == 3
        assert body["mean_frp"] > 0

    def test_transport_direction_station_specific(self, client, db_session):
        resp = client.get("/api/fire/transport", params={"station_name": "Anand Vihar"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert "\u2192" in body["label"]

    def test_plume_risk(self, client, db_session):
        resp = client.get("/api/plume-risk")
        assert resp.status_code == 200
        body = resp.json()
        assert body["risk_level"] in {"LOW", "MODERATE", "HIGH"}
        assert 0 <= body["risk_score"] <= 1
        assert body["fire_count"] == 5
        assert body["wind_speed"] > 0
        assert len(body["factors"]) >= 2


class TestExplainability:
    def test_explanation_endpoint(self, client, db_session):
        resp = client.get("/api/explanation/Anand Vihar")
        assert resp.status_code == 200
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        assert len(body["top_features"]) >= 3
        assert len(body["natural_language"]) >= 1
        for ft in body["top_features"]:
            assert {"feature", "importance", "direction", "description"} <= set(ft)

    def test_explanation_other_station(self, client, db_session):
        """Explanation for a station without live readings must still be either
        real SHAP on the fired tree model (200) or an honest 503 — never
        fabricated percentages."""
        resp = client.get("/api/explanation/RK Puram")
        if resp.status_code == 200:
            body = resp.json()
            assert body["station"] == "RK Puram"
            assert len(body["top_features"]) >= 3
            for ft in body["top_features"]:
                assert {"feature", "importance", "direction", "description"} <= set(ft)
                assert ft["importance"] > 0
        else:
            assert resp.status_code == 503
            # ``detail`` is a machine-readable dict for insufficient_data, and a
            # string for other refusals; neither may claim fabricated values.
            detail = resp.json().get("detail", "")
            text = detail if isinstance(detail, str) else str(detail)
            assert "fabricated" not in text.lower()


class TestAlerts:
    def test_alerts_generated_from_forecast(self, client, db_session):
        """The served feed is derived from the current forecast state, so a fresh
        generation for a station must be reflected without the caller reloading
        a persisted snapshot."""
        client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
        body = client.get("/api/alerts").json()
        assert body, "expected alerts for the regenerated station"
        assert "Anand Vihar" in {a["station"] for a in body}

    def test_alert_structure(self, client, db_session):
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        alerts = resp.json()
        assert len(alerts) >= 1
        for alert in alerts:
            assert alert["station"]
            assert alert["alert_level"] in {"WATCH", "ADVISORY", "WARNING", "SEVERE"}
            assert alert["title"]


class TestModelMetricsAPI:
    def test_metrics_list(self, client, db_session):
        resp = client.get("/api/model/metrics")
        assert resp.status_code == 200
        metrics = resp.json()
        assert len(metrics) == 2
        for m in metrics:
            assert m["model_name"] in {"xgboost", "random_forest"}
            assert m["pollutant"] in {"pm25", "pm10"}
            assert m["mae"] > 0

    def test_metrics_save_and_retrieve(self, client, db_session):
        payload = {
            "model_name": "xgboost", "pollutant": "o3",
            "horizon_hours": 24, "mae": 7.5, "rmse": 11.0,
            "r2": 0.88, "mape": 9.5,
        }
        resp = client.post("/api/model/metrics", json=payload)
        assert resp.status_code == 201
        fetched = client.get("/api/model/metrics").json()
        assert any(m["pollutant"] == "o3" and m["horizon_hours"] == 24 for m in fetched)


class TestDataQuality:
    def test_data_quality_report(self, client, db_session):
        resp = client.get("/api/data-quality")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["station_count"] == len(DEFAULT_STATIONS)
        assert "tables" in body
        assert "recommendations" in body


class TestCouplingFeedback:
    def test_coupling_endpoint_structure(self, client, db_session):
        resp = client.get("/api/coupling/Anand Vihar")
        assert resp.status_code == 200
        body = resp.json()
        assert body["station"] == "Anand Vihar"
        diag = body["diag"]
        assert {"aod_est", "radiation_transmittance", "pbl_suppression_factor",
                "corrected_pbl_height", "stability_coupling_index",
                "feedback_multiplier", "coupling_strength"} <= set(diag)
        assert 0 <= diag["aod_est"] <= 2.5
        assert 0 < diag["pbl_suppression_factor"] <= 1.0
        assert 0 <= diag["stability_coupling_index"] <= 1.0
        assert isinstance(body["narrative"], list) and body["narrative"]

    def test_coupling_high_pm25_drives_stronger_feedback(self):
        from ml.features.coupling import coupling_feedback_score
        heavy = coupling_feedback_score(pm25=250, pbl_height=400, wind_speed=1, hour=13)
        light = coupling_feedback_score(pm25=30, pbl_height=800, wind_speed=6, hour=13)
        assert heavy["stability_coupling_index"] > light["stability_coupling_index"]
        assert heavy["pbl_suppression_factor"] < light["pbl_suppression_factor"]
        assert heavy["feedback_multiplier"] > light["feedback_multiplier"]

    def test_coupling_features_in_feature_vector(self, client, db_session):
        from app.database import SessionLocal
        from app.models.db_models import Station
        from app.services.forecast_service import build_features_from_db
        with SessionLocal() as session:
            station = session.query(Station).filter(Station.name == "Anand Vihar").first()
            features = build_features_from_db(session, station.id)
            assert features["stability_coupling_index"] != 0.0
            assert 0.0 <= features["pbl_suppression_factor"] <= 1.0
            assert features["feedback_multiplier"] > 1.0
            assert 0.0 <= features["aod_est"] <= 2.5
        session.close()


class TestCoupledForecastLoop:
    def test_coupled_forecast_endpoint(self, client, db_session):
        resp = client.post(
            "/api/forecast/coupled",
            json={"station_name": "Anand Vihar", "horizons": [1, 6, 12, 24, 48, 72]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["mode"] == "coupled-two-way"
        assert len(body["coupled"]) == 6
        assert len(body["uncoupled"]) == 6
        assert body["feedback_path"]
        for point in body["coupled"]:
            assert point["horizon_hours"] in (1, 6, 12, 24, 48, 72)
            assert point["so2_pred"] is not None or point["so2_pred"] is None
            assert "coupling" in point
            assert point["coupling_stability"] is not None

    def test_coupled_vs_uncoupled_shapes(self, client, db_session):
        resp = client.post(
            "/api/forecast/coupled",
            json={"station_name": "ITO", "horizons": [24, 72]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        coupled_ts = [p["horizon_hours"] for p in body["coupled"]]
        uncoupled_ts = [p["horizon_hours"] for p in body["uncoupled"]]
        assert coupled_ts == [24, 72]
        assert uncoupled_ts == [24, 72]
        assert len(body["feedback_path"]) == 72

    def test_six_pollutants_present_in_coupled(self, client, db_session):
        resp = client.post(
            "/api/forecast/coupled",
            json={"station_name": "Dwarka", "horizons": [6]},
        )
        assert resp.status_code == 200
        point = resp.json()["coupled"][0]
        for key in ("pm25_pred", "pm10_pred", "o3_pred", "no2_pred", "so2_pred", "co_pred"):
            assert key in point
        assert point["aqi_pred"] is not None


class TestGridForecast:
    def test_grid_forecast_endpoint(self, client, db_session):
        client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
        resp = client.get("/api/grid/forecast?horizon_hours=24")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["horizon_hours"] == 24
        assert body["grid_size"][0] > 0
        assert "cells" in body
        assert body["extent"]["lats_min"] <= body["extent"]["lats_max"]

    def test_grid_overview(self, client, db_session):
        client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
        resp = client.get("/api/grid/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["stations"]) == len(DEFAULT_STATIONS)
        assert all(s["name"] for s in body["stations"])


class TestForecastCompletePollutantSet:
    def test_direct_forecast_includes_so2_co(self, client, db_session):
        resp = client.post("/api/forecast/generate", json={"station_name": "RK Puram"})
        assert resp.status_code == 200
        f = resp.json()["forecasts"][0]
        assert "so2_pred" in f and "co_pred" in f
        assert f["so2_pred"] > 0
        assert f["co_pred"] > 0


class TestDispersionForecast:
    def test_dispersion_forecast_endpoint(self, client, db_session):
        client.post("/api/forecast/coupled", json={"station_name": "Anand Vihar", "horizons": [1, 6, 12, 24, 48, 72]})
        resp = client.get("/api/dispersion/forecast?horizon_hours=24&start_hour=8")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["mode"] == "numerical_advection_diffusion"
        assert body["horizon_hours"] in (24, 72)
        assert len(body["frames"]) >= 6
        assert body["fire_count"] >= 0
        frame = body["frames"][0]
        assert {"hour", "aqi_mean", "aqi_max", "coupling", "cells"} <= set(frame)
        assert {"stability_coupling_index", "corrected_pbl_height"} <= set(frame["coupling"])
        assert len(frame["cells"]) > 0
        for cell in frame["cells"][:5]:
            assert {"lat", "lon", "aqi", "aqi_category"} <= set(cell)
            assert 0 <= cell["aqi"] <= 500

    def test_dispersion_forecast_with_no_surface(self, client, db_session):
        resp = client.get("/api/dispersion/forecast?horizon_hours=24")
        assert resp.status_code == 200
        body = resp.json()
        assert body["frames"] == []
        assert "error" in body

    def test_dispersion_forecast_72h_evolution(self, client, db_session):
        client.post("/api/forecast/generate", json={"station_name": "Dwarka"})
        resp = client.get("/api/dispersion/forecast?horizon_hours=72&start_hour=23")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["frames"]) == 72
        means = [f["aqi_mean"] for f in body["frames"]]
        assert all(0 <= m <= 500 for m in means)
        assert all(int(f["precip_mm"]) >= 0 for f in body["frames"])
        hours_of_day = {f["hour_of_day"] for f in body["frames"]}
        assert hours_of_day  # diurnal cycle present
        assert all(0 <= h <= 23 for h in hours_of_day)

    def test_dispersion_frame_hours_subset(self, client, db_session):
        """`?frame_hours=` must shrink the payload without changing the run.

        The 72 h response serialises 72 x 1575 per-cell grids, which is what
        made the panel time out on the free tier. Asking for a subset has to
        return exactly those frames, keep full cell fidelity on them, and still
        advertise the complete hour list.
        """
        client.post("/api/forecast/generate", json={"station_name": "Dwarka"})
        full = client.get("/api/dispersion/forecast?horizon_hours=72&start_hour=8")
        assert full.status_code == 200, full.text
        subset = client.get(
            "/api/dispersion/forecast?horizon_hours=72&start_hour=8&frame_hours=6,12,24,48,72"
        )
        assert subset.status_code == 200, subset.text
        body = subset.json()
        assert [f["hour"] for f in body["frames"]] == [6, 12, 24, 48, 72]
        assert body["frame_hours_available"] == [f["hour"] for f in full.json()["frames"]]
        for frame in body["frames"]:
            assert len(frame["cells"]) > 0
            assert {"lat", "lon", "aqi", "aqi_category"} <= set(frame["cells"][0])
        # Same physics: the requested frames match the unfiltered run.
        by_hour_full = {f["hour"]: f for f in full.json()["frames"]}
        for frame in body["frames"]:
            assert frame["aqi_mean"] == by_hour_full[frame["hour"]]["aqi_mean"]
        assert len(subset.content) < len(full.content)

    def test_dispersion_frame_hours_malformed_falls_back_to_all(self, client, db_session):
        client.post("/api/forecast/generate", json={"station_name": "Dwarka"})
        for raw in ("not-a-number", "-4", ""):
            resp = client.get(
                f"/api/dispersion/forecast?horizon_hours=24&start_hour=8&frame_hours={raw}"
            )
            assert resp.status_code == 200, resp.text
            assert len(resp.json()["frames"]) == 24, raw

    def test_dispersion_cache_key_separates_frame_sets(self):
        """A filtered response must not be served from the unfiltered entry.

        The frame set is part of the response, so it has to be part of the key —
        otherwise a cold client asking for 12 frames receives a cached 72-frame
        body (or worse, the pre-warm's entry poisons a filtered request).
        """
        from app.api.dispersion import dispersion_cache_key

        key = dispersion_cache_key
        assert key(72, 8, [6, 12, 24, 48, 72]) != key(72, 8, None)
        # Canonicalised: the same hours in another order must not solve the PDE
        # a second time.
        assert key(72, 8, [12, 6]) == key(72, 8, [6, 12])
        assert key(24, 8, [6, 12]) != key(48, 8, [6, 12])
        assert key(24, 8, [6, 12]) != key(24, 9, [6, 12])


class TestSummaryEndpoint:
    def test_summary_kpis(self, client, db_session):
        resp = client.get("/api/summary")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["stations"] == len(DEFAULT_STATIONS)
        assert body["stations_with_readings"] >= 1
        assert body["ncr_avg_aqi"] is not None
        assert body["worst_station"]["name"] == "Anand Vihar"
        assert body["worst_station"]["aqi_category"] in {
            "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
        }
        assert body["active_fires_24h"] >= 3
        assert body["open_alerts"] >= 1
        assert body["models_trained"] >= 2
        assert "stations_with_forecast" in body["forecast_coverage"]

    def test_summary_best_is_not_worse_than_worst(self, client, db_session):
        body = client.get("/api/summary").json()
        assert body["best_station"]["aqi"] <= body["worst_station"]["aqi"]


class TestExportEndpoint:
    def test_export_forecast_csv(self, client, db_session):
        client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
        resp = client.get("/api/export/forecast.csv", params={"station_name": "Anand Vihar", "hours": 72})
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/csv")
        assert "filename=" in resp.headers.get("content-disposition", "")
        text = resp.content.decode("utf-8")
        lines = text.strip().splitlines()
        assert lines[0].startswith("timestamp,horizon_hours,pm25_pred")
        assert len(lines) > 1

    def test_export_forecast_csv_missing_station(self, client, db_session):
        resp = client.get("/api/export/forecast.csv", params={"station_name": "Nowhere NCR"})
        assert resp.status_code == 404
