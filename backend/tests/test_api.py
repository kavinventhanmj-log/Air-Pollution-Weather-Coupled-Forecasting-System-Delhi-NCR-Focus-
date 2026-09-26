

from app.database import DEFAULT_STATIONS


def test_health(client, db_session):
    # /api/health is the minimal liveness contract
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # /health is the rich probe (used by Docker healthchecks)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "AeroCast-NCR API"
    assert body["database"] == "connected"


def test_health_reports_degraded_when_db_down(client, db_session, monkeypatch):
    monkeypatch.setattr("app.main.database_reachable", lambda: False)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "disconnected"


def test_data_quality(client, db_session):
    response = client.get("/api/data-quality")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["station_count"] == len(DEFAULT_STATIONS)
    assert "tables" in body
    assert "recommendations" in body
    assert body["tables"]["stations"]["total"] == len(DEFAULT_STATIONS)
    assert body["tables"]["pollution_observations"]["total"] == 12
    assert "Anand Vihar" in body["forecast_coverage"]


def test_get_stations(client, db_session):
    response = client.get("/api/stations")
    assert response.status_code == 200
    bodies = response.json()
    assert isinstance(bodies, list)
    assert len(bodies) == len(DEFAULT_STATIONS)
    names = {b["name"] for b in bodies}
    assert names == {s["name"] for s in DEFAULT_STATIONS}
    for b in bodies:
        assert {"id", "name", "latitude", "longitude", "city"} <= set(b)
        assert isinstance(b["latitude"], float)
        assert isinstance(b["longitude"], float)


def test_get_station_by_name(client, db_session):
    response = client.get("/api/stations/Anand Vihar")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Anand Vihar"
    assert abs(body["latitude"] - 28.6492) < 0.0001
    assert abs(body["longitude"] - 77.2918) < 0.0001


def test_get_station_not_found(client, db_session):
    response = client.get("/api/stations/Noida Sector 62")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_current_aqi_by_station(client, db_session):
    response = client.get("/api/current/Anand Vihar")
    assert response.status_code == 200
    body = response.json()
    assert body["station"] == "Anand Vihar"
    assert body["timestamp"] is not None
    assert body["pm25"] is not None and body["pm25"] > 0
    assert body["pm10"] is not None and body["pm10"] > 0
    assert isinstance(body["aqi"], int) and body["aqi"] > 0
    assert body["aqi_category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
    assert body["dominant_pollutant"] in {"pm25", "pm10", "o3", "no2", "so2", "co"}


def test_get_current_aqi_station_not_found(client, db_session):
    response = client.get("/api/current/Noida Sector 62")
    assert response.status_code == 404


def test_get_forecast_by_station(client, db_session):
    response = client.get("/api/forecast/Anand Vihar")
    assert response.status_code == 200
    bodies = response.json()
    assert isinstance(bodies, list)
    assert len(bodies) >= 12
    for b in bodies:
        assert {"timestamp", "horizon_hours", "pm25_pred", "pm10_pred", "aqi_pred", "aqi_category"} <= set(b)


def test_get_forecast_station_not_found(client, db_session):
    response = client.get("/api/forecast/Gurugram")
    assert response.status_code == 404


def test_get_forecast_ncr(client, db_session):
    response = client.get("/api/forecast/ncr")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == {s["name"] for s in DEFAULT_STATIONS}
    assert isinstance(body["Anand Vihar"], list)


def test_generate_forecast_default(client, db_session):
    response = client.post("/api/forecast/generate")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["station"] == "Anand Vihar"
    assert body["horizons"] == [1, 6, 12, 24, 48, 72]
    assert len(body["forecasts"]) == 6
    for f in body["forecasts"]:
        assert f["pm25_pred"] is not None and f["pm25_pred"] > 0
        assert f["aqi_pred"] is not None and f["aqi_pred"] > 0
        assert f["aqi_category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
        assert f["horizon_hours"] in body["horizons"]


def test_generate_forecast_for_station(client, db_session):
    response = client.post(
        "/api/forecast/generate",
        json={"station_name": "Dwarka", "horizons": [3, 24]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["station"] == "Dwarka"
    assert body["horizons"] == [3, 24]
    assert {f["horizon_hours"] for f in body["forecasts"]} == {3, 24}
    generated = client.get("/api/forecast/Dwarka")
    assert generated.status_code == 200
    assert len(generated.json()) >= 2


def test_generate_forecast_persists_alerts(client, db_session):
    """``POST /api/forecast/generate`` still appends to the ``alerts`` table for
    audit history, even though the served feed is now evaluated live."""
    from app.database import SessionLocal
    from app.models.db_models import Alert
    with SessionLocal() as session:
        before = session.query(Alert).count()
    client.post("/api/forecast/generate", json={"station_name": "Anand Vihar"})
    with SessionLocal() as session:
        assert session.query(Alert).count() > before


def test_generate_forecast_station_not_found(client, db_session):
    response = client.post("/api/forecast/generate", json={"station_name": "Noida"})
    assert response.status_code == 404


def test_generate_forecast_invalid_horizon(client, db_session):
    response = client.post("/api/forecast/generate", json={"horizons": [100]})
    assert response.status_code == 400
    response = client.post("/api/forecast/generate", json={"horizons": []})
    assert response.status_code == 400


def test_forecast_comparison(client, db_session):
    response = client.get("/api/forecast/comparison/Anand Vihar", params={"hours": 72})
    assert response.status_code == 200
    body = response.json()
    assert body["station"] == "Anand Vihar"
    assert len(body["points"]) >= 12
    for p in body["points"]:
        assert {"timestamp", "actual_aqi", "predicted_aqi", "actual_pm25", "predicted_pm25", "delta"} <= set(p)
        assert p["actual_aqi"] is not None
        assert p["predicted_aqi"] is not None


def test_forecast_comparison_not_found(client, db_session):
    response = client.get("/api/forecast/comparison/Noida")
    assert response.status_code == 404


def test_forecast_comparison_no_data(client, db_session):
    response = client.get("/api/forecast/comparison/Punjabi Bagh")
    assert response.status_code == 404


def test_get_weather(client, db_session):
    response = client.get("/api/weather/Anand Vihar")
    assert response.status_code == 200
    body = response.json()
    assert body["station"] == "Anand Vihar"
    assert body["temperature"] is not None
    assert body["wind_speed"] is not None
    assert body["pbl_height"] == 180.0


def test_get_weather_latest_across_stations(client, db_session):
    response = client.get("/api/weather/latest")
    assert response.status_code == 200
    bodies = response.json()
    assert isinstance(bodies, list)
    # only the seeded station has weather, but latest is still a list keyed per station
    assert len(bodies) == 1
    body = bodies[0]
    assert body["station"] == "Anand Vihar"
    assert body["station_id"] is not None
    assert body["latitude"] == 28.6492
    assert body["longitude"] == 77.2918
    assert body["timestamp"] is not None
    assert body["pbl_height"] == 180.0


def test_get_weather_not_found(client, db_session):
    response = client.get("/api/weather/Gurugram")
    assert response.status_code == 404


def test_get_weather_history(client, db_session):
    response = client.get("/api/weather/Anand Vihar/history", params={"hours": 24})
    assert response.status_code == 200
    bodies = response.json()
    assert isinstance(bodies, list)
    assert len(bodies) == 12
    for b in bodies:
        assert b["station"] == "Anand Vihar"
        assert b["pbl_height"] is not None


def test_get_weather_history_not_found(client, db_session):
    response = client.get("/api/weather/RK Puram/history")
    assert response.status_code == 404


def test_get_inversion(client, db_session):
    response = client.get("/api/inversion/Anand Vihar")
    assert response.status_code == 200
    body = response.json()
    assert body["station"] == "Anand Vihar"
    assert body["pbl_height"] == 180.0
    assert body["inversion_detected"] is True
    assert body["inversion_strength"] == "Moderate"


def test_get_inversion_reports_geometry_fields(client, db_session):
    """Layer geometry is always present in the contract, even when null."""
    body = client.get("/api/inversion/Anand Vihar").json()
    for field in (
        "inversion_base_height_m",
        "inversion_top_height_m",
        "inversion_thickness_m",
        "inversion_thickness_hpa",
    ):
        assert field in body
    # Seeded rows carry no pressure-level profile, so the PBL proxy is used and
    # must not invent a thickness.
    assert body["inversion_source"] == "pbl_proxy"
    assert body["inversion_thickness_m"] is None


def test_get_inversion_reports_duration_and_persistence(client, db_session):
    body = client.get("/api/inversion/Anand Vihar").json()
    assert body["inversion_duration_h"] >= 0.0
    assert body["inversion_persistence_window_h"] == 24
    assert body["inversion_measured_window_h"] > 0.0
    assert body["inversion_episode_samples"] == 12
    # 12 hourly readings is an 11 h span, which cannot support a 24 h claim and
    # the response must say so rather than implying a full day was observed.
    assert body["inversion_history_sufficient"] is False
    assert body["inversion_persistence"] is not None


def test_get_inversion_not_found(client, db_session):
    response = client.get("/api/inversion/Faridabad")
    assert response.status_code == 404


def test_get_fire_activity(client, db_session):
    response = client.get("/api/fire-activity")
    assert response.status_code == 200
    body = response.json()
    assert body["total_fires"] >= 4
    assert body["high_confidence_fires"] >= 2
    assert body["mean_frp"] > 0


def test_get_latest_fires(client, db_session):
    response = client.get("/api/fires/latest", params={"hours": 48})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 5
    assert body["region"]
    assert body["generated_at"]
    event = body["fires"][0]
    assert {"id", "latitude", "longitude", "acq_date", "confidence", "frp",
            "brightness", "satellite", "instrument", "daynight"} <= set(event)
    assert event["acq_date"]
    ids = [f["id"] for f in body["fires"]]
    assert ids == sorted(ids, reverse=True)  # newest first


def test_get_latest_fires_lookback_excludes_old(client, db_session):
    # Seeded fires are ~2h old -> a 1h look-back window must return nothing.
    response = client.get("/api/fires/latest", params={"hours": 1})
    assert response.status_code == 200
    assert response.json()["count"] == 0

    response = client.get("/api/fires/latest", params={"hours": 7 * 24, "limit": 2})
    assert response.status_code == 200
    assert len(response.json()["fires"]) <= 2


def test_get_latest_fires_query_validation(client, db_session):
    assert client.get("/api/fires/latest", params={"hours": 0}).status_code == 422
    assert client.get("/api/fires/latest", params={"hours": 721}).status_code == 422


def test_get_fire_transport_direction(client, db_session):
    response = client.get("/api/fire/transport")
    assert response.status_code == 200
    body = response.json()
    assert body["from_direction"] == "NW"
    assert body["to_direction"] == "SE"
    assert "→" in body["label"]
    assert body["wind_speed"] is not None
    assert body["basis"] == "live weather data"


def test_get_plume_risk(client, db_session):
    response = client.get("/api/plume-risk")
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert 0 <= body["risk_score"] <= 1
    assert body["fire_count"] >= 4
    assert "→" in body["transport_direction"]
    assert body["wind_speed"] > 0
    assert body["distance_nearest_fire"] > 0
    assert len(body["factors"]) >= 2


def test_get_explanation(client, db_session):
    response = client.get("/api/explanation/Anand Vihar")
    assert response.status_code == 200
    body = response.json()
    assert body["station"] == "Anand Vihar"
    assert body["prediction"]["pm25_pred"] is not None
    assert body["prediction"]["aqi_pred"] is not None
    assert len(body["top_features"]) >= 3
    prev_importance = float("inf")
    total_pct = 0.0
    for feature in body["top_features"]:
        assert {"feature", "importance", "direction", "description"} <= set(feature)
        importance = feature["importance"]
        assert isinstance(importance, (int, float)) and importance > 0
        assert importance <= prev_importance, "top_features must be sorted by descending importance"
        prev_importance = importance
        pct = feature.get("importance_pct")
        if pct is not None and isinstance(pct, (int, float)):
            assert 0 <= pct <= 100
            total_pct += pct
    assert len(body["natural_language"]) >= 1


def test_get_explanation_not_found(client, db_session):
    response = client.get("/api/explanation/Ghaziabad")
    assert response.status_code == 404


def test_get_alerts(client, db_session):
    response = client.get("/api/alerts")
    assert response.status_code == 200
    bodies = response.json()
    assert isinstance(bodies, list)
    assert len(bodies) >= 1
    for b in bodies:
        assert b["station"] == "Anand Vihar"
        assert b["alert_level"] in {"WATCH", "ADVISORY", "WARNING", "SEVERE"}
        assert b["title"]


def test_get_alerts_empty(client, db_session):
    """The feed is evaluated live per station, so an empty ``alerts`` table no
    longer means an empty response. A station with neither a forecast nor an
    observation has nothing to alert on and must return an empty list."""
    from app.database import SessionLocal
    from app.models.db_models import Alert, Forecast, PollutionReading
    with SessionLocal() as session:
        session.query(Alert).delete()
        session.query(Forecast).delete()
        session.query(PollutionReading).delete()
        session.commit()
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert response.json() == []


def test_get_alerts_covers_every_station(client, db_session):
    """Regression: the feed used to replay the persisted ``alerts`` table, which
    only ever held one station's rows, so 16 of 17 NCR stations were missing."""
    from app.database import SessionLocal
    from app.models.db_models import Station
    with SessionLocal() as session:
        session.query(Station).filter(Station.name == "Faridabad").delete()
        session.commit()
        session.add(Station(name="Faridabad", latitude=28.4089, longitude=77.3178))
        session.commit()
    response = client.get("/api/alerts")
    assert response.status_code == 200
    body = response.json()
    assert body, "expected the seeded stations to raise alerts"
    assert "Anand Vihar" in {a["station"] for a in body}


def test_get_alerts_station_filter(client, db_session):
    all_alerts = client.get("/api/alerts").json()
    assert all_alerts
    station = all_alerts[0]["station"]
    response = client.get("/api/alerts", params={"station": station})
    assert response.status_code == 200
    body = response.json()
    assert {a["station"] for a in body} == {station}


def test_get_alerts_unknown_station(client, db_session):
    response = client.get("/api/alerts", params={"station": "Not A Station"})
    assert response.status_code == 404


def test_get_model_metrics(client, db_session):
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    bodies = response.json()
    assert isinstance(bodies, list)
    assert len(bodies) == 2
    for b in bodies:
        assert b["model_name"] in {"xgboost", "random_forest"}
        assert b["pollutant"] in {"pm25", "pm10"}
        assert b["mae"] > 0
        assert 0 < b["r2"] <= 1


def test_save_model_metrics(client, db_session):
    payload = {
        "model_name": "xgboost",
        "pollutant": "no2",
        "horizon_hours": 48,
        "mae": 8.2,
        "rmse": 12.7,
        "r2": 0.91,
        "mape": 11.3,
    }
    response = client.post("/api/model/metrics", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] is not None
    assert body["model_name"] == "xgboost"
    assert body["pollutant"] == "no2"
    assert body["r2"] == 0.91

    fetched = client.get("/api/model/metrics").json()
    assert any(m["pollutant"] == "no2" and m["horizon_hours"] == 48 for m in fetched)


def test_save_model_metrics_validation(client, db_session):
    response = client.post("/api/model/metrics", json={"model_name": "", "pollutant": "pm2", "horizon_hours": 24})
    assert response.status_code == 400
    response = client.post("/api/model/metrics", json={"model_name": "nn", "pollutant": "pm2"})
    assert response.status_code in (400, 422)


def test_unknown_route(client, db_session):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404


def test_get_model_performance(client, db_session, monkeypatch, tmp_path):
    # measured values (this is a recorded dataset, not recomputed at request time)
    import json as _json

    ev = {
        "schema_version": 1,
        "target": "pm25",
        "model_dir": str(tmp_path),
        "data_source": "data/ml/training_dataset.csv",
        "generated_at": "2026-09-12T00:00:00Z",
        "feature_count": 44,
        "features": ["pm25_lag1", "hour_sin", "month_sin"],
        "horizons": [1, 6],
        "evaluated_models": ["persistence", "random_forest", "xgboost"],
        "split_type": "chronological",
        "split_ratios": [0.6, 0.2, 0.2],
        "split_ranges": {
            "train": {"start": "2024-01-01T00:00:00", "end": "2025-03-17T00:00:00", "n_rows": 51007},
            "validation": {"start": "2025-03-17T00:00:00", "end": "2025-08-10T00:00:00", "n_rows": 17002},
            "test": {"start": "2025-08-10T00:00:00", "end": "2026-09-08T00:00:00", "n_rows": 17002},
        },
        "results": [
            {
                "horizon_hours": h,
                "n_train": 51007,
                "n_val": 17002,
                "n_test": 16999,
                "test_period_start": "2025-08-10T00:00:00",
                "test_period_end": "2026-09-08T00:00:00",
                "metrics": {
                    "persistence": {"mae": 60.0 + h, "rmse": 90.0 + h, "r2": 0.5, "n": 16999},
                    "random_forest": {"mae": 50.0 + h, "rmse": 80.0 + h, "r2": 0.6, "n": 16999},
                    "xgboost": {"mae": 45.0 + h, "rmse": 75.0 + h, "r2": 0.7, "n": 16999},
                },
            }
            for h in (1, 6)
        ],
    }
    eval_json = tmp_path / "evaluation.json"
    eval_json.write_text(_json.dumps(ev), encoding="utf-8")
    monkeypatch.setenv("AEROCAST_PM25_MODEL_DIR", str(tmp_path))

    response = client.get("/api/model/performance")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["target"] == "pm25"
    assert body["feature_count"] == 44
    assert body["horizons"] == [1, 6]
    assert body["evaluated_models"] == ["persistence", "random_forest", "xgboost"]
    assert set(body["split_ranges"]) == {"train", "validation", "test"}
    assert body["split_ranges"]["test"]["n_rows"] == 17002
    assert len(body["results"]) == 2
    h1 = next(r for r in body["results"] if r["horizon_hours"] == 1)
    assert h1["metrics"]["xgboost"]["mae"] == 46.0
    assert h1["metrics"]["persistence"]["n"] == 16999


def test_model_performance_missing_returns_404(client, db_session, monkeypatch, tmp_path):
    monkeypatch.setenv("AEROCAST_PM25_MODEL_DIR", str(tmp_path))
    response = client.get("/api/model/performance")
    assert response.status_code == 404
