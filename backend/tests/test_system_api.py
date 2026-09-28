def test_system_status(client):
    response = client.get("/api/system")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "AeroCast-NCR"
    assert body["database"] in {"connected", "disconnected"}
    assert "weather_forecast" in body["engines"]
    assert "ctm_hysplit" in body["engines"]
    assert "ctm_wrf_chem" in body["engines"]
    assert "imd" in body["engines"]
    assert "cpcb" in body["engines"]
    assert "firms" in body["engines"]
    for _key, engine in body["engines"].items():
        assert {"source", "status", "note"} <= set(engine)
    assert set(body["run_mode"]) >= {
        "environment",
        "live_refresh_enabled",
        "demo_hydrate_empty_db",
        "explanation",
    }
    # The schema version is read from the migration chain, not a hardcoded
    # string, so it cannot drift from the code that is actually running.
    assert isinstance(body["schema_version"], str)
    assert body["schema_version"]
    # The pre-warm state has to be readable over HTTP: it is the only way to tell
    # a working cache warm-up from a dead one without hand-timing requests, and
    # `setting`/`default_applied` say whether it is on because of the environment
    # default or because it was configured explicitly.
    assert isinstance(body["prewarm"]["enabled"], bool)
    assert body["prewarm"]["setting"] in {True, False, None}
    assert isinstance(body["prewarm"]["default_applied"], bool)
    assert body["prewarm"]["state"] in {"disabled", "scheduled", "running", "complete", "cancelled"}


def test_system_status_reports_the_prewarm_sweep(client, monkeypatch):
    from app.services import prewarm

    monkeypatch.setattr(
        prewarm, "_STATUS",
        {"enabled": True, "state": "complete", "entries_warmed": 13, "entries_failed": 0, "seconds": 71.4},
    )
    body = client.get("/api/system").json()
    assert body["prewarm"]["state"] == "complete"
    assert body["prewarm"]["entries_warmed"] == 13
    assert body["prewarm"]["seconds"] == 71.4


def test_system_status_reports_disconnected_db(client, monkeypatch):
    monkeypatch.setattr("app.api.system.database_reachable", lambda: False)
    response = client.get("/api/system")
    assert response.status_code == 200
    assert response.json()["database"] == "disconnected"


def test_engine_reports_never_leak_absolute_paths(client, monkeypatch):
    """CTM engine entries must not expose server filesystem layout.

    The old response embedded the absolute HYSPLIT binary path and the full
    WRF-Chem netCDF path, which told an anonymous caller where to keep digging.
    The sanitised response only carries leaf names and status/notes.
    """
    from app.config import Settings

    fake = Settings(environment="development", database_url="sqlite:///./x.db")
    fake.hysplit_home = r"C:\Users\someone\hysplit4"
    fake.hysplit_met_dir = r"C:\Users\someone\met"
    fake.wrf_output_dir = r"C:\Users\someone\wrfout"
    monkeypatch.setattr("app.config.get_settings", lambda: fake)
    body = client.get("/api/system").json()
    for key, engine in body["engines"].items():
        for field in ("status", "note", "detail"):
            value = (engine.get(field) or "").lower()
            assert "\\users\\" not in value, f"{key}.{field} leaked a path"
    assert body["engines"]["ctm_hysplit"]["status"] == "surrogate"
    assert body["engines"]["ctm_wrf_chem"]["status"] == "gated"
