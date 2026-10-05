
from app.database import DEFAULT_STATIONS
from app.services import cpcb_service
from app.services.cpcb_service import CpcbError


def _records_fixture():
    return [
        {"state": "Delhi", "city": "Delhi", "station": "Anand Vihar, Delhi - DPCC",
         "last_update": "09-09-2026 14:00:00", "latitude": "28.6492", "longitude": "77.2918",
         "pollutant_id": "PM2.5", "min_value": "95", "max_value": "120", "avg_value": "98"},
        {"state": "Delhi", "city": "Delhi", "station": "Anand Vihar, Delhi - DPCC",
         "last_update": "09-09-2026 14:00:00", "latitude": "28.6492", "longitude": "77.2918",
         "pollutant_id": "PM10", "min_value": "180", "max_value": "220", "avg_value": "190"},
        {"state": "Delhi", "city": "Delhi", "station": "ITO, Delhi - DPCC",
         "last_update": "09-09-2026 14:00:00", "latitude": "28.629", "longitude": "77.241",
         "pollutant_id": "PM2.5", "min_value": "70", "max_value": "90", "avg_value": "75"},
        {"state": "Uttar Pradesh", "city": "Noida", "station": "Sector-62, Noida - UPPCB",
         "last_update": "09-09-2026 14:00:00", "latitude": "28.590", "longitude": "77.326",
         "pollutant_id": "CO", "min_value": "1", "max_value": "3", "avg_value": "1.8"},
    ]


def test_pollution_latest_seeded(client, db_session):
    response = client.get("/api/pollution/latest")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    row = body[0]
    assert row["station"] == "Anand Vihar"
    assert row["pm25"] is not None
    assert row["pm10"] is not None
    assert row["o3"] is not None
    assert row["no2"] is not None
    assert row["so2"] is not None
    assert row["co"] is not None
    assert {"station_id", "station", "city", "timestamp", "aqi"} <= set(row)


def test_pollution_stations(client, db_session):
    response = client.get("/api/pollution/stations")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(DEFAULT_STATIONS)
    for station in body:
        assert "state" in station


def test_pollution_history(client, db_session):
    response = client.get("/api/pollution/1/history")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 12
    timestamps = [row["timestamp"] for row in body]
    assert timestamps == sorted(timestamps, reverse=True)


def test_pollution_history_not_found(client, db_session):
    response = client.get("/api/pollution/999/history")
    assert response.status_code == 404


def test_pollution_ingest_mocked(client, db_session, monkeypatch, auth_headers):
    monkeypatch.setattr(cpcb_service, "fetch_ncr_records", lambda **k: _records_fixture())
    response = client.post("/api/pollution/ingest", headers=auth_headers)
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["records_fetched"] == 4
    assert summary["stations_processed"] == 3
    assert summary["inserted"] == 3
    assert summary["skipped"] == 0
    assert summary["errors"] == []

    latest = client.get("/api/pollution/latest").json()
    assert len(latest) == 3
    names = {row["station"] for row in latest}
    assert names == {"Anand Vihar", "ITO", "Noida Sector-62"}
    noida = next(row for row in latest if row["station"] == "Noida Sector-62")
    assert noida["state"] == "Uttar Pradesh"
    assert noida["co"] == 1.8


def test_pollution_ingest_is_idempotent(client, db_session, monkeypatch, auth_headers):
    monkeypatch.setattr(cpcb_service, "fetch_ncr_records", lambda **k: _records_fixture())
    first = client.post("/api/pollution/ingest", headers=auth_headers).json()
    second = client.post("/api/pollution/ingest", headers=auth_headers).json()
    assert first["inserted"] == 3
    assert second["inserted"] == 0
    assert second["updated"] == 0
    assert second["skipped"] == 3
    latest = client.get("/api/pollution/latest").json()
    assert len(latest) == 3


def test_pollution_ingest_missing_key(client, db_session, monkeypatch, auth_headers):
    def boom(api_key=None, cities=None):
        raise CpcbError("missing_key", "DATA_GOV_API_KEY is not configured.")
    monkeypatch.setattr(cpcb_service, "fetch_ncr_records", boom)
    response = client.post("/api/pollution/ingest", headers=auth_headers)
    assert response.status_code == 400
    assert "DATA_GOV_API_KEY" in response.json()["detail"]


def test_pollution_ingest_fetch_failure(client, db_session, monkeypatch, auth_headers):
    def boom(api_key=None, cities=None):
        raise CpcbError("empty", "No CPCB records returned (network=ConnectionError).")
    monkeypatch.setattr(cpcb_service, "fetch_ncr_records", boom)
    response = client.post("/api/pollution/ingest", headers=auth_headers)
    assert response.status_code == 502
    assert "No CPCB records" in response.json()["detail"]
