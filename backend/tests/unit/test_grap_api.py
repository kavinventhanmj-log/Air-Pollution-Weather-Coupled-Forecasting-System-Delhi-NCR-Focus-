"""End-to-end tests for the GRAP API endpoints."""


def test_grap_stages_matrix(client, db_session):
    response = client.get("/api/grap/stages")
    assert response.status_code == 200
    body = response.json()
    stages = body["stages"]
    assert [s["stage"] for s in stages] == [0, 1, 2, 3, 4]
    assert stages[0]["title"] == "Not invoked"
    assert stages[1]["aqi_range_low"] == 201
    assert stages[4]["aqi_range_high"] is None
    for s in stages:
        assert s["color"]
        assert s["summary"]
        assert s["measures"]


def test_grap_current_from_seeded_state(client, db_session):
    response = client.get("/api/grap/current")
    assert response.status_code == 200
    body = response.json()

    # Seeded data: Anand Vihar latest AQI 215 (Poor) -> Stage I invoked.
    # Latest row (i=0) is pm25=95 -> 201 + 99/29 * 4 = 214.66 -> 215, using the
    # published 91-120 band.
    assert body["status"] == "ACTIVE"
    assert body["stage"] == 1
    assert body["aqi"] == 215
    assert body["aqi_category"] == "Poor"
    assert body["dominant_pollutant"] == "pm25"
    assert body["measures"]
    assert "CAQM" in body["source"]
    # Seed weather is homogeneous (pbl 180 m) -> strong PBL-proxy inversion.
    assert body["inversion_strength"] is not None
    assert body["inversion_strength"] > 0.5
    # Seed fire mean FRP is ~66.6 MW.
    assert body["fire_mean_frp_mw"] is not None


def test_grap_station(client, db_session):
    response = client.get("/api/grap/Anand%20Vihar")
    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == 1
    assert body["aqi"] == 215
    assert body["aqi_category"] == "Poor"


def test_grap_station_matches_supplied_aqi(client, db_session):
    body = client.get("/api/grap/anand vihar").json()
    stage_one = client.get("/api/grap/stages").json()["stages"][1]
    assert body["title"] == stage_one["title"]


def test_grap_station_not_found(client, db_session):
    response = client.get("/api/grap/NoSuchPlace")
    assert response.status_code == 404
