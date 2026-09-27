"""End-to-end tests for the GRAP API endpoints."""

from datetime import UTC, datetime, timedelta

import pytest
from app.models.db_models import PollutionReading, Station

# The seeded latest Anand Vihar reading is pm25=95, pm10=180, o3=55, no2=80,
# so2=16.0, co=2.2. Its sub-indices, worked out by hand from the CPCB tables,
# are pm25 215, pm10 153, o3 55, no2 50, so2 20, co 102 - so the AQI is 215 and
# PM2.5 dominates. Several tests below corrupt the stored `aqi` column and
# expect exactly that number back.
SEEDED_AQI = 215


def _station(session):
    return session.query(Station).filter(Station.name == "Anand Vihar").first()


def _latest(session):
    station = _station(session)
    return (
        session.query(PollutionReading)
        .filter(PollutionReading.station_id == station.id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )


def _corrupt_stored_aqi(session, value=500):
    """Overwrite the denormalised `aqi` column on the latest reading.

    The seeded value is derived from `calculate_aqi` itself (see conftest), so
    stored and recalculated agree by construction and the column's corruption
    cannot surface unless a test writes a value the current calculator would
    never produce. 500 is exactly such a value: the pre-fix gappy breakpoints
    fell through to it for 9.4% of the shipped archive.
    """
    reading = _latest(session)
    reading.aqi = value
    session.commit()
    return value


def _reseed(session, o3_series, **overrides):
    """Replace Anand Vihar's readings with one O3 series, oldest first.

    Every other pollutant is left clean so the O3 sub-index is the AQI, which
    keeps the expected values derivable by hand.
    """
    station = _station(session)
    session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    base = (
        datetime.now(UTC)
        .replace(tzinfo=None)
        .replace(minute=0, second=0, microsecond=0)
    )
    count = len(o3_series)
    defaults = dict(pm25=10.0, pm10=20.0, no2=5.0, so2=2.0, co=0.4)
    for i, o3 in enumerate(o3_series):
        session.add(
            PollutionReading(
                station_id=station.id,
                timestamp=base - timedelta(hours=count - 1 - i),
                o3=o3,
                **{**defaults, **overrides},
            )
        )
    session.commit()


def _current(client, name="Anand%20Vihar"):
    response = client.get(f"/api/current/{name}")
    assert response.status_code == 200, response.text
    return response.json()


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


# ---------------------------------------------------------------------------
# Step 4: GRAP recomputes AQI instead of trusting the stored `aqi` column.
#
# `PollutionReading.aqi` is a denormalised cache written at ingest time. Rows
# ingested before the breakpoint fix hold the old gappy table's fallthrough - a
# spurious 500 for readings that fell in a band gap (O3 = 0.1 ug/m3 -> aqi 500).
# Averaging that column into the NCR figure escalated the GRAP stage on 12.7% of
# sampled archive days, so both GRAP endpoints now recompute from the stored
# concentrations, matching what `/api/current/{station}` has always done.
# ---------------------------------------------------------------------------


def test_grap_current_ignores_a_corrupt_stored_aqi(client, db_session):
    """A bogus 500 in the denormalised column must not reach the assessment."""
    assert _latest(db_session).aqi == SEEDED_AQI  # the fixture is self-consistent
    assert _corrupt_stored_aqi(db_session, 500) == 500
    assert _latest(db_session).aqi == 500  # ...and is genuinely corrupt now

    body = client.get("/api/grap/current").json()
    # pm25=95 -> PM2.5 band 91-120 -> 201 + 99/29 * 4 = 214.66 -> 215.
    assert body["aqi"] == SEEDED_AQI
    assert body["aqi"] != 500
    assert body["aqi_category"] == "Poor"
    assert body["dominant_pollutant"] == "pm25"
    assert body["stage"] == 1


def test_grap_current_and_station_agree_despite_a_corrupt_column(client, db_session):
    """The invariant that was broken: one endpoint read the column, one didn't.

    They are the same assessment at two granularities, so a single station's
    figure must be identical whether it is read alone or folded into the NCR
    mean.
    """
    _corrupt_stored_aqi(db_session, 500)

    ncr = client.get("/api/grap/current").json()
    one = client.get("/api/grap/Anand%20Vihar").json()
    assert ncr["aqi"] == one["aqi"] == SEEDED_AQI
    assert ncr["stage"] == one["stage"] == 1
    assert ncr["dominant_pollutant"] == one["dominant_pollutant"] == "pm25"
    assert ncr["aqi_category"] == one["aqi_category"] == "Poor"


def test_grap_current_does_not_over_escalate_on_a_bad_column(client, db_session):
    """Quantifies the severity: 500 is a Stage IV trigger, 215 is Stage I."""
    _corrupt_stored_aqi(db_session, 500)

    stages = client.get("/api/grap/stages").json()["stages"]
    stage_of_500 = next(
        s["stage"]
        for s in stages
        if s["aqi_range_low"] is not None
        and s["aqi_range_low"] <= 500 < (s["aqi_range_high"] or 10**9)
    )
    assert stage_of_500 == 4  # what the stale column would have produced

    body = client.get("/api/grap/current").json()
    assert body["aqi"] == SEEDED_AQI
    assert body["stage"] == 1
    assert body["stage"] != stage_of_500


def test_rising_o3_divergence_between_current_and_grap_is_intentional(
    client, db_session
):
    """8 h O3 mean of 150 against a latest reading of 300.

    `/current` scores the CPCB 8-hour mean: O3 band 101-168 gives
    101 + 99/67 * (150-101) = 173.4. GRAP scores the latest observed reading:
    band 209-748 gives 301 + 99/539 * (300-209) = 317.7 -> 318.

    The two disagree *on purpose*. `/current` is a window-mean surface and GRAP
    assesses the latest observed state, so they are not expected to match, and
    the O3 8h->1h substitution is a window-mean rule that is correctly absent
    from the instantaneous GRAP path. This test pins that intent so a future
    "make them consistent" change has to be deliberate.
    """
    # Nine hourly readings. The 8 h window is inclusive at both ends
    # (`aqi_averaging.rolling_means`), so it spans the whole series and averages
    # to (132 * 7 + 126 + 300) / 9 = 150, while the latest reading is 300.
    _reseed(db_session, [132.0] * 7 + [126.0, 300.0])

    current = _current(client)
    assert current["averaged_concentrations"]["o3"] == pytest.approx(150.0)
    assert current["o3_averaging_basis"] == "8h"
    assert current["sub_indices"]["o3"] == pytest.approx(173.4, abs=0.05)
    assert current["averaging_windows"]["o3"] == 8
    assert current["dominant_pollutant"] == "o3"
    # The raw latest observation is still reported verbatim alongside it.
    assert current["o3"] == 300.0

    grap = client.get("/api/grap/Anand%20Vihar").json()
    assert grap["aqi"] == 318
    assert grap["dominant_pollutant"] == "o3"


def test_grap_scores_every_pollutant_and_leaves_nh3_pb_unscored(client, db_session):
    """Recomputation must not drop a pollutant, nor start scoring NH3/Pb.

    PM2.5 has to win the seeded mix on its own merit (215 against pm10 153,
    co 102, o3 55, no2 50, so2 20), and an extreme Pb reading - retained at
    ingest but carrying no CPCB sub-index - must not be able to win instead.
    """
    _corrupt_stored_aqi(db_session, 500)
    body = client.get("/api/grap/current").json()
    assert body["aqi"] == SEEDED_AQI
    assert body["dominant_pollutant"] == "pm25"

    _reseed(
        db_session,
        [132.0] * 7 + [126.0, 300.0],
        nh3=900.0,
        pb=900.0,
    )
    after = client.get("/api/grap/Anand%20Vihar").json()
    assert after["aqi"] == 318
    assert after["dominant_pollutant"] == "o3"


def test_grap_reports_no_aqi_when_nothing_is_scorable(client, db_session):
    """A row with no usable pollutant is "no AQI", not an AQI of 0.

    Zero is a legitimate clean-air reading and reports "Good"; conflating it
    with absent data would assert Stage I non-invocation on missing evidence.
    """
    _reseed(
        db_session,
        [None] * 9,
        pm25=None,
        pm10=None,
        no2=None,
        so2=None,
        co=None,
    )

    station = client.get("/api/grap/Anand%20Vihar").json()
    assert station["aqi"] is None
    assert station["aqi_category"] is None
    assert station["stage"] == 0
    assert station["status"] == "NOT_INVOKED"

    ncr = client.get("/api/grap/current").json()
    assert ncr["aqi"] is None
    assert ncr["stage"] == 0
    assert ncr["status"] == "NOT_INVOKED"
