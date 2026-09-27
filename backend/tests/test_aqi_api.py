"""API-contract tests for the AQI fix.

Requirement G: existing response fields must keep their names and types.
Requirement J: compare the real endpoint path before/after, especially CO.

The seeded fixture in conftest builds 12 hourly Anand Vihar readings whose CO
starts at 2.2 and rises by 0.1 an hour. Under the old gappy breakpoints the
values 2.2..2.4 fell in the (2, 2.1)-style holes in some bands; the CO range
(1, 2) / (2, 10) pair now interpolates continuously.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.models.db_models import PollutionReading, Station
from app.services.aqi_calculator import calculate_iaqi

LEGACY_FIELDS = {
    "station", "timestamp", "pm25", "pm10", "o3", "no2", "so2", "co",
    "aqi", "aqi_category", "dominant_pollutant",
}
NEW_FIELDS = {
    "nh3", "pb", "instantaneous_aqi", "aqi_basis", "sub_indices",
    "averaged_concentrations", "instantaneous_concentrations",
    "pollutant_units", "data_availability", "averaging_windows",
    "o3_averaging_basis",
}

#: CPCB's O3 rule is 8 h by default; an 8 h mean above 208 ug/m3 is scored from
#: the 1 h value instead. The other two are project reporting states.
O3_BASES = {"8h", "1h_fallback", "8h_fallback_unavailable", "unavailable"}


def _seed(db_session, readings, **overrides):
    """Replace Anand Vihar's history with `readings`, a list of hourly dicts."""
    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    db_session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    count = len(readings)
    defaults = dict(pm25=10.0, pm10=20.0, no2=10.0, so2=5.0, co=0.5)
    for i, row in enumerate(readings):
        db_session.add(PollutionReading(
            station_id=station.id,
            timestamp=base - timedelta(hours=count - 1 - i),
            **{**defaults, **row, **overrides},
        ))
    db_session.commit()
    return station


def _current(client, name="Anand%20Vihar"):
    response = client.get(f"/api/current/{name}")
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# G. Backwards compatibility
# ---------------------------------------------------------------------------


def test_legacy_fields_still_present(client, db_session):
    body = _current(client)
    assert LEGACY_FIELDS <= set(body)


def test_legacy_field_types_unchanged(client, db_session):
    body = _current(client)
    assert isinstance(body["station"], str)
    assert isinstance(body["aqi"], int)
    assert isinstance(body["aqi_category"], str)
    assert body["dominant_pollutant"] in {
        "pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb", None
    }
    for key in ("pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb"):
        assert body[key] is None or isinstance(body[key], (int, float))


def test_additive_fields_expose_averaging_provenance(client, db_session):
    body = _current(client)
    assert NEW_FIELDS <= set(body)
    assert body["aqi_basis"] in {"window_average", "instantaneous"}
    assert body["o3_averaging_basis"] in O3_BASES
    assert set(body["averaging_windows"]) == {
        "pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb"
    }
    assert body["averaging_windows"]["pm25"] == 24
    assert body["averaging_windows"]["o3"] == 8
    assert body["averaging_windows"]["co"] == 8
    assert body["pollutant_units"]["co"] == "mg/m3"
    assert body["pollutant_units"]["pm25"] == "ug/m3"


def test_sub_indices_match_the_reported_aqi(client, db_session):
    body = _current(client)
    sub = body["sub_indices"]
    assert sub, "at least one pollutant should be scorable"
    assert body["aqi"] == round(max(sub.values()))


def test_nh3_pb_reported_unavailable_not_scored(client, db_session):
    body = _current(client)
    assert "nh3" not in body["sub_indices"]
    assert "pb" not in body["sub_indices"]
    assert body["data_availability"]["nh3"] == (
        "official_cpcb_table_not_vendored_in_repository"
    )
    assert body["data_availability"]["pb"] == (
        "official_cpcb_table_not_vendored_in_repository"
    )


# ---------------------------------------------------------------------------
# D. The endpoint really uses trailing windows
# ---------------------------------------------------------------------------


def test_endpoint_uses_window_averages(client, db_session):
    body = _current(client)
    assert body["aqi_basis"] == "window_average"
    assert body["averaged_concentrations"] is not None
    assert body["instantaneous_concentrations"] is not None
    # The instantaneous reading differs from the 24-hour mean for PM2.5 because
    # the fixture ramps pm25 up by 6 ug/m3 each hour.
    assert body["averaged_concentrations"]["pm25"] != body["instantaneous_concentrations"]["pm25"]
    assert body["instantaneous_aqi"] != body["aqi"]


def test_averaging_shrinks_towards_the_window_mean(client, db_session):
    body = _current(client)
    instant_pm25 = body["instantaneous_concentrations"]["pm25"]
    avg_pm25 = body["averaged_concentrations"]["pm25"]
    # The fixture ramps pm25 upward as i grows while the timestamp goes *back*
    # in time, so the series descends: the latest row is the cleanest and the
    # 24-hour mean sits above it.
    assert instant_pm25 < avg_pm25


def test_instantaneous_fields_still_report_the_raw_reading(client, db_session):
    """The existing per-pollutant fields stay instantaneous, as before."""
    body = _current(client)
    latest = body["instantaneous_concentrations"]
    assert body["pm25"] == latest["pm25"]
    assert body["co"] == latest["co"]


def test_stale_station_falls_back_to_instantaneous(client, db_session):
    """With no history, the endpoint degrades to instantaneous, not a crash."""
    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    db_session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    db_session.add(PollutionReading(
        station_id=station.id,
        timestamp=datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0),
        pm25=60.0, pm10=120.0, o3=40.0, no2=30.0, so2=8.0, co=0.9, aqi=121,
    ))
    db_session.commit()

    body = _current(client)
    assert body["aqi_basis"] == "instantaneous"
    # No window mean exists, so the pollutant is simply absent - never back-filled
    # from the instantaneous value under a misleading key.
    assert body["averaged_concentrations"].get("pm25") is None
    assert body["instantaneous_concentrations"]["pm25"] == 60.0
    # `data_availability` lists exclusion reasons only, so a scored pollutant
    # is absent from it; NH3/Pb are present because they are excluded.
    assert "pm25" not in body["data_availability"]
    assert body["data_availability"]["nh3"] == (
        "official_cpcb_table_not_vendored_in_repository"
    )
    # 60 is the top of the published 31-60 band -> IAQI 100
    assert body["sub_indices"]["pm25"] == 100.0
    # pm10=120 -> 101 + 99/149*19 = 113.62 outranks pm25
    assert body["sub_indices"]["pm10"] == 113.6
    assert body["dominant_pollutant"] == "pm10"
    assert body["aqi"] == 114
    assert body["aqi"] != 121  # the stale stored column is not echoed back


def test_sparse_history_returns_null_averages_not_a_fabricated_mean(client, db_session):
    """Two readings cannot make a 24-hour PM average."""
    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    db_session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    for i, pm in enumerate((30.0, 60.0)):
        db_session.add(PollutionReading(
            station_id=station.id, timestamp=base - timedelta(hours=i),
            pm25=pm, pm10=100.0, o3=30.0, no2=20.0, so2=5.0, co=0.8,
        ))
    db_session.commit()

    body = _current(client)
    assert body["averaged_concentrations"].get("pm25") is None
    assert "pm25" not in body["sub_indices"]
    assert body["data_availability"]["pm25"] == "no_valid_observation"
    # 8-hour O3 with two readings is still valid.
    assert body["averaged_concentrations"]["o3"] == pytest.approx(30.0)


def test_window_mean_never_uses_rows_after_the_evaluation_instant(client, db_session):
    """The window is anchored on the reported timestamp and extends backwards."""
    body = _current(client)
    as_of = datetime.fromisoformat(body["timestamp"])
    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    rows = (
        db_session.query(PollutionReading)
        .filter(
            PollutionReading.station_id == station.id,
            PollutionReading.timestamp <= as_of,
            PollutionReading.timestamp >= as_of - timedelta(hours=24),
        )
        .order_by(PollutionReading.timestamp.desc())
        .all()
    )
    expected = sum(r.pm25 for r in rows) / len(rows)
    assert body["averaged_concentrations"]["pm25"] == pytest.approx(expected, abs=1e-6)
    assert body["timestamp"] == rows[0].timestamp.isoformat()


def test_future_rows_never_leak_into_the_window(client, db_session):
    """A row newer than the evaluated instant becomes the new anchor only.

    The endpoint anchors on the newest observation, so a later row legitimately
    becomes the new anchor; the guarantee is that the mean is built only from
    rows at or before that anchor - one spike cannot define a 24-hour average.
    """
    body = _current(client)
    as_of = datetime.fromisoformat(body["timestamp"])
    before_mean = body["averaged_concentrations"]["pm25"]
    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    db_session.add(PollutionReading(
        station_id=station.id, timestamp=as_of + timedelta(minutes=30),
        pm25=500.0, pm10=200.0, o3=60.0, no2=50.0, so2=10.0, co=1.2,
    ))
    db_session.commit()
    try:
        moved = _current(client)
        assert moved["averaged_concentrations"]["pm25"] > before_mean
        # 500 is the top breakpoint, but a single point cannot drag a 24-hour
        # mean of ~13 hourly readings all the way to 500.
        assert moved["averaged_concentrations"]["pm25"] < 500
        assert moved["aqi"] != 500
    finally:
        db_session.query(PollutionReading).filter(
            PollutionReading.station_id == station.id,
            PollutionReading.timestamp == as_of + timedelta(minutes=30),
        ).delete()
        db_session.commit()


# ---------------------------------------------------------------------------
# J. Before/after regression on the reported case
# ---------------------------------------------------------------------------


def test_co_in_the_500_range_no_longer_forces_aqi_500(client, db_session):
    """CO=1.05 mg/m3 is normal Delhi air and must not produce AQI 500.

    The old table stopped at CO 1.0 and returned 500 for everything above it.
    """
    from app.services.aqi_calculator import calculate_iaqi

    # Guard the underlying regression independently of the API.
    # CO is published to 1 decimal, so 1.05 truncates to 1.0 and stays inside
    # the published 0-1.0 "Good" band -> IAQI 50.
    assert calculate_iaqi("co", 1.05) == pytest.approx(50, abs=0.01)

    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    db_session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    for i in range(12):
        db_session.add(PollutionReading(
            station_id=station.id, timestamp=base - timedelta(hours=i),
            pm25=40.0, pm10=90.0, o3=30.0, no2=25.0, so2=6.0, co=1.05, aqi=500,
        ))
    db_session.commit()

    body = _current(client)
    assert body["aqi"] != 500
    assert body["sub_indices"]["co"] == pytest.approx(50, abs=0.1)
    assert body["dominant_pollutant"] == "pm10"
    assert body["aqi_category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor"}


def test_stored_legacy_aqi_500_does_not_override_the_computed_value(client, db_session):
    """Historical rows that already stored 500 must not persist that verdict."""
    station = db_session.query(Station).filter(Station.name == "Anand Vihar").first()
    db_session.query(PollutionReading).filter(
        PollutionReading.station_id == station.id
    ).delete()
    base = datetime.now(UTC).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    for i in range(12):
        db_session.add(PollutionReading(
            station_id=station.id, timestamp=base - timedelta(hours=i),
            pm25=40.0, pm10=90.0, o3=30.0, no2=25.0, so2=6.0, co=1.05,
            aqi=500,  # stale value written by the old buggy calculator
        ))
    db_session.commit()

    body = _current(client)
    # pm10=90 -> 51 + 49/50 * 40 = 90.2, and pm10 outranks pm25 (67).
    assert body["aqi"] == 90
    assert body["aqi"] != 500
    assert body["dominant_pollutant"] == "pm10"


# ---------------------------------------------------------------------------
# CPCB O3 rule: 8-hour mean above 208 ug/m3 -> score the 1-hour value instead
# ---------------------------------------------------------------------------


def test_api_scores_the_1h_value_when_the_8h_mean_exceeds_208(client, db_session):
    """Nine hourly readings: a 315 background and a final 38 after the episode
    collapsed. The 8 h mean is 284.2, so CPCB requires the 1 h value."""
    _seed(db_session, [{"o3": 315.0}] * 8 + [{"o3": 38.0}])

    body = _current(client)
    assert body["o3_averaging_basis"] == "1h_fallback"
    # The reported window mean stays the genuine 8 h mean, not the substituted
    # value, so a client can still see what triggered the rule.
    assert body["averaged_concentrations"]["o3"] == pytest.approx(2558 / 9)
    assert body["sub_indices"]["o3"] == pytest.approx(38.0)
    # Not the 8 h value: 284.2 would score in the 301-400 band.
    assert body["sub_indices"]["o3"] < 100
    assert body["dominant_pollutant"] == "o3"
    assert body["aqi"] == 38
    # The legacy raw field still carries the latest observation.
    assert body["o3"] == 38.0


def test_api_uses_the_8h_value_when_the_threshold_is_not_crossed(client, db_session):
    _seed(db_session, [{"o3": 150.0}] * 9)

    body = _current(client)
    assert body["o3_averaging_basis"] == "8h"
    assert body["averaged_concentrations"]["o3"] == pytest.approx(150.0)
    # `sub_indices` are reported rounded to one decimal, like every other
    # pollutant in this response.
    assert body["sub_indices"]["o3"] == round(calculate_iaqi("o3", 150.0), 1)
    # averaging_windows keeps advertising the normal 8 h period regardless.
    assert body["averaging_windows"]["o3"] == 8


def test_api_boundary_208_9_truncates_and_does_not_trigger(client, db_session):
    """208.9 truncates to 208, which is not *above* 208."""
    _seed(db_session, [{"o3": 208.9}] * 9)
    assert _current(client)["o3_averaging_basis"] == "8h"

    _seed(db_session, [{"o3": 209.0}] * 9)
    assert _current(client)["o3_averaging_basis"] == "1h_fallback"


def test_api_1h_value_substitutes_downwards_not_max(client, db_session):
    """8 h mean 300 with a 1 h value of 100: 100 must win, not max(300, 100).

    The two sit in different CPCB bands (209-748 against 51-100), so a
    ``max()`` implementation would be visibly wrong here rather than off by a
    fraction of a point.
    """
    background = (300.0 * 9 - 100.0) / 8
    _seed(db_session, [{"o3": background}] * 8 + [{"o3": 100.0}])

    body = _current(client)
    assert body["o3_averaging_basis"] == "1h_fallback"
    assert body["averaged_concentrations"]["o3"] == pytest.approx(300.0)
    assert body["sub_indices"]["o3"] == round(calculate_iaqi("o3", 100.0), 1)
    assert body["sub_indices"]["o3"] == 100.0
    # The 8-hour value would have stayed in the 301-400 band.
    assert calculate_iaqi("o3", 300.0) > 300.0


def test_api_8h_unavailable_does_not_score_o3_from_a_1h_value(client, db_session):
    """One valid reading cannot make an 8 h mean, so O3 is not scored at all."""
    _seed(db_session, [{"o3": None}] * 8 + [{"o3": 300.0}])

    body = _current(client)
    assert body["o3_averaging_basis"] == "unavailable"
    assert "o3" not in body["sub_indices"]
    assert body["averaged_concentrations"].get("o3") is None
    assert body["data_availability"]["o3"] == "no_valid_observation"


def test_api_both_periods_unavailable(client, db_session):
    _seed(db_session, [{"o3": None}] * 9)

    body = _current(client)
    assert body["o3_averaging_basis"] == "unavailable"
    assert "o3" not in body["sub_indices"]


def test_legacy_instantaneous_path_is_not_labelled_an_o3_fallback(client, db_session):
    """The sparse-station mechanism must never look like a CPCB substitution."""
    _seed(db_session, [{"o3": 500.0}])

    body = _current(client)
    assert body["aqi_basis"] == "instantaneous"
    assert body["o3_averaging_basis"] == "unavailable"
    assert body["o3"] == 500.0


def test_api_o3_fallback_leaves_other_pollutants_unchanged(client, db_session):
    """Only the O3 sub-index may move when the 208 trigger fires."""
    _seed(db_session, [{"o3": 60.0}] * 9, pm25=70.0, pm10=140.0, no2=55.0, so2=20.0, co=1.4)
    before = _current(client)
    assert before["o3_averaging_basis"] == "8h"

    _seed(db_session, [{"o3": 315.0}] * 8 + [{"o3": 38.0}],
          pm25=70.0, pm10=140.0, no2=55.0, so2=20.0, co=1.4)
    after = _current(client)
    assert after["o3_averaging_basis"] == "1h_fallback"

    for pollutant in ("pm25", "pm10", "no2", "so2", "co"):
        assert after["sub_indices"][pollutant] == before["sub_indices"][pollutant]
        assert after["averaged_concentrations"][pollutant] == (
            before["averaged_concentrations"][pollutant]
        )
    assert after["o3_averaging_basis"] != before["o3_averaging_basis"]


def test_api_o3_fallback_exposes_every_legacy_field(client, db_session):
    """A triggered episode must not drop or rename any pre-existing field."""
    _seed(db_session, [{"o3": 315.0}] * 8 + [{"o3": 38.0}])

    body = _current(client)
    assert LEGACY_FIELDS <= set(body)
    assert NEW_FIELDS <= set(body)
    assert body["aqi_category"] in {
        "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
    }
    assert body["aqi_basis"] == "window_average"
    assert body["averaging_windows"] == {
        "pm25": 24, "pm10": 24, "o3": 8, "no2": 24, "so2": 24, "co": 8,
        "nh3": 24, "pb": 24,
    }
