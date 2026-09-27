"""Regression tests for the remaining stored-``PollutionReading.aqi`` readers.

The shared conftest seeds every ``aqi`` column with ``calculate_aqi(...)`` (see
``conftest._calculate_current_aqi``), so stored and recalculated AQI are identical
*by construction* in every other test. That makes the whole defect invisible to
the normal fixtures. Every test here therefore writes a deliberately corrupt
stored value first and asserts the endpoints ignore it.

Expected values are hand-derived from the CPCB India National AQI breakpoint
table rather than taken from the implementation, so the tests fail if the
calculation ever drifts:

    PM2.5  ug/m3   0-30 -> 0-50    31-60 -> 51-100   61-90 -> 101-200
                   91-120 -> 201-300                 121-250 -> 301-400
    O3     ug/m3   0-50 -> 0-50    51-100 -> 51-100  101-168 -> 101-200

    sub-index = (I_hi - I_lo) / (B_hi - B_lo) * (C - B_lo) + I_lo, rounded.

    PM2.5 = 77.0   (99/29) * 16 + 101 = 155.4 -> 156
    PM2.5 = 200.0  (99/129) * 79 + 301 = 361.6 -> 362
    PM2.5 = 10.0   (50/30) * 10 = 16.7 -> 17
    PM2.5 = 0.0    0 -> 0
    O3     = 140.0 (99/67) * 39 + 101 = 158.6 -> 159
    NH3/Pb only    no scorable pollutant -> category "Unknown" -> AQI is None
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.models.db_models import Forecast, PollutionReading, Station

# The corrupt sentinel that the pre-fix ingest wrote into the archive.
CORRUPT_AQI = 500


def _base_hour() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, minute=0, second=0, microsecond=0)


def _station(db, name: str) -> Station:
    station = db.query(Station).filter(Station.name == name).first()
    assert station is not None, f"seed station {name!r} is missing"
    return station


def _wipe(db) -> None:
    """Drop every seeded observation/forecast so each test controls its own."""
    db.query(Forecast).delete()
    db.query(PollutionReading).delete()
    db.commit()


def _add_reading(db, name: str, stored_aqi, *, hours_ago: int = 0, **conc) -> None:
    station = _station(db, name)
    db.add(PollutionReading(
        station_id=station.id,
        timestamp=_base_hour() - timedelta(hours=hours_ago),
        aqi=stored_aqi,
        pm25=conc.get("pm25"), pm10=conc.get("pm10"), o3=conc.get("o3"),
        no2=conc.get("no2"), so2=conc.get("so2"), co=conc.get("co"),
        nh3=conc.get("nh3"), pb=conc.get("pb"),
        data_source="test",
    ))
    db.commit()


def _add_forecast(db, name: str, pm25_pred, aqi_pred) -> None:
    station = _station(db, name)
    db.add(Forecast(
        station_id=station.id,
        forecast_timestamp=_base_hour(),
        horizon_hours=1,
        pm25_pred=pm25_pred,
        aqi_pred=aqi_pred,
        aqi_category="Moderate",
        dominant_pollutant="pm25",
    ))
    db.commit()


def _summary(client):
    response = client.get("/api/summary")
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# /api/summary
# --------------------------------------------------------------------------

def test_summary_ignores_a_corrupt_stored_aqi_column(client, db_session):
    """/summary reports the recalculated AQI, not the stored 500 sentinel."""
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=77.0)   # 156
    _add_reading(db_session, "RK Puram", CORRUPT_AQI, pm25=200.0)     # 362
    _add_reading(db_session, "ITO", CORRUPT_AQI, pm25=10.0)           # 17
    _add_reading(db_session, "Dwarka", CORRUPT_AQI, nh3=40.0, pb=2.0) # None

    body = _summary(client)

    assert body["worst_station"]["aqi"] == 362
    assert body["best_station"]["aqi"] == 17


def test_summary_aqi_category_and_dominant_pollutant_agree(client, db_session):
    """AQI, category and dominant pollutant must all come from one calculation.

    This is the self-consistency property: the dominant pollutant has to be the
    one that actually drives the reported AQI. A low-PM2.5 / high-O3 station
    proves the dominant pollutant is not just hard-coded to pm25.
    """
    _wipe(db_session)
    # PM2.5 sub-index 17, O3 sub-index 159 -> AQI 159 driven by O3.
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=10.0, o3=140.0)

    worst = _summary(client)["worst_station"]

    assert worst["aqi"] == 159
    assert worst["aqi_category"] == "Moderate"   # 101-200 band
    assert worst["dominant_pollutant"] == "o3"


def test_summary_aggregates_follow_the_recalculated_aqi(client, db_session):
    """ncr_avg_aqi / worst / best must all be derived from the corrected values."""
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=77.0)   # 156
    _add_reading(db_session, "RK Puram", CORRUPT_AQI, pm25=200.0)     # 362
    _add_reading(db_session, "ITO", CORRUPT_AQI, pm25=10.0)           # 17
    _add_reading(db_session, "Dwarka", CORRUPT_AQI, nh3=40.0, pb=2.0) # None

    body = _summary(client)

    # Only the three scorable stations count: (156 + 362 + 17) / 3 = 178.33.
    # Counting the unscorable station as 0 would give 133.8 instead.
    assert body["ncr_avg_aqi"] == pytest.approx(178.3)
    assert body["worst_station"]["name"] == "RK Puram"
    assert body["best_station"]["name"] == "ITO"
    assert body["stations_with_readings"] == 4


def test_summary_reports_none_not_zero_for_an_unscorable_station(client, db_session):
    """No scorable pollutant must yield ``None``, never a misleading AQI of 0.

    ``calculate_aqi()`` returns ``(0, "Unknown", "pm25")`` for an unscorable row,
    so the 0 has to be mapped to None or the station reads as "Good".
    """
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=77.0)            # 156
    _add_reading(db_session, "RK Puram", CORRUPT_AQI, nh3=40.0, pb=2.0)       # None

    body = _summary(client)

    # Average over the single scorable station (156), not (156 + 0) / 2 = 78.
    assert body["ncr_avg_aqi"] == pytest.approx(156.0)
    assert body["worst_station"]["name"] == "Anand Vihar"
    assert body["worst_station"]["aqi"] == 156


def test_summary_does_not_elect_an_unscorable_station_as_best(client, db_session):
    """An unscorable station must not win ``best_station`` by defaulting to 0."""
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=10.0)           # 17
    _add_reading(db_session, "RK Puram", CORRUPT_AQI, nh3=40.0, pb=2.0)       # None

    body = _summary(client)

    assert body["best_station"]["name"] == "Anand Vihar"
    assert body["best_station"]["aqi"] == 17


def test_summary_orders_a_legitimate_zero_aqi_above_an_unscorable_station(client, db_session):
    """A real AQI of 0 must not collapse to the None sentinel during ordering.

    ``max(..., key=lambda x: x.aqi or -1)`` turns a legitimate 0 into -1, tying
    with an unscorable station; the tie is then broken by alphabetical order and
    the unscorable station can be reported as the worst. Stations are iterated
    alphabetically (summary orders by name), and "Anand Vihar" precedes
    "RK Puram", so the unscorable station is seen first.
    """
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, nh3=40.0, pb=2.0)  # None
    _add_reading(db_session, "RK Puram", CORRUPT_AQI, pm25=0.0)             # 0

    body = _summary(client)

    assert body["worst_station"]["name"] == "RK Puram"
    assert body["worst_station"]["aqi"] == 0
    assert body["worst_station"]["aqi_category"] == "Good"
    assert body["ncr_avg_aqi"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# /api/forecast/comparison
# --------------------------------------------------------------------------

def _points(client, station="Anand Vihar", hours=72):
    response = client.get(f"/api/forecast/comparison/{station}", params={"hours": hours})
    assert response.status_code == 200, response.text
    return response.json()["points"]


def test_forecast_comparison_ignores_a_corrupt_stored_actual_aqi(client, db_session):
    """actual_aqi is recalculated instead of read from the stored column."""
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=77.0)   # 156
    _add_forecast(db_session, "Anand Vihar", pm25_pred=90.0, aqi_pred=200)

    point = _points(client)[0]

    assert point["actual_aqi"] == 156
    assert point["actual_aqi"] != CORRUPT_AQI


def test_forecast_comparison_uses_one_aqi_basis_for_actual_and_predicted(client, db_session):
    """Identical concentrations on both sides must produce an identical AQI.

    ``predicted_aqi`` is ``calculate_aqi`` of the predicted concentrations, so
    matching concentrations have to match exactly. Any drift between the two
    sides means one of them is using a different calculation.
    """
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=77.0)   # 156
    _add_forecast(db_session, "Anand Vihar", pm25_pred=77.0, aqi_pred=156)

    point = _points(client)[0]

    assert point["actual_aqi"] == point["predicted_aqi"] == 156


def test_forecast_comparison_reports_none_for_an_unscorable_actual(client, db_session):
    """An actual row with no scorable pollutant must yield ``actual_aqi=None``."""
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, nh3=40.0, pb=2.0)
    _add_forecast(db_session, "Anand Vihar", pm25_pred=90.0, aqi_pred=200)

    point = _points(client)[0]

    assert point["actual_aqi"] is None


def test_forecast_comparison_delta_is_unaffected_by_the_stored_aqi(client, db_session):
    """``delta`` is a PM2.5 difference and must not change with the AQI source."""
    _wipe(db_session)
    _add_reading(db_session, "Anand Vihar", CORRUPT_AQI, pm25=77.0)
    _add_forecast(db_session, "Anand Vihar", pm25_pred=90.0, aqi_pred=200)

    point = _points(client)[0]

    assert point["actual_pm25"] == 77.0
    assert point["predicted_pm25"] == 90.0
    assert point["delta"] == pytest.approx(13.0)
