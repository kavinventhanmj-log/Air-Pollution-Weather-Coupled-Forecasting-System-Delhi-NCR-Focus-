from datetime import datetime

import pytest
import requests
from app.database import DEFAULT_STATIONS
from app.services import cpcb_service
from app.services.cpcb_service import (
    CpcbError,
    NormalizedObservation,
    _parse_timestamp,
    _short_station_name,
    _to_float,
    normalize_records,
    upsert_ncr_data,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, exc=None):
        self.payload = payload
        self.status_code = status_code
        self.exc = exc

    def raise_for_status(self):
        if self.exc is not None:
            raise self.exc
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self.payload


def sample_record(**overrides):
    base = {
        "country": "India",
        "state": "Delhi",
        "city": "Delhi",
        "station": "Anand Vihar, Delhi - DPCC",
        "last_update": "09-09-2026 14:00:00",
        "latitude": "28.6492",
        "longitude": "77.2918",
        "pollutant_id": "PM2.5",
        "min_value": "7",
        "max_value": "35",
        "avg_value": "22",
    }
    base.update(overrides)
    return base


def fake_settings(**overrides):
    class _S:
        data_gov_api_key = overrides.get("data_gov_api_key", "test-key")
        data_gov_api_url = "https://api.data.gov.in"
        data_gov_resource_id = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
        data_gov_ncr_cities = overrides.get("data_gov_ncr_cities", "Delhi,Noida")
        data_gov_timeout = 5
    return _S()


# --------------------------------------------------------------------------- helpers


def test_to_float_conversions():
    assert _to_float("7") == 7.0
    assert _to_float("22.5") == 22.5
    assert _to_float(None) is None
    assert _to_float("NA") is None
    assert _to_float("") is None
    assert _to_float("abc") is None
    assert _to_float("NaN") is None


def test_parse_timestamp():
    ts = _parse_timestamp("09-09-2026 14:00:00")
    assert ts is not None
    assert ts.hour == 14
    assert ts.strftime("%d-%m-%Y %H:%M:%S") == "09-09-2026 14:00:00"
    assert _parse_timestamp("NA") is None
    assert _parse_timestamp("not a date") is None
    assert _parse_timestamp("") is None


def test_short_station_name():
    assert _short_station_name("Anand Vihar, Delhi - DPCC", "Delhi") == "Anand Vihar"
    assert _short_station_name("Sector-51, Gurugram - HSPCB", "Gurugram") == "Sector-51"
    assert _short_station_name("Knowledge Park - V, Greater Noida - UPPCB", "Greater Noida") == "Knowledge Park - V"
    assert _short_station_name("Single Name", "") == "Single Name"
    assert _short_station_name("Sector-62, Noida - UPPCB", "Noida") == "Sector-62"


# --------------------------------------------------------------------------- normalize_records


def test_normalize_merges_row_per_pollutant_into_one_observation():
    records = [
        sample_record(pollutant_id="PM2.5", avg_value="22"),
        sample_record(pollutant_id="PM10", avg_value="90"),
        sample_record(pollutant_id="NO2", avg_value="45"),
        sample_record(pollutant_id="SO2", avg_value="8"),
        sample_record(pollutant_id="CO", avg_value="1.2"),
        sample_record(pollutant_id="OZONE", avg_value="18"),
    ]
    obs = normalize_records(records)
    assert len(obs) == 1
    o = obs[0]
    assert o.station_name == "Anand Vihar"
    assert o.city == "Delhi"
    assert o.state == "Delhi"
    assert o.latitude == 28.6492
    assert o.longitude == 77.2918
    assert o.values == {"pm25": 22.0, "pm10": 90.0, "no2": 45.0, "so2": 8.0, "co": 1.2, "o3": 18.0}


def test_normalize_NA_values_stay_none_not_zero():
    records = [
        sample_record(pollutant_id="PM2.5", avg_value="NA", max_value="NA", min_value="NA"),
        sample_record(pollutant_id="PM10", avg_value="90"),
    ]
    obs = normalize_records(records)
    assert len(obs) == 1
    assert obs[0].values == {"pm10": 90.0}
    assert obs[0].values.get("pm25") is None


def test_normalize_retains_nh3_and_pb():
    """NH3 and Pb are published by the official feed, so they are kept.

    They are stored (nullable) even though they cannot be scored yet, because
    the repository has no verified CPCB breakpoint table for them. Dropping
    them here would make the data permanently unrecoverable.
    """
    records = [
        sample_record(pollutant_id="NH3", avg_value="33"),
        sample_record(pollutant_id="Pb", avg_value="0.5"),
        sample_record(pollutant_id="PM2.5", avg_value="10"),
    ]
    obs = normalize_records(records)
    assert len(obs) == 1
    assert obs[0].values == {"pm25": 10.0, "nh3": 33.0, "pb": 0.5}


def test_normalize_pb_alias_lead():
    """The feed spells it "Pb"; tolerate "Lead" too."""
    obs = normalize_records([sample_record(pollutant_id="Lead", avg_value="0.4")])
    assert obs[0].values == {"pb": 0.4}


def test_normalize_ignores_genuinely_unmapped_pollutants():
    """Non-criteria pollutants are still discarded, not invented."""
    records = [
        sample_record(pollutant_id="CO2", avg_value="400"),
        sample_record(pollutant_id="Benzene", avg_value="3"),
        sample_record(pollutant_id="PM2.5", avg_value="10"),
    ]
    obs = normalize_records(records)
    assert len(obs) == 1
    assert obs[0].values == {"pm25": 10.0}


def test_normalize_nh3_missing_stays_none_not_zero():
    records = [
        sample_record(pollutant_id="NH3", avg_value="NA", max_value="NA", min_value="NA"),
        sample_record(pollutant_id="PM2.5", avg_value="10"),
    ]
    obs = normalize_records(records)
    assert obs[0].values == {"pm25": 10.0}
    assert obs[0].values.get("nh3") is None


def test_normalize_value_fallback_min_max_legacy_names():
    r = sample_record(pollutant_id="PM2.5", avg_value="NA", max_value="35", min_value="7")
    obs = normalize_records([r])
    assert obs[0].values == {"pm25": 35.0}  # max fallback
    r2 = sample_record(pollutant_id="PM2.5", avg_value="NA", max_value="NA", min_value="7")
    obs2 = normalize_records([r2])
    assert obs2[0].values == {"pm25": 7.0}  # min fallback


def test_normalize_legacy_pollutant_field_names():
    r = {
        "country": "India", "state": "Delhi", "city": "Delhi",
        "station": "ITO, Delhi - DPCC", "last_update": "09-09-2026 14:00:00",
        "pollutant_id": "PM2.5", "pollutant_min": "10", "pollutant_max": "40",
        "pollutant_avg": "25",
    }
    obs = normalize_records([r])
    assert obs[0].values == {"pm25": 25.0}


def test_normalize_groups_by_station_and_timestamp():
    records = [
        sample_record(pollutant_id="PM2.5", avg_value="10"),
        sample_record(pollutant_id="CO", avg_value="2.0", last_update="09-09-2026 13:00:00"),
    ]
    obs = normalize_records(records)
    assert len(obs) == 2
    assert {tuple(sorted(o.values.items())) for o in obs} == {
        (("pm25", 10.0),),
        (("co", 2.0),),
    }


def test_normalize_skips_rows_without_station_or_timestamp():
    records = [
        sample_record(station=""),
        sample_record(last_update="NA"),
        sample_record(),
    ]
    obs = normalize_records(records)
    assert len(obs) == 1


# --------------------------------------------------------------------------- fetch_ncr_records


def test_fetch_requires_api_key(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_api_key=""))
    with pytest.raises(CpcbError) as exc:
        cpcb_service.fetch_ncr_records()
    assert exc.value.kind == "missing_key"


def test_fetch_success_single_page(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    responses = iter([
        FakeResponse(payload={"status": "ok", "total": 1, "records": [sample_record()]}),
    ])
    monkeypatch.setattr(cpcb_service.requests, "get", lambda *a, **k: next(responses))
    records = cpcb_service.fetch_ncr_records(api_key="key")
    assert len(records) == 1


def test_fetch_paginates_until_short_page(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    full_page = [sample_record(station=f"S{i}", last_update="09-09-2026 14:00:00") for i in range(cpcb_service.PAGE_LIMIT)]
    pages = [
        FakeResponse(payload={"status": "ok", "records": full_page}),
        FakeResponse(payload={"status": "ok", "records": [sample_record()]}),
    ]
    monkeypatch.setattr(
        cpcb_service.requests,
        "get",
        lambda *a, **k: pages.pop(0) if pages else FakeResponse(payload={"status": "ok", "records": []}),
    )
    records = cpcb_service.fetch_ncr_records(api_key="key")
    assert len(records) == cpcb_service.PAGE_LIMIT + 1


def test_fetch_http_error(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    resp = FakeResponse(payload=None, status_code=403, exc=requests.exceptions.HTTPError(response=FakeResponse(status_code=403)))
    monkeypatch.setattr(cpcb_service.requests, "get", lambda *a, **k: resp)
    with pytest.raises(CpcbError) as exc:
        cpcb_service.fetch_ncr_records(api_key="key")
    assert exc.value.kind == "empty"
    assert "403" in exc.value.message


def test_fetch_connection_error(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    monkeypatch.setattr(cpcb_service, "RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(cpcb_service, "RETRY_DELAY", 0)
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("boom")
    monkeypatch.setattr(cpcb_service.requests, "get", boom)
    with pytest.raises(CpcbError) as exc:
        cpcb_service.fetch_ncr_records(api_key="key")
    assert exc.value.kind == "empty"
    assert "network" in exc.value.message


def test_fetch_invalid_json(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    monkeypatch.setattr(cpcb_service, "RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(cpcb_service, "RETRY_DELAY", 0)
    def bad_json(*a, **k):
        raise ValueError("Expecting value")
    monkeypatch.setattr(cpcb_service.requests, "get", bad_json)
    with pytest.raises(CpcbError) as exc:
        cpcb_service.fetch_ncr_records(api_key="key")
    assert exc.value.kind == "empty"
    assert "invalid-json" in exc.value.message


def test_fetch_error_status_raises(monkeypatch):
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    resp = FakeResponse(payload={"status": "error", "message": "bad"}, status_code=200)
    monkeypatch.setattr(cpcb_service.requests, "get", lambda *a, **k: resp)
    with pytest.raises(CpcbError) as exc:
        cpcb_service.fetch_ncr_records(api_key="key")
    assert exc.value.kind == "empty"


def test_fetch_url_has_no_key_in_query_but_uses_params(monkeypatch):
    seen = {}
    monkeypatch.setattr(cpcb_service, "get_settings", lambda: fake_settings(data_gov_ncr_cities="Delhi"))
    monkeypatch.setattr(cpcb_service, "RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(cpcb_service, "RETRY_DELAY", 0)
    def capture(url, params=None, timeout=None, headers=None):
        seen["url"] = url
        seen["params"] = params
        return FakeResponse(payload={"status": "ok", "total": 1, "records": [sample_record()]})
    monkeypatch.setattr(cpcb_service.requests, "get", capture)
    cpcb_service.fetch_ncr_records(api_key="supersecret")
    assert "supersecret" not in seen["url"]
    assert seen["params"]["api-key"] == "supersecret"


# --------------------------------------------------------------------------- upsert_ncr_data


def _obs(ts_str, **values):
    return NormalizedObservation(
        station_name="Anand Vihar",
        city="Delhi",
        state="Delhi",
        latitude=28.6492,
        longitude=77.2918,
        timestamp=_parse_timestamp(ts_str),
        values=values,
    )


def test_upsert_inserts_station_and_reading(db_session):
    obs = _obs("09-09-2026 14:00:00", pm25=22.0, pm10=90.0)
    counters = upsert_ncr_data(db_session, [obs])
    assert counters["inserted"] == 1
    assert counters["station_created"] == 0  # Anand Vihar already seeded
    assert counters["station_updated"] == 1  # city "Delhi NCR" -> "Delhi"
    from app.models.db_models import PollutionReading
    rows = db_session.query(PollutionReading).filter(PollutionReading.station_id == 1).all()
    assert len(rows) == 12 + 1  # seeded 12 + one new


def test_upsert_is_idempotent_no_duplicates(db_session):
    obs = _obs("09-09-2026 14:00:00", pm25=22.0, pm10=90.0)
    upsert_ncr_data(db_session, [obs])
    second = upsert_ncr_data(db_session, [obs])
    assert second["inserted"] == 0
    assert second["skipped"] == 1
    from app.models.db_models import PollutionReading
    count = (
        db_session.query(PollutionReading)
        .filter(PollutionReading.station_id == 1, PollutionReading.timestamp == datetime(2026, 9, 9, 14, 0, 0))
        .count()
    )
    assert count == 1


def test_upsert_skips_unknown_station(db_session):
    # The station set is curated; unknown monitors are skipped, never created.
    obs = NormalizedObservation(
        station_name="Sector-62, Noida",
        city="Noida",
        state="Uttar Pradesh",
        latitude=28.5900,
        longitude=77.3260,
        timestamp=_parse_timestamp("09-09-2026 14:00:00"),
        values={"pm25": 55.0},
    )
    counters = upsert_ncr_data(db_session, [obs])
    assert counters["station_skipped"] == 1
    assert counters["inserted"] == 0
    from app.models.db_models import Station
    assert db_session.query(Station).count() == len(DEFAULT_STATIONS)


def test_upsert_persists_nh3_and_pb(db_session):
    """NH3/Pb must round-trip into the nullable DB columns."""
    from app.models.db_models import PollutionReading

    obs = _obs("09-09-2026 14:30:00", pm25=40.0, nh3=31.5, pb=0.42)
    assert upsert_ncr_data(db_session, [obs])["inserted"] == 1
    row = (
        db_session.query(PollutionReading)
        .filter(PollutionReading.timestamp == datetime(2026, 9, 9, 14, 30, 0))
        .first()
    )
    assert row is not None
    assert row.nh3 == 31.5
    assert row.pb == 0.42
    # NH3/Pb must not drag the AQI: only the six verified pollutants decide it.
    # pm25=40 -> 51 + 49/29 * 9 = 66.21 -> 66  (published 31-60 band)
    assert row.aqi == 66
    row_id = row.id
    db_session.delete(row)
    db_session.commit()
    assert row_id is not None


def test_upsert_nh3_only_reading_is_not_scored_but_is_stored(db_session):
    """A NH3-only reading must be stored, yet must not produce AQI 0 as 'Good'."""
    from app.models.db_models import PollutionReading

    obs = _obs("09-09-2026 15:00:00", nh3=250.0)
    assert upsert_ncr_data(db_session, [obs])["inserted"] == 1
    row = (
        db_session.query(PollutionReading)
        .filter(PollutionReading.timestamp == datetime(2026, 9, 9, 15, 0, 0))
        .first()
    )
    assert row is not None
    assert row.nh3 == 250.0
    assert row.pm25 is None and row.co is None
    assert row.aqi == 0  # nothing scorable -> 0/Unknown, never a real "Good"
    db_session.delete(row)
    db_session.commit()


def test_upsert_maps_dataset_monitor_names_to_curated_stations(db_session):
    from app.models.db_models import PollutionReading, Station

    aliases = {
        "IMD Lodhi Road": "Lodhi Road",
        "R K Puram": "RK Puram",
        "Dwarka-Sector 8": "Dwarka",
        "Sector - 62": "Noida Sector-62",
        "Sector 11": "Faridabad",
    }
    for source, canonical in aliases.items():
        obs = NormalizedObservation(
            station_name=source,
            city="Delhi",
            state="Delhi",
            latitude=28.0,
            longitude=77.0,
            timestamp=_parse_timestamp("09-09-2026 14:00:00"),
            values={"pm25": 40.0},
        )
        counters = upsert_ncr_data(db_session, [obs])
        assert counters["inserted"] == 1, source
        target = db_session.query(Station).filter(Station.name == canonical).first()
        assert target is not None, canonical
        row = db_session.query(PollutionReading).filter(PollutionReading.station_id == target.id).first()
        assert row is not None and row.pm25 == 40.0
        db_session.query(PollutionReading).filter(PollutionReading.station_id == target.id).delete()
        db_session.commit()
