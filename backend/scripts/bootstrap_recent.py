"""Demo bootstrap: restamp the newest observations onto the last 24h.

Live CPCB feeds can lag far behind the system clock (e.g. the opencity.in
resources end months earlier), which leaves the summary's 24-hour "live AQI"
block and the dashboard's current-reading panels empty even though the app is
fully functional. This script re-stamps each station's most recent observations
(pollution + weather) onto the last 24 hourly slots so the dashboard renders a
live-looking current scenario for a demo.

This is a presentation aid only — it is NOT part of the real ingestion path
(live refresh_service still writes genuine feed readings).

Usage:
    python -m backend.scripts.bootstrap_recent
"""

from datetime import UTC, datetime, timedelta

from backend.app.database import SessionLocal
from backend.app.models.db_models import (
    PollutionReading,
    Station,
    WeatherReading,
)


def _aligned_slots(n_hours: int, anchor_minute: int) -> list[datetime]:
    now = datetime.now(UTC).replace(tzinfo=None).replace(minute=anchor_minute, second=0, microsecond=0)
    return [now - timedelta(hours=h) for h in range(n_hours - 1, -1, -1)]


def bootstrap_pollution(db, anchor_minute: int) -> int:
    inserted = 0
    for s in db.query(Station).all():
        latest = (
            db.query(PollutionReading)
            .filter(PollutionReading.station_id == s.id)
            .order_by(PollutionReading.timestamp.desc())
            .first()
        )
        if latest is None:
            continue
        existing = {
            ts
            for (ts,) in db.query(PollutionReading.timestamp)
            .filter(
                PollutionReading.station_id == s.id,
                PollutionReading.timestamp >= datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24),
            )
            .all()
        }
        rows = []
        for ts in _aligned_slots(24, anchor_minute):
            if ts in existing:
                continue
            rows.append(PollutionReading(
                station_id=s.id,
                timestamp=ts,
                pm25=latest.pm25,
                pm10=latest.pm10,
                o3=latest.o3,
                no2=latest.no2,
                so2=latest.so2,
                co=latest.co,
                aqi=latest.aqi,
                data_source=latest.data_source,
                re_stamped=True,
            ))
            existing.add(ts)
        if rows:
            db.add_all(rows)
            db.commit()
            inserted += len(rows)
    return inserted


def bootstrap_weather(db, anchor_minute: int) -> int:
    inserted = 0
    for s in db.query(Station).all():
        latest = (
            db.query(WeatherReading)
            .filter(WeatherReading.station_id == s.id)
            .order_by(WeatherReading.timestamp.desc())
            .first()
        )
        if latest is None:
            continue
        existing = {
            ts
            for (ts,) in db.query(WeatherReading.timestamp)
            .filter(
                WeatherReading.station_id == s.id,
                WeatherReading.timestamp >= datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24),
            )
            .all()
        }
        rows = []
        for ts in _aligned_slots(24, anchor_minute):
            if ts in existing:
                continue
            rows.append(WeatherReading(
                station_id=s.id,
                timestamp=ts,
                temperature=latest.temperature,
                humidity=latest.humidity,
                pressure_msl=latest.pressure_msl,
                surface_pressure=latest.surface_pressure,
                wind_speed=latest.wind_speed,
                wind_direction=latest.wind_direction,
                precipitation=latest.precipitation,
                cloud_cover=latest.cloud_cover,
                pbl_height=latest.pbl_height,
            ))
            existing.add(ts)
        if rows:
            db.add_all(rows)
            db.commit()
            inserted += len(rows)
    return inserted


def main():
    db = SessionLocal()
    try:
        anchor_latest = (
            db.query(PollutionReading)
            .order_by(PollutionReading.timestamp.desc())
            .first()
        )
        anchor_minute = anchor_latest.timestamp.minute if anchor_latest else 0
        print(f"Anchor minute : {anchor_minute}")
        n_p = bootstrap_pollution(db, anchor_minute)
        n_w = bootstrap_weather(db, anchor_minute)
        print(f"Pollution rows re-stamped into last 24h : {n_p}")
        print(f"Weather rows re-stamped into last 24h   : {n_w}")
    finally:
        db.close()
    print("\nDemo bootstrap complete — dashboard now shows a live-looking 24h scenario.")


if __name__ == "__main__":
    main()
