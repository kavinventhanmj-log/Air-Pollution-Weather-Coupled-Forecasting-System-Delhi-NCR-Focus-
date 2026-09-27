from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.db_models import FireReading, Forecast, ModelMetrics, PollutionReading, Station
from ..schemas.schemas import StationAQISummary, SummaryResponse
from ..services import alert_service
from ..services.aqi_calculator import calculate_aqi

settings = get_settings()

router = APIRouter()


@router.get("/summary", response_model=SummaryResponse)
def get_summary(db: Session = Depends(get_db)):
    """Aggregate NCR-wide key performance indicators for the dashboard.

    Centre-stage summary of the current air-quality scenario: how many
    stations are reporting, the current network-average AQI, the worst and
    best station, live fire forcing, open alerts, and forecast/model
    coverage. Backed entirely by persisted service-state tables.

    Cached for 60 s — the per-station latest-reading scans are a few seconds
    on the pooled Neon Postgres.
    """
    from ..services.ttl_cache import cached

    return cached("summary", 60, lambda: _build_summary(db))


def _build_summary(db: Session) -> object:
    stations = db.query(Station).order_by(Station.name).all()

    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24)
    latest_by_station = {}
    for s in stations:
        r = (
            db.query(PollutionReading)
            .filter(PollutionReading.station_id == s.id, PollutionReading.timestamp >= since)
            .order_by(PollutionReading.timestamp.desc())
            .first()
        )
        if r is not None:
            latest_by_station[s.id] = r

    summaries = []
    for s in stations:
        r = latest_by_station.get(s.id)
        if r is None:
            continue
        aqi, category, dominant = calculate_aqi(
            pm25=r.pm25, pm10=r.pm10, o3=r.o3, no2=r.no2, so2=r.so2, co=r.co)
        summaries.append(StationAQISummary(
            name=s.name,
            aqi=None if category == "Unknown" else aqi,
            aqi_category=category,
            dominant_pollutant=dominant,
        ))

    aqi_values = [x.aqi for x in summaries if x.aqi is not None]
    worst = max(summaries, key=lambda x: -1 if x.aqi is None else x.aqi) if summaries else None
    best = min(summaries, key=lambda x: x.aqi if x.aqi is not None else 10**9) if summaries else None

    active_fires = (
        db.query(FireReading)
        .filter(FireReading.acq_date >= since)
        .count()
    )
    open_alerts = len(alert_service.all_station_alerts(db))
    models_trained = db.query(ModelMetrics).count()

    stations_with_forecast = (
        db.query(Forecast.station_id).distinct().count()
    )
    latest_forecast = db.query(Forecast).order_by(Forecast.forecast_timestamp.desc()).first()

    # Demo hydration is checked FIRST on purpose. On the hosted demo both
    # DEMO_HYDRATE_EMPTY_DB and LIVE_REFRESH_ENABLED are set, and the
    # re-stamped archive rows are what actually surface as "current" readings —
    # so reporting "live" there would overstate the provenance. Order matters:
    # the more specific, more conservative mode wins.
    if getattr(settings, "demo_hydrate_empty_db", False):
        data_mode = "demo_seeded"
        data_mode_note = (
            "Demo-seeded mode (DEMO_HYDRATE_EMPTY_DB=true): stored historical "
            "CPCB/fire/weather records are re-stamped into the recent window so "
            "a fresh database renders a live-looking demo. Not real-time data."
        )
    elif getattr(settings, "live_refresh_enabled", False):
        data_mode = "live"
        data_mode_note = (
            "Live-refresh scheduler is enabled; observations are re-pulled from "
            "the upstream feed on a schedule."
        )
    else:
        data_mode = "static_archive"
        data_mode_note = "Historical archive only; no live refresh or demo re-stamping."

    return SummaryResponse(
        generated_at=datetime.now(UTC).replace(tzinfo=None),
        stations=len(stations),
        stations_with_readings=len(summaries),
        ncr_avg_aqi=round(float(sum(aqi_values) / len(aqi_values)), 1) if aqi_values else None,
        worst_station=worst,
        best_station=best,
        active_fires_24h=active_fires,
        open_alerts=open_alerts,
        models_trained=models_trained,
        forecast_coverage={
            "stations_with_forecast": stations_with_forecast,
            "latest_forecast_at": latest_forecast.forecast_timestamp if latest_forecast else None,
        },
        data_mode=data_mode,
        data_mode_note=data_mode_note,
    )
