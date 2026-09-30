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


def _naive_utc(value: datetime | None) -> datetime | None:
    """Normalise a stored timestamp to naive UTC.

    `PollutionReading.timestamp` is a timezone-aware column, so PostgreSQL hands
    back aware datetimes while SQLite hands back naive ones. The rest of this
    module does its arithmetic in naive UTC, so coerce before subtracting
    instead of assuming a single dialect's behaviour.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


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

    now = datetime.now(UTC).replace(tzinfo=None)

    # Each station's MOST RECENT stored observation, regardless of age.
    #
    # This deliberately does not filter on a recency window. The upstream CKAN
    # archive (services/refresh_service.py CKAN_BASE) stopped publishing on
    # 2025-12-31, so a 24 h window returned nothing for every station and blanked
    # the whole dashboard - even though ~70k historical rows per station were
    # sitting in the table. Reporting the newest observation we actually hold,
    # together with its age (see `observation_age_hours` below), is strictly more
    # informative than reporting nothing.
    latest_by_station = {}
    for s in stations:
        r = (
            db.query(PollutionReading)
            .filter(PollutionReading.station_id == s.id)
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
        aqi, category, dominant = calculate_aqi(pm25=r.pm25, pm10=r.pm10, o3=r.o3, no2=r.no2, so2=r.so2, co=r.co)
        summaries.append(
            StationAQISummary(
                name=s.name,
                aqi=None if category == "Unknown" else aqi,
                aqi_category=category,
                dominant_pollutant=dominant,
            )
        )

    aqi_values = [x.aqi for x in summaries if x.aqi is not None]
    worst = max(summaries, key=lambda x: -1 if x.aqi is None else x.aqi) if summaries else None
    best = min(summaries, key=lambda x: x.aqi if x.aqi is not None else 10**9) if summaries else None

    # Newest observation across the whole network, and how old it is. The
    # frontend renders this next to the AQI so a nine-month-old number can
    # never be mistaken for a live reading.
    latest_observation = db.query(PollutionReading).order_by(PollutionReading.timestamp.desc()).first()
    latest_observation_at = _naive_utc(latest_observation.timestamp if latest_observation else None)
    observation_age_hours = (
        round((now - latest_observation_at).total_seconds() / 3600.0, 1) if latest_observation_at is not None else None
    )

    # `active_fires_24h` is a genuinely 24-hour metric (FIRMS keeps publishing,
    # unlike the pollution archive), so it keeps its own window rather than
    # reusing the old pollution `since`.
    fires_since = now - timedelta(hours=24)
    active_fires = db.query(FireReading).filter(FireReading.acq_date >= fires_since).count()
    open_alerts = len(alert_service.all_station_alerts(db))
    models_trained = db.query(ModelMetrics).count()

    stations_with_forecast = db.query(Forecast.station_id).distinct().count()
    latest_forecast = db.query(Forecast).order_by(Forecast.forecast_timestamp.desc()).first()

    # Demo hydration is checked FIRST on purpose. On the hosted demo both
    # DEMO_HYDRATE_EMPTY_DB and LIVE_REFRESH_ENABLED are set, and the
    # re-stamped archive rows are what actually surface as "current" readings -
    # so reporting "live" there would overstate the provenance. Order matters:
    # the more specific, more conservative mode wins.
    #
    # Beyond that ordering, an enabled refresh scheduler is no longer taken as
    # proof that the data is current. It used to report a bare "live" whenever
    # LIVE_REFRESH_ENABLED was set, which stayed true even after the upstream
    # archive silently stopped publishing - the dashboard claimed live data while
    # showing nothing. A scheduler that is running but has not produced a
    # network observation inside the window is reported as "stale".
    stale_cutoff_hours = 24.0
    if getattr(settings, "demo_hydrate_empty_db", False):
        data_mode = "demo_seeded"
        data_mode_note = (
            "Demo-seeded mode (DEMO_HYDRATE_EMPTY_DB=true): stored historical "
            "CPCB/fire/weather records are re-stamped into the recent window so "
            "a fresh database renders a live-looking demo. Not real-time data."
        )
    elif observation_age_hours is None:
        data_mode = "empty"
        data_mode_note = "No pollution observations are stored for any station, so no network average can be computed."
    elif observation_age_hours > stale_cutoff_hours and getattr(settings, "live_refresh_enabled", False):
        # The scheduler is on but the newest observation is outside the window:
        # the upstream feed is not publishing. Say so instead of implying the
        # number below is current.
        data_mode = "stale"
        data_mode_note = (
            f"Live refresh is enabled, but the newest stored observation is "
            f"{observation_age_hours:.0f} h old (outside the "
            f"{stale_cutoff_hours:.0f} h window), so the upstream feed is not "
            f"currently publishing. The AQI below is the latest observation on "
            f"record, not a live reading."
        )
    elif getattr(settings, "live_refresh_enabled", False):
        data_mode = "live"
        data_mode_note = (
            "Live-refresh scheduler is enabled; observations are re-pulled from the upstream feed on a schedule."
        )
    else:
        data_mode = "static_archive"
        data_mode_note = (
            f"Historical archive only; no live refresh or demo re-stamping. "
            f"Newest observation is {observation_age_hours:.0f} h old."
            if observation_age_hours is not None
            else "Historical archive only; no live refresh or demo re-stamping."
        )

    return SummaryResponse(
        generated_at=now,
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
        latest_observation_at=latest_observation_at,
        observation_age_hours=observation_age_hours,
    )
