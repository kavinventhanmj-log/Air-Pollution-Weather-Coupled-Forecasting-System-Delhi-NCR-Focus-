import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import Alert, Forecast, PollutionReading, Station
from ..schemas.schemas import (
    ForecastComparisonPoint,
    ForecastComparisonResponse,
    ForecastContextResponse,
    ForecastGenerateRequest,
    ForecastGenerateResponse,
    ForecastPoint,
)
from ..services import alert_service, forecast_service
from ..services.aqi_calculator import calculate_aqi

logger = logging.getLogger("aerocast.forecast")

router = APIRouter()

def _round_hour(dt: datetime) -> datetime:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt.replace(minute=0, second=0, microsecond=0)

def _station_or_404(db: Session, station_name: str) -> Station:
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")
    return station

def _to_forecast_point(f) -> ForecastPoint:
    return ForecastPoint(
        timestamp=f.forecast_timestamp,
        horizon_hours=f.horizon_hours,
        pm25_pred=f.pm25_pred,
        pm10_pred=f.pm10_pred,
        o3_pred=f.o3_pred,
        no2_pred=f.no2_pred,
        so2_pred=f.so2_pred,
        co_pred=f.co_pred,
        aqi_pred=f.aqi_pred,
        aqi_category=f.aqi_category or "",
        dominant_pollutant=f.dominant_pollutant,
        coupling_stability=f.coupling_stability,
        coupling_mode=f.coupling_mode,
    )

@router.post("/forecast/coupled", response_model=dict)
def generate_coupled_forecast(
    req: ForecastGenerateRequest = ForecastGenerateRequest(),
    db: Session = Depends(get_db),
):
    """Run the time-stepped two-way coupled weather-chemistry forecast.

    Steps meteorology + chemistry forward hour-by-hour, correcting PBL height,
    temperature, and stability from the freshly forecast aerosol load, then
    persists the coupled forecast points. Returns both the coupled series and
    the direct (uncoupled) series for comparison, plus the feedback path.
    """
    if not req.horizons:
        raise HTTPException(status_code=400, detail="horizons must be a non-empty list")
    invalid = [h for h in req.horizons if not 1 <= h <= 72]
    if invalid:
        raise HTTPException(status_code=400, detail=f"invalid horizon values: {invalid}")

    if req.station_name:
        station = _station_or_404(db, req.station_name)
    else:
        station = db.query(Station).order_by(Station.id).first()
        if not station:
            raise HTTPException(status_code=503, detail="No stations available; seed the database first")

    horizons = list(dict.fromkeys(req.horizons))
    result = forecast_service.generate_coupled_forecast(db, station.id, horizons)
    rows = forecast_service.save_coupled_forecasts(db, station.id, result["coupled"])

    coverage = forecast_service.station_data_sufficiency(db, station.id)

    return {
        "station": station.name,
        "generated_at": datetime.now(UTC).replace(tzinfo=None),
        "horizons": horizons,
        "mode": "coupled-two-way",
        "coupled": result["coupled"],
        "uncoupled": result["uncoupled"],
        "feedback_path": result["feedback_path"],
        "saved_points": len(rows),
        "pooled_features": bool(coverage["pooled"]),
        "local_readings": int(coverage["local_readings"]),
        "history_days": coverage.get("history_days"),
    }

@router.post("/forecast/generate", response_model=ForecastGenerateResponse)
def generate_forecast(
    req: ForecastGenerateRequest = ForecastGenerateRequest(),
    db: Session = Depends(get_db),
):
    if not req.horizons:
        raise HTTPException(status_code=400, detail="horizons must be a non-empty list")
    invalid = [h for h in req.horizons if not 1 <= h <= 72]
    if invalid:
        raise HTTPException(status_code=400, detail=f"invalid horizon values: {invalid}")

    if req.station_name:
        station = _station_or_404(db, req.station_name)
    else:
        station = db.query(Station).order_by(Station.id).first()
        if not station:
            raise HTTPException(status_code=503, detail="No stations available; seed the database first")

    horizons = list(dict.fromkeys(req.horizons))
    coverage = forecast_service.station_data_sufficiency(db, station.id)

    # Operational outlook: run the time-stepped two-way coupled forecast so the
    # served 72h AQI reflects aerosol-radiative PBL/weather feedback (the core
    # requirement). Falls back to the direct per-horizon ML forecast if the
    # coupled engine cannot run for any reason.
    try:
        result = forecast_service.generate_coupled_forecast(db, station.id, horizons)
        predictions = result["coupled"]
        forecast_service.save_coupled_forecasts(db, station.id, predictions)
    except Exception:
        logger.warning("coupled forecast failed for %s; falling back to direct ML", station.name, exc_info=True)
        _, predictions = forecast_service.generate_forecast(db, station.id, horizons)

    weather = forecast_service.get_weather_context(db, station.id)
    fire = forecast_service.get_fire_context(db)
    aqi_series = [p["aqi_pred"] for p in predictions]
    if len(aqi_series) >= 2:
        if aqi_series[-1] > aqi_series[0] * 1.05:
            trend = "rising"
        elif aqi_series[-1] < aqi_series[0] * 0.95:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "stable"

    worst = max(predictions, key=lambda p: p["aqi_pred"])
    alert_inputs = {**worst, "trend": trend}
    for alert in alert_service.generate_alerts(alert_inputs, weather, fire):
        db.add(Alert(
            station_id=station.id,
            alert_level=alert["alert_level"],
            title=alert["title"],
            description=alert.get("description", ""),
            forecast_horizon_hours=alert.get("forecast_horizon_hours"),
            factors=alert.get("factors"),
            recommendation=alert.get("recommendation"),
        ))
    db.commit()

    return ForecastGenerateResponse(
        station=station.name,
        generated_at=datetime.now(UTC).replace(tzinfo=None),
        horizons=horizons,
        pooled_features=bool(coverage["pooled"]),
        local_readings=int(coverage["local_readings"]),
        history_days=coverage.get("history_days"),
        forecasts=[
            ForecastPoint(
                timestamp=datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=p["horizon_hours"]),
                horizon_hours=p["horizon_hours"],
                pm25_pred=p["pm25_pred"],
                pm10_pred=p["pm10_pred"],
                o3_pred=p["o3_pred"],
                no2_pred=p["no2_pred"],
                so2_pred=p.get("so2_pred"),
                co_pred=p.get("co_pred"),
                aqi_pred=p["aqi_pred"],
                aqi_category=p["aqi_category"],
                coupling_stability=p.get("coupling_stability"),
                coupling_mode="coupled" if p.get("coupling_stability") is not None else "direct",
            )
            for p in predictions
        ],
    )

@router.get("/forecast/comparison/{station_name}", response_model=ForecastComparisonResponse)
def get_forecast_comparison(
    station_name: str,
    hours: int = Query(default=72, ge=1, le=720),
    db: Session = Depends(get_db),
):
    station = _station_or_404(db, station_name)
    start = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)

    forecasts = (
        db.query(Forecast)
        .filter(Forecast.station_id == station.id, Forecast.forecast_timestamp >= start)
        .order_by(Forecast.forecast_timestamp)
        .all()
    )
    readings = (
        db.query(PollutionReading)
        .filter(
            PollutionReading.station_id == station.id,
            PollutionReading.timestamp >= start - timedelta(hours=2),
        )
        .order_by(PollutionReading.timestamp)
        .all()
    )

    reading_by_hour = {}
    for r in readings:
        key = _round_hour(r.timestamp)
        reading_by_hour.setdefault(key, r)

    points = []
    for f in forecasts:
        key = _round_hour(f.forecast_timestamp)
        actual = reading_by_hour.get(key)
        delta = None
        if actual and actual.pm25 is not None and f.pm25_pred is not None:
            delta = round(f.pm25_pred - actual.pm25, 1)
        actual_aqi = None
        if actual is not None:
            observed_aqi, observed_category, _ = calculate_aqi(
                pm25=actual.pm25, pm10=actual.pm10, o3=actual.o3,
                no2=actual.no2, so2=actual.so2, co=actual.co)
            if observed_category != "Unknown":
                actual_aqi = observed_aqi
        points.append(ForecastComparisonPoint(
            timestamp=f.forecast_timestamp,
            actual_aqi=actual_aqi,
            predicted_aqi=f.aqi_pred,
            actual_pm25=actual.pm25 if actual else None,
            predicted_pm25=f.pm25_pred,
            delta=delta,
        ))

    if not points:
        raise HTTPException(status_code=404, detail=f"No forecasts within the last {hours} hours for station '{station_name}'")
    return ForecastComparisonResponse(station=station.name, points=points)

@router.get("/forecast/ncr", response_model=dict[str, list[ForecastPoint]])
def get_ncr_forecast(hours: int = Query(default=72, ge=1, le=72), db: Session = Depends(get_db)):
    from ..services.ttl_cache import cached

    return cached(f"forecast-ncr:{hours}", 120, lambda: _read_ncr_forecast(db, hours))


def _read_ncr_forecast(db: Session, hours: int) -> dict[str, list[ForecastPoint]]:
    stations = db.query(Station).order_by(Station.name).all()
    result = {}
    for station in stations:
        forecasts = (
            db.query(Forecast)
            .filter(Forecast.station_id == station.id, Forecast.horizon_hours <= hours)
            .order_by(Forecast.forecast_timestamp)
            .all()
        )
        result[station.name] = [_to_forecast_point(f) for f in forecasts]
    return result

@router.get("/forecast/{station_name}/context", response_model=ForecastContextResponse)
def get_forecast_context(station_name: str, db: Session = Depends(get_db)):
    """Per-horizon atmospheric + coupling context for the 72h forecast window.

    For each horizon (1..72h) returns the nearest stored weather observation's
    atmosphere (temperature, humidity, pressure, wind, PBL, lapse-rate
    inversion status) plus the coupling-engine features derived from it.
    Missing rows are reported as nulls (UI renders "Data unavailable") and are
    never filled with synthetic values.
    """
    station = _station_or_404(db, station_name)
    from ..services.coupling_service import get_forecast_context as _context
    from ..services.ttl_cache import cached

    return cached(f"forecast-context:{station_name}", 300, lambda: _context(db, station))

@router.get("/forecast/{station_name}", response_model=list[ForecastPoint])
def get_forecast(station_name: str, hours: int = Query(default=72, ge=1, le=72), db: Session = Depends(get_db)):
    station = _station_or_404(db, station_name)
    from ..services.ttl_cache import cached

    return cached(f"forecast:{station_name}:{hours}", 120, lambda: _read_station_forecast(db, station, hours))


def _read_station_forecast(db: Session, station: Station, hours: int) -> list[ForecastPoint]:
    forecasts = (
        db.query(Forecast)
        .filter(Forecast.station_id == station.id, Forecast.horizon_hours <= hours)
        .order_by(Forecast.forecast_timestamp)
        .all()
    )
    return [_to_forecast_point(f) for f in forecasts]
