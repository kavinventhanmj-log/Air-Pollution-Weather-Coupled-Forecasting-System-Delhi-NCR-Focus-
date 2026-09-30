from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import Station, WeatherReading
from ..schemas.schemas import WeatherDetailResponse

router = APIRouter()

def _station_or_404(db: Session, station_name: str) -> Station:
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")
    return station

def _to_detail(station: Station, r) -> WeatherDetailResponse:
    return WeatherDetailResponse(
        station=station.name,
        station_id=station.id,
        latitude=station.latitude,
        longitude=station.longitude,
        reading_latitude=r.latitude,
        reading_longitude=r.longitude,
        timestamp=r.timestamp,
        temperature=r.temperature,
        humidity=r.humidity,
        pressure_msl=r.pressure_msl,
        surface_pressure=r.surface_pressure,
        wind_speed=r.wind_speed,
        wind_direction=r.wind_direction,
        precipitation=r.precipitation,
        cloud_cover=r.cloud_cover,
        pbl_height=r.pbl_height,
    )

@router.get("/weather/latest", response_model=list[WeatherDetailResponse])
def get_weather_latest(db: Session = Depends(get_db)):
    """Latest weather observation for every station in Delhi NCR.

    Serves a single NCR-wide snapshot: the most recent ``weather_observations``
    row per station, newest observed time first. Stations without any weather
    data simply do not appear in the result.
    """
    stations = {s.id: s for s in db.query(Station).all()}
    readings = (
        db.query(WeatherReading)
        .order_by(WeatherReading.station_id, WeatherReading.timestamp.desc())
        .all()
    )
    seen: set[int] = set()
    latest = []
    for reading in readings:
        if reading.station_id in seen:
            continue
        seen.add(reading.station_id)
        station = stations.get(reading.station_id)
        if station is None:
            continue
        latest.append(_to_detail(station, reading))
    return latest

@router.get("/weather/{station_name}", response_model=WeatherDetailResponse)
def get_weather(station_name: str, db: Session = Depends(get_db)):
    station = _station_or_404(db, station_name)
    reading = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station.id)
        .order_by(WeatherReading.timestamp.desc())
        .first()
    )
    if not reading:
        raise HTTPException(status_code=404, detail=f"No weather data for station '{station_name}'")
    return _to_detail(station, reading)

@router.get("/weather/{station_name}/history", response_model=list[WeatherDetailResponse])
def get_weather_history(
    station_name: str,
    hours: int = Query(default=24, ge=1, le=720),
    db: Session = Depends(get_db),
):
    station = _station_or_404(db, station_name)
    readings = (
        db.query(WeatherReading)
        .filter(WeatherReading.station_id == station.id)
        .order_by(WeatherReading.timestamp.desc())
        .limit(hours)
        .all()
    )
    if not readings:
        raise HTTPException(status_code=404, detail=f"No historical weather for station '{station_name}'")
    return [_to_detail(station, r) for r in readings]
