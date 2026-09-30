import math
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import FireReading, PollutionReading, Station, WeatherReading
from ..schemas.schemas import (
    FireActivityResponse,
    FireEvent,
    FireHotspot,
    FireHotspotsResponse,
    FiresLatestResponse,
    PlumeRiskResponse,
    TransportDirectionResponse,
)

router = APIRouter()

DELHI_LAT = 28.6139
DELHI_LON = 77.2090

COMPASS_POINTS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

UPWIND_SOURCES = {"N", "NNE", "NW", "NNW", "W", "WNW"}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

def estimate_transport_direction(wind_direction):
    if wind_direction is None:
        return {"source": "Unknown", "target": "Unknown", "label": "Unknown"}
    from_idx = int((float(wind_direction) % 360) / 22.5 + 0.5) % 16
    to_idx = (from_idx + 8) % 16
    source = COMPASS_POINTS[from_idx]
    target = COMPASS_POINTS[to_idx]
    return {"source": source, "target": target, "label": f"{source} \u2192 {target}"}

def _latest_wind(db: Session, station_name: str = None):
    query = db.query(WeatherReading)
    if station_name:
        station = db.query(Station).filter(Station.name == station_name).first()
        if station:
            query = query.filter(WeatherReading.station_id == station.id)
    reading = query.order_by(WeatherReading.timestamp.desc()).first()
    if not reading:
        return None, None, None
    return reading, reading.wind_speed, reading.wind_direction

def _latest_network_pm25(db: Session) -> float | None:
    """Latest PM2.5 per station averaged across the network (µg/m³)."""
    rows = db.query(PollutionReading.station_id, PollutionReading.timestamp, PollutionReading.pm25).all()
    last_by_station: dict[int, float] = {}
    for station_id, _ts, pm25 in rows:
        if pm25 is None:
            continue
        last_by_station[station_id] = pm25
    if not last_by_station:
        return None
    return sum(last_by_station.values()) / len(last_by_station)

@router.get("/fire-activity", response_model=FireActivityResponse)
def get_fire_activity(db: Session = Depends(get_db)):
    from ..services.ttl_cache import cached

    return cached("fire-activity", 300, lambda: _compute_fire_activity(db))


def _compute_fire_activity(db: Session) -> FireActivityResponse:
    # `FireReading.acq_date` is naive UTC (FIRMS acquisition times are UTC, see
    # firms_service._parse_acq_time). The fallback therefore has to be UTC too:
    # `datetime.now()` returns the *server's* local time, so the same field would
    # mean two different things depending on where the process runs.
    now_utc = datetime.now(UTC).replace(tzinfo=None)

    fires = db.query(FireReading).order_by(FireReading.acq_date.desc()).limit(1000).all()
    if not fires:
        return FireActivityResponse(total_fires=0, high_confidence_fires=0, mean_frp=0.0, region="Delhi NCR", date=now_utc)
    high_conf = sum(1 for f in fires if f.confidence and f.confidence.lower() == "high")
    frps = [f.frp for f in fires if f.frp is not None]
    return FireActivityResponse(
        total_fires=len(fires),
        high_confidence_fires=high_conf,
        mean_frp=sum(frps) / len(frps) if frps else 0.0,
        region="Punjab/Haryana/Rajasthan",
        date=fires[0].acq_date if fires else now_utc,
    )

@router.get("/fire/transport", response_model=TransportDirectionResponse)
def get_transport_direction(station_name: str = "Anand Vihar", db: Session = Depends(get_db)):
    reading, speed, wind_deg = _latest_wind(db, station_name)
    est = estimate_transport_direction(wind_deg)
    basis = "live weather data"
    if wind_deg is None:
        est = {"source": "NW", "target": "SE", "label": "NW \u2192 SE"}
        basis = "climatological default (NW winter winds)"
    return TransportDirectionResponse(
        station=station_name,
        from_direction=est["source"],
        to_direction=est["target"],
        label=est["label"],
        wind_speed=speed,
        wind_direction=wind_deg,
        basis=basis,
    )

@router.get("/plume-risk", response_model=PlumeRiskResponse)
def get_plume_risk(db: Session = Depends(get_db)):
    from ..services.ttl_cache import cached

    return cached("plume-risk", 300, lambda: _compute_plume_risk(db))


def _compute_plume_risk(db: Session) -> PlumeRiskResponse:
    fires = db.query(FireReading).order_by(FireReading.acq_date.desc()).limit(500).all()
    if not fires:
        return PlumeRiskResponse(
            risk_level="LOW",
            risk_score=0.1,
            fire_count=0,
            transport_direction="Unknown",
            wind_speed=0,
            distance_nearest_fire=999,
            confidence=0,
            factors=["No fire data available"],
        )

    reading, wind_speed, wind_deg = _latest_wind(db)
    est = estimate_transport_direction(wind_deg)
    if wind_deg is None:
        est = {"source": "NW", "target": "SE", "label": "NW \u2192 SE"}
    transport_direction = est["label"]

    # Real fire-impact / transport features (FRP-weighted, wind-aligned)
    from ml.features.fire_impact import compute_fire_impact

    fires_df = __import__("pandas").DataFrame([{
        "lat": f.latitude, "lon": f.longitude, "frp": f.frp or 1.0,
    } for f in fires])
    impact = compute_fire_impact(
        fires_df, DELHI_LAT, DELHI_LON,
        wind_dir=wind_deg if wind_deg is not None else 0.0,
        wind_speed=wind_speed or 0.0,
    )
    fire_count = impact["fire_count"]
    min_dist = impact["nearest_fire_distance"]

    base = (fire_count / 100) * (1 / max(min_dist / 100, 0.1))
    alignment = 1.0 if est["source"] in UPWIND_SOURCES else 0.9
    risk_score = min(1.0, base * alignment)

    if risk_score > 0.7:
        risk_level = "HIGH"
    elif risk_score > 0.4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Merge ML transport risk into the headline risk level (additive, honest:
    # both derive from real fires + wind).
    transport_risk = impact["transport_risk"]
    if transport_risk and transport_risk >= 0.7 and risk_level == "LOW":
        risk_level = "MODERATE"
    if transport_risk and transport_risk >= 0.4 and risk_level == "LOW":
        risk_level = "MODERATE"

    factors = []
    if fire_count > 50:
        factors.append(f"High fire activity: {fire_count} active hotspots")
    if min_dist < 200:
        factors.append(f"Nearest fire only {min_dist:.0f}km from Delhi")
    if wind_deg is not None:
        factors.append(f"Wind {wind_speed:.1f} m/s from {est['source']}; plume transport toward {est['target']}")
    else:
        factors.append("Wind data unavailable; assuming climatological NW \u2192 SE transport")
    if impact["wind_alignment_pct"]:
        factors.append(f"{impact['wind_alignment_pct']:.0f}% of nearby fires upwind (winds from {est['source']})")
    if impact["transport_time_hours"]:
        factors.append(f"Nearest fire smoke advective arrival ~{impact['transport_time_hours']:.1f}h at current wind")
    if impact["stubble_impact_score"]:
        factors.append(f"Stubble-smoke PM impact proxy: {impact['stubble_impact_score']:.2f}")

    # Estimated smoke-attributed contribution to today's PM2.5 load.
    # Transparent proxy: stubble_impact_score (0-1, FRP + wind-alignment
    # weighted) applied to the current network-mean PM2.5. NOT measured.
    network_pm25 = _latest_network_pm25(db)
    est_contribution = None
    if network_pm25 and impact["stubble_impact_score"]:
        est_contribution = round(impact["stubble_impact_score"] * network_pm25, 1)
        factors.append(f"Estimated smoke-attributed PM2.5 contribution (proxy): +{est_contribution:.1f} \u00b5g/m\u00b3 of current {network_pm25:.0f} \u00b5g/m\u00b3 network mean")

    return PlumeRiskResponse(
        risk_level=risk_level,
        risk_score=round(risk_score, 3),
        fire_count=fire_count,
        transport_direction=transport_direction,
        wind_speed=round(wind_speed or 0, 1),
        distance_nearest_fire=round(min_dist, 1),
        confidence=round(min(risk_score, 0.95), 2),
        factors=factors,
        wind_alignment_pct=round(impact["wind_alignment_pct"], 1),
        transport_time_hours=impact["transport_time_hours"],
        transport_risk=transport_risk,
        transport_risk_level=impact["transport_risk_level"],
        stubble_impact_score=round(impact["stubble_impact_score"], 3),
        estimated_pm25_contribution_ugm3=est_contribution,
    )


@router.get("/fire/hotspots", response_model=FireHotspotsResponse)
def get_fire_hotspots(db: Session = Depends(get_db)):
    """Recent FIRMS active-fire locations for map overlay (SIH26082)."""
    from ..services.ttl_cache import cached

    return cached("fire-hotspots", 300, lambda: _compute_fire_hotspots(db))


def _compute_fire_hotspots(db: Session) -> FireHotspotsResponse:
    fires = db.query(FireReading).order_by(FireReading.acq_date.desc()).limit(1000).all()
    hotspots = [
        FireHotspot(
            lat=f.latitude,
            lon=f.longitude,
            frp=f.frp,
            confidence=f.confidence,
            acq_date=f.acq_date,
        )
        for f in fires
    ]
    return FireHotspotsResponse(region="Punjab/Haryana/Rajasthan", hotspots=hotspots)


@router.get("/fires/latest", response_model=FiresLatestResponse)
def get_latest_fires(
    hours: int = Query(default=48, ge=1, le=24 * 30, description="Look-back window in hours"),
    limit: int = Query(default=1000, ge=1, le=5000, description="Max hotspot rows to return"),
    db: Session = Depends(get_db),
):
    """Latest stored NASA FIRMS fire observations for the NCR + upwind region.

    Purely observational: returns what was detected (time, location, FRP,
    confidence, satellite/instrument). It states only that a fire was observed
    at that location/time — it does not assert that any fire caused or
    contributed to Delhi pollution.
    """
    from ..config import get_settings

    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)
    # SQLite stores naive datetimes; PostgreSQL (timezone=True) can hold aware.
    if get_settings().database_url.startswith("sqlite"):
        since = since.replace(tzinfo=None)
    fires = (
        db.query(FireReading)
        .filter(FireReading.acq_date >= since)
        .order_by(FireReading.acq_date.desc(), FireReading.id.desc())
        .limit(limit)
        .all()
    )
    return FiresLatestResponse(
        region="Delhi NCR + upwind tract (Punjab/Haryana/north Rajasthan)",
        generated_at=now,
        count=len(fires),
        fires=[
            FireEvent(
                id=f.id,
                latitude=f.latitude,
                longitude=f.longitude,
                acq_date=f.acq_date,
                confidence=f.confidence,
                frp=f.frp,
                brightness=f.brightness,
                satellite=f.satellite,
                instrument=f.instrument,
                daynight=f.daynight,
            )
            for f in fires
        ],
    )
