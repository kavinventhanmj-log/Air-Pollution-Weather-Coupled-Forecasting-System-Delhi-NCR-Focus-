"""GRAded Response Action Plan (GRAP) endpoints for Delhi NCR.

Exposes the static CAQM stage matrix and live assessments. The live assessment
is derived from persisted observation state (current AQI, latest
PBL-derived inversion proxy and recent FIRMS fire intensity) with no external
network calls, so it is safe to call from the dashboard on every refresh.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import FireReading, PollutionReading, Station, WeatherReading
from ..schemas.schemas import GrapAssessment, GrapStageOut, GrapStagesResponse
from ..services.aqi_calculator import calculate_aqi, get_aqi_category, get_dominant_pollutant
from ..services.grap_service import assess_grap, get_grap_stages

router = APIRouter()

INVERSION_PBL_REFERENCE_M = 500  # mirrors forecast_service's PBL proxy


def _reading_aqi(
    reading: PollutionReading | None,
) -> tuple[int | None, str | None, str | None]:
    """``(aqi, category, dominant_pollutant)`` recalculated from one reading.

    The ``PollutionReading.aqi`` column is a denormalised cache written at ingest
    time, and rows ingested before the breakpoint fix still carry the old
    gappy-table calculator's fallthrough: 9.4% of the shipped archive stores a
    spurious ``500`` for readings whose concentrations fell in a band gap (for
    example O3 = 0.1 ug/m3 -> aqi 500). Averaging that column into the NCR figure
    escalated the GRAP stage on 12.7% of sampled days, so both GRAP endpoints
    now recompute from the stored concentrations instead - which is what
    ``get_current_aqi`` has always done, and why it ignores the column too.

    This is the same *instantaneous latest-reading* basis the per-station
    endpoint already used, so the two GRAP endpoints now share one definition
    and cannot drift apart again. It is deliberately not the window-mean basis
    of ``/current``: GRAP assesses the latest observed state, not a trailing
    average, and the O3 8h->1h substitution belongs to the window-mean path.
    """
    if reading is None:
        return None, None, None
    aqi, category, dominant = calculate_aqi(
        reading.pm25,
        reading.pm10,
        reading.o3,
        reading.no2,
        reading.so2,
        reading.co,
    )
    if category == "Unknown":
        # Nothing was scorable. That is not the same as clean air, which scores
        # 0 and reports "Good", so it is reported as "no AQI" rather than 0.
        return None, None, None
    return aqi, category, dominant


def _inversion_proxy_from_pbl(pbl_height_m: float | None) -> float | None:
    """Normalised 0..1 inversion proxy from the shallowest recent PBL height."""
    if pbl_height_m is None or pbl_height_m != pbl_height_m:
        return None
    return max(0.0, min(1.0, (INVERSION_PBL_REFERENCE_M - pbl_height_m) / INVERSION_PBL_REFERENCE_M))


@router.get("/grap/stages", response_model=GrapStagesResponse)
def grap_stages():
    """The full GRAP stage matrix (not-invoked + Stages I-IV)."""
    return GrapStagesResponse(stages=[GrapStageOut(**s) for s in get_grap_stages()])


@router.get("/grap/current", response_model=GrapAssessment)
def grap_current(db: Session = Depends(get_db)):
    """Assess the operative GRAP stage for NCR from current persisted state."""
    from ..services.ttl_cache import cached

    return cached("grap:current", 300, lambda: _compute_grap_current(db))


def _compute_grap_current(db: Session) -> GrapAssessment:
    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24)

    stations = db.query(Station).order_by(Station.name).all()
    aqi_values: list[int] = []
    latest: list[PollutionReading] = []
    for s in stations:
        r = (
            db.query(PollutionReading)
            .filter(PollutionReading.station_id == s.id, PollutionReading.timestamp >= since)
            .order_by(PollutionReading.timestamp.desc())
            .first()
        )
        if r is not None:
            latest.append(r)
            # Recalculated, not read from the denormalised `aqi` column - see
            # `_reading_aqi`. Stations with nothing scorable are skipped so they
            # cannot drag the NCR mean down with a spurious 0.
            station_aqi, _cat, _dom = _reading_aqi(r)
            if station_aqi is not None:
                aqi_values.append(station_aqi)

    ncr_aqi = round(sum(aqi_values) / len(aqi_values)) if aqi_values else None
    category, _ = get_aqi_category(ncr_aqi) if ncr_aqi is not None else (None, None)

    def _avg(attr):
        values = [getattr(r, attr) for r in latest if getattr(r, attr) is not None]
        return sum(values) / len(values) if values else None

    dominant = get_dominant_pollutant(
        _avg("pm25"), _avg("pm10"), _avg("o3"), _avg("no2"), _avg("so2"), _avg("co"),
    ) if latest else None

    shallowest_pbl = (
        db.query(func.min(WeatherReading.pbl_height))
        .filter(WeatherReading.timestamp >= since)
        .scalar()
    )
    mean_frp = (
        db.query(func.avg(FireReading.frp))
        .filter(FireReading.acq_date >= since, FireReading.synthetic.is_(False))
        .scalar()
    )
    fire_mean_frp_mw = float(mean_frp) if mean_frp is not None else None

    result = assess_grap(
        aqi=ncr_aqi,
        inversion_strength=_inversion_proxy_from_pbl(shallowest_pbl),
        fire_mean_frp_mw=fire_mean_frp_mw,
    )
    result["aqi_category"] = category
    result["dominant_pollutant"] = dominant
    return GrapAssessment(**result)


@router.get("/grap/{station_name}", response_model=GrapAssessment)
def grap_for_station(station_name: str, db: Session = Depends(get_db)):
    """Assess the operative GRAP stage for one station's latest reading."""
    station = db.query(Station).filter(func.lower(Station.name) == station_name.lower()).first()
    if station is None:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station_name}")

    r = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station.id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )
    aqi, category, dominant = _reading_aqi(r)

    latest_pbl = (
        db.query(WeatherReading.pbl_height)
        .filter(WeatherReading.station_id == station.id)
        .order_by(WeatherReading.timestamp.desc())
        .first()
    )

    result = assess_grap(
        aqi=aqi if r else None,
        inversion_strength=_inversion_proxy_from_pbl(latest_pbl[0] if latest_pbl else None),
        fire_mean_frp_mw=None,
    )
    result["aqi_category"] = category if category and category != "Unknown" else None
    result["dominant_pollutant"] = dominant
    return GrapAssessment(**result)
