from datetime import UTC, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.db_models import PollutionReading, Station
from ..schemas.schemas import CurrentAQI, StationResponse
from ..services.aqi_averaging import rolling_means, select_o3_concentration
from ..services.aqi_calculator import AVERAGING_WINDOW_HOURS, evaluate_aqi

router = APIRouter()

# Pollutant columns exposed on PollutionReading, in canonical order.
_POLLUTANTS = ("pm25", "pm10", "o3", "no2", "so2", "co", "nh3", "pb")

# How far back to read when forming the trailing averaging windows. 24 h is the
# longest CPCB window, so this covers every pollutant with a little slack.
_WINDOW_LOOKBACK_HOURS = 30


@router.get("/stations", response_model=list[StationResponse])
def get_stations(db: Session = Depends(get_db)):
    stations = db.query(Station).order_by(Station.name).all()
    return stations

@router.get("/stations/{station_name}", response_model=StationResponse)
def get_station_by_name(station_name: str, db: Session = Depends(get_db)):
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")
    return station

@router.get("/current/{station_name}", response_model=CurrentAQI)
def get_current_aqi(station_name: str, db: Session = Depends(get_db)):
    """Current AQI for a station, scored from CPCB window means.

    The CPCB National AQI is defined on *averaged* concentrations - 24-hour
    means for PM2.5/PM10/NO2/SO2 and 8-hour means for O3/CO - not on the single
    most recent sensor value. This endpoint therefore builds trailing window
    means ending at the latest observation (see
    ``app.services.aqi_averaging.rolling_means``) and scores those.

    Averaging happens exactly once, here, and the resulting means go straight to
    ``aqi_calculator``. No downstream layer re-averages them.

    Response compatibility: ``station``, ``timestamp``, the six pollutant
    fields, ``aqi``, ``aqi_category`` and ``dominant_pollutant`` keep their
    original names and meanings. ``pm25``/``pm10``/``o3``/``no2``/``so2``/``co``
    still carry the raw latest observation. The additive fields
    (``sub_indices``, ``averaged_concentrations``, ``data_availability``, ...)
    expose the scored values and their provenance.
    """
    station = db.query(Station).filter(Station.name == station_name).first()
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_name}' not found")
    reading = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station.id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )
    if not reading:
        raise HTTPException(status_code=404, detail=f"No pollution readings for station '{station_name}'")

    as_of = reading.timestamp
    if getattr(as_of, "tzinfo", None) is not None:
        as_of = as_of.astimezone(UTC).replace(tzinfo=None)

    # Trailing windows ending at the latest observation. The upper bound keeps
    # the query strictly historical (no future rows), and the lower bound keeps
    # it to one row per station per hour rather than the whole table.
    # `rolling_means` re-checks both bounds per pollutant, so correctness does
    # not depend on this pre-filter.
    cutoff = as_of - timedelta(hours=_WINDOW_LOOKBACK_HOURS)
    history = (
        db.query(PollutionReading)
        .filter(
            PollutionReading.station_id == station.id,
            PollutionReading.timestamp <= as_of,
            PollutionReading.timestamp >= cutoff,
        )
        .all()
    )

    averaged = rolling_means(history, as_of)

    # CPCB's O3 rule: the 8-hour mean is the normal scoring period, but an
    # 8-hour mean above 208 ug/m3 must be replaced by the 1-hour value for the
    # sub-index. The choice is made in the averaging layer; this endpoint only
    # scores whatever concentration comes back. Note this is *not* the legacy
    # instantaneous fallback below - it fires on the 208 threshold alone.
    o3_concentration, o3_basis = select_o3_concentration(history, as_of)

    # Raw latest values, kept for the legacy response fields.
    instantaneous = {
        p: getattr(reading, p, None) for p in _POLLUTANTS
    }
    instantaneous = {k: v for k, v in instantaneous.items() if v is not None}

    # Score the window means when at least one pollutant has a usable window;
    # otherwise fall back to the instantaneous row so a station with a single
    # sparse reading still reports *something* (clearly flagged as such) rather
    # than an empty shell. The fallback is driven by history coverage, not by
    # whether the resulting AQI happens to be 0 - clean air legitimately
    # produces AQI 0 and must not be treated as "no data".
    has_averaged = any(v is not None for v in averaged.values())
    if has_averaged:
        # Copy, so the reported `averaged_concentrations` below keeps exposing
        # the genuine 8-hour window mean. `o3_concentration` is None only when
        # the 8-hour window was itself unavailable, in which case O3 is not
        # scored at all - a 1-hour value never stands in for it.
        scored_input = dict(averaged)
        scored_input["o3"] = o3_concentration
        basis = "window_average"
    else:
        scored_input = instantaneous
        basis = "instantaneous"

    result = evaluate_aqi(scored_input, averaged=has_averaged)
    instant_result = evaluate_aqi(instantaneous, averaged=False)

    # `averaged_concentrations` reports the real window means, including None
    # where history was insufficient. It is never back-filled with
    # instantaneous values, so a consumer can always tell a genuine window mean
    # apart from a single raw reading.
    window_means = {k: v for k, v in averaged.items() if v is not None}
    detail = result.as_dict()

    return CurrentAQI(
        station=station.name,
        timestamp=reading.timestamp,
        pm25=reading.pm25,
        pm10=reading.pm10,
        o3=reading.o3,
        no2=reading.no2,
        so2=reading.so2,
        co=reading.co,
        nh3=reading.nh3,
        pb=reading.pb,
        # `aqi_calculator` is the single source of truth. The stored
        # `reading.aqi` column is deliberately not used as a fallback: it was
        # written by the old gappy-breakpoint calculator and can hold a
        # spurious 500 that this endpoint exists to correct.
        aqi=result.aqi,
        aqi_category=result.category,
        dominant_pollutant=result.dominant_pollutant,
        instantaneous_aqi=instant_result.aqi,
        aqi_basis=basis,
        sub_indices=detail["sub_indices"],
        averaged_concentrations=window_means,
        instantaneous_concentrations=instantaneous,
        pollutant_units=detail["pollutant_units"],
        data_availability=detail["data_availability"],
        averaging_windows={p: AVERAGING_WINDOW_HOURS.get(p) for p in _POLLUTANTS},
        o3_averaging_basis=o3_basis,
    )
