"""Threshold alert engine plus the live, all-station alert builder.

``generate_alerts`` is the pure rule engine: it maps one station's forecast /
weather / fire context onto advisory records. ``all_station_alerts`` is the
read-side driver used by ``GET /api/alerts`` — it evaluates the rules for
*every* station on demand so the Alerts view is never limited to whichever
station happened to be persisted last.
"""

from datetime import UTC, datetime
from typing import Any

ALERT_LEVEL_RANK = {"WATCH": 1, "ADVISORY": 2, "WARNING": 3, "SEVERE": 4}

# A run is the set of horizon rows written by one forecast generation. Six
# horizons are persisted in a single commit, so they share one ``created_at``.
_MAX_RUN_ROWS = 24


def generate_alerts(
    forecast_data: dict[str, Any], weather_data: dict[str, Any], fire_data: dict[str, Any]
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    forecast_data = forecast_data or {}
    weather_data = weather_data or {}
    fire_data = fire_data or {}

    aqi = forecast_data.get("aqi_pred", 0) or 0
    dominant = forecast_data.get("dominant_pollutant", "")
    trend = forecast_data.get("trend", "stable")

    if aqi >= 401:
        alerts.append(
            {
                "alert_level": "SEVERE",
                "title": "Severe Pollution Alert",
                "description": f"Severe AQI conditions expected. Forecast AQI: {aqi}.",
                "factors": f"Severe concentrations; dominant pollutant: {dominant or 'n/a'}",
                "recommendation": "Avoid outdoor activities. Sensitive groups should remain indoors with air purifiers.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )
    elif aqi >= 301:
        alerts.append(
            {
                "alert_level": "WARNING",
                "title": "Very Poor AQI Warning",
                "description": f"Very Poor AQI conditions expected. Forecast AQI: {aqi}.",
                "factors": f"Elevated pollution levels; dominant pollutant: {dominant or 'n/a'}",
                "recommendation": "Limit prolonged outdoor exertion. Use N95 masks outdoors.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )
    elif aqi >= 201:
        alerts.append(
            {
                "alert_level": "ADVISORY",
                "title": "Poor AQI Advisory",
                "description": f"Poor AQI conditions expected. Forecast AQI: {aqi}.",
                "factors": f"Elevated pollution driven by {dominant or 'multiple pollutants'}",
                "recommendation": "Consider reducing strenuous outdoor activity.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )

    if trend == "rising":
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "Rising Pollution Trend",
                "description": "Pollution levels are forecast to increase across the forecast window.",
                "factors": "Concentrations trending upward hour over hour",
                "recommendation": "Monitor future forecasts; pollution controls may be needed if the trend persists.",
            }
        )
    elif trend == "falling":
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "Improving Air Quality",
                "description": "Pollution levels are forecast to decline over the coming hours.",
                "factors": "Concentrations trending downward",
                "recommendation": "Conditions are expected to improve; verify before resuming outdoor activity.",
            }
        )

    if dominant == "pm25" and aqi >= 201:
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "PM2.5 Dominated Pollution",
                "description": "PM2.5 is the dominant pollutant and is in the Poor-to-Severe range.",
                "factors": "Fine particulate matter primary driver",
                "recommendation": "Avoid intense physical activity; fine particles penetrate indoors.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )

    wind = weather_data.get("wind_speed")
    if wind is not None and wind < 2:
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "Low Wind Speed Watch",
                "description": f"Wind speed is {wind:.1f} m/s, limiting pollutant dispersion.",
                "factors": "Low wind speed",
                "recommendation": "Pollution may accumulate due to poor ventilation.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )
    if wind is not None and wind > 15:
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "High Wind / Dust Watch",
                "description": f"Wind speed of {wind:.1f} m/s may resuspend dust and aggravate PM10.",
                "factors": "Strong winds",
                "recommendation": "Expect elevated PM10; cover windows near construction sites.",
            }
        )

    pbl = weather_data.get("pbl_height")
    if pbl is not None and pbl < 150:
        alerts.append(
            {
                "alert_level": "WARNING",
                "title": "Strong Inversion Trapping",
                "description": f"A strong inversion with PBL at {pbl:.0f}m is trapping pollutants.",
                "factors": "Strong inversion layer",
                "recommendation": "Expect rapid AQI deterioration overnight into morning.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )
    elif pbl is not None and pbl < 300:
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "Weak Inversion / Low PBL",
                "description": f"Planetary boundary layer at {pbl:.0f}m limits vertical mixing.",
                "factors": "Compressed boundary layer",
                "recommendation": "Pollution trapping likely near the surface.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )

    humidity = weather_data.get("humidity")
    if humidity is not None and humidity > 80:
        alerts.append(
            {
                "alert_level": "ADVISORY",
                "title": "High Humidity / Secondary Aerosol",
                "description": f"Relative humidity of {humidity:.0f}% promotes secondary aerosol formation.",
                "factors": "High relative humidity",
                "recommendation": "Secondary particles may push AQI higher than direct-emission forecasts.",
            }
        )

    precipitation = weather_data.get("precipitation")
    if precipitation is not None and precipitation <= 0.5 and aqi >= 201:
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "No Rain Scavenging",
                "description": "Absence of precipitation means no washout of accumulated pollutants.",
                "factors": "Dry conditions",
                "recommendation": "Pollution is unlikely to be flushed out; expect persistence.",
            }
        )

    fire_count = fire_data.get("fire_count", 0) or 0
    nearest = fire_data.get("distance_nearest_fire")
    if fire_count > 50 and nearest is not None and nearest < 300:
        alerts.append(
            {
                "alert_level": "WARNING",
                "title": "Approaching Smoke Plume",
                "description": f"{fire_count} regional fire hotspots with nearest fire {nearest:.0f}km from Delhi NCR.",
                "factors": "Stubble burning plume advection",
                "recommendation": "Fire smoke may compound local pollution; prepare for elevated AQI.",
                "forecast_horizon_hours": forecast_data.get("horizon_hours"),
            }
        )
    elif fire_count > 20:
        alerts.append(
            {
                "alert_level": "WATCH",
                "title": "Elevated Regional Burning",
                "description": f"{fire_count} fire hotspots detected across the northern plains.",
                "factors": "Regional agricultural burning",
                "recommendation": "Monitor winds; smoke may drift into the NCR.",
            }
        )

    alerts.sort(key=lambda a: ALERT_LEVEL_RANK.get(a["alert_level"], 0), reverse=True)
    return alerts


# ---------------------------------------------------------------------------
# Live all-station evaluation (read path for GET /api/alerts)
# ---------------------------------------------------------------------------
def _naive_utc(value: datetime | None) -> datetime | None:
    """Normalise a DB timestamp to naive UTC.

    The project stores observation/forecast timestamps as naive UTC, but
    PostgreSQL ``DateTime(timezone=True)`` columns hand back aware values. The
    two can never be compared, so strip the tzinfo at the boundary.
    """
    if value is None:
        return None
    return value.replace(tzinfo=None) if value.tzinfo else value


def _trend(aqi_series: list[float | None]) -> str:
    values = [v for v in aqi_series if v is not None]
    if len(values) < 2 or not values[0]:
        return "stable"
    if values[-1] > values[0] * 1.05:
        return "rising"
    if values[-1] < values[0] * 0.95:
        return "falling"
    return "stable"


def _latest_forecast_run(db, station_id: int) -> list[Any]:
    """Return the horizon rows of the most recent forecast run for a station.

    Rows are grouped by ``created_at`` so a partially-written run (or several
    runs committed inside the same clock second) still yields one coherent
    outlook instead of a mix of two runs.
    """
    from ..models.db_models import Forecast

    rows = (
        db.query(Forecast)
        .filter(Forecast.station_id == station_id)
        .order_by(Forecast.created_at.desc(), Forecast.horizon_hours.asc())
        .limit(_MAX_RUN_ROWS)
        .all()
    )
    if not rows:
        return []
    newest = rows[0].created_at
    run = [r for r in rows if r.created_at == newest]
    # A single-row group means the run was truncated; fall back to the whole
    # window so the outlook still spans multiple horizons.
    if len(run) == 1 and len(rows) > 1:
        run = rows
    return run


def _forecast_inputs(db, station_id: int) -> dict[str, Any] | None:
    """Build the forecast-side alert inputs for one station.

    Prefers the latest persisted forecast run. When a station has no forecast
    at all (for example a freshly seeded station) it falls back to the latest
    pollution observation so the station is still represented instead of
    silently disappearing from the Alerts view.
    """
    from ..models.db_models import PollutionReading
    from .aqi_calculator import calculate_aqi, get_dominant_pollutant

    run = _latest_forecast_run(db, station_id)
    if run:
        worst = max(run, key=lambda r: r.aqi_pred if r.aqi_pred is not None else -1)
        if worst.aqi_pred is not None:
            return {
                "aqi_pred": int(worst.aqi_pred),
                "dominant_pollutant": worst.dominant_pollutant or "",
                "trend": _trend([r.aqi_pred for r in run]),
                "horizon_hours": worst.horizon_hours,
                "as_of": _naive_utc(worst.created_at),
            }

    reading = (
        db.query(PollutionReading)
        .filter(PollutionReading.station_id == station_id)
        .order_by(PollutionReading.timestamp.desc())
        .first()
    )
    if reading is None:
        return None
    aqi, category, dominant = calculate_aqi(
        pm25=reading.pm25,
        pm10=reading.pm10,
        o3=reading.o3,
        no2=reading.no2,
        so2=reading.so2,
        co=reading.co,
    )
    if category == "Unknown":
        return None
    return {
        "aqi_pred": int(aqi),
        "dominant_pollutant": dominant or get_dominant_pollutant(
            reading.pm25, reading.pm10, reading.o3, reading.no2, reading.so2, reading.co
        ),
        "trend": "stable",
        "horizon_hours": None,
        "as_of": _naive_utc(reading.timestamp),
    }


def build_station_alerts(
    db, station, fire_data: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Evaluate the alert rules for a single station.

    ``fire_data`` is the NCR-wide fire context; pass the same dict for every
    station in a batch so the 500-row fire scan is only paid once.
    """
    from . import forecast_service

    inputs = _forecast_inputs(db, station.id)
    if inputs is None:
        return []
    weather = forecast_service.get_weather_context(db, station.id)
    if fire_data is None:
        fire_data = forecast_service.get_fire_context(db)
    as_of = inputs.get("as_of") or datetime.now(UTC).replace(tzinfo=None)
    return [
        {
            "station": station.name,
            "station_id": station.id,
            "alert_level": alert["alert_level"],
            "title": alert["title"],
            "description": alert.get("description", ""),
            "factors": alert.get("factors"),
            "recommendation": alert.get("recommendation"),
            "forecast_horizon_hours": alert.get("forecast_horizon_hours"),
            "created_at": as_of,
        }
        for alert in generate_alerts(inputs, weather, fire_data)
    ]


def all_station_alerts(db, station_name: str | None = None) -> list[dict[str, Any]]:
    """Evaluate the alert rules across the whole NCR network.

    Stations without a forecast or an observation are skipped (there is nothing
    to alert on); every other station is always represented, so the Alerts view
    covers the full network rather than a single persisted station.

    The expensive part is the 17-station sweep (latest forecast run + observation
    per station, plus one shared NCR-wide fire scan), so it is computed **once**
    and cached here rather than per caller. On the pooled Neon Postgres the first
    sweep costs several seconds, and ``/api/alerts`` and ``/api/summary`` both
    need it; caching inside the service means the second caller is free and a
    ``?station=`` filter reuses the same sweep instead of recomputing the whole
    network. The cache is bypassed on SQLite (local dev / pytest) by
    :func:`..services.ttl_cache.cached`.
    """
    from ..models.db_models import Station
    from . import forecast_service
    from .ttl_cache import cached

    def _sweep() -> list[dict[str, Any]]:
        stations = db.query(Station).order_by(Station.name).all()
        # One NCR-wide fire scan shared by every station.
        fire_data = forecast_service.get_fire_context(db)
        rows: list[dict[str, Any]] = []
        for station in stations:
            rows.extend(build_station_alerts(db, station, fire_data))
        rows.sort(key=lambda a: (-ALERT_LEVEL_RANK.get(a["alert_level"], 0), a["station"]))
        return rows

    rows = cached("alerts:all_stations", 120, _sweep)
    if station_name:
        return [row for row in rows if row["station"] == station_name]
    return list(rows)
