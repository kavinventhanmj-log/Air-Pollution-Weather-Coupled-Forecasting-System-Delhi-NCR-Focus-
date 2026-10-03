import sqlalchemy as sa
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from ..database import Base


class Station(Base):
    __tablename__ = "stations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    city = Column(String, default="Delhi NCR")
    state = Column(String)

class PollutionReading(Base):
    __tablename__ = "pollution_observations"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    # Concentration units follow the CPCB sub-index tables, i.e. ug/m3 for
    # every criterion pollutant EXCEPT co which is mg/m3. See
    # app/services/aqi_calculator.POLLUTANT_UNITS.
    pm25 = Column(Float)
    pm10 = Column(Float)
    o3 = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    co = Column(Float)  # mg/m3 (not ug/m3)
    # NH3 and Pb are CPCB criteria pollutants that the data.gov.in feed
    # publishes but this project previously discarded at ingest. They are
    # stored so the measurements are not lost. They are NOT scored into the
    # AQI: no verified CPCB sub-index breakpoint table for them exists in this
    # repository, so aqi_calculator reports them as unavailable rather than
    # inventing bands. Nullable throughout - never backfilled with zeros.
    nh3 = Column(Float)  # ug/m3
    pb = Column(Float)  # ug/m3
    aqi = Column(Integer)
    # Provenance tag: which official source produced this reading
    # (data_gov_in | opencity_ckan | cpcb_dataset | cpcb_live). NULL for legacy
    # rows ingested before the column existed.
    data_source = Column(String)
    # True when the row is a forward re-stamp of an older observation (see
    # backend/scripts/bootstrap_recent.py) rather than a fresh measurement.
    # Surfaced in forecast provenance so a synthetic "recent" history can never
    # be presented as live sensor data. NULL on real readings.
    re_stamped = Column(Boolean, default=False, nullable=False, server_default=sa.false())
    __table_args__ = (
        # Declared as a unique INDEX rather than a UniqueConstraint because the
        # Alembic migration (b9c9f4d1a7e2) creates it that way. A table
        # constraint added via ALTER TABLE is equivalent for enforcement, but
        # expressing it as an index here keeps `alembic revision --autogenerate`
        # from reporting a permanent index <-> constraint mismatch.
        Index("uq_pollution_station_ts", "station_id", "timestamp", unique=True),
        Index("idx_pollution_station_time", "station_id", "timestamp"),
    )

class WeatherReading(Base):
    __tablename__ = "weather_observations"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(
        Integer,
        ForeignKey("stations.id", name="fk_weather_station_id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp = Column(DateTime(timezone=True), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    pressure_msl = Column(Float)
    surface_pressure = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    precipitation = Column(Float)
    cloud_cover = Column(Float)
    pbl_height = Column(Float)
    # Vertical pressure-level temperature (degC) used for lapse-rate inversion
    # (SIH26082). Open-Meteo / ERA5 style: temperature at standard pressure
    # levels. All optional — when NULL the PBL-height proxy is used.
    temperature_1000hPa = Column(Float)
    temperature_925hPa = Column(Float)
    temperature_850hPa = Column(Float)
    temperature_700hPa = Column(Float)
    geopotential_height_925hPa = Column(Float)
    geopotential_height_850hPa = Column(Float)
    __table_args__ = (
        UniqueConstraint("station_id", "timestamp", name="uq_weather_station_ts"),
        Index("idx_weather_station_time", "station_id", "timestamp"),
    )

class FireReading(Base):
    """A single stored fire-hostspot observation (hotspot event).

    Stores the raw fire *observation* only — no attribution to pollution is
    implied here. ``acq_date`` is kept as naive-UTC (matching the weather
    convention) and the row key (satellite, latitude, longitude, acq_date)
    prevents duplicate ingestion of the same hotspot detection.

    Provenance (SIH26082): ``synthetic`` marks rows that were *simulated* for
    the 2023-2024 training history rather than observed by NASA FIRMS;
    ``source`` names the origin — ``"synthetic_sim"`` (generated stubble-fire
    history), ``"firms_csv"`` (real observations loaded from a FIRMS CSV
    export), or ``"firms_live"`` (written from live FIRMS refresh). Legacy
    rows default to ``synthetic=False``. Operational readers MUST exclude
    synthetic rows unless the caller explicitly requests the simulated
    overlay, so a simulated fire is never presented as a live FIRMS detection.
    """
    __tablename__ = "fire_readings"
    id = Column(Integer, primary_key=True, index=True)
    satellite = Column(String)
    instrument = Column(String)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    acq_date = Column(DateTime(timezone=True), nullable=False)
    confidence = Column(String)
    frp = Column(Float)
    brightness = Column(Float)
    daynight = Column(String)
    synthetic = Column(Boolean, nullable=False, default=False)
    source = Column(String)
    __table_args__ = (
        UniqueConstraint("satellite", "latitude", "longitude", "acq_date", name="uq_fire_lat_lon_time"),
        Index("idx_fire_lat_lon_time", "latitude", "longitude", "acq_date"),
    )

class Forecast(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(
        Integer,
        ForeignKey("stations.id", name="fk_forecast_station_id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    forecast_timestamp = Column(DateTime(timezone=True), nullable=False)
    horizon_hours = Column(Integer, nullable=False)
    pm25_pred = Column(Float)
    pm10_pred = Column(Float)
    o3_pred = Column(Float)
    no2_pred = Column(Float)
    so2_pred = Column(Float)
    co_pred = Column(Float)
    aqi_pred = Column(Integer)
    aqi_category = Column(String)
    dominant_pollutant = Column(String)
    inversion_detected = Column(Integer)
    inversion_strength = Column(Float)
    pbl_height = Column(Float)
    coupling_stability = Column(Float)
    coupling_mode = Column(String)
    __table_args__ = (
        # One row per (station, horizon): regenerating a forecast replaces it
        # instead of accumulating duplicates that make history look like a
        # single day. Enforced in the DB, not just in the upsert helper.
        UniqueConstraint(
            "station_id", "horizon_hours", name="uq_forecast_station_horizon"
        ),
        Index("idx_forecast_station_time", "station_id", "forecast_timestamp"),
        Index("idx_forecast_horizon", "horizon_hours"),
    )

class ForecastRun(Base):
    """Audit trail of every forecast generation attempt (SIH honesty requirement).

    A run is written whether the forecast succeeded, fell back, or was refused
    for insufficient data. ``status`` is one of ``succeeded`` / ``fallback`` /
    ``refused``; the provenance columns mirror the ``ForecastProvenance`` block
    returned to the caller, so a reviewer can reconstruct what the system knew
    at the moment a number was published instead of trusting the number alone.
    """

    __tablename__ = "forecast_runs"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(
        Integer,
        ForeignKey("stations.id", name="fk_forecast_run_station_id", ondelete="CASCADE"),
        nullable=False,
    )
    station_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status = Column(String, nullable=False)  # succeeded | fallback | refused
    refusal_code = Column(String)  # e.g. insufficient_data
    refusal_reason = Column(String)
    model = Column(String)
    model_artifact = Column(String)
    model_artifact_sha256 = Column(String)
    fallback_used = Column(Boolean, default=False, nullable=False, server_default=sa.false())
    fallback_reason = Column(String)
    horizons = Column(String)  # comma-separated horizon list
    history_rows = Column(Integer, default=0, nullable=False, server_default="0")
    pollution_rows = Column(Integer, default=0, nullable=False, server_default="0")
    weather_rows = Column(Integer, default=0, nullable=False, server_default="0")
    fire_rows = Column(Integer, default=0, nullable=False, server_default="0")
    window_start = Column(DateTime)
    window_end = Column(DateTime)
    latest_observation = Column(DateTime)
    observation_age_hours = Column(Float)
    is_stale = Column(Boolean, default=False, nullable=False, server_default=sa.false())
    is_demo = Column(Boolean, default=False, nullable=False, server_default=sa.false())
    is_re_stamped = Column(Boolean, default=False, nullable=False, server_default=sa.false())
    data_source = Column(String)
    __table_args__ = (
        Index("idx_forecast_run_station_time", "station_id", "created_at"),
    )

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(
        Integer,
        ForeignKey("stations.id", name="fk_alert_station_id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    alert_level = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(String)
    forecast_horizon_hours = Column(Integer)
    factors = Column(String)
    recommendation = Column(String)

class User(Base):
    """Portal account used by the UI authentication layer (SIH26082).

    Stored passwords are scrypt-hashed with a per-user random salt; only the
    hex-encoded salt and hash are persisted. See ``backend/app/security.py``.
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="Analyst")
    password_hash = Column(String, nullable=False)
    password_salt = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    pollutant = Column(String, nullable=False)
    horizon_hours = Column(Integer, nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    r2 = Column(Float)
    mape = Column(Float)
    test_period_start = Column(DateTime)
    test_period_end = Column(DateTime)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())


class CouplingState(Base):
    """Latest persisted meteorology-pollution-fire coupling snapshot per station.

    A write-through row produced by the coupling service every time the
    coupling features are computed (SIH26082 Phase 30 persistence): the nine
    coupling features, the key atmospheric inputs, the invert fire-transport
    fields and provenance timestamps. The row is keyed on ``station_id`` —
    each station keeps its most recent snapshot.

    ``coupling_state`` is the data-driven feedback-surrogate band of the
    composite ``meteorology_pollution_interaction`` feature (NONE / LOW /
    MODERATE / HIGH); ``coupling_domains`` lists which feature domains
    (aerosol, atmospheric, feedback, fire, ozone) were present from stored
    data; ``data_quality`` reflects how many of the nine features were
    computable (GOOD / PARTIAL / SPARSE / UNAVAILABLE). These are labels of
    the coupling *engine*, never of a physics simulation — see
    ``docs/SCIENTIFIC_METHODOLOGY.md``.
    """
    __tablename__ = "coupling_states"
    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)
    # -- atmospheric inputs (latest stored observations) -----------------
    wind_speed_mps = Column(Float)
    wind_direction_deg = Column(Float)
    pbl_height_m = Column(Float)
    # -- inversion --------------------------------------------------------
    inversion_detected = Column(Boolean)
    inversion_strength = Column(Float)
    inversion_category = Column(String)
    inversion_source = Column(String)
    # -- fire transport ----------------------------------------------------
    fire_count = Column(Integer)
    upwind_fire_count = Column(Integer)
    nearest_fire_distance_km = Column(Float)
    fire_impact_score = Column(Float)
    wind_alignment_pct = Column(Float)
    fire_transport_direction = Column(String)   # compass the plume would travel toward
    fire_transport_time_hours = Column(Float)
    fire_transport_influence = Column(Float)
    # -- nine coupling features (0..1, None = data unavailable) -----------
    dispersion_potential = Column(Float)
    accumulation_potential = Column(Float)
    inversion_trapping_potential = Column(Float)
    pollution_stagnation_index = Column(Float)
    aerosol_accumulation_potential = Column(Float)
    regional_transport_potential = Column(Float)
    ozone_photochemical_potential = Column(Float)
    meteorology_pollution_interaction = Column(Float)
    # -- labels + provenance -----------------------------------------------
    coupling_state = Column(String)
    coupling_domains = Column(String)
    data_quality = Column(String)
    weather_reading_timestamp = Column(DateTime(timezone=True))
    pollution_reading_timestamp = Column(DateTime(timezone=True))
    __table_args__ = (
        UniqueConstraint("station_id", name="uq_coupling_state_station"),
        Index("ix_coupling_states_station_id", "station_id"),
    )
