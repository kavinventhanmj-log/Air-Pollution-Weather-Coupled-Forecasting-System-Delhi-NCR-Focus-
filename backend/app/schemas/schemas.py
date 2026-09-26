from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ImdForecastDay(BaseModel):
    day: int
    date: str | None = None
    max_temp_c: float | None = None
    min_temp_c: float | None = None
    condition: str = ""


class ImdForecastResponse(BaseModel):
    available: bool
    reasons: list[str] = Field(default_factory=list)
    station_id: str | None = None
    station_name: str | None = None
    fetched_at: str | None = None
    source: str | None = None
    days: list[ImdForecastDay] = Field(default_factory=list)


class StationResponse(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float
    city: str
    state: str | None = None

class PollutionReadingResponse(BaseModel):
    station_id: int
    station: str
    city: str | None = None
    state: str | None = None
    timestamp: datetime
    pm25: float | None
    pm10: float | None
    o3: float | None
    no2: float | None
    so2: float | None
    co: float | None
    aqi: int | None
    data_source: str | None = None

class PollutionIngestResponse(BaseModel):
    records_fetched: int
    observations_normalized: int
    stations_processed: int
    inserted: int
    updated: int
    skipped: int
    station_created: int
    station_updated: int
    errors: list[str]

class StationPollutionCoverage(BaseModel):
    station_id: int
    station: str
    city: str | None = None
    readings: int
    last_timestamp: datetime | None = None
    hours_since_last: float | None = None
    history_days: float | None = None
    sufficiency: str
    sources: list[str] = Field(default_factory=list)

class PollutionCoverageResponse(BaseModel):
    stations_total: int
    stations_with_readings: int
    stations_recent: int
    stations_insufficient: int
    coverage_pct: float
    stations: list[StationPollutionCoverage]

class StationAQISummary(BaseModel):
    name: str
    aqi: int | None
    aqi_category: str
    dominant_pollutant: str | None

class SummaryResponse(BaseModel):
    generated_at: datetime
    stations: int
    stations_with_readings: int
    ncr_avg_aqi: float | None
    worst_station: StationAQISummary | None
    best_station: StationAQISummary | None
    active_fires_24h: int
    open_alerts: int
    models_trained: int
    forecast_coverage: dict
    data_mode: str = "static_archive"
    data_mode_note: str = ""

class CurrentAQI(BaseModel):
    station: str
    timestamp: datetime
    pm25: float | None
    pm10: float | None
    o3: float | None
    no2: float | None
    so2: float | None
    co: float | None
    aqi: int | None
    aqi_category: str
    dominant_pollutant: str

class ForecastPoint(BaseModel):
    timestamp: datetime
    horizon_hours: int
    pm25_pred: float | None
    pm10_pred: float | None
    o3_pred: float | None
    no2_pred: float | None
    so2_pred: float | None = None
    co_pred: float | None = None
    aqi_pred: int | None
    aqi_category: str
    dominant_pollutant: str | None = None
    coupling_stability: float | None = None
    coupling_mode: str | None = None

class GridForecastPoint(BaseModel):
    timestamp: datetime
    horizon_hours: int
    aqi_pred: int | None
    aqi_category: str
    lat: float | None = None
    lon: float | None = None

class ForecastGenerateRequest(BaseModel):
    station_name: str | None = None
    horizons: list[int] = Field(default_factory=lambda: [1, 6, 12, 24, 48, 72])

class ForecastGenerateResponse(BaseModel):
    station: str
    generated_at: datetime
    horizons: list[int]
    forecasts: list[ForecastPoint]
    # "Pooled" flags: set when the station's own pollution history is too thin
    # for a stable local model, so the forecast is driven by the NCR regional
    # composite series instead (honest fallback, never invented values).
    pooled_features: bool = False
    local_readings: int = 0
    history_days: float | None = None

class ForecastComparisonPoint(BaseModel):
    timestamp: datetime
    actual_aqi: int | None
    predicted_aqi: int | None
    actual_pm25: float | None
    predicted_pm25: float | None
    delta: float | None

class ForecastComparisonResponse(BaseModel):
    station: str
    points: list[ForecastComparisonPoint]

class Pm25ForecastPoint(BaseModel):
    timestamp: datetime
    forecast_horizon: int
    predicted_pm25: float
    pm25_lower_bound: float
    pm25_upper_bound: float
    baseline_persistence: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_r2: float | None = None
    test_n: int | None = None

class Pm25ForecastResponse(BaseModel):
    station: str
    station_id: int
    model: str
    forecast_strategy: str
    uncertainty_method: str
    coverage_target: float | None = None
    feature_version: str | None = None
    release_time: datetime
    data_as_of: datetime | None = None
    generated_at: datetime
    requested_hours: int
    served_horizons: list[int]
    context: dict | None = None
    forecasts: list[Pm25ForecastPoint]

class Pm25ModelCardResponse(BaseModel):
    station: str | None = None
    model: str
    forecast_strategy: str
    uncertainty_method: str
    coverage_target: float | None = None
    n_horizons: int
    horizons: list[int]
    n_features: int | None = None
    available: bool
    message: str | None = None

class ShapContribution(BaseModel):
    feature: str
    value: float | None = None
    shap_value: float
    share_of_abs_contributions_pct: float
    direction: str
    description: str

class Pm25ForecastExplanationResponse(BaseModel):
    station: str
    station_id: int
    model: str
    explanation_method: str
    horizon_hours: int
    forecast_timestamp: datetime
    forecast_pm25: float
    base_value: float
    summary: str
    top_positive_drivers: list[ShapContribution]
    top_negative_drivers: list[ShapContribution]
    contributions_by_magnitude: list[ShapContribution]
    n_features: int
    feature_values_used: dict | None = None
    data_as_of: datetime
    generated_at: datetime
    test_metrics: dict | None = None

class ForecastExplanationResponse(Pm25ForecastExplanationResponse):
    """SHAP explanation keyed by a stored ``forecasts`` row id.

    ``forecast_pm25`` is the model-derived prediction on the exact feature row
    being explained (forecast = base_value + sum of shap values);
    ``stored_pm25_pred`` is the value persisted on the original forecast row.
    """
    forecast_id: int
    stored_pm25_pred: float | None = None

class PollutionEventFactor(BaseModel):
    """One contributing factor of an event, with real evidence from stored data."""
    factor: str
    status: str
    value: float | None = None
    evidence: str | None = None
    description: str | None = None

class PollutionEventConfidence(BaseModel):
    """Model-uncertainty-based confidence for an event (from conformal bounds)."""
    label: str
    basis: str
    margin_ugm3: float | None = None
    conformal_half_width_ugm3: float | None = None
    lower_bound_ugm3: float | None = None
    upper_bound_ugm3: float | None = None
    test_r2: float | None = None
    coverage_target: float | None = None
    uncertainty_method: str | None = None

class PollutionEvent(BaseModel):
    """A detected pollution event (surge / relief / high-risk episode).

    ``expected_peak``/``expected_trough`` and their timestamps are the model's
    predicted extreme over the event window; ``severity`` labels the tier of
    that predicted extreme vs documented CPCB/NAAQS thresholds; contributing
    factors carry the actual stored values that triggered each classification.
    """
    event_type: str
    station: str | None = None
    status: str
    start_time: datetime
    end_time: datetime | None = None
    expected_peak: float | None = None
    expected_peak_time: datetime | None = None
    expected_trough: float | None = None
    expected_trough_time: datetime | None = None
    severity: str
    severity_label: str
    confidence: PollutionEventConfidence
    contributing_factors: list[PollutionEventFactor]

class PollutionEventsCurrentResponse(BaseModel):
    """Current/upcoming pollution events for one station + methodology."""
    station: str
    station_id: int
    generated_at: datetime
    release_time: datetime
    data_as_of: datetime | None = None
    model: str
    forecast_strategy: str | None = None
    uncertainty_method: str | None = None
    coverage_target: float | None = None
    horizon_hours: int
    baseline_pm25_ugm3: float | None = None
    forecast_peak_pm25_ugm3: float | None = None
    events: list[PollutionEvent]
    atmosphere: dict | None = None
    methodology: dict
    notes: list[str] | None = None

class WeatherResponse(BaseModel):
    station: str
    timestamp: datetime
    temperature: float | None
    humidity: float | None
    pressure_msl: float | None
    wind_speed: float | None
    wind_direction: float | None
    precipitation: float | None
    cloud_cover: float | None

class WeatherDetailResponse(BaseModel):
    station: str
    station_id: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    reading_latitude: float | None = None
    reading_longitude: float | None = None
    timestamp: datetime
    temperature: float | None
    humidity: float | None
    pressure_msl: float | None
    surface_pressure: float | None
    wind_speed: float | None
    wind_direction: float | None
    precipitation: float | None
    cloud_cover: float | None
    pbl_height: float | None

class InversionResponse(BaseModel):
    station: str
    timestamp: datetime
    pbl_height: float | None
    inversion_detected: bool
    inversion_strength: str
    trapping_risk: str
    inversion_strength_score: float | None = None
    inversion_category: str | None = None
    inversion_source: str | None = None
    inversion_base_pressure: float | None = None
    inversion_top_pressure: float | None = None
    inversion_base_height_m: float | None = None
    inversion_top_height_m: float | None = None
    inversion_thickness_m: float | None = None
    inversion_thickness_hpa: float | None = None
    strongest_layer_gradient: float | None = None
    low_pbl_flag: bool | None = None
    pbl_category: str | None = None
    dispersion_condition: str | None = None
    # Duration / persistence of the current episode, measured over the stored
    # hourly history. Measured window and sample count travel with the values so
    # a short archive is never presented as a long episode.
    inversion_duration_h: float = 0.0
    inversion_persistence: float | None = None
    inversion_persistence_window_h: int = 24
    inversion_measured_window_h: float = 0.0
    inversion_episode_samples: int = 0
    inversion_episode_onset: datetime | None = None
    inversion_history_sufficient: bool = False

class WindCondition(BaseModel):
    wind_speed_mps: float | None = None
    wind_direction_deg: float | None = None
    compass_from: str | None = None
    category: str
    label: str
    normalized: float | None = None
    provenance: str
    notes: list[str] | None = None

class PblCondition(BaseModel):
    pbl_height_m: float | None = None
    category: str
    label: str
    normalized: float | None = None
    provenance: str
    notes: list[str] | None = None

class VentilationCondition(BaseModel):
    ventilation_coefficient_m2s: float | None = None
    category: str
    label: str
    normalized: float | None = None
    provenance: str
    notes: list[str] | None = None

class InversionIndicator(BaseModel):
    detected: bool | None = None
    category: str
    strength: float | None = None
    source: str
    provenance: str
    base_pressure_hpa: float | None = None
    top_pressure_hpa: float | None = None
    strongest_gradient_k100hpa: float | None = None
    lapse_unit: str | None = None
    profile_available: bool | None = None
    pbl_category: str | None = None
    dispersion_condition: str | None = None
    normalized: float | None = None
    limitations: list[str] | None = None

class TrappingIndicator(BaseModel):
    score: float | None = None
    category: str
    label: str
    normalized: float | None = None
    provenance: str
    factors: list[str] | None = None

class NormalizedFeatures(BaseModel):
    """Normalized (0..1) features for later ML use."""
    wind: float | None = None
    pbl: float | None = None
    ventilation: float | None = None
    inversion: float | None = None
    trapping: float | None = None

class StationAtmosphere(BaseModel):
    station: str
    station_id: int
    analyzed_at: datetime
    weather_timestamp: datetime | None = None
    pollution_timestamp: datetime | None = None
    weather_age_hours: float | None = None
    pollution_age_hours: float | None = None
    flags: list[str] | None = None
    inputs: dict
    input_basis: dict | None = None
    wind: WindCondition
    pbl: PblCondition
    ventilation: VentilationCondition
    inversion: InversionIndicator | None = None
    trapping: TrappingIndicator
    features: NormalizedFeatures

class AtmosphereCurrentResponse(BaseModel):
    generated_at: datetime
    region: str
    methodology: dict
    summary: dict
    stations: list[StationAtmosphere]

class CouplingDiagnostics(BaseModel):
    aod_est: float
    radiation_transmittance: float
    pbl_suppression_factor: float
    corrected_pbl_height: float
    stability_coupling_index: float
    feedback_multiplier: float
    coupling_strength: str

class CouplingResponse(BaseModel):
    station: str
    timestamp: datetime
    pm25: float | None
    pbl_height: float | None
    wind_speed: float | None
    diag: CouplingDiagnostics
    narrative: list[str]


# SIH26082: deterministic, real-data coupling-feature packet.
class CouplingFeature(BaseModel):
    value: float | None
    available: bool
    basis: str


class CouplingFeaturesResponse(BaseModel):
    station: str
    timestamp: datetime
    features: dict[str, CouplingFeature]
    inputs: dict
    methodology: dict
    provenance: dict
    coupling_state: str
    coupling_domains: str
    data_quality: str


class CouplingStateSnapshot(BaseModel):
    """One persisted coupling snapshot (SIH26082 Phase 30 persistence)."""
    station: str
    station_id: int | None = None
    computed_at: datetime | None = None
    wind_speed_mps: float | None = None
    wind_direction_deg: float | None = None
    pbl_height_m: float | None = None
    inversion_detected: bool | None = None
    inversion_strength: float | None = None
    inversion_category: str | None = None
    inversion_source: str | None = None
    fire_count: int | None = None
    upwind_fire_count: int | None = None
    nearest_fire_distance_km: float | None = None
    fire_impact_score: float | None = None
    wind_alignment_pct: float | None = None
    fire_transport_direction: str | None = None
    fire_transport_time_hours: float | None = None
    fire_transport_influence: float | None = None
    dispersion_potential: float | None = None
    accumulation_potential: float | None = None
    inversion_trapping_potential: float | None = None
    pollution_stagnation_index: float | None = None
    aerosol_accumulation_potential: float | None = None
    regional_transport_potential: float | None = None
    ozone_photochemical_potential: float | None = None
    meteorology_pollution_interaction: float | None = None
    coupling_state: str
    coupling_domains: str
    data_quality: str
    weather_reading_timestamp: datetime | None = None
    pollution_reading_timestamp: datetime | None = None


class CouplingStateListResponse(BaseModel):
    generated_at: datetime
    count: int
    states: list[CouplingStateSnapshot]


class ForecastHorizonContext(BaseModel):
    horizon_hours: int
    target_timestamp: datetime | None
    weather_match_timestamp: datetime | None
    temperature_c: float | None
    humidity_pct: float | None
    pressure_hpa: float | None
    wind_speed_mps: float | None
    wind_direction_deg: float | None
    pbl_height_m: float | None
    inversion_detected: bool | None
    inversion_strength: float | None
    inversion_category: str | None
    inversion_source: str | None
    dispersion_potential: float | None
    accumulation_potential: float | None
    inversion_trapping_potential: float | None
    pollution_stagnation_index: float | None
    fire_transport_influence: float | None
    regional_transport_potential: float | None
    ozone_photochemical_potential: float | None
    meteorology_pollution_interaction: float | None


class ForecastContextResponse(BaseModel):
    station: str
    generated_at: datetime
    units: dict
    regional_note: str
    horizons: list[ForecastHorizonContext]

class FireActivityResponse(BaseModel):
    total_fires: int
    high_confidence_fires: int
    mean_frp: float
    region: str
    date: datetime

class FireHotspot(BaseModel):
    lat: float
    lon: float
    frp: float | None = None
    confidence: str | None = None
    acq_date: datetime | None = None

class FireHotspotsResponse(BaseModel):
    region: str
    hotspots: list[FireHotspot]

class FireEvent(BaseModel):
    """One stored FIRMS fire observation (facts only — no attribution)."""
    id: int
    latitude: float
    longitude: float
    acq_date: datetime
    confidence: str | None = None
    frp: float | None = None
    brightness: float | None = None
    satellite: str | None = None
    instrument: str | None = None
    daynight: str | None = None

class FiresLatestResponse(BaseModel):
    region: str
    generated_at: datetime
    count: int
    fires: list[FireEvent]

class TransportRiskCurrentResponse(BaseModel):
    """Estimated Regional Pollution Transport Risk — transparent 0-100 estimate."""
    risk_score: int | None
    risk_level: str
    main_contributing_factors: list[str]
    upwind_fire_count: int
    fire_count: int
    dominant_wind_direction: dict
    atmospheric_condition: dict
    generated_at: datetime
    region: str
    disclaimer: str
    inputs: dict
    components: dict
    methodology: dict
    station_detail: list[dict]

class PlumeRiskResponse(BaseModel):
    risk_level: str
    risk_score: float
    fire_count: int
    transport_direction: str
    wind_speed: float
    distance_nearest_fire: float
    confidence: float
    factors: list[str]
    wind_alignment_pct: float | None = None
    transport_time_hours: float | None = None
    transport_risk: float | None = None
    transport_risk_level: str | None = None
    stubble_impact_score: float | None = None
    estimated_pm25_contribution_ugm3: float | None = None

class TransportDirectionResponse(BaseModel):
    station: str
    from_direction: str
    to_direction: str
    label: str
    wind_speed: float | None
    wind_direction: float | None
    basis: str

class ExplanationResponse(BaseModel):
    station: str
    timestamp: datetime
    prediction: dict
    top_features: list[dict]
    natural_language: list[str]

class AlertResponse(BaseModel):
    id: int
    station: str
    alert_level: str
    title: str
    description: str
    forecast_horizon_hours: int | None
    factors: str | None
    recommendation: str | None
    created_at: datetime

class ModelMetricCreate(BaseModel):
    model_name: str
    pollutant: str
    horizon_hours: int
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    mape: float | None = None
    test_period_start: datetime | None = None
    test_period_end: datetime | None = None

class ModelMetricResponse(BaseModel):
    id: int | None = None
    model_name: str
    pollutant: str
    horizon_hours: int
    mae: float | None
    rmse: float | None
    r2: float | None
    mape: float | None
    test_period_start: datetime | None
    test_period_end: datetime | None
    trained_at: datetime | None = None

class SplitRangeInfo(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    n_rows: int

class PerformanceMetrics(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    mape: float | None = None
    nmae: float | None = None
    n: int = 0

class HorizonPerformance(BaseModel):
    horizon_hours: int
    n_train: int
    n_val: int
    n_test: int
    test_period_start: datetime | None = None
    test_period_end: datetime | None = None
    metrics: dict[str, PerformanceMetrics]

class ModelPerformanceResponse(BaseModel):
    schema_version: int = 1
    target: str
    model_dir: str
    data_source: str
    generated_at: datetime
    feature_count: int
    features: list[str]
    horizons: list[int]
    evaluated_models: list[str]
    split_type: str
    split_ratios: list[float] | None = None
    split_ranges: dict[str, SplitRangeInfo]
    results: list[HorizonPerformance]


class ScenarioPerturbation(BaseModel):
    """A controlled change to a continuous environmental input.

    ``absolute`` replaces the stored baseline value outright. ``relative``
    multiplies the baseline value (2.0 = double, 0.5 = half); for
    ``wind_direction`` only, ``relative`` adds a rotation in degrees.
    """

    mode: Literal["absolute", "relative"]
    value: float


class FireActivityChange(BaseModel):
    """Regional fire-activity change expressed as an FRP intensity multiplier.

    Only the FRP-weighted intensity terms are scaled; detected-fire counts and
    geometry-derived terms are left at their real observed values.
    """

    mode: Literal["relative"] = "relative"
    multiplier: float = Field(default=1.0, gt=0.0)


class InversionChange(BaseModel):
    detected: bool
    strength: float | None = Field(default=None, ge=0.0, le=1.0)


class ScenarioChanges(BaseModel):
    wind_speed: ScenarioPerturbation | None = None
    wind_direction: ScenarioPerturbation | None = None
    pbl_height: ScenarioPerturbation | None = None
    fire_activity: FireActivityChange | None = None
    inversion: InversionChange | None = None


class ScenarioAnalysisRequest(BaseModel):
    station_name: str | None = None
    hours: int = Field(default=72, ge=1, le=72)
    changes: ScenarioChanges = Field(
        ...,
        description="At least one field of ``changes`` must be set.",
    )


class ScenarioForecastSummary(BaseModel):
    peak_pm25: float | None = None
    mean_pm25: float | None = None
    forecasts: list[Pm25ForecastPoint]


class ScenarioDifferencePoint(BaseModel):
    timestamp: datetime
    forecast_horizon: int
    baseline_pm25: float
    scenario_pm25: float
    difference_pm25: float
    baseline_lower_bound: float | None = None
    baseline_upper_bound: float | None = None
    scenario_lower_bound: float | None = None
    scenario_upper_bound: float | None = None


class ScenarioDifference(BaseModel):
    peak_difference_pm25: float | None = None
    mean_difference_pm25: float | None = None
    points: list[ScenarioDifferencePoint]


class ScenarioChangeEffect(BaseModel):
    feature: str
    unit: str | None = None
    baseline_value: float | None = None
    scenario_value: float | None = None


class ScenarioObserved(BaseModel):
    pm25_last_observed_ugm3: float | None = None
    pm25_lag1_anchor_ugm3: float | None = None
    timestamp: datetime | None = None


class ScenarioAnalysisResponse(BaseModel):
    label: str = "SCENARIO ANALYSIS"
    disclaimer: str
    station: str
    station_id: int
    release_time: datetime
    data_as_of: datetime | None = None
    generated_at: datetime
    model: str
    forecast_strategy: str | None = None
    uncertainty_method: str | None = None
    coverage_target: float | None = None
    horizon_hours: int
    served_horizons: list[int]
    value_kinds: dict
    observed: ScenarioObserved
    baseline_forecast: ScenarioForecastSummary
    scene_forecast: ScenarioForecastSummary
    difference: ScenarioDifference
    input_changes: list[ScenarioChangeEffect]
    notes: list[str]
    data_integrity: dict


class GrapStageOut(BaseModel):
    stage: int
    title: str
    aqi_range_low: int | None
    aqi_range_high: int | None
    categories: list[str]
    color: str
    summary: str
    measures: list[str]


class GrapStagesResponse(BaseModel):
    stages: list[GrapStageOut]


class GrapAssessment(BaseModel):
    assessed_at: datetime
    stage: int
    status: Literal["ACTIVE", "NOT_INVOKED"]
    title: str
    color: str
    aqi: int | None
    aqi_category: str | None
    dominant_pollutant: str | None
    inversion_strength: float | None
    inversion_note: str | None
    fire_mean_frp_mw: float | None
    fire_note: str | None
    advisory: str
    rationale: list[str]
    measures: list[str]
    source: str
