export interface Station {
  id: number
  name: string
  latitude: number
  longitude: number
  city: string
  state?: string | null
}

export interface PollutionReading {
  station_id: number
  station: string
  city: string | null
  state: string | null
  timestamp: string
  pm25: number | null
  pm10: number | null
  o3: number | null
  no2: number | null
  so2: number | null
  co: number | null
  aqi: number | null
}

export interface PollutionIngestSummary {
  records_fetched: number
  observations_normalized: number
  stations_processed: number
  inserted: number
  updated: number
  skipped: number
  station_created: number
  station_updated: number
  errors: string[]
}

export interface DataImportSummary {
  dataset: 'weather' | 'pollution'
  rows: number
  inserted: number
  updated: number
  unchanged: number
  unknown_stations: string[]
  errors: string[]
}

export interface CurrentAQI {
  station: string
  timestamp: string
  pm25: number | null
  pm10: number | null
  o3: number | null
  no2: number | null
  so2: number | null
  co: number | null
  aqi: number | null
  aqi_category: string
  dominant_pollutant: string
}

export interface ForecastPoint {
  timestamp: string
  horizon_hours: number
  pm25_pred: number | null
  pm10_pred: number | null
  o3_pred: number | null
  no2_pred: number | null
  so2_pred?: number | null
  co_pred?: number | null
  aqi_pred: number | null
  aqi_category: string
  coupling_stability?: number | null
  coupling_mode?: string | null
}

export interface CouplingDiagnostics {
  aod_est: number
  radiation_transmittance: number
  pbl_suppression_factor: number
  corrected_pbl_height: number
  stability_coupling_index: number
  feedback_multiplier: number
}

export interface CoupledForecastPoint extends ForecastPoint {
  so2_pred: number | null
  co_pred: number | null
  coupling: CouplingDiagnostics
  pbl_effective: number
  coupling_stability: number
}

export interface CoupledForecastResult {
  station: string
  generated_at: string
  horizons: number[]
  mode: string
  coupled: CoupledForecastPoint[]
  uncoupled: ForecastPoint[]
  feedback_path: Array<{
    t_plus: number
    pm25: number
    pbl_effective: number
    stability: number
    feedback_multiplier: number
  }>
  saved_points: number
}

export interface GridCell {
  lat: number
  lon: number
  aqi: number
  aqi_category: string
}

export interface GridForecast {
  horizon_hours: number
  step_deg: number
  bounds: Record<string, number>
  wind_dir: number | null
  wind_speed: number | null
  cells: GridCell[]
  grid_size: number[]
  extent: { lats_min: number; lats_max: number; lons_min: number; lons_max: number }
}

export interface DispersionFrame {
  hour: number
  hour_of_day: number
  wind_speed: number
  wind_dir_deg: number
  pbl_height: number
  precip_mm: number
  coupling: {
    mean_pm25: number
    stability_coupling_index: number
    pbl_suppression_factor: number
    corrected_pbl_height: number
  }
  aqi_mean: number
  aqi_max: number
  cells: GridCell[]
}

export interface DispersionForecast {
  mode: string
  horizon_hours: number
  start_hour: number
  domain: Record<string, number>
  step_deg: number
  wx: { wind_speed: number; wind_direction: number; pbl_height: number; precipitation: number }
  fire_count: number
  fires: Array<{ lat: number; lon: number; frp: number; confidence: string | null; satellite: string | null }>
  dt_used: number
  steps_per_hour: number
  /** Every hour the solver produced, even when the request asked for a subset. */
  frame_hours_available?: number[]
  frames: DispersionFrame[]
}

export interface WeatherData {
  station: string
  timestamp: string
  temperature: number | null
  humidity: number | null
  pressure_msl: number | null
  surface_pressure: number | null
  wind_speed: number | null
  wind_direction: number | null
  precipitation: number | null
  cloud_cover: number | null
  pbl_height: number | null
}

export interface InversionData {
  station: string
  timestamp: string
  pbl_height: number | null
  inversion_detected: boolean
  inversion_strength: string
  trapping_risk: string
  inversion_strength_score?: number | null
  inversion_category?: string | null
  inversion_source?: string | null
  inversion_base_pressure?: number | null
  inversion_top_pressure?: number | null
  strongest_layer_gradient?: number | null
  low_pbl_flag?: boolean | null
  pbl_category?: string | null
  dispersion_condition?: string | null
}

export interface CouplingDiagnostics {
  aod_est: number
  radiation_transmittance: number
  pbl_suppression_factor: number
  corrected_pbl_height: number
  stability_coupling_index: number
  feedback_multiplier: number
  coupling_strength: string
}

export interface CouplingData {
  station: string
  timestamp: string
  pm25: number | null
  pbl_height: number | null
  wind_speed: number | null
  diag: CouplingDiagnostics
  narrative: string[]
}

export interface CouplingFeature {
  value: number | null
  available: boolean
  basis: string
}

export interface CouplingFeaturesResponse {
  station: string
  timestamp: string
  features: Record<string, CouplingFeature>
  inputs: Record<string, number | string | boolean | null>
  methodology: { note: string; constants: Record<string, number> }
  provenance: Record<string, string | number | boolean | null>
  coupling_state?: string
  coupling_domains?: string
  data_quality?: string
}

export interface ForecastHorizonContext {
  horizon_hours: number
  target_timestamp: string | null
  weather_match_timestamp: string | null
  temperature_c: number | null
  humidity_pct: number | null
  pressure_hpa: number | null
  wind_speed_mps: number | null
  wind_direction_deg: number | null
  pbl_height_m: number | null
  inversion_detected: boolean | null
  inversion_strength: number | null
  inversion_category: string | null
  inversion_source: string | null
  dispersion_potential: number | null
  accumulation_potential: number | null
  inversion_trapping_potential: number | null
  pollution_stagnation_index: number | null
  fire_transport_influence: number | null
  regional_transport_potential: number | null
  ozone_photochemical_potential: number | null
  meteorology_pollution_interaction: number | null
}

export interface ForecastContextResponse {
  station: string
  generated_at: string
  units: Record<string, string>
  regional_note: string
  horizons: ForecastHorizonContext[]
}

export interface FireActivity {
  total_fires: number
  high_confidence_fires: number
  mean_frp: number
  region: string
  date: string
}

export interface FireHotspot {
  lat: number
  lon: number
  frp?: number | null
  confidence?: string | null
  acq_date?: string | null
  synthetic?: boolean
  source?: string | null
}

export interface FireHotspotsResponse {
  region: string
  hotspots: FireHotspot[]
}

export interface PlumeRisk {
  risk_level: string
  risk_score: number
  fire_count: number
  transport_direction: string
  wind_speed: number
  distance_nearest_fire: number
  confidence: number
  factors: string[]
  wind_alignment_pct?: number | null
  transport_time_hours?: number | null
  transport_risk?: number | null
  transport_risk_level?: string | null
  stubble_impact_score?: number | null
  estimated_pm25_contribution_ugm3?: number | null
  synthetic_fire_count?: number | null
  fire_basis?: string | null
}

export interface PollutionEventFactor {
  factor: string
  status: string
  value: number | null
  evidence: string | null
  description: string | null
}

export interface PollutionEventConfidence {
  label: string
  basis: string
  margin_ugm3: number | null
  conformal_half_width_ugm3: number | null
  lower_bound_ugm3: number | null
  upper_bound_ugm3: number | null
  test_r2: number | null
  coverage_target: number | null
  uncertainty_method: string | null
}

export interface PollutionEvent {
  event_type: string
  station: string | null
  status: string
  start_time: string
  end_time: string | null
  expected_peak: number | null
  expected_peak_time: string | null
  expected_trough: number | null
  expected_trough_time: string | null
  severity: string
  severity_label: string
  confidence: PollutionEventConfidence
  contributing_factors: PollutionEventFactor[]
}

export interface PollutionEventsCurrent {
  station: string
  station_id: number
  generated_at: string
  release_time: string
  data_as_of: string | null
  model: string
  forecast_strategy: string | null
  uncertainty_method: string | null
  coverage_target: number | null
  horizon_hours: number
  baseline_pm25_ugm3: number | null
  forecast_peak_pm25_ugm3: number | null
  events: PollutionEvent[]
  atmosphere: Record<string, any> | null
  methodology: Record<string, any>
  notes: string[] | null
}

export interface DataQualityResponse {
  status: string
  generated_at: string
  station_count: number
  stations_with_forecasts: number
  forecast_coverage: Record<string, number>
  latest_pollution_reading: string | null
  latest_weather_reading: string | null
  latest_fire_reading: string | null
  tables: Record<string, { total: number; missing_values: Record<string, number> }>
  recommendations: string[]
}

export interface Explanation {
  station: string
  timestamp: string
  prediction: Record<string, any>
  top_features: Array<{
    feature: string
    importance: number
    direction: string
    description: string
  }>
  natural_language: string[]
}

export interface ShapContribution {
  feature: string
  value: number | null
  shap_value: number
  share_of_abs_contributions_pct: number
  direction: 'positive' | 'negative' | 'zero'
  description: string
}

export interface ForecastExplanation {
  forecast_id: number
  station: string
  station_id: number
  model: string
  explanation_method: string
  horizon_hours: number
  forecast_timestamp: string
  forecast_pm25: number
  stored_pm25_pred: number | null
  base_value: number
  summary: string
  top_positive_drivers: ShapContribution[]
  top_negative_drivers: ShapContribution[]
  contributions_by_magnitude: ShapContribution[]
  n_features: number
  feature_values_used: Record<string, any> | null
  data_as_of: string
  generated_at: string
  test_metrics: Record<string, any> | null
}

export interface Alert {
  id: number
  station: string
  alert_level: string
  title: string
  description: string
  forecast_horizon_hours: number | null
  factors: string | null
  recommendation: string | null
  created_at: string
}

export interface ModelMetric {
  model_name: string
  pollutant: string
  horizon_hours: number
  mae: number | null
  rmse: number | null
  r2: number | null
  mape: number | null
  test_period_start: string | null
  test_period_end: string | null
}

export interface StationAQISummary {
  name: string
  aqi: number | null
  aqi_category: string
  dominant_pollutant: string | null
}

export interface SummaryResponse {
  generated_at: string
  stations: number
  stations_with_readings: number
  ncr_avg_aqi: number | null
  worst_station: StationAQISummary | null
  best_station: StationAQISummary | null
  active_fires_24h: number
  open_alerts: number
  models_trained: number
  forecast_coverage: {
    stations_with_forecast: number
    latest_forecast_at: string | null
  }
  data_mode?: string
  data_mode_note?: string
  /**
   * Timestamp of the newest pollution observation the summary was built from,
   * and its age at the moment the summary was generated. The backend averages
   * each station's most recent stored reading rather than a fixed recency
   * window, so these two fields are the only way to tell whether the reported
   * `ncr_avg_aqi` is a live reading or a historical one.
   */
  latest_observation_at?: string | null
  observation_age_hours?: number | null
}

export interface SystemEngineStatus {
  source: string
  status: string
  note: string
}

export interface SystemRunMode {
  environment: string
  live_refresh_enabled: boolean
  live_refresh_interval_hours: number
  demo_hydrate_empty_db: boolean
  explanation: string
}

export interface SystemResponse {
  generated_at: string
  service: string
  database: string
  run_mode: SystemRunMode
  engines: Record<string, SystemEngineStatus>
  schema_version: string
}

export interface ModelPerformanceSplitRange {
  start: string | null
  end: string | null
  n_rows: number
}

export interface ModelPerformanceMetrics {
  mae: number | null
  rmse: number | null
  r2: number | null
  mape?: number | null
  nmae?: number | null
  n: number
}

export interface ModelCoverageInfo {
  coverage_target: number | null
  quantile: number | null
  n_cal: number | null
  test_coverage: number | null
  n_test: number | null
}

export interface ModelHorizonPerformance {
  horizon_hours: number
  n_train: number
  n_val: number
  n_test: number
  test_period_start: string | null
  test_period_end: string | null
  metrics: Record<string, ModelPerformanceMetrics>
  coverage?: {
    target: number | null
    models: Record<string, ModelCoverageInfo>
  } | null
}

export interface ModelPerformanceResponse {
  schema_version: number
  target: string
  model_dir: string
  data_source: string
  generated_at: string
  feature_count: number
  features: string[]
  horizons: number[]
  evaluated_models: string[]
  split_type: string
  split_ratios: number[] | null
  split_ranges: Record<string, ModelPerformanceSplitRange>
  results: ModelHorizonPerformance[]
}

export interface Pm25ForecastPoint {
  timestamp: string
  forecast_horizon: number
  predicted_pm25: number
  pm25_lower_bound: number
  pm25_upper_bound: number
  baseline_persistence: number | null
  test_mae: number | null
  test_rmse: number | null
  test_r2: number | null
  test_n: number | null
}

export interface Pm25ForecastResponse {
  station: string
  station_id: number
  model: string
  forecast_strategy: string
  uncertainty_method: string
  coverage_target: number | null
  feature_version: string | null
  release_time: string
  data_as_of: string | null
  generated_at: string
  requested_hours: number
  served_horizons: number[]
  context: Record<string, unknown> | null
  forecasts: Pm25ForecastPoint[]
}

export interface WindCondition {
  wind_speed_mps: number | null
  wind_direction_deg: number | null
  compass_from: string | null
  category: string
  label: string
  normalized: number | null
  provenance: string
  notes: string[] | null
}

export interface PblCondition {
  pbl_height_m: number | null
  category: string
  label: string
  normalized: number | null
  provenance: string
  notes: string[] | null
}

export interface VentilationCondition {
  ventilation_coefficient_m2s: number | null
  category: string
  label: string
  normalized: number | null
  provenance: string
  notes: string[] | null
}

export interface InversionIndicator {
  detected: boolean | null
  category: string
  strength: number | null
  source: string
  provenance: string
  base_pressure_hpa: number | null
  top_pressure_hpa: number | null
  strongest_gradient_k100hpa: number | null
  lapse_unit: string | null
  profile_available: boolean | null
  pbl_category: string | null
  dispersion_condition: string | null
  normalized: number | null
  limitations: string[] | null
}

export interface TrappingIndicator {
  score: number | null
  category: string
  label: string
  normalized: number | null
  provenance: string
  factors: string[] | null
}

export interface AtmosphereFeatures {
  wind: number | null
  pbl: number | null
  ventilation: number | null
  inversion: number | null
  trapping: number | null
}

export interface StationAtmosphere {
  station: string
  station_id: number
  analyzed_at: string
  weather_timestamp: string | null
  pollution_timestamp: string | null
  weather_age_hours: number | null
  pollution_age_hours: number | null
  flags: string[] | null
  inputs: Record<string, unknown>
  input_basis: Record<string, unknown> | null
  wind: WindCondition
  pbl: PblCondition
  ventilation: VentilationCondition
  inversion: InversionIndicator | null
  trapping: TrappingIndicator
  features: AtmosphereFeatures
}

export interface AtmosphereCurrentResponse {
  generated_at: string
  region: string
  methodology: Record<string, unknown>
  summary: Record<string, unknown>
  stations: StationAtmosphere[]
}

export interface TransportRiskResponse {
  risk_score: number | null
  risk_level: string
  main_contributing_factors: string[]
  upwind_fire_count: number
  fire_count: number
  dominant_wind_direction: Record<string, unknown>
  atmospheric_condition: Record<string, unknown>
  generated_at: string
  region: string
  disclaimer: string
  inputs: Record<string, unknown>
  components: Record<string, unknown>
  methodology: Record<string, unknown>
  station_detail: Array<Record<string, unknown>>
}

export interface GrapStage {
  stage: number
  title: string
  aqi_range_low: number | null
  aqi_range_high: number | null
  categories: string[]
  color: string
  summary: string
  measures: string[]
}

export interface GrapStagesResponse {
  stages: GrapStage[]
}

export interface GrapAssessment {
  assessed_at: string
  stage: number
  status: 'ACTIVE' | 'NOT_INVOKED'
  title: string
  color: string
  aqi: number | null
  aqi_category: string | null
  dominant_pollutant: string | null
  inversion_strength: number | null
  inversion_note: string | null
  fire_mean_frp_mw: number | null
  fire_note: string | null
  advisory: string
  rationale: string[]
  measures: string[]
  source: string
}

export interface AuthUser {
  id: number
  email: string
  name: string
  role: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  user: AuthUser
  expires_in: number
}

export interface DemoCredentials {
  email: string
  password: string
  name: string
  role: string
}

export interface ScenarioPerturbationInput {
  mode: 'absolute' | 'relative'
  value: number
}

export interface ScenarioFireInput {
  mode: 'relative'
  multiplier: number
}

export interface ScenarioInversionInput {
  detected: boolean
  strength: number | null
}

export interface ScenarioChangesInput {
  wind_speed?: ScenarioPerturbationInput
  wind_direction?: ScenarioPerturbationInput
  pbl_height?: ScenarioPerturbationInput
  fire_activity?: ScenarioFireInput
  inversion?: ScenarioInversionInput
}

export interface ScenarioAnalysisRequest {
  station_name: string | null
  hours: number
  changes: ScenarioChangesInput
}

export interface ScenarioForecastPoint {
  timestamp: string
  forecast_horizon: number
  predicted_pm25: number
  pm25_lower_bound: number | null
  pm25_upper_bound: number | null
}

export interface ScenarioForecastSummary {
  peak_pm25: number | null
  mean_pm25: number | null
  forecasts: ScenarioForecastPoint[]
}

export interface ScenarioDiffPoint {
  timestamp: string
  forecast_horizon: number
  baseline_pm25: number
  scenario_pm25: number
  difference_pm25: number
  baseline_lower_bound: number | null
  baseline_upper_bound: number | null
  scenario_lower_bound: number | null
  scenario_upper_bound: number | null
}

export interface ScenarioDifference {
  peak_difference_pm25: number | null
  mean_difference_pm25: number | null
  points: ScenarioDiffPoint[]
}

export interface ScenarioChangeEffect {
  feature: string
  unit: string | null
  baseline_value: number | null
  scenario_value: number | null
}

export interface ScenarioAnalysisResponse {
  label: string
  disclaimer: string
  station: string
  station_id: number
  release_time: string
  data_as_of: string | null
  generated_at: string
  model: string
  forecast_strategy: string | null
  uncertainty_method: string | null
  coverage_target: number | null
  horizon_hours: number
  served_horizons: number[]
  value_kinds: Record<string, string>
  observed: {
    pm25_last_observed_ugm3: number | null
    pm25_lag1_anchor_ugm3: number | null
    timestamp: string | null
  }
  baseline_forecast: ScenarioForecastSummary
  scene_forecast: ScenarioForecastSummary
  difference: ScenarioDifference
  input_changes: ScenarioChangeEffect[]
  notes: string[]
  data_integrity: Record<string, boolean | string>
}
