# SIH26082 — Requirement Traceability Matrix

> **PS:** *Air Pollution–Weather Coupled Forecasting System (Delhi NCR Focus)* —
> Ministry of Earth Sciences / NCMRWF, Smart India Hackathon 2026.
>
> This matrix maps every explicit clause of SIH26082 to the module, API, UI,
> evidence and test that discharge it. Statuses are limited to `IMPLEMENTED` /
> `PARTIALLY IMPLEMENTED` / `NOT IMPLEMENTED`. Supplemental formal formulas are in
> [`docs/SCIENTIFIC_METHODOLOGY.md`](SCIENTIFIC_METHODOLOGY.md); the requirement-level
> audit is [`docs/SIH26082_IMPLEMENTATION_AUDIT.md`](SIH26082_IMPLEMENTATION_AUDIT.md).

## L1 — System-level clauses (every sentence of the PS)

| PS clause | Implementation | File / module | API | UI | Evidence / test | Status |
|-----------|----------------|---------------|-----|----|-----------------|--------|
| 72-hour air-quality forecast for Delhi NCR | Horizon models (persistence/RF/XGBoost) × horizons {1,6,12,24,48,72} × 6 pollutants, with Indian AQI computed from predicted concentrations | `ml/training/trainer.py`, `ml/inference/predictor.py`, `backend/app/services/forecast_service.py` | `GET /api/forecast/{station}?hours=72`, `GET /api/forecast/ncr`, `POST /api/forecast/generate` | 72H Forecast page (hourly chart, pollutant selector, AQI timeline, weather overlay, per-horizon bands) | `backend/tests/test_forecast_api.py`, `test_forecast_service.py`, `unit/test_forecast_pipeline.py` | IMPLEMENTED |
| PM2.5 | Direct PM2.5 engine with split-conformal prediction intervals; feature in all tree models | `ml/training/train_pm25.py`, `ml/inference/pm25_forecaster.py` | `GET /api/forecast/pm25{,/model-card,/explanation}` | Dashboard PM2.5 card, Forecast charts, PM2.5 selection | `unit/test_pm25_forecaster.py`, `test_pm25_forecast_api.py` | IMPLEMENTED |
| PM10 | Trained + evaluated across all horizons | `ml/training/trainer.py` | `GET /api/forecast/{station}` | Pollutant selector, table columns | `unit/test_models.py::test_forecast_complete_pollutant_set` | IMPLEMENTED |
| Ground-level O3 | Trained + evaluated (photochemical potential feature) | `ml/training/trainer.py`, `ml/features/coupling_engine.py` (ozone-photochemical) | `GET /api/forecast/{station}` | Pollutant selector | `unit/test_fire_coupling.py`, model-performance page | IMPLEMENTED |
| NOx / NO2 | NO2 trained + evaluated; NOx reported honestly when absent | `ml/training/trainer.py` | `GET /api/forecast/{station}` | Pollutant selector | Model-performance page | IMPLEMENTED |
| Temperature | Open-Meteo surface + 1000/925/850/700 hPa fields in feature pipeline | `ml/features/feature_engineering.py`, `backend/app/services/refresh_service.py` | `GET /api/weather/{station}` (typo-hostile), `GET /api/atmosphere/current`, forecast context | Atmosphere page, forecast-context table | `unit/test_atmospheric_state.py` | IMPLEMENTED |
| Wind speed | u/v decomposition from speed+direction; feature + coupling anchors | `ml/features/wind.py` (u/v), feature engineering | forecast context, `GET /api/coupling/{station}` | Transport page wind vector, forecast context | `unit/test_wind_vector.py` | IMPLEMENTED |
| Wind direction | Stored as degrees-from; decomposed to sin/cos; "FROM <compass>" | `ml/features/wind.py` | context + coupling payloads | Transport / Fire & Plume overlay | `unit/test_wind_vector.py` | IMPLEMENTED |
| Humidity | Stored, feature, forecast-context, coupling humidity feature | feature engineering, `coupling_engine.humidity_boost` | context, coupling features | 72h table Hum % column (added Phase 40) | `unit/test_coupling_engine.py` | IMPLEMENTED |
| Pressure | Surface + pressure levels stored; feature + context | `WeatherReading` columns | context | 72h table Pressure hPa column (added Phase 40) | `unit/test_vertical_atmosphere.py` | IMPLEMENTED |
| PBL height (PBLH) | Stored from Open-Meteo `boundary_layer_height`; classified LOW/MODERATE/HIGH; ventilation potential; trend + anomaly in atmosphere summary | `ml/features/atmospheric_profile.py`, `backend/app/services/atmosphere_service.py`, `coupling_service.py` | `GET /api/atmosphere/current`, context, coupling features/state | Atmosphere page PBL panel (value, trend, ventilation, confidence) | `unit/test_atmospheric_profile.py`, `test_coupling_state.py` | IMPLEMENTED |
| Atmospheric inversion detection + strength | Vertical lapse-rate gradient (dT/dp × 100 K/100 hPa) from stored pressure levels; PBL-height proxy fallback clearly labelled; categories NO/WEAK/MODERATE/STRONG; strength 0..1 ramp | `ml/features/atmospheric_profile.py`, `backend/app/api/inversion.py`, `ml/features/inversion.py` | `GET /api/inversion/{station}`, atmosphere summary | Inversion panel (category, source `lapse_rate|pbl_proxy`, strength bar, trapping) | `unit/test_vertical_atmosphere.py`, `test_inversion_api.py` | IMPLEMENTED |
| Stubble burning / regional fire influence | NASA FIRMS VIIRS ingestion + fire count, FRP-weighted impact, alignment, transport time (`nearest/(wind×3.6)`), transport risk bands, stubble-impact score; transport pathway overlay (≤500 km upwind, ±90°, top-6 FRP corridors) | `ml/features/fire_impact.py`, `backend/app/api/fire.py`, `ml/features/coupling_engine.py` | `GET /api/fire/hotspots`, `/fire-activity`, `/plume-risk`, `GET /api/dispersion/forecast` | Transport (/Fire & Plume) page; map corridors + 500 km ring + influence zone | `unit/test_fire_impact.py`, `test_fire_api.py`, `unit/test_coupling_engine.py` | IMPLEMENTED |
| Pollution dispersion | Numerical finite-difference advection–diffusion–deposition solver (~2.2 km NCR grid, CFL-safe, non-negative) + honest engine compositing | `ml/features/dispersion_solver.py`, `backend/app/services/dispersion_service.py` | `GET /api/dispersion/forecast` | Atmosphere / Transport pages; spatial outlook | `unit/test_dispersion_solver.py` | IMPLEMENTED |
| Two-way weather ↔ pollution coupling | **met→chem:** features feed all forecasters; **chem→met:** AOD proxy → radiation transmittance → PBL suppression → stability-coupling-index → feedback multiplier inside the online loop; labelled *data-driven surrogate* | `ml/features/coupling.py`, `ml/features/coupled_loop.py`, `ml/features/coupling_engine.py`, `backend/app/services/coupling_service.py` | `GET /api/coupling/{station}`, `/coupling/features/{station}`, `GET /api/coupling/state`, `POST /api/forecast/coupled` (returns coupled + uncoupled series) | Coupling panel, forecast-context, coupling-state | `test_coupled_forecast_api.py`, `unit/test_coupling_engine.py`, `test_coupling_state.py` | IMPLEMENTED (documented surrogate) |
| Real-time dashboard | Operational control-room layout (current AQI + 72h max AQI, PM2.5, O3, PBLH, inversion, dispersion, fire influence; 8 visualisations) | `frontend/src/pages/Dashboard.tsx`, `Overview.tsx`, components (`KpiCard`, `AtmosphericState`, `CouplingPanel`, `InversionPanel`, `GrapPanel`) | aggregation endpoints `/api/summary`, `/api/current/*`, `/api/system` | Dashboard | `test_root.py`, `test_auth_api.py` | IMPLEMENTED |
| High-resolution NCR spatial visualisation | 2×2 km IDW gridded surface from per-station forecasts + numerical field; Delhi/Gurugram/Noida/Ghaziabad/Faridabad + all NCR stations; time slider NOW/+6..+72h; AQI/PM/weather/wind/PBLH/inversion/fire/fire-influence/dispersion layers | `backend/app/services/grid_service.py`, `ml/features/interpolation.py`, `frontend/src/pages/SpatialForecastPage.tsx`, `NCRMap.tsx`, `StationMap.tsx`, `ml/features/dispersion_solver.py` | `GET /api/grid/forecast`, `GET /api/dispersion/forecast`, `/api/coupling/state` | Spatial AQ Outlook page (time slider, layer toggles, always-visible legend), NCR Map (rings, corridors) | `test_grid_api.py`, `unit/test_interpolation.py`, `unit/test_dispersion_solver.py` | IMPLEMENTED |
| Explainable forecasts | Real SHAP (`TreeExplainer`) with honest fallback; driver + evidence narrative built from actual feature values | `backend/app/services/explanation_service.py`, forecast-reason panel | `GET /api/explanation/{station}`, `/api/forecast/{station}/context`, `/api/forecast/pm25/explanation` | Forecast Explainability page, ForecastReasonPanel, AI Explanation link | `test_explainability_api.py`, `unit/test_explanation_service.py` | IMPLEMENTED |
| Forecast uncertainty / confidence | Split-conformal prediction intervals (calibrated on validation, measured coverage on test) for the direct PM2.5 engine; honest "Uncertainty estimate unavailable" elsewhere | `ml/inference/pm25_forecaster.py`, `backend/app/api/model_performance.py` | `GET /api/forecast/pm25` (± bands), model-performance coverage table | Dashboard / pm25 forecast charts (band), uncertainty disclosure on 72H page | `unit/test_pm25_forecaster.py`, `test_pm25_forecast_api.py` | IMPLEMENTED |
| Model performance evaluation | MAE/RMSE/R² by pollutant × horizon × model (persistence/RF/XGB/GRU) on chronological held-out test; actual-vs-predicted + residual + error distribution + conformal coverage | `ml/training/evaluator.py`, `backend/app/api/model_performance.py` | `GET /api/model/performance?target=pm25|pm10|o3|no2` | Model Performance page | `test_model_performance_api.py`, `ml/training/tests/` | IMPLEMENTED |

## L2 — Evidence that every PS paragraph is discharged

| PS paragraph | Where addressed | Key vectors |
|--------------|-----------------|-------------|
| "WRF-Chem or similar coupled frameworks" | Honest adapter (`validate_configuration`/`run_forecast`/`get_output`, `CtmUnavailable` when gated) + physics-informed numerical surrogate | `ml/ctm/wrfchem_adapter.py`, `ml/ctm/*`, `docs/wrfchem_adapter.md`, `docs/methodology.md` §9 |
| Meteorology–chemistry interlink | Coupling engine (9 features) + coupled/uncoupled forecast loop | `ml/features/coupling_engine.py`, `ml/features/coupled_loop.py`, `/api/forecast/coupled` |
| Inversion impact on trapping | Lapse-rate inversion + `inversion_trapping_potential` feature + stability coupling | `ml/features/atmospheric_profile.py`, `coupling_engine.py`, `/api/inversion/{station}` |
| Stubble-plume dispersion under prevailing weather | FRP fire sources + wind/PBL/rain forced solver + transport-time estimation | `ml/features/dispersion_solver.py`, `ml/features/fire_impact.py`, `/api/dispersion/forecast` |
| NCR-wide coverage (not just a single station) | IDW grid + numerical domain; city markers incl. NCP cities | `ml/features/interpolation.py`, `grid_service.py`, `/api/grid/forecast`, `/spatial` |
| Actionable alerting / public-health interpretation | Alert rules (AQI/trend/wind/PBL/inversion/humidity/fire) + severity ranks + GRAP staging | `backend/app/services/alert_service.py`, `backend/app/api/alerts.py`, `grap/` endpoints, `Alerts.tsx`, `GrapPanel.tsx` |
| Real monitoring + satellite fire data | CPCB (data.gov.in CKAN), Open-Meteo, NASA FIRMS in DB + download scripts + provenance page | `scripts/download_*.py`, `frontend/src/pages/DataMethodology.tsx` (`/data-methodology`), `/api/data-quality` |

## L3 — UI surfaces that prove the chain end-to-end

1. **Login/Demo** → `/login` (JWT demo credentials via `/api/auth/demo`).
2. **Dashboard** → `/` current AQI + 72h max AQI + inversion/dispersion/fire tier, 72h AQI chart, alerts strip.
3. **72H Forecast** → `/forecast` hourly multi-pollutant + AQI timeline + weather overlay + risk bands + uncertainty note.
4. **Atmosphere** → `/atmosphere` PBL (height/trend/ventilation/confidence), inversion (category/source/strength/trapping), wind, humidity/pressure, dispersion/accumulation panels.
5. **Transport** → `/fire-plume`, `/stubble` fire points + influence zone + transport corridor + wind arrows over Delhi NCR.
6. **NCR Map** → `/map` station clustering, AQI, 500 km influence ring, modelling-domain polygon.
7. **Forecast Explainability** → `/explanation` SHAP + driver/evidence narrative.
8. **Alerts** → `/alerts` severity badges + reasons + GRAP.
9. **Model Performance** → `/model-performance` MAE/RMSE/R² by horizon + splits + conformal coverage.
10. **Spatial AQ Outlook** → `/spatial` gridded NCR forecast with time slider.
11. **Scenario** → `/scenario` what-if engine (explicitly labelled, read-only over DB).
12. **Data & Methodology** → `/data-methodology` provenance (observed / reanalysis / forecast / derived / ML-predicted / scenario-simulated).

## Verification run used to produce this matrix

```bash
python -m pytest backend/tests -q                    # 748 passed
python -m ruff check backend/app backend/tests ml     # clean
cd frontend && npx tsc --noEmit && npx vite build     # clean
```