# SIH26082 — AeroCast-NCR Implementation Audit (Phase 41)

> Problem Statement: **Air Pollution–Weather Coupled Forecasting System (Delhi NCR Focus)** —
> Ministry of Earth Sciences / NCMRWF (SIH 2026).
>
> Phase-41 section audit of every SIH26082 capability against the codebase. Statuses are
> limited to **`IMPLEMENTED`** / **`PARTIALLY IMPLEMENTED`** / **`NOT IMPLEMENTED`** /
> **`NOT AVAILABLE`**. Formal formulas, constants, units and assumptions for every indicator
> are in [`docs/SCIENTIFIC_METHODOLOGY.md`](SCIENTIFIC_METHODOLOGY.md); requirement-level
> narrative and evidence are in [`docs/SIH_FINAL_COMPLIANCE.md`](SIH_FINAL_COMPLIANCE.md).
>
> **Honesty contract (unchanged):** no value is fabricated — missing inputs ⇒ `None` ⇒
> "Data unavailable"; no invented accuracy (R² is reported only from chronological held-out
> splits); proxies/potentials/surrogates are explicitly labelled; real-engine adapters
> (HYSPLIT / WRF-Chem / ERA5 / IMD) are operator-gated and always report *why* when gated.

---

## 1. Requirement-by-requirement status (24 rows)

| # | Capability | Status | Implementation / evidence | Remaining limitation |
|---|------------|--------|---------------------------|----------------------|
| 1 | **Two-way air pollution ↔ weather coupling** (chemistry→meteorology: AOD → radiation transmittance → PBL suppression → stability/coupling index; meteorology→chemistry: features feed forecasters) | IMPLEMENTED | `ml/features/coupling.py`, `ml/features/coupled_loop.py`, `backend/app/api/coupling.py` | Analytic surrogate, not a compiled coupled CTM |
| 1a | **Coupling feature group measured by ablation** (paired with/without the 6 coupling features on one identical chronological split) | IMPLEMENTED | `ml/evaluation/ablation.py` (`run_coupling_ablation`), `scripts/run_coupling_ablation.py`, `reports/coupling_ablation_pm25.json`, `backend/tests/unit/test_coupling_ablation.py` (32) | t+24 h: MAE 41.364 vs 42.510 (−2.70%, material). t+6 h: −0.12%, below the 0.5% materiality threshold → reported as no material change. Single seed / RandomForest only |
| 2 | **Nine named coupling features** (dispersion, accumulation, inversion-trapping, stagnation, aerosol-accumulation, fire-transport, regional-transport, ozone-photochemical, meteorology-pollution interaction) | IMPLEMENTED | `ml/features/coupling_engine.py`; computed from stored obs, 0..1 with `basis` string | Features are potentials/tendencies, not measurements |
| 3 | **Coupling features computed from stored observations only** — never overwriting a measured value, never inventing missing inputs | IMPLEMENTED | `backend/app/services/coupling_service.py` (`get_coupling_inputs` provenance), `backend/tests/unit/test_coupling_engine.py` (15) | `None` reported as "Data unavailable" when inputs missing |
| 4 | **Meteorology → chemistry forward path** (wind/PBL/inversion/humidity/fire features consumed by the ML forecasters at training + inference) | IMPLEMENTED | `ml/features/feature_engineering.py`, `backend/app/services/forecast_service.py` (`build_features_from_db`) | — |
| 5 | **Chemistry → meteorology backward path** (aerosol feedback: `aod_est`, `radiation_transmittance`, `pbl_suppression_factor`, `corrected_pbl_height`, `stability_coupling_index`, `feedback_multiplier`) | IMPLEMENTED | `ml/features/coupling.py`, `/api/coupling/{station}` | Physics-informed analytic closure, not a radiative-transfer scheme |
| 6 | **Online time-stepped coupled 72h loop** with coupled-vs-uncoupled skill comparison and hour-by-hour `feedback_path` | IMPLEMENTED | `ml/features/coupled_loop.py`, `GET /api/forecast/coupled` | Loop inherits per-model forecast skill |
| 7 | **Coupling-state persistence (Phase 30)** — per-station latest snapshot with `coupling_state` / `coupling_domains` / `data_quality` labels and retrieval endpoints | IMPLEMENTED | `backend/app/models/db_models.py` (`CouplingState`), Alembic `5c1b7d9a2f6e`, `backend/app/services/coupling_service.py`, `GET /api/coupling/state`, `GET /api/coupling/state/{station}`, `backend/tests/test_coupling_state.py` (7) | Snapshot updated on demand (features fetch), not a scheduled cron |
| 8 | **72-hour multi-pollutant forecasting** for Delhi NCR (PM2.5, PM10, O3, NO2, SO2, CO × horizons {1,6,12,24,48,72}, persistence / RF / XGBoost) | IMPLEMENTED | `ml/training/trainer.py`, `ml/inference/predictor.py`, `GET /api/forecast/{station}?hours=72`, `GET /api/forecast/ncr` | Long-horizon NO₂/SO₂ skill modest (0.21 / 0.15 R² at 72 h) |
| 9 | **Direct PM2.5 forecast engine** with distribution-free split-conformal prediction intervals + model card + explanation | IMPLEMENTED | `ml/training/train_pm25.py`, `ml/inference/pm25_forecaster.py`, `GET /api/forecast/pm25{,/model-card,/explanation}` | Interval coverage degrades at longer horizons |
| 10 | **Vertical pressure-level atmospheric data** (1000/925/850/700 hPa temperature + geopotential) per station | IMPLEMENTED | Open-Meteo pressure-level ingestion in `backend/app/services/refresh_service.py`, `WeatherReading` profile columns, `docs/methodology.md` | Very recent archive levels may be NULL → live-forecast supplement; ERA5 (CDS) documented as production upgrade |
| 11 | **Lapse-rate inversion detection** (`dT/dp × 100` K/100 hPa, `T↑` with height ⇒ inversion; source `lapse_rate \| pbl_proxy`) | IMPLEMENTED | `ml/features/atmospheric_profile.py`, `backend/app/api/inversion.py`, `backend/tests/unit/test_vertical_atmosphere.py` | Grades (1.5/0.6/0.0 K per 100 hPa) are heuristic, stated as such |
| 11a | **Inversion layer geometry** (base/top height in m, thickness in m and hPa) | IMPLEMENTED | `ml/features/atmospheric_profile.py` (`pressure_to_height_m`, `layer_heights_m`, `classify_gradient`), `backend/app/api/inversion.py` (`_geopotential_by_level`), `backend/tests/unit/test_inversion_geometry.py` | Heights use real geopotential when the feed supplies it, else the hypsometric equation anchored at 1000 hPa; `null` (never fabricated) under the PBL proxy |
| 11b | **Inversion episode duration & persistence** (current unbroken run in hours, share of trailing 24 h inverted, onset) | IMPLEMENTED | `ml/features/inversion.py` (`summarise_inversion_episode`), `backend/app/api/inversion.py` (`_episode_stats`, `EPISODE_HISTORY_HOURS=48`), `backend/tests/unit/test_inversion_geometry.py`, `backend/tests/unit/test_inversion_api_helpers.py` | Bounded by the stored archive: `inversion_measured_window_h`, `inversion_episode_samples` and `inversion_history_sufficient` disclose what was actually read, and a >3 h sampling gap breaks the run rather than being counted as persistence |
| 12 | **PBL classification / dispersion condition** (`low_pbl_flag`, `pbl_category`, `dispersion_condition`) | IMPLEMENTED | `ml/features/atmospheric_profile.py` (`classify_pbl`, `combine_inversion`) | Thresholds (150/300/500 m) heuristic |
| 13 | **Fire (stubble) ingestion + full transport feature set** (fire_count, FRP impact, nearest distance, `wind_alignment_pct`, `transport_time_hours`, `transport_risk` + level, `stubble_impact_score`, 500 km radius, ±90° upwind) | IMPLEMENTED | NASA FIRMS ingestion, `ml/features/fire_impact.py`, `backend/app/api/fire.py`, `docs/SCIENTIFIC_METHODOLOGY.md` §4 | Transport time is an advective estimate (no boundary-layer diffusion) |
| 14 | **Plume-transport pathway layer on the Leaflet map** (upwind ≤500 km fires ranked by FRP, top-6 dashed corridors, wind vector, "FROM <compass>") | IMPLEMENTED | `frontend/src/lib/geo.ts`, `frontend/src/components/StationMap.tsx`, `NCRMap.tsx`, `StubblePlumePage.tsx` | Straight-line advective corridor, not a dispersion result |
| 15 | **500 km influence ring + NCR modelling-domain boundary on maps** (ring == `fire_impact.DEFAULT_MAX_DISTANCE_KM`; polygon == 28.2–28.9°N / 76.6–77.5°E numerical grid) | IMPLEMENTED | `frontend/src/lib/geo.ts` (`INFLUENCE_RADIUS_KM`, `NCR_MODELING_BOUNDARY`), `StationMap.tsx` (`Circle`, `Polygon`) | Polygon is the modelling domain, not an administrative boundary |
| 16 | **WRF-Chem / coupled-CTM integration** (HYSPLIT `hycs_std` real binary + genuine CDMP parsing; WRF-Chem `wrfout_d01_*.nc` consumption; spec interface `validate_configuration` / `run_forecast` / `get_output`; `CtmUnavailable` when gated; 50/50 disclosed composite) | PARTIALLY IMPLEMENTED | `ml/ctm/*`, `backend/app/services/dispersion_service.py`, `ml/ctm/wrfchem_adapter.py`, `backend/tests/unit/test_ctm_engines.py`, `docs/hysplit.md`, `wrfchem_adapter.md` | Live engine run needs an operator-provided HYSPLIT build / real WRF-Chem output (external HPC by design); surrogate PDE is the always-available fallback |
| 17 | **Numerical dispersion surrogate** (vectorised finite-difference advection–diffusion–deposition PDE on the ~2.2 km NCR grid, CFL-safe, non-negativity preserving) | IMPLEMENTED | `ml/features/dispersion_solver.py`, `GET /api/dispersion/forecast`, `backend/tests/unit/test_dispersion_solver.py` | Parametric chemistry, not a gas/aerosol scheme |
| 18 | **GRU / LSTM deep-learning model** (custom NumPy GRU trained + evaluated across all pollutants/horizons; honest underperformance vs XGBoost documented) | IMPLEMENTED | `ml/training/train_gru.py`, `ml/models/gru_model.py`, `models/pm25/evaluation.json` | GRU is a documented ensemble candidate, not the serving model |
| 19 | **Data ingestion (CPCB + Open-Meteo + ERA5 + IMD + FIRMS)** pairwise with honesty gates | PARTIALLY IMPLEMENTED | `scripts/download_*.py`, `backend/app/services/refresh_service.py`, `ml/features/era5_surface.py`, `backend/app/services/imd_weather.py`, `backend/tests/unit/test_era5_surface.py`, `test_imd_weather.py` | ERA5 is single-level reanalysis (vertical profiles documented as the production upgrade); ERA5 and IMD are credential-gated (honest `reasons`, never fabricated); IMD raw radar not directly consumed |
| 20 | **Indian (CPCB breakpoint) AQI** — sub-index, AQI, category, dominant pollutant | IMPLEMENTED | `backend/app/services/aqi_calculator.py`, `backend/tests/test_aqi.py` | — |
| 21 | **Explainability** — real `shap.TreeExplainer` when a tree model loads + honest fallback | IMPLEMENTED | `backend/app/services/explanation_service.py`, `frontend/src/pages/AIExplanation.tsx` | SHAP on tree models only (not persistence) |
| 22 | **Alerts** — deterministic rules (copied thresholds: AQI 201/301/401; trend; PM2.5 dominance; wind <2/>15 m/s; PBL <150/<300 m; humidity >80 %; no-rain washout; fires >20 / >50 & <300 km), severity rank, dedicated unit suite | IMPLEMENTED | `backend/app/services/alert_service.py`, `backend/app/api/alerts.py`, `backend/tests/unit/test_alert_service.py` (30) | Thresholds are configurable constants, not regulatory mandates |
| 23 | **Chronological validation / backtesting** + persisted cross-model metrics + performance dashboard | IMPLEMENTED | `ml/training/evaluator.py`, `backend/app/api/model_performance.py`, `model_metrics` tables, `frontend/src/pages/ModelPerformancePage.tsx` | Metrics are from the most recent chronological evaluation run only |
| 24 | **Command dashboard & map layers** (FIRMS hotspots, wind vectors, transport pathways, influence ring, atmosphere/inversion panel, GRAP panel, forecast context, alert severity labels, per-horizon humidity/pressure, uncertainty disclosure, keyboard-accessible maps) | IMPLEMENTED | `frontend/src/pages/*`, `frontend/src/components/*` (StationMap, AlertList, InversionPanel, GrapPanel, Forecast72h, AboutModal), `frontend/src/lib/{aqi,geo,theme}.ts` | Live metrics masked to latest chronological evaluation; some grids render only on data availability |

**Summary:** 23 `IMPLEMENTED`, 1 `PARTIALLY IMPLEMENTED` (row 16 — real-engine CTM run), 0 `NOT
IMPLEMENTED`, 0 `NOT AVAILABLE`.

---

## 2. Verification runs (Phase 41 session)

| Check | Command | Result |
|-------|---------|--------|
| Full backend suite | `python -m pytest backend/tests -q` | **748 passed** |
| CI lint scope | `python -m ruff check backend/app backend/tests` | All checks passed |
| ML module lint (town fix) | `python -m ruff check ml` | All checks passed (pre-existing B023/F841 fixed) |
| Coupling-state + features API | `python -m pytest backend/tests/test_coupling_state.py backend/tests/test_coupling_features_api.py -q` | 14 passed |
| Alert-engine suite | `python -m pytest backend/tests/unit/test_alert_service.py -q` | 30 passed |
| Alembic chain (Postgres dialect) | `python -m alembic upgrade head --sql` | full chain renders `coupling_states` DDL; offline SQL valid |
| Frontend typecheck | `cd frontend && npx tsc --noEmit` | clean |
| Frontend build | `cd frontend && npx vite build` | success (chunk-size warning only) |

---

## 3. Files changed / created (Phase 30 + Phase 40/41)

**Created**
- `alembic/versions/5c1b7d9a2f6e_coupling_state.py` — `coupling_states` migration.
- `backend/tests/test_coupling_state.py` (7), `backend/tests/unit/test_alert_service.py` (30).
- `docs/SCIENTIFIC_METHODOLOGY.md` — formal formulas/constants/units/assumptions.

**Modified**
- `backend/app/models/db_models.py` — `CouplingState` model (`uq_coupling_state_station`).
- `backend/app/services/coupling_service.py` — write-through persistence + state/domain/quality labels + `_parse_dt`.
- `backend/app/schemas/schemas.py` — `coupling_state` / `coupling_domains` / `data_quality` on `CouplingFeaturesResponse`; added `CouplingStateSnapshot`, `CouplingStateListResponse`.
- `backend/app/api/coupling.py` — `GET /api/coupling/state[+/{station}]` (registered before the dynamic route).
- `frontend/src/lib/geo.ts` — `DELHI_NCR_CENTROID`, `INFLUENCE_RADIUS_KM`, `NCR_MODELING_BOUNDARY`.
- `frontend/src/components/StationMap.tsx` — influence ring + modelling-boundary polygon + map a11y (`keyboard={false}` removed).
- `frontend/src/pages/Forecast72h.tsx` — humidity/pressure columns, uncertainty disclosure, chart/table a11y.
- `frontend/src/components/AlertList.tsx` — severity text badges + missing ADVISORY level styling.
- `README.md` — SIH 2026, coupling-state API rows, scientific limitations, test count 748.
- `CHANGELOG.md` — 1.10.0 entry. `ml/preprocessing/training_dataset.py`, `ml/training/evaluate_pm25.py` — lint fixes.
- `docs/SIH26082_IMPLEMENTATION_AUDIT.md` — this file.

**Preserved (not modified unnecessarily)** — `dispersion_solver.py`, `ml/features/coupling.py`,
`ml/preprocessing/*`, trained `.joblib` models, `aerocast_ncr.db`, existing endpoints/models.

---

## 4. Honest-disclosure statements (unchanged)

1. **No synthetic data.** Every coupling feature, atmosphere indicator and alert is computed from
   stored CPCB / Open-Meteo / FIRMS observations; a missing input ⇒ `None` ⇒ "Data unavailable".
2. **No fabricated WRF-Chem runs.** Adapters consume genuine engine output and raise
   `CtmUnavailable` with an exact `reason` otherwise.
3. **No invented accuracy.** R² / MAE are live `/api/model/performance` values from the most
   recent chronological hold-out; never reinterpreted as "accuracy".
4. **Proxies labelled.** PBL/inversion grades, advective transport times, coupling potentials and
   the meteorology–pollution surrogate carry explicit "estimate / proxy / potential" wording in
   code, payloads and UI (see also `docs/SCIENTIFIC_METHODOLOGY.md` §0).