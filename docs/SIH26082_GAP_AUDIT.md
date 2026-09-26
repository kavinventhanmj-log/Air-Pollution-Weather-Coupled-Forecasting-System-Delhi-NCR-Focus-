# SIH26082 — GAP AUDIT (pre-change snapshot)

**AeroCast-NCR · Air Pollution–Weather Coupled Forecasting System (Delhi NCR Focus)**
Phase-1 gap audit performed against the committed baseline (v1.11.0) before the
Phase 19/20 coupling-status + flow additions described at the end.

> Complementary docs: [`SIH26082_IMPLEMENTATION_AUDIT.md`](SIH26082_IMPLEMENTATION_AUDIT.md)
> (24-row capability status), [`SIH26082_TRACEABILITY.md`](SIH26082_TRACEABILITY.md) (PS→code→
> UI→test matrix), [`SIH26082_FINAL_AUDIT.md`](SIH26082_FINAL_AUDIT.md) (A–P classification),
> [`SCIENTIFIC_METHODOLOGY.md`](SCIENTIFIC_METHODOLOGY.md) (formulas), and the earlier
> [`SIH_GAP_AUDIT.md`](SIH_GAP_AUDIT.md) (historic gap list).

## 1. Already implemented (real, connected)

- **72 h multi-pollutant forecasting** — PM2.5/PM10/O3/NO2/SO2/CO × horizons {1,6,12,24,48,72},
  persistence/RF/XGBoost, chronological splits (`ml/training/trainer.py`, `forecast_service.py`).
- **Direct PM2.5 engine** with split-conformal prediction intervals + model card + explanation
  (`ml/inference/pm25_forecaster.py`, `/api/forecast/pm25*`).
- **Two-way coupling surrogate** — met→chem features in all forecasters; chem→met feedback
  (AOD proxy → transmittance → PBL suppression → stability-coupling index → multiplier) in an
  online loop; `POST /api/forecast/coupled` returns coupled + uncoupled series
  (`ml/features/coupling.py`, `coupling_engine.py`, `coupled_loop.py`).
- **Coupling-state persistence** (`coupling_states`, `/api/coupling/state[+/{station}]`, labels
  `coupling_state`/`coupling_domains`/`data_quality`) — Phase 30.
- **Nine coupling features** from stored observations only, each with a `basis` string
  (`ml/features/coupling_engine.py`).
- **Vertical lapse-rate inversion** (dT/dp × 100 K/100 hPa over 1000/925/850/700 hPa) +
  documented PBL-proxy fallback + categories NO/WEAK/MODERATE/STRONG (`atmospheric_profile.py`).
- **PBLH first-class** — Open-Meteo field, LOW/MODERATE/HIGH, ventilation potential, trend,
  anomaly, confidence (`atmosphere_service.py`, `coupling_service.py`).
- **Wind u/v vector** processing + dispersion potential + transport direction (`ml/features/wind.py`).
- **Fire/stubble engine** — FIRMS VIIRS, ≤500 km, upwind ±90°, transport time, transport-risk
  bands, stubble-impact score, estimated transport pathway + influence ring + modelling-domain
  polygon on maps (`fire_impact.py`, geo.ts, StationMap).
- **Numerical dispersion solver** — finite-difference advection–diffusion–deposition on the
  ~2.2 km grid, CFL-safe, non-negativity preserving (`dispersion_solver.py`).
- **Indian AQI** from predicted concentrations (never AQI-only target) (`aqi_calculator.py`).
- **Alerts** with deterministic rules + 30-test unit suite (`alert_service.py`).
- **SHAP explainability** with honest fallback (`explanation_service.py`).
- **Auth (JWT + demo), CORS via env, Pydantic validation** — no keys in source.
- **Dashboard, maps, 72h page, atmosphere, transport, alerts, performance, spatial outlook,
  scenario, data & methodology** pages (all with Loading/Error/Empty states).
- **748 tests passing** — AQI, inversion, PBL, wind vectors, dispersion, fire, coupling,
  forecast pipeline, time-series split, leakage checks, API validation.

## 2. Partially implemented (honest boundaries)

| Area | Partial part | Boundary documented |
|------|--------------|---------------------|
| Vertical inversion | Only 4 pressure levels; inversion base/top/thickness/duration/persistence not computed | `atmospheric_profile.py`, FINAL_AUDIT §F |
| Real-engine dispersion | Surrogate PDE runs; WRF-Chem/HYSPLIT run is gated (adapter contract only) | `wrfchem_adapter*, hysplit.md, methodology §9` |
| Two-way coupling | Functional surrogate, not physical radiative transfer/chemistry | All UI + docs label it a surrogate |
| Data ingestion | ERA5 single-level offline; IMD/ERA5-CD live credential-gated | `era5_surface.py`, `imd_weather.py` |
| Model comparison | Persistence/RF/XGB/GRU by horizon; no measured with- vs without-coupling-features ablation | documented as limitation |
| Long-horizon skill | NO₂/SO₂ R² ↔ 0.21/0.15 @72 h shown as-is | Model Performance page |

## 3. Missing (verified absent at baseline)

- Explicit **directional coupling status** (Meteorology→Pollution / Pollution→Meteorology
  ACTIVE/LIMITED/UNAVAILABLE) on the dashboard — **added in this phase**.
- A **visual scientific flow** (weather → dispersion → pollutants → feedback → response) with
  per-stage variables, current values and source/method — **added in this phase**.
- Inversion thickness/base-top/persistence (requires multi-day vertical archive).
- Measured coupling-ablation metrics on test sets (serves honesty; not fabricated).
- Rate limiting on public read endpoints.

## 4. Only UI-level (what is visually present but not real output)

- No fabricated sensor/forecast/fire values anywhere. Any "Only UI-level" claim is consciously
  avoided — displayed numbers trace to DB rows, NWP fields, or documented surrogates; missing
  inputs render "Unavailable"/"Data unavailable".

## 5. Connected to the ML pipeline (feature-level)

- Lags (pollution, weather), PBLH, inversion, dispersion/ventilation, fire-transport, coupling
  features, temporal encodings, weather (T/humidity/pressure/wind/precip) — all in
  `feature_engineering.py` and consumed by trainer/predictor. PBLH appears in training,
  inference, dispersion, fire transport, coupling and explanation engines.

## 6. Connected to real data

- CPCB (data.gov.in CKAN) live + archived; Open-Meteo hourly (incl. pressure levels + PBLH);
  NASA FIRMS VIIRS (FRP/confidence); ERA5 reanalysis offline; IMD opt-in. All download scripts
  in `scripts/`. Data-provenance page documents observed/reanalysis/forecast/derived/ML/scenario.

## 7. Scientifically weak (without changes)

- Inversion diagnostics lack vertical thickness/base/top and persistence.
- No live multi-level vertical reanalysis (ERA5 CDS) to replace the 4-pressure-level archive.
- NO₂/SO₂ long-horizon skill and ML-ablation-vs-coupled claims are honest but unproven — kept
  truthful rather than inflated.

## 8. Currently simulated (labelled as such)

- Meteorology–pollution feedback and fire transport are documented **surrogates/estimates**; the
  dispersion solver is a real numerical model but **not** WRF-Chem; scenario page is explicitly
  hypothetical. No simulation is ever presented as an observation or a physical CTM result.

## 9. Improvable without breaking anything

- Directional coupling status + flow visualization (the Phase 19/20 deliverable below).
- Continuing to add provenance columns on any page showing derived indicators.
- More unit tests around directional-status derivation once present.

---

## This-phase change set (Phase 19/20)

- `frontend/src/types/index.ts` — `CouplingFeaturesResponse` carries `coupling_state`,
  `coupling_domains`, `data_quality`.
- `frontend/src/components/CouplingStatusFlow.tsx` (new) — directional ACTIVE/LIMITED/UNAVAILABLE
  status derived from the live coupling-features payload + aerosol-feedback diagnostics, plus the
  clickable 5-stage flow (weather → atmospheric state → pollutants → aerosol feedback →
  atmospheric response) with per-stage variables, current values and source/method.
- `frontend/src/components/CouplingPanel.tsx` — renders `CouplingStatusFlow`; optional `features`
  prop.
- `frontend/src/pages/Atmosphere.tsx` — passes `features` to `CouplingPanel`.

Validation after change: `tsc --noEmit`, `vite build`, `ruff check backend/app backend/tests ml`,
`pytest backend/tests -q` (664 at 1.16.0; 635 when this phase landed). No backend/DB/API-shape changes — this phase is frontend/docs only.
