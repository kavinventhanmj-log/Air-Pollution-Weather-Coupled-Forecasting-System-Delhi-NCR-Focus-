# AeroCast-NCR

### AI-Powered 72-Hour Air Quality & Pollution-Plume Forecasting System for Delhi NCR

**Problem Statement — [SIH 2026 · #26082](docs/PS_SUBMISSION.md)** · Ministry of Earth Sciences (MoES) · National Centre for Medium Range Weather Forecasting (NCMRWF)

---

[![CI](https://github.com/methila-2056/Air-Pollution-Weather-Coupled-Forecasting-System-Delhi-NCR-Focus-/actions/workflows/ci.yml/badge.svg)](https://github.com/methila-2056/Air-Pollution-Weather-Coupled-Forecasting-System-Delhi-NCR-Focus-/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-941%20passed-success)](backend/tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-compose%20ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Frontend](https://img.shields.io/badge/frontend-Vercel%20%E2%9C%93-success?logo=vercel)](https://air-pollution-weather-coupled-forec-eight.vercel.app)
[![Backend](https://img.shields.io/badge/backend-status%3A%20unverified-lightgrey?logo=render)](docs/deployment.md)

AeroCast-NCR fuses **official CPCB real-time monitoring (data.gov.in)**,
**Open-Meteo weather**, **NASA FIRMS active fires**, **ERA5 meteorology**,
**NOAA HYSPLIT** and **WRF-Chem output** into a machine-learning
**72-hour pollutant forecast** for every station in the Delhi NCR belt — with an
explicit **two-way weather–chemistry coupling** module, a **high-resolution
gridded spatial surface**, and a **numerical advection–diffusion dispersion
core** that simulates how stubble-burning plumes disperse under prevailing
weather. Every result is explainable (SHAP), bounded (split-conformal
prediction intervals) and honest (gated real engines, openly-reported skill).

> ### Deployment status (verified 2026-09-28)
>
> The **frontend** is live and serving: <https://air-pollution-weather-coupled-forec-eight.vercel.app>
>
> The **backend** at `air-pollution-weather-coupled.onrender.com` was **not
> responding** at the time of writing (both `/health` and `/api/health` timed
> out; the Vercel `/api/*` proxy times out as a consequence). Treat the hosted
> API as unverified — run the stack locally with the Quick Start below, or follow
> [`docs/deployment.md`](docs/deployment.md) to bring the backend up and verify
> it yourself before relying on any hosted number.
>
> The database credentials previously committed to local environment files have
> been treated as compromised and should be **rotated**; this repository ships
> only `.env.example` templates, and `.env` / `*.db` are git-ignored.

---

## Table of Contents

- [Why AeroCast-NCR](#why-aerocast-ncr)
- [Key Capabilities](#key-capabilities)
- [Refusal Over Fabrication](#refusal-over-fabrication)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Data Sources & Honesty Gates](#data-sources--honesty-gates)
- [Model Performance](#model-performance)
- [Data & Machine-Learning Pipeline](#data--machine-learning-pipeline)
- [Database & Migrations](#database--migrations)
- [Repository Layout](#repository-layout)
- [Quick Start — Docker](#quick-start--docker)
- [Quick Start — Bare Metal](#quick-start--bare-metal)
- [API Summary](#api-summary)
- [Frontend Pages](#frontend-pages)
- [Testing & Quality Gates](#testing--quality-gates)
- [Deployment & CI/CD](#deployment--cicd)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Honest Scope](#honest-scope)
- [Known Issues](#known-issues)
- [Contributing & Security](#contributing--security)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Why AeroCast-NCR

Delhi NCR experiences among the worst air-quality episodes on Earth. The
October–December "stubble season" routinely pushes the AQI past **450
(Severe+)** — a level driven almost as much by *weather* as by emissions:

- **Temperature inversions** suppress the planetary boundary layer (PBL) and
  trap pollutants within a few hundred metres of the ground.
- **North-westerly winds** advect hundreds of thousands of tonnes of
  agricultural-residue smoke from Punjab & Haryana into the region.
- **Calm, humid, foggy nights** with low ventilation convert an ordinary
  emissions day into an air-quality emergency.

Forecasting these episodes requires more than statistical extrapolation — it
requires coupling **meteorology with chemistry** in both directions:

> Pollution absorbs sunlight → the boundary layer collapses → winds calm →
> ventilation drops → pollution accumulates further.

AeroCast-NCR is a production-grade, containerised forecasting system that
operationalises this physics:

1. Ingests **real, official observations** (CPCB via data.gov.in, Open-Meteo,
   NASA FIRMS, monthly ERA5) through an idempotent, deduplicated
   time-zone-aware refresh pipeline.
2. Trains and serves **per-pollutant, per-horizon ML models** (XGBoost,
   Random Forest, Persistence, plus a custom GRU trained & evaluated as a
   candidate ensemble member) for all six criteria pollutants — **108 models**.
3. Provides a **direct PM2.5 forecast engine** for every hour 1→72 with
   split-conformal prediction intervals **and** SHAP explanations.
4. **Couples weather and chemistry** in a closed feedback loop — aerosol
   optical depth attenuates solar radiation, suppressing the boundary layer
   and reinforcing stability.
5. **Numerically simulates plume dispersion** on a ~2.2 km NCR grid
   (advection + diffusion + deposition + emission) driven by live stubble-fire
   detections and prevailing meteorology — usable as an honest surrogate for,
   or composited with, real HYSPLIT / WRF-Chem engines.
6. Detects **pollution events**, runs read-only **what-if scenarios**, aligns
   alerts to the CAQM **Graded Response Action Plan (GRAP)** stages I–IV, and
   surfaces all of it through a **real-time command dashboard** and a suite of
   REST APIs.

## Key Capabilities

| Capability | How it works | Verify |
|------------|--------------|--------|
| **Real-time command dashboard** | One-screen station picker, PM2.5 forecast with conformal bands, atmospheric conditions, fire intelligence, transport risk, model performance, wind-arrow map | `GET /api/forecast/pm25`, `/` page |
| **Two-way weather–chemistry coupling** | Aerosol AOD → solar attenuation → PBL suppression → stability feedback (`ml/features/coupling.py`, `coupled_loop.py`); nine interpretable coupling features are computed from stored observations and the latest state is **persisted per station** (`coupling_states`) with `coupling_state` / `coupling_domains` / `data_quality` labels | `GET /api/coupling/features/{station}`, `GET /api/coupling/state`, `GET /api/coupling/{station}`, `POST /api/forecast/coupled` |
| **72-hour AQI forecast** | Persistence / Random Forest / XGBoost (GRU trained & evaluated), all **six criteria pollutants** × horizons {1, 6, 12, 24, 48, 72} | `POST /api/forecast/generate`, `GET /api/forecast/{station}` |
| **Direct PM2.5 forecast engine** | Dedicated per-hour XGBoost with split-conformal prediction intervals | `GET /api/forecast/pm25` + model-card + explanation |
| **Boundary-layer inversion detection** | Lapse-rate computed from 700/850/925/1000 hPa vertical profiles; strength ≥ 0.6 K/100 hPa flagged | `GET /api/inversion/{station}`, `docs/methodology.md` |
| **Pollution event detection** | Statistical surge / relief / sustained high-risk episode detection with atmospheric attribution | `GET /api/events/current`, `docs/events.md` |
| **Scenario (what-if) analysis** | Read-only perturbation engine across wind / PBL / fire / inversion | `POST /api/scenario/analysis`, `docs/scenario_analysis.md` |
| **Cross-model performance** | 4-model evaluation (persistence / RF / XGBoost / GRU) with MAE/RMSE/R² | `GET /api/model/performance`, `/performance` page |
| **Stubble-plume dispersion** | Finite-difference advection–diffusion–deposition solver with FRP fire point sources, wind/PBL/rain forcing | `GET /api/dispersion/forecast`, `ml/features/dispersion_solver.py` |
| **Real CTM engine adapters** | Genuine NOAA HYSPLIT (`hycs_std`, validated byte-for-byte vs ARL `cdump.bin`) and WRF-Chem (`wrfout_d01_*.nc`) — strictly gated, never simulated | `/api/system` engines, `docs/hysplit.md`, `docs/wrfchem_adapter.md` |
| **NCR-wide spatial mapping** | ~2.2 km gridded AQI surface (IDW + downwind advection) + live numerical field | `GET /api/grid/forecast`, `/spatial` page |
| **All six criteria pollutants** | PM2.5, PM10, O₃, NO₂, SO₂, CO | `predict_pollutants`; `TestForecastCompletePollutantSet` |
| **Actionable alerts & explainability** | Alert engine (INFO→WATCH→WARNING→SEVERE) + SHAP feature attribution + natural-language explanations | `GET /api/alerts`, `GET /api/explanation/{station}` |
| **Graded Response Action Plan** | CAQM stage matrix (I ≥201, II ≥301, III ≥401, IV >450) with live NCR assessment | `GET /api/grap/current` |
| **Official IMD gateway** | `api.imd.gov.in` adapter (station 42182 Delhi) — honest 401-gated, never fabricated | `GET /api/imd/forecast`, `docs/imd.md` |
| **Live data refresh** | Scheduler + one-shot CLI pulls weather / fire / pollution idempotently (tz-aware dedup + DB-unique guards) | `make refresh-data`, `LIVE_REFRESH_ENABLED=true` |
| **Official CPCB ingestion** | data.gov.in-backed idempotent upsert with per-station+timestamp uniqueness | `POST /api/pollution/ingest`, `docs/api.md` |

See [`docs/ps_mapping.md`](docs/ps_mapping.md) for the requirement-by-requirement
mapping to the official problem statement, and
[`docs/SIH_FINAL_COMPLIANCE.md`](docs/SIH_FINAL_COMPLIANCE.md) /
[`docs/SIH_GAP_AUDIT.md`](docs/SIH_GAP_AUDIT.md) for evidence and audit.

## Refusal Over Fabrication

The single most important design decision in this system: **when the data cannot
support a forecast, the API refuses instead of producing a plausible-looking
number.** A dashboard that invents clean PM2.5 curves from a sensor that has not
reported for three weeks is worse than no dashboard at all — it is actively
misleading during exactly the episodes where a wrong call costs the most.

| Guard | Behaviour | Where |
|--------|-----------|-------|
| **Minimum history** | Fewer than `MIN_FEATURE_ROWS` (2) aligned observation rows ⇒ `503 insufficient_data`. No rows ⇒ no response body with numbers. | `forecast_service.py` |
| **Staleness** | Newest observation older than `STALE_AFTER_HOURS` (6 h) ⇒ refused, with `is_stale: true` in provenance. | `forecast_service.py` |
| **Missing values stay missing** | `None`/`NaN` propagate as `None`; they are never coerced to `0.0`. Fallbacks are explicit, labelled, and reported in provenance. | `_feature()`, `_upsert_forecasts()` |
| **No borrowed provenance** | A station's provenance is built from *that station's* rows. An empty station reports empty provenance rather than inheriting a neighbour's timestamp. | `build_provenance()` |
| **Re-stamp transparency** | Rows copied forward by the demo bootstrap carry `re_stamped = true`; forecasts built on them are flagged, never presented as live sensor data. | `re_stamped` column, `bootstrap_recent.py` |
| **Audit trail** | Every generation attempt — succeeded, fallback, or refused — is written to `forecast_runs` with the model, artifact SHA-256, row counts, window, and provenance flags. A published number can always be traced to the data behind it. | `ForecastRun`, `record_forecast_run()` |
| **Real zero ≠ missing** | A genuine `0.0` reading (calm wind, no fire) is preserved, not treated as "no data". | `transport_risk_service.py`, `dispersion_service.py` |
| **Gated engines stay gated** | HYSPLIT / WRF-Chem / IMD / ERA5 report a `reason` when credentials or binaries are absent. They never simulate a result and label it real. | `docs/hysplit.md`, `docs/imd.md` |

Refusals return a machine-readable payload so the UI can distinguish "no data
yet" from "backend is broken":

```json
{
  "detail": {
    "code": "insufficient_data",
    "message": "Anand Vihar has 0 aligned observation rows; 2 are required.",
    "station": "Anand Vihar",
    "required_hours": 2,
    "available_hours": 0
  }
}
```

Because a refusal is a legitimate outcome, it is *recorded*, not swallowed — see
`forecast_runs` above. Query it with `GET /api/forecast/{station}` provenance or
directly in SQL to answer "why was there no forecast on Tuesday?".

## Architecture

```
 Official data sources
 ┌──────────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐
 │ CPCB · data.gov.in   │   │ Open-Meteo      │   │ NASA FIRMS · ERA5 · IMD  │
 │ (pollution)          │   │ (weather / PBL) │   │ (fire / atmos / official)│
 └──────────┬───────────┘   └────────┬────────┘   └─────────────┬────────────┘
            │                        │                          │
            ▼                        ▼                          ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │      Ingest & Refresh pipeline  (idempotent · dedup · UTC-aligned)        │
 │   refresh_service → CPCB upserter → AQI engine → feature builder          │
 └───────────────────────────────────┬───────────────────────────────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
 ┌──────────────────────────────┐        ┌──────────────────────────────────┐
 │   ML forecasters (108 models) │        │  Physics & numerical modules      │
 │   XGBoost · RandomForest     │        │  two-way coupling · lapse-rate    │
 │   Persistence · GRU          │        │  dispersion PDE · HYSPLIT/WRF-Chem│
 │   + PM2.5 conformal engine   │        │  grid · events · scenarios · SHAP │
 └──────────────┬───────────────┘        └─────────────────┬────────────────┘
                │                                          │
                ▼                                          ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                     FastAPI backend (backend/app)                         │
 │   api · services · models · SQLAlchemy · Alembic migrations               │
 └───────────────┬────────────────────────────────────┬──────────────────────┘
                 ▼                                    ▼
       ┌─────────────────────┐                ┌─────────────────────────┐
       │  React + TS + Vite  │   proxy ◄──►  │ PostgreSQL (prod) /     │
       │  Recharts · Leaflet │   /api/*      │ SQLite (dev)            │
       └─────────────────────┘                └─────────────────────────┘
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend API** | Python 3.11+ (3.12 in CI), FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, Alembic |
| **Frontend** | React 18, TypeScript, Vite 7, Tailwind CSS, Recharts, Leaflet/react-leaflet, react-router-dom, lucide-react, axios |
| **Machine Learning** | XGBoost, scikit-learn (RandomForest, split-conformal), SHAP, NumPy, Pandas, joblib, custom NumPy GRU |
| **Data Stores** | PostgreSQL 16 (prod / Docker) · SQLite (dev) |
| **External data** | CPCB (data.gov.in), Open-Meteo (+ pressure levels), NASA FIRMS VIIRS, Copernicus ERA5 (CDS), IMD, NOAA HYSPLIT GDAS, WRF-Chem `wrfout_d01_*.nc` |
| **Orchestration** | Docker Compose (Postgres · backend · frontend), GitHub Actions CI |
| **Deployment** | Vercel (frontend SPA) · Render (FastAPI) · Neon (managed PostgreSQL) |

## Data Sources & Honesty Gates

| Source | Data | Stage |
|--------|------|-------|
| **CPCB / data.gov.in** | 6 pollutants, hourly, per station `3b01bcb8-…` | Real-time feed if `DATA_GOV_API_KEY` set; persisted archive otherwise — **never fabricated** (400 without key) |
| **Open-Meteo** | Ambient weather + **vertical 1000/925/850/700 hPa profiles** per station | Active primary weather path (verified live, HTTP 200) |
| **NASA FIRMS (VIIRS)** | Fire radiative power detections, Punjab–Haryana → NCR domain | Live pull if `NASA_FIRMS_MAP_KEY` set; 58k+ bundled archive otherwise |
| **Copernicus ERA5 (CDS)** | `t2m` / `sp` / `blh` single-level reanalysis, sampled at the 17 stations | Offline ingestion (`ml/features/era5_surface.py`) — gated on CDS credentials, honest empty placeholder otherwise |
| **IMD (official)** | `api.imd.gov.in` city forecast (station 42182, Delhi) | Gated — 401 without key/IP whitelist; reason surfaced on `/api/imd/forecast` |
| **NOAA HYSPLIT** | Lagrangian dispersion / back-trajectory | Plug-in adapter launches real `hycs_std` + parses genuine binary `cdump.bin` (validated against ARL); **surrogate PDE mode** if the binary/met files are absent |
| **WRF-Chem** | Convective transport surface | Only real `wrfout_d01_*.nc` files are absorbed (`ml/ctm/wrfchem_adapter.py`); status honestly "gated" otherwise |

**17 stations:** Anand Vihar, RK Puram, ITO, Dwarka, Punjabi Bagh, Lodhi Road,
Sirifort, Shadipur, Okhla Phase-2, Ashok Vihar, Mundka, Jahangirpuri, Aya
Nagar, Vivek Vihar (Delhi) · Teri Gram (Gurugram) · Noida Sector-62 (Noida)
· Faridabad (Faridabad).

## Model Performance

Evaluated on a held-out **chronological** test split (no random shuffle; ~21k
out-of-sample rows per model). **R² (coefficient of determination, XGBoost):**

| Pollutant | 1 h | 6 h | 12 h | 24 h | 48 h | 72 h |
|-----------|------|------|------|------|------|------|
| **PM2.5** | 0.949 | 0.787 | 0.741 | 0.686 | 0.617 | 0.556 |
| **PM10**  | 0.924 | 0.775 | 0.726 | 0.696 | 0.608 | 0.575 |
| **O₃**    | 0.920 | 0.734 | 0.703 | 0.670 | 0.635 | 0.599 |
| **NO₂**   | 0.894 | 0.667 | 0.638 | 0.588 | 0.363 | 0.207 |
| **SO₂**   | 0.749 | 0.375 | 0.213 | 0.205 | 0.224 | 0.154 |
| **CO**    | 0.866 | 0.626 | 0.585 | 0.544 | 0.480 | 0.451 |

**Takeaways (honestly reported):**

- Short-horizon skill is strong across all six pollutants (PM2.5 R² ≈ 0.95 at
  1 h, MAE ≈ 15.7 µg/m³), far above persistence (which is *negative* R² —
  e.g. PM2.5 −0.43 at 1 h).
- Skill decays gracefully toward 72 h as microphysics become unpredictable.
  Long-horizon **NO₂** (0.207) and **SO₂** (0.154) are the acknowledged weak
  spots — chemically short-lived species with sparse spatio-temporal coverage.
- The dedicated **PM2.5 hourly engine** (73 horizon-specific XGBoost models,
  split-conformal intervals) scores **R² = 0.879 @ 1 h, 0.615 @ 24 h,
  0.424 @ 72 h** on a separate 15,767-hour out-of-sample window.
- The custom NumPy **GRU** was trained and evaluated alongside (2-layer,
  hidden 128, seq_len 48) — it honestly underperforms the trees (R² = 0.336
  vs 0.879 at 1 h) and is retained as a documented ensemble candidate rather
  than overstated.

Full MAE / RMSE / R² / MAPE per model (persistence, RF, XGBoost, GRU) are in
[`models/metrics.json`](models/metrics.json) and
[`models/pm25/evaluation.json`](models/pm25/evaluation.json).

## Data & Machine-Learning Pipeline

1. **Acquire** — real data: the official GoI CPCB feed
   (`backend/app/services/cpcb_service.py`, resource
   [`3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`](https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69)),
   Open-Meteo weather + vertical profiles, NASA FIRMS fires, and Copernicus
   ERA5 — or the live refresh service (`backend/app/services/refresh_service.py`).
2. **Build** — `scripts/build_dataset.py` (feat. ~131k rows × 124 features)
   fuses raw readings into time-aligned station series with engineered
   features (AOD, transmittance, corrected PBL, stability, ventilation,
   fire-impact, transport-risk, cyclic temporal encodings).
3. **Train** — `python -m ml.training.trainer` fits per-pollutant,
   per-horizon models; `scripts/evaluate_models.py` re-scores them.
   `python -m ml.training.train_pm25` builds the direct PM2.5 conformal engine;
   `python -m ml.training.train_gru` the GRU candidate.
4. **Serve** — FastAPI loads the persisted artifacts and forecasts on demand,
   feeding AQI, alerts, coupling, grid, dispersion, events, GRAP and scenario
   endpoints.

### Official CPCB pollution feed (data.gov.in)

The pollution cursor points at the **official** Government dataset "Real time
Air Quality Index from various locations" (`CPCB/DPCC`). Readings are upserted
idempotently into `stations` / `pollution_observations`, enforced by a
`(station_id, timestamp)` unique constraint.

```bash
# 1. Get a free API key: https://api.data.gov.in → register → MyAccount
# 2. Configure it:
DATA_GOV_API_KEY=your_key_here
# 3. Trigger the official CPCB ingestion:
curl -X POST http://localhost:8000/api/pollution/ingest
# 4. Verify:
curl http://localhost:8000/api/pollution/latest
```

Ingestion fetches paginated, per-city records for
`Delhi, Gurugram, Noida, Ghaziabad, Faridabad` (configurable via
`DATA_GOV_NCR_CITIES`), normalizes one row per pollutant per timestamp into
single observations, and upserts them idempotently. Without a key the ingest
endpoint returns `400` (`DATA_GOV_API_KEY is not set`) rather than inventing
data.

## Database & Migrations

Schema is versioned with **Alembic** and is the single source of truth for both
PostgreSQL (production) and SQLite (development). Migrations run automatically:

- **Render** — `preDeployCommand: python backend/scripts/migrate_safely.py` in
  [`render.yaml`](render.yaml), so a release never serves traffic against a stale
  schema.
- **Docker** — [`backend/scripts/entrypoint.sh`](backend/scripts/entrypoint.sh)
  migrates, then `exec`s uvicorn (keeping it PID 1 so Docker signals reach it).
  Set `SKIP_MIGRATIONS=1` to bypass. The entrypoint **refuses to start** if
  migrations fail, rather than serving confusing 500s.
- **Local** — `python -m alembic upgrade head`, or let
  `backend/app/database.py::apply_migrations()` reconcile a SQLite dev database.

Current head: **`b8d3f1a9c4e2`** — station foreign keys, forecast uniqueness, the
`forecast_runs` audit table, and `pollution_observations.re_stamped`.

**`migrate_safely.py`, not a bare `alembic upgrade head`.** Deploy targets run
`migrate_safely.py`, which inspects the database and branches three ways:

| Detected state | How | Action |
|----------------|-----|--------|
| `alembic_version` present | `versioned` | `alembic upgrade head` |
| Tables exist, no `alembic_version` | `unversioned` | `alembic stamp head` |
| No application tables | `empty` | `alembic upgrade head` |

The `unversioned` branch is the one that matters, and the reason a bare upgrade
is unsafe to ship. A pre-existing database that has the tables but no version
mark would have the **whole chain replayed against itself** — every
`CREATE TABLE` fails against a table that already exists, and because the
failure is non-idempotent the next deploy fails identically. Stamping is correct
there because those tables were built by `create_all()` from the same ORM
metadata the chain is generated from, so they are already at head. Blindly
stamping is *also* wrong, though, so the script refuses when a legacy database
is missing tables added by recent migrations (e.g. `forecast_runs`) — claiming
head there would hide a real gap until runtime. Use `--status` to inspect
without writing, and `--force-upgrade` to replay deliberately.

**Cross-dialect correctness.** The chain is verified in both directions against a
throwaway PostgreSQL 16 *and* SQLite: `upgrade head` → `downgrade base` →
`upgrade head`, followed by `alembic revision --autogenerate` confirming **zero
drift** against the ORM metadata on both engines. Three portability rules are
load-bearing and easy to regress:

1. **No `DELETE ... USING`** (PostgreSQL-only) — dedup uses the correlated
   `WHERE id NOT IN (SELECT MAX(id) … GROUP BY …)` form, valid on both engines.
2. **No `ALTER SEQUENCE` / `RENAME CONSTRAINT` on SQLite** — those are guarded by
   a dialect check; SQLite has no named sequences.
3. **Batch mode is SQLite-only.** Alembic's `batch_alter_table` rebuilds the
   table on PostgreSQL, which **drops the id sequence the table owns** and
   replaces it with `_alembic_tmp_<table>_id_seq` — silently orphaning every
   later `INSERT`. PostgreSQL uses plain `ALTER TABLE … ADD CONSTRAINT`.

> A historical note worth keeping: because the baseline originally created `id`
> as a bare `Integer` (no `SERIAL`) and `main.py` *swallowed* migration
> exceptions, the production schema was actually being built by
> `Base.metadata.create_all()` with **no `alembic_version` table at all**. The
> baseline now declares `autoincrement=True` so the chain and `create_all` agree,
> and `migrate_safely.py` handles databases left in that state.

## Repository Layout

```
aerocast-ncr/
├── backend/
│   ├── app/                   # FastAPI application
│   │   ├── api/               #   REST route modules (auth, forecast, GRAP, IMD…)
│   │   ├── services/          #   domain logic (refresh, events, coupling, dispersion…)
│   │   ├── models/            #   SQLAlchemy models (db_models.py)
│   │   ├── schemas/           #   Pydantic v2 request/response schemas
│   │   ├── main.py            #   app factory + lifespan (migrations → schema → seed)
│   │   └── database.py        #   engine/session + SQLite apply_migrations
│   ├── scripts/               #   bootstrap_recent.py · migrate_safely.py · audit_re_stamps.py · entrypoint.sh · …
│   └── tests/                 # unit + integration pytest suites
├── ml/
│   ├── features/              # coupling, coupled_loop, dispersion_solver, grid, …
│   ├── models/                # xgboost, random_forest, persistence, gru
│   ├── preprocessing/         # weather/pollution/fire/atmosphere processors
│   ├── training/              # trainer, train_pm25, train_gru, evaluate
│   ├── inference/             # forecasters (predictor, pm25_forecaster)
│   ├── ctm/                   # HYSPLIT & WRF-Chem real-engine adapters
│   └── evaluation/            # metrics
├── frontend/src/              # React + TS SPA (pages/, components/, hooks/, auth/)
├── alembic/versions/          # versioned schema migrations (head: b8d3f1a9c4e2)
├── data/                      # processed datasets (gitignored large exports)
├── models/                    # trained artifacts (joblib / JSON model cards)
├── docs/                      # methodology, API, events, scenarios, deploy, ERA5, IMD, HYSPLIT…
├── scripts/                   # CLI tools (download*, build_dataset, refresh, fetch_*)
├── tests/                     # top-level integration tests
├── app/main.py                # Render shim: `uvicorn app.main:app` from repo root
├── .github/workflows/         # CI (ruff · pytest · Alembic-on-Postgres · tsc/vite)
├── render.yaml                # Render blueprint (pre-deploy migrations)
├── docker-compose.yml         # full-stack compose (db · backend · frontend)
├── Makefile                   # developer command centre
└── pyproject.toml             # packaging + ruff/pytest/cov config
```

> The root-level `app/main.py` is a one-line shim that re-exports
> `backend.app.main:app`, so Render's stock `uvicorn app.main:app` start command
> works from the repository root. `render.yaml` and the Docker image use the
> explicit `backend.app.main:app` / `app.main:app` paths instead and do not
> depend on it. Note that this shim *shadows* `backend/app` if `backend/` is not
> first on `sys.path` — which is why `alembic/env.py` enforces that ordering
> explicitly.

## Quick Start — Docker

```bash
# 1. Clone and configure
git clone https://github.com/methila-2056/Air-Pollution-Weather-Coupled-Forecasting-System-Delhi-NCR-Focus-.git
cd aerocast-ncr
cp .env.example .env            # local defaults
cp docs/deploy.env.example deploy.env   # (optional) production overrides

# 2. Build and start the full stack (Postgres + API + frontend)
docker compose up -d --build

# 3. The entrypoint applies `alembic upgrade head` before serving;
#    station seeding and the live-refresh scheduler then run in-app.

# 4. Access
#    Frontend : http://localhost:5173
#    API docs  : http://localhost:8000/docs
#    Liveness  : http://localhost:8000/api/health
#    Readiness : http://localhost:8000/health   (round-trips the database)
```

`/api/health` is a **liveness** probe and answers as soon as the process accepts
connections. `/health` is a **readiness** probe that round-trips Postgres, so it
is the right target for a human but the wrong target for a container or platform
health check — a cold or pooled database will exceed the timeout and trigger a
restart loop. `docker-compose.yml`, the `Dockerfile` `HEALTHCHECK`, and
`render.yaml` all deliberately point at `/api/health`.

Stop with `docker compose down` (add `-v` to also drop the `pgdata` volume).

## Quick Start — Bare Metal

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
python -m alembic upgrade head                       # create/migrate the schema
uvicorn app.main:app --host 0.0.0.0 --port 8000      # run from backend/

# Frontend (separate terminal)
cd frontend && npm install && npm run dev            # http://localhost:5173
```

Download data and build the dataset with
`scripts/fetch_all.{sh,ps1}` or `python scripts/build_dataset.py`. For local
development against SQLite, `backend/app/database.py` `apply_migrations()`
keeps the schema in sync with the Alembic chain (including the
`uq_weather_station_ts` unique index).

> **Cloning is slow by design.** `models/` (~1.26 GB) and `data/` are committed
> so a fresh clone is immediately runnable — model artifacts are joblib
> binaries that a Docker build cannot regenerate. If clone time becomes a
> problem, narrow it with `git clone --filter=blob:none` plus a sparse checkout
> of `backend/`, `frontend/`, `ml/`, and `docs/`, then fetch `models/` on demand.
> The ~194 MB SQLite database (`aerocast_ncr.db`) is git-ignored and must be
> built locally.

## API Summary

All endpoints live under `/api` (interactive docs at `/docs`):

| Area | Endpoints |
|------|-----------|
| Auth & system | `POST /api/auth/login`, `POST /api/auth/demo`, `GET /api/auth/me`, `GET /api/system`, `GET /api/health` (liveness), `GET /health` (readiness) |
| Station & current AQI | `GET /stations`, `GET /stations/{station}`, `GET /current/{station}` |
| Official CPCB pollution | `POST /pollution/ingest`, `GET /pollution/latest`, `GET /pollution/stations`, `GET /pollution/{station_id}/history` |
| Forecast | `POST /forecast/generate`, `GET /forecast/{station}`, `GET /forecast/ncr`, `GET /forecast/comparison/{station}`, `POST /forecast/coupled` |
| PM2.5 forecast engine | `GET /forecast/pm25`, `GET /forecast/pm25/model-card`, `GET /forecast/pm25/explanation` |
| Weather | `GET /weather/{station}`, `GET /weather/{station}/history` |
| Atmosphere & coupling | `GET /atmosphere/current`, `GET /coupling/features/{station}`, `GET /coupling/state`, `GET /coupling/state/{station}`, `GET /coupling/{station}`, `GET /inversion/{station}` |
| Fire / transport / plume | `GET /fire-activity`, `GET /fire/transport`, `GET /plume-risk`, `GET /fire/hotspots`, `GET /fires/latest`, `GET /transport-risk/current`, `GET /stubble` |
| Spatial & dispersion | `GET /grid/forecast`, `GET /grid/overview`, `GET /dispersion/forecast` |
| Events & scenarios | `GET /events/current`, `POST /scenario/analysis` |
| IMD official gateway | `GET /imd/forecast` |
| GRAP (CAQM) | `GET /grap/stages`, `GET /grap/current`, `GET /grap/{station}` |
| Explainability | `GET /explanation/{station}` |
| Alerts & metrics | `GET /alerts`, `GET /model/performance`, `GET / POST /model/metrics` |
| Summary & export | `GET /summary`, `GET /export/forecast.csv` |

Request/response schemas are described in [`docs/api.md`](docs/api.md) and are
introspectable at `/docs`.

## Frontend Pages

| Route | Page |
|-------|------|
| `/login` | Auth (demo login available) |
| `/` | **Command Dashboard** — station picker, PM2.5 forecast + conformal band, atmospheric conditions, fire intelligence & transport risk, SHAP explainability, GRAP panel, model performance, wind-arrow station map |
| `/overview` | NCR KPIs & regional summary |
| `/forecast` | 72-hour multi-horizon forecast charts |
| `/map` | Leaflet station map + FIRMS hotspots + plume transport |
| `/spatial` | Gridded/dispersion heatmap with hour scrubber |
| `/atmosphere` | Per-station atmospheric conditions (wind, PBL, ventilation, inversion, trapping) |
| `/stubble` & `/fire-plume` | Stubble-plume / fire activity & dispersion |
| `/explanation` | SHAP feature attribution (AI explanation) |
| `/alerts` | Alert centre |
| `/events` | Pollution events (surge / relief / high-risk episode) |
| `/performance` | 4-model (persistence / RF / XGBoost / GRU) evaluation comparison |
| `/architecture` | Honest engine status (what is really wired vs gated) |
| `/data` | Data tools / CSV export |
| `/profile` | Analyst profile & session |

## Testing & Quality Gates

```bash
python -m pytest backend/tests -q                   # 974 unit + integration tests
python -m ruff check backend/app backend/tests       # lint (CI-scoped)
cd frontend && npm run build                         # tsc type-check + production build
```

The suite covers the refusal contract explicitly — `test_forecast_data_sufficiency.py`
asserts that under-powered or stale stations return `503 insufficient_data`, that
genuine `0.0` values are preserved, and that provenance never leaks across
stations. Forecast regeneration is tested for idempotency against the
`(station_id, horizon_hours)` unique constraint.

Convenience targets: `make test`, `make test-unit`, `make test-integration`,
`make build`, `make refresh-data`, `make pip-audit` — see the `Makefile`.

CI additionally verifies **`alembic upgrade head` against a fresh PostgreSQL
16** and re-runs the integration/API suite against it, so schema migrations are
proven on every push — not just on the developer's laptop.

## Deployment & CI/CD

### Vercel + Render (managed cloud)

Production is split across two platforms — see
[`render.yaml`](render.yaml) and [`frontend/vercel.json`](frontend/vercel.json):

| App | Host | How |
|-----|------|-----|
| **Frontend** (React SPA) | **Vercel** | Import the repo, root dir = `frontend`, framework preset *Vite*, build `npm run build`, output dir `dist`. `vercel.json` rewrites `/api/*` → the Render backend URL and falls back to `index.html` for SPA routes. |
| **Backend** (FastAPI) | **Render Web Service** | Blueprint `render.yaml` → build `pip install -r backend/requirements.txt`, **pre-deploy `python backend/scripts/migrate_safely.py`**, start `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` (repo-root cwd so `ml/`, `models/`, `data/`, `alembic.ini` resolve), health check `/api/health`. |
| **Database** | **Neon / Supabase** | External Postgres. Set `DATABASE_URL` on Render (overriding it disables the SQLite default and triggers migrations + seeding at boot). |

Mandatory env vars on Render: `DATABASE_URL`, `CORS_ORIGINS`
(`https://<your-app>.vercel.app`), `SECRET_KEY` (long random). Optional:
`NASA_FIRMS_MAP_KEY`, `DATA_GOV_API_KEY`, `IMD_API_KEY`,
`LIVE_REFRESH_ENABLED=true`, `DEMO_USER_PASSWORD`.

> **Before your first deploy to an existing database:** if it was created by an
> older build of this app, it may have **no `alembic_version` table** (see
> [Database & Migrations](#database--migrations)). Nothing to do by hand —
> `preDeployCommand` runs `migrate_safely.py`, which detects that state and
> stamps instead of replaying. To check ahead of a deploy without writing:
> `python backend/scripts/migrate_safely.py --status`.

> Free-tier caveats: Render free Postgres expires after 90 days, and the free
> web service idles after ~15 min (cold start = migrations + model load, a few
> seconds). The repository ships ~1.5 GB of committed `models/` + `data/`, so
> clones are slow; keep `data/` lean if builds become a bottleneck.

### CI/CD

- **Compose runbook** — [`docs/deployment.md`](docs/deployment.md) (migration
  flow, live refresh, reverse-proxy/HTTPS, `pgdata` backup/restore,
  first-boot checks).
- **Production env template** — [`docs/deploy.env.example`](docs/deploy.env.example).
- **CI** — `.github/workflows/ci.yml`:
  1. *Backend:* ruff + full pytest suite (`974 passed`).
  2. *Migrations:* `alembic upgrade head` against a fresh Postgres 16,
     then integration/API tests against it.
  3. *Frontend:* `tsc` + `vite build`.

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/methodology.md`](docs/methodology.md) | AQI, features, models, SHAP, coupling, dispersion (§9 surrogate vs. WRF-Chem) |
| [`docs/SCIENTIFIC_METHODOLOGY.md`](docs/SCIENTIFIC_METHODOLOGY.md) | Formal formulas, constants, units & assumptions (coupling engine, inversion, fire impact, AQI, alerts, models) |
| [`docs/SIH26082_TRACEABILITY.md`](docs/SIH26082_TRACEABILITY.md) | PS clause → implementation → file → API → UI → evidence/test → status matrix |
| [`docs/SIH26082_GAP_AUDIT.md`](docs/SIH26082_GAP_AUDIT.md) | Nine-point gap audit (implemented / partial / missing / UI-only / ML-connected / real-data / weak / simulated / improvable) |
| [`docs/SIH26082_FINAL_AUDIT.md`](docs/SIH26082_FINAL_AUDIT.md) | Final audit (A–P) with FULLY / PARTIALLY / NOT IMPLEMENTED classification |
| [`docs/SIH26082_IMPLEMENTATION_AUDIT.md`](docs/SIH26082_IMPLEMENTATION_AUDIT.md) | Requirement-by-requirement 24-row status table + verification runs |
| [`docs/architecture.md`](docs/architecture.md) | System layers & component diagram |
| [`docs/ps_mapping.md`](docs/ps_mapping.md) | Requirement → implementation mapping |
| [`docs/PS_SUBMISSION.md`](docs/PS_SUBMISSION.md) | Problem-statement submission summary |
| [`docs/SIH_FINAL_COMPLIANCE.md`](docs/SIH_FINAL_COMPLIANCE.md) | Final compliance evidence (R1–R20, verified runs) |
| [`docs/SIH_GAP_AUDIT.md`](docs/SIH_GAP_AUDIT.md) | Gap audit vs. a literal WRF-Chem deployment |
| [`docs/deployment.md`](docs/deployment.md) | Deploy runbook & environment reference |
| [`docs/api.md`](docs/api.md) | API reference (incl. PM2.5 engine, events, scenario) |
| [`docs/events.md`](docs/events.md) | Pollution event detection specification |
| [`docs/scenario_analysis.md`](docs/scenario_analysis.md) | What-if scenario engine methodology |
| [`docs/dataset.md`](docs/dataset.md) | Data sources & schema |
| [`docs/era5.md`](docs/era5.md) | ERA5 reanalysis ingestion |
| [`docs/imd.md`](docs/imd.md) | Official IMD gateway adapter |
| [`docs/hysplit.md`](docs/hysplit.md) | NOAA HYSPLIT real-engine integration |
| [`docs/wrfchem_adapter.md`](docs/wrfchem_adapter.md) | WRF-Chem output adapter |
| [`docs/UI_REDESIGN_AUDIT.md`](docs/UI_REDESIGN_AUDIT.md) | Frontend UX audit & redesign notes |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproducible build/train pipeline |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

## Roadmap

- **Gridded full-domain PM2.5 0–72 h** — extend the point-station engine to a
  seamless hourly NCR grid (dispersion-solver-guided interpolation).
- **Mobile / public advisory slice** — citizen-facing AQI + school-closure
  guidance backed by the same APIs.
- **Source-apportionment tab** — stubble vs. local-emissions decomposition
  using wind-aligned fire impact + dispersion attribution.
- **Ensemble conformal bands** — multi-seed XGBoost + GRU mixtures for tighter
  calibrated intervals.
- **CAQM e-portal integration** — machine-readable GRAP alert feed for
  automated action workflow.
- **Annual retraining pipeline** — scheduled feature-store refresh + model
  versioning (the system already logs per-model metrics for drift review).

## Honest Scope

The problem statement names "WRF-Chem or similar coupled frameworks". Shipping
compiled WRF-Chem is an HPC-scale effort (Fortran toolchain, MOZART/GOCART
chemistry, full emission inventories). AeroCast-NCR therefore implements a
**physics-informed numerical surrogate** — a vectorised finite-difference
transport core with explicit two-way coupling — that delivers the PS's
*functional* outcomes (plume dispersion prediction, meteorological interlink,
72 h AQI, NCR-wide coverage) reproducibly on commodity hardware. When a real
engine is available (HYSPLIT binary + GDAS/EDAS met, or WRF-Chem `wrfout`
files), the adapters exercise it genuinely — validated, composited and plainly
disclosed. See [`docs/methodology.md` §9](docs/methodology.md) for the
tradeoff table, validation strategy, era5/imd/hysplit adapters, and
[`docs/SIH_GAP_AUDIT.md`](docs/SIH_GAP_AUDIT.md) for what a literal WRF-Chem
deployment would require. **Nothing in this repository claims results it
cannot produce.**

### Scientific limitations (stated honestly)

- **Coupling features are potentials/tendencies**, not measurements; the
  meteorology–pollution term is an explicit *data-driven surrogate*, not a
  physics-based chemistry loop. Missing inputs render "Data unavailable".
- **PBL and pressure-level temperatures are Open-Meteo model fields**, not
  in-situ instruments; "Strong/Moderate/Weak" inversion grades and PBL bands
  are heuristic thresholds, never reanalysis climatology.
- **Inversion uses a documented PBL-height proxy** when fewer than two stored
  pressure levels exist.
- **Transport times and pathway corridors are advective estimates** (straight
  line at the surface wind speed) — not a boundary-layer diffusion result.
- **R² is goodness-of-fit on a chronological held-out split**, not "accuracy";
  long-horizon NO₂/SO₂ skill is modest, and forecast uncertainty grows with
  horizon (conformal intervals are provided for the direct PM2.5 engine).
- **WRF-Chem / HYSPLIT / ERA5 / IMD are operator-gated**: live runs occur only
  when the operator provides the external engine output or credentials; the
  system reports `reason` strings when gated and never fabricates data. Exact
  formulas, constants and units for every indicator are in
  [`docs/SCIENTIFIC_METHODOLOGY.md`](docs/SCIENTIFIC_METHODOLOGY.md).

### Known issues

Stated plainly rather than buried — each is a real limitation of the current
tree, not a roadmap item.

- **The flat `xgboost_pm25_*` artifacts are stale relative to the serving
  feature contract.** They declare 114 feature names; the current feature
  builder supplies 107, missing `latitude`, `longitude`, `mean_frp`,
  `max_bright`, and three season dummies. The `/api/explanation/{station}`
  endpoint was repointed to the matched `models/pm25` family and is no longer
  affected, but the **direct forecast path still loads the flat artifacts**, and
  `_model_predict()` can substitute `0.0` for a feature the builder did not
  produce. The correct fix is to retrain the flat model on the current contract
  (or retire it in favour of the per-horizon family) — not to keep the
  substitution. Until then, treat `mean_frp` / `max_bright` / season-dummy
  contributions in a served PM2.5 explanation as unvalidated.
- **`re_stamped` is not backfilled for historical rows.** The migration
  deliberately leaves pre-existing rows at `false` rather than guessing: a
  pm25-only heuristic measured 87% false positives against the real database,
  and a full-value-tuple heuristic mostly matched all-`NULL`/`aqi=0` rows. Legacy
  re-stamps are therefore indistinguishable from real measurements *after the
  fact*. New rows are stamped correctly by `bootstrap_recent.py`; use the
  read-only `backend/scripts/audit_re_stamps.py` to review candidates manually.
  Legacy rows remain covered by the 6-hour staleness refusal.
- **Empty-state GET endpoints are not uniformly explicit.** `/api/forecast/{station}`,
  `/api/summary`, `/api/grid/*`, and `/api/pollution/*` do not all yet return a
  dedicated empty/provenance payload the way the forecast-generation paths do.
- **The hosted backend is unverified** — see
  [Deployment status](#deployment-status-verified-2026-09-28) above.

## Contributing & Security

- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development workflow,
  conventions and how to report issues.
- See [`SECURITY.md`](SECURITY.md) for the responsible-disclosure policy.

## Acknowledgments

- **Problem statement SIH26082** — Ministry of Earth Sciences / NCMRWF.
- **Data** — CPCB via data.gov.in, Open-Meteo, NASA FIRMS (VIIRS), Copernicus
  CDS (ERA5), IMD, NOAA ARL (HYSPLIT / GDAS).
- **Ecosystem** — XGBoost, scikit-learn, SHAP, FastAPI, React, and the open
  science community.

## License

MIT — see [`LICENSE`](LICENSE).