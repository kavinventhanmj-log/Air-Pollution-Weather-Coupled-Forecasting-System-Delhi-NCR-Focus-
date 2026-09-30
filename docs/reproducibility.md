# Reproducibility

This document explains how every artifact is produced and how to re-run the
pipeline from source data to served forecasts.

## Data pipeline

1. **Fetch sources**
   ```bash
   python scripts/download_weather.py
   python scripts/download_pollution.py
   python scripts/download_fire.py
   python scripts/download_atmosphere.py
   ```
   Outputs land under `data/{weather,pollution,fire}/<station>_*.csv`.
   `scripts/generate_fire_data.py` synthesizes deterministic demo fire
   readings when no FIRMS key is available.

2. **Build the featured dataset**
   ```bash
   python scripts/build_dataset.py
   ```
   Produces `data/processed/featured_dataset.csv` (120 MB, ~131k rows,
   124 features). Because GitHub rejects pushes >100 MB, the committed export
   is split into `featured_dataset_part1.csv` + `featured_dataset_part2.csv`;
   reassemble with `cat`/`copy /b` per `data/processed/SPLIT_NOTE.txt`.

## Model training

```bash
python -m ml.training.trainer
```

- Trains persistence, random-forest and XGBoost regressors for **6 pollutants
  × 6 horizons** (108 models total).
- Saved to `models/{model_type}_{target}_{h}h.joblib` with a sidecar JSON;
  `models/metrics.json` tracks MAE/RMSE/R²/MAPE.
- `scripts/evaluate_models.py` recomputes evaluation tables against held-out
  actuals (`models/predicted_vs_actual_*.csv`).

## Backend runtime

```bash
cp .env.example .env          # defaults to a local SQLite DB
cd backend
uvicorn app.main:app --reload --port 8000
```

On startup the app:
1. creates tables (`Base.metadata.create_all`) and applies additive migrations
   (`apply_migrations` in `app/database.py`);
2. seeds Delhi NCR baseline stations/readings when the DB is empty;
3. optionally starts the live-refresh scheduler
   (`LIVE_REFRESH_ENABLED=true`, interval `LIVE_REFRESH_INTERVAL_HOURS`).

## Frontend

```bash
cd frontend
npm ci
npm run dev      # dev server on :5173 (proxies /api → :8000)
npm run build    # type-check + production bundle
```

## End-to-end

Run the backend and frontend as two processes (Docker was removed; see
`docs/deployment.md`):

```bash
cd backend && uvicorn app.main:app --port 8000   # backend   http://localhost:8000/docs
cd frontend && npm run dev                       # frontend  http://localhost:5173
```
`ml/`, `models/` and `data/` are mounted read-only into the backend container,
so rebuilt/retrained artifacts are picked up without rebuilding the image.

## Test suite

```bash
python -m pip install -e ".[dev]"
python -m pytest backend/tests -q
```
Unit tests (fast, offline) live in `backend/tests/unit/`; integration tests
(FastAPI `TestClient` + seeded temp SQLite) in `backend/tests/integration/`.
Determinism: no network in tests, fixed seed for the demo fire data.

## Versions pinned

- Python ≥ 3.11 (dev/tested on 3.12)
- Node ≥ 18 (dev/tested on 20)
- See `pyproject.toml` / `backend/requirements.txt` / `frontend/package.json`