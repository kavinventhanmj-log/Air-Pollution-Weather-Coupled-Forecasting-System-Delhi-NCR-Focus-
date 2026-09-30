# Deployment

Docker was removed. The supported paths are **Render** (the hosted production
deployment) and **local development** (system Python + npm against SQLite, or
PostgreSQL with Alembic).

The backend runs Alembic migrations automatically at startup against
PostgreSQL, through the guarded entrypoint `backend/scripts/migrate_safely.py`
rather than a bare `alembic upgrade head`.

## Render (production)

`render.yaml` is the source of truth for the hosted deployment:

- **Web service** — native Python (no container image built from this repo)
- **PostgreSQL** — provisioned database
- **Pre-deploy** — `preDeployCommand: python backend/scripts/migrate_safely.py`

Required environment variables are set in the Render dashboard (or via
`render.yaml` for non-secret defaults). See `docs/deploy.env.example` for the
full list and defaults.

Set a real `NASA_FIRMS_MAP_KEY` for reliable live fire ingestion, and
`LIVE_REFRESH_ENABLED=true` to run the background refresh scheduler.

`render.yaml` sets `autoDeploy: false`, so application changes do not redeploy on
their own — trigger a deploy from the Render dashboard when you want one.

## Local development

```bash
python -m pip install -e ".[dev]"
cp .env.example .env

# backend (SQLite dev default)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# frontend (separate terminal)
cd frontend && npm ci && npm run dev
```

The dashboard is at `http://localhost:5173`, which proxies `/api` to the backend
via the Vite dev server.

### PostgreSQL with Alembic (optional)

```bash
export DATABASE_URL=postgresql://aerocast:pass@localhost:5432/aerocast_ncr
python -m alembic upgrade head   # or: python backend/scripts/migrate_safely.py
```

Prefer `migrate_safely.py` — it detects a database created by the legacy
`create_all()` path (application tables, no `alembic_version`) and stamps
instead of replaying the chain, which is exactly the production failure this
guards against.

### Live data refresh

For a one-shot refresh, or a dry run to preview before committing:

```bash
cd backend
python -m scripts.refresh_once --dry-run   # preview
python -m scripts.refresh_once             # commit
```

### Backup / restore (PostgreSQL)

```bash
pg_dump -U aerocast -d aerocast_ncr > backup_$(date +%Y%m%d).sql
pg_restore -U aerocast -d aerocast_ncr backup_YYYYMMDD.sql
```

For a managed database (Neon, Render, Supabase) use the provider's own
point-in-time recovery and snapshot tooling instead of local dumps.

## Startup verification

```bash
curl -fsS http://localhost:8000/health            # {"status":"healthy",...}
curl -fsS http://localhost:8000/api/data-quality  # per-table row counts + gaps
curl -fsS http://localhost:8000/api/summary       # NCR KPIs
curl -fsS http://localhost:8000/api/stations      # 17 seeded stations
```

Look in the backend log for `Alembic migrations applied at startup` (Postgres) or
`Seeded 5 default Delhi NCR stations`.

## Environment reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///./aerocast_ncr.db` | SQLAlchemy connection string; Postgres triggers Alembic migrations |
| `SECRET_KEY` | *(dev placeholder)* | JWT signing key. Production boot is fail-closed if this is the placeholder or <32 chars |
| `NASA_FIRMS_MAP_KEY` | *(empty)* | Optional FIRMS API key for live fire data |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed browser origins |
| `ENVIRONMENT` | `development` | App runtime environment label. Must be exactly `production` for the fail-closed startup guard; anything else (incl. misspellings) boots with the guard off and emits a warning |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LIVE_REFRESH_ENABLED` | `false` | Whether the refresh scheduler runs |
| `LIVE_REFRESH_INTERVAL_HOURS` | `3` | Scheduler cadence |
| `DEMO_HYDRATE_EMPTY_DB` | `false` | Re-stamp the bundled archive into the recent window when the database has no fresh readings |
| `ENABLE_DEMO_USER` | unset (= off) | Loud opt-in to the public demo login at `GET /api/auth/demo`. Required for the hosted demo deployment |
| `CONTROL_ROOM_PREWARM` | unset (= on in production) | Fill the TTL cache with the heavy control-room reads in the background after startup. Worth it on scale-to-zero hosts (Render free tier), where a cold wake otherwise leaves the first dashboard load queueing behind cold ~12 s aggregations. Never blocks readiness; failures are logged and skipped. Unset, the sweep follows the environment: on when `ENVIRONMENT=production`, off for local dev/pytest/CI. Set `false` to opt out on a production host. Check it took effect with `GET /api/system` → `prewarm`. |

See `docs/deploy.env.example` for a production template.

## Production tips

- **TLS and CORS.** Render terminates TLS. Set `CORS_ORIGINS` to the real public
  origin (e.g. `https://forecast.example`); wildcard origins are rejected at
  boot in production.
- **Production startup is fail-closed.** With `ENVIRONMENT=production` the app
  refuses to boot — rather than starting "healthy" — if `SECRET_KEY` is the
  published dev placeholder or shorter than 32 chars, `DATABASE_URL` is missing
  or not PostgreSQL, `CORS_ORIGINS` is unset/empty/wildcard, or demo hydration
  is switched on. An unrecognised `ENVIRONMENT` string (e.g. a typo
  `productionn`) warns on every boot because it silently disables those guards.
  `ENABLE_DEMO_USER=true` is a deliberate, loud opt-in: it boots (with a
  `RuntimeWarning` that the published demo credential is live via
  `GET /api/auth/demo`) for the SIH26082 demo deployment, and unset stays off
  in production. See `test_production_config_guard.py`.
- **Secrets.** Do not commit `deploy.env`. Rotate any previously-exposed values
  before production use. Provide `NASA_FIRMS_MAP_KEY` via the environment.
- **Sizing.** `models/` (200+ model files) and `data/` are read from the
  repository; on a small host, expect a slow first boot.

## Health & operations

- `GET /health` — liveness
- `GET /api/data-quality` — per-table row counts and missing-value audit
- `GET /api/summary` — NCR KPIs (station count, AQI, alerts), including
  `data_mode` and `observation_age_hours` so stale data is labelled rather than
  presented as live
- Logs: `server.log` / `server_err.log` (backend), `vite_dev.log` /
  `vite_dev_err.log` (frontend); Render dashboard logs on the hosted service

## Security notes

- Never commit real API keys. `.env` is git-ignored and only ever holds local
  placeholders; copy `.env.example`/`docs/deploy.env.example` and fill in real
  values on each host. Rotate any previously-exposed values before production
  use.
- Change the development-only Postgres credentials for any shared or production
  environment.
