# Deployment

Two supported paths: **Docker Compose** (self-contained, recommended) and
**bare metal** (system Python + npm). Compose is the documented production path;
the backend runs **Alembic migrations automatically at container start** against
PostgreSQL.

## Docker Compose (recommended)

### 1. Configure

```bash
cp .env.example .env                 # local defaults (optional real FIRMS key)
cp docs/deploy.env.example deploy.env
```

`docker-compose.yml` reads `NASA_FIRMS_MAP_KEY` from your shell/environment. For
production, prefer passing a private env file (see "Production tips" below).

### 2. Build and start

```bash
docker compose up --build -d
```

Services:

- `db` — PostgreSQL 16 (healthchecked), volume `pgdata`
- `backend` — FastAPI on `:8000`; `ml/`, `models/`, `data/` mounted read-only;
  runs as an **unprivileged** user and performs migrations at boot
- `frontend` — nginx on `:5173` (host → 80) serving the SPA and proxying `/api`

All services are `restart: unless-stopped`; `backend` only starts after `db` is
healthy, and `frontend` only after `backend` reports healthy.

### 3. Migration flow

On container start, `app.main.lifespan` calls `run_migrations()` which runs
`alembic upgrade head` against `DATABASE_URL` (PostgreSQL in Compose) using the
migration scripts baked into the image (`alembic/versions/`). A fresh deployment
builds the full 7-table schema; an existing deployment applies only pending
revisions. SQLite (bare-metal dev) keeps `create_all` + additive
`apply_migrations()` as a lightweight fallback and skips Alembic.

To run migrations on demand (e.g. before scaling replicas):

```bash
docker compose run --rm backend python -m alembic upgrade head
docker compose run --rm backend python -m alembic current
```

### 4. Live data refresh

Compose sets `LIVE_REFRESH_ENABLED=true` and
`LIVE_REFRESH_INTERVAL_HOURS=3`, so the scheduler pulls weather, fire and
pollution in the background. For a one-shot/manual refresh (e.g. after a gap or
in dry-run to preview):

```bash
docker compose run --rm backend python -m scripts.refresh_once --dry-run   # preview
docker compose run --rm backend python -m scripts.refresh_once              # commit
```

Provide a `NASA_FIRMS_MAP_KEY` for reliable live fire ingestion.

### 5. First-boot verification

```bash
curl -fsS http://localhost:8000/health                 # {"status":"healthy",...}
curl -fsS http://localhost:8000/api/data-quality       # per-table row counts + gaps
curl -fsS http://localhost:8000/api/summary            # NCR KPIs
curl -fsS http://localhost:8000/api/stations           # 5 seeded stations
```

Open the dashboard at `http://localhost:5173`. In `docker compose logs -f
backend`, look for `Alembic migrations applied at startup` (Postgres) or
`Seeded 5 default Delhi NCR stations`.

### 6. `pgdata` backup / restore

Backup the named volume (take a filesystem-level snapshot for a consistent run):

```bash
# dump to a file via the container
docker compose exec -T db pg_dump -U aerocast -d aerocast_ncr > backup_$(date +%Y%m%d).sql
# or snapshot the volume
docker run --rm -v aerocast-ncr_pgdata:/data -v "$PWD":/backup alpine \
  tar czf /backup/pgdata_$(date +%Y%m%d).tar.gz -C /data .
```

Restore:

```bash
docker compose down
docker run --rm -v aerocast-ncr_pgdata:/data -v "$PWD":/backup alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/pgdata_YYYYMMDD.tar.gz -C /data"
docker compose up -d
# or from a plain SQL dump:
cat backup_YYYYMMDD.sql | docker compose exec -T db psql -U aerocast -d aerocast_ncr
```

Stop all services before a volume-level restore. For a fresh SQL dump restore,
drop/recreate the DB first to avoid conflicts.

### Production tips

- **Reverse proxy / HTTPS.** Keep `:8000`/`:5173` on the host private and put a
  TLS-terminating reverse proxy (Caddy, nginx, Traefik) in front; set
  `CORS_ORIGINS` to the real public origin (e.g. `https://forecast.example`).
- **Production startup is fail-closed.** With `ENVIRONMENT=production` the app
  refuses to boot — rather than starting "healthy" — if `SECRET_KEY` is the
  published dev placeholder or shorter than 32 chars, `DATABASE_URL` is missing
  or not PostgreSQL, `CORS_ORIGINS` is unset/empty/wildcard, or the demo
  account / demo hydration are switched on. An unrecognised `ENVIRONMENT`
  string (e.g. a typo `productionn`) warns on every boot because it silently
  disables those guards. See `test_production_config_guard.py`.
- **Secrets.** Do not commit `deploy.env`; rotate the development-only Postgres
  password before exposure. Provide `NASA_FIRMS_MAP_KEY` via the env file.
- **Sizing.** `models/` (200+ model files) and `data/` are mounted read-only, so
  keep them on the host; bump container resources if needed.

## Bare metal

```bash
python -m pip install -e ".[dev]"
cp .env.example .env

# backend (SQLite dev default; use Alembic for Postgres below)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# frontend (separate terminal)
cd frontend && npm ci && npm run dev

# Postgres with Alembic (optional)
export DATABASE_URL=postgresql://aerocast:pass@localhost:5432/aerocast_ncr
python -m alembic upgrade head
```

## Environment reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///./aerocast_ncr.db` | SQLAlchemy connection string; Postgres triggers Alembic migrations |
| `NASA_FIRMS_MAP_KEY` | *(empty)* | Optional FIRMS API key for live fire data |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed browser origins |
| `ENVIRONMENT` | `development` | App runtime environment label. Must be exactly `production` for the fail-closed startup guard; anything else (incl. misspellings) boots with the guard off and emits a warning |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LIVE_REFRESH_ENABLED` | `false` | Whether the refresh scheduler runs |
| `LIVE_REFRESH_INTERVAL_HOURS` | `3` | Scheduler cadence |
| `DEMO_HYDRATE_EMPTY_DB` | `false` | Re-stamp the bundled archive into the recent window when the database has no fresh readings |
| `CONTROL_ROOM_PREWARM` | unset (= on in production) | Fill the TTL cache with the heavy control-room reads in the background after startup. Worth it on scale-to-zero hosts (Render free tier), where a cold wake otherwise leaves the first dashboard load queueing behind cold ~12 s aggregations. Never blocks readiness; failures are logged and skipped. Unset, the sweep follows the environment: on when `ENVIRONMENT=production`, off for local dev/pytest/CI. Set `false` to opt out on a production host. Check it took effect with `GET /api/system` → `prewarm`. |

See `docs/deploy.env.example` for a production template.

## Health & operations

- `GET /health` — liveness (also the container HEALTHCHECK).
- `GET /api/data-quality` — per-table row counts and missing-value audit.
- `GET /api/summary` — NCR KPIs (station count, AQI, alerts).
- Logs: `docker compose logs -f backend`; bare-metal `server.log` /
  `server_err.log` (backend), `vite_dev.log` / `vite_dev_err.log` (frontend).

## Security notes

- Never commit real API keys. `.env` is git-ignored and only ever holds local
  placeholders; copy `.env.example`/`docs/deploy.env.example` and fill in real
  values on each host. Rotate any previously-exposed values before production
  use.
- Postgres credentials in `docker-compose.yml` are development-only defaults;
  change them for any shared/`production` environment.
- The backend container runs as a non-root user; keep `ml/`, `models/`, `data/`
  mounts read-only.
