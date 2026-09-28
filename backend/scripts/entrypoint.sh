#!/bin/sh
# Container entrypoint: apply schema migrations, then hand off to uvicorn.
#
# Migrations must run before the app serves traffic, otherwise a fresh database
# boots against a schema that lacks re_stamped / forecast_runs and every request
# touching those columns fails at runtime rather than at deploy time.
#
# `alembic/env.py` resolves the metadata from `app.*`, which inside the image
# lives at /app/app (the image copies backend/app into /app/app), so no
# PYTHONPATH juggling is needed here - but it is set anyway to stay explicit.
#
# SKIP_MIGRATIONS=1 opts out for local compose runs that manage the schema
# separately, or when a migration is known-bad and you need the old behaviour to
# diagnose it.
set -e

cd /app

if [ "${SKIP_MIGRATIONS:-0}" = "1" ]; then
    echo "[entrypoint] SKIP_MIGRATIONS=1 - not running migrations"
else
    echo "[entrypoint] applying migrations..."
    # migrate_safely.py rather than a bare `alembic upgrade head`: it stamps
    # instead of replaying when pointed at a legacy database that has tables but
    # no alembic_version (built by the old create_all() path). A bare upgrade
    # would try to recreate those tables and fail. On a fresh database it is
    # equivalent to `alembic upgrade head`.
    if ! python backend/scripts/migrate_safely.py; then
        # Refuse to serve: booting against a half-migrated schema produces
        # confusing 500s that look like application bugs. Crash loudly instead.
        echo "[entrypoint] FATAL: migrations failed; refusing to start." >&2
        exit 1
    fi
    echo "[entrypoint] migrations applied"
fi

# exec so uvicorn becomes PID 1 and receives Docker's SIGTERM directly.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
