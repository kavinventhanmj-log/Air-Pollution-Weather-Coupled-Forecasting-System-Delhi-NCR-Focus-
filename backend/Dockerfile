FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# runtime dependencies only (models/ml/data arrive via docker-compose volumes)
COPY backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# application + Alembic migration scripts (migrations run at container start)
COPY backend/app ./app
COPY backend/scripts/entrypoint.sh ./entrypoint.sh
# migrate_safely.py is invoked by entrypoint.sh to decide between
# `alembic upgrade head` and `alembic stamp head` (legacy databases have tables
# but no alembic_version). It must be present at backend/scripts/ to match the
# path entrypoint.sh calls.
COPY backend/scripts/migrate_safely.py ./backend/scripts/migrate_safely.py
COPY alembic.ini .
COPY alembic ./alembic

# run as an unprivileged user
RUN useradd --create-home --uid 10001 aerocast \
    && chown -R aerocast:aerocast /app \
    && chmod +x /app/entrypoint.sh
USER aerocast

EXPOSE 8000

# Liveness only, never `/health`: that probe round-trips Postgres, which can
# exceed this 5s timeout while a database is cold and would mark a perfectly
# healthy container as unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# The entrypoint applies migrations first, then execs uvicorn. `exec` matters:
# it makes uvicorn PID 1 so Docker signals reach it directly instead of being
# swallowed by a shell wrapper.
ENTRYPOINT ["/app/entrypoint.sh"]
