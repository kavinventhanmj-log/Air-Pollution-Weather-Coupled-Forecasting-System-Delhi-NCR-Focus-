"""Optional demo-data hydration for the running app.

When ``DEMO_HYDRATE_EMPTY_DB=true`` the backend checks shortly after startup
whether either observation stream (pollution or weather) has any rows within
the last 24 hours and, if not, loads the bundled coupled dataset + FIRMS fire
archive + model metrics + seed alerts and re-stamps the newest observations
into the last 24 hourly slots. This renders the dashboard as a live-looking
demo on a brand-new database and self-heals stale CPCB feeds.

On every boot (observations fresh or not) it then idempotently ensures the
demo auxiliary payloads — model metrics, seed alerts, the recent-fire archive
and persisted per-station 72h forecasts — so a database with live observations
still renders every demo panel (see :func:`_ensure_auxiliary_demo_data`).

Deliberately opt-in: the live-refresh ingestion path is never touched, and the
default (flag unset) keeps the app behaviour identical to before.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..database import SessionLocal
from ..utils.helpers import haversine_distance

logger = logging.getLogger("aerocast.hydration")

RECENCY_WINDOW_HOURS = 24
BOOT_GRACE_SECONDS = 45


def _demo_observation_is_stale(db) -> bool:
    """True when pollution or weather has no reading within the last 24h."""
    from ..models.db_models import PollutionReading, WeatherReading

    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=RECENCY_WINDOW_HOURS)
    recent_poll = (
        db.query(PollutionReading.id)
        .filter(PollutionReading.timestamp >= cutoff)
        .first()
    )
    recent_wx = (
        db.query(WeatherReading.id)
        .filter(WeatherReading.timestamp >= cutoff)
        .first()
    )
    return not (recent_poll and recent_wx)


def _load_demo_data() -> None:
    from backend.app.models.db_models import PollutionReading
    from backend.scripts import bootstrap_recent
    from backend.scripts import load_data as demo

    csv_path = Path(demo.DEFAULT_CSV)

    db = SessionLocal()
    try:
        counts = demo.load_coupled_data(db, csv_path, chunksize=25_000)
    finally:
        db.close()
    logger.info(
        "demo coupled load complete: pollution=%d weather=%d",
        counts["pollution"],
        counts["weather"],
    )

    # Re-stamp NOW so the dashboard goes live-looking before the slower fire
    # archive load finishes (firms_fires.csv can be hundreds of MB on a cold
    # cache; a fresh/edge database should still render a live scenario fast).
    anchor_minute = 0
    db = SessionLocal()
    try:
        latest = (
            db.query(PollutionReading.timestamp)
            .order_by(PollutionReading.timestamp.desc())
            .first()
        )
        if latest is not None and latest[0] is not None:
            anchor_minute = latest[0].minute
        n_poll = bootstrap_recent.bootstrap_pollution(db, anchor_minute)
        n_wx = bootstrap_recent.bootstrap_weather(db, anchor_minute)
        logger.info("demo re-stamp into last 24h: pollution=%d weather=%d", n_poll, n_wx)
        n_pf, n_wf = _fill_missing_stations(db, anchor_minute)
        logger.info("demo neighbor-fill: pollution=%d weather=%d", n_pf, n_wf)
    finally:
        db.close()


def _fill_missing_stations(db, anchor_minute: int) -> tuple[int, int]:
    """Give every station a last-24h pollution+weather picture.

    The bundled CPCB dataset covers only the five anchor monitoring sites, so a
    brand-new database would otherwise leave the other stations' cards on
    "Awaiting live data". This demo proxy copies the *nearest data-bearing
    station's* latest readings onto the missing stations' hourly slots (the
    same re-stamp used by :mod:`bootstrap_recent`). It is marked with
    ``data_source='demo_proxy'`` so it can never be mistaken for a genuine
    feed reading, and it only ever back-fills a station, never overwrites.
    """
    from backend.app.models.db_models import PollutionReading, Station, WeatherReading
    from backend.scripts import bootstrap_recent

    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=RECENCY_WINDOW_HOURS)
    slots = bootstrap_recent._aligned_slots(RECENCY_WINDOW_HOURS, anchor_minute)

    stations = db.query(Station).all()
    by_id = {s.id: s for s in stations}

    def _anchor_ids(model) -> set[int]:
        rows = (
            db.query(model.station_id)
            .filter(model.timestamp >= cutoff)
            .distinct()
            .all()
        )
        return {r[0] for r in rows}

    def _existing(model, sid: int) -> set:
        rows = (
            db.query(model.timestamp)
            .filter(model.station_id == sid, model.timestamp >= cutoff)
            .all()
        )
        return {r[0] for r in rows}

    def _nearest(id_pool: set[int], target) -> int | None:
        best, best_d = None, float("inf")
        for sid in id_pool:
            s = by_id[sid]
            d = haversine_distance(target.latitude, target.longitude, s.latitude, s.longitude)
            if d < best_d:
                best, best_d = sid, d
        return best

    def _latest(model, sid: int):
        return (
            db.query(model)
            .filter(model.station_id == sid)
            .order_by(model.timestamp.desc())
            .first()
        )

    poll_anchors = _anchor_ids(PollutionReading)
    wx_anchors = _anchor_ids(WeatherReading)

    poll_rows, wx_rows = [], []
    for s in stations:
        if s.id not in poll_anchors:
            src = _nearest(poll_anchors, s)
            latest = _latest(PollutionReading, src) if src is not None else None
            if latest is not None:
                existing = _existing(PollutionReading, s.id)
                for ts in slots:
                    if ts in existing:
                        continue
                    poll_rows.append(PollutionReading(
                        station_id=s.id, timestamp=ts,
                        pm25=latest.pm25, pm10=latest.pm10, o3=latest.o3,
                        no2=latest.no2, so2=latest.so2, co=latest.co,
                        aqi=latest.aqi, data_source="demo_proxy",
                    ))
                    existing.add(ts)

        if s.id not in wx_anchors:
            wsrc = _nearest(wx_anchors, s)
            wlatest = _latest(WeatherReading, wsrc) if wsrc is not None else None
            if wlatest is not None:
                wexisting = _existing(WeatherReading, s.id)
                for ts in slots:
                    if ts in wexisting:
                        continue
                    wx_rows.append(WeatherReading(
                        station_id=s.id, timestamp=ts,
                        temperature=wlatest.temperature,
                        humidity=wlatest.humidity,
                        pressure_msl=wlatest.pressure_msl,
                        surface_pressure=wlatest.surface_pressure,
                        wind_speed=wlatest.wind_speed,
                        wind_direction=wlatest.wind_direction,
                        precipitation=wlatest.precipitation,
                        cloud_cover=wlatest.cloud_cover,
                        pbl_height=wlatest.pbl_height,
                    ))
                    wexisting.add(ts)

    if poll_rows:
        db.bulk_save_objects(poll_rows)
    if wx_rows:
        db.bulk_save_objects(wx_rows)
    if poll_rows or wx_rows:
        db.commit()
    return len(poll_rows), len(wx_rows)


def _ensure_auxiliary_demo_data(db) -> dict:
    """Idempotently ensure metrics, seed alerts, fires and persisted forecasts.

    Runs on *every* boot while ``DEMO_HYDRATE_EMPTY_DB`` is on, independent of
    how fresh the observations are. A database that hydrated once (or is fed by
    the live-refresh ingest) has recent readings but none of the demo artifacts,
    so panels stay on "no metrics / no persisted forecast / seed alerts are
    missing". Every loader here is idempotent (keyed upserts/skip-if-present),
    and each one is tolerated on failure so a transient model/file problem can
    never break the boot.
    """
    from backend.app.models.db_models import FireReading, Forecast, Station
    from backend.app.services.forecast_service import generate_forecast
    from backend.scripts import load_data as demo

    fire_csv = Path(demo.MODELS_DIR).parent / "data" / "fire" / "firms_fires.csv"
    metrics_path = demo.MODELS_DIR / "metrics.json"

    result = {}

    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=RECENCY_WINDOW_HOURS)
    recent_fire = (
        db.query(FireReading.id)
        .filter(FireReading.acq_date >= cutoff)
        .first()
        is not None
    )
    if recent_fire:
        result["fire"] = 0
    else:
        try:
            result["fire"] = demo.load_fire_data(db, fire_csv, chunksize=50_000)
        except Exception:
            result["fire"] = -1
            logger.exception("demo fire load failed (continuing)")

    try:
        result["metrics"] = demo.load_metrics(db, metrics_path)
    except Exception:
        result["metrics"] = -1
        logger.exception("demo metrics load failed (continuing)")
    try:
        result["alerts"] = demo.load_alerts_seed(db)
    except Exception:
        result["alerts"] = -1
        logger.exception("demo alerts seed failed (continuing)")

    stations = db.query(Station).order_by(Station.id).all()
    have = {r[0] for r in db.query(Forecast.station_id).distinct().all()}
    result["forecasts"] = 0
    for s in stations:
        if s.id in have:
            continue
        try:
            generate_forecast(db, s.id)
            have.add(s.id)
            result["forecasts"] += 1
        except Exception:
            logger.exception("demo forecast generation failed for station %s (continuing)", s.name)
    return result


async def hydrate_demo_if_empty(stop: asyncio.Event) -> None:
    """Load demo data once when observations are missing or stale (background).

    The load is deferred by ``BOOT_GRACE_SECONDS`` so it never competes with
    Render's boot/health-check window on a 512 MB free instance — the service
    comes up hot and responsive first, then hydrates quietly with chunked,
    batched inserts (see :mod:`backend.scripts.load_data`). If it were allowed
    to spike memory during startup it would OOM-kill the container (the crash
    loop that made the deployed demo intermittently 502).
    """
    del stop
    try:
        # Belt-and-braces on top of config.demo_hydration_enabled: if this ever
        # runs in production the process is misconfigured and must not start
        # synthesising rows. The config validator already refuses to boot with
        # DEMO_HYDRATE_EMPTY_DB=true+production; this guard covers a future
        # caller that forgets to check the setting first.
        from ..config import get_settings

        if get_settings().is_production:
            logger.warning(
                "demo hydration requested in production; refusing to synthesise observations"
            )
            return

        await asyncio.sleep(BOOT_GRACE_SECONDS)

        db = SessionLocal()
        try:
            stale = _demo_observation_is_stale(db)
        finally:
            db.close()

        if stale:
            logger.info("demo hydration: recent observations absent, loading bundled dataset ...")
            await asyncio.to_thread(_load_demo_data)
            logger.info("demo hydration: observations loaded")

        # Always (re-)ensure the auxiliary demo payloads and persisted
        # per-station 72h forecasts — idempotent, so a database with live
        # observations still gains the metrics/alerts/fire archive/forecasts
        # that the demo panels render.
        db = SessionLocal()
        try:
            n = await asyncio.to_thread(_ensure_auxiliary_demo_data, db)
            logger.info("demo aux ensure complete: %s", n)
        finally:
            db.close()

        from .ttl_cache import invalidate_all

        invalidate_all()
        logger.info("demo hydration complete")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("demo hydration failed")
