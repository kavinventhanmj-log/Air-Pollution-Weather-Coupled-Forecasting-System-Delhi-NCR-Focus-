"""Cold-start cache pre-warm for the control room.

Render's free tier scales the service to zero after ~15 minutes idle, and a
measured production cold wake takes ~76 s. The Control Room then fans out ~14
requests at once, several of which are heavy aggregations over the pooled Neon
Postgres (``/api/summary`` ~12 s, ``/api/atmosphere/current`` ~11 s for all 17
stations, ``/api/data-quality`` ~11 s). On a cold instance every one of those
caches is empty, so the fan-out serialises on a 0.5-CPU box, individual requests
run past the browser's timeout, and panels are left permanently empty — the
exact failure this module exists to prevent.

Every panel already reads through :func:`..services.ttl_cache.cached`. This
module simply runs the same builders, with the *same cache keys*, in the
background right after the app reports ready. The first dashboard load after a
cold start is then served from a warm cache.

Design notes:

* Opt-in via ``CONTROL_ROOM_PREWARM`` (see :mod:`..config`). It is off by
  default so local development, pytest and CI never pay for it.
* Runs off the event loop in a worker thread, one entry at a time, with a pause
  between entries so a real user request that arrives mid-pre-warm is not
  starved of the single CPU.
* Best-effort: a failing entry is logged and skipped. A pre-warm that throws
  must never take the service down.
* Nothing is fabricated or mutated — these are the same read-only builders the
  HTTP handlers call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger("aerocast.prewarm")

# Pause between entries. Long enough that an inbound request is served without
# queueing behind a 10 s aggregation, short enough that the whole sweep still
# finishes inside a typical cold-boot window.
_ENTRY_GAP_S = 1.0

# Last sweep's outcome, surfaced on /api/system. A cache warm-up is invisible by
# construction -- if it silently stops working, the only symptom is a slow first
# load, which is exactly the thing nobody is watching. Making the state
# inspectable is what turns "the pre-warm should help" into something that can
# be verified against production.
_STATUS: dict[str, object] = {"enabled": False, "state": "disabled"}


def prewarm_status() -> dict[str, object]:
    """Snapshot of the last control-room sweep. Never raises."""
    return dict(_STATUS)


def _mark(**fields: object) -> None:
    _STATUS.update(fields)


def note_scheduled() -> None:
    """Record that the sweep has been handed to a worker thread.

    Without this the state reads ``disabled`` for the fraction of a second
    between scheduling and the thread starting, which is exactly the window where
    someone checks whether the deployment has the feature on.
    """
    _mark(enabled=True, state="scheduled")


def note_disabled() -> None:
    """Record that this environment runs without the sweep (dev, test, opt-out)."""
    _STATUS.clear()
    _STATUS.update({"enabled": False, "state": "disabled"})


def _entries() -> list[tuple[str, str, float, Callable[[object], object]]]:
    """``(label, cache_key, ttl, builder)`` for every heavy read-only panel.

    The keys must stay identical to the ones the HTTP layer uses, otherwise the
    pre-warm fills an entry nobody reads.
    """
    from ..api import fire as fire_api
    from ..api import grap as grap_api
    from ..api import grid as grid_api
    from ..api import summary as summary_api

    # Imported here rather than at module scope: ``main`` imports the API
    # routers, so a top-level import would be circular. By the time the sweep
    # runs the app is fully imported.
    from ..main import _data_quality_report
    from ..services import alert_service, atmosphere_service, transport_risk_service

    return _dispersion_entries() + [
        ("atmosphere:current", "atmosphere:current", 300, lambda db: atmosphere_service.get_current_atmosphere(db)),
        ("alerts:all_stations", "alerts:all_stations", 120, lambda db: alert_service.all_station_alerts(db)),
        ("summary", "summary", 60, lambda db: summary_api._build_summary(db)),
        # Opens its own session (it is defined in main.py that way).
        ("data-quality", "data-quality", 300, lambda _db: _data_quality_report()),
        ("transport-risk:current:72", "transport-risk:current:72", 300,
         lambda db: transport_risk_service.get_current_transport_risk(db, fire_window_hours=72)),
        ("grap:current", "grap:current", 300, lambda db: grap_api._compute_grap_current(db)),
        ("fire-activity", "fire-activity", 300, lambda db: fire_api._compute_fire_activity(db)),
        # Only the default (real-only) variant is pre-warmed: it is the one the
        # map requests. The key must come from the endpoint's own helper so a
        # change there cannot silently warm an entry nobody reads.
        ("fire-hotspots", fire_api.fire_hotspots_cache_key(False), 300,
         lambda db: fire_api._compute_fire_hotspots(db, include_synthetic=False)),
        ("plume-risk", "plume-risk", 300, lambda db: fire_api._compute_plume_risk(db)),
        ("grid:forecast:24", "grid:forecast:24", 120, lambda db: grid_api._compute_grid(db, 24)),
    ]


def _dispersion_frame_hours(horizon_hours: int) -> list[int]:
    """Six-hourly frames — must match the Spatial Forecast page's cadence.

    Kept as a function of the horizon so it cannot drift from the client's
    ``FRAME_STEP_H`` the way a hard-coded list would.
    """
    return list(range(6, horizon_hours + 1, 6))


def _dispersion_entries() -> list[tuple[str, str, float, Callable[[object], object]]]:
    """Pre-warm the dispersion keys the UI actually requests, and warm them first.

    Two rules, both learned the hard way from a production measurement:

    * **Only keys the client asks for.** The solver is the most expensive read in
      the app (~10-12 s for 72 h on the production box), so warming a key nobody
      requests is the worst outcome available: it burns the one CPU during the
      cold-boot window and speeds nothing up. An earlier version warmed
      ``dispersion:72:8:all`` while the page had already moved to filtered
      ``frame_hours`` requests, so every entry it created was unreadable garbage.
      Keys now come from :func:`..api.dispersion.dispersion_cache_key`.
    * **Before the cheap reads, not after.** The whole sweep runs in background
      against a single CPU, and the first real user request lands ~76 s after the
      wake. Cheapest-first is the usual instinct, but measured cold-boot timings
      put the three dispersion solves at ~33 s of that window on their own --
      last-in-list they would still have been running when the user's request
      arrived. Dispatch is the one panel a retry cannot rescue (a cold 12 s
      serialisation either clears or blows the client timeout), so it goes first
      and the rest of the list follows.
    """
    from ..api.dispersion import DISPERSION_TTL_SECONDS, dispersion_cache_key
    from .dispersion_service import run_dispersion_forecast_service

    return [
        (
            f"dispersion:{horizon}:8:{'-'.join(str(h) for h in hours)}",
            dispersion_cache_key(horizon, 8, hours),
            DISPERSION_TTL_SECONDS,
            lambda db, horizon=horizon, hours=hours: run_dispersion_forecast_service(
                db, horizon_hours=horizon, start_hour=8, frame_hours=hours
            ),
        )
        # Cheapest first: 24 h is the horizon the page opens on, so it goes warm
        # before the 48/72 h solvers take the CPU.
        for horizon, hours in ((24, _dispersion_frame_hours(24)), (48, _dispersion_frame_hours(48)), (72, _dispersion_frame_hours(72)))
    ]


def prewarm_control_room(stop: threading.Event | None = None) -> None:
    """Populate the TTL cache with the control room's heavy read payloads.

    ``stop`` (a :class:`threading.Event`) lets the caller abandon the sweep at
    shutdown; it is checked between entries only, never mid-query.
    """
    from ..database import SessionLocal
    from .ttl_cache import cached

    started = time.monotonic()
    warmed, failed = 0, 0
    # Recorded per entry so /api/system can show how far the sweep got, not just
    # whether it finished: a sweep that dies on entry 2 and a sweep that is still
    # running at 40 s look identical from the outside otherwise.
    _mark(enabled=True, state="running", entries_warmed=0, entries_failed=0)
    for label, key, ttl, builder in _entries():
        if stop is not None and stop.is_set():
            _mark(state="cancelled", seconds=round(time.monotonic() - started, 1))
            logger.info("Pre-warm cancelled after %d entries", warmed)
            return
        try:
            with SessionLocal() as db:
                cached(key, ttl, lambda b=builder: b(db))
            warmed += 1
            logger.info("Pre-warmed %s", label)
        except Exception:
            failed += 1
            logger.exception("Pre-warm failed for %s (continuing)", label)
        _mark(entries_warmed=warmed, entries_failed=failed, last_entry=label)
        time.sleep(_ENTRY_GAP_S)
    _mark(state="complete", entries_warmed=warmed, entries_failed=failed,
          seconds=round(time.monotonic() - started, 1))
    logger.info(
        "Pre-warm complete: %d entries warm, %d failed, %.1fs", warmed, failed, time.monotonic() - started
    )
