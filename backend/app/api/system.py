from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter

from ..config import get_settings
from ..database import database_reachable

router = APIRouter()


@router.get("/system")
def system_status():
    """Honest engine + integration status for the architecture page.

    Reports which real third-party systems are wired in via environment
    configuration. Nothing is fabricated: an engine shows as ``available`` only
    when the corresponding setting was actually provided by the deployment.
    """
    settings = get_settings()

    def _dir_exists(raw: str) -> bool:
        return bool(raw and Path(raw).expanduser().is_dir())

    def _file_exists(raw: str) -> bool:
        return bool(raw and Path(raw).expanduser().exists())

    # Chemical-transport engines (SIH26082 R6). A HYSPLIT executable only counts
    # as usable when it physically exists alongside its meteorological inputs.
    # Only booleans and leaf names are reported: absolute installation paths
    # leak server filesystem layout to anonymous callers.
    hysplit_name = ""
    hysplit_usable = False
    if settings.hysplit_home and settings.hysplit_met_dir:
        for candidate in ("hycs_std", "hycs_std.exe", "hycs_std_32766"):
            p = Path(settings.hysplit_home).expanduser() / "exec" / candidate
            if p.exists():
                hysplit_name = candidate
                hysplit_usable = _dir_exists(settings.hysplit_met_dir)
                break
        if not hysplit_name and _file_exists(settings.hysplit_home):
            hysplit_name = Path(settings.hysplit_home).name

    # WRFs chem/CTM surface: only real `wrfout_d01_*.nc` output is ever absorbed.
    wrf_name = ""
    if _dir_exists(settings.wrf_output_dir):
        matches = sorted(Path(settings.wrf_output_dir).expanduser().glob("wrfout_d01_*.nc"))
        if matches:
            wrf_name = matches[-1].name

    # Official IMD weather API (SIH26082 R9): key must have been provided.
    imd_configured = bool(settings.imd_api_key)

    # CPCB / data.gov.in live AQ feed.
    cpcb_configured = bool(settings.data_gov_api_key)

    # NASA FIRMS fire archive.
    firms_configured = bool(settings.nasa_firms_map_key)

    db_status = "connected" if database_reachable() else "disconnected"

    # The schema the running code expects, taken from the migration chain on
    # disk. This is honest to ask for: a boot that runs ``run_migrations`` has
    # already upgraded the database to this head (and refuses to serve when the
    # stamped revision disagrees), so "head == running code" is the verified
    # invariant rather than a hardcoded version string that can drift.
    from ..database import _alembic_heads

    try:
        heads = sorted(_alembic_heads())
        schema_version = heads[0] if heads else "unmanaged"
    except Exception:
        schema_version = "unknown"

    # Cold-start cache pre-warm. Reported because a cache warm-up is invisible
    # from the outside: if it stops working, the only symptom is a slow first
    # load, which is exactly what nobody is watching. The state also tells you
    # whether the instance you are talking to has finished warming yet, and
    # `setting`/`default_applied` distinguish "on because production" from
    # "on because somebody set it".
    from ..services.prewarm import prewarm_status

    prewarm = prewarm_status()
    prewarm["setting"] = settings.control_room_prewarm
    prewarm["default_applied"] = settings.control_room_prewarm is None

    engines = {
        "weather_forecast": {
            "source": "Open-Meteo",
            "status": "active",
            "note": "Ambient plus 900/1000 hPa vertical profile for every NCR station. Used by atmosphere, inversion and feature builder.",
        },
        "ctm_hysplit": {
            "source": "NOAA HYSPLIT",
            "status": "usable" if hysplit_usable else "surrogate",
            "detail": hysplit_name or None,
            "note": (
                f"Executable {hysplit_name} with GDAS/EDAS met found -> genuine dispersion runs."
                if hysplit_usable
                else "No HYSPLIT install supplied; analytic dispersion surrogate is active (feature-flagged in the UI)."
            ),
        },
        "ctm_wrf_chem": {
            "source": "WRF-Chem",
            "status": "available" if wrf_name else "gated",
            "detail": wrf_name or None,
            "note": (
                f"Absorbed real surface from {wrf_name}."
                if wrf_name
                else "Set WRF_OUTPUT_DIR to a folder of wrfout_d01_*.nc from an external run to absorb the CTM surface here."
            ),
        },
        "imd": {
            "source": "IMD api.imd.gov.in",
            "status": "configured" if imd_configured else "not-configured",
            "note": (
                "Official IMD forecast key present; real IMD outputs are fetched."
                if imd_configured
                else "No IMD key supplied (the API returns 401 without registration/IP whitelisting); Open-Meteo path stays active. Honest reason is surfaced on /api/imd/forecast."
            ),
        },
        "cpcb": {
            "source": "CPCB / data.gov.in",
            "status": "configured" if cpcb_configured else "archive",
            "note": (
                "Live CPCB AQ feed configured."
                if cpcb_configured
                else "Live feed key not supplied; persisted CPCB-format archive drives the dashboard."
            ),
        },
        "firms": {
            "source": "NASA FIRMS",
            "status": "configured" if firms_configured else "archive",
            "note": (
                "FIRMS map key configured; live VIIRS fire pull."
                if firms_configured
                else "No FIRMS key supplied; bundled VIIRS fire archive (58k+ detections) drives plume risk."
            ),
        },
    }

    return {
        "generated_at": datetime.now(UTC).replace(tzinfo=None),
        "service": "AeroCast-NCR",
        "database": db_status,
        "run_mode": {
            "environment": settings.environment,
            "live_refresh_enabled": getattr(settings, "live_refresh_enabled", False),
            "live_refresh_interval_hours": getattr(settings, "live_refresh_interval_hours", 3),
            "demo_hydrate_empty_db": getattr(settings, "demo_hydrate_empty_db", False),
            "explanation": "live_refresh_enabled re-pulls upstream feeds on a schedule; demo_hydrate_empty_db re-stamps the historical archive into the recent window for a fresh database.",
        },
        "prewarm": prewarm,
        "engines": engines,
        "schema_version": schema_version,
    }
