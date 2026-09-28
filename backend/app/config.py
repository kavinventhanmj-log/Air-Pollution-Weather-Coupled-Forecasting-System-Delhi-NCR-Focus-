import pathlib
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings

_ENV_CANDIDATES = [
    pathlib.Path(__file__).resolve().parents[2] / ".env",
    pathlib.Path(__file__).resolve().parents[1] / ".env",
    pathlib.Path.cwd() / ".env",
]
_ENV_FILE = next((str(p) for p in _ENV_CANDIDATES if p.exists()), ".env")

#: The development default JWT signing key. Published in this repository, so it
#: is only ever a placeholder: production must override it, and
#: ``Settings.validate_production_config`` refuses to boot while it is in use.
DEV_SECRET_KEY = "aerocast-dev-secret-change-me-in-production"

#: The published demo password, also a development-only placeholder.
DEV_DEMO_PASSWORD = "AeroCast@2026"


class Settings(BaseSettings):
    database_url: str = "sqlite:///./aerocast_ncr.db"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    environment: str = "development"
    log_level: str = "INFO"
    nasa_firms_map_key: str = ""
    live_refresh_enabled: bool = False
    live_refresh_interval_hours: int = 3
    # Optional demo self-hydration: when true, the app loads the bundled coupled
    # dataset + model metrics + alerts and re-stamps recent observations into the
    # last 24h whenever pollution or weather has no reading in that window. This
    # makes a brand-new or stale database render a live-looking demo with no
    # manual steps (see backend/app/services/demo_hydration.py).
    demo_hydrate_empty_db: bool = False
    # Cold-start cache pre-warm (Render free tier). When enabled, a background
    # task populates the TTL cache with the control room's heavy read-only
    # payloads right after startup, so the first dashboard load after a ~76 s
    # cold wake is served from cache instead of serialising ~12 s aggregations on
    # 0.5 CPU.
    #
    # Tri-state on purpose. `None` means "follow the default for this
    # environment" (on in production, off everywhere else), and an explicit
    # true/false always wins. It used to be a plain `bool = False`, which meant
    # the feature was silently off in production: Render does not push newly
    # added `render.yaml` env vars to an already-created service, so the sweep
    # never ran on the live deployment and `/api/system` reported
    # `{"enabled": false}` for the whole life of the release. A cold-start fix
    # that quietly depends on a manual dashboard toggle is not a fix.
    control_room_prewarm: bool | None = None
    # Public URL of the deployed frontend (Vercel). When set, `GET /` on the
    # API redirects the browser there instead of answering a bare 404.
    frontend_url: str = ""
    # data.gov.in / CPCB "Real time Air Quality Index from various locations"
    data_gov_api_key: str = ""
    data_gov_api_url: str = "https://api.data.gov.in"
    data_gov_resource_id: str = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    data_gov_ncr_cities: str = "Delhi,Gurugram,Noida,Ghaziabad,Faridabad"
    data_gov_timeout: int = 30

    # --- Chemical transport model (CTM) engines (SIH26082 R6) ---
    # Physical copy of NOAA HYSPLIT (e.g. C:\hysplit4). When set and a real
    # `exec/hycs_std(.exe)` plus GDAS/EDAS met files exist, dispersion runs use
    # genuine HYSPLIT output; otherwise the analytic surrogate stays active.
    hysplit_home: str = ""
    hysplit_met_dir: str = ""
    # Directory holding genuine WRF-Chem `wrfout_d01_*.nc` output from an
    # external run to be absorbed as the CTM surface (never fabricated).
    wrf_output_dir: str = ""

    # --- IMD official weather API (SIH26082 R9) ---
    # api.imd.gov.in requires registration + key/IP whitelisting (returns 401
    # otherwise). When set, `/api/imd/forecast` fetches *real* IMD outputs;
    # otherwise it reports honest reasons and the Open-Meteo path stays.
    imd_api_key: str = ""
    # Delhi/Safdarjung (the anchor IMD city for the NCR domain).
    imd_station_id: str = "42182"
    imd_api_base: str = "https://api.imd.gov.in/api/v1"

    # --- Authentication (SIH26082 UI login layer) ---
    # The default below is a development placeholder. It is public in the
    # repository, so it must never sign a token in production --
    # ``validate_production_config`` refuses to boot if it is still in use.
    secret_key: str = DEV_SECRET_KEY
    access_token_expire_minutes: int = 480  # 8 hours — one operational shift
    # Demo account seeded at startup; override via DEMO_USER_* env vars.
    demo_user_email: str = "analyst@aerocast.in"
    demo_user_name: str = "Demo Analyst"
    demo_user_role: str = "Analyst"
    # Also a published default. Production must supply its own value, and
    # ``enable_demo_user`` must be false, so the seeded account cannot
    # authenticate with a credential that is published in the source.
    demo_user_password: str = DEV_DEMO_PASSWORD
    # Whether startup seeds the demo account at all. Default follows the
    # environment: off in production, on for local dev/CI. An explicit value
    # always wins, so a staging deployment can opt in deliberately.
    enable_demo_user: bool | None = None

    model_config = {"env_file": _ENV_FILE}

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == "production"

    @property
    def demo_user_enabled(self) -> bool:
        """Whether the demo account is created at startup.

        An explicit ``ENABLE_DEMO_USER`` always wins. Unset, it follows the
        environment: production never seeds it, because both the address and the
        default password are published in this repository, so anyone could log
        in to a live deployment. Local dev and CI keep it so the login page
        remains usable out of the box.
        """
        if self.enable_demo_user is not None:
            return self.enable_demo_user
        return not self.is_production

    @property
    def demo_hydration_enabled(self) -> bool:
        """Whether startup may synthesise observations to fill data gaps.

        An explicit ``DEMO_HYDRATE_EMPTY_DB`` wins for every non-production
        environment. In production it is forced off: re-stamping an archive
        into the recent window makes synthetic rows look like fresh sensor
        measurements, which is the fabrication the provenance columns
        (``re_stamped``, ``data_source``) exist to prevent.
        """
        if self.is_production:
            return False
        return self.demo_hydrate_empty_db

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        """Fail closed on configurations that are unsafe in production.

        Each of these is a silent-failure mode found in the release audit:
        a known JWT signing key lets anyone mint an admin token; a missing
        ``DATABASE_URL`` falls back to an empty local SQLite file while the
        service still reports healthy; the published demo password would seed
        a working login; and demo hydration would manufacture observations.

        Refusing to start is the only reliable response -- every one of these
        otherwise starts "successfully" and fails open at runtime.
        """
        if not self.is_production:
            return self

        problems: list[str] = []

        if self.secret_key == DEV_SECRET_KEY or self.secret_key.strip() == "":
            problems.append(
                "SECRET_KEY is unset or still the published development default. "
                "Set SECRET_KEY to a long random value in the deployment "
                "provider's secret store."
            )
        elif len(self.secret_key) < 32:
            problems.append(
                f"SECRET_KEY is only {len(self.secret_key)} characters; "
                "at least 32 are required to sign HS256 tokens in production."
            )

        if not self.database_url or self.database_url.startswith("sqlite"):
            problems.append(
                "DATABASE_URL is unset or points at SQLite. Production must "
                "use PostgreSQL; a SQLite fallback serves an empty database "
                "while reporting healthy."
            )

        if self.demo_user_enabled:
            problems.append(
                "ENABLE_DEMO_USER is on in production, and the demo address and "
                "password are published in this repository. Set "
                "ENABLE_DEMO_USER=false."
            )

        if self.demo_hydrate_empty_db:
            problems.append(
                "DEMO_HYDRATE_EMPTY_DB is on in production, which synthesises "
                "observations and presents them as recent measurements. Set "
                "DEMO_HYDRATE_EMPTY_DB=false."
            )

        if problems:
            raise ValueError(
                "Refusing to start in production with an unsafe configuration:"
                + "".join(f"\n  - {p}" for p in problems)
            )
        return self

    @property
    def prewarm_enabled(self) -> bool:
        """Whether the cold-start cache sweep should run.

        An explicit ``CONTROL_ROOM_PREWARM`` always wins. Unset, the sweep
        follows the environment: production is exactly where a 76 s cold wake
        makes it worth ~75 s of off-request-path CPU, and local dev / pytest /
        CI are exactly where nobody wants it. The sweep is best-effort,
        cancellable and never blocks readiness, so the production default is
        safe; ``CONTROL_ROOM_PREWARM=false`` turns it off if a deployment would
        rather not pay for it.
        """
        if self.control_room_prewarm is not None:
            return self.control_room_prewarm
        return self.environment.strip().lower() == "production"

@lru_cache
def get_settings() -> Settings:
    return Settings()
