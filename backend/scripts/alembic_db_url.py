"""Resolve the database URL Alembic should migrate, with no silent fall-through.

Why this module exists
----------------------
``alembic -x db_url=...`` does **not** populate the ini-file options, so
``config.get_main_option("db_url")`` never sees it. The documented way to read
``-x`` is ``context.get_x_argument(as_dictionary=True)`` (which reads
``Config.cmd_opts.x``). The previous ``env.py`` called ``get_main_option`` and
therefore ignored the override entirely, silently falling through to
``DATABASE_URL`` -- which pointed at a live database at the time. A mistargeted
``alembic upgrade head`` then ran against it.

To make that failure mode impossible, resolution here is *explicit-aware*: once a
``db_url`` has been supplied on the command line it is authoritative. If it is
blank or unparseable the helper raises :class:`DbUrlResolutionError` instead of
quietly migrating a different database.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

#: Command-line / ini key that overrides the migration target.
DB_URL_KEY = "db_url"


class DbUrlResolutionError(RuntimeError):
    """An explicit ``db_url`` was unusable. Never fall back to another database."""


def parse_x_arguments(raw: Mapping[str, str] | Sequence[str] | None) -> dict[str, str]:
    """Parse ``-x`` values the same way Alembic does.

    Accepts either the list Alembic stores in ``Config.cmd_opts.x`` (``["a=1"]``)
    or an already-parsed mapping. Items without ``=`` map to an empty string.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(k).strip(): str(v) for k, v in raw.items()}
    parsed: dict[str, str] = {}
    for item in raw:
        key, sep, value = str(item).partition("=")
        parsed[key.strip()] = value if sep else ""
    return parsed


def _safe_target(url: str) -> str:
    """Render a URL for logging with any password masked."""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except ArgumentError:
        return "<unparseable>"


def _validated(
    value: str | None,
    *,
    origin: str,
    explicit: bool,
    log: Callable[[str], None],
) -> str:
    if value is None or not str(value).strip():
        if explicit:
            raise DbUrlResolutionError(
                f"an explicit database URL was supplied via {origin} but is empty; "
                "refusing to fall back to another database. Pass a valid URL or "
                "remove the override."
            )
        raise DbUrlResolutionError(
            f"the database URL from {origin} is empty; refusing to run migrations "
            "without an explicit target."
        )
    candidate = str(value).strip()
    try:
        make_url(candidate)
    except ArgumentError:
        hint = " (explicit override; no fallback attempted)" if explicit else ""
        raise DbUrlResolutionError(
            f"the database URL from {origin} is not a valid SQLAlchemy URL: "
            f"{_safe_target(candidate)}{hint}"
        ) from None
    log(f"[alembic] database target ({origin}): {_safe_target(candidate)}")
    return candidate


def resolve_db_url(
    *,
    x_arguments: Mapping[str, str] | Sequence[str] | None,
    main_option: str | None,
    environ: Mapping[str, str],
    settings_loader: Callable[[], str],
    log: Callable[[str], None] = print,
) -> str:
    """Return the URL to migrate, honouring ``-x db_url=...`` above all else.

    Precedence: ``-x db_url`` > ``alembic.ini`` ``db_url`` > ``DATABASE_URL``
    environment variable > application settings. The first two are *explicit*:
    a blank or invalid value there raises rather than falling through.
    """
    x_args = parse_x_arguments(x_arguments)
    if DB_URL_KEY in x_args:
        return _validated(
            x_args[DB_URL_KEY], origin="-x db_url override", explicit=True, log=log
        )

    if main_option is not None and main_option.strip():
        return _validated(
            main_option, origin="alembic.ini db_url", explicit=True, log=log
        )

    env_url = environ.get("DATABASE_URL")
    if env_url is not None and env_url.strip():
        return _validated(
            env_url, origin="DATABASE_URL environment variable", explicit=False, log=log
        )

    return _validated(
        settings_loader(), origin="application settings", explicit=False, log=log
    )
