# Alembic database-URL override safety fix

## Problem

`alembic -x db_url=...` was accepted by the CLI but never read by
`alembic/env.py`. Alembic exposes `-x key=value` values through
`Config.cmd_opts.x` and `EnvironmentContext.get_x_argument()`; the ini-file
accessor `config.get_main_option("db_url")` cannot see them. `env.py` used the
latter, so the override was silently ignored and resolution fell through to
`DATABASE_URL`.

This was not theoretical: a smoke test intended for a throwaway SQLite file
silently connected to the live `DATABASE_URL` and ran
`alembic upgrade head` against it (see
[`FIRE_PROVENANCE_FIX.md`](./FIRE_PROVENANCE_FIX.md) §4). The migration was
additive and was left in place by explicit decision, but the silent
fall-through is the hazard.

## Fix

New module `backend/scripts/alembic_db_url.py::resolve_db_url`.

Resolution order:

1. `-x db_url=...` — read via `context.get_x_argument(as_dictionary=True)`.
2. `db_url` in `alembic.ini`.
3. `DATABASE_URL` environment variable.
4. Application settings (`app.config.get_settings().database_url`).

The first two are **explicit and authoritative**. If an explicit value is blank
or not parseable by `sqlalchemy.engine.make_url`, resolution raises
`DbUrlResolutionError` instead of migrating a different database. Every
resolution logs the chosen target with any password masked, so a mistarget is
visible before DDL runs.

`alembic/env.py` calls the resolver and keeps the existing
offline/online migration paths unchanged.

## Usage

```powershell
# Explicit target (authoritative). Never falls back to DATABASE_URL.
python -m alembic -x db_url=sqlite:///./dev.db upgrade head

# Blank or malformed override is a hard error, not a fallback:
python -m alembic -x db_url= upgrade head      # raises DbUrlResolutionError

# No override: DATABASE_URL, then app settings.
python -m alembic upgrade head
```

## Tests

`backend/tests/unit/test_alembic_db_url_override.py` (12 tests):

* resolver precedence: `-x` beats `DATABASE_URL`; ini option; env; settings;
* `parse_x_arguments` matches Alembic semantics (`.split("=")`, no `=` → `""`);
* passwords are masked in the log line; the selected target is logged;
* safety: blank and malformed explicit overrides raise instead of falling back;
* end to end through the real CLI, with `DATABASE_URL` set to an intentionally
  unreachable PostgreSQL URL (`postgresql://…@127.0.0.1:1/never_used`) so any
  fall-through fails loudly:
  * `-x db_url=sqlite:///<tmp> current` succeeds and the SQLite file is opened;
  * `-x db_url=sqlite:///<tmp> upgrade head` then `downgrade -1` run the full
    chain on the disposable file (head `c5d7e9f1a3b0`, columns added then
    removed);
  * `-x db_url=` aborts with the resolution error and never contacts the
    fallback host.

No test connects to Neon. The exact disposable targets are pytest `tmp_path`
SQLite files, e.g.
`sqlite:///C:/Users/methi/AppData/Local/Temp/pytest-of-methi/pytest-<n>/test_cli_explicit_override_mig0/chain.db`.

## Verification

| check | result |
| --- | --- |
| `pytest backend/tests/unit/test_alembic_db_url_override.py` | 12 passed |
| Migration suites (`test_migration_integrity.py`, override tests, both PG-gated migration files) | 36 passed, 17 skipped |
| Full backend suite | 1190 passed, 17 skipped |
| Ruff (`alembic/env.py`, `backend/scripts/alembic_db_url.py`, new test) | clean |
| Manual CLI check: `-x db_url=sqlite:///<tmp> upgrade head` with `DATABASE_URL` unreachable | exit 0; logged target `sqlite:///…/override_manual.db`; ran `a1b2c3d4e5f6 → … → c5d7e9f1a3b0` |

The PG-gated migration tests skip locally by design (they require a PostgreSQL
`DATABASE_URL`); none were pointed at Neon.