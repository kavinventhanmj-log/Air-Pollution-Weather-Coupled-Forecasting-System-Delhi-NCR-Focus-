# Fire Observation Provenance Fix — Completion Report

Scope: end-to-end remediation of the defect where 383,093 synthetic fire
observations (generated stubble-fire history, dated 2023-01-01 …, Aqua/Terra)
were stored in `fire_readings` indistinguishable from FIRMS satellite
detections, and were consequently served as real observations by operational
endpoints.

Outcome: **synthetic observations can no longer be presented as real FIRMS
data anywhere in the product.** Provenance is stamped at write time, persisted,
excluded from every operational read, and surfaced in the API and UI when a
user explicitly opts in.

---

## 1. Root cause

`backend/scripts/load_data.py::load_fire_data` parsed `data/fire/firms_fires.csv`
but silently discarded the CSV's own `synthetic` provenance column. The
synthetic history therefore landed in `fire_readings` looking exactly like a
live FIRMS detection. Two consequences:

* every fire endpoint returned simulated 2023–2024 hotspots as current events;
* `backend/app/api/fire.py::_compute_plume_risk` consumed the last 500 rows with
  no recency window and no provenance filter, so the score was permanently
  clamped to HIGH (1.0) regardless of real conditions.

## 2. What changed

### 2.1 Schema — revision `c5d7e9f1a3b0` (new linear head)

`alembic/versions/c5d7e9f1a3b0_fire_provenance.py` (down_revision
`b8d3f1a9c4e2`) adds to `fire_readings`:

| column | type | notes |
| --- | --- | --- |
| `synthetic` | BOOLEAN NOT NULL DEFAULT 0 | legacy rows left `false` = "not known to be synthetic" |
| `source` | VARCHAR NULL | free-text provenance label |

`downgrade()` drops both columns. `backend/app/database.py::apply_migrations()`
mirrors both columns for SQLite so file databases stay usable without Alembic.
`backend/app/models/db_models.py::FireReading` declares them.

Legacy rows are deliberately **not** backfilled by the migration: no heuristic
can tell a legacy synthetic row from a real one after the fact, and guessing
would label real measurements as simulated. Restoration is a separate,
explicit, exact-match operator step (§2.3).

### 2.2 Writers — stamp provenance at ingest

* `scripts/generate_fire_data.py` — emitted history is tagged
  `source="synthetic_sim"`.
* `backend/scripts/load_data.py::load_fire_data` — reads and persists the CSV
  `synthetic` flag and `source`; a CSV without the column is treated as **real**
  FIRMS data (real-safe default); also persists `brightness`/`instrument`.
* `backend/app/services/firms_service.py` — live FIRMS download normalise/upsert
  stamps `synthetic=False, source="firms_live"`.

Both loaders gained a `_col(df, name, fallback)` helper. Without it,
`pd.DataFrame.get()` returned an *empty* Series for any optional column and the
strict `zip()` aborted the entire ingest with
`ValueError: zip() argument N is shorter than arguments 1-N`. This was not
hypothetical: the shipped `firms_fires.csv` has no `source` column, so the
backfill script crashed on the real dataset until it was fixed (§2.3).

### 2.3 Backfill — `backend/scripts/backfill_fire_provenance.py`

Deliberate, non-destructive, manual-only (never run automatically).

* **Exact match only** on `(satellite, latitude→4dp, longitude→4dp, naive-UTC
  acq_date)`, reconstructed from `acq_date`+`acq_time` with the same
  `acq_time` 9999/2400 fallback as `load_data`. Rows with no CSV match are left
  **completely untouched** — nothing is inferred or heuristically rewritten.
* `--dry-run` reports what would change and writes nothing; `--csv` and
  `--db-url` are explicit.
* Refuses to run (RuntimeError) when `fire_readings` lacks the provenance
  columns.
* Handles SQLite drivers that return `acq_date` as a raw ISO string, which
  would otherwise break every key silently.

The safety hardening added on top of the original backfill (exact-match,
explicit-target, dry-run-default) is:

* **Expectations gate the write.** `--min-matched`, `--expect-db-rows` and
  `--expect-matched` are checked against the read phase *before* `UPDATE` is
  issued. A failed expectation (including a zero-match target) aborts with exit
  code `3` and the table completely untouched. Exit codes are: `0` success,
  `2` refusal before any connection (bad flags/target/CSV), `3` anomaly.
  **A non-zero exit therefore always means nothing was written.**
* **Warnings do not fail the run.** Rows already carrying
  `source='firms_live'` (never rewritten) and rows with no CSV match (left
  untouched) are printed under `WARNING`. They are expected on a real target.
  An `--execute` run that updated 0 rows is still an error (exit `3`) — a
  no-op is not a success — *except* when the rows you would have written were
  all deliberately protected, which is the reconciliation working correctly and
  exits `0`.
* **Timezone fail-closed.** `fire_readings.acq_date` is TIMESTAMPTZ on
  PostgreSQL while the keys are naive UTC. The script probes the session
  timezone with a read-only `SELECT current_setting('TimeZone')`. An
  inconclusive probe on a non-loopback (production) `--execute`, or a
  positively non-UTC session, now **refuses the write** instead of silently
  assuming UTC; `--allow-non-utc-timezone` is the explicit waiver.
* **No credential chains.** Every DB-error path raises `BackfillError` with the
  password-masked target and `from None`, so a raw driver exception (whose text
  can include the DSN) is never chained into the operator-facing error or its
  traceback. The same applies to `alembic_db_url.py`.

Verified against `data/fire/firms_fires.csv`: 383,103 signatures =
383,093 synthetic / 10 real.

### 2.4 Readers — synthetic excluded from all operational output

Real-only filters added to: `api/fire.py`, `api/grap.py`, `api/summary.py`,
`main.py` (`latest_fire`, data-quality), and services `coupling_service`,
`forecast_service`, `dispersion_service`, `transport_risk_service`,
`pm25_forecast_service`, `scenario_service`.

### 2.5 API — `backend/app/schemas/schemas.py`

* `FireHotspot`, `FireEvent` — `synthetic: bool = False`, `source: str | None`.
* `PlumeRiskResponse` — `synthetic_fire_count`, `fire_basis` (e.g.
  `"Real FIRMS observations only (synthetic history excluded)"`), so the caller
  can see what the score was computed from and what was held back.
* `GET /api/fire/hotspots` — new opt-in `include_synthetic` query parameter,
  default `false`.

### 2.6 Frontend

* `types/index.ts`, `api/client.ts` — provenance fields; `getFireHotspots(includeSynthetic = false)`.
* `StationMap.tsx` — real hotspots are coloured solid markers with a "Live FIRMS
  Hotspot" popup; synthetic markers are grey dashed with "Simulated hotspot
  (not a live detection)".
* `StubblePlumePage.tsx` — off-by-default toggle *"Show simulated (2023–24
  synthetic history) overlay"* with a legend; `StubblePlume.tsx` footnotes the
  fire basis; `DataMethodology.tsx` documents provenance in the methodology.

The synthetic overlay is never on by default and is always visibly labelled.

## 3. Verification

| check | result |
| --- | --- |
| Full backend suite (`pytest backend/tests`) | **1272 passed, 17 skipped, 0 failed** (`DATABASE_URL` unset) |
| New provenance tests | 19 passed (`test_fire_provenance.py`, `test_fire_provenance_api.py`) |
| Migration chain on SQLite (`test_migration_integrity.py`) | 24 passed, incl. upgrade→downgrade of head `c5d7e9f1a3b0` |
| Ruff (all changed + new files) | clean |
| Mypy (`backend/app`, project config) | 277 errors vs **273 pre-existing at HEAD** — no new error classes; the +4 are the same pre-existing SQLAlchemy `Column`-kwarg pattern used by the surrounding serializer lines. The gate already fails at HEAD and is not enforced. |
| Frontend `npx tsc --noEmit` | clean |
| Frontend `npm run build` (tsc + vite) | built in 5.19 s |
| Backfill `--dry-run` (temp SQLite, 3-row fixture) | 3 scanned / 2 matched / 1 would-update / 1 untouched; **0 rows written** |
| Backfill live run (same temp DB) | exactly the mislabeled row corrected → `synthetic=1, source='synthetic_sim'`; real row untouched; no-match row untouched |
| Backfill signature load on the real 383k CSV | 383,103 signatures (383,093 synthetic / 10 real) |
| API smoke (TestClient, temp SQLite: 3 real + 4 synthetic fires) | see below |
| Backfill safety regressions (`test_fire_provenance_backfill_safety.py`) | 72 passed |
| Disposable-DB guard (`test_db_guard.py`) | 8 passed; a stray non-SQLite `DATABASE_URL` aborts the run (exit 3) with no credential in the message |

API smoke output (4 synthetic rows deliberately dated *after* the real ones —
the exact shape that previously saturated plume risk to HIGH):

```
hotspots        default: real: 3 | synthetic-leaked: 0
hotspots+syn    include:  total: 7 | synthetic-labeled: 4
   synth sample source: synthetic_sim
plume-risk      score: 0.089 | fire_count: 3 | synthetic_fire_count: 4
                fire_basis: Real FIRMS observations only (synthetic history excluded)
fire-activity   total_fires: 3 | hotspot_days: 0
fires/latest    count: 3 | all-real: True | sample-source: firms_live
data-quality    fire_readings: total 7, synthetic 4, real 3
```

A regression test asserts the same: a synthetic-only hotspot cluster leaves
`fire_count == 0` and the score at the LOW tier instead of HIGH 1.0.

### 3.1 Test-harness safety — disposable databases only

Adding a recovery path is not enough: the test harness itself must never be
able to touch a real database. `backend/conftest.py::db_session` calls
`Base.metadata.drop_all`, so a test run with `DATABASE_URL` pointing at
production would recreate the original incident in reverse. The harness now
refuses any non-SQLite target unless the operator explicitly opts in:

* `AEROCAST_ALLOW_LIVE_TEST_DB` must be truthy for `DATABASE_URL` to be
  anything other than SQLite. CI's throwaway PostgreSQL service sets it to
  `"1"` in the `migrations` job; the `backend` job runs with no `DATABASE_URL`
  and therefore exercises SQLite only.
* `pytest_configure` checks the environment before collection, and the
  `db_session` fixture re-checks the live engine's backend, so a stray
  `DATABASE_URL` fails the run instead of dropping tables.
* The refusal never echoes the URL or its password.

`backend/tests/unit/test_db_guard.py` pins both the refusal and the opt-in
without opening a connection.

## 4. Deployment note — incident

While attempting an isolated migration smoke test on a throwaway SQLite file,
revision `c5d7e9f1a3b0` was **applied to the live Neon database**. Cause:
`alembic/env.py` read the override with `config.get_main_option("db_url")`,
but `alembic -x db_url=...` is exposed through `Config.cmd_opts.x` /
`EnvironmentContext.get_x_argument()` — never the ini options — so the override
was silently ignored and the migration fell through to the `DATABASE_URL`
fallback.

Verified resulting state: `alembic_version = c5d7e9f1a3b0`, `fire_readings`
carries `synthetic` + `source`, 382,960 rows all `synthetic = false`. The change
is additive, non-destructive, and matches what a deploy would apply. Per explicit
user decision it was **left in place**, not reverted.

**Fixed (safety):** `alembic/env.py` now resolves the target through
`backend/scripts/alembic_db_url.py`, which reads `-x db_url` via
`context.get_x_argument(as_dictionary=True)`. An explicitly supplied `db_url`
is authoritative: if it is blank or not a valid SQLAlchemy URL, resolution
raises `DbUrlResolutionError` instead of falling through to `DATABASE_URL`. The
chosen target is logged with the password masked. Regression tests
(`backend/tests/unit/test_alembic_db_url_override.py`) run the real Alembic CLI
against a disposable SQLite file while `DATABASE_URL` points at an unreachable
host, proving both that the override is honoured and that it never falls
through. See `docs/ALEMBIC_DB_URL_OVERRIDE.md`.

## 5. Remaining limitations

1. **Neon rows still have `source = NULL`.** The schema migration alone does not
   restore provenance on existing rows. Until the operator runs the backfill
   against Neon (`--dry-run` first), legacy rows report `synthetic = false` —
   i.e. "not known to be synthetic". That default is honest but permissive: for
   the 2023–2024 synthetic history it is wrong until the backfill runs. The
   backfill is exact-match only and has been validated, but it has **not** been
   executed against production.
2. **PG-gated migration tests skipped locally** (`test_c5d7_fire_provenance_migration_postgres.py`,
   6 tests). They run only when `DATABASE_URL` is PostgreSQL (CI's migrations
   job); the revision itself has now been exercised on PostgreSQL in
   production, and on SQLite via the integrity suite.
3. **Synthetic overlay is opt-in**, so default UI/API output contains no
   simulated fire data at all.
4. Pre-existing, unrelated: 7 untracked scratch scripts remain in the repo root
   (`check_pg.py`, `inspect_db.py`, `neon_health_check.py`, …). Untouched.