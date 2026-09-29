# Changelog

All notable changes to **AeroCast-NCR** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [1.22.0] - 2026-09-29

### Fixed

- **The site no longer hangs on "Connecting to the API server" after a cold
  start.** The shared warm-up gate gave itself a 25 s budget, but a Render
  free-tier wake takes far longer than that — measured live at **105.6 s**
  through the Vercel proxy on this deployment. The gate therefore expired while
  the container was still booting, every panel then re-issued its request into a
  dead socket, and the dashboard stayed on the connecting banner indefinitely
  until a manual reload. The budget is now 150 s with 60 attempts, so the
  deadline rather than the attempt cap governs the wait, and the banner shows a
  live elapsed-time counter so a genuine one-to-two minute wait reads as
  progress instead of a broken page.
- **The warm-up gate no longer trusts the CDN alone.** Real traffic reaches the
  backend through the Vercel rewrite, and that edge answers `502`/`504` with its
  own shorter timeout while Render is still booting — so the proxy could report
  "down" for an instance that had in fact just come up, and the gate would sit
  there for the full budget and then give up on a live backend. The gate now
  probes the backend origin directly as well, and either signal counts.
- Per-request transient retries raised 3 -> 4 as a second line of defence behind
  the shared gate.

### Changed

- `.vercel/` is now ignored. The Vercel project link file carries the project and
  org identifiers and must not be committed.

## [1.21.0] - 2026-09-29

### Fixed

- **The PM2.5 forecast, its SHAP explanation and the observed-vs-forecast
  verification panel no longer 500 on a lagging weather feed.** The feature row
  is anchored on the latest *pollution* observation, so whenever the pollution
  feed lags the weather feed the weather query returns zero rows — and
  `pd.DataFrame([])` has no columns at all, while `align_observations` indexes
  `wx["timestamp"]` unconditionally. That raised `KeyError: 'timestamp'` inside
  `build_feature_row`, which the API surfaced as
  `500 PM2.5 forecast failed: 'timestamp'` and
  `500 PM2.5 explanation failed: 'timestamp'` (the dashboard's "Why this
  forecast" panel then reported "Couldn't reach the explainability service",
  and the verification chart stayed empty because it needs a loaded PM2.5
  forecast). `_query_weather`/`_query_pollution` now return a zero-row frame
  that still carries its columns, so the exogenous features arrive as `None`
  and the model is still run on the real target-derived features rather than
  the request failing.

### Changed

- **Pollution refresh covers every Delhi station the opencity.in package
  publishes.** `CKAN_RESOURCES` listed only the five regional-composite
  monitors, so the other stations in the `stations` table had *zero* pollution
  rows and the PM2.5 forecast, its explanation and the verification panel all
  refused with `No pollution readings for station`. Nine further per-station
  datastore resources are now configured (Ashok Vihar, Aya Nagar,
  Jahangirpuri, Lodhi Road, Mundka, Okhla Phase-2, Shadipur, Sirifort,
  Vivek Vihar), bringing coverage to 14 of 17 stations. The map is now keyed by
  our own station names, so the hardcoded rename table is gone. Faridabad,
  Noida Sector-62 and Teri Gram have no resource in that package (they sit
  outside Delhi) and are still reported as `insufficient_history` by
  `/api/pollution/coverage` rather than being filled in.

## [1.20.0] - 2026-09-29

### Changed

- **`ENABLE_DEMO_USER=true` is now an explicit, allowed opt-in in production.**
  SIH26082 ships a demo login so reviewers can sign in and explore the portal.
  Unset still resolves to off in production (and the login-path/credential
  endpoint refusals are unchanged), but an explicit `true` no longer refuses to
  boot: the config emits a `RuntimeWarning` instead, noting that the published
  demo address and password will be served by `GET /api/auth/demo`. The demo
  hydration fabrication guard (`DEMO_HYDRATE_EMPTY_DB` in production) is
  unchanged and remains a hard refusal. `render.yaml` now declares
  `ENABLE_DEMO_USER: "true"`.

## [1.19.0] - 2026-09-28

### Added

- **Production startup is fail-closed on CORS too.** `validate_production_config`
  now refuses to boot with `CORS_ORIGINS` unset, empty, or `*` (a wildcard
  lets any site read the API), and rejects `DATABASE_URL` on any non-PostgreSQL
  scheme — not just the SQLite case. An unrecognised `ENVIRONMENT` string
  (e.g. a typo `productionn`) emits a `RuntimeWarning` on every boot because it
  silently disables the fail-closed guard; it warns rather than refusing so
  local dev is unaffected.
- **Belt-and-braces production guard inside `hydrate_demo_if_empty`.** The
  background task now refuses to synthesise observations when
  `get_settings().is_production` is true, even if a future caller forgets to
  check `demo_hydration_enabled` first (the config validator already refuses
  `DEMO_HYDRATE_EMPTY_DB=true` + `production` at boot).
- **`/api/system` schema honesty.** `schema_version` is now the current Alembic
  head read from the migration chain on disk (falling back to `"unmanaged"` /
  `"unknown"`) instead of a hardcoded `"1.0.0"` that could drift from the code.
- **CTM engine status no longer leaks server paths.** `/api/system` reports only
  the HYSPLIT executable leaf name and WRF-Chem `wrfout` file name, with
  statuses/notes, instead of absolute installation paths.
- **The demo credential is no longer shipped to browsers.** `LoginPage` fetches
  the demo account from `GET /api/auth/demo` at runtime instead of embedding
  `analyst@aerocast.in` / the published password in the client bundle. The
  endpoint 404s when the demo account is disabled, so in production the demo
  affordance stays hidden and no password rides the deployed frontend.
- Regression tests: CORS wildcard/empty refusals, unsupported DB scheme,
  unknown-environment warning, demo-hydration production refuset, system path
  sanitisation, plus the existing guard/contract suite extended.

### Changed

- **`_flush_json_value` never fabricates `0.0` for a gap.** `None`, NaN, ±inf,
  and non-floatable values now pass through as `None` (genuinely absent),
  which is what the refusal machinery in `_model_predict` needs to see:
  a missing measurement stays missing instead of becoming a confident zero.
- **`build_features_from_db_with_meta` drops fictional coordinates.** When the
  station record is missing, `latitude`/`longitude` are left absent (blocking
  the prediction via the required-features contract) instead of serving the
  NCR centroid `28.6139/77.2090` under composite data. Fire-impact feature
  builders tolerate a frame without a coordinate column; coupling never reads
  them.
- **Imputation is now limited to the two validated fire features.** A `None`
  feature fills to `0.0` only for `mean_frp`/`max_bright`, the values whose
  `0.0` is a truthful "no fire that hour" encoding matching training-time
  `fillna(0)`; any other absent feature stays blocking.
- **coupling_service stops inventing coordinates:** both the `get_coupling_inputs`
  centroid fallback and the `get_forecast_context` `or 28.6139` / `or 77.2090`
  coercions are removed, and `_fire_impact_from_rows` returns its honest empty
  result when station lat/lon are missing.

## [1.18.0] - 2026-09-28

### Added

- **Refusal-over-fabrication contract** in the forecast path. `MIN_FEATURE_ROWS = 2`
  and `STALE_AFTER_HOURS = 6.0`; under-powered or stale stations return
  `503 insufficient_data` with a machine-readable payload instead of a number.
  `None`/`NaN` propagate as `None` rather than being coerced to `0.0`.
  New suite `backend/tests/unit/test_forecast_data_sufficiency.py` (20 tests).
- **`forecast_runs` audit table** (`ForecastRun`) recording every generation
  attempt — succeeded / fallback / refused — with model name, artifact SHA-256,
  history row counts, observation window, and provenance flags. Both forecast
  endpoints now write a run; refusals are recorded, not swallowed.
- **`pollution_observations.re_stamped`** so rows copied forward by the demo
  bootstrap are distinguishable from live sensor data in forecast provenance.
  `bootstrap_recent.py` stamps every row it writes.
- `backend/scripts/audit_re_stamps.py` — read-only helper that lists candidate
  legacy re-stamp rows for manual review instead of mutating them.
- `backend/scripts/entrypoint.sh` — container entrypoint that migrates, then
  `exec`s uvicorn (staying PID 1) and refuses to start on migration failure.
- `backend/scripts/migrate_safely.py` — guarded migration entrypoint used by both
  `render.yaml` (`preDeployCommand`) and `entrypoint.sh`. Detects whether the
  database is `versioned`, `unversioned` (tables present, no `alembic_version`),
  or `empty`, and upgrades or stamps accordingly. `--status` reports without
  writing; `--force-upgrade` replays the chain deliberately. 19 regression tests
  in `backend/tests/unit/test_migrate_safely.py` pin the decision, including
  assertions that the deploy configs and Dockerfile do not regress to a bare
  `alembic upgrade head`.
- `preDeployCommand: python backend/scripts/migrate_safely.py` in `render.yaml`.
- Station foreign keys on `weather_observations`, `forecasts`, and `alerts`;
  unique `(station_id, horizon_hours)` on `forecasts` with an idempotent
  `_upsert_forecasts()` replacing append-per-regeneration.
- Parametrised regression tests pinning that episode duration equals the sample
  count on every supported pandas, plus a sub-hourly case proving the added
  term is the *observed* interval rather than a hard-coded hour.
- Alembic revision `b8d3f1a9c4e2` (new head).

### Fixed

- **The Alembic chain had never completed on either engine.** Four independent
  defects:
  1. `alembic/env.py` skipped its `sys.path` insert when `PYTHONPATH` already
     contained the backend root, so the repo-root `app/` Render shim shadowed
     `backend/app` → `ModuleNotFoundError: No module named 'app.database'`.
  2. `f6a2e7b3c8d9` and `b9c9f4d1a7e2` used PostgreSQL-only `DELETE ... USING`.
  3. `d4a1e4c9f0b2` and `e2b1c3d4a5f7` used `ALTER SEQUENCE` /
     `RENAME CONSTRAINT` / plain constraint DDL that SQLite rejects.
  4. The baseline created `id` as a bare `Integer` with no `SERIAL`, so
     `d4a1e4c9f0b2` failed on a missing `pollution_readings_id_seq`. Because
     `main.py` swallowed migration exceptions, the production schema was in fact
     built by `Base.metadata.create_all()` with **no `alembic_version` table**.
     The baseline now declares `autoincrement=True`.
- **A bare `alembic upgrade head` in `preDeployCommand` would have failed every
  deploy against the existing production database.** Because that database has
  tables but no `alembic_version`, Alembic replays the whole chain against
  itself, every `CREATE TABLE` collides with an existing table, and the failure
  is non-idempotent — so each subsequent deploy fails the same way. Deploy
  targets now run `migrate_safely.py`, which stamps that state instead. It also
  refuses to stamp when recently added tables are missing, since claiming head
  there would hide a real schema gap until runtime.
- `migrate_safely.py` had to place `backend/` ahead of the repo root on
  `sys.path`, mirroring `alembic/env.py`; a repo-root-first ordering made
  `import app` resolve to the Render shim and raised
  `ModuleNotFoundError: No module named 'app.database'`. Its `alembic_version`
  probe uses the SQLAlchemy inspector rather than PostgreSQL's `to_regclass()`
  so the SQLite development path works too. The `Dockerfile` now copies the
  script, which `entrypoint.sh` invokes.
- **The PostgreSQL CI job leaked cached responses between tests.** The TTL cache
  is enabled on PostgreSQL and disabled on SQLite, and the rationale given was
  that tests need isolation. Gating on the *database dialect* satisfied that
  goal only on the developer's machine: the full local suite runs on SQLite with
  the cache off, while the `Migrations on PostgreSQL` job runs on Postgres with
  the cache **on**, so a payload computed for one test was served to the next.
  Two tests asserting an empty result failed there while passing locally:
  `test_dispersion_forecast_with_no_surface` expected `frames == []` and got 24
  cached frames, and `test_get_alerts_empty` expected `[]` and got 2 alerts.
  Caching is now refused while pytest is executing, whatever the dialect, and
  an autouse fixture clears the store around every test as a second line of
  defence. Production behaviour is unchanged: a PostgreSQL deployment still
  caches, which is the point of the cache. 8 tests in
  `test_ttl_cache_isolation.py` pin both halves, including a test that
  reproduces the original failure under a PostgreSQL URL.
- **Every inversion episode duration was one hour short on pandas 3.**
  `summarise_inversion_episode` computed its typical sampling interval as
  `np.diff(series.astype("int64")) / 3.6e12`, which hard-codes a nanosecond
  divisor. `Series.astype("int64")` is resolution-dependent rather than
  nanosecond-guaranteed: pandas 3 resolves `datetime64` to **microseconds**, so
  every 1 h gap became `0.001 h`, the median interval was `0.001`, and a 19-hour
  inversion was reported as `18.001` → `18.0`. The tests passed on the pandas
  2.2 pinned locally and failed on the pandas 3 that CI's `pandas>=2.2.0`
  resolves to, which is how the divergence surfaced. Now computed with
  `diff().dt.total_seconds()`, which is resolution-independent.
  `ml/preprocessing/training_dataset.py` already documented this exact hazard
  for its own epoch conversion; the inversion code did not follow it.
- **Batch mode on PostgreSQL silently dropped owned id sequences.**
  `batch_alter_table` rebuilds the table and replaces the sequence with
  `_alembic_tmp_<table>_id_seq`, orphaning later inserts. Batch mode is now
  SQLite-only; PostgreSQL uses plain `ALTER TABLE … ADD CONSTRAINT`.
  `b9c9f4d1a7e2` now creates a unique *index* for the same reason.
- `docker-compose.yml` healthcheck pointed at `/health`, which round-trips
  Postgres and exceeds the 5 s timeout on a cold database — a restart loop. Now
  uses `/api/health` (liveness), matching the `Dockerfile` and `render.yaml`.
- Station-scoped provenance: an empty station no longer inherits a neighbour's
  latest-reading timestamp. Missing PBL persists as `NULL` instead of `500.0`.
  SHAP no longer zero-fills unavailable inputs. Calm wind (a genuine `0.0`) is no
  longer treated as missing. `transport_risk_service` preserves genuine zeros.
- `/api/explanation/{station}` repointed to the matched `models/pm25` feature +
  SHAP path (was reading a mismatched per-station feature row) and now maps
  missing-data errors to a structured `503 insufficient_data`.
- ORM/migration naming parity for all station FKs and `ix_forecast_runs_id`;
  `alembic revision --autogenerate` now reports **zero drift** on both SQLite
  and PostgreSQL 16.

### Changed

- `re_stamped` is **not** backfilled for historical rows. The original pm25
  heuristic was measured against the real database and flagged 116,029 of
  133,407 rows (87%); a full-value-tuple fallback mostly matched all-`NULL` /
  `aqi=0` rows. No heuristic can distinguish a legacy re-stamp from a genuine
  repeat, so existing rows stay `false` and `audit_re_stamps.py` supports manual
  review instead.
- README rewritten: accurate test count (974), corrected health-probe guidance,
  new "Refusal Over Fabrication" and "Database & Migrations" sections, the
  `migrate_safely.py` decision table, a candid "Known Issues" section, and a
  deployment-status note recording that the hosted Render backend was
  unreachable when verified on 2026-09-28.

## [1.17.0] - 2026-09

### Fixed (live dashboard URL pointed at a stale Vercel project)

The documented dashboard URL resolved to
`air-pollution-weather-coupled-forecasting-system-methila.vercel.app`, a Vercel
project that is **not** connected to the repository and was still serving a
pre-`1.15.2` build (bundle `index-BXCbbjES.js`, 979 KB, built 2026-09-26 05:03Z
— none of the `frame_hours`, warm-up backoff, or `Retrying` UI markers present).

The git-connected project is
`https://air-pollution-weather-coupled-forec-eight.vercel.app`, which serves
`index-CzMUUmgp.js` (1,042,004 bytes, 2026-09-26 21:51Z) with all markers
present, and whose `/api/*` rewrite proxy returns HTTP 200 for `/api/health`,
`/api/stations`, and `/api/inversion/{station}` carrying the new geometry and
episode fields.

Updated `README.md`, `docs/PS_SUBMISSION.md`, and `render.yaml`
(`FRONTEND_URL`) to the current origin. `CORS_ORIGINS` remains
`sync: false` and still needs to be set in the Render dashboard if the
backend is ever called cross-origin; the Vercel rewrite proxy makes normal
dashboard traffic same-origin, so it is not required for the live path.

Two further Vercel projects exist with the same base name and older bundles
(`…-forec-coral`, not git-connected, and `aerocast-ncr`, last built
2026-09-20). They are not referenced by the documentation and can be deleted.

### Added (inversion layer geometry, episode duration/persistence, and a measured coupling ablation)

Closes the two implementation partials recorded in `docs/SIH26082_FINAL_AUDIT.md`:
inversion **base/top height, thickness, duration and persistence** were listed as
"not implemented — requires a multi-day vertical archive", and the two-way coupling
features were present but never measured. Both are now implemented, and the coupling
claim is backed by a number rather than an assertion.

**Inversion layer geometry.** `ml/features/atmospheric_profile.py` gained
`pressure_to_height_m` (hypsometric, `dz = (R_d·T̄/g)·ln(p_ref/p)`, with
`R_d = 287.05 J kg⁻¹ K⁻¹` and `g = 9.80665 m s⁻²`) and `layer_heights_m`, which
prefers real geopotential heights and otherwise integrates upward from 1000 hPa
using each layer's mean temperature. `classify_gradient` and `combine_inversion` now
report `inversion_base_height_m`, `inversion_top_height_m`, `inversion_thickness_m`
and `inversion_thickness_hpa` for the strongest layer. `add_lapse_rate_inversion_features`
carries the same columns into the training matrix.

**Inversion episode statistics.** `summarise_inversion_episode`
(`ml/features/inversion.py`) measures the current episode over the stored hourly
history: `current_duration_h` for the unbroken run of inversion samples and
`persistence_fraction` over the trailing 24 h. A sampling gap > 3 h
(`EPISODE_GAP_TOLERANCE_H`) breaks the run and the gap is *not* added to the
duration, because nothing was observed during it. `GET /api/inversion/{station}`
returns `inversion_duration_h`, `inversion_persistence`,
`inversion_persistence_window_h`, `inversion_measured_window_h`,
`inversion_episode_samples`, `inversion_episode_onset` and
`inversion_history_sufficient` (48 h bounded history query), so a short archive can
never be presented as a long episode. Geometry and duration are `null` under the
PBL proxy rather than fabricated.

**Coupling ablation.** `ml/evaluation/ablation.py` (`run_coupling_ablation`) fits
the same model twice on one identical chronological split — with and without the six
coupling features — selecting on validation and scoring paired held-out test rows.
Reproduce with `python scripts/run_coupling_ablation.py`. Measured on the real
`featured_dataset.csv` (RandomForest, pm2.5, 21,299-row test split):

| Horizon | MAE with | MAE without | Δ MAE | Δ % | Verdict |
|---------|----------|-------------|-------|-----|---------|
| t+6h  | 34.729 | 34.770 | −0.041 | −0.12% | no material change |
| t+24h | 41.364 | 42.510 | −1.146 | −2.70% | material improvement |

The report names the direction of the effect in both cases. Because two fits of the
same model differ slightly for reasons unrelated to the features, a difference below
`MATERIALITY_FRACTION = 0.5%` is reported as **no material change** instead of a
positive contribution — which is why the t+6 h result is not claimed as support. The
ablation trains throwaway models and writes only `reports/coupling_ablation_pm25.json`;
shipped models and the live API are untouched.

### Fixed (flaky TTL cache test)
`test_cached_rebuilds_after_ttl_expiry` slept 60 ms against a 50 ms TTL. That 10 ms
margin sits inside the Windows ~15.6 ms timer granularity, so the case failed roughly
1 run in 6 in full-suite runs. It now drives `ttl_cache.time.monotonic` with an
injected clock and asserts both the inside-TTL and past-TTL cases, with no sleep and
no timing dependency. 12/12 consecutive runs green.

### Changed
- `docs/SIH26082_FINAL_AUDIT.md` § F moves from `PARTIALLY IMPLEMENTED` to
  `FULLY IMPLEMENTED` (geometry + episode statistics now computed and exposed), with
  the two genuinely remaining gaps named: live multi-level ERA5 pressure-level
  ingestion, and a confidence measure on the episode statistics.
- `docs/SIH26082_FINAL_AUDIT.md` § J and `docs/SCIENTIFIC_METHODOLOGY.md` § 11.1 gain
  the measured ablation table; known-limitations list the single-seed caveat.
- Test count 664 → **748** (`backend/tests/unit/test_inversion_geometry.py` 32,
  `test_inversion_api_helpers.py` 22, `test_coupling_ablation.py` 32, plus API and TTL
  coverage). `ruff check backend ml scripts` clean.

## [1.16.0] - 2026-09

### Fixed (dead panels on a cold start: "Atmospheric profile unavailable", "No active alerts", an empty verification chart)

Reported from the live deployment: opening the Control Room during a Render
cold wake left whole panels permanently empty — the three atmospheric panels
("Atmospheric profile unavailable"), the alerts strip ("No active alerts" while
`/api/summary` reported 115 open alerts), the verification chart, and the
numerical dispersion panel. Every one of those endpoints answers in **under 12 s
once the instance is awake** (verified directly against production), so none of
it was missing data.

Three separate defects added up:

1. **The self-heal window was shorter than the cold start it was meant to
   survive.** The retry backoff (4/10/20/30 s) fired its last attempt 64 s after
   mount. The measured production cold wake is 76 s, so the final retry landed
   while the instance was still booting and then gave up for good. The schedule
   now spans ~229 s, and per-request retries went from 2 to 3 — with a 25 s
   warm-up budget per attempt, two retries gave up at roughly the 50 s mark.
2. **A panel claimed data was missing when the request had simply failed.** The
   "Why this forecast" card had exactly one attempt and reported any failure as
   *"No stored SHAP explanation for … yet — run the explainability pipeline to
   populate it"*. The pipeline is fine: `/api/forecast/pm25/explanation` returns
   a live `shap.TreeExplainer` pass in ~3 s. The panel now retries through the
   shared warm-up gate and, if it still cannot reach the service, says so and
   offers a retry instead of asserting something false about the database.
3. **`/api/dispersion/forecast` serialised 113,400 grid cells.** A 72 h run is
   72 frames × 1575 cells — **7.3 MB of JSON and ~11 s** of serialisation on a
   0.5-CPU instance — for a page that renders exactly one frame at a time. That
   is why the numerical dispersion panel was the one panel that could not be
   rescued by retrying.

### Added

- `GET /api/dispersion/forecast?frame_hours=6,12,24,48,72` — the solver still
  integrates the full horizon (the field evolves hourly, so every frame depends
  on the one before it), but only the requested hours get their per-cell grid
  materialised. `frame_hours_available` in the body advertises every hour the
  run produced. Omitting the parameter keeps the previous every-frame
  behaviour, so existing consumers and the integration tests are unaffected. The
  Spatial Forecast page asks for 6-hourly frames: 12 grids instead of 72.
- `CONTROL_ROOM_PREWARM`: a background task fills the TTL cache with the heaviest
  read-only payloads right after startup — atmosphere, alerts, summary,
  data-quality, transport risk, GRAP, the three fire aggregates, the statistical
  grid, and the dispersion runs for the 24/48/72 h horizons the Spatial Forecast
  page requests. Every key is built by the same `dispersion_cache_key()` helper
  the endpoint uses, so the sweep can only ever warm entries that are actually
  read. The first dashboard load after a cold wake is then served warm instead of
  queueing behind a set of cold ~12 s aggregations. Runs off the event loop, one
  entry at a time with a pause between, best-effort per entry, and never blocks
  readiness. Unset, it follows the environment: **on when `ENVIRONMENT=production`**
  (where the ~76 s cold wake makes ~75 s of off-request-path CPU worth paying),
  off for local dev, pytest and CI. `CONTROL_ROOM_PREWARM=false` opts out. It
  dispatches the solver first — the three dispersion solves are ~33 s of the 76 s
  wake, and dispatch is the one panel a retry cannot rescue — then the cheap reads.
- `GET /api/system` now reports a `prewarm` block (`enabled`, `state`,
  `setting`, `default_applied`, `entries_warmed`, `entries_failed`, `seconds`,
  `last_entry`). A cache warm-up is invisible by construction: verified against
  production, a sweep that was not warming anything looked exactly like one that
  was, and the only way to tell was to time requests by hand. The state also
  tells you whether the instance you are talking to has finished warming yet,
  and whether it is on because of the environment default or explicit config.

  This block immediately paid for itself: it reported `{"enabled": false}` on the
  live deployment, proving the sweep had never run there. The cause was that
  `CONTROL_ROOM_PREWARM` defaulted to `false` and Render does not push newly added
  `render.yaml` env vars to an already-created service — so the whole cold-start
  half of this release was inert in production. Hence the environment-aware
  default above rather than a silent `false`.
- `resilientGet()` in the API client: warm-gate + widening-delay retry for
  panels that live outside the dashboard's fan-out.
- A **Retry** button on the Spatial Forecast error banner, and honest
  "couldn't be computed" state (with retry) for the coupled two-way forecast,
  which previously swallowed its failure and left a permanent *"Run the coupled
  forecast…"* prompt that read like an action still owed rather than a dropped
  request. A station switch no longer leaves the previous station's coupled
  result under the new station's heading.

### Tests

664 passing (was 635). New: dispersion `frame_hours` subset (payload strictly
smaller, frames numerically identical to the unfiltered run, malformed values
fall back to every frame), dispersion cache-key separation/canonicalisation, and
a `prewarm` suite (the sweep runs the expensive solver first, pre-warmed keys are
exactly the keys the endpoint serves for the client's six-hourly frame sets, a
failing entry does not abort the sweep, the stop event is honoured, the setting
defaults on in production and honours an explicit opt-out, and its progress is
reported on `/api/system`).

Also worth recording from production verification of this release: an earlier
build warmed `dispersion:72:8:all`, which no client requested any more once the
page moved to filtered `frame_hours` — spending the most expensive read in the app
on an entry nothing could read. Fixed, and the key/frame-set equivalence is now
asserted in the suite.

## [1.15.2] - 2026-09

### Fixed (deadlock in the warm-up gate, and a blank page that never recovered)

A cold start could hang the dashboard **permanently**. The warm-up gate
`ensureWarm()` probed `/api/health` through the same axios instance that
carries the retry interceptor, and the interceptor calls `ensureWarm()` on a
transient failure. A 502/503/429 from the probe -- exactly what a waking
Render instance returns -- therefore sent the interceptor to `await
ensureWarm()`, which returned the very `warmUpPromise` the probe was already
awaiting. Neither could ever settle. Any page opened during a cold start hung
on its empty state instead of rendering, and `Dashboard` blocked on
`await ensureWarm()` before requesting a single panel.

- The probe now uses a bare same-origin `fetch('/api/health')`. It has no
  interceptors, so the gate is structurally incapable of re-entering itself,
  and an `AbortController` guarantees each attempt settles even if the socket
  hangs.
- Warm-up gate cut from `12 x 8 s` (up to **96 s** of blank dashboard) to a
  hard 25 s budget, after which the per-request retries take over.

### Fixed (control room stayed empty after a slow cold start)

The failure mode behind a fully blank Control Room was permanent: panels that
failed were recorded and shown in a banner, but nothing ever re-polled, so the
page sat on `--` / "Awaiting live data" / "No active alerts" until the user
manually reloaded -- even once the instance was up and serving.

- `Dashboard` now retries the whole fan-out on a widening backoff
  (4 s / 10 s / 20 s / 30 s) while any panel is failing, then stops and leaves
  the manual retry to the user. The backoff deliberately spans the 76 s cold
  wake measured in production.

## [1.15.1] - 2026-09

### Fixed (free-tier cold start: the warm path was using the wrong probe)

A measured live cold wake took **76.4 s** (then 818 ms warm). Python import is
only 2.6 s, so the cost was not the app -- it was the wake path itself.

The app exposes two probes, and everything on the wake path was using the wrong
one:

| probe | behaviour |
| --- | --- |
| `GET /api/health` | DB-free, returns as soon as the process accepts connections |
| `GET /health` | additionally round-trips Postgres to report `"database"` |

`client.ts ensureWarm()`, the `warmup.ts` keep-alive pinger and the Render
`healthCheckPath` all used `/health`. So every wake demanded a database
round-trip at the exact moment the connection pool had no connection yet,
which kept the client on "API offline" for longer than the process actually
needed and held the service "not ready" in Render's eyes. Warm, before any cold
start, the difference is already visible: `/health` 975-1910 ms vs
`/api/health` 306-1533 ms.

- Warm gate, keep-alive pinger, `render.yaml healthCheckPath` and
  `scripts/run_dev.py` now all use `/api/health`.
- `backend/Dockerfile` `HEALTHCHECK` also moved to `/api/health`. It ran the
  DB-backed probe against a **5 s** timeout, so a slow first connect would mark
  a healthy container unhealthy and risk a needless restart loop.

### Fixed (duplicate 17-station sweeps)

`/api/alerts` cached its result under `alerts:{station}`, so every distinct
station filter ran its own full network sweep; `/api/summary` then called the
same sweep **uncached** just to produce `open_alerts`. On the pooled Neon
Postgres that showed up as `/api/alerts` 7.8 s and `/api/summary` 11.9 s.

The 120 s cache now lives inside `alert_service.all_station_alerts`, keyed once
on the unfiltered sweep, with the station name applied afterwards. Both
endpoints share one computation, and browsing stations from the new menu
reuses the sweep instead of recomputing the network.

## [1.15.0] - 2026-09

### Fixed (429 / "API offline", alerts coverage, station menu)

Three defects surfaced during a live demo of the deployed Vercel + Render
free-tier stack. All three were reproducible against production.

**1. HTTP 429 was never treated as retryable, so panels died permanently.**
Render's free tier answers a waking instance with 429 as well as 502/503, but
the client only classified `0/502/503/504` as transient. A rate-limited wake
was therefore surfaced as a hard error: a red "API offline" badge plus
"Request failed with status code 429" that never recovered, even though the
instance came up seconds later. The same missing case made `AuthContext`
treat a 429 as a rejected session (spurious mid-demo logout) and `LoginPage`
report it as **"Invalid email or password"**.

- `api/client.ts` -- `isTransientStatus()` now covers `0/408/429/500/502/503/504`;
  up to 2 retries with exponential backoff; upstream `Retry-After` honoured. A
  429 backs off directly instead of paying the full wake-up wait, because the
  instance is already up.
- The shared warm-up gate now probes `/health` instead of `/system`. `/system`
  runs a real database round-trip, which is precisely what should not be
  hammered while the container is still importing.
- Request storm removed: a dashboard mount fanned out ~14 requests at once --
  the exact shape that trips the free-tier rate limiter. Requests now queue
  behind a 4-in-flight cap, duplicate in-flight GETs are de-duplicated by URL,
  and POSTs are staggered. Every panel still loads; they just no longer arrive
  as a thundering herd.
- `api/warmup.ts` -- keep-warm pinger on `/health` every 3 min, skipped while
  the tab is hidden.
- `pages/LoginPage.tsx` -- demo sign-in no longer issues a `GET /api/auth/demo`
  on mount (the value is a published constant; the request only added a second
  competing retry loop during cold start). Progressive backoff, and the button
  now shows a spinner plus "Server is waking up" / "retrying" instead of
  greying out silently.

**2. Alerts covered 1 of 17 stations.** `GET /api/alerts` replayed the persisted
`alerts` table, which is only appended to by `POST /api/forecast/generate` -- an
endpoint the UI never calls. Production returned 6 alerts, all for Anand Vihar,
all frozen at a single timestamp; the other 16 stations had none.

- `services/alert_service.py` -- added `all_station_alerts()` /
  `build_station_alerts()`, which evaluate the existing rule engine for every
  station on demand from the latest forecast run, with a latest-observation
  fallback for stations that have no forecast. One NCR-wide fire scan is shared
  across the batch. Measured locally: 53 alerts across 15 stations, up from 6 on 1.
- `GET /api/alerts` serves the live evaluation (120 s TTL cache) and accepts
  `?station=` to scope the feed; unknown stations 404.
- `api/summary.py` `open_alerts` now counts the same live set, so the header
  figure and the Alerts view agree.
- `POST /api/forecast/generate` still appends to the table for audit history.

**3. No way to reach a station other than the default.** New **Stations**
dropdown in the primary and mobile navigation listing all 17 monitoring stations
(session-cached, no refetch per menu open) which routes to
`/alerts?station=<name>`; the Alerts page also gained a station `<select>` and a
station badge on the active-alert count.

### Fixed (provenance honesty)

- `api/summary.py` checked `live_refresh_enabled` **before**
  `demo_hydrate_empty_db`, and the hosted demo sets both -- so the UI always
  reported `data_mode: "live"` even though demo re-stamping is what makes the
  observations look current, and the honest `demo_seeded` branch was unreachable
  in production. The more specific, more conservative mode now wins.

## [1.14.0] - 2026-09

### Fixed (backend latency + cold-start consistency)

Measured warm latencies showed the Render free tier (0.5 CPU) answering slow
for several uncached read endpoints (plume-risk ~6s, grid ~7s, events ~10s,
pm25 forecast ~7s, GRAP ~4.5s) and a 45–120 s boot after idle. Now:

- TTL cache (already used for `/summary`, `/data-quality`, `/atmosphere/current`,
  `/transport-risk/current`) is applied to the remaining slow reads: `/plume-risk`,
  `/fire-activity`, `/fire/hotspots`, `/grid/forecast` (per horizon),
  `/grid/overview`, `/dispersion/forecast` (per horizon/start), `/grap/current`,
  `/events/current` (per station/hours), `/forecast/pm25` (per station/hours),
  `/forecast/{station}`, `/forecast/ncr`. Cache is keyed by query params, bypassed
  on SQLite (tests keep full isolation), invalidated on live-refresh/seed.
- Frontend: transient failures now go through a SINGLE shared warm-up loop
  (`ensureWarm`) that pings `/system` until the scale-to-zero Render instance has
  booted, then every queued panel retries exactly once — no more per-panel retry
  storm leaving a patchwork of dead cards, and no multi-minute per-request hangs.

Verified: warm POST `/forecast/coupled` (coupled-two-way) + `/scenario/analysis`
return 200; all GET surfaces return 200 direct and via the Vercel proxy.

## [1.13.0] - 2026-09

### Fixed (frontend resilience — cold-start recovery)

The Render free-tier backend scales to zero after ~15 min idle; its cold boot
(heavy ML imports) can exceed the old 32 s retry budget, leaving panels frozen
on empty states ("No stations loaded", hotspots 0, plume/smoke "--") until a
manual reload. Deployed endpoints were verified working (all GET + coupled/
scenario POSTs return 200 both direct and via the Vercel proxy); the app now
survives cold starts instead of leaving permanent placeholders.

- `api/client` — transient retry budget raised to 8 attempts × 10 s (covers
  90–120 s boots); the two idempotent compute POSTs (`/forecast/coupled`,
  `/scenario/analysis`) now also retry on 502/503/504.
- `api/warmup.ts` + `main.tsx` — while any tab is open, a background pinger
  polls `/system` every 4 min so the demo never hits a cold start mid-session.
- `pages/NCRMap.tsx` — `load()` extracted, Retry actions on the map error and
  "No stations loaded", auto-refetch on tab visibility when data is empty.
- `pages/Dashboard.tsx` — refetch stations on tab visibility + Retry control on
  the map empty state.
- `pages/SpatialForecastPage.tsx` — separate station-error banner with Retry
  button + auto-refetch on tab visibility.

Verified live (proxy + direct): `/stations`, `/pollution/latest`,
`/fire/hotspots`, `/plume-risk`, `/fire-activity`, `/summary`, `/grid/forecast`,
`/transport-risk/current`, `/alerts`, `/atmosphere/current`, `/coupling[+/features]`,
`/events/current`, `/data-quality`, `/grap/current`, `/dispersion/forecast`,
`/model/performance`, `/model/metrics`, `POST /forecast/coupled` (coupled+
uncoupled series), `POST /scenario/analysis` — all 200.

## [1.12.0] - 2026-09

### Added (frontend)
- **Coupling directional status + scientific flow (Phases 19/20).** New `CouplingStatusFlow`
  component on the Atmosphere page (inside `CouplingPanel`): directional availability chips
  (Meteorology → Pollution / Pollution → Meteorology = ACTIVE / LIMITED / UNAVAILABLE) derived
  only from the live coupling-features payload (`coupling_state`, `coupling_domains`,
  `data_quality` + input availability) and the aerosol-feedback diagnostics; plus a clickable
  5-stage flow (weather → atmospheric state → pollutants → aerosol feedback → atmospheric
  response) where every stage shows its actual variables, current values and source/method.
  Unavailable inputs render "Unavailable" — never a fabricated value; the feedback term is
  explicitly labelled an ML surrogate, not WRF-Chem.
- `CouplingFeaturesResponse` type now carries `coupling_state` / `coupling_domains` /
  `data_quality` (present in the backend response since 1.10.0).

### Docs
- `docs/SIH26082_GAP_AUDIT.md` — nine-point gap audit (implemented / partially implemented /
  missing / UI-only / ML-connected / real-data / scientifically weak / simulated / improvable)
  with the Phase 19/20 change set appended.

## [1.11.0] - 2026-09

### Added (frontend)
- **What-if Scenario page** (`/scenario`, Tools menu) wired to `POST /api/scenario/analysis`:
  per-station baseline-vs-scenario PM2.5 outlook for 6–72 h with wind speed, wind direction,
  PBL height, regional fire-activity and inversion perturbations; baseline/scenario chart,
  per-horizon Δ bars, input-change table, hourly detail. Explicitly labelled "Scenario
  simulation — not a real forecast".
- **Data & Methodology page** (`/data-methodology`, Tools menu): data provenance with value-kind
  classification (observed / reanalysis / forecast / derived / ML predicted / scenario simulated),
  per-source table (provides / update frequency / spatial / temporal / limitations), pipeline
  diagram, scientific-methodology summaries, honesty commitment.

### Docs
- `docs/SIH26082_TRACEABILITY.md` — PS clause → implementation → file → API → UI → evidence/test →
  status matrix covering every explicit SIH26082 clause.
- `docs/SIH26082_FINAL_AUDIT.md` — final audit (sections A–P) with FULLY / PARTIALLY /
  NOT IMPLEMENTED classification; 23 IMPLEMENTED, 1 PARTIALLY (real-engine WRF-Chem run is the
  gated adapter), 0 NOT IMPLEMENTED.

## [1.10.0] - 2026-09

### Added (backend)
- **Coupling-state persistence (Phase 30).** New `coupling_states` table (one latest row per
  station, Alembic `5c1b7d9a2f6e`) written through on every `get_coupling_features` call,
  storing the raw inputs (wind, PBL, inversion, fires), the nine coupling features and the
  labelled `coupling_state` / `coupling_domains` / `data_quality`. New endpoints
  `GET /api/coupling/state` and `GET /api/coupling/state/{station}` (404-safe), schemas
  `CouplingStateSnapshot` / `CouplingStateListResponse`.
- **Label resolution.** `coupling_state` (NONE/LOW/MODERATE/HIGH from the feedback-surrogate
  band), `coupling_domains` (aerosol/atmospheric/feedback/fire/ozone), `data_quality`
  (GOOD/PARTIAL/SPARSE/UNAVAILABLE) exposed on the features payload.
- **Alert-engine unit suite.** `backend/tests/unit/test_alert_service.py` (30 tests) covering
  every deterministic alert trigger (AQI thresholds 201/301/401, trend, PM2.5 dominance,
  wind, inversion/PBL, humidity, precipitation washout, regional fires), severity ordering
  and resilience to missing/empty inputs.

### Added (frontend)
- **500 km influence ring + NCR modelling-domain boundary** on the Leaflet maps – the ring
  exactly matches `fire_impact.DEFAULT_MAX_DISTANCE_KM`; the dashed polygon marks the
  ~2.2 km numerical grid domain (28.2–28.9°N, 76.6–77.5°E), each with an explanatory popup.
- **Humidity + pressure columns** in the 72h atmospheric-context table.
- **Alert severity text badges** (and the previously-missing ADVISORY level styling) in the
  Alert centre.
- **Uncertainty note + chart/table accessibility** on the 72h Forecast page (point-vs-band
  disclosure, `aria-label` chart, hidden table captions); map keyboard navigation restored
  (`keyboard={false}` removed).

### Docs
- `docs/SCIENTIFIC_METHODOLOGY.md` – formal formulas, constants, units and assumptions for
  the coupling engine, inversion (lapse-rate + PBL proxy), fire impact, AQI, alerts, models,
  transport risk and dispersion surrogate.
- `SIH26082_IMPLEMENTATION_AUDIT.md` rewritten to the Phase-41 24-row status table
  (`IMPLEMENTED` / `PARTIALLY IMPLEMENTED` / `NOT IMPLEMENTED` / `NOT AVAILABLE` only).
- README: SIH 2026 correction, new coupling-state API rows, scientific limitations section.

### Quality
- Full suite green: **635 tests passed**; `ruff check backend/app backend/tests` and
  `ruff check ml` both clean; frontend `tsc --noEmit` + `vite build` clean.

## [1.9.0] - 2026-09

### Added (backend)
- **Coupling engine — nine named meteorology–pollution–fire features.** New pure module
  `ml/features/coupling_engine.py` computes dispersion/accumulation potential,
  inversion trapping, pollution stagnation, aerosol accumulation, fire-transport influence,
  regional transport, ozone-photochemical potential and a meteorology–pollution interaction
  surrogate (all 0..1 with an explicit `basis` string). Missing inputs → `None`
  ("Data unavailable"), never invented.
- **Endpoints** `GET /api/coupling/features/{station}` and
  `GET /api/forecast/{station}/context` (72 horizons, each with the nearest stored weather
  row ±2 h up to ±6 h, lapse-rate inversion state and coupling features). New schemas:
  `CouplingFeaturesResponse`, `ForecastHorizonContext`, `ForecastContextResponse`.
- **WRF-Chem spec interface.** `WRFChemAdapter.validate_configuration()`,
  `run_forecast()` and `get_output()` now exist and surface exact configuration failures —
  still honest: no genuine `wrfout_d01_*.nc` output → `CtmUnavailable`.

### Added (frontend)
- **Atmospheric-context table** on the 72h Forecast page (per-horizon temp, wind, PBL,
  inversion category/source, dispersion, accumulation, stagnation, fire/regional transport,
  O3 potential) with Low/Moderate/High bands.
- **Dispersion/accumulation + regional-influence panels** on the Atmosphere page
  (feature bars + per-feature basis, provenance for fire data).
- **Plume-transport pathway overlay**: upwind (≤500 km, wind-FROM within ±90°) FIRMS fires
  ranked by FRP render as dashed amber corridors to the Delhi NCR centroid (top 6), with a
  wind vector, "regional mean wind FROM <compass>" and an explanatory centroid popup.

### Changed (frontend)
- **About modal** now shows a data-source table + a six-item scientific limitations list
  (honest WRF-Chem adapter, PBL/inversion proxies, coupling potentials).
- **Architecture page** loads live `/api/model/performance` XGBoost metrics instead of
  hardcoded accuracy claims; Overview page deduplicated its AQI categories onto `lib/aqi`.

### Tests
- `backend/tests/unit/test_coupling_engine.py` (15) and `backend/tests/test_coupling_features_api.py`
  (8) added; CTM engine suite extended with the WRF-Chem spec-interface contract. Frontend
  `tsc --noEmit` + `vite build` verified.

## [1.8.8] - 2026-09

### Added (frontend)
- **About AeroCast-NCR modal.** An About entry in the top bar, the mobile
  user menu and the footer opens an accessible modal (Esc / backdrop close)
  that ties the app back to the SIH 2026 problem statement (SIH26082) and
  summarizes the platform's features and data sources.

### Changed (frontend)
- **Removed the unused monitoring-station dropdown** from the Overview page;
  it duplicated the station pickers on the Map and Forecast pages.

### Tests
- Frontend production build verified (`tsc && vite build`, 2341 modules
  transformed, dist generated cleanly).

## [1.8.7] - 2026-09

### Fixed (production / CI)
- **PostgreSQL 500 on pooled (data-sparse) stations.** The forecast/explanation
  history build merged a naive-UTC regional composite frame against tz-aware
  weather columns; PostgreSQL returns `DateTime(timezone=True)` as aware while
  SQLite returns naive, so the `pd.merge` raised
  `ValueError: merge on datetime64[us] and datetime64[us, UTC]`. Both frames are
  now normalised to the repo's naive-UTC convention before the outer join.
- **CI ruff job is green again.** Files that shipped without a trailing newline
  (`model_performance.py`, `model_performance_service.py`, `system.py`,
  `test_system_api.py`) and an unused loop variable were repaired; the
  front-door `ruff check backend/app backend/tests` step no longer blocks every
  push with W292/B007.

### Changed (observability)
- **Single database-liveness helper.** `/health` and `/api/system` now share
  `database_reachable()` instead of each re-implementing the `SELECT 1` check,
  so the readiness probe and the architecture-page engine report can never
  drift apart. Corrupted characters in the probe docstrings were cleaned up.

### Tests
- Verified full suite: **573 passed** (unit + integration). New coverage pins
  the mixed-tz merge, the liveness helper's connected/degraded paths for both
  endpoints, and probe behaviour when the database is unreachable.

## [1.8.6] - 2026-09

### Fixed (developer tooling)
- **`make clean-data` no longer fails.** The target referenced a missing
  `scripts/clean_generated.py`; the script now exists and is documented in
  `scripts/README.md`. It removes only regenerable artifacts (Python/tooling
  caches, runtime logs, local `*.db`, `frontend/dist`, engineered datasets and
  ML byproducts), never source CSVs or committed model weights. Dry-run is the
  default; pass `--exec` to apply. Virtualenv-internal caches are excluded.
- **Pytest coverage artifacts are now gitignored** (`.coverage`, `.coverage.*`),
  keeping `git status` clean after `pytest --cov` runs.

## [1.8.5] - 2026-09

### Fixed (demo panels on warm databases + honest status badge)
- **Demo panels no longer stay empty on a database that already has live
  observations.** The demo hydration previously skipped *everything* when the
  last 24 h already contained pollution+weather, so a warm DB never gained the
  seeded alerts, model metrics, fire archive or persisted forecasts — stations
  showed "No forecast stored", "Forecast engine not available" and missing
  metrics. Hydration now runs an idempotent auxiliary pass on **every** boot
  (`_ensure_auxiliary_demo_data`) that seeds metrics/alerts/24 h fires and
  persists vanilla 72 h forecast rows for every station missing them.
- **Status badge stops crying "API offline" through a cold start.** A single
  failed `GET /summary` (the first request on a waking Render instance) no
  longer flips the header pill to red; it only reports offline after two
  consecutive exhausted loads (~>2 min), by which time the instance is truly
  unresponsive.

## [1.8.4] - 2026-09

### Fixed (cold-start resilience for live demo)
- **Every panel now self-heals through a Render cold start.** The axios client
  retries idempotent `GET`s up to 4 times (8 s apart) on transient gateway
  502/503/504 or network-timeout failures, so the first page loaded right after
  the free-tier backend wakes no longer strands panels on "API offline",
  "Loading…", "No forecast stored" or "No active hotspots".
- **Keepalive workflow** (`.github/workflows/keepalive.yml`) pings the Render
  `/health` endpoint every 5 min (and on demand via `workflow_dispatch`), so
  the backend never idles to sleep mid-demo.

## [1.8.3] - 2026-09

### Fixed (cold-start session + deploy wiring)
- **Boot-time `getMe()` no longer logs the analyst out while the Render
  free-tier backend is still waking up from idle.** A transient 502/503/504 or
  network timeout now keeps the stored session (user + token) so panels can
  retry instead of kicking the analyst back to sign-in mid-demo — matching the
  login page's existing cold-start retry from 1.8.1.
- **`render.yaml` `FRONTEND_URL` corrected** to the live Vercel deployment
  (`air-pollution-weather-coupled-forecasting-system-methila.vercel.app`) so
  `GET /` on the Render backend 307-redirects to the real frontend.
- **`frontend/.gitignore` added** so Vercel build artifacts (`.vercel`,
  `.env*`) are never committed.

## [1.8.2] - 2026-09

### Fixed (deployed frontend UX)
- **`GET /` on the deployed backend no longer answers `{"detail": "Not Found"}`**
  �?" a browser hit on `https://air-pollution-weather-coupled.onrender.com` now
  307-redirects to the Vercel frontend (`FRONTEND_URL`, already set in
  `render.yaml` and `.env`). Local dev with no `FRONTEND_URL` gets a small
  landing JSON pointing at `/docs` and `/health` instead of a bare 404.
- New `backend/tests/unit/test_root.py` (2 tests) locking both behaviours.

## [1.8.1] - 2026-09

### Fixed (deployed demo stability)
- **OOM crash-loop on Render free-tier** — the startup demo self-hydration
  read the ~131k-row coupled dataset and ~383k-row FIRMS archive fully into
  pandas on a 512 MB instance, OOM-killing the container every boot (the
  intermittent **502** that made demo sign-in fail). The loaders
  (`backend/scripts/load_data.py`) now **stream CSV in chunks (25k/50k rows)
  and insert in small resilient batches** (2k rows each) with `gc.collect()`
  between chunks — a Neon-free-tier statement/connection hiccup skips one batch
  instead of aborting the whole hydration; the hydration task
  (`demo_hydration.py`) is deferred by a 45 s boot-grace so it never competes
  with Render's health-check window.
- **Demo sign-in resilient to cold starts** — the login page auto-retries once
  (~12 s) after a transient 502/503/504 or network timeout (Render free back-
  ends sleep after ~15 min idle), showing "server is waking up …" instead of an
  immediate "Demo sign-in failed".
- New `backend/tests/unit/test_load_data.py` (6 tests) locking chunked insert,
  idempotency and missing-file behaviour.

## [1.8.0] - 2026-09

### Added
- **Gated IMD official weather API adapter (WS-3, R9).**
  - `backend/app/services/imd_weather.py`: fetches genuine 7-day city forecasts
    from `api.imd.gov.in/api/v1/cityforecast` (station `42182`
    Delhi/Safdarjung, the NCR anchor). The gateway authenticates via key/IP
    whitelist and returns **HTTP 401 otherwise** — verified live with urllib in
    this environment; `imd_reasons()`/`IMDApiUnavailable` report exactly that,
    and nothing is ever fabricated.
  - `backend/app/api/imd.py` — `GET /api/imd/forecast` returns `{available,
    station, days:[7d max/min/condition], reasons}` (registered in `main.py`);
    schemas `ImdForecastDay`/`ImdForecastResponse`; config `imd_api_key`,
    `imd_station_id`, `imd_api_base`.
  - `scripts/fetch_imd_weather.py` — offline CLI writing
    `data/imd/imd_forecast.csv` (empty placeholder + honest reason on failure);
    `scripts/build_dataset.py` gains `load_imd()` and merges `imd_*` columns
    into the coupled dataset when real rows exist, warning + Open-Meteo
    fallback otherwise.
  - New `docs/imd.md`; `SIH_FINAL_COMPLIANCE.md` R9 note; 13 new unit tests
    (`backend/tests/unit/test_imd_weather.py`: 401/no-key reasons, real-shape
    parsing, malformed/non-JSON/HTTP-error rejection, router response,
    dataset glue).

## [1.7.0] - 2026-09

### Added
- **Real ERA5 reanalysis ingestion (WS-2, R9 subset).**
  - New gated reader `ml/features/era5_surface.py`: samples genuine
    Copernicus-CDS `reanalysis-era5-single-levels` NetCDF grids (`blh`, `t2m`,
    `sp`) at the 17 curated NCR stations (nearest grid cell), converts units
    (K→°C, Pa→hPa), and returns per-station, per-hour rows
    (`time, station, era5_temperature, era5_surface_pressure, era5_blh`).
    Reads NetCDF3-classic via `scipy.io.netcdf_file` with **zero extra
    dependencies** (prefers netCDF4/xarray if installed); handles the CDS
    `expver` split; reports honest `era5_reasons()` and returns an empty frame
    when no real file exists — nothing is ever fabricated.
  - `scripts/download_atmosphere.py` rewritten: proper CDS request (one NetCDF
    per archive year, idempotent, `--no-download`/`--force`), then extracts the
    station CSV through the shared reader; unreachable/unauthorised CDS writes
    a clearly-empty placeholder so `build_dataset.py` keeps using Open-Meteo.
  - `scripts/build_dataset.py::load_atmosphere` now consumes the real NetCDF
    samples via the shared reader (previously the NetCDF path was
    non-functional — downloaded `blh/t2m/sp` grids never became `era5_*`
    columns), with an honest "ERA5 atmosphere unavailable" warning + reason.
  - New `docs/era5.md`; `SIH_FINAL_COMPLIANCE.md` R9 note + 8 new unit tests
    (`backend/tests/unit/test_era5_surface.py`: gating, multi-year glob,
    nearest-cell sampling/units, descending latitude, expver, CSV glue).

**528 tests pass** (was 520). No DB migration, no frontend change.

## [1.6.0] - 2026-09

### Added
- **Real chemical-transport engine layer — NOAA HYSPLIT + WRF-Chem (WS-4, R6).**
  - New `ml/ctm/` package: `ctm_interface.py` (`CtmResult`, `CtmUnavailable`,
    `register`, `run_best_engine` with deterministic WRFChem→HYSPLIT priority),
    `regrid.py` (IDW gridding for sparse plumes).
  - `hysplit_adapter.py`: renders a standard HYSPLIT CONTROL file (line layout
    mirrors NOAA ARL `utilhysplit`), runs the real `exec/hycs_std(.exe)`, and
    parses the binary `cdump` using ARL's own record layout — validated against
    the real `cdump.bin` archived in `noaa-oar-arl/utilhysplit` — then regrids
    the plume onto the NCR domain. Strictly gated on executable + genuine ARL
    met files; never simulates; `CtmUnavailable` otherwise.
  - `wrfchem_adapter.py`: consumes genuine external `wrfout_d01_*.nc` output
    (xarray/netCDF4), units taken verbatim from the file.
  - `scripts/download_hysplit_gdas.py`: idempotent GDAS1 ARL archive fetcher
    from the verified ready.noaa.gov file scheme (`gdas1.<mon><yy>.w<k>`).
  - `dispersion_service.py` now tries genuine engines first and, on success,
    composites the engine plume *pattern* 50/50 with the coupled forecast
    surface (`mode`, `composite`, and `ctm` blocks disclose engine/units; the
    analytic surrogate stays the documented fallback when no engine is
    runnable). `mode="numerical_advection_diffusion"` fallback unchanged.
  - New `docs/hysplit.md` and `docs/wrfchem_adapter.md`; `SIH_FINAL_COMPLIANCE.md`
    R6 now ✅ (engine-gated). 14 new unit tests (`test_ctm_engines.py`) covering
    CONTROL layout, cdump round-trip + malformed rejection, surface regrid,
    gating, registry order, and composite logic.

## [1.5.0] - 2026-09

### Added
- **Pollution coverage for all 17 curated NCR stations (WS-1).**
  - New careful aliases map every data.gov.in / CPSB station name to the 17
    canonical monitors (adds Lodhi Road, Sirifort, Shadipur, Okhla Phase-2,
    Ashok Vihar, Mundka, Jahangirpuri, Aya Nagar, Vivek Vihar, Teri Gram /
    Vikas Sadan, Noida Sector-62, Faridabad/Sector 11).
  - Provenance tagging: every `pollution_observations` row now carries
    `data_source` (`data_gov_in`, `opencity_ckan`, `cpcb_dataset`); additive
    schema change only (alembic `d3e5f7a4b8c2` + SQLite `apply_migrations`).
  - New `scripts/backfill_pollution.py`: keyless, idempotent historical AQI
    backfill from the community CPCB hourly-AQI dataset (Vonter/india-cpcb-aqi,
    ODbL) for all sparse/empty stations; network-gated, offline-safe, honest
    (hourly AQI only, never fabricated).
  - Sparse-station resilience in forecasting: when a station has fewer than 24
    local readings it falls back to a regional composited signal from the five
    core stations (Anand Vihar, RK Puram, ITO, Dwarka, Punjabi Bagh) so the
    72-hour model is never run on empty history.
  - New `GET /api/pollution/coverage` endpoint reporting per-station reading
    counts, first/last timestamps, source breakdown, and sufficiency status
    (adequate / limited / insufficient_history / stale / no_data); model
    responses for `/api/forecast/generate` and `/api/forecast/coupled` now
    surface `pooled_features`, `local_readings`, and `history_days`.
  - Unit + API tests for aliases, coverage endpoint, and pooled fallback (506
    passing).

## [1.4.0] - 2026-09

### Added
- **Institutional light-theme UI redesign (full frontend overhaul).** The dark
  developer dashboard is replaced with a credible government/NGO atmospheric-
  services portal: deep institutional blue (`inst` palette) + white/light
  surfaces, national header with SIH26082 badge, secondary section navigation,
  breadcrumb-ready pages, footer disclaimer, `Inter`/Noto Sans font stack,
  WCAG-aware focus rings, and `prefers-reduced-motion` support.
  - New reusable components: `PageHeader`, `KpiCard`, `LoadingState` (skeleton),
    `ErrorState` (retry), `EmptyState`, `AQIBadge`.
  - Light CARTO basemap + light-themed Leaflet overrides; canonical CPCB AQI
    colour table centralised in `frontend/src/lib/aqi.ts`.
  - Legacy science pages (Overview, Spatial Forecast, AI Explanation, Data
    Tools) preserved and reachable under their original routes.
- **Portal authentication layer (additive, zero new dependencies).**
  - Backend: `hashlib.scrypt` password hashing + hand-rolled HS256 JWT (stdlib
    only) in `backend/app/security.py`; new `users` table (alembic
    `a1b2c3d4e5f6` + SQLite `apply_migrations`); idempotent demo-user seeding at
    startup; endpoints `POST /api/auth/login`, `GET /api/auth/me`,
    `POST /api/auth/logout`, `GET /api/auth/demo`. All data APIs remain public.
  - Frontend: `AuthContext`, `ProtectedRoute` guard, `/login` page with
    password visibility toggle + demo autofill, `/profile` page, user menu with
    avatar initials and sign-out. Credentials fully env-driven (`SECRET_KEY`,
    `DEMO_USER_*`).

### Changed
- `docs/UI_REDESIGN_AUDIT.md` — frontend/UX + authentication audit and plan.

## [1.3.0] - 2026-09

### Added
- **Expanded monitoring network: 5 → 17 stations.** Added Delhi sites (Lodhi
  Road, Sirifort, Shadipur, Okhla Phase-2, Ashok Vihar, Mundka, Jahangirpuri,
  Aya Nagar, Vivek Vihar), Gurugram (Teri Gram), Noida (Sector-62) and
  Faridabad — wired across `DEFAULT_STATIONS`, the live refresh service
  (`refresh_service.STATIONS`), `scripts/download_weather.py` and
  `backend/scripts/load_data.py`.
- **Historic weather for all new stations** — full Open-Meteo archive history
  (2023-01-01 → present) downloaded and loaded into the running database.
- **`scripts/seed_stations.py`** — idempotent seeding of stations + historic
  weather CSVs (`data/weather/*_weather.csv`), skipping already-loaded
  timestamps.
- **CSV import / export ("Data Tools").** New `GET /api/export/weather.csv` and
  `GET /api/export/pollution.csv` (alongside the existing
  `GET /api/export/forecast.csv`) download persisted observations per station,
  and new `POST /api/import/weather` / `POST /api/import/pollution` accept the
  same formats back as `text/csv`, so series round-trip cleanly. Imports are
  idempotent (keyed on `(station, timestamp)`, existing rows updated only when
  they changed), timestamps are stored per the app-wide IST-naive convention,
  blank `aqi` is recomputed from the six criteria pollutants, and unknown (non
  curated) stations are skipped and reported. New `frontend DataTools` page
  (`/data`) provides export buttons, CSV upload, templates and import summaries.

### Changed
- The `/api/forecast/ncr` aggregate, station list, grid overview, summary and
  data-quality reports now span all 17 stations; tests assert counts against
  `DEFAULT_STATIONS` instead of hard-coded five.
- **Live CPCB pollution now covers all 17 stations.** The data.gov.in ingestion
  maps the platform's monitor display names onto the curated canonical stations
  (`_STATION_ALIASES`: e.g. `IMD Lodhi Road` -> `Lodhi Road`, `R K Puram` ->
  `RK Puram`, `Dwarka-Sector 8` -> `Dwarka`, `Sector - 62` ->
  `Noida Sector-62`, `Sector 11` -> `Faridabad`). Ingestion no longer
  auto-creates ad-hoc stations — unknown monitors are skipped and counted as
  `station_skipped`, keeping the network exactly at the curated 17.
- `DATA_GOV_API_KEY` is configured with the shared public demo key; for stable
  scheduled ingestion register a free personal key at data.gov.in and replace
  it in `.env` (the demo key is rate-limited and intermittently returns 429s).

## [1.2.0] - 2026-09

### Added
- **Graded Response Action Plan (GRAP)** — CAQM stage matrix (Oct-2024
  revision: Stage I ≥201, Stage II ≥301, Stage III ≥401, Stage IV >450) as a
  pure service (`grap_service.py`), three API endpoints, and a dashboard panel.
  - `GET /api/grap/stages` — full referential matrix (incl. not-invoked).
  - `GET /api/grap/current` — live NCR assessment from persisted 24-hour
    average AQI, shallowest-PBL inversion proxy and FIRMS mean FRP, with
    rationale and an actionable measures list.
  - `GET /api/grap/{station}` — per-station assessment.
- Dashboard section 4 "Graded Response Action Plan" with stage badge, advisory,
  measures and why-this-stage rationale (`GrapPanel`).

## [1.1.1] - 2026-09

### Added
- **Command dashboard** — new `/` route (`frontend/src/pages/Dashboard.tsx`)
  aggregating the direct PM2.5 forecast with conformal bands, atmospheric
  conditions, regional fire intelligence + transport risk, SHAP
  explainability, cross-model performance and a wind-arrow station map.
- **Client + types wiring** — `getPm25Forecast`, `getAtmosphereCurrent`,
  `getTransportRisk` in `frontend/src/api/client.ts` with full response
  typings; `StationMap` renders flow-direction wind arrows.

### Fixed
- **Weather-ingestion dedup** — the pre-insert lookup compared tz-aware
  PostgreSQL datetimes against naive-UTC source timestamps (always unequal),
  so duplicate weather rows accumulated. Existing timestamps are now normalized
  to naive-UTC before the set-membership check.
- **Weather uniqueness enforced** — Alembic migration
  `f6a2e7b3c8d9_weather_unique_ts` adds `(station_id, timestamp)` on
  `weather_observations` after deduplicating pre-existing rows; the same dedup +
  unique index is applied to SQLite dev DBs via `apply_migrations()`.
- **Junk pollution rows** — the CKAN feed occasionally returns rows with a
  valid timestamp but every sensor value NULL; these are now skipped instead of
  inserting empty observations.
- **Pollution-events 500** — the forecaster serialises timestamps as ISO-8601
  strings, which broke run-gap arithmetic (`str - str`) in the event rules;
  `_parse_ts` normalises them to naive-UTC datetimes (regression tests added).

## [1.1.0] - 2026-09

### Added
- **GRU deep-learning model** — custom NumPy 2-layer GRU (`ml/training/train_gru.py`,
  `ml/models/gru_model.py`) trained at all 6 horizons for all 6 pollutants
  (no PyTorch/TensorFlow dependency). Honestly underperforms XGBoost
  (h-1 R² 0.336 vs 0.879); retained as a candidate ensemble member while the
  live PM2.5 endpoint serves the higher-accuracy XGBoost direct models.
- **Direct PM2.5 forecast engine** — dedicated multi-horizon XGBoost with
  conformal prediction intervals, model-card and feature explanation
  endpoints (`GET /api/forecast/pm25`, `/api/forecast/pm25/model-card`,
  `/api/forecast/pm25/explanation`).
- **4-model evaluation** — persistence / RF / XGBoost / GRU across all
  horizons written to `models/pm25/evaluation.json` + `.csv`.
- **Pollution event detection** — surge / relief / sustained high-risk
  episodes with confidence and atmospheric contributors
  (`GET /api/events/current`, `docs/events.md`).
- **Scenario (what-if) analysis engine** — read-only perturbation of wind /
  PBL / fire / inversion propagated through the coupled model
  (`POST /api/scenario/analysis`, `docs/scenario_analysis.md`).
- **Cross-model performance dashboard** — `GET /api/model/performance` and
  a frontend page comparing MAE/RMSE/R² across models and horizons.
- **Fire hotspots endpoint** — `GET /api/fire/hotspots` feeding the Leaflet
  map overlay with FRP-sized markers, plus `GET /api/transport-risk/current`.

### Changed
- `docs/api.md` covers all new endpoints (PM2.5 engine, events, scenario,
  model performance, transport risk, atmosphere/current, fire hotspots).

### Fixed
- (none)

## [1.0.0] - 2026-09

### Added
- Numerical dispersion transport core (`ml/features/dispersion_solver.py`) —
  finite-difference advection–diffusion–deposition–emission surrogate of the
  WRF-Chem dynamical core for the NCR grid, with lateral inflow boundary
  conditions, stubble-fire FRP point sources, urban emission, and
  non-negativity-preserving explicit numerics.
- API endpoint `GET /api/dispersion/forecast` (72h hourly AQI frames,
  per-frame two-way coupling diagnostics) and `backend/app/services/dispersion_service.py`.
- Frontend `/spatial` toggle between the statistical IDW grid and the live
  numerical dispersion field, with fire-plume overlay and hour scrubber.
- Full containerization: backend and frontend Dockerfiles, nginx `/api`
  reverse proxy, build-context exclusions.
- GitHub Actions CI (backend pytest + frontend build) and Dependabot config.
- MIT license, contributor guide, changelog, and expanded documentation
  (API reference, PS-to-implementation mapping, reproducibility, deployment).
- Unit-test layer covering the dispersion solver, coupling, grid service,
  fire impact, feature engineering and inversion modules.

### Changed
- Online two-way coupled forecast loop (`ml/features/coupled_loop.py`) with a
  time-stepped feedback path; all six criteria pollutants (PM2.5, PM10, O3,
  NO2, SO2, CO) across horizons {1, 6, 12, 24, 48, 72}.
- NCR spatial grid (`backend/app/services/grid_service.py`) at ~2.2 km with
  wind-advected IDW interpolation.
- `data/processed/featured_dataset.csv` (120 MB, over GitHub's 100 MB push
  limit) committed as two verbatim halves — see `data/processed/SPLIT_NOTE.txt`.

### Removed
- (none)

### Fixed
- Numerical stability of the dispersion core: corrected upwind flux indexing,
  CFL-safe adaptive time step, per-second wet-scavenging rates, and domain
  re-population via lateral inflow boundary conditions.

## [0.3.0] - 2026-07
### Added
- Two-way weather–chemistry coupling module (`ml/features/coupling.py`) and
  `/api/coupling/{station}` diagnostics.

## [0.2.0] - 2026-06
### Added
- Live refresh scheduler, packaging via `pyproject.toml`, frontend robustness.

## [0.1.0] - 2026-05
### Added
- Initial XGBoost/random-forest/persistence forecasters, AQI engine,
  explainability (SHAP), alerts, plume-risk, inversion detection.