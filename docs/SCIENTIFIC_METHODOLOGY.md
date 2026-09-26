# AeroCast-NCR — Scientific Methodology Reference

> Single source of truth for every formula, threshold, constant, unit and
> assumption used by the SIH26082-coupled forecasting system. Each section is
> traceable to the implementing module. The system is built entirely from
> **stored, real observations** (CPCB pollution, Open-Meteo weather +
> vertical profile, NASA FIRMS fires); no value is ever fabricated, and every
> derived quantity carries an explicit basis / provenance tag.

## 0. Honesty contract

1. **No synthetic data.** A missing input ⇒ `None` ⇒ UI renders "Data unavailable". Nothing is interpolated or invented in the coupling engine, atmosphere analysis, or alerts.
2. **No invented accuracy.** R² / MAE are reported only from held-out temporal test splits of real CPCB data (`/api/model-metrics`). They are never reinterpreted as "accuracy".
3. **Proxies are labelled.** PBL-height inversion, lapse-rate gradients, transport times, coupling potentials, and the meteorology–pollution surrogate all carry explicit "estimate / proxy / potential" wording.
4. **WRF-Chem is an honest adapter.** The dispersion core is a physics-informed numeric surrogate, not a compiled WRF-Chem run; genuine `wrfout_*.nc` files are consumed when present, otherwise `CtmUnavailable` is raised.

## 1. Input data and units

| Source | Table / channel | Fields used | Units |
|---|---|---|---|
| CPCB | `PollutionReading` | PM2.5, PM10, O3, NO2, SO2, CO | µg/m³ (CO in mg/m³ at input) |
| Open-Meteo | `WeatherReading` | temperature, humidity, pressure_msl, wind_speed, wind_direction, pbl_height, precipitation + vertical `temperature_{1000,925,850,700}hPa` | °C, %, hPa, m/s, ° (meteorological FROM), m, mm, °C |
| NASA FIRMS | `FireReading` | lat, lon, FRP | °, °, MW |
| CPCB | `Station` | name, latitude, longitude | °N/°E |

Wind direction is meteorological **FROM** direction throughout. Fire bearing
`_angle_between(station → fire)` is the great-circle bearing of the fire as seen
from the station.

## 2. AQI calculation (`backend/app/services/aqi_calculator.py`)

Sub-index for pollutant `p` with concentration `c` via CPCB breakpoint bands:

```
IAQI_p = ((IAQI_hi - IAQI_lo) / (BP_hi - BP_lo)) · (c − BP_lo) + IAQI_lo
```

`BP`/`IAQI` breakpoint tables are defined per pollutant in `IAQI_BREAKPOINTS`
(e.g. PM2.5: 0–30→0–50, 31–60→51–100, …, 251–500→401–500).

```
AQI      = max_p IAQI_p          (int; 500 if above the top band)
dominant = argmax_p IAQI_p
category = AQI_CATEGORIES band    Good ≤50, Satisfactory ≤100, Moderate ≤200,
                                  Poor ≤300, Very Poor ≤400, Severe ≤500
```

Missing pollutants are simply excluded; no pollutant present ⇒ `(0, "Unknown", "pm25")`.

## 3. Inversion detection (`ml/features/atmospheric_profile.py`)

Layer gradient (linear in pressure), per adjacent pair of stored levels
(1000/925/850/700 hPa, at least two required):

```
gradient [K/100 hPa] = (T_top − T_base) / (p_base − p_top) · 100
```

Temperature **increases with height** ⇒ positive gradient ⇒ inversion. The
strongest (most positive) layer governs:

| Gradient (K/100 hPa) | Category |
|---|---|
| > 1.5 | strong |
| > 0.6 | moderate |
| > 0.0 | weak (T increases with height) |
| ≤ 0.0 | none |

```
inversion_strength = clip((gmax − 0.0) / (4.0 − 0.0), 0, 1)      # ramp 0→1 across 0..4 K/100 hPa
```

When fewer than two pressure levels are stored, a **documented PBL-height proxy**
is used (`inversion_source = "pbl_proxy"`; never a surface-temperature threshold):

| PBL height | Category | Strength formula |
|---|---|---|
| < 150 m | strong | `(500 − pbl) / 500` clipped to 0..1 |
| 150–300 m | moderate | same |
| 300–500 m | weak | same |
| ≥ 500 m | none / 0 | — |

`combine_inversion(pbl, temps)` returns the lapse-rate result as authoritative
when `profile_available`, else the proxy. PBL dispersion condition:
`<150 m` TRAPPED, `<300 m` LIMITED, `<500 m` MODERATE, else GOOD.

### 3.1 Layer geometry (base/top height, thickness)

The strongest layer's depth is reported in metres as well as hPa. Heights use
real geopotential heights when the feed supplies them, otherwise the
first-order **hypsometric equation** integrated upward from 1000 hPa with each
layer's mean temperature:

```
dz [m]      = (R_d · T̄ [K] / g) · ln(p_base / p_top)
R_d         = 287.05 J kg⁻¹ K⁻¹      # ICAO dry-air specific gas constant
g           = 9.80665 m s⁻²
T̄           = (T_base + T_top) / 2   # layer mean, +273.15 to K
inversion_thickness_hpa = p_base − p_top
inversion_thickness_m    = z_top − z_base
```

A supplied geopotential height anchors that level, and the levels above it are
integrated from that anchor, so a partially populated profile still yields a
complete set. When the lowest stored level is not 1000 hPa the heights are
*depths* relative to that level rather than altitudes. Under the PBL proxy
there is no profile, so all geometry fields are `null` — never a fabricated
number.

### 3.2 Episode duration and persistence (`ml/features/inversion.py`)

`summarise_inversion_episode` measures the current episode over the stored
hourly history (`GET /api/inversion/{station}` reads a bounded 48 h window and
re-analyses the profile at every reading):

```
current_duration_h = (t_newest − t_onset) + median_sampling_interval
persistence_fraction = inverted samples / samples in trailing 24 h
```

A sampling gap larger than `EPISODE_GAP_TOLERANCE_H = 3 h` **breaks** the run and
the gap is *not* added to the duration: nothing was observed during it, so
counting it would claim persistence that was never measured. The response
carries `inversion_measured_window_h`, `inversion_episode_samples` and
`inversion_history_sufficient` alongside the values, so an episode is never
presented as longer than the archive behind it. No extrapolation past the
newest sample is performed.

## 4. Fire impact (`ml/features/fire_impact.py`)

Radius `MAX_DISTANCE_KM = 500` km; fires beyond are ignored.

```
haversine_distance   — great-circle km (Earth radius 6371 km)
bearing(station→fire) — atan2 great-circle bearing, [0, 360)°
is_upwind            — |angular diff(bearing − wind_dir)| < 180 parallel test ≤ 90°
alignment             = cos(radians(angle_diff))     (clipped ≥ 0)
weight_i              = FRP_i · alignment_i / (dist_i + 1)
raw_score             = Σ weight_i over fires ≤ 500 km
impact_score          = 1 / (1 + exp(−1.0 · (raw_score − 1.0)))     # logistic, clipped 0..1
nearest_fire_distance  = min dist (km)
wind_alignment_pct     = upwind_count / fire_count · 100
transport_time_hours   = nearest_km / (wind_speed_mps · 3.6)         # advective estimate, wind ≤ 0.5 m/s ⇒ None
transport_risk         = 0.4·impact + 0.3·proximity + 0.2·alignment + 0.1·time_term
                         proximity = 1 − min(1, nearest/500); time_term = 1 − min(1, t/24)
transport_risk_level   ≥0.7 severe | ≥0.4 high | ≥0.15 moderate | else low
stubble_impact_score   = impact · (0.5 + 0.5·alignment)
```

## 5. Coupling engine — nine features (`ml/features/coupling_engine.py`)

Normalized anchors: `WIND_REF = 7 m/s`, `PBL_REF = 1500 m`, `VENT_REF = 6000 m²/s`,
`PM25_GOOD = 35`, `PM25_SEVERE = 300 µg/m³`, `NO2_REF = 120 µg/m³`,
`T_WARM = 25 °C`, `T_HOT = 40 °C`, `UPWIND_SAT = 50`, `FIRE_DIST_REF = 500 km`.

Normalized primitives: `wind_norm = wind/7`, `pbl_norm = pbl/1500`,
`vent_norm = (wind·pbl)/6000` (0..1, clipped). All features are clipped to
0..1, rounded to 4 dp, and `None` — rendered "Data unavailable" — whenever a
required input is missing.

| # | Feature | Formula | Inputs required |
|---|---|---|---|
| 1 | `dispersion_potential` | `0.50·vent_norm + 0.20·pbl_norm + 0.30·(1 − inversion_strength)` | wind, pbl |
| 2 | `accumulation_potential` | `1 − dispersion_potential` | as #1 |
| 3 | `inversion_trapping_potential` | `0.50·inversion_strength + 0.50·(1 − pbl_norm)` | inversion_strength or pbl |
| 4 | `pollution_stagnation_index` | `0.40·(1 − wind_norm) + 0.40·(1 − pbl_norm) + 0.20·inversion_strength` | any of wind/pbl/inversion |
| 5 | `aerosol_accumulation_potential` | `0.50·pm25_norm + 0.50·accumulation_potential`, `pm25_norm = (pm25 − 35)/(300 − 35)` | pm25 + #2 |
| 6 | `fire_transport_influence` | `0.50·fire_impact_score + 0.25·proximity + 0.15·upwind_count/50 + 0.10·alignment` | fire inputs; `proximity = 1 − min(1, nearest/500)` |
| 7 | `regional_transport_potential` | `0.60·fire_transport_influence + 0.25·(1 − vent_norm) + 0.15·(1 − wind_norm)` | #6 or wind/vent |
| 8 | `ozone_photochemical_potential` | `0.50·(t−25)/(40−25) + 0.30·(1 − wind_norm) + 0.20·no2/120` — a *potential*, not a formation rate | temp, wind and/or no2 |
| 9 | `meteorology_pollution_interaction` | mean of available {#2, #4, #8, #6} — explicit **data-driven feedback surrogate**, not a physics chemistry loop | ≥1 of those features |

**Band labels** (and the `coupling_state` thresholds in the service): `Low` < 0.33,
`Moderate` < 0.66, else `High`.

**Persistence layer** (`backend/app/services/coupling_service.py`): every
`get_coupling_features` call write-through upserts one row per station in the
`coupling_states` table. `coupling_state ∈ {NONE, LOW, MODERATE, HIGH}` from the
band of feature #9; `coupling_domains` = joined present domains among
`aerosol / atmospheric / feedback / fire / ozone` (`"none"` if none);
`data_quality` = GOOD (9/9 available), PARTIAL (≥6), SPARSE (≥1), UNAVAILABLE (0).
`fire_transport_direction` = compass of `(wind_direction + 180)°` (the downwind
direction the smoke travels). Queryable via `GET /api/coupling/state`.

## 6. Atmosphere analysis (`backend/app/services/atmosphere_service.py`)

Wind/ventilation/PBL reference points and formula mirror §5. Indicators:

```
ventilation_coefficient [m²/s] = wind_speed · pbl_height
   poor <3000 | moderate 3000–6000 | good ≥6000   (vent_norm = vc/6000)
inversion gradient [K/100 hPa] = (T_top − T_base)/(p_base − p_top)·100   (§3)
dispersion_quality q           = 0.50·vent_norm + 0.20·pbl_norm + 0.30·(1 − inv_norm)
trapping_index                 = clip(0.70·(1 − q) + 0.30·pm25_norm, 0, 1)   (PM2.5 known)
                                 clip(1 − q, 0, 1)                            (PM2.5 unknown)
   bands <0.33 low | <0.55 moderate | <0.75 high | else severe
```

Provenance taxonomy: `OBSERVED`, `DERIVED` (deterministic formula), `ESTIMATED`
(PBL height is an Open-Meteo model field; inversion proxy).

## 7. Regional transport risk (`backend/app/services/transport_risk_service.py`)

Score 0–100 = 100 · available-forecast ratio of three components:

```
fire_component      = 0.50·impact + 0.25·proximity + 0.15·min(upwind/50,1) + 0.10·alignment
atmo_component      = 0.45·(1 − vent_norm) + 0.30·inversion_norm + 0.25·(1 − pbl_norm)
pollution_component = clip((pm25 − 35)/(300 − 35), 0, 1)
risk = (0.45·fire + 0.35·atmo + 0.20·pollution) / Σ(weights of available inputs)
```

Bands: `0–20 LOW | 21–40 MODERATE | 41–60 ELEVATED | 61–80 HIGH | 81–100 VERY HIGH`.
Regional wind is the circular mean of stored per-station winds (vector-averaged
in the blowing-TOWARD sense, converted back to FROM).

## 8. Alerts (`backend/app/services/alert_service.py`)

Deterministic rules; outputs sorted by severity rank
`WATCH(1) < ADVISORY(2) < WARNING(3) < SEVERE(4)`:

| Trigger | Condition | Level |
|---|---|---|
| Forecast AQI | ≥ 401 / ≥ 301 / ≥ 201 | SEVERE / WARNING / ADVISORY |
| Trend | `rising` / `falling` | WATCH |
| PM2.5 dominance | dominant == pm25 **and** AQI ≥ 201 | WATCH |
| Wind | < 2 m/s (poor dispersion) | WATCH |
| Wind | > 15 m/s (dust resuspension) | WATCH |
| Inversion | PBL < 150 m (strong trapping) | WARNING |
| Inversion | 150 ≤ PBL < 300 m | WATCH |
| Humidity | > 80 % (secondary aerosol potential) | ADVISORY |
| Precipitation | ≤ 0.5 mm **and** AQI ≥ 201 (no washout) | WATCH |
| Fires | fire_count > 50 **and** nearest < 300 km | WARNING |
| Fires | fire_count > 20 | WATCH |

## 9. Forecasting models

- **Persistence baseline** — `PM2.5(t+h) = PM2.5(t)` (sanity check for ML gains).
- **Random Forest** — nonlinear baseline with feature importances.
- **XGBoost (primary)** — gradient-boosted trees; one model per pollutant
  (PM2.5, PM10, O3, NO2, SO2, CO) per horizon (1, 6, 12, 24, 48, 72 h);
  trained via `ml/training/*`, served by `/api/forecast`. Trained artifacts are
  `.joblib` files; no student-hackathon weights are fabricated.
- **Input features** — lagged pollutants (t−1, t−3, t−6, t−12, t−24),
  meteorological fields, inversion metrics, fire-impact features, cyclic
  hour/day/month encodings (see `docs/methodology.md` §2).
- **Evaluation** — MAE, RMSE, R², MAPE on temporal holdouts; rolling-window
  backtesting; live metrics at `/api/model/performance` (and `/api/model-metrics`).

## 10. Numerical dispersion surrogate (`ml/features/dispersion_solver.py`)

Same 36×46 NCR grid (~2.2 km cells), explicit finite differences:

```
∂C/∂t = −u·∂C/∂x − v·∂C/∂y                    wind advection (upwind flux)
        + K_h·∇²C                              horizontal turbulent diffusion
        − (λ_dep + λ_wet)·C                    dry deposition + wet scavenging
        + E(x, y, t)                           fire (FRP) point + urban area sources
```

Numerics: upwind advection with CFL-safe `dt = 0.5·dx/max_wind` (≈277 s at
4 m/s), 5-point diffusive operator, reflecting interior / open outflow
boundaries, non-negativity preserving. Live PBL/wind/precipitation drive a
diurnal PBL cycle. This is a **physics-informed surrogate** — not compiled
WRF-Chem; the documented honest-scope tradeoff and the exact literal-WRF-Chem
requirements are in `docs/methodology.md` §9.1.

## 11. Two-way coupling loop (`ml/features/coupled_loop.py`)

Hourly online integration: (1) predict next-hour pollutants from current
features (meteorology → chemistry); (2) derive AOD, radiation transmittance,
PBL suppression and stability index from the fresh PM2.5 (chemistry →
meteorology); (3) correct PBL/temperature/inversion/stability; (4) persist the
corrected fields and re-enter (two-way feedback). `/api/forecast/coupled`
returns both the coupled and the uncoupled series plus the hour-by-hour
`feedback_path`.

### 11.1 Coupling ablation (`ml/evaluation/ablation.py`)

`run_coupling_ablation` measures whether the six coupling features actually
improve forecasts. The same model type and hyper-parameters are fitted twice on
the **same** chronological split (`chronological_split_by_time`, 60/15/25),
selecting on validation and scoring on the held-out test set; the only
difference between arms is whether the feature group
(`aod_est`, `radiation_transmittance`, `pbl_suppression_factor`,
`corrected_pbl_height`, `stability_coupling_index`, `feedback_multiplier`) is in
the design matrix. Because the split and evaluation rows are identical, the
metric delta is attributable to the feature group rather than to a re-rolled
split.

```
Δ = metric(with coupling) − metric(without coupling)
```

Sign is read against the metric: for MAE/RMSE (lower is better) a **negative**
Δ means the features helped; for R² (higher is better) a **positive** Δ means
they helped. Differences below `MATERIALITY_FRACTION = 0.5%` of the
without-coupling metric are reported as **no material change** rather than as a
positive contribution, because two fits of the same model differ slightly for
reasons unrelated to the features.

Measured (RandomForest, pm2.5, 21,299-row test split, reproduce with
`python scripts/run_coupling_ablation.py`):

| Horizon | MAE with | MAE without | Δ MAE | Δ % | R² Δ | Verdict |
|---------|----------|-------------|-------|-----|------|---------|
| t+6h  | 34.729 | 34.770 | −0.041 | −0.12% | −0.0005 | no material change |
| t+24h | 41.364 | 42.510 | −1.146 | −2.70% | +0.0307 | material improvement |

The experiment trains throwaway models in memory and writes only a JSON report
(`reports/coupling_ablation_pm25.json`); it does not modify the shipped models
or the live API. Single seed and one model type: the t+6 h null result is within
fit-to-fit variation and would need a multi-seed run to be stated as firmly as
t+24 h.

## 12. Spatial forecast (`backend/app/services/grid_service.py`)

Per-station coupled forecasts are interpolated by inverse-distance weighting
onto a ~2.2 km grid over (28.2–28.9°N, 76.6–77.5°E), optionally advectively
shifted downwind, and served as GeoJSON-style cells via `/api/grid/forecast`.

## 13. Constants reference (shared anchors)

| Constant | Value | Meaning |
|---|---|---|
| `WIND_REF_MPS` | 7.0 | wind speed = "full" horizontal dispersion |
| `PBL_REF_M` | 1500.0 | mixing depth = "full" vertical dispersion |
| `VENT_REF_M2S` | 6000.0 | ventilation coefficient reference |
| `PBL thresholds` | 150 / 300 / 500 m | strong / moderate / weak trapping |
| `PM25_REF_GOOD` / `PM25_REF_SEVERE` | 35 / 300 µg/m³ | CPCB 24h satisfactory / emergency |
| `MAX_DISTANCE_KM` | 500 km | regional fire radius (also UI influence ring) |
| `UPWIND_ANGLE_DEG` | ±90° | upwind sector (UI geo.ts matches §4) |
| `UPWIND_COUNT_SATURATION` | 50 | upwind fires saturating the count term |
| `STRONG/MODERATE/WEAK INV K` | 1.5 / 0.6 / 0.0 | lapse-rate inversion categories |
| `TRANSPORT_TIME_REF_H` | 24 | advective arrival time reference |

## 14. Related docs

- `docs/methodology.md` — high-level methodology narrative (§0–§11 above are its formal specification).
- `docs/SIH26082_IMPLEMENTATION_AUDIT.md` — requirement-by-requirement compliance and honest-disclosure statements.
- `docs/architecture.md`, `docs/api.md` — system structure and endpoints.