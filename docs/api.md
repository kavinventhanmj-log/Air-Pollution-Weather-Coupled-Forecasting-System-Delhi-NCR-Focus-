# API Reference

Base URL (dev): `http://localhost:8000/api`
The Vite dev server proxies `/api` to the backend on port 8000.

All endpoints return JSON. Errors use FastAPI's standard `{"detail": ...}` shape.

Authoritative definitions live in `backend/app/api/*.py` and
`backend/app/schemas/schemas.py`.

## System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe: status, service, version |
| GET | `/api/data-quality` | Row counts + missing-value audit per table |

## Stations
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stations` | List NCR monitoring stations (lat/lon/city) |

## Current conditions
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/current/{station_name}` | Latest pollutant readings + AQI sub-indices |

## Weather & atmosphere
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/weather/{station_name}` | Latest weather (wind, PBL, precip, temp…) |
| GET | `/api/weather/{station_name}/history?hours=24` | Historical weather series |
| GET | `/api/inversion/{station_name}` | Inversion detection + trapping risk |

## Forecasting
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/forecast/generate` | `{station_name}` → 6-horizon pollutant forecasts |
| GET | `/api/forecast/{station_name}?hours=72` | Persisted forecasts for a station |
| GET | `/api/forecast/ncr` | Per-station forecasts across NCR |
| GET | `/api/forecast/comparison/{station_name}?hours=72` | Predicted vs actual series |
| POST | `/api/forecast/coupled` | `{station_name, horizons?}` → two-way coupled loop |

## Spatial & dispersion
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/grid/forecast?horizon_hours=24` | Statistical IDW AQI surface (cells) |
| GET | `/api/grid/overview` | Grid coverage summary |
| GET | `/api/dispersion/forecast?horizon_hours=72&start_hour=8` | Numerical advection–diffusion run; hourly frames + coupling diagnostics |
| GET | `/api/dispersion/forecast?horizon_hours=72&frame_hours=6,12,24,48,72` | Same run, but only the listed hours get their per-cell AQI grid serialised. The solver still integrates the whole horizon; this bounds the response, which for 72 h is otherwise 72 × 1575 cells (~7 MB). `frame_hours_available` in the body lists every hour the run produced. Omit the parameter for all frames. |

## Fire & plume
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fire/hotspots` | Current NASA FIRMS hotspot aggregate (count, FRP, nearest station, upwind share) |
| GET | `/api/fires/latest` | Latest per-fire activity rows (satellite, FRP, confidence, day/night) |
| GET | `/api/fire/transport?station_name=…` | FIRMS → station advective transport estimate (alignment, transport time, transport risk, stubble impact) |
| GET | `/api/plume-risk` | Composite plume-risk score + contributors |
| GET | `/api/fire-activity` | Latest FIRMS fire-aggregation summary |
| GET | `/api/fire/{station_name}` | Fire context for a station |

## Inversion / atmosphere (vertical, pressure-level)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/inversion/{station_name}` | Lapse-rate-based inversion detection, strength, base/top pressure, category + PBL/dispersion conditions |
| GET | `/api/atmosphere/current` | Current vertical-atmosphere profile summary (pressure levels, inversion, PBL, dispersion) |
| GET | `/api/transport-risk/current` | Current transport-risk aggregate per station |
| GET | `/api/coupling/{station_name}` | Two-way weather–chemistry coupling diagnostics + narrative |

## Events & scenarios
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events/current` | Pollution events (surge / relief / sustained high-risk episode) with confidence + atmospheric contributors |
| POST | `/api/scenario/analysis` | Read-only what-if scenario engine (wind / PBL / fire / inversion perturbations) |

## Explainability, alerts, metrics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/explanation/{station_name}` | Top features + natural-language explanation |
| GET | `/api/forecast/pm25/explanation?station_name=…` | PM2.5-specific feature explanation |
| GET | `/api/model/performance` | Latest persisted model-performance comparison |
| GET | `/api/model/metrics` | Persisted MAE/RMSE/R² per model |
| POST | `/api/model/metrics` | Save a model metric row |
| GET | `/api/alerts` | Generated alert feed |

## PM2.5 forecast engine (Phase-3)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/forecast/pm25?station_name=…&hours=…` | Direct multi-horizon PM2.5 forecast (serving model: XGBoost + conformal intervals; hours ∈ {1,6,12,24,48,72}, default 72) |
| GET | `/api/forecast/pm25/model-card` | Deployed model card (features, horizons, uncertainty metadata) |
| GET | `/api/forecast/{forecast_id}/explanation` | Forecast-specific SHAP explanation |

## Authentication (additive portal layer — data APIs stay public)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/login` | `{email, password}` → `{access_token, token_type, user, expires_in}` (HS256 JWT, stdlib-signed) |
| GET | `/api/auth/me` | Current user profile — requires `Authorization: Bearer <token>` |
| POST | `/api/auth/logout` | Idempotent logout (stateless tokens; returns user), requires bearer token |
| GET | `/api/auth/demo` | Env-configured demo credentials for the login page hint (dev only) |

All forecasting / observation endpoints remain public so scripts and tests keep working unchanged;
route protection is enforced client-side on the protected portal pages.

## Notes
- Horizon values: `[1, 6, 12, 24, 48, 72]` hours.
- AQI categories: `Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`.
- Interactive docs: `http://localhost:8000/docs` (Swagger UI).
- `GET /api/system` reports run mode and integration status, including a
  `prewarm` block (`enabled`, `state`, `entries_warmed`, `entries_failed`,
  `seconds`) for the `CONTROL_ROOM_PREWARM` cache sweep. On a scale-to-zero host
  this is how you tell a warm instance from one whose caches are still cold.
