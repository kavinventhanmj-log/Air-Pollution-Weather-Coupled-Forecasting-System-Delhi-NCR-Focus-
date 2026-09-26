# SIH26082 — Problem Statement Mapping

Official problem statement (SIH26082, NCMRWF / Ministry of Earth Sciences):
*"AI-Powered 72-Hour Air Quality and Pollution-Plume Forecasting for Delhi NCR"*.
This table maps every explicit requirement to its implementation and where to
verify it.

| # | Requirement (PS) | Implementation | Proof |
|---|------------------|----------------|-------|
| 1 | Couple weather forecasts with pollution chemistry; two-way interlink | `ml/features/coupling.py` (aerosol AOD → solar attenuation → PBL suppression → stability feedback) + `ml/features/coupled_loop.py` (online time-stepped loop); numerical core feedback in `dispersion_solver.advance()` | `GET /api/coupling/{station}`, `POST /api/forecast/coupled`, per-frame coupling fields in `GET /api/dispersion/forecast`; tests: `TestCouplingFeedback`, `TestCoupledForecastLoop` |
| 2 | 72-hour AQI prediction | Persistence/RF/XGB forecasters, 6 pollutants × horizons {1,6,12,24,48,72} | `POST /api/forecast/generate`, `GET /api/forecast/{station}`; `TestForecastPipeline` |
| 3 | Predict how stubble-burning plumes disperse under prevailing weather | Numerical advection–diffusion–deposition solver with FRP fire point sources, wind/PBL/rain forcing, lateral inflow BC | `GET /api/dispersion/forecast`; `ml/features/dispersion_solver.py`; `TestDispersionForecast` |
| 4 | Inversion impact on pollutant trapping | `ml/features/inversion.py`; stability index couples inversion to PBL | `GET /api/inversion/{station}`; `TestInversion*` |
| 5 | NCR-wide (spatial) coverage, not just station points | IDW gridded surface (~2.2 km) + numerical field | `GET /api/grid/forecast`, `/spatial` page; `TestGridForecast` |
| 6 | All criteria pollutants | PM2.5, PM10, O3, NO2, SO2, CO | `predict_pollutants`; `TestForecastCompletePollutantSet` |
| 7 | Actionable alerting / public-health interpretation | Alert engine + explainability (SHAP) | `GET /api/alerts`, `GET /api/explanation/{station}`; `TestAlerts`, `TestExplainability` |
| 8 | Use real monitoring + fire satellite data | CPCB readings, Open-Meteo weather, NASA FIRMS fires in DB + download scripts | `scripts/download_*.py`, `data/`, `backend/app/models` |

## Honest scope note

The PS names "WRF-Chem or similar coupled frameworks". Shipping a compiled
Fortran WRF-Chem would require an HPC cluster, GOCART/MOZART chemistry and full
emission inventories — outside a student hackathon. AeroCast-NCR implements a
**physics-informed numerical surrogate** (finite-difference transport core with
explicit two-way coupling) that satisfies the functional outcomes (plume
dispersion prediction, meteorological interlink, 72 h AQI, NCR-wide mapping).
This is documented transparently in `docs/methodology.md` §9.

## Regression verification

```bash
python -m pytest backend/tests -q    # 748 tests green
cd frontend && npm run build          # type-checked production build
```