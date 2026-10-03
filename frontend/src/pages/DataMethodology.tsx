import {
  Database,
  Satellite,
  CloudSun,
  BarChart3,
  Layers,
  FlaskConical,
  ShieldAlert,
} from 'lucide-react'
import PageHeader from '../components/PageHeader'

const KINDS = [
  { icon: Satellite, label: 'Observed', color: 'text-emerald-700 bg-emerald-50 border-emerald-200', note: 'Direct measurements — CPCB stations, FIRMS hotspots' },
  { icon: Database, label: 'Reanalysis', color: 'text-sky-700 bg-sky-50 border-sky-200', note: 'ERA5 (offline/archival feature pipeline)' },
  { icon: CloudSun, label: 'Forecast', color: 'text-violet-700 bg-violet-50 border-violet-200', note: 'NWP fields — Open-Meteo surface + pressure levels' },
  { icon: Layers, label: 'Derived', color: 'text-amber-700 bg-amber-50 border-amber-200', note: 'Computed from stored inputs — AQI, inversion gradient, PBL class, ventilation' },
  { icon: BarChart3, label: 'ML predicted', color: 'text-inst-700 bg-inst-50 border-inst-200', note: 'Model output on feature rows — 72 h pollutant forecasts, with uncertainty bands' },
  { icon: FlaskConical, label: 'Scenario simulated', color: 'text-fuchsia-700 bg-fuchsia-50 border-fuchsia-200', note: 'What-if only — never shown as a real forecast' },
]

const SOURCES = [
  {
    name: 'CPCB / data.gov.in',
    provides: 'Live + archived AQI and pollutant concentrations (PM2.5, PM10, O3, NO2, SO2, CO) for Delhi NCR monitoring stations',
    freq: 'hourly slot (CKAN feed)',
    spatial: 'Station points (city-level NCR)',
    temporal: 'Hourly',
    limits: 'Station coverage varies; gaps render as "Data unavailable"; observational values are never overwritten by the forecast pipeline',
  },
  {
    name: 'Open-Meteo',
    provides: 'Surface weather (T, humidity, pressure, wind speed/direction, precipitation, cloud) + planetary boundary-layer height + pressure-level temperature/geopotential at 1000/925/850/700 hPa',
    freq: 'hourly (NWP)',
    spatial: 'Gridded, nearest-cell per station',
    temporal: 'Hourly, 7-day horizon',
    limits: 'Analysis fields, not in-situ instruments; pressure levels may be NULL shortly after release; PBLH is a model field (interpretation thresholds documented as heuristics)',
  },
  {
    name: 'NASA FIRMS (VIIRS)',
    provides: 'Active-fire detections: lat/lon, detection time, FRP (MW), confidence — upwind ≥ 40° exclusion, 500 km radius around Delhi NCR centroid',
    freq: '≈ 3 h per overpass',
    spatial: '375 m pixels',
    temporal: 'Per-overpass',
    limits: 'Satellite-only radiance detection; low-FRP or obscured fires can be missed; attribution to specific farms is never claimed',
  },
  {
    name: 'ERA5 (Copernicus CDS)',
    provides: 'Single-level atmospheric reanalysis (offline feature pipeline / training window)',
    freq: 'archival',
    spatial: '≈ 31 km (0.25°)',
    temporal: 'Hourly (archival)',
    limits: 'Not a live feed; CDS credentials required (honest `reason` when absent). Multi-level live ingestion is the documented production upgrade',
  },
  {
    name: 'IMD (opt-in)',
    provides: 'Optional observed weather ingest when credentials are configured',
    freq: 'n/a (opt-in)',
    spatial: 'Station points',
    temporal: 'Hourly when active',
    limits: 'Credential-gated; absent → clearly reported, never fabricated',
  },
]

const STAGES = [
  { step: 'Data sources', detail: 'CPCB + Open-Meteo + FIRMS (+ optional ERA5/IMD) with timestamps normalised to UTC' },
  { step: 'Quality + alignment', detail: 'Missing-value policy, outlier flagging (IQR), station consistency, weather↔pollution timestamp matching, fire-data temporal matching' },
  { step: 'Atmospheric state engine', detail: 'PBLH, lapse-rate inversion (+ PBL proxy fallback), wind vector, stability, humidity/pressure/temperature' },
  { step: 'Coupling engine', detail: 'Nine features — dispersion/accumulation/inversion/trapping/stagnation/aerosol/fire/regional/O3 potential + meteorology–pollution feedback surrogate' },
  { step: 'ML / hybrid forecast engine', detail: 'Persistence, Random Forest, XGBoost (serving) per pollutant × horizon on chronological splits; GRU evaluated; direct PM2.5 engine with conformal intervals' },
  { step: 'Indian AQI', detail: 'CPCB breakpoint sub-indices from predicted concentrations → AQI + category + dominant pollutant' },
  { step: '72 h spatial outlook', detail: 'Per-station forecasts → 2×2 km IDW grid + numerical dispersion field' },
  { step: 'Alerts + explanation + dashboard', detail: 'Deterministic alert rules, SHAP-driven narrative, operational dashboard' },
]

const METHODS = [
  {
    title: 'AQI (Indian / CPCB)',
    body: 'Sub-index per pollutant via breakpoint interpolation (IAQI), AQI = maximum sub-index, dominant pollutant from the max. Concentrations are predicted first; AQI is never predicted directly.',
  },
  {
    title: 'Inversion',
    body: 'Vertical temperature gradient dT/dp × 100 (K/100 hPa) across stored 1000/925/850/700 hPa layers. T increasing with height ⇒ inversion; categories NO/WEAK/MODERATE/STRONG at 0.6/1.5 K per 100 hPa; strength ramp 0..1. When fewer than two pressure levels exist a clearly-labelled PBL-height proxy is used with an "estimate limited" note.',
  },
  {
    title: 'PBL height',
    body: 'Real NWP field (Open-Meteo boundary_layer_height). Classified LOW/ MODERATE/HIGH (150/300/500 m heuristics); ventilation potential = wind speed × PBLH; trend over last 24 h; anomaly vs climatological band; caution — no direct causality, supporting variables shown.',
  },
  {
    title: 'Wind / dispersion',
    body: 'u/v components from speed+direction; dominant transport direction; dispersion potential from wind + PBLH + stability (+ rain washout where present); numerical advection–diffusion–deposition solver on the ~2.2 km NCR grid (CFL-safe, non-negative). Surrogate — not WRF-Chem chemistry.',
  },
  {
    title: 'Two-way coupling',
    body: 'met→chem: features feed all forecasters. chem→met: AOD proxy → radiation transmittance → PBL suppression → stability-coupling index → feedback multiplier inside the online loop (coupled vs uncoupled series returned). This is a data-driven surrogate, never physical WRF-Chem.',
  },
  {
    title: 'Fire / stubble influence',
    body: 'Live FIRMS detections ≤ 500 km, upwind ±90° of transport direction; FRP-weighted impact; transport time = nearest distance ÷ wind speed; transport-risk weighting (0.4/0.3/0.2/0.1); stubble-impact score. Only real FIRMS observations feed these metrics. Rows marked synthetic are 2023–24 simulated training history (clearly labelled, excluded from all operational metrics; visible at most as a grey "simulated" map overlay when explicitly toggled on). Wording is "estimated fire-related transport influence".',
  },
  {
    title: 'Uncertainty & validation',
    body: 'Split-conformal intervals calibrated on validation, measured coverage reported on test (never claimed beyond measured). MAE/RMSE/R² from chronological held-out splits by pollutant × horizon × model. R² is goodness-of-fit, not "accuracy".',
  },
]

export default function DataMethodologyPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Data & Methodology"
        subtitle="Data provenance and the scientific basis of every value shown on the dashboard."
        breadcrumbs={[{ label: 'Tools' }, { label: 'Data & Methodology' }]}
      />

      <div className="card">
        <h3 className="card-header">What you are looking at — value kinds</h3>
        <p className="text-sm text-slate-600">
          Every dashboard value falls into exactly one of these classes. Observed pollution is
          never overwritten; missing data renders as &ldquo;Data unavailable&rdquo;, never as a filler value.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {KINDS.map(({ icon: Icon, label, color, note }) => (
            <div key={label} className={`rounded-xl border p-4 ${color.split(' ')[1]} ${color.split(' ')[2]}`}>
              <div className="flex items-center gap-2">
                <Icon className={`h-5 w-5 ${color.split(' ')[0]}`} aria-hidden="true" />
                <span className="font-semibold text-slate-900">{label}</span>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">{note}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 className="card-header">Pipeline</h3>
        <ol className="space-y-2">
          {STAGES.map((s, i) => (
            <li key={s.step} className="flex items-start gap-3 text-sm">
              <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-inst-700 text-xs font-bold text-white">
                {i + 1}
              </span>
              <div>
                <span className="font-semibold text-slate-900">{s.step}</span>
                <span className="block text-xs leading-relaxed text-slate-600">{s.detail}</span>
              </div>
            </li>
          ))}
        </ol>
      </div>

      <div className="card">
        <h3 className="card-header">Data sources</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 pr-3">Source</th>
                <th className="py-2 pr-3">Provides</th>
                <th className="py-2 pr-3">Update</th>
                <th className="py-2 pr-3">Spatial</th>
                <th className="py-2 pr-3">Temporal</th>
                <th className="py-2">Limitations</th>
              </tr>
            </thead>
            <tbody>
              {SOURCES.map((s) => (
                <tr key={s.name} className="border-b border-slate-100 align-top last:border-0">
                  <td className="py-2.5 pr-3 font-semibold text-slate-900">{s.name}</td>
                  <td className="py-2.5 pr-3 text-slate-700">{s.provides}</td>
                  <td className="py-2.5 pr-3 text-slate-700">{s.freq}</td>
                  <td className="py-2.5 pr-3 text-slate-700">{s.spatial}</td>
                  <td className="py-2.5 pr-3 text-slate-700">{s.temporal}</td>
                  <td className="py-2.5 text-xs text-slate-500">{s.limits}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3 className="card-header">Scientific methodology</h3>
        <div className="grid gap-4 md:grid-cols-2">
          {METHODS.map((m) => (
            <div key={m.title} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-sm font-bold text-slate-900">{m.title}</h4>
              <p className="mt-1 text-xs leading-relaxed text-slate-600">{m.body}</p>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-slate-500">
          Exact formulas, constants, units and assumptions: <code>docs/SCIENTIFIC_METHODOLOGY.md</code>.
          Trade-off against full WRF-Chem: <code>docs/methodology.md §9</code> · adapter contract:{' '}
          <code>docs/wrfchem_adapter.md</code> · traceability: <code>docs/SIH26082_TRACEABILITY.md</code> ·{' '}
          final audit: <code>docs/SIH26082_FINAL_AUDIT.md</code>.
        </p>
      </div>

      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        <div>
          <p className="font-semibold">Honesty commitment</p>
          <p className="mt-1 text-xs leading-relaxed">
            This prototype never claims WRF-Chem execution, plumes guaranteed to arrive, or physical
            two-way coupling. Coupling, dispersion and fire-transport outputs are transparent
            data-driven surrogates/estimates with the equations documented. Metrics reflect the most
            recent chronological evaluation; R² is goodness-of-fit, not accuracy. Prototype developed
            for Smart India Hackathon 2026 — SIH26082. Not an official Government of India service.
          </p>
        </div>
      </div>
    </div>
  )
}