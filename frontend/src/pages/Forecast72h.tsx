import { useCallback, useEffect, useState } from 'react'
import { Download, Info } from 'lucide-react'
import { getStations, getForecast, getForecastContext, getForecastExportUrl } from '../api/client'
import PageHeader from '../components/PageHeader'
import LoadingState from '../components/LoadingState'
import ErrorState from '../components/ErrorState'
import EmptyState from '../components/EmptyState'
import ForecastRiskBand from '../components/ForecastRiskBand'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { fmt } from '../lib/aqi'
import { CHART, pollutantColor } from '../lib/theme'
import type { Station, ForecastPoint, ForecastContextResponse } from '../types'

const tabs = [
  { key: 'aqi_pred', label: 'AQI' },
  { key: 'pm25_pred', label: 'PM2.5' },
  { key: 'pm10_pred', label: 'PM10' },
  { key: 'o3_pred', label: 'O₃' },
  { key: 'no2_pred', label: 'NO₂' },
]

const ctxNum = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? '--' : v.toFixed(digits))

/**
 * Colour a 0..1 tendency value by severity.
 *
 * `higherIsBetter` is needed because `dispersion_potential` is the complement of
 * the rest: the coupling engine defines it as how readily the atmosphere
 * disperses pollutants (ml/features/coupling_engine.py), and
 * `accumulation_potential` as `1 - dispersion_potential`. So a high dispersion
 * value is the favourable end, while high values in the other five columns are
 * the concerning end. Applying one "high = red" rule to all six made
 * dispersion_potential red precisely when dispersion was strong.
 */
function bandClass(v: number | null | undefined, higherIsBetter = false): string {
  if (v === null || v === undefined) return 'text-slate-400'
  const severity = higherIsBetter ? 1 - v : v
  if (severity < 0.33) return 'text-emerald-600'
  if (severity < 0.66) return 'text-amber-600'
  return 'text-rose-600'
}

export default function Forecast72h() {
  const [stations, setStations] = useState<Station[]>([])
  const [selectedStation, setSelectedStation] = useState('Anand Vihar')
  const [forecast, setForecast] = useState<ForecastPoint[]>([])
  const [context, setContext] = useState<ForecastContextResponse | null>(null)
  const [pollutant, setPollutant] = useState<string>('aqi_pred')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
    getForecast(selectedStation)
      .then((r) => setForecast(r.data))
      .catch(() => setError('Failed to load forecast data'))
      .finally(() => setLoading(false))
    getForecastContext(selectedStation)
      .then((r) => setContext(r.data))
      .catch(() => setContext(null))
  }, [selectedStation])

  useEffect(() => {
    getStations().then((r) => setStations(r.data)).catch(() => {})
  }, [])

  useEffect(() => { load() }, [load])

  const chartData = forecast.map((f) => ({
    time: `+${f.horizon_hours}h`,
    value: f[pollutant as keyof ForecastPoint] ?? null,
    category: f.aqi_category,
  }))

  return (
    <div className="space-y-6">
      <PageHeader
        title="72-Hour Forecast"
        subtitle="Hourly ML forecasts for every NCR monitoring station, with CPCB risk categorisation and per-horizon atmospheric context"
        breadcrumbs={[{ label: 'Dashboard', to: '/dashboard' }, { label: 'Forecast' }]}
        lastUpdated={forecast[0]?.timestamp}
        actions={
          <div className="flex items-center gap-2">
            <select
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
              className="select"
              aria-label="Select station"
            >
              {stations.map((s) => (
                <option key={s.id} value={s.name}>{s.name}</option>
              ))}
            </select>
            <a
              href={getForecastExportUrl(selectedStation, 72)}
              className="btn-outline inline-flex items-center gap-1.5"
            >
              <Download className="h-4 w-4" aria-hidden="true" /> CSV
            </a>
          </div>
        }
      />

      {error && <ErrorState title="Forecast unavailable" message={error} onRetry={load} />}

      <ForecastRiskBand forecast={forecast} />

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Pollutant selection">
        {tabs.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={pollutant === t.key}
            onClick={() => setPollutant(t.key)}
            className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
              pollutant === t.key
                ? 'border-inst-700 bg-inst-700 text-white'
                : 'border-slate-300 bg-white text-slate-600 hover:border-inst-300 hover:text-inst-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <LoadingState label="Loading forecast" rows={2} />
      ) : error ? null : forecast.length ? (
        <>
          <div className="card">
            <h2 className="mb-3 text-base font-bold text-slate-900">
              {tabs.find((t) => t.key === pollutant)?.label} forecast — {selectedStation}
            </h2>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData} margin={{ left: -12, right: 12 }} accessibilityLayer role="img" aria-label={`${tabs.find((t) => t.key === pollutant)?.label} forecast for ${selectedStation} across 72 hours`}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
                <XAxis dataKey="time" stroke={CHART.axis} fontSize={11} />
                <YAxis stroke={CHART.axis} fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8 }}
                  labelStyle={{ color: CHART.tooltipLabel }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="value" stroke={pollutantColor(pollutant)} strokeWidth={2.5} dot={false} name={pollutant.replace('_pred', '').toUpperCase()} />
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
              Point forecast from the persisted 72-hour ML run. Uncertainty grows with horizon —
              the +72h point is materially less certain than +1h. Distribution-free split-conformal
              prediction intervals for the direct PM2.5 engine are shown on the Dashboard
              (&lsquo;/api/forecast/pm25&rsquo;). This horizon-tab view is a point series only.
            </p>
          </div>

          <div className="card">
            <h2 className="card-header">Forecast table</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">
                  72-hour pollutant and AQI forecast for {selectedStation}.
                </caption>
                <thead>
                  <tr className="border-b border-slate-200 text-left text-slate-500">
                    <th className="py-2">Horizon</th>
                    <th className="py-2 text-right">PM2.5</th>
                    <th className="py-2 text-right">PM10</th>
                    <th className="py-2 text-right">O₃</th>
                    <th className="py-2 text-right">NO₂</th>
                    <th className="py-2 text-right">AQI</th>
                    <th className="py-2 text-right">Category</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.map((f) => (
                    <tr key={f.horizon_hours} className="border-b border-slate-100">
                      <td className="py-2 font-medium text-slate-900">+{f.horizon_hours}h</td>
                      <td className="py-2 text-right tabular-nums text-slate-700">{fmt(f.pm25_pred, 1)}</td>
                      <td className="py-2 text-right tabular-nums text-slate-700">{fmt(f.pm10_pred, 1)}</td>
                      <td className="py-2 text-right tabular-nums text-slate-700">{fmt(f.o3_pred, 1)}</td>
                      <td className="py-2 text-right tabular-nums text-slate-700">{fmt(f.no2_pred, 1)}</td>
                      <td className="py-2 text-right font-bold tabular-nums text-slate-900">{f.aqi_pred ?? '--'}</td>
                      <td className="py-2 text-right"><span className="capitalize text-slate-700">{f.aqi_category}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h2 className="card-header flex items-center gap-2">
              Atmospheric context
              <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">
                <Info className="h-3 w-3" aria-hidden="true" />
                real stored weather
              </span>
            </h2>
            <p className="mb-3 text-xs text-slate-500">
              Nearest stored weather observation per horizon, feeding the coupling-engine
              dispersion / accumulation / stagnant / fire-transport tendencies. A dash
              (&lsquo;--&rsquo;) means the stored observation is missing for that horizon
              — it is never filled with a synthetic value.
            </p>
            {context && context.horizons.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <caption className="sr-only">
                    Per-horizon atmospheric context: temperature, humidity, pressure, wind, PBL,
                    inversion and coupling tendencies for {selectedStation}.
                  </caption>
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-slate-500">
                      <th className="py-2">Horizon</th>
                      <th className="py-2 text-right">Temp °C</th>
                      <th className="py-2 text-right">Hum %</th>
                      <th className="py-2 text-right">Pressure hPa</th>
                      <th className="py-2 text-right">Wind m/s</th>
                      <th className="py-2 text-right">PBL m</th>
                      <th className="py-2">Inversion</th>
                      <th className="py-2 text-right">Dispersion</th>
                      <th className="py-2 text-right">Accumulation</th>
                      <th className="py-2 text-right">Stagnation</th>
                      <th className="py-2 text-right">Fire transport</th>
                      <th className="py-2 text-right">Regional transport</th>
                      <th className="py-2 text-right">O₃ potential</th>
                    </tr>
                  </thead>
                  <tbody>
                    {context.horizons.map((h) => (
                      <tr key={h.horizon_hours} className="border-b border-slate-100">
                        <td className="py-1.5 font-medium text-slate-900">+{h.horizon_hours}h</td>
                        <td className="py-1.5 text-right tabular-nums text-slate-700">{ctxNum(h.temperature_c)}</td>
                        <td className="py-1.5 text-right tabular-nums text-slate-700">{ctxNum(h.humidity_pct, 0)}</td>
                        <td className="py-1.5 text-right tabular-nums text-slate-700">{ctxNum(h.pressure_hpa, 0)}</td>
                        <td className="py-1.5 text-right tabular-nums text-slate-700">{ctxNum(h.wind_speed_mps)}</td>
                        <td className="py-1.5 text-right tabular-nums text-slate-700">{ctxNum(h.pbl_height_m, 0)}</td>
                        <td className="py-1.5 capitalize text-slate-600">
                          {h.inversion_category ?? '--'}
                          {h.inversion_source === 'lapse_rate' ? ' • lapse-rate' : h.inversion_source ? ' • proxy' : ''}
                        </td>
                        <td className={`py-1.5 text-right font-semibold tabular-nums ${bandClass(h.dispersion_potential, true)}`}>{ctxNum(h.dispersion_potential, 2)}</td>
                        <td className={`py-1.5 text-right font-semibold tabular-nums ${bandClass(h.accumulation_potential)}`}>{ctxNum(h.accumulation_potential, 2)}</td>
                        <td className={`py-1.5 text-right tabular-nums ${bandClass(h.pollution_stagnation_index)}`}>{ctxNum(h.pollution_stagnation_index, 2)}</td>
                        <td className={`py-1.5 text-right tabular-nums ${bandClass(h.fire_transport_influence)}`}>{ctxNum(h.fire_transport_influence, 2)}</td>
                        <td className={`py-1.5 text-right tabular-nums ${bandClass(h.regional_transport_potential)}`}>{ctxNum(h.regional_transport_potential, 2)}</td>
                        <td className={`py-1.5 text-right tabular-nums ${bandClass(h.ozone_photochemical_potential)}`}>{ctxNum(h.ozone_photochemical_potential, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title="Atmospheric context unavailable" hint="No stored weather rows were found for this station's 72-hour window." />
            )}
          </div>
        </>
      ) : (
        <EmptyState title="No forecast stored" hint="Persisted forecasts for this station have not been generated yet." />
      )}
    </div>
  )
}