import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  AlertTriangle,
  Flame,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'
import {
  getStations,
  getCurrentAQI,
  getForecast,
  getPm25Forecast,
  getAtmosphereCurrent,
  getFireActivity,
  getFireHotspots,
  getTransportRisk,
  getModelPerformance,
  getPollutionLatest,
  getPollutionHistory,
  getGrapCurrent,
  getSummary,
  getAlerts,
  getDataQuality,
  ensureWarm,
} from '../api/client'
import PageHeader from '../components/PageHeader'
import AQIBadge from '../components/AQIBadge'
import LoadingState from '../components/LoadingState'
import ErrorState from '../components/ErrorState'
import EmptyState from '../components/EmptyState'
import StationMap from '../components/StationMap'
import AtmosphericState from '../components/AtmosphericState'
import AtmosphericStory from '../components/AtmosphericStory'
import DispersionMeter from '../components/DispersionMeter'
import ForecastRiskBand from '../components/ForecastRiskBand'
import TransportChain from '../components/TransportChain'
import AlertsSection from '../components/AlertsSection'
import ForecastReasonPanel from '../components/ForecastReasonPanel'
import { useIntervalRefresh } from '../hooks/useIntervalRefresh'
import { aqiStyle, fmt, readableOnHex, tsFmt } from '../lib/aqi'
import { CHART } from '../lib/theme'
import {
  ComposedChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot, Legend,
} from 'recharts'
import type {
  Station, CurrentAQI, ForecastPoint, Pm25ForecastResponse, Pm25ForecastPoint,
  AtmosphereCurrentResponse, FireActivity, FireHotspot, TransportRiskResponse,
  ModelPerformanceResponse, PollutionReading, Alert, SummaryResponse, GrapAssessment, DataQualityResponse,
} from '../types'

// Backoff schedule for the self-heal fan-out below. Render's cold wake measured
// 76 s in production, so the retries have to span that window *from the end of
// the failed attempt*, not from page load: the previous 4/10/20/30 schedule
// fired its last retry 64 s after mount — while the instance was still booting —
// and then gave up for good, which is how a page opened during a cold start
// stayed permanently empty (dead atmosphere panels, "No active alerts", an empty
// verification chart) even after the backend was serving normally. Each entry is
// the wait *before* the next attempt, so the sweep now runs to ~229 s and every
// retry is issued against an instance that has had time to finish booting.
const SELF_HEAL_DELAYS_MS = [4_000, 10_000, 20_000, 30_000, 45_000, 60_000, 60_000]

interface WindVectorInput {
  lat: number
  lon: number
  direction_deg: number | null
  speed: number | null
}

function buildWindArrows(stations: Station[], atmosphere: AtmosphereCurrentResponse | null): WindVectorInput[] {
  if (!atmosphere) return []
  const out: WindVectorInput[] = []
  for (const st of stations) {
    const a = atmosphere.stations.find((x) => x.station === st.name)
    if (a && a.wind?.wind_direction_deg != null) {
      out.push({
        lat: st.latitude,
        lon: st.longitude,
        direction_deg: a.wind.wind_direction_deg,
        speed: a.wind.wind_speed_mps,
      })
    }
  }
  return out
}

function ChartTooltip() {
  return {
    contentStyle: { backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8 },
    labelStyle: { color: CHART.tooltipLabel },
  }
}

export default function Dashboard() {
  const [stations, setStations] = useState<Station[]>([])
  const [selected, setSelected] = useState('Anand Vihar')
  const [current, setCurrent] = useState<CurrentAQI | null>(null)
  const [forecast, setForecast] = useState<ForecastPoint[]>([])
  const [pm25, setPm25] = useState<Pm25ForecastResponse | null>(null)
  const [atmosphere, setAtmosphere] = useState<AtmosphereCurrentResponse | null>(null)
  const [fireActivity, setFireActivity] = useState<FireActivity | null>(null)
  const [hotspots, setHotspots] = useState<FireHotspot[]>([])
  const [transport, setTransport] = useState<TransportRiskResponse | null>(null)
  const [grap, setGrap] = useState<GrapAssessment | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [modelPerf, setModelPerf] = useState<ModelPerformanceResponse | null>(null)
  const [pollution, setPollution] = useState<PollutionReading[]>([])
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [dataQuality, setDataQuality] = useState<DataQualityResponse | null>(null)
  const [history, setHistory] = useState<PollutionReading[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [failedPanels, setFailedPanels] = useState<string[]>([])
  const [selfHealAttempt, setSelfHealAttempt] = useState(0)
  const [warming, setWarming] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const stationId = stations.find((s) => s.name === selected)?.id ?? null

  useEffect(() => {
    getStations()
      .then((r) => setStations(r.data))
      .catch(() => setStations([]))
  }, [])

  // If the (scale-to-zero) backend was cold on first paint, stations can be
  // empty; refetch when the user returns to a visible tab.
  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === 'visible' && stations.length === 0) {
        getStations().then((r) => setStations(r.data)).catch(() => {})
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [stations.length])

  const refreshAll = useCallback(() => {
    setError(null)
    const failures: string[] = []
    // Runs one panel fetch, keeps the last good data on failure and records
    // the panel name so the banner below can offer a retry (nothing is
    // silently dropped any more).
    const settle = <T,>(label: string, promise: Promise<{ data: T }>, set: (v: T) => void) =>
      promise
        .then((r) => set(r.data))
        .catch(() => { failures.push(label) })
    const run = async () => {
      setWarming(true)
      await ensureWarm()
      setWarming(false)
      await Promise.allSettled([
        settle('Summary', getSummary(), setSummary),
        settle('Current AQI', getCurrentAQI(selected), setCurrent),
        settle('72h forecast', getForecast(selected), setForecast),
        settle('PM2.5 forecast', getPm25Forecast(selected), setPm25),
        settle('Atmospheric state', getAtmosphereCurrent(selected), setAtmosphere),
        settle('Fire activity', getFireActivity(), setFireActivity),
        settle('Fire hotspots', getFireHotspots().then((r) => ({ data: r.data.hotspots })), setHotspots),
        settle('Transport risk', getTransportRisk(), setTransport),
        settle('Model performance', getModelPerformance(), setModelPerf),
        settle('Latest pollution', getPollutionLatest(), setPollution),
        stationId != null
          ? settle('Pollution history', getPollutionHistory(stationId, 168), setHistory)
          : Promise.resolve(),
        settle('GRAP', getGrapCurrent(), setGrap),
        settle('Alerts', getAlerts(), setAlerts),
        settle('Data quality', getDataQuality(), setDataQuality),
      ])
      setFailedPanels([...failures])
    }
    return run()
  }, [selected, stationId])

  useEffect(() => {
    setLoading(true)
    refreshAll().finally(() => setLoading(false))
  }, [refreshAll])

  // Self-heal. A scale-to-zero cold start can outlast the warm-up gate and the
  // per-request retries, and the failure mode was permanent: every panel stayed
  // empty until the user manually reloaded, because nothing re-polled after the
  // instance finally came up. Retry the whole fan-out on a widening backoff a
  // few times, then stop and let the banner offer a manual retry.
  useEffect(() => {
    if (failedPanels.length === 0) {
      setSelfHealAttempt(0)
      return
    }
    if (selfHealAttempt >= SELF_HEAL_DELAYS_MS.length) return
    const timer = setTimeout(() => {
      setWarming(true)
      refreshAll().finally(() => {
        setWarming(false)
        setSelfHealAttempt((n) => n + 1)
      })
    }, SELF_HEAL_DELAYS_MS[selfHealAttempt])
    return () => clearTimeout(timer)
  }, [failedPanels.length, selfHealAttempt, refreshAll])

  useIntervalRefresh(refreshAll, 60_000, autoRefresh)

  const modelLabels: Record<string, string> = {
    persistence: 'Persistence',
    random_forest: 'Random Forest',
    xgboost: 'XGBoost',
    gru: 'GRU',
  }

  const rankStyle = aqiStyle(current?.aqi ?? null)

  return (
    <div className="space-y-6">
      <PageHeader
        title="Atmospheric Intelligence Control Room"
        subtitle="Delhi NCR · one consolidated view of live CPCB observations, 72-hour ML forecasts, atmospheric physics and regional fire-plume transport"
        breadcrumbs={[{ label: 'Control Room' }]}
        lastUpdated={summary?.generated_at ?? current?.timestamp}
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex cursor-pointer select-none items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="h-3.5 w-3.5 accent-inst-700"
              />
              Auto-refresh (1 min)
            </label>
          </div>
        }
      />

      {error && (
        <ErrorState
          title="Control room is unavailable"
          message="The API server could not be reached. Verify the backend is running on :8000."
          onRetry={() => {
            setLoading(true)
            refreshAll().finally(() => setLoading(false))
          }}
        />
      )}

      {warming && (
        <div className="flex items-center gap-2 rounded-lg border border-inst-200 bg-inst-50 px-4 py-2.5 text-sm text-inst-800" role="status">
          <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-inst-600" aria-hidden="true" />
          Connecting to the API server — the free-tier backend may be waking up…
        </div>
      )}

      {!error && failedPanels.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-300 bg-amber-50 px-4 py-2.5 text-sm text-amber-900" role="alert">
          <p>
            {failedPanels.length === 1 ? '1 panel' : `${failedPanels.length} panels`} couldn&apos;t load:&nbsp;
            <span className="font-medium">{failedPanels.join(', ')}</span>. Everything else is still live.
          </p>
          <button
            type="button"
            onClick={() => { setLoading(true); refreshAll().finally(() => setLoading(false)) }}
            className="rounded-lg border border-amber-400 bg-white px-3 py-1 text-xs font-semibold text-amber-800 hover:bg-amber-100"
          >
            Retry
          </button>
        </div>
      )}

      {loading && !current && !summary ? (
        <LoadingState label="Loading control room" rows={3} />
      ) : (
        <>
          {/* ---------- Regional pulse ---------- */}
          {summary && (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <div className="card flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-inst-50">
                  <Sparkles className="h-5 w-5 text-inst-700" aria-hidden="true" />
                </span>
                <div>
                  <p className="text-xs text-slate-500">NCR average AQI</p>
                  <div className="flex items-center gap-2">
                    <p className="text-xl font-bold text-slate-900">{fmt(summary.ncr_avg_aqi, 0)}</p>
                    <AQIBadge category={aqiStyle(summary.ncr_avg_aqi).label} aqi={summary.ncr_avg_aqi} />
                  </div>
                </div>
              </div>

              <StationRank summary={summary} best />
              <StationRank summary={summary} />

              <div className="card flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-50">
                  <Flame className="h-5 w-5 text-orange-600" aria-hidden="true" />
                </span>
                <div>
                  <p className="text-xs text-slate-500">Active fires (24 h)</p>
                  <p className="text-xl font-bold text-slate-900">{summary.active_fires_24h}</p>
                </div>
              </div>

              <div className="card flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-red-50">
                  <AlertTriangle className="h-5 w-5 text-red-600" aria-hidden="true" />
                </span>
                <div>
                  <p className="text-xs text-slate-500">Open alerts</p>
                  <p className="text-xl font-bold text-slate-900">{summary.open_alerts}</p>
                </div>
              </div>
            </div>
          )}

          {/* ---------- Data sources & freshness ---------- */}
          {(dataQuality || summary) && (
            <section className="card">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-bold text-slate-900">Data sources & freshness</h2>
                <div className="flex flex-wrap items-center gap-2">
                {(summary?.data_mode === 'live' || summary?.data_mode === 'demo_seeded') && (
                  <span
                    title={summary.data_mode_note}
                    className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                      summary.data_mode === 'live'
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
                        : 'border-amber-300 bg-amber-50 text-amber-700'
                    }`}
                  >
                    {summary.data_mode === 'live' ? 'Live data' : 'Demo-seeded — not real-time'}
                  </span>
                )}
                <Link to="/data" className="text-xs font-semibold text-inst-700 hover:underline">
                  Data & system sources →
                </Link>
              </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Stations</p>
                  <p className="text-base font-bold tabular-nums text-slate-900">{dataQuality?.station_count ?? summary?.stations}</p>
                  <p className="text-xs text-slate-500">CPCB live network</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Pollution observations</p>
                  <p className="text-base font-bold tabular-nums text-slate-900">{dataQuality?.tables?.pollution_observations?.total?.toLocaleString() ?? '—'}</p>
                  <p className="truncate text-xs text-slate-500">{dataQuality?.latest_pollution_reading ? tsFmt(dataQuality.latest_pollution_reading) : 'CPCB'}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Weather observations</p>
                  <p className="text-base font-bold tabular-nums text-slate-900">{dataQuality?.tables?.weather_observations?.total?.toLocaleString() ?? '—'}</p>
                  <p className="truncate text-xs text-slate-500">{dataQuality?.latest_weather_reading ? tsFmt(dataQuality.latest_weather_reading) : 'Open-Meteo'}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Stored fires</p>
                  <p className="text-base font-bold tabular-nums text-slate-900">{dataQuality?.tables?.fire_readings?.total?.toLocaleString() ?? summary?.active_fires_24h}</p>
                  <p className="truncate text-xs text-slate-500">{dataQuality?.latest_fire_reading ? tsFmt(dataQuality.latest_fire_reading) : 'NASA FIRMS'}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Forecast rows</p>
                  <p className="text-base font-bold tabular-nums text-slate-900">{dataQuality?.tables?.forecasts?.total?.toLocaleString() ?? '—'}</p>
                  <p className="truncate text-xs text-slate-500">{dataQuality?.stations_with_forecasts != null ? `${dataQuality.stations_with_forecasts} stations covered` : 'ML engine'}</p>
                </div>
              </div>
            </section>
          )}

          {/* ---------- Station grid ---------- */}
          <div>
            <h2 className="mb-3 text-base font-semibold text-slate-900">Monitoring stations</h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6 xl:grid-cols-6">
              {stations.map((s) => {
                const r = pollution.find((p) => p.station_id === s.id)
                const style = aqiStyle(r?.aqi ?? null)
                const isSel = s.name === selected
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setSelected(s.name)}
                    aria-pressed={isSel}
                    className={`rounded-xl border bg-white p-3 text-left shadow-card transition-all ${
                      isSel ? 'border-inst-600 ring-2 ring-inst-200' : 'border-slate-200 hover:border-inst-300'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <p className="truncate text-xs font-semibold text-slate-900">{s.name}</p>
                      <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${style.bar}`} aria-hidden="true" />
                    </div>
                    <p className="mt-1 text-lg font-bold tabular-nums text-slate-900">
                      {fmt(r?.aqi ?? null, 0)}
                      <span className="ml-1 text-[10px] font-normal text-slate-500">Recorded AQI</span>
                    </p>
                    <p className="truncate text-xs text-slate-700">
                      PM2.5 {fmt(r?.pm25 ?? null, 0)}<span className="text-slate-500"> μg/m³</span>
                    </p>
                    {r ? (
                      <p className={`truncate text-xs font-semibold capitalize ${style.text}`}>Recorded · {style.label}</p>
                    ) : (
                      <p className="truncate text-xs italic text-slate-500">Awaiting live data</p>
                    )}
                  </button>
                )
              })}
            </div>
          </div>

          {/* ---------- Station banner ---------- */}
          <div className="grid gap-6 lg:grid-cols-3">
            <section className="card lg:col-span-1">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h2 className="text-base font-bold text-slate-900">{selected}</h2>
                  <p className="text-xs text-slate-500">
                    {stations.find((s) => s.name === selected)?.city ?? 'Delhi NCR'} · updated {tsFmt(current?.timestamp)}
                  </p>
                </div>
                {current && <AQIBadge category={current.aqi_category} aqi={current.aqi} />}
              </div>

              <div className="mt-3 flex items-center gap-3">
                <div className={`flex h-16 w-16 shrink-0 flex-col items-center justify-center rounded-xl ${rankStyle.bar}`} style={{ color: readableOnHex(rankStyle.hex) }}>
                  <span className="text-2xl font-bold leading-none tabular-nums">{fmt(current?.aqi, 0)}</span>
                  <span className="text-[10px] font-medium uppercase">AQI</span>
                </div>
                <div className="min-w-0">
                  <p className={`text-sm font-semibold capitalize ${rankStyle.text}`}>{rankStyle.label}</p>
                  <p className="truncate text-xs text-slate-500">
                    Dominant: {current?.dominant_pollutant ?? '—'}
                  </p>
                  <p className="text-xs text-slate-500">{tsFmt(current?.timestamp)}</p>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2">
                {[
                  { label: 'PM2.5', v: current?.pm25 },
                  { label: 'PM10', v: current?.pm10 },
                  { label: 'O₃', v: current?.o3 },
                  { label: 'NO₂', v: current?.no2 },
                ].map((p) => (
                  <div key={p.label} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">{p.label}</p>
                    <p className="text-base font-bold tabular-nums text-slate-900">{fmt(p.v, 1)}</p>
                  </div>
                ))}
              </div>

              {!current && (
                <p className="mt-3 text-xs leading-relaxed text-slate-500">
                  No CPCB readings have been published for this station yet — the PMC uses NWP data for its
                  forecast until live observations arrive.
                </p>
              )}
            </section>

            <div className="lg:col-span-2">
              <AtmosphericState atmosphere={atmosphere?.stations.find((s) => s.station === selected) ?? null} />
            </div>
          </div>

          {/* ---------- Atmospheric narrative + dispersion ---------- */}
          <div className="grid gap-6 lg:grid-cols-2">
            <AtmosphericStory atmosphere={atmosphere?.stations.find((s) => s.station === selected) ?? null} />
            <DispersionMeter atmosphere={atmosphere?.stations.find((s) => s.station === selected) ?? null} />
          </div>

          {/* ---------- 72-hour outlook ---------- */}
          <div className="grid gap-6 lg:grid-cols-2">
            <ForecastRiskBand forecast={forecast} />
            <section className="card">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-bold text-slate-900">72-hour PM2.5 forecast — {selected}</h2>
                {pm25 && <span className="text-xs text-slate-500">{pm25.uncertainty_method} · {pm25.forecast_strategy}</span>}
                <Link to="/forecast" className="text-xs font-semibold text-inst-700 hover:underline">
                  Open full forecast →
                </Link>
              </div>
              <ForecastPreview point={pm25?.forecasts?.slice(0, 24) ?? forecast.slice(0, 24)} observed={current?.pm25 ?? null} uncertaintyMethod={pm25?.uncertainty_method} />
            </section>
          </div>

          {/* ---------- Why this forecast ---------- */}
          <ForecastReasonPanel station={selected} />

          {/* ---------- Observed vs forecast verification ---------- */}
          <VerificationChart history={history} pm25={pm25} />

          {/* ---------- Regional transport ---------- */}
          <TransportChain transport={transport} fire={fireActivity} />

          {/* ---------- Alerts strip ---------- */}
          <AlertsSection alerts={alerts} grap={grap} />

          {/* ---------- Model performance summary ---------- */}
          <section className="card">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-base font-bold text-slate-900">Forecast model quality</h2>
              <Link to="/model-performance" className="text-xs font-semibold text-inst-700 hover:underline">
                Full metrics →
              </Link>
            </div>
            {modelPerf?.results?.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 text-left">Horizon</th>
                      {modelPerf.evaluated_models.map((m) => (
                        <th key={m} className="px-2 py-2 text-left">{modelLabels[m] ?? m}</th>
                      ))}
                      <th className="py-2 text-right">N (test)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelPerf.results.slice(0, 12).map((r) => (
                      <tr key={r.horizon_hours} className="border-b border-slate-100">
                        <td className="py-2 font-semibold text-slate-900">+{r.horizon_hours}h</td>
                        {modelPerf.evaluated_models.map((m) => {
                          const mm = r.metrics[m]
                          return (
                            <td key={m} className="px-2 py-2 text-xs text-slate-700">
                              MAE {fmt(mm?.mae, 1)} · R² {fmt(mm?.r2, 2)}
                            </td>
                          )
                        })}
                        <td className="py-2 text-right text-slate-700">{r.n_test?.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-slate-500">No measured metrics available yet.</p>
            )}
          </section>

          {/* ---------- Map ---------- */}
          <section className="card overflow-hidden p-0">
            <div className="px-6 pt-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-bold text-slate-900">NCR monitoring map</h2>
                <span className="text-xs text-slate-500">stations = recorded AQI colour · circles = FIRMS hotspots · cyan arrows = wind</span>
              </div>
            </div>
            <div className="p-3">
              {stations.length ? (
                <StationMap
                  stations={stations}
                  pollution={pollution}
                  fires={hotspots}
                  wind={buildWindArrows(stations, atmosphere)}
                  onSelectStation={setSelected}
                />
              ) : (
                <div className="text-center">
                  <EmptyState title="No stations loaded" />
                  <button
                    type="button"
                    onClick={() => { getStations().then((r) => setStations(r.data)).catch(() => {}); refreshAll() }}
                    className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-inst-300 bg-white px-3 py-1.5 text-xs font-semibold text-inst-700 hover:bg-inst-50"
                  >
                    Retry
                  </button>
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

function StationRank({ summary, best = false }: { summary: SummaryResponse; best?: boolean }) {
  const target = best ? summary.best_station : summary.worst_station
  if (!target || target.aqi == null) return null
  const style = aqiStyle(target.aqi)
  const Icon = best ? TrendingDown : TrendingUp
  return (
    <div className="card flex items-center gap-3">
      <span className={`flex h-10 w-10 items-center justify-center rounded-lg ${best ? 'bg-green-50' : 'bg-red-50'}`}>
        <Icon className={`h-5 w-5 ${best ? 'text-green-600' : 'text-red-600'}`} aria-hidden="true" />
      </span>
      <div className="min-w-0">
        <p className="text-xs text-slate-500">{best ? 'Cleanest station' : 'Most polluted'}</p>
        <p className="truncate text-sm font-bold text-slate-900">{target.name}</p>
        <p className={`text-xs font-semibold capitalize ${style.text}`}>{style.label} · {fmt(target.aqi, 0)}</p>
      </div>
    </div>
  )
}

function ForecastPreview({ point, observed, uncertaintyMethod }: { point: Pm25ForecastPoint[] | ForecastPoint[]; observed: number | null; uncertaintyMethod?: string | null }) {
  if (!point.length) {
    return (
      <EmptyState
        title="Forecast engine not available"
        hint="Train the PM2.5 horizon models to populate the 72-hour forecast chart."
      />
    )
  }
  const first = point[0] as Partial<Pm25ForecastPoint>
  const isPm25 = typeof first.predicted_pm25 === 'number'
  const rows = point.map((p) => {
    const pm = p as Partial<Pm25ForecastPoint>
    const fc = p as Partial<ForecastPoint>
    return {
      time: `+${pm.forecast_horizon ?? fc.horizon_hours ?? 0}h`,
      pred: isPm25 ? (pm.predicted_pm25 ?? 0) : (fc.pm25_pred ?? fc.aqi_pred ?? 0),
      lower: isPm25 ? (pm.pm25_lower_bound ?? null) : null,
      upper: isPm25 ? (pm.pm25_upper_bound ?? null) : null,
    }
  })
  const hasBounds = isPm25 && rows.some((r) => r.lower != null && r.upper != null && r.upper > r.lower)
  if (observed !== null && observed !== undefined) {
    rows.unshift({ time: 'now', pred: observed, lower: observed, upper: observed })
  }
  const tooltip = ChartTooltip()
  return (
    <div>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={rows} margin={{ left: -20, right: 8 }}>
          <defs>
            <linearGradient id="fcastFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART.brand} stopOpacity={0.28} />
              <stop offset="95%" stopColor={CHART.brand} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART.brand} stopOpacity={0.18} />
              <stop offset="95%" stopColor={CHART.brand} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis dataKey="time" stroke={CHART.axis} fontSize={11} />
          <YAxis stroke={CHART.axis} fontSize={11} />
          <Tooltip {...tooltip} />
          <ReferenceLine y={60} stroke={CHART.naaqsPm25} strokeDasharray="4 4" label={{ value: 'NAAQS 60', position: 'insideTopRight', fill: CHART.naaqsPm25, fontSize: 10 }} />
          {observed !== null && observed !== undefined && (
            <ReferenceDot x="now" y={observed} r={4} fill={CHART.naaqsPm25} stroke="#fff" strokeWidth={1.5} />
          )}
          {hasBounds && (
            <>
              <Area type="monotone" dataKey="upper" stroke="none" fill="url(#bandFill)" name="Upper bound" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="transparent" name="Lower bound" />
            </>
          )}
          <Area type="monotone" dataKey="pred" stroke={CHART.brand} strokeWidth={2} fill={hasBounds ? 'transparent' : 'url(#fcastFill)'} name="PM2.5 (μg/m³)" />
          {hasBounds && <Legend wrapperStyle={{ fontSize: 11 }} />}
        </AreaChart>
      </ResponsiveContainer>
      {hasBounds && uncertaintyMethod && (
        <p className="mt-1 text-[11px] leading-snug text-slate-500">
          Shaded band = {uncertaintyMethod} prediction interval (observed dot at “now”).
        </p>
      )}
    </div>
  )
}

interface VerifyRow {
  h: number
  obs: number | null
  fct: number | null
  lo: number | null
  hi: number | null
}

function buildVerifyRows(history: PollutionReading[], pm25: Pm25ForecastResponse | null): VerifyRow[] {
  if (!pm25) return []
  const anchor = new Date(pm25.release_time).getTime()
  if (Number.isNaN(anchor)) return []
  const perHour = new Map<number, number>()
  for (const r of history) {
    if (r.pm25 === null || r.pm25 === undefined) continue
    const t = new Date(r.timestamp).getTime()
    if (Number.isNaN(t)) continue
    const h = Math.round((t - anchor) / 3_600_000)
    if (h < -168 || h > 0) continue
    if (!perHour.has(h)) perHour.set(h, r.pm25)
  }
  const rows: VerifyRow[] = []
  for (let h = -168; h <= 0; h++) {
    rows.push({ h, obs: perHour.get(h) ?? null, fct: null, lo: null, hi: null })
  }
  for (const p of pm25.forecasts) {
    rows.push({ h: p.forecast_horizon, obs: null, fct: p.predicted_pm25, lo: p.pm25_lower_bound, hi: p.pm25_upper_bound })
  }
  return rows
}

function VerificationChart({ history, pm25 }: { history: PollutionReading[]; pm25: Pm25ForecastResponse | null }) {
  const rows = buildVerifyRows(history, pm25)
  const observedCount = rows.filter((r) => r.obs != null).length
  const hasForecast = rows.some((r) => r.fct != null)
  const hasBand = rows.some((r) => r.lo != null && r.hi != null && r.hi > r.lo)
  if (!hasForecast || observedCount === 0) {
    return (
      <section className="card">
        <h2 className="mb-3 text-base font-bold text-slate-900">Verification — observed vs forecast (last 7 days)</h2>
        <p className="text-sm text-slate-500">
          Observed-vs-forecast series becomes available once pollution history and a PM2.5 forecast are loaded for the selected station.
        </p>
      </section>
    )
  }
  const tooltip = ChartTooltip()
  return (
    <section className="card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-bold text-slate-900">Verification — observed vs forecast (last 7 days)</h2>
        <span className="text-xs text-slate-500">Amber = CPCB observed · brand = model forecast · shaded band = conformal interval</span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={rows} margin={{ left: -20, right: 8 }}>
          <defs>
            <linearGradient id="obsFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART.naaqsPm25} stopOpacity={0.22} />
              <stop offset="95%" stopColor={CHART.naaqsPm25} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="vfyBand" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART.brand} stopOpacity={0.18} />
              <stop offset="95%" stopColor={CHART.brand} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis
            dataKey="h"
            type="number"
            domain={[-168, 72]}
            ticks={[-168, -120, -72, -48, -24, 0, 24, 48, 72]}
            tickFormatter={(h: number) => (h === 0 ? 'now' : h < 0 ? `${-h}h` : `+${h}h`)}
            stroke={CHART.axis}
            fontSize={11}
          />
          <YAxis stroke={CHART.axis} fontSize={11} />
          <Tooltip
            {...tooltip}
            labelFormatter={(h: number) => (h === 0 ? 'now' : h < 0 ? `${-h}h ago` : `+${h}h ahead`)}
          />
          <ReferenceLine y={60} stroke={CHART.naaqsPm25} strokeDasharray="4 4" label={{ value: 'NAAQS 60', position: 'insideTopRight', fill: CHART.naaqsPm25, fontSize: 10 }} />
          <ReferenceLine x={0} stroke={CHART.axis} strokeDasharray="3 3" label={{ value: 'now', position: 'insideTopLeft', fill: CHART.axis, fontSize: 10 }} />
          {hasBand && (
            <>
              <Area type="monotone" dataKey="hi" stroke="none" fill="url(#vfyBand)" name="Upper bound" />
              <Area type="monotone" dataKey="lo" stroke="none" fill="transparent" name="Lower bound" />
            </>
          )}
          <Area type="monotone" dataKey="obs" stroke={CHART.naaqsPm25} strokeWidth={2} fill="url(#obsFill)" name="Observed PM2.5 (μg/m³)" />
          <Line type="monotone" dataKey="fct" stroke={CHART.brand} strokeWidth={2} dot={false} name="Forecast PM2.5 (μg/m³)" />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="mt-1 text-[11px] leading-snug text-slate-500">
        Observed from the persisted CPCB-format archive for the selected station (hourly), forecast = ML prediction
        {pm25 ? ` (${pm25.model} · ${pm25.uncertainty_method})` : ''} against the forecast timestamp at “now”.
      </p>
    </section>
  )
}