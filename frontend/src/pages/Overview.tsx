import { useState, useEffect } from 'react'
import { getStations, getPollutionLatest, getForecast, getWeather, getInversion, getFireActivity, getPlumeRisk, getExplanation, getCoupling, getSummary, getForecastExportUrl } from '../api/client'
import AQICard from '../components/AQICard'
import AQIBadge from '../components/AQIBadge'
import StatCard from '../components/StatCard'
import ForecastChart from '../components/ForecastChart'
import InversionPanel from '../components/InversionPanel'
import CouplingPanel from '../components/CouplingPanel'
import StubblePlume from '../components/StubblePlume'
import ExplainabilityPanel from '../components/ExplainabilityPanel'
import PageHeader from '../components/PageHeader'
import EmptyState from '../components/EmptyState'
import { useIntervalRefresh } from '../hooks/useIntervalRefresh'
import { CHART } from '../lib/theme'
import { aqiCategory } from '../lib/aqi'
import { formatAge, isStale, modeChip, modeLabel } from '../lib/freshness'
import type { Station, PollutionReading, ForecastPoint, WeatherData, InversionData, FireActivity, PlumeRisk, Explanation, CouplingData, SummaryResponse } from '../types'

export default function Overview() {
  const [stations, setStations] = useState<Station[]>([])
  const [selectedStation, setSelectedStation] = useState('Anand Vihar')
  const [pollution, setPollution] = useState<PollutionReading[]>([])
  const [forecast, setForecast] = useState<ForecastPoint[]>([])
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [inversion, setInversion] = useState<InversionData | null>(null)
  const [fire, setFire] = useState<FireActivity | null>(null)
  const [plumeRisk, setPlumeRisk] = useState<PlumeRisk | null>(null)
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [coupling, setCoupling] = useState<CouplingData | null>(null)
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getStations()
      .then(res => setStations(res.data))
      .catch(() => setStations([]))
  }, [])

  // Provenance of the summary numbers, derived once so the cards, the banner
  // and the status chip can never disagree about whether the data is current.
  const summaryAgeHours = summary?.observation_age_hours ?? null
  const ageLabel = formatAge(summaryAgeHours)
  const stale = isStale(summaryAgeHours)
  const mode = summary?.data_mode

  const refreshAll = (): Promise<void> => {
    setError(null)
    return Promise.allSettled([
      getPollutionLatest().then(r => setPollution(r.data)),
      getForecast(selectedStation).then(r => setForecast(r.data)),
      getWeather(selectedStation).then(r => setWeather(r.data)),
      getInversion(selectedStation).then(r => setInversion(r.data)),
      getFireActivity().then(r => setFire(r.data)),
      getPlumeRisk().then(r => setPlumeRisk(r.data)),
      getExplanation(selectedStation).then(r => setExplanation(r.data)),
      getCoupling(selectedStation).then(r => setCoupling(r.data)),
      getSummary().then(r => setSummary(r.data)),
    ]).then(results => {
      const failures = results.filter(r => r.status === 'rejected')
      if (failures.length === results.length) {
        setError('Unable to reach the backend. Make sure the API server is running.')
      }
    })
  }

  useEffect(() => {
    setLoading(true)
    refreshAll().finally(() => setLoading(false))
  }, [selectedStation])

  const latestByStation = new Map<string, PollutionReading>()
  pollution.forEach(p => {
    latestByStation.set(String(p.station_id), p)
    latestByStation.set(p.station, p)
  })
  const selectedReading =
    latestByStation.get(String(stations.find(s => s.name === selectedStation)?.id ?? '')) ??
    latestByStation.get(selectedStation) ??
    null

  useIntervalRefresh(refreshAll, 60_000, autoRefresh)

  return (
    <div className="space-y-6">
      <PageHeader
        title="Delhi NCR Air Intelligence"
        subtitle="Real-time monitoring & 72-hour forecasting — consolidated operations view"
        breadcrumbs={[{ label: 'Dashboard', to: '/dashboard' }, { label: 'Overview' }]}
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex cursor-pointer select-none items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={e => setAutoRefresh(e.target.checked)}
                className="h-3.5 w-3.5 accent-inst-700"
              />
              Auto-refresh (1 min)
            </label>
            <a
              href={getForecastExportUrl(selectedStation, 72)}
              className="btn-outline"
            >
              Export CSV
            </a>
            
          </div>
        }
      />

      {/* When the upstream feed stops publishing, the AQI figures below are the
          newest readings on record rather than current conditions. Without this
          banner a months-old average is indistinguishable from a live one. */}
      {summary && mode === 'stale' && (
        <div
          role="status"
          className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"
        >
          <div>
            <p className="font-semibold">Upstream air-quality feed is not publishing</p>
            <p className="mt-0.5 text-amber-800">
              {summary.data_mode_note
                ?? `The newest station observation is ${ageLabel ?? 'of unknown age'}.`}
            </p>
          </div>
          <span
            className={`whitespace-nowrap rounded-full border px-2.5 py-1 text-xs font-semibold ${modeChip(mode)}`}
            title={summary.latest_observation_at ?? undefined}
          >
            {modeLabel(mode)}
          </span>
        </div>
      )}

      {summary && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatCard
            label="NCR Average AQI"
            value={summary.ncr_avg_aqi?.toFixed(0) ?? '--'}
            sub={
              <>
                {`${summary.stations_with_readings} / ${summary.stations} stations reporting`}
                {/* The average is built from each station's newest stored reading,
                    which may be old when the upstream feed stops publishing. Say
                    how old, so a historical number is never read as current. */}
                {ageLabel && (
                  <>
                    {' · '}
                    <span className={stale ? 'font-medium text-amber-700' : undefined}>
                      as of {ageLabel}
                    </span>
                  </>
                )}
              </>
            }
            tone={summary.ncr_avg_aqi != null ? (summary.ncr_avg_aqi <= 100 ? 'good' : summary.ncr_avg_aqi >= 200 ? 'bad' : 'warn') : 'default'}
          />
          <div className="card">
            <p className="card-header">Worst Station</p>
            {summary.worst_station ? (
              <div className="mt-1 flex items-center gap-2">
                <span className="stat-value">{summary.worst_station.name}</span>
                <AQIBadge category={summary.worst_station.aqi_category} aqi={summary.worst_station.aqi} />
              </div>
            ) : <p className="stat-value">--</p>}
          </div>
          <StatCard label="Active Fires (24h)" value={summary.active_fires_24h} sub="NASA FIRMS hotspots" tone={summary.active_fires_24h > 10 ? 'bad' : 'default'} />
          <StatCard label="Open Alerts" value={summary.open_alerts} sub={`${summary.models_trained} models trained`} tone={summary.open_alerts > 0 ? 'warn' : 'good'} />
        </div>
      )}

      {error && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <span>{error}</span>
          <button
            onClick={() => refreshAll()}
            className="whitespace-nowrap rounded bg-red-600 px-3 py-1 text-xs font-medium text-white hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      )}

      {loading && <div className="py-12 text-center text-slate-400">Loading data...</div>}

      {!loading && !error && !selectedReading && (
        <div className="card border-dashed">
          <EmptyState
            title="No pollution data for this station yet"
            hint="Run the official CPCB ingestion once the backend has a data.gov.in API key, then refresh this page."
          />
        </div>
      )}

      <div className="flex items-center justify-between text-sm text-slate-500">
        <h2 className="text-base font-bold text-slate-900">Latest observation</h2>
        {selectedReading ? (
          <span>
            {selectedReading.station} · {selectedReading.timestamp.slice(0, 16)}Z
          </span>
        ) : (
          <span>CPCB live network</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <AQICard label="Recorded AQI" value={selectedReading?.aqi} />
        <AQICard label="PM2.5" value={selectedReading?.pm25} unit="μg/m³" />
        <AQICard label="PM10" value={selectedReading?.pm10} unit="μg/m³" />
        <AQICard label="O₃" value={selectedReading?.o3} unit="μg/m³" />
        <AQICard label="NO₂" value={selectedReading?.no2} unit="μg/m³" />
        <AQICard label="SO₂" value={selectedReading?.so2} unit="μg/m³" />
        <AQICard label="CO" value={selectedReading?.co} unit="mg/m³" />
        <AQICard label="Category" value={selectedReading ? aqiCategory(selectedReading.aqi, '--') : '--'} />
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <AQICard label="Temperature" value={weather?.temperature} unit="°C" />
        <AQICard label="Humidity" value={weather?.humidity} unit="%" />
        <AQICard label="Wind Speed" value={weather?.wind_speed} unit="m/s" />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h2 className="card-header">72-Hour AQI Forecast</h2>
          <ForecastChart data={forecast} pollutant="aqi_pred" color={CHART.brand} />
        </div>
        <InversionPanel data={inversion} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <StubblePlume data={plumeRisk} />
        <ExplainabilityPanel data={explanation} />
      </div>

      <CouplingPanel data={coupling} />
    </div>
  )
}