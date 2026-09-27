import { useCallback, useEffect, useState } from 'react'
import { MapPin, Route, Wind, RefreshCcw } from 'lucide-react'
import { getStations, getFireHotspots, getPlumeRisk, getPollutionLatest, getTransportRisk } from '../api/client'
import StationMap from '../components/StationMap'
import PageHeader from '../components/PageHeader'
import ErrorState from '../components/ErrorState'
import EmptyState from '../components/EmptyState'
import { aqiStyle, fmt } from '../lib/aqi'
import { buildTransportPathways } from '../lib/geo'
import type { Station, FireHotspot, PlumeRisk, PollutionReading, TransportRiskResponse } from '../types'

const DELHI: { lat: number; lon: number } = { lat: 28.6139, lon: 77.209 }

export default function NCRMap() {
  const [stations, setStations] = useState<Station[]>([])
  const [fires, setFires] = useState<FireHotspot[]>([])
  const [plume, setPlume] = useState<PlumeRisk | null>(null)
  const [pollution, setPollution] = useState<PollutionReading[]>([])
  const [transport, setTransport] = useState<TransportRiskResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [retrying, setRetrying] = useState(false)

  const load = useCallback(() => {
    setError(null)
    setRetrying(true)
    Promise.allSettled([
      getStations().then((r) => setStations(r.data)).catch(() => {}),
      getFireHotspots().then((r) => setFires(r.data.hotspots)).catch(() => {}),
      getPlumeRisk().then((r) => setPlume(r.data)).catch(() => {}),
      getPollutionLatest().then((r) => setPollution(r.data)).catch(() => {}),
      getTransportRisk().then((r) => setTransport(r.data)).catch(() => {}),
    ]).then((results) => {
      const failed = results.filter((r) => r.status === 'rejected').length
      if (failed >= 3) setError('Database or service unavailable — the backend may have been sleeping. Click Retry.')
    }).finally(() => setRetrying(false))
  }, [])

  useEffect(() => {
    load()
  }, [load])

  // If the (scale-to-zero) backend was cold on first paint, the panels can be
  // empty; retry automatically when the user returns to a visible tab.
  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === 'visible' && stations.length === 0) load()
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [stations.length, load])

  const latestByStation = new Map<number, PollutionReading>()
  pollution.forEach((p) => {
    if (!latestByStation.has(p.station_id)) latestByStation.set(p.station_id, p)
  })

  const windFromDeg = typeof transport?.dominant_wind_direction?.from_degrees === 'number'
    ? (transport.dominant_wind_direction.from_degrees as number)
    : null
  const windSpeed = typeof transport?.dominant_wind_direction?.wind_speed_mps === 'number'
    ? (transport.dominant_wind_direction.wind_speed_mps as number)
    : null
  const compassFrom = typeof transport?.dominant_wind_direction?.compass_from === 'string'
    ? (transport.dominant_wind_direction.compass_from as string)
    : null

  const pathways = buildTransportPathways(fires, DELHI, windFromDeg, windSpeed)
  const windVector = windFromDeg != null ? [{ lat: DELHI.lat, lon: DELHI.lon, direction_deg: windFromDeg, speed: windSpeed }] : []

  const riskTone =
    plume?.risk_level === 'HIGH' ? 'bg-red-100 text-red-800' :
    plume?.risk_level === 'MODERATE' ? 'bg-amber-100 text-amber-800' :
    'bg-green-100 text-green-800'

  return (
    <div className="space-y-6">
      <PageHeader
        title="Delhi NCR Monitoring Map"
        subtitle="CPCB air-quality stations, NASA FIRMS fire hotspots and estimated plume-transport pathways"
        breadcrumbs={[{ label: 'Dashboard', to: '/dashboard' }, { label: 'NCR Map' }]}
        lastUpdated={transport?.generated_at ?? pollution[0]?.timestamp}
      />

      {error && <ErrorState message={error} onRetry={load} />}

      {retrying && !stations.length && (
        <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-4 py-2 text-xs text-slate-500">
          <RefreshCcw className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> Loading NCR data…
        </div>
      )}

      <div className="card overflow-hidden p-0">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-6 pt-4 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1.5"><span className="h-2 w-4 border-t-2 border-dashed border-amber-700" aria-hidden="true" /> estimated advective transport pathway</span>
          <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-4 bg-cyan-600" aria-hidden="true" /> surface wind vector</span>
          {compassFrom && windSpeed != null && (
            <span className="ml-auto">regional mean wind FROM {compassFrom} ({windSpeed.toFixed(1)} m/s)</span>
          )}
        </div>
        <div className="p-3">
          <StationMap stations={stations} fires={fires} pollution={pollution} wind={windVector} pathways={pathways} />
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="card flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-inst-50">
            <MapPin className="h-5 w-5 text-inst-700" aria-hidden="true" />
          </span>
          <div>
            <p className="text-xs text-slate-500">Hotspots (24 h region)</p>
            <p className="text-xl font-bold text-slate-900">{fires.length}</p>
          </div>
        </div>
        <div className={`card flex items-center gap-3 ${plume ? '' : 'opacity-60'}`}>
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100">
            <Wind className={riskTone.split(' ')[1] ?? 'text-slate-600'} aria-hidden="true" />
          </span>
          <div>
            <p className="text-xs text-slate-500">Plume transport risk</p>
            <p className={`text-xl font-bold ${riskTone.split(' ')[1] ?? 'text-slate-900'}`}>{plume?.risk_level ?? '--'}</p>
          </div>
        </div>
        <div className="card flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100">
            <Route className="h-5 w-5 text-slate-600" aria-hidden="true" />
          </span>
          <div>
            <p className="text-xs text-slate-500">Smoke arrival (nearest)</p>
            <p className="text-xl font-bold text-slate-900">
              {plume?.transport_time_hours != null ? `${plume.transport_time_hours.toFixed(1)}h` : '--'}
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="mb-3 text-base font-bold text-slate-900">Station AQI snapshot</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
          {stations.map((s) => {
            const reading = latestByStation.get(s.id)
            const style = aqiStyle(reading?.aqi ?? null)
            return (
              <div key={s.id} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center justify-between gap-1">
                  <p className="truncate text-xs font-semibold text-slate-800">{s.name}</p>
                  <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${style.bar}`} aria-hidden="true" />
                </div>
                <p className="text-[10px] text-slate-500">
                  {s.latitude.toFixed(3)}, {s.longitude.toFixed(3)}{s.state ? ` · ${s.state}` : ''}
                </p>
                <p className="mt-1 text-sm text-slate-700">
                  {reading ? (
                    <>
                      Recorded AQI <span className="font-bold tabular-nums text-slate-900">{reading.aqi?.toFixed(0) ?? '--'}</span>
                      <span className="text-xs text-slate-500"> · PM2.5 {fmt(reading.pm25, 0)}</span>
                    </>
                  ) : (
                    <span className="italic text-slate-400">No reading yet</span>
                  )}
                </p>
              </div>
            )
          })}
          {!stations.length && (
            <div className="col-span-full">
              <EmptyState title="No stations loaded" />
              <div className="mt-2 text-center">
                <button type="button" onClick={load} className="btn-outline">Retry loading stations</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}