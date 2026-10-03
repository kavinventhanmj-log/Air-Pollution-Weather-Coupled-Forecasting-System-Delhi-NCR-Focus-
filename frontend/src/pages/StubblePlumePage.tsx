import { useCallback, useEffect, useState } from 'react'
import { Flame, Info } from 'lucide-react'
import { getPlumeRisk, getFireActivity, getFireHotspots, getTransportRisk } from '../api/client'
import PageHeader from '../components/PageHeader'
import ErrorState from '../components/ErrorState'
import LoadingState from '../components/LoadingState'
import EmptyState from '../components/EmptyState'
import StubblePlume from '../components/StubblePlume'
import StationMap from '../components/StationMap'
import KpiCard from '../components/KpiCard'
import { fmt } from '../lib/aqi'
import { buildTransportPathways } from '../lib/geo'
import type { PlumeRisk, FireActivity, FireHotspot, TransportRiskResponse } from '../types'

const DELHI: { lat: number; lon: number } = { lat: 28.6139, lon: 77.209 }

function fetchAll(showSynthetic: boolean) {
  return Promise.allSettled([
    getPlumeRisk().then((r) => r.data).catch(() => null as PlumeRisk | null),
    getFireActivity().then((r) => r.data).catch(() => null as FireActivity | null),
    getFireHotspots(showSynthetic)
      .then((r) => r.data.hotspots)
      .catch(() => [] as FireHotspot[]),
    getTransportRisk().then((r) => r.data).catch(() => null as TransportRiskResponse | null),
  ])
}

export default function StubblePlumePage() {
  const [risk, setRisk] = useState<PlumeRisk | null>(null)
  const [fire, setFire] = useState<FireActivity | null>(null)
  const [hotspots, setHotspots] = useState<FireHotspot[]>([])
  const [transport, setTransport] = useState<TransportRiskResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showSynthetic, setShowSynthetic] = useState(false)

  const load = useCallback((attempt = 1) => {
    setLoading(true)
    setError(null)
    fetchAll(showSynthetic).then((results) => {
      const ok = results.filter(
        (r): r is PromiseFulfilledResult<any> => r.status === 'fulfilled',
      ).length
      setRisk((results[0] as any)?.value ?? null)
      setFire((results[1] as any)?.value ?? null)
      setHotspots((results[2] as any)?.value ?? [])
      setTransport((results[3] as any)?.value ?? null)

      // A waking Render free backend answers the first burst of requests with
      // gateway errors; retry once after a pause so the page never stays on a
      // spinner (or zeros) just because it raced the cold start.
      if (ok === 0 && attempt < 2) {
        setTimeout(() => load(attempt + 1), 10000)
        return
      }
      if (ok === 0) setError('Failed to load fire data')
      setLoading(false)
    })
  }, [showSynthetic])

  useEffect(() => { load() }, [load])

  const windFromDeg = typeof transport?.dominant_wind_direction?.from_degrees === 'number'
    ? (transport.dominant_wind_direction.from_degrees as number)
    : null
  const windSpeed = typeof transport?.dominant_wind_direction?.wind_speed_mps === 'number'
    ? (transport.dominant_wind_direction.wind_speed_mps as number)
    : null
  const compassFrom = typeof transport?.dominant_wind_direction?.compass_from === 'string'
    ? (transport.dominant_wind_direction.compass_from as string)
    : null

  const pathways = buildTransportPathways(hotspots, DELHI, windFromDeg, windSpeed)
  const windVector = windFromDeg != null ? [{ lat: DELHI.lat, lon: DELHI.lon, direction_deg: windFromDeg, speed: windSpeed }] : []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Fire & Plume Intelligence"
        subtitle="Crop-residue burning hotspots, fire radiative power and estimated transport toward Delhi NCR"
        breadcrumbs={[{ label: 'Dashboard', to: '/dashboard' }, { label: 'Fire & Plume' }]}
        lastUpdated={fire?.date ?? undefined}
      />

      {error && <ErrorState title="Fire data unavailable" message={error} onRetry={() => load()} />}

      {loading ? (
        <LoadingState label="Loading fire intelligence" rows={2} />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <KpiCard label="Total fires (24 h)" value={fire?.total_fires ?? 0} tone={fire?.total_fires ? 'warn' : 'default'} />
            <KpiCard label="High confidence" value={fire?.high_confidence_fires ?? '--'} />
            <KpiCard label="Mean fire radiative power" value={fmt(fire?.mean_frp, 1)} unit="MW" />
            <KpiCard label="Source region" value={fire?.region ?? 'Unknown'} sub={fire?.date?.replace('T', ' ').slice(0, 16) ?? undefined} />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <StubblePlume data={risk} />
            <div className="card">
              <div className="mb-2 flex items-center gap-2">
                <Info className="h-4 w-4 text-inst-700" aria-hidden="true" />
                <h3 className="card-header mb-0">Methodology</h3>
              </div>
              <div className="space-y-2 text-sm text-slate-600">
                <p>This module estimates regional fire-plume transport risk based on:</p>
                <ul className="list-inside list-disc space-y-1">
                  <li>Live NASA FIRMS active-fire hotspots (real observations only)</li>
                  <li>Fire Radiative Power (FRP) intensity</li>
                  <li>Wind direction and speed alignment</li>
                  <li>Distance from Delhi NCR</li>
                  <li>Atmospheric dispersion conditions</li>
                </ul>
                <p className="text-xs text-slate-500">
                  Simulated (2023–24 synthetic) fire history is never used in these metrics; maps may show
                  it only as a clearly-labelled grey overlay when toggled on.
                </p>
                <p className="mt-3 text-amber-800">
                  Note: this is an estimated transport-risk indicator derived from satellite
                  hotspots + NWP winds — not a full regional chemical-transport simulation.
                </p>
                {compassFrom && windSpeed != null && (
                  <p className="text-xs text-slate-500">
                    Regional mean wind currently FROM {compassFrom} ({windSpeed.toFixed(1)} m/s).
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className="card overflow-hidden p-0">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-6 pt-5">
              <div className="flex items-center gap-2">
                <Flame className="h-4 w-4 text-orange-600" aria-hidden="true" />
                <h2 className="text-base font-bold text-slate-900">Active hotspot map</h2>
              </div>
              <label className="ml-auto inline-flex cursor-pointer items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={showSynthetic}
                  onChange={(e) => setShowSynthetic(e.target.checked)}
                  className="h-3.5 w-3.5"
                />
                Show simulated (2023–24 synthetic history) overlay
              </label>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-6 pt-2 text-xs text-slate-500">
              <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-4 bg-red-500" aria-hidden="true" /> live FIRMS hotspot (circle size/colour = FRP)</span>
              <span className="inline-flex items-center gap-1.5"><span className="h-3 w-3 rounded-full border border-dashed border-slate-400 bg-slate-200" aria-hidden="true" /> simulated hotspot (not a live detection)</span>
              <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-4 border-t-2 border-dashed border-amber-700" aria-hidden="true" /> estimated advective pathway to Delhi</span>
            </div>
            <div className="p-3">
              {hotspots.length ? (
                <StationMap stations={[]} fires={hotspots} wind={windVector} pathways={pathways} />
              ) : (
                <EmptyState title="No active hotspots" hint="No FIRMS detections in the recent window." />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}