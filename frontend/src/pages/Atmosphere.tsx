import { useCallback, useEffect, useState } from 'react'
import { CloudSun, Droplets, Gauge, Thermometer, Wind } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import ErrorState from '../components/ErrorState'
import LoadingState from '../components/LoadingState'
import EmptyState from '../components/EmptyState'
import InversionPanel from '../components/InversionPanel'
import CouplingPanel from '../components/CouplingPanel'
import DispersionMeter from '../components/DispersionMeter'
import KpiCard from '../components/KpiCard'
import { fmt } from '../lib/aqi'
import { STATUS } from '../lib/theme'
import { getStations, getWeather, getInversion, getCoupling, getCouplingFeatures, getAtmosphereCurrent } from '../api/client'
import type { Station, WeatherData, InversionData, CouplingData, CouplingFeaturesResponse, AtmosphereCurrentResponse } from '../types'

/**
 * A labelled 0..1 feature bar.
 *
 * `higherIsBetter` marks the one feature whose polarity is the reverse of the
 * rest. The coupling engine documents the contract in ml/features/coupling_engine.py:
 * "higher = more of the named tendency", and accumulation_potential is defined as
 * the complement of dispersion_potential (`1 - dispersion`). So for
 * `dispersion_potential` a high value is strong ventilation and a deep boundary
 * layer, which is the favourable end - while the other features are all
 * "higher = worse". Painting both with one "high = red" rule made
 * dispersion_potential render red at exactly the moment it was good, and
 * disagreed with this page's own explanatory copy directly above it.
 */
function FeatureBar({
  label,
  feature,
  higherIsBetter = false,
}: {
  label: string
  feature: { value: number | null; available: boolean; basis?: string } | undefined
  higherIsBetter?: boolean
}) {
  const value = feature?.value ?? null
  const width = value == null ? 0 : Math.max(4, Math.min(100, value * 100))
  // Band on the unfavourable end so the colour is right for either polarity.
  // Band on the unfavourable end so the colour is right for either polarity.
  const severity = value == null ? null : higherIsBetter ? 1 - value : value
  const color =
    value == null || severity == null
      ? STATUS.muted
      : severity < 0.33
        ? STATUS.good
        : severity < 0.66
          ? STATUS.warn
          : STATUS.bad
  return (
    <div className="py-1">
      <div className="mb-0.5 flex items-center justify-between text-xs">
        <span className="font-medium text-slate-600">{label}</span>
        <span className="font-semibold tabular-nums text-slate-900">{value == null ? '--' : value.toFixed(2)}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200">
        <div className="h-2 rounded-full transition-all" style={{ width: `${width}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

export default function Atmosphere() {
  const [stations, setStations] = useState<Station[]>([])
  const [selected, setSelected] = useState('Anand Vihar')
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [inversion, setInversion] = useState<InversionData | null>(null)
  const [coupling, setCoupling] = useState<CouplingData | null>(null)
  const [features, setFeatures] = useState<CouplingFeaturesResponse | null>(null)
  const [atmosphere, setAtmosphere] = useState<AtmosphereCurrentResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
    Promise.allSettled([
      getWeather(selected).then((r) => setWeather(r.data)).catch(() => setWeather(null)),
      getInversion(selected).then((r) => setInversion(r.data)).catch(() => setInversion(null)),
      getCoupling(selected).then((r) => setCoupling(r.data)).catch(() => setCoupling(null)),
      getCouplingFeatures(selected).then((r) => setFeatures(r.data)).catch(() => setFeatures(null)),
      getAtmosphereCurrent(selected).then((r) => setAtmosphere(r.data)).catch(() => setAtmosphere(null)),
    ]).then((results) => {
      if (results.every((r) => r.status === 'rejected')) setError('Failed to load atmospheric data')
    }).finally(() => setLoading(false))
  }, [selected])

  useEffect(() => { getStations().then((r) => setStations(r.data)).catch(() => {}) }, [])
  useEffect(() => { load() }, [load])

  const pbl = weather?.pbl_height ?? null
  const pblWidth = pbl != null ? Math.max(4, Math.min(100, (pbl / 1500) * 100)) : 0
  const pblColor = pbl == null ? STATUS.muted : pbl < 300 ? STATUS.bad : pbl < 600 ? STATUS.warn : STATUS.good
  const pblNote =
    pbl == null
      ? 'PBL height data unavailable'
      : pbl < 300
        ? 'Compressed boundary layer — pollutants trapped near the surface'
        : pbl < 600
          ? 'Moderate boundary layer — limited vertical mixing'
          : 'Elevated boundary layer — good dispersion conditions'

  const ventilation = pbl != null && (weather?.wind_speed ?? null) != null ? pbl * (weather?.wind_speed ?? 0) : null

  return (
    <div className="space-y-6">
      <PageHeader
        title="Atmospheric Conditions"
        subtitle="Vertical structure, inversion trapping and the two-way aerosol–PBL coupling that shapes dispersion"
        breadcrumbs={[{ label: 'Dashboard', to: '/dashboard' }, { label: 'Atmosphere' }]}
        lastUpdated={weather?.timestamp}
        actions={
          <select value={selected} onChange={(e) => setSelected(e.target.value)} className="select" aria-label="Select station">
            {stations.map((s) => (
              <option key={s.id} value={s.name}>{s.name}</option>
            ))}
          </select>
        }
      />

      {error && <ErrorState title="Atmospheric data unavailable" message={error} onRetry={load} />}

      {loading && !weather ? (
        <LoadingState label="Loading atmospheric data" rows={2} />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <KpiCard label="Temperature" value={fmt(weather?.temperature, 1)} unit="°C" tone={weather?.temperature != null && weather.temperature > 35 ? 'warn' : 'default'} />
            <KpiCard label="Humidity" value={fmt(weather?.humidity, 0)} unit="%" />
            <KpiCard label="Wind speed" value={fmt(weather?.wind_speed, 1)} unit="m/s" />
            <KpiCard label="Pressure (MSL)" value={fmt(weather?.pressure_msl, 0)} unit="hPa" />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <CouplingPanel data={coupling} features={features} />

            <div className="card">
              <div className="flex items-center gap-2">
                <CloudSun className="h-4 w-4 text-inst-700" aria-hidden="true" />
                <h3 className="card-header mb-0">PBL height & ventilation</h3>
              </div>
              <div className="mt-4">
                <div className="mb-1 flex justify-between text-sm">
                  <span className="text-slate-500">Current PBL height</span>
                  <span className="font-semibold text-slate-900">{pbl != null ? `${pbl}m` : '--'}</span>
                </div>
                <div className="h-3 w-full rounded-full bg-slate-200">
                  <div className="h-3 rounded-full transition-all" style={{ width: `${pblWidth}%`, backgroundColor: pblColor }} />
                </div>
                <p className="mt-2 text-xs text-slate-500">{pblNote}</p>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Ventilation index</p>
                  <p className="text-base font-bold text-slate-900">{ventilation != null ? fmt(ventilation, 0) : '--'} <span className="text-xs text-slate-400">m²/s</span></p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Wind direction</p>
                  <p className="text-base font-bold text-slate-900">{fmt(weather?.wind_direction, 0)}°</p>
                </div>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                Ventilation index = wind speed × PBL height. Low values indicate limited
                atmospheric dilution capacity, which the aerosol-PBL coupling module further modulates.
              </p>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <InversionPanel data={inversion} />
            <div className="card">
              <h3 className="card-header">What to look for</h3>
              <ul className="space-y-2 text-sm text-slate-600">
                <li className="flex items-start gap-2"><Thermometer className="mt-0.5 h-4 w-4 shrink-0 text-inst-600" aria-hidden="true" /> Inversion strength from the vertical lapse rate or PBL proxy.</li>
                <li className="flex items-start gap-2"><Wind className="mt-0.5 h-4 w-4 shrink-0 text-inst-600" aria-hidden="true" /> Whether regional winds advect plume-loads toward/away from NCR.</li>
                <li className="flex items-start gap-2"><Droplets className="mt-0.5 h-4 w-4 shrink-0 text-inst-600" aria-hidden="true" /> Both pollution trapping and dispersion are scored for each station.</li>
                <li className="flex items-start gap-2"><Gauge className="mt-0.5 h-4 w-4 shrink-0 text-inst-600" aria-hidden="true" /> Coupling panel shows the two-way aerosol–PBL feedback used by forecasts.</li>
              </ul>
            </div>
          </div>

          <DispersionMeter atmosphere={atmosphere?.stations.find((s) => s.station === selected) ?? null} />

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="card">
              <h3 className="card-header">Dispersion / accumulation state</h3>
              <p className="mb-2 text-xs text-slate-500">
                Coupling-engine tendencies from the latest stored weather + pollution
                (0–1 scale). A lower dispersion value with higher accumulation /
                stagnation means the current atmosphere retains pollutants.
              </p>
              {features ? (
                <>
                  <FeatureBar label="Dispersion potential" feature={features.features.dispersion_potential} higherIsBetter />
                  <FeatureBar label="Accumulation potential" feature={features.features.accumulation_potential} />
                  <FeatureBar label="Inversion trapping potential" feature={features.features.inversion_trapping_potential} />
                  <FeatureBar label="Pollution stagnation index" feature={features.features.pollution_stagnation_index} />
                  <FeatureBar label="Aerosol accumulation potential" feature={features.features.aerosol_accumulation_potential} />
                  <p className="mt-3 text-[10px] leading-relaxed text-slate-400">
                    {features.methodology?.note ?? 'Features from stored observations.'}
                  </p>
                </>
              ) : (
                <EmptyState title="Coupling features unavailable" hint="No stored weather or pollution data for this station yet." />
              )}
            </div>

            <div className="card">
              <h3 className="card-header">Regional influence</h3>
              <p className="mb-2 text-xs text-slate-500">
                Estimated influence of regional stubble fires and the synoptic wind on
                NCR pollution — a transparent heuristic, not a chemical plume model.
              </p>
              {features ? (
                <>
                  <FeatureBar label="Fire transport influence" feature={features.features.fire_transport_influence} />
                  <FeatureBar label="Regional transport potential" feature={features.features.regional_transport_potential} />
                  <FeatureBar label="Ozone photochemical potential" feature={features.features.ozone_photochemical_potential} />
                  <FeatureBar label="Meteorology–pollution interaction" feature={features.features.meteorology_pollution_interaction} />

                  <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-600">
                    <p className="font-semibold text-slate-700">Input provenance</p>
                    <dl className="mt-1 space-y-0.5">
                      <div className="flex justify-between gap-2"><dt className="text-slate-500">Weather observed</dt><dd className="tabular-nums">{features.provenance?.weather_reading_timestamp ? new Date(String(features.provenance.weather_reading_timestamp)).toLocaleString() : '--'}</dd></div>
                      <div className="flex justify-between gap-2"><dt className="text-slate-500">Inversion source</dt><dd className="tabular-nums">{String(features.provenance?.inversion_source ?? '--')}</dd></div>
                      <div className="flex justify-between gap-2"><dt className="text-slate-500">Fires (≤500 km)</dt><dd className="tabular-nums">{features.inputs?.fire_count ?? '--'}</dd></div>
                      <div className="flex justify-between gap-2"><dt className="text-slate-500">Upwind fires</dt><dd className="tabular-nums">{features.inputs?.upwind_fire_count ?? '--'}</dd></div>
                    </dl>
                  </div>
                </>
              ) : (
                <EmptyState title="Regional influence unavailable" hint="Fire or weather observations are not stored for this station yet." />
              )}
            </div>
          </div>

          {!weather && !inversion && !coupling && (
            <EmptyState title="No atmospheric data" hint="This station has no weather or vertical-profile observations stored yet." />
          )}
        </>
      )}
    </div>
  )
}