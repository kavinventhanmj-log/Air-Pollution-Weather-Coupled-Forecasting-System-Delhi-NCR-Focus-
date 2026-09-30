import { Gauge } from 'lucide-react'
import type { StationAtmosphere } from '../types'
import { fmt } from '../lib/aqi'
import { STATUS } from '../lib/theme'
import EmptyState from './EmptyState'

interface Props {
  atmosphere: StationAtmosphere | null
}

export default function DispersionMeter({ atmosphere }: Props) {
  if (!atmosphere) {
    return (
      <section className="card">
        <div className="card-header mb-4">Dispersion potential</div>
        <EmptyState
          title="Atmospheric profile unavailable"
          hint="Requires stored weather and pressure-level observations for this station."
        />
      </section>
    )
  }

  // Dispersion drivers do not share a polarity. Per the backend contract
  // (atmosphere_service: "higher wind/pbl/ventilation mean more dispersion,
  // higher inversion/trapping mean more trapping"), a *high* normalised wind,
  // PBL or ventilation is favourable, while a *high* normalised trapping is not.
  //
  // Averaging the four raw values - as this component used to - mixes
  // favourable and unfavourable drivers into one mean and then inverted it, so
  // a strong-wind day and a trapped day could score the same. Invert only the
  // trapping term, average the four dispersion-favourable drivers, and use that
  // directly rather than subtracting it from 1.
  const drivers = [
    atmosphere.wind.normalized,
    atmosphere.pbl.normalized,
    atmosphere.ventilation.normalized,
    atmosphere.trapping.normalized == null
      ? null
      : 1 - atmosphere.trapping.normalized,
  ].filter((v): v is number => v != null)
  const meanDriver = drivers.length ? drivers.reduce((s, v) => s + v, 0) / drivers.length : null
  const potential = meanDriver != null ? Math.max(0, Math.min(100, Math.round(meanDriver * 100))) : null

  const tone = potential == null ? 'text-slate-400' : potential >= 65 ? 'text-green-700' : potential >= 40 ? 'text-amber-700' : 'text-red-700'
  const bar = potential == null ? STATUS.muted : potential >= 65 ? STATUS.good : potential >= 40 ? STATUS.warn : STATUS.bad

  return (
    <section className="card">
      <div className="mb-3 flex items-center gap-2">
        <Gauge className="h-4 w-4 text-inst-700" aria-hidden="true" />
        <h2 className="text-base font-bold text-slate-900">Dispersion potential</h2>
        <span className="ml-auto rounded-full bg-inst-50 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-inst-800">
          Estimated
        </span>
      </div>

      <div className="flex items-end justify-between gap-4">
        <div>
          <p className={`text-4xl font-bold tabular-nums ${tone}`}>
            {potential ?? '--'}
            <span className="text-lg font-semibold text-slate-400">/100</span>
          </p>
          <p className="mt-1 text-xs leading-snug text-slate-500">
            {potential == null
              ? 'Index cannot be estimated without normalised drivers.'
              : potential >= 65
                ? 'The atmosphere can flush emissions relatively easily.'
                : potential >= 40
                  ? 'Dilution is partial — expect slower pollutant build-down.'
                  : 'Dilution is weak — emissions are likely to accumulate near the surface.'}
          </p>
        </div>
        <div className="w-28">
          <div className="flex justify-between text-[10px] font-medium text-slate-400">
            <span>0</span>
            <span>100</span>
          </div>
          <div className="mt-1 h-3 w-full overflow-hidden rounded-full bg-slate-200">
            <div className="h-3 rounded-full transition-all" style={{ width: `${potential ?? 0}%`, backgroundColor: bar }} />
          </div>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-3 gap-3 text-center">
        <div className="rounded-lg bg-slate-50 p-2">
          <dt className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Ventilation</dt>
          <dd className="mt-0.5 text-sm font-bold text-slate-900">{fmt(atmosphere.ventilation.ventilation_coefficient_m2s)} m²/s</dd>
          <dd className="truncate text-[10px] text-slate-500">{atmosphere.ventilation.label}</dd>
        </div>
        <div className="rounded-lg bg-slate-50 p-2">
          <dt className="text-[10px] font-medium uppercase tracking-wide text-slate-500">PBL height</dt>
          <dd className="mt-0.5 text-sm font-bold text-slate-900">{fmt(atmosphere.pbl.pbl_height_m)} m</dd>
          <dd className="truncate text-[10px] text-slate-500">{atmosphere.pbl.label}</dd>
        </div>
        <div className="rounded-lg bg-slate-50 p-2">
          <dt className="text-[10px] font-medium uppercase tracking-wide text-slate-500">Inversion</dt>
          <dd className="mt-0.5 text-sm font-bold text-slate-900">{atmosphere.inversion?.detected ? 'Capped' : 'None'}</dd>
          <dd className="truncate text-[10px] text-slate-500">{atmosphere.inversion?.dispersion_condition ?? 'No profile'}</dd>
        </div>
      </dl>

      <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
        Index = 100 × mean of the normalised wind, PBL and ventilation drivers plus the inverted trapping
        index, so a strong-wind day scores high and a trapped day low. Estimated for operational awareness only;
        not a regulatory metric.
      </p>
    </section>
  )
}