import { useEffect, useState } from 'react'
import { getSummary } from '../api/client'
import { useIntervalRefresh } from '../hooks/useIntervalRefresh'
import { fmt } from '../lib/aqi'
import { formatAge, isStale, modeLabel } from '../lib/freshness'
import type { SummaryResponse } from '../types'

interface Props {
  className?: string
}

export default function SystemStatus({ className = '' }: Props) {
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  // Render free-tier instances take 45-90 s to wake; the client already retries
  // transient 5xx internally for ~40 s per load. Declaring the API offline on a
  // single failed load would flash a false red badge right after a cold start,
  // so we only flip to "API offline" after >=2 back-to-back exhausted loads.
  const [consecutiveFails, setConsecutiveFails] = useState(0)

  const load = () => {
    getSummary()
      .then((r) => {
        setSummary(r.data)
        setConsecutiveFails(0)
      })
      .catch(() => setConsecutiveFails((n) => n + 1))
  }

  useEffect(() => { load() }, [])
  useIntervalRefresh(load, 60_000, true)

  const offline = consecutiveFails >= 2

  // Two independent staleness questions, previously conflated into one flag:
  //  - `apiStale`  : the backend itself has not rebuilt the summary recently
  //                  (a service-health problem).
  //  - `dataStale` : the newest stored observation is old, i.e. the upstream
  //                  feed is not publishing (a data-provenance problem).
  // Either one must suppress the reassuring green "Live" treatment.
  let apiStale = false
  if (!offline && summary?.generated_at) {
    const t = new Date(summary.generated_at).getTime()
    if (!Number.isNaN(t)) apiStale = Date.now() - t > 2 * 60 * 60 * 1000
  }
  const dataStale = isStale(summary?.observation_age_hours)

  const dot = offline ? 'bg-red-500' : apiStale || dataStale ? 'bg-amber-400' : 'bg-emerald-400'

  // `modeLabel` maps every known mode to its own label. The previous ternary
  // chain fell through to 'Live' for stale/archive/empty, so an offline feed or a
  // months-old archive was reported as live.
  const statusText = offline
    ? 'API offline'
    : dataStale
      ? 'Data stale'
      : apiStale
        ? 'Summary stale'
        : modeLabel(summary?.data_mode)

  const ageLabel = formatAge(summary?.observation_age_hours)

  return (
    <div className={`hidden items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-xs text-white lg:flex ${className}`}>
      <span className="relative flex h-2 w-2">
        {!offline && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${dot}`} />}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${dot}`} />
      </span>
      <span className="font-semibold uppercase tracking-wide">
        {statusText}
      </span>
      {!offline && summary && (
        <span className="text-inst-100" title={summary.data_mode_note ?? undefined}>
          {summary.data_mode === 'demo_seeded' ? 'archive → now · ' : ''}
          NCR AQI {fmt(summary.ncr_avg_aqi, 0)} · {summary.stations_with_readings}/{summary.stations} stations
          {ageLabel ? ` · data ${ageLabel}` : ''}
        </span>
      )}
    </div>
  )
}