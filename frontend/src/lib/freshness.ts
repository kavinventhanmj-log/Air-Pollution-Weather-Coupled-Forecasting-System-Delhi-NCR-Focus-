/**
 * Freshness helpers for summary data.
 *
 * The backend computes `ncr_avg_aqi` from each station's most recent *stored*
 * observation rather than a fixed recency window, and reports the age of that
 * observation alongside it. That is deliberate: the upstream CPCB/CKAN archive
 * can stop publishing, and a windowed average would then blank the dashboard
 * even though real rows are on disk.
 *
 * The trade-off is that the number shown may be old, so every place that renders
 * it has to say how old. The helpers here are the single source of truth for
 * that wording, so a stale reading is never presented as a live one.
 */

export type DataMode = 'live' | 'stale' | 'demo_seeded' | 'static_archive' | 'empty' | string

/** Observations older than this are treated as not current. */
export const STALE_AFTER_HOURS = 24

/** True when the reported data is old enough that it must not be called live. */
export function isStale(ageHours: number | null | undefined): boolean {
  return typeof ageHours === 'number' && Number.isFinite(ageHours) && ageHours > STALE_AFTER_HOURS
}

/**
 * Render an observation age as a short human label ("18 min ago", "9 months ago").
 * Returns null when the age is unknown so callers can omit the label entirely
 * rather than printing a misleading "unknown ago".
 */
export function formatAge(ageHours: number | null | undefined): string | null {
  if (typeof ageHours !== 'number' || !Number.isFinite(ageHours) || ageHours < 0) return null

  const minutes = ageHours * 60
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${Math.round(minutes)} min ago`
  if (ageHours < 24) return `${Math.round(ageHours)} h ago`

  const days = ageHours / 24
  if (days < 7) return `${Math.round(days)} d ago`
  if (days < 31) return `${Math.round(days / 7)} wk ago`
  if (days < 365) return `${Math.round(days / 30.44)} mo ago`
  return `${(days / 365.25).toFixed(1)} yr ago`
}

/**
 * Label for the provenance of a summary, used for the status chip and tooltip.
 * `live` is the only mode that warrants a reassuring green "Live" treatment.
 */
export function modeLabel(mode: DataMode | null | undefined): string {
  switch (mode) {
    case 'live':
      return 'Live'
    case 'stale':
      return 'Stale'
    case 'demo_seeded':
      return 'Demo data'
    case 'static_archive':
      return 'Archive'
    case 'empty':
      return 'No data'
    default:
      return 'Unknown'
  }
}

/** Tailwind classes for the provenance chip. */
export function modeChip(mode: DataMode | null | undefined): string {
  switch (mode) {
    case 'live':
      return 'bg-green-100 text-green-800 border-green-300'
    case 'stale':
      return 'bg-amber-100 text-amber-800 border-amber-300'
    case 'demo_seeded':
      return 'bg-sky-100 text-sky-800 border-sky-300'
    case 'empty':
      return 'bg-slate-100 text-slate-500 border-slate-200'
    default:
      return 'bg-slate-100 text-slate-600 border-slate-300'
  }
}
