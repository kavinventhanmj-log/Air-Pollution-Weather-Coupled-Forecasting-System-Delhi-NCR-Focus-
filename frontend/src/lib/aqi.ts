export interface AqiStyle {
  label: string
  hex: string
  chip: string   // tailwind classes for badges (light theme)
  text: string   // tailwind text color
  bar: string    // solid background
}

// CPCB / MoEFCC air quality category thresholds (1-hour AQI, 8 pollutants).
// The Tailwind chip/text/bar classes below are kept visually consistent with each
// row's `hex` value (e.g. `#ca8a04` == Tailwind yellow-600 -> yellow classes).
export const AQI_CATEGORIES: Array<{ min: number; max: number } & AqiStyle> = [
  { min: 0, max: 50, label: 'Good', hex: '#16a34a', chip: 'bg-green-100 text-green-800 border-green-300', text: 'text-green-700', bar: 'bg-green-600' },
  { min: 51, max: 100, label: 'Satisfactory', hex: '#ca8a04', chip: 'bg-yellow-100 text-yellow-800 border-yellow-300', text: 'text-yellow-700', bar: 'bg-yellow-600' },
  { min: 101, max: 200, label: 'Moderate', hex: '#d97706', chip: 'bg-amber-100 text-amber-800 border-amber-300', text: 'text-amber-700', bar: 'bg-amber-600' },
  { min: 201, max: 300, label: 'Poor', hex: '#dc2626', chip: 'bg-red-100 text-red-800 border-red-300', text: 'text-red-700', bar: 'bg-red-600' },
  { min: 301, max: 400, label: 'Very Poor', hex: '#9333ea', chip: 'bg-purple-100 text-purple-800 border-purple-300', text: 'text-purple-700', bar: 'bg-purple-600' },
  { min: 401, max: Infinity, label: 'Severe', hex: '#7f1d1d', chip: 'bg-red-200 text-red-900 border-red-400', text: 'text-red-900', bar: 'bg-red-900' },
]

// Resolve the canonical AQI hex for a category label (used by spatial heatmaps etc.).
export function aqiCategoryHex(category: string | null | undefined): string {
  if (!category) return '#94a3b8'
  return AQI_CATEGORIES.find((c) => c.label.toLowerCase() === category.toLowerCase())?.hex ?? '#94a3b8'
}

// Pick a readable text colour for a solid background hex (WCAG-ish luminance).
// Returns dark navy for light backgrounds, white for dark backgrounds.
export function readableOnHex(hex: string): string {
  const h = hex.replace('#', '')
  if (h.length < 6) return '#ffffff'
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return lum > 0.45 ? '#0f172a' : '#ffffff'
}

export function aqiStyle(aqi: number | null | undefined): AqiStyle {
  if (typeof aqi !== 'number' || Number.isNaN(aqi)) {
    return { label: 'No data', hex: '#94a3b8', chip: 'bg-slate-100 text-slate-500 border-slate-200', text: 'text-slate-400', bar: 'bg-slate-300' }
  }
  // The CPCB bands above are integer-edged (0-50, 51-100, ...) so a fractional
  // value such as 50.5 falls into the gap between two ranges and would fall
  // through to the Severe fallback. Interpolated fields (IDW grid/dispersion
  // AQI) and the 1-decimal NCR average are both genuinely fractional, so floor
  // first and resolve the value into the band it rounds down to. This mirrors
  // `get_aqi_category` in backend/app/services/aqi_calculator.py, which also
  // resolves in-between values into the lower band. Band edges are unchanged.
  const whole = Math.floor(aqi)
  return AQI_CATEGORIES.find((c) => whole >= c.min && whole <= c.max) ?? AQI_CATEGORIES[5]
}

export function aqiCategory(aqi: number | null | undefined, fallback = 'No data'): string {
  if (typeof aqi !== 'number' || Number.isNaN(aqi)) return fallback
  return aqiStyle(aqi).label
}

export function fmt(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '--'
  return v.toFixed(digits)
}

export function tsFmt(s: string | null | undefined, showSeconds = false): string {
  if (!s) return '--'
  const cleaned = s.endsWith('Z') || s.includes('+00:00') ? s : `${s}Z`
  const d = new Date(cleaned)
  if (Number.isNaN(d.getTime())) return s.replace('T', ' ').slice(0, 16)
  const p = (n: number) => String(n).padStart(2, '0')
  const local =
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
  return showSeconds ? `${local}:${p(d.getSeconds())}` : local
}