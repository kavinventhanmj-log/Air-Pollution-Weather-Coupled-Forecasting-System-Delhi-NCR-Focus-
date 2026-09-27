import type { GrapAssessment } from '../types'
import { readableOnHex } from '../lib/aqi'

export default function GrapPanel({ data }: { data: GrapAssessment | null }) {
  if (!data) {
    return (
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
        GRAP assessment unavailable — no current AQI data yet.
      </div>
    )
  }

  const active = data.status === 'ACTIVE'
  const statusColor = active
    ? { backgroundColor: data.color, color: readableOnHex(data.color) }
    : { backgroundColor: '#e2e8f0', color: '#334155' }

  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center gap-3">
        <span
          className="inline-flex items-center rounded-full px-4 py-1 text-sm font-bold"
          style={statusColor}
        >
          {data.title}
        </span>
        <span className="text-xs text-slate-600">
          {data.aqi != null ? `NCR avg AQI ${data.aqi}` : 'No AQI yet'} · {data.aqi_category ?? '—'} · GRAP {data.status === 'ACTIVE' ? 'operative' : 'not invoked'}
        </span>
      </div>

      <p className="text-sm text-slate-700 mt-3">{data.advisory}</p>

      {data.inversion_note && (
        <p className="text-xs text-amber-800 mt-2">🧊 {data.inversion_note}</p>
      )}
      {data.fire_note && (
        <p className="text-xs text-orange-800 mt-1">🔥 {data.fire_note}</p>
      )}

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Action measures
          </h4>
          <ul className="space-y-1.5 text-sm">
            {data.measures.map((m, i) => (
              <li key={i} className="flex gap-2 text-slate-700">
                <span className="text-inst-700">·</span>
                <span>{m}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Why this stage
          </h4>
          <ul className="space-y-1.5 text-sm">
            {data.rationale.map((r, i) => (
              <li key={i} className="flex gap-2 text-slate-600">
                <span className="text-inst-700">·</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
          <p className="text-[10px] text-slate-400 mt-3">{data.source} · {data.assessed_at?.replace('T', ' ').slice(0, 16)}Z</p>
        </div>
      </div>
    </div>
  )
}