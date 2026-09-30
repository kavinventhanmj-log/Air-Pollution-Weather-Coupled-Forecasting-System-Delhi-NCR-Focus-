import type { InversionData } from '../types'

function layerLabel(data: InversionData): string {
  if (data.inversion_base_pressure && data.inversion_top_pressure) {
    return `${data.inversion_base_pressure.toFixed(0)}–${data.inversion_top_pressure.toFixed(0)} hPa`
  }
  return '--'
}

export default function InversionPanel({ data }: { data: InversionData | null }) {
  if (!data) return <div className="card"><p className="text-slate-500">Loading inversion data...</p></div>

  const strengthColor = {
    'Strong': 'bg-red-600',
    'Moderate': 'bg-orange-500',
    'Weak': 'bg-amber-500',
    'None': 'bg-green-600',
  }[data.inversion_strength] || 'bg-slate-400'

  const riskColor = {
    'HIGH': 'text-red-700',
    'MEDIUM': 'text-orange-700',
    'LOW': 'text-amber-700',
    'MINIMAL': 'text-green-700',
  }[data.trapping_risk] || 'text-slate-500'

  const dispColor = {
    'TRAPPED': 'text-red-700',
    'LIMITED': 'text-orange-700',
    'MODERATE': 'text-amber-700',
    'GOOD': 'text-green-700',
  }[data.dispersion_condition ?? ''] || 'text-slate-500'

  const pct = data.inversion_strength_score != null ? Math.round(data.inversion_strength_score * 100) : null
  // Drive the bar from the numeric strength score rather than from label buckets,
  // so it agrees with the percentage printed right next to it. The label buckets
  // gave every Moderate inversion the same 65% regardless of its actual score,
  // and disagreed with the score whenever the two fell in different buckets.
  // Keep the label buckets only as the fallback when no score is available.
  const width = pct !== null
    ? `${Math.max(2, Math.min(100, pct))}%`
    : data.inversion_strength === 'Strong'
    ? '100%'
    : data.inversion_strength === 'Moderate'
    ? '65%'
    : data.inversion_strength === 'Weak'
    ? '35%'
    : '10%'

  return (
    <div className="card">
      <h3 className="card-header">Inversion Status</h3>
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-500">Strength</span>
            <span className="font-medium text-slate-900">{data.inversion_strength}{pct !== null ? ` (${pct}%)` : ''}</span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-3">
            <div className={`${strengthColor} h-3 rounded-full transition-all`} style={{ width }} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-slate-500">PBL Height: </span>
            <span className="font-medium text-slate-900">{data.pbl_height ? `${data.pbl_height}m` : '--'}</span>
          </div>
          <div>
            <span className="text-slate-500">Inversion layer: </span>
            <span className="font-medium text-slate-900">{data.inversion_source === 'lapse_rate' ? layerLabel(data) : '--'}</span>
          </div>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-500">Basis</span>
          <span className="font-medium text-slate-900">
            {data.inversion_source === 'lapse_rate'
              ? `Vertical lapse rate (${data.strongest_layer_gradient?.toFixed(2) ?? '--'} K/100hPa)`
              : 'PBL proxy (no vertical profile)'}
          </span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-500">Pollution Trapping</span>
          <span className={`font-bold ${riskColor}`}>{data.trapping_risk}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-500">Dispersion Condition</span>
          <span className={`font-bold ${dispColor}`}>{data.dispersion_condition ?? '--'}</span>
        </div>
      </div>
    </div>
  )
}