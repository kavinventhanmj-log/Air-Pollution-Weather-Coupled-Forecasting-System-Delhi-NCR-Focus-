import type { PlumeRisk } from '../types'

export default function StubblePlume({ data }: { data: PlumeRisk | null }) {
  if (!data) return <div className="card"><p className="text-slate-500">Loading fire data...</p></div>
  
  const riskColor = {
    'HIGH': 'bg-red-100 text-red-800',
    'MODERATE': 'bg-amber-100 text-amber-800',
    'LOW': 'bg-green-100 text-green-800',
  }[data.risk_level] || 'bg-slate-100 text-slate-700'

  return (
    <div className="card">
      <h3 className="card-header">Regional Fire-Plume Impact</h3>
      <div className="space-y-4">
        <div className={`text-center py-3 rounded-lg ${riskColor}`}>
          <p className="text-2xl font-bold uppercase">{data.risk_level}</p>
          <p className="text-sm">Delhi NCR Impact</p>
        </div>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-slate-500">Fire Count</p>
            <p className="text-lg font-bold text-slate-900">{data.fire_count}</p>
          </div>
          <div>
            <p className="text-slate-500">Transport</p>
            <p className="text-lg font-bold text-slate-900">{data.transport_direction}</p>
          </div>
          <div>
            <p className="text-slate-500">Nearest Fire</p>
            <p className="text-lg font-bold text-slate-900">{data.distance_nearest_fire}km</p>
          </div>
          <div>
            <p className="text-slate-500">Confidence</p>
            <p className="text-lg font-bold text-slate-900">{(data.confidence * 100).toFixed(0)}%</p>
          </div>
          {data.wind_alignment_pct != null && (
            <div>
              <p className="text-slate-500">Winds Aligned</p>
              <p className="text-lg font-bold text-slate-900">{data.wind_alignment_pct.toFixed(0)}%</p>
            </div>
          )}
          {data.transport_time_hours != null && (
            <div>
              <p className="text-slate-500">Smoke Arrival</p>
              <p className="text-lg font-bold text-slate-900">{data.transport_time_hours.toFixed(1)}h</p>
            </div>
          )}
          {data.stubble_impact_score != null && (
            <div>
              <p className="text-slate-500">Stubble Smoke Proxy</p>
              <p className="text-lg font-bold text-slate-900">{data.stubble_impact_score.toFixed(2)}</p>
            </div>
          )}
          {data.estimated_pm25_contribution_ugm3 != null && (
            <div className="col-span-2">
              <p className="text-slate-500">Estimated smoke-attributed PM2.5 contribution</p>
              <p className="text-lg font-bold text-slate-900">+{data.estimated_pm25_contribution_ugm3.toFixed(1)} μg/m³</p>
              <p className="text-xs text-slate-500">Proxy estimate from FIRMS intensity · wind alignment · proximity — not a measured quantity</p>
            </div>
          )}
        </div>
      </div>
      {data.fire_basis != null && (
        <p className="mt-3 text-xs text-slate-500">
          Basis: {data.fire_basis}
          {data.synthetic_fire_count != null && data.synthetic_fire_count > 0
            ? ` (includes ${data.synthetic_fire_count} synthetic hotspots to fill the model window — real detections only in the headline figures above)`
            : ' — real FIRMS detections only'}
        </p>
      )}
    </div>
  )
}
