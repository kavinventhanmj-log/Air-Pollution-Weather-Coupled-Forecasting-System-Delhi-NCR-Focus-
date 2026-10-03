import { MapContainer, TileLayer, Popup, CircleMarker, Polyline, Circle, Polygon } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import type { Station, FireHotspot, PollutionReading } from '../types'
import { aqiStyle } from '../lib/aqi'
import {
  DELHI_NCR_CENTROID,
  INFLUENCE_RADIUS_KM,
  NCR_MODELING_BOUNDARY,
  type TransportPathway,
} from '../lib/geo'

const DELHI_CENTER: [number, number] = [28.6139, 77.209]

interface WindVector {
  lat: number
  lon: number
  direction_deg: number | null
  speed: number | null
}

interface Props {
  stations: Station[]
  onSelectStation?: (name: string) => void
  fires?: FireHotspot[]
  pollution?: PollutionReading[]
  wind?: WindVector[]
  pathways?: TransportPathway[]
}

function frpColor(frp?: number | null): string {
  const v = frp ?? 1
  if (v >= 100) return '#ff0000'
  if (v >= 50) return '#ff7f00'
  if (v >= 20) return '#ffd700'
  return '#ff4444'
}

function frpRadius(frp?: number | null): number {
  const v = frp ?? 1
  return Math.min(14, Math.max(3, 2 + Math.log10(1 + v) * 2.5))
}

function aqiColor(aqi?: number | null): string {
  return aqiStyle(aqi ?? null).hex
}

function windArrowPoints(w: WindVector): [number, number][] {
  // Meteorological direction: angle the wind blows FROM (0=N, clockwise).
  // The arrow shows the flow direction (TO = FROM + 180).
  const speed = w.speed ?? 1
  const deg = w.direction_deg ?? 0
  const towardRad = ((deg + 180) % 360) * (Math.PI / 180)
  const len = Math.min(40, 6 + speed * 3)
  const latScale = len / 111320
  const lonScale = len / (111320 * Math.cos((w.lat * Math.PI) / 180))
  const tip: [number, number] = [w.lat + latScale * Math.cos(towardRad), w.lon + lonScale * Math.sin(towardRad)]
  const back: [number, number] = [w.lat - latScale * 0.45 * Math.cos(towardRad), w.lon - lonScale * 0.45 * Math.sin(towardRad)]
  const headRad = (Math.PI * 5) / 6
  const headL = len * 0.3
  const left: [number, number] = [
    tip[0] + headL * Math.cos(towardRad + headRad) * (latScale / lonScale || 1),
    tip[1] + headL * Math.sin(towardRad + headRad),
  ]
  const right: [number, number] = [
    tip[0] + headL * Math.cos(towardRad - headRad) * (latScale / lonScale || 1),
    tip[1] + headL * Math.sin(towardRad - headRad),
  ]
  return [back, tip, left, tip, right]
}

function keyOf(lat: number, lon: number): string {
  return `${lat.toFixed(3)},${lon.toFixed(3)}`
}

export default function StationMap({ stations, onSelectStation, fires = [], pollution = [], wind = [], pathways = [] }: Props) {
  const latest = new Map<number, PollutionReading>()
  pollution.forEach(p => {
    if (!latest.has(p.station_id)) latest.set(p.station_id, p)
  })

  const pathwayOrigins = new Set(pathways.map((p) => keyOf(p.from.lat, p.from.lon)))
  const ncrBoundary: [number, number][] = NCR_MODELING_BOUNDARY.map((p) => [p.lat, p.lon])

  return (
    <MapContainer
      center={DELHI_CENTER}
      zoom={10}
      dragging={true}
      touchZoom={true}
      scrollWheelZoom={true}
      doubleClickZoom={true}
      boxZoom={true}
      zoomControl={true}
      inertia={true}
      worldCopyJump={true}
      className="h-96 rounded-xl z-0"
    >
      <TileLayer
        attribution='&copy; OpenStreetMap contributors &copy; CARTO'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />
      <Polygon
        positions={ncrBoundary}
        pathOptions={{ color: '#0f766e', weight: 1.5, dashArray: '5 4', fill: true, fillColor: '#0f766e', fillOpacity: 0.04 }}
      >
        <Popup>
          <div className="text-sm">
            <p className="font-bold text-slate-900">NCR modelling domain</p>
            <p className="text-xs text-slate-500">
              28.2–28.9°N, 76.6–77.5°E — the ~2.2 km numerical grid used by grid_service.py and
              the dispersion solver (not an administrative boundary).
            </p>
          </div>
        </Popup>
      </Polygon>
      <Circle
        center={[DELHI_NCR_CENTROID.lat, DELHI_NCR_CENTROID.lon]}
        radius={INFLUENCE_RADIUS_KM * 1000}
        pathOptions={{ color: '#b45309', weight: 1.5, dashArray: '3 5', fill: false }}
      >
        <Popup>
          <div className="text-sm">
            <p className="font-bold text-slate-900">500 km influence ring</p>
            <p className="text-xs text-slate-500">
              FIRMS fire hotspots within this radius are considered regional and feed the
              fire-transport / regional-transport coupling features.
            </p>
          </div>
        </Popup>
      </Circle>
      {pathways.length > 0 && (
        <CircleMarker
          center={DELHI_CENTER}
          pathOptions={{ color: '#b45309', weight: 2, fill: false, dashArray: '4 3' }}
          radius={16}
        >
          <Popup>
            <div className="text-sm">
              <p className="font-bold text-slate-900">Delhi NCR (centroid)</p>
              <p className="text-xs text-slate-500">Dashed lines mark the highest-FRP upwind fires whose smoke regional winds would advect toward NCR (advective estimate, not a plume chemistry model).</p>
            </div>
          </Popup>
        </CircleMarker>
      )}
      {pathways.map((p, i) => (
        <Polyline
          key={`pathway-${i}`}
          positions={[[p.from.lat, p.from.lon], [p.to.lat, p.to.lon]]}
          pathOptions={{ color: '#b45309', weight: 1.5, dashArray: '6 6', opacity: 0.7 }}
        />
      ))}
      {fires.filter((f) => !f.synthetic).map((f, i) => (
        <CircleMarker
          key={`${f.lat}-${f.lon}-${i}`}
          center={[f.lat, f.lon]}
          pathOptions={{ color: frpColor(f.frp), fillColor: frpColor(f.frp), fillOpacity: 0.55 }}
          radius={frpRadius(f.frp)}
        >
          <Popup>
            <div className="text-xs">
              <p className="font-bold">Live FIRMS Hotspot</p>
              <p>FRP: {f.frp?.toFixed(1) ?? '--'} MW</p>
              <p>Confidence: {f.confidence ?? '--'}</p>
              {pathwayOrigins.has(keyOf(f.lat, f.lon)) && (
                <p className="text-amber-700">Winds would advect this smoke toward Delhi NCR (estimate)</p>
              )}
              {f.acq_date && <p>{String(f.acq_date).slice(0, 16)}</p>}
              {f.source && <p className="text-slate-500">source: {f.source}</p>}
            </div>
          </Popup>
        </CircleMarker>
      ))}
      {fires.filter((f) => f.synthetic).map((f, i) => (
        <CircleMarker
          key={`syn-${f.lat}-${f.lon}-${i}`}
          center={[f.lat, f.lon]}
          pathOptions={{ color: '#9ca3af', weight: 1, dashArray: '3 3', fillColor: '#d1d5db', fillOpacity: 0.5 }}
          radius={frpRadius(f.frp)}
        >
          <Popup>
            <div className="text-xs">
              <p className="font-bold text-slate-600">Simulated hotspot (not a live detection)</p>
              <p>FRP: {f.frp?.toFixed(1) ?? '--'} MW</p>
              <p>2023–24 synthetic training history — shown only because the simulated overlay is enabled.</p>
              {f.acq_date && <p>{String(f.acq_date).slice(0, 10)}</p>}
              {f.source && <p className="text-slate-500">source: {f.source}</p>}
            </div>
          </Popup>
        </CircleMarker>
      ))}
      {fires.filter((f) => !f.synthetic).map((f, i) => {
        if (!pathwayOrigins.has(keyOf(f.lat, f.lon))) return null
        return (
          <CircleMarker
            key={`ring-${i}`}
            center={[f.lat, f.lon]}
            pathOptions={{ color: '#b45309', weight: 1.5, fill: false, dashArray: '4 3' }}
            radius={frpRadius(f.frp) + 4}
          />
        )
      })}
      {wind.map((w, i) => (
        <Polyline
          key={`wind-${i}`}
          positions={windArrowPoints(w)}
          pathOptions={{ color: '#0891b2', weight: 2, opacity: 0.8 }}
        />
      ))}
      {stations.map(s => {
        const reading = latest.get(s.id)
        return (
          <CircleMarker
            key={s.id}
            center={[s.latitude, s.longitude]}
            pathOptions={{ color: aqiColor(reading?.aqi), fillColor: aqiColor(reading?.aqi), fillOpacity: 0.85 }}
            radius={9}
            eventHandlers={{ click: () => onSelectStation?.(s.name) }}
          >
            <Popup>
              <div className="text-sm">
                <p className="font-bold text-slate-900">{s.name}</p>
                <p className="text-slate-500">{s.city}{s.state ? `, ${s.state}` : ''}</p>
                {reading ? (
                  <>
                    <p className="text-slate-800">Recorded AQI: {reading.aqi?.toFixed(0) ?? '--'}</p>
                    <p className="text-slate-800">PM2.5: {reading.pm25?.toFixed(1) ?? '--'} μg/m³</p>
                    <p className="text-slate-500">{reading.timestamp.slice(0, 16)}</p>
                  </>
                ) : (
                  <p className="text-slate-400">No pollution reading yet</p>
                )}
              </div>
            </Popup>
          </CircleMarker>
        )
      })}
    </MapContainer>
  )
}