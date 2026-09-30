import { api } from './client'

// Render free-tier instances scale to zero after ~15 min idle. While any
// browser tab is open this pinger polls the cheap liveness endpoint on a
// cadence comfortably inside that window, so the demo never eats a full cold
// boot mid-session. The ping is fire-and-forget: failures (including the 429s
// Render emits while an instance is still waking) are swallowed here because
// the shared warm-up gate in client.ts already owns the retry-on-wake logic.
let started = false

const DEFAULT_INTERVAL_MS = 180_000

export function startKeepWarm(intervalMs: number = DEFAULT_INTERVAL_MS): void {
  if (started) return
  started = true
  const ping = () => {
    // Never ping a hidden tab — background tabs do not need the instance kept
    // warm, and a ping per hidden tab is pure rate-limit pressure.
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return
    // `/health` is the DB-free liveness probe; `/health` at the server root
    // (bare `/api/health`) would add a Postgres round-trip to every keep-alive
    // ping, which is wasted work on a 0.5-CPU instance and cannot succeed any
    // faster than the process itself.
    //
    // The path is relative to `baseURL: '/api'` (client.ts), so it must be
    // `/health` and not `/api/health` - the latter resolved to `/api/api/health`
    // and 404'd on every ping, leaving the instance to scale to zero anyway.
    api.get('/health', { timeout: 15000 }).catch(() => {
      // Backend sleeping or waking; the axios warm-up gate handles recovery.
    })
  }
  ping()
  setInterval(ping, intervalMs)
}
