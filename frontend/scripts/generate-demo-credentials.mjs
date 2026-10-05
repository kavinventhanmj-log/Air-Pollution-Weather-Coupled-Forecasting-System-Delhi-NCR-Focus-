/**
 * Write `public/demo-credentials.json` at build time so the login page can show
 * the demo account without waiting on the backend.
 *
 * Why this exists
 * ---------------
 * The demo credential used to be fetched from `GET /api/auth/demo`, which only
 * answers once the FastAPI container is serving. On Render's free tier that
 * instance sleeps after ~15 min idle and takes 45-120 s to wake, and the
 * `keepalive` workflow meant to hold it awake declares a 5-minute cron but
 * actually runs on a median 260 min cadence (60 runs measured; zero failures, so
 * it is GitHub throttling the cron rather than the job breaking). Measured
 * against production: 83,397 ms for the first request, 435 ms once warm. The demo
 * affordance was therefore invisible on the login page for well over a minute,
 * and nothing in the frontend could fix it -- the credential only existed
 * behind the booting container.
 *
 * Vercel serves the built bundle from its own CDN, so reading the credential
 * from a static file costs a same-origin request in milliseconds regardless of
 * whether Render is awake.
 *
 * Security posture
 * ----------------
 * This does not make the credential secret that it was not already. It is
 * published in `render.yaml` in this public repository, and any visitor could
 * read it by waiting out the cold start. What it does change is that the value
 * ships in the built frontend, so it is only written when the build is
 * explicitly configured for it:
 *
 *   ENABLE_DEMO_USER=true        opt in; anything else (including unset) writes
 *                                `{"enabled": false}`, matching the backend where
 *                                `enable_demo_user` unset means "off in
 *                                production"
 *   DEMO_USER_PASSWORD          required when enabled
 *   DEMO_USER_EMAIL              defaults to the backend's own default
 *   DEMO_USER_NAME / _ROLE       optional, default to the backend's defaults
 *
 * `/api/auth/demo` remains authoritative: the frontend only uses this file as a
 * fast path and still falls back to the endpoint when the file carries no
 * credential, so turning the demo off on the backend still hides the affordance.
 *
 * The output file is git-ignored; it is a build artifact, never a tracked one.
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// Mirrors `backend/app/config.py`. Keeping the defaults here in step with the
// backend means a Vercel project only has to set ENABLE_DEMO_USER and the
// password, and cannot end up advertising a different address than the one the
// API actually seeded.
const DEFAULT_EMAIL = 'analyst@aerocast.in'
const DEFAULT_NAME = 'Demo Analyst'
const DEFAULT_ROLE = 'Analyst'

const OUT_FILE = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'demo-credentials.json')

// Same spelling as `Settings.demo_user_enabled`: an explicit value wins, and
// unset means off.
const TRUTHY = new Set(['1', 'true', 'yes', 'on'])

function readEnv(name) {
  const raw = process.env[name]
  return typeof raw === 'string' ? raw.trim() : ''
}

function main() {
  const enabled = TRUTHY.has(readEnv('ENABLE_DEMO_USER').toLowerCase())
  const password = readEnv('DEMO_USER_PASSWORD')
  const email = readEnv('DEMO_USER_EMAIL') || DEFAULT_EMAIL

  // Written unconditionally, including when the build has no credential at all.
  // `vercel.json` rewrites every unmatched path to `/index.html`, so a *missing*
  // manifest reaches the browser as HTML with a 200 status; a real JSON file
  // that happens to say "disabled" is unambiguous.
  const payload = enabled && password
    ? {
        enabled: true,
        email,
        password,
        name: readEnv('DEMO_USER_NAME') || DEFAULT_NAME,
        role: readEnv('DEMO_USER_ROLE') || DEFAULT_ROLE,
      }
    : { enabled: false }

  mkdirSync(dirname(OUT_FILE), { recursive: true })
  writeFileSync(OUT_FILE, `${JSON.stringify(payload, null, 2)}\n`, 'utf8')

  if (payload.enabled) {
    console.log('[demo-credentials] wrote public/demo-credentials.json (demo enabled)')
  } else {
    // Say which reason applied -- a silent "disabled" is how the SIH demo
    // deployment ended up advertising nothing in the first place.
    const why = !enabled
      ? 'ENABLE_DEMO_USER is not set to a truthy value'
      : 'DEMO_USER_PASSWORD is empty'
    console.log(`[demo-credentials] wrote public/demo-credentials.json as disabled (${why})`)
  }
}

try {
  main()
} catch (error) {
  // A hint that loads slowly must never be the reason a build fails.
  console.warn(`[demo-credentials] skipped: ${error instanceof Error ? error.message : error}`)
}