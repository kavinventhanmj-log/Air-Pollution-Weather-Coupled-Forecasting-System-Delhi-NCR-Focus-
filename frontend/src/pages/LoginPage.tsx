import { FormEvent, useEffect, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { Activity, AlertCircle, Eye, EyeOff, Loader2, LockKeyhole, Mail } from 'lucide-react'
import { useAuth } from '../auth/AuthContext'
import { isTransientStatus, getDemoCredentials, resilientGet } from '../api/client'
import type { DemoCredentials } from '../types'

// Edge-served copy of the demo credential, written at build time by
// `scripts/generate-demo-credentials.mjs`. Vercel serves the built bundle from
// its own CDN, so reading this costs a same-origin request in milliseconds,
// whereas `/api/auth/demo` only answers once the Render free-tier container is
// serving -- measured at 83,397 ms cold against 435 ms warm, because the
// keepalive cron meant to hold it awake runs on a median 260 min cadence
// against an intended 5 min. A plain `fetch` is used rather than the axios
// instance: this is a static asset, not an API call, so it must not be queued
// behind the shared warm-up gate or rewritten by the `/api/:path*` proxy.
const DEMO_MANIFEST_URL = '/demo-credentials.json'

async function readDemoManifest(signal: AbortSignal): Promise<DemoCredentials | null> {
  try {
    const res = await fetch(DEMO_MANIFEST_URL, { signal, cache: 'no-store' })
    if (!res.ok) return null
    const data = await res.json()
    if (data?.enabled !== true) return null
    if (typeof data.email !== 'string' || typeof data.password !== 'string') return null
    return {
      email: data.email,
      password: data.password,
      name: typeof data.name === 'string' ? data.name : '',
      role: typeof data.role === 'string' ? data.role : '',
    }
  } catch {
    // No manifest, or one that is not JSON: `vercel.json` rewrites unmatched
    // paths to /index.html, so an absent file arrives as HTML with a 200 and
    // `res.json()` throws. Either way the endpoint below is still authoritative.
    return null
  }
}

export default function LoginPage() {
  const { user, token, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const params = new URLSearchParams(location.search)
  const next = params.get('next')

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [status, setStatus] = useState<string | null>(null)

  // The demo login is fetched from the backend rather than embedded in the
  // bundle. Publishing the credential in client source put the password in
  // front of every visitor of a production deployment, which is why
  // ``/api/auth/demo`` returns 404 whenever the demo account is switched off.
  // In development / staging the endpoint answers and the button appears;
  // in production it stays hidden and no credential is shipped.
  const [demo, setDemo] = useState<DemoCredentials | null>(null)

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    // Prefer the edge-served manifest, and only fall back to the endpoint when
    // it carries no credential. That keeps the demo affordance visible on the
    // first paint instead of after a cold start, while `/api/auth/demo` stays
    // authoritative: a deployment without the manifest, or one that writes
    // `{"enabled": false}`, still resolves through the endpoint exactly as
    // before, so switching the demo off on the backend still hides the button.
    void (async () => {
      const fromManifest = await readDemoManifest(controller.signal)
      if (cancelled) return
      if (fromManifest) {
        setDemo(fromManifest)
        return
      }

      // Retry through the cold-start window instead of fetching once.
      //
      // A single fire-and-forget request made the demo button's visibility
      // depend on whether the backend happened to be awake: one transient 502/503
      // during a Render wake left `demo` null for the rest of the page's life, so
      // the button never appeared and the demo looked broken. `resilientGet`
      // waits on the shared warm-up gate and retries, and still lets a genuine
      // 404 (demo disabled) through immediately, so the credential stays hidden
      // in production exactly as before.
      try {
        const res = await resilientGet(getDemoCredentials)
        if (!cancelled) setDemo(res.data)
      } catch {
        // 404 (demo disabled) or a genuinely unavailable backend: keep the demo
        // affordance hidden. Manual sign-in always remains available.
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [])

  if (user && token) {
    return <Navigate to={next && !next.startsWith('/login') ? next : '/dashboard'} replace />
  }

  function isTransient(error: any) {
    if (!error || !error.response) return true
    return isTransientStatus(error.response.status)
  }

  // Render free back-ends sleep after ~15 min idle and can take 45-120 s to
  // wake; the first sign-in attempts are answered with gateway 502/503/429
  // while the container boots. Keep retrying inside that wake window so demo
  // sign-in works on the first click even right after a cold start.
  const WAKE_MAX_ATTEMPTS = 12

  async function signIn(useEmail: string, usePassword: string) {
    for (let attempt = 1; attempt <= WAKE_MAX_ATTEMPTS; attempt++) {
      try {
        setStatus(attempt === 1 ? 'Signing in…' : 'Server is waking up — retrying…')
        await login(useEmail.trim(), usePassword)
        navigate(next && !next.startsWith('/login') ? next : '/dashboard', { replace: true })
        return
      } catch (err) {
        if (attempt === WAKE_MAX_ATTEMPTS || !isTransient(err)) throw err
        setError(
          (err as any)?.response?.status === 429
            ? 'Server is rate-limiting requests — retrying in a moment…'
            : 'Air-quality server is waking up — signing you in shortly…',
        )
        setSubmitting(true)
        // Back off progressively so a rate-limited instance is not hammered
        // into staying rate-limited, while still retrying quickly enough to
        // land within the wake window.
        await new Promise((r) => setTimeout(r, Math.min(1000 * 2 ** (attempt - 1), 8000)))
      }
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setStatus(null)
    const useEmail = email.trim() || demo?.email || ''
    const usePassword = password || demo?.password || ''
    if (!useEmail || !usePassword) {
      setError('Please enter your email and password.')
      return
    }
    setSubmitting(true)
    try {
      await signIn(useEmail, usePassword)
    } catch (err) {
      if (isTransient(err)) {
        setError('Server is still waking up — please try again in a moment.')
      } else {
        setError('Invalid email or password. Please try again.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  async function demoSignIn() {
    if (!demo) return
    setError(null)
    setStatus(null)
    setSubmitting(true)
    try {
      await signIn(demo.email, demo.password)
    } catch (err) {
      if (isTransient(err)) {
        setError('Server is still waking up — please try again in a moment.')
      } else {
        setError('Demo sign-in failed. Please try again.')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-inst-50/60">
      <header className="border-b border-white/10 bg-inst-800 text-white">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-4">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/10">
            <Activity className="h-6 w-6 text-cyan-300" aria-hidden="true" />
          </span>
          <div>
            <p className="text-sm font-bold leading-tight uppercase tracking-wide">AeroCast-NCR</p>
            <p className="text-xs text-inst-200">National Air Quality Forecasting · Delhi NCR</p>
          </div>
          <span className="ml-auto hidden rounded-full border border-inst-300/40 px-3 py-1 text-xs text-inst-100 sm:inline-block">
            SIH 2026 · PS SIH26082
          </span>
        </div>
      </header>

      <main className="flex flex-1 items-center justify-center px-4 py-12">
        <div className="w-full max-w-md">
          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 bg-slate-50 px-8 py-6 text-center">
              <h1 className="text-xl font-bold text-slate-900">Analyst Sign In</h1>
              <p className="mt-1 text-sm text-slate-500">
                Restricted access for authorised air-quality analysts
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5 px-8 py-7" aria-label="Sign in">
              <div>
                <label htmlFor="email" className="mb-1.5 block text-sm font-medium text-slate-700">
                  Email address
                </label>
                <div className="relative">
                  <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
                  <input
                    id="email"
                    type="email"
                    autoComplete="username"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="analyst@aerocast.in"
                    className="w-full rounded-lg border border-slate-300 bg-white py-2.5 pl-10 pr-3 text-sm text-slate-900 outline-none ring-inst-600/30 focus:border-inst-500 focus:ring-2"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="password" className="mb-1.5 block text-sm font-medium text-slate-700">
                  Password
                </label>
                <div className="relative">
                  <LockKeyhole className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
                  <input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full rounded-lg border border-slate-300 bg-white py-2.5 pl-10 pr-10 text-sm text-slate-900 outline-none ring-inst-600/30 focus:border-inst-500 focus:ring-2"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 hover:text-slate-600"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {error && (
                <div
                  role="alert"
                  className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-700"
                >
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  <span>{error}</span>
                </div>
              )}

              {demo && (
                <button
                  type="button"
                  onClick={demoSignIn}
                  disabled={submitting}
                  className="flex w-full items-center justify-center gap-2 rounded-lg border border-inst-300 bg-inst-50 px-4 py-2.5 text-sm font-semibold text-inst-800 transition-colors hover:bg-inst-100 focus:outline-none focus:ring-2 focus:ring-inst-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {submitting ? (
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  ) : (
                    <Activity className="h-4 w-4" aria-hidden="true" />
                  )}
                  {submitting ? (status ?? 'Signing in…') : 'Continue with Demo Account'}
                </button>
              )}

              {submitting && (
                <p className="text-center text-xs text-slate-500">
                  Cold starts on the free tier can take up to 2 minutes. This page will
                  sign you in automatically — no need to click again.
                </p>
              )}

              {demo && (
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  <span className="h-px flex-1 bg-slate-200" />
                  OR
                  <span className="h-px flex-1 bg-slate-200" />
                </div>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="w-full rounded-lg bg-inst-700 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-inst-800 focus:outline-none focus:ring-2 focus:ring-inst-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {submitting ? 'Signing in…' : 'Sign in'}
              </button>
            </form>
          </div>

          {demo && (
            <div className="mt-4 rounded-xl border border-inst-100 bg-inst-50 px-6 py-4">
                <p className="text-sm font-semibold text-inst-900">Hackathon demo access</p>
                <p className="mt-1 break-all text-xs leading-relaxed text-inst-800">
                  Email: <span className="font-mono">{demo.email}</span>&nbsp;· Password:{' '}
                  <span className="font-mono">{demo.password}</span>
                </p>
                <p className="mt-1 text-xs text-inst-700">
                  These are shown for reference — just press Enter or click
                  &quot;Continue with Demo Account&quot; to sign in instantly.
                </p>
              </div>
          )}

          <p className="mt-6 text-center text-xs leading-relaxed text-slate-500">
            Prototype developed for Smart India Hackathon 2026 — SIH26082.
            <br />
            Not an official Government of India service.
          </p>
        </div>
      </main>
    </div>
  )
}