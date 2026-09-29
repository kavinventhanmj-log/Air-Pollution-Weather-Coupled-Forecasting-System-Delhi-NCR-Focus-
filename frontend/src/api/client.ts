import axios, { type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'
import { latestForecastRun } from '../lib/forecast'
import type { Station, CurrentAQI, ForecastPoint, WeatherData, InversionData, FireActivity, FireHotspotsResponse, PlumeRisk, Explanation, ForecastExplanation, Alert, ModelMetric, CouplingData, CouplingFeaturesResponse, ForecastContextResponse, CoupledForecastResult, GridForecast, DispersionForecast, SummaryResponse, PollutionReading, PollutionIngestSummary, DataImportSummary, ModelPerformanceResponse, Pm25ForecastResponse, AtmosphereCurrentResponse, TransportRiskResponse, GrapAssessment, GrapStagesResponse, LoginResponse, AuthUser, DemoCredentials, PollutionEventsCurrent, DataQualityResponse, SystemResponse, ScenarioAnalysisRequest, ScenarioAnalysisResponse } from '../types'

const TOKEN_KEY = 'aerocast_token'
const USER_KEY = 'aerocast_user'

export const getStoredToken = () => sessionStorage.getItem(TOKEN_KEY) ?? localStorage.getItem(TOKEN_KEY)
export const getStoredUser = (): AuthUser | null => {
  const raw = sessionStorage.getItem(USER_KEY) ?? localStorage.getItem(USER_KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw) as AuthUser
  } catch {
    return null
  }
}
export const storeSession = (token: string, user: AuthUser) => {
  sessionStorage.setItem(TOKEN_KEY, token)
  sessionStorage.setItem(USER_KEY, JSON.stringify(user))
}
export const clearSession = () => {
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(USER_KEY)
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

// Render free-tier instances sleep after ~15 min idle and take tens of
// seconds to wake; a page load right after a cold start must not drop every
// panel because of the client timeout. The heavy control-room endpoints
// (/events/current, /dispersion/forecast, /data-quality, /transport-risk/current,
// /summary, ...) legitimately take 30-50 s on a cold cache, so the default
// timeout is set well above that. Transient wake failures are retried by the
// interceptor below.
const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

api.interceptors.request.use((config) => {
  const token = getStoredToken()
  if (token) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Render free-tier instances scale to zero after ~15 min idle; the first
// requests after a cold start are answered with gateway 502/503/504, a
// network timeout, or 429 while the container boots (often 45-120 s). Instead
// of every panel retrying independently (a request storm that hammers a
// 512 MB box and leaves a patchwork of dead panels), a SINGLE shared warm-up
// loop pings the cheap liveness probe until the backend is awake; every panel
// then retries its request. Idempotent GETs (and the two pure compute POSTs
// /forecast/coupled and /scenario/analysis, which only read stored data and
// compute) take part in this wake-and-retry.
// The budget must EXCEED the wake it is waiting for, or the gate is useless.
// A production Render cold wake measured 76 s and Render's own spin-up runs
// 45-120 s, so a 25 s budget guaranteed failure: the gate expired while the
// container was still booting, every panel then re-issued into a dead socket,
// and the page was stuck on "Connecting to the API server" for good. 150 s
// covers the slow end of Render's range with margin; the gate is only ever
// reached by a request that has already failed once, so a generous budget
// costs nothing when the backend is healthy (the first probe returns at once).
const MAX_WARM_ATTEMPTS = 60
const WARM_RETRY_DELAY_MS = 2500
const WARM_BUDGET_MS = 150_000
const WARM_PROBE_TIMEOUT_MS = 8000

// Per-request retries for a transient failure. Each retry waits behind the
// shared gate first, so this is a second line of defence behind
// WARM_BUDGET_MS rather than a way to escape a still-booting instance.
const MAX_RETRIES = 4

// A sleeping instance answers with 502/503 *and* 429 (Render throttles the
// wake-up burst). 429 used to fall through as a hard error, which is what put
// "Request failed with status code 429" and a permanent red "API offline" badge
// on screen even though the instance came up seconds later. 408/500 are
// included for the same reason — the edge proxy emits them while a cold boot is
// still in progress.
const TRANSIENT_STATUSES = new Set([0, 408, 429, 500, 502, 503, 504])

export const isTransientStatus = (status: number | undefined): boolean =>
  status === undefined || TRANSIENT_STATUSES.has(status)

// The Render free tier also rate-limits bursts. A dashboard mount fans out
// ~14 requests at once, which is exactly the shape that trips the limiter, so
// requests are queued behind a small in-flight cap and GETs are de-duplicated
// by URL. Panels still all load; they just stop arriving as a thundering herd.
const MAX_INFLIGHT = 4
const QUEUE_GAP_MS = 120

let inflight = 0
const waiters: Array<() => void> = []

function acquireSlot(): Promise<void> {
  if (inflight < MAX_INFLIGHT) {
    inflight += 1
    return Promise.resolve()
  }
  return new Promise<void>((resolve) => {
    waiters.push(() => {
      inflight += 1
      resolve()
    })
  })
}

function releaseSlot(): void {
  inflight -= 1
  const next = waiters.shift()
  if (next) next()
}

let warmUpPromise: Promise<boolean> | null = null

// The warm-up probe deliberately bypasses the axios instance.
//
// `ensureWarm` is called *from* the response interceptor below, so probing
// through `api` let the gate re-enter itself: a transient failure of the probe
// (502/503/429 while Render is waking) sent the interceptor to
// `await ensureWarm()`, which returned the very `warmUpPromise` that the probe
// was already awaiting. Nothing could ever settle, so any page opened during a
// cold start hung forever on its empty state instead of rendering. A bare
// same-origin `fetch` carries no interceptors, so the gate is structurally
// incapable of re-entering itself, and `AbortController` guarantees each
// attempt settles even if the socket hangs.
// The backend origin is public — it is already hardcoded in vercel.json and in
// the deployment docs — so probing it directly adds no secret to the bundle.
const DIRECT_API_ORIGIN = 'https://air-pollution-weather-coupled.onrender.com'

const WARM_PROBE_URLS = ['/api/health', `${DIRECT_API_ORIGIN}/api/health`]

async function probeOne(url: string, timeoutMs: number): Promise<boolean> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: controller.signal, cache: 'no-store' })
    return res.ok
  } catch {
    return false
  } finally {
    clearTimeout(timer)
  }
}

// The gate does NOT rely on the same-origin path alone. Real requests reach the
// backend through the Vercel rewrite, and that edge answers 502/504 with its own
// (shorter) timeout while Render is still booting — so the proxy reports "down"
// for an instance that has in fact just come up, and a gate that believed it
// would sit there for the whole budget and then give up. The backend origin is
// probed directly as well and either signal is enough, which keeps the gate
// independent of how the CDN happens to be behaving that minute.
async function probeAlive(timeoutMs: number): Promise<boolean> {
  const results = await Promise.all(WARM_PROBE_URLS.map((url) => probeOne(url, timeoutMs)))
  return results.some(Boolean)
}

async function ensureWarm(): Promise<boolean> {
  if (!warmUpPromise) {
    warmUpPromise = (async () => {
      const deadline = Date.now() + WARM_BUDGET_MS
      for (let i = 0; i < MAX_WARM_ATTEMPTS; i++) {
        if (await probeAlive(WARM_PROBE_TIMEOUT_MS)) return true
        const remaining = deadline - Date.now()
        if (remaining <= 0 || i === MAX_WARM_ATTEMPTS - 1) return false
        await sleep(Math.min(WARM_RETRY_DELAY_MS, remaining))
      }
      return false
    })().finally(() => { warmUpPromise = null })
  }
  return warmUpPromise
}

const IDEMPOTENT_POST = /^\/(forecast\/coupled|scenario\/analysis)(\?|$)/

// In-flight GET de-duplication: several panels ask for the same endpoint on one
// mount (e.g. /stations from the header, the map and the dashboard), and every
// duplicate is another unit of load on a 0.5-CPU instance.
const pendingGets = new Map<string, Promise<unknown>>()

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

// Honour an upstream Retry-After when present, otherwise back off so a
// rate-limited instance is not hammered into staying rate-limited.
function backoffFor(attempt: number, error: any): number {
  const header = error?.response?.headers?.['retry-after']
  const seconds = Number(header)
  if (Number.isFinite(seconds) && seconds > 0) return Math.min(seconds * 1000, 30000)
  return Math.min(1000 * 2 ** attempt, 15000)
}

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const config = error?.config as (InternalAxiosRequestConfig & { retryCount?: number }) | undefined
    if (!config) return Promise.reject(error)
    const method = (config.method ?? 'get').toLowerCase()
    const isGet = method === 'get'
    const isIdempotentPost = method === 'post' && typeof config.url === 'string' && IDEMPOTENT_POST.test(config.url)
    if (!isGet && !isIdempotentPost) return Promise.reject(error)
    const status = error?.response?.status
    if (!isTransientStatus(status)) return Promise.reject(error)
    if ((config.retryCount ?? 0) >= MAX_RETRIES) return Promise.reject(error)
    config.retryCount = (config.retryCount ?? 0) + 1
    // For a wake-up failure, gate every queued panel behind the single shared
    // warm-up loop first. For a plain 429 the instance is already up, so just
    // back off — paying the full wake-up wait there would stall every panel.
    if (status === 429) {
      await sleep(backoffFor(config.retryCount, error))
    } else {
      const warm = await ensureWarm()
      if (!warm) return Promise.reject(error)
    }
    return api.request(config)
  },
)

async function request<T>(
  method: 'get' | 'post',
  url: string,
  body?: unknown,
  config?: Record<string, unknown>,
): Promise<AxiosResponse<T>> {
  await acquireSlot()
  try {
    // Small stagger keeps a 14-request mount from hitting the limiter as one burst.
    if (method === 'post') await sleep(QUEUE_GAP_MS)
    return method === 'get'
      ? await api.get<T>(url, config)
      : await api.post<T>(url, body, config)
  } finally {
    releaseSlot()
  }
}

const get = <T>(url: string, config?: Record<string, unknown>): Promise<AxiosResponse<T>> => {
  const key = `${url}::${config?.params ? JSON.stringify(config.params) : ''}`
  const existing = pendingGets.get(key)
  if (existing) return existing as Promise<AxiosResponse<T>>
  const pending = (async (): Promise<AxiosResponse<T>> => {
    try {
      return await request<T>('get', url, undefined, config)
    } finally {
      pendingGets.delete(key)
    }
  })()
  pendingGets.set(key, pending)
  return pending
}

const post = <T>(url: string, body?: unknown, config?: Record<string, unknown>): Promise<AxiosResponse<T>> =>
  request<T>('post', url, body, config)

export const login = (email: string, password: string) =>
  post<LoginResponse>('/auth/login', { email, password })
export const getMe = () => get<AuthUser>('/auth/me')
export const logout = () => post<AuthUser>('/auth/logout')
export const getDemoCredentials = () => get<DemoCredentials>('/auth/demo')

export const getStations = () => get<Station[]>('/stations')
export const getPollutionLatest = () => get<PollutionReading[]>('/pollution/latest')
export const getPollutionStations = () => get<Station[]>('/pollution/stations')
export const getPollutionHistory = (stationId: number, limit = 168) =>
  get<PollutionReading[]>(`/pollution/${stationId}/history`, { params: { limit } })
export const ingestPollution = () => post<PollutionIngestSummary>('/pollution/ingest')
export const getCurrentAQI = (station: string) => get<CurrentAQI>(`/current/${station}`)
export const getForecast = async (station: string, hours = 72) => {
  const res = await get<ForecastPoint[]>(`/forecast/${station}`, { params: { hours } })
  return { ...res, data: latestForecastRun(res.data) }
}
export const getNCRForecast = (hours = 72) => get<Record<string, ForecastPoint[]>>(`/forecast/ncr`, { params: { hours } })
export const getWeather = (station: string) => get<WeatherData>(`/weather/${station}`)
export const getInversion = (station: string) => get<InversionData>(`/inversion/${station}`)
export const getFireActivity = () => get<FireActivity>('/fire-activity')
export const getFireHotspots = () => get<FireHotspotsResponse>('/fire/hotspots')
export const getPlumeRisk = () => get<PlumeRisk>('/plume-risk')
export const getExplanation = (station: string) => get<Explanation>(`/explanation/${station}`)
export const getForecastExplanation = (forecastId: number) =>
  get<ForecastExplanation>(`/forecast/${forecastId}/explanation`)
export const getPm25ForecastExplanation = (station: string, horizon = 24) =>
  get<ForecastExplanation>('/forecast/pm25/explanation', { params: { station_name: station, horizon } })
export const getCoupling = (station: string) => get<CouplingData>(`/coupling/${station}`)
export const getCouplingFeatures = (station: string) => get<CouplingFeaturesResponse>(`/coupling/features/${station}`)
export const getForecastContext = (station: string) => get<ForecastContextResponse>(`/forecast/${station}/context`)
export const generateCoupledForecast = (station: string, horizons?: number[]) =>
  post<CoupledForecastResult>('/forecast/coupled', { station_name: station, horizons: horizons ?? [1, 6, 12, 24, 48, 72] }, { timeout: 120000 })
export const getGridForecast = (horizon = 24) => get<GridForecast>('/grid/forecast', { params: { horizon_hours: horizon } })
// `frameHours` limits how many hourly AQI grids the server materialises. The
// solver always integrates the full horizon, but building the per-cell grid is
// what costs: 72 h x 1575 cells is 113,400 cell objects (~7 MB of JSON and ~10 s
// of serialisation on the 0.5-CPU instance). A client that renders one frame at
// a time asks for the hours it can actually display.
export const getDispersionForecast = (horizon = 72, startHour = 8, frameHours?: number[]) =>
  get<DispersionForecast>('/dispersion/forecast', {
    params: {
      horizon_hours: horizon,
      start_hour: startHour,
      ...(frameHours?.length ? { frame_hours: frameHours.join(',') } : {}),
    },
  })
export const getAlerts = (station?: string) =>
  get<Alert[]>('/alerts', station ? { params: { station } } : undefined)
export const getModelMetrics = () => get<ModelMetric[]>('/model/metrics')
export const getModelPerformance = (target = 'pm25') =>
  get<ModelPerformanceResponse>('/model/performance', { params: { target } })
export const getPm25Forecast = (station: string, hours = 72) =>
  get<Pm25ForecastResponse>('/forecast/pm25', { params: { station_name: station, hours } })
export const getAtmosphereCurrent = (station?: string) =>
  get<AtmosphereCurrentResponse>('/atmosphere/current', { params: station ? { station_name: station } : {} })
export const getTransportRisk = (hours = 72) =>
  get<TransportRiskResponse>('/transport-risk/current', { params: { hours } })
export const getSummary = () => get<SummaryResponse>('/summary')
export const getEvents = (station?: string, hours = 48) =>
  get<PollutionEventsCurrent>('/events/current', { params: { station_name: station, hours } })
export const getDataQuality = () => get<DataQualityResponse>('/data-quality')
export const getSystemStatus = () => get<SystemResponse>('/system')
export const getGrapCurrent = () => get<GrapAssessment>('/grap/current')
export const getGrapStages = () => get<GrapStagesResponse>('/grap/stages')

export const getForecastExportUrl = (station: string, hours = 72) =>
  `/api/export/forecast.csv?station_name=${encodeURIComponent(station)}&hours=${hours}`
export const getWeatherExportUrl = (station: string, hours = 72) =>
  `/api/export/weather.csv?station_name=${encodeURIComponent(station)}&hours=${hours}`
export const getPollutionExportUrl = (station: string, hours = 72) =>
  `/api/export/pollution.csv?station_name=${encodeURIComponent(station)}&hours=${hours}`

const csvHeaders = { 'Content-Type': 'text/csv' }
export const importWeatherCsv = (csv: string) =>
  post<DataImportSummary>('/import/weather', csv, { headers: csvHeaders })
export const importPollutionCsv = (csv: string) =>
  post<DataImportSummary>('/import/pollution', csv, { headers: csvHeaders })

export const postScenarioAnalysis = (payload: ScenarioAnalysisRequest) =>
  post<ScenarioAnalysisResponse>('/scenario/analysis', payload, { timeout: 120000 })

// Exported for the keep-warm pinger (see ../warmup). While a browser tab is
// open the Render instance is polled so the demo never hits a cold start
// mid-session.
export { api, ensureWarm }

/**
 * Fetch for panels that live outside the dashboard's fan-out.
 *
 * The dashboard retries its whole panel set on a widening backoff while
 * anything is failing, but a standalone panel (the "Why this forecast"
 * explanation, the coupled two-way feedback card) had exactly one shot: a
 * single failure during a cold start left it on its empty state forever, and
 * in the explanation panel's case the empty state actively claimed the
 * explainability pipeline had never been run — a confident, false statement
 * about data that exists and loads in under 3 s once the instance is awake.
 *
 * This gives those panels the same recovery behaviour: wait for the shared
 * warm-up gate, then retry a few times on a widening delay. It resolves as soon
 * as the request succeeds and only rejects once the budget is spent, so the
 * caller's error state means "genuinely unavailable" rather than "arrived during
 * a wake".
 */
export async function resilientGet<T>(
  run: () => Promise<AxiosResponse<T>>,
  attempts = 4,
): Promise<AxiosResponse<T>> {
  let lastError: unknown
  for (let attempt = 0; attempt < attempts; attempt++) {
    if (attempt > 0) {
      await ensureWarm()
      await sleep(Math.min(2000 * 2 ** (attempt - 1), 15000))
    }
    try {
      return await run()
    } catch (error) {
      lastError = error
      if (!isTransientStatus((error as { response?: { status?: number } })?.response?.status)) throw error
    }
  }
  throw lastError
}
