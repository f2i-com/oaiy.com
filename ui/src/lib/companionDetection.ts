/**
 * OAIY Companion detection.
 *
 * The companion is a tray-resident Tauri app (see oaiy-web/desktop/) that
 * exposes a localhost HTTP API at http://127.0.0.1:17972. When it's
 * running on the user's machine, oaiy-web can:
 *
 *   - List the user's locally-managed services in the palette (Phase 3)
 *   - Spawn / monitor / stop those services from a UI button (Phase 2/3)
 *   - Route browser_action / browser_extract / browser_session /
 *     browser_page nodes through the companion's Playwright sidecar
 *     (Phase 4)
 *
 * This module is the detection probe — a small reactive helper any
 * component can subscribe to. It does NOT block app load; the probe
 * runs in the background and updates listeners when the companion's
 * status changes.
 *
 * Discovery contract:
 *   GET http://127.0.0.1:17972/api/health
 *     → 200 { status: 'ok', companion: 'oaiy-companion', version: 'x.y.z' }
 *     → anything else / no response → assume not running
 */

const COMPANION_BASE = 'http://127.0.0.1:17972';
const POLL_INTERVAL_MS = 10_000;
const FETCH_TIMEOUT_MS = 1500;

export interface CompanionInfo {
  /** True if the last health probe succeeded. */
  available: boolean;
  /** Version string from the companion, only set when `available`. */
  version?: string;
  /** Base URL — useful for downstream code that wants to call other
   *  companion endpoints (`/api/services`, `/api/browser/...`). */
  baseUrl: string;
  /** Last time the status changed, in ms-since-epoch. */
  lastChange: number;
}

type Listener = (info: CompanionInfo) => void;

let current: CompanionInfo = {
  available: false,
  baseUrl: COMPANION_BASE,
  lastChange: Date.now(),
};

const listeners = new Set<Listener>();
let pollTimer: number | null = null;
let pollPromise: Promise<void> | null = null;

async function probeOnce(): Promise<void> {
  try {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    const resp = await fetch(`${COMPANION_BASE}/api/health`, {
      method: 'GET',
      signal: controller.signal,
      // The companion's API is on a different origin (localhost:17972 vs
      // oaiy-web's hosting origin). The companion sets CORS for any origin,
      // so credentials: 'omit' is fine and minimal.
      credentials: 'omit',
      cache: 'no-store',
    });
    window.clearTimeout(timeout);
    if (resp.ok) {
      const body = (await resp.json().catch(() => null)) as {
        companion?: string;
        version?: string;
      } | null;
      const isOaiyCompanion = body?.companion === 'oaiy-companion';
      const next: CompanionInfo = {
        available: isOaiyCompanion,
        version: isOaiyCompanion ? body?.version : undefined,
        baseUrl: COMPANION_BASE,
        lastChange:
          current.available !== isOaiyCompanion || current.version !== body?.version
            ? Date.now()
            : current.lastChange,
      };
      publish(next);
      return;
    }
  } catch {
    // Network error, timeout, or CORS rejection — all mean "not
    // available". Don't log to console; this probe runs on a 10s loop
    // and would flood the console otherwise.
  }
  publish({
    available: false,
    baseUrl: COMPANION_BASE,
    lastChange: current.available ? Date.now() : current.lastChange,
  });
}

function publish(next: CompanionInfo): void {
  if (
    next.available === current.available &&
    next.version === current.version
  ) {
    // No state change — keep the original lastChange.
    current = { ...current };
    return;
  }
  current = next;
  for (const listener of listeners) {
    try {
      listener(current);
    } catch (e) {
      // A listener throwing shouldn't break the probe.
      // eslint-disable-next-line no-console
      console.warn('[companion-detect] listener threw:', e);
    }
  }
}

/**
 * Start the periodic probe. Safe to call multiple times — only one
 * underlying timer runs at any moment. The first probe fires immediately
 * so the initial UI doesn't wait for the first interval tick.
 */
export function startCompanionDetection(): void {
  if (pollTimer !== null) return;
  // Fire one probe right away, then on the interval.
  pollPromise = probeOnce();
  pollTimer = window.setInterval(() => {
    pollPromise = probeOnce();
  }, POLL_INTERVAL_MS);
}

/** Stop the periodic probe. */
export function stopCompanionDetection(): void {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

/** Snapshot of the current companion status. */
export function getCompanionInfo(): CompanionInfo {
  return current;
}

/**
 * Subscribe to companion-status changes. Returns an unsubscribe
 * function. Listener fires only when `available` or `version` change —
 * not on every poll. The current status is passed to the listener
 * immediately so callers don't need to also call getCompanionInfo().
 */
export function subscribeCompanionStatus(listener: Listener): () => void {
  listeners.add(listener);
  try {
    listener(current);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[companion-detect] initial listener call threw:', e);
  }
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Force an immediate probe (independent of the poll interval). Useful
 * after the user has manually started/stopped the companion and wants
 * the UI to refresh without waiting for the next tick.
 */
export async function refreshCompanionStatus(): Promise<CompanionInfo> {
  await probeOnce();
  return current;
}

// Re-export the base URL so other modules building companion API calls
// can use a single source of truth. Phase 2/3/4 modules will add their
// own helpers (e.g. `fetchCompanionServices`, `companionBrowserGoto`)
// that build on this.
export const COMPANION_API_BASE = COMPANION_BASE;

/** Internal: lets tests/dev tools await an in-flight probe. */
export function _currentProbePromise(): Promise<void> | null {
  return pollPromise;
}
