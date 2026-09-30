/**
 * OAIY Desktop detection.
 *
 * OAIY Desktop is a tray-resident Tauri app (see oaiy-web/desktop/) that
 * exposes a localhost HTTP API at http://127.0.0.1:17972 (or wherever the
 * engine address in Settings points: lib/engineEndpoint.ts). When it's
 * running on the user's machine, oaiy-web can:
 *
 *   - List the user's locally-managed services in the palette (Phase 3)
 *   - Spawn / monitor / stop those services from a UI button (Phase 2/3)
 *   - Route browser_action / browser_extract / browser_session /
 *     browser_page nodes through OAIY Desktop's Playwright sidecar
 *     (Phase 4)
 *
 * This module is the detection probe — a small reactive helper any
 * component can subscribe to. It does NOT block app load; the probe
 * runs in the background and updates listeners when OAIY Desktop's
 * status changes.
 *
 * Discovery contract:
 *   GET <engine address>/api/health   (http://127.0.0.1:17972 by default)
 *     → 200 { status: 'ok', product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: 'x.y.z' }
 *     → anything else / no response → assume not running
 *
 * The engine address is a setting the person can change, so nothing here (or
 * anywhere that reaches the desktop) may keep a copy of it: ask
 * `getEngineBase()` each time, or read `DesktopInfo.baseUrl` from here.
 *
 * WHEN it looks. OAIY's own window (and the desktop shell) look as soon as the
 * page opens, as they always have. A tab in a browser does not: on a public
 * address that request is a permission prompt for the visitor and shows the
 * site what is on their network. A tab looks when its person presses Connect
 * (lib/desktopConnect.ts), and from then on whenever the editor opens (the
 * link kept in lib/desktopLink.ts, or an engine address given in Settings).
 * `refreshDesktopStatus()` is the person asking, and always asks.
 */

import { looksOnLoad, readHost } from '@oaiy/shared/capabilities/host';
import { getEngineBase, subscribeEngineBase } from './engineEndpoint';
import { hasSavedDesktopLink } from './desktopLink';

const POLL_INTERVAL_MS = 10_000;
const FETCH_TIMEOUT_MS = 1500;

export interface DesktopInfo {
  /**
   * False until the desktop has been asked. A tab in a browser asks only when its person presses Connect, so until then
   * the desktop is neither there nor not there, and the page says so.
   */
  checked: boolean;
  /** True if the last health probe succeeded. */
  available: boolean;
  /** Version string from OAIY Desktop, only set when `available`. */
  version?: string;
  /** Base URL — useful for downstream code that wants to call other
   *  companion endpoints (`/api/services`, `/api/browser/...`). */
  baseUrl: string;
  /** Last time the status changed, in ms-since-epoch. */
  lastChange: number;
}

type Listener = (info: DesktopInfo) => void;

let current: DesktopInfo = {
  checked: false,
  available: false,
  baseUrl: getEngineBase(),
  lastChange: Date.now(),
};

const listeners = new Set<Listener>();
let pollTimer: number | null = null;
let pollPromise: Promise<void> | null = null;

async function probeOnce(): Promise<void> {
  // The address this probe asks about: the setting as it is now. If the person
  // changes it while the answer is on its way, that answer is about a desktop
  // they no longer point at (a probe of the new address is already running, see
  // subscribeEngineBase below) and must not be published over it.
  const base = getEngineBase();
  try {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    const resp = await fetch(`${base}/api/health`, {
      method: 'GET',
      signal: controller.signal,
      // OAIY Desktop's API is on a different origin (localhost:17972 vs
      // oaiy-web's hosting origin). OAIY Desktop sets CORS for any origin,
      // so credentials: 'omit' is fine and minimal.
      credentials: 'omit',
      cache: 'no-store',
    });
    window.clearTimeout(timeout);
    if (resp.ok) {
      const body = (await resp.json().catch(() => null)) as {
        product?: string;
        protocol?: string;
        version?: string;
      } | null;
      // A 200 from a fixed loopback port is not proof it is us — assert the
      // identity. Reporting a squatter as "available" would silently route
      // desktop-backed nodes at a stranger.
      const isOaiyCompanion = body?.product === 'oaiy-desktop';
      if (getEngineBase() !== base) return;
      const next: DesktopInfo = {
        checked: true,
        available: isOaiyCompanion,
        version: isOaiyCompanion ? body?.version : undefined,
        baseUrl: base,
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
  if (getEngineBase() !== base) return;
  publish({
    checked: true,
    available: false,
    baseUrl: base,
    lastChange: current.available ? Date.now() : current.lastChange,
  });
}

function publish(next: DesktopInfo): void {
  if (
    next.checked === current.checked &&
    next.available === current.available &&
    next.version === current.version &&
    next.baseUrl === current.baseUrl
  ) {
    // No state change — keep the original lastChange.
    current = { ...current };
    return;
  }
  // A new address is a change even when the desktop there answers just as the
  // last one did: whoever shows the address (the dock, the settings card) has
  // to be told, or it keeps showing the old one.
  current = next;
  for (const listener of listeners) {
    try {
      listener(current);
    } catch (e) {
      // A listener throwing shouldn't break the probe.
      // eslint-disable-next-line no-console
      console.warn('[desktop-detect] listener threw:', e);
    }
  }
}

/**
 * Whether this page may look for the desktop as it opens: OAIY's own window and the desktop shell always may; a tab in a
 * browser only where its person has said (a link kept from Connect, or an engine address given in Settings).
 */
export function mayLookOnLoad(): boolean {
  return looksOnLoad(readHost()) || hasSavedDesktopLink();
}

/**
 * Start the periodic probe, where this page may look (see `mayLookOnLoad`; a tab with no link does nothing here).
 * Safe to call multiple times — only one underlying timer runs at any moment.
 * The first probe fires immediately so the initial UI doesn't wait for the
 * first interval tick; `probeNow: false` starts the timer alone, for a caller
 * that has just asked (Connect).
 */
export function startDesktopDetection(options: { probeNow?: boolean } = {}): void {
  if (pollTimer !== null) return;
  if (!mayLookOnLoad()) return;
  // Fire one probe right away, then on the interval.
  if (options.probeNow !== false) pollPromise = probeOnce();
  pollTimer = window.setInterval(() => {
    pollPromise = probeOnce();
  }, POLL_INTERVAL_MS);
}

/** Stop the periodic probe. */
export function stopDesktopDetection(): void {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

/** Disconnect: the probe stops and the desktop is neither there nor not there again, until someone asks. */
export function resetDesktopDetection(): void {
  stopDesktopDetection();
  publish({ checked: false, available: false, baseUrl: getEngineBase(), lastChange: current.available ? Date.now() : current.lastChange });
}

/** Snapshot of the current companion status. */
export function getDesktopInfo(): DesktopInfo {
  return current;
}

/**
 * Subscribe to companion-status changes. Returns an unsubscribe
 * function. Listener fires only when `available` or `version` change —
 * not on every poll. The current status is passed to the listener
 * immediately so callers don't need to also call getDesktopInfo().
 */
export function subscribeDesktopStatus(listener: Listener): () => void {
  listeners.add(listener);
  try {
    listener(current);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[desktop-detect] initial listener call threw:', e);
  }
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Force an immediate probe (independent of the poll interval). Useful
 * after the user has manually started/stopped OAIY Desktop and wants
 * the UI to refresh without waiting for the next tick.
 */
export async function refreshDesktopStatus(): Promise<DesktopInfo> {
  await probeOnce();
  return current;
}

// Re-probe immediately when the endpoint changes, rather than making the user
// wait out the 10s poll after typing a new address. Not when the change leaves a
// tab with no link (Reset to the default address, with nothing kept from Connect):
// that tab looks at nothing (see lib/desktopConnect.ts).
subscribeEngineBase(() => {
  if (mayLookOnLoad()) void probeOnce();
});

/** Internal: lets tests/dev tools await an in-flight probe. */
export function _currentProbePromise(): Promise<void> | null {
  return pollPromise;
}
