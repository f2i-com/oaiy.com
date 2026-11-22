/**
 * The guard on the one automatic reload a first visit makes, when a static host
 * gives no isolation headers and the service worker has to supply them: at most
 * one reload a minute, so a browser that refuses isolation cannot loop.
 *
 * localStorage, not sessionStorage: switching into a cross-origin isolated
 * context can start a fresh session store.
 */
export const RELOAD_KEY = 'oaiy-agent.isolation-reload';
const WINDOW_MS = 60_000;

type Store = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

/**
 * True when the page may reload now (and records it); false when it reloaded
 * within the last minute or storage cannot say.
 */
export function mayReloadForIsolation(store: Store, now: number): boolean {
  try {
    const last = Number(store.getItem(RELOAD_KEY)) || 0;
    if (now - last < WINDOW_MS) return false;
    store.setItem(RELOAD_KEY, String(now));
    return true;
  } catch {
    return false;
  }
}
