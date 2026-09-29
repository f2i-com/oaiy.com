/**
 * The guard on the one automatic reload a first visit makes, when a static host
 * gives no isolation headers and the service worker has to supply them: at most
 * one reload a minute, so a browser that refuses isolation cannot loop.
 *
 * localStorage, not sessionStorage: switching into a cross-origin isolated
 * context can start a fresh session store.
 */
export const RELOAD_KEY = 'oaiy-agent.isolation-reload';
/** The key before the app was called OAIY. A reload it recorded a moment ago still counts. */
export const OLD_RELOAD_KEY = 'bot.computer.isolation-reload';
const WINDOW_MS = 60_000;

type Store = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

/**
 * True when the page may reload now (and records it); false when it reloaded
 * within the last minute or storage cannot say. The old key is read once and
 * moved to the new one, so the change of name cannot start a reload loop nor
 * leave a stale key behind.
 */
export function mayReloadForIsolation(store: Store, now: number): boolean {
  try {
    const old = store.getItem(OLD_RELOAD_KEY);
    const last = Math.max(Number(store.getItem(RELOAD_KEY)) || 0, Number(old) || 0);
    if (old !== null) {
      store.removeItem(OLD_RELOAD_KEY);
      if (last) store.setItem(RELOAD_KEY, String(last));
    }
    if (now - last < WINDOW_MS) return false;
    store.setItem(RELOAD_KEY, String(now));
    return true;
  } catch {
    return false;
  }
}
