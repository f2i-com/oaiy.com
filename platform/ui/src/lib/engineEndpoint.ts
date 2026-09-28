/**
 * Where the OAIY engine lives.
 *
 * Historically this was the constant `http://127.0.0.1:17972`, compiled in.
 * That is right for the common case — browser and engine on one machine — but
 * it makes one genuinely useful arrangement impossible: run the engine on a
 * desktop and drive the flow editor from a phone on the same network. No URL
 * the phone could be given would have been used, because there was nowhere to
 * put one.
 *
 * So the base URL is a setting, defaulting to loopback. Two rules:
 *
 *   1. A stored value that does not parse is IGNORED, loudly, and loopback is
 *      used instead. A typo must not leave the editor pointing at nothing with
 *      no way back — the default is always reachable on the machine that is
 *      running the engine.
 *   2. Only the ORIGIN is kept (scheme + host + port). Paths, query strings,
 *      credentials and fragments are stripped rather than honoured: every
 *      caller appends its own `/api/...`, so a stored path would silently
 *      produce `http://host/foo/api/health` and 404 forever.
 */

/** The engine on this machine. Also the fallback whenever a stored value is unusable. */
export const DEFAULT_ENGINE_BASE = 'http://127.0.0.1:17972';

const KEY = 'oaiy.engineBase';

type Listener = (base: string) => void;
const listeners = new Set<Listener>();

/**
 * Normalise anything the user typed into a bare origin, or null if it cannot
 * be one.
 *
 * Deliberately permissive about what it ACCEPTS (a bare `192.168.1.50:17972`
 * is what someone reads off a screen and types) and strict about what it
 * RETURNS (always scheme + host + optional port, never a trailing slash).
 */
export function normalizeEngineBase(raw: string): string | null {
  const text = String(raw ?? '').trim();
  if (text === '') return null;
  // A bare host[:port] is the realistic input; give it a scheme so URL can parse.
  const withScheme = /^https?:\/\//i.test(text) ? text : `http://${text}`;
  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  if (!url.hostname) return null;
  // Credentials in a stored endpoint would be written to localStorage in the
  // clear and sent on every poll. Refuse rather than quietly dropping them.
  if (url.username || url.password) return null;
  return `${url.protocol}//${url.host}`;
}

/** The configured engine origin, or the loopback default. Never throws. */
export function getEngineBase(): string {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(KEY);
  } catch {
    // Private mode / storage disabled: the default is still correct.
    return DEFAULT_ENGINE_BASE;
  }
  if (!stored) return DEFAULT_ENGINE_BASE;
  const normalized = normalizeEngineBase(stored);
  if (!normalized) {
    // eslint-disable-next-line no-console
    console.warn(`[engine] ignoring unusable stored endpoint ${JSON.stringify(stored)}; using ${DEFAULT_ENGINE_BASE}`);
    return DEFAULT_ENGINE_BASE;
  }
  return normalized;
}

/** True when the engine is somewhere other than this machine's loopback. */
export function isRemoteEngine(base = getEngineBase()): boolean {
  try {
    const h = new URL(base).hostname.toLowerCase();
    const bare = h.startsWith('[') && h.endsWith(']') ? h.slice(1, -1) : h;
    return !(h === 'localhost' || bare === '::1' || /^127(?:\.\d{1,3}){3}$/.test(h));
  } catch {
    return false;
  }
}

/**
 * Point the editor at a different engine. Pass null/'' to return to loopback.
 *
 * Returns the origin actually stored so a caller can show it back; throws on
 * input that cannot be an origin, because silently keeping the old endpoint
 * after the user pressed Save is the one outcome that reads as a bug.
 */
export function setEngineBase(raw: string | null): string {
  const next = raw === null || String(raw).trim() === '' ? DEFAULT_ENGINE_BASE : normalizeEngineBase(raw);
  if (!next) {
    throw new Error(`"${raw}" is not a usable address — expected something like 192.168.1.50:17972`);
  }
  try {
    if (next === DEFAULT_ENGINE_BASE) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, next);
  } catch {
    throw new Error('this browser will not persist settings, so the address could not be saved');
  }
  for (const fn of listeners) {
    try {
      fn(next);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[engine] endpoint listener threw:', e);
    }
  }
  return next;
}

/** Re-run when the endpoint changes (the detection probe uses this to re-probe). */
export function subscribeEngineBase(fn: Listener): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}
