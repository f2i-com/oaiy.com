/**
 * Incognito: while an incognito project is open, requests to OAIY carry
 * `X-OAIY-Incognito: 1`, so OAIY keeps nothing of them either (no media
 * saved, no job list, no prompt cache, no logs). Only OAIY is told: another
 * service may refuse a request with a header it does not expect.
 */
let incognito = false;
/** The incognito session (the incognito project): OAIY keeps its prompt state in memory for its next request, and nowhere else. */
let session: string | null = null;
const oaiyOrigins = new Set<string>();

export function setIncognito(on: boolean, id: string | null = null): void {
  incognito = on;
  session = on ? id : null;
}

export function isIncognito(): boolean {
  return incognito;
}

/** An origin OAIY answers at (its media service, or a chat provider on it). */
export function addOaiyOrigin(url: string | undefined): void {
  try {
    if (url) oaiyOrigins.add(new URL(url).origin);
  } catch {
    /* not a URL */
  }
}

/**
 * The headers for a request to `url`: OAIY's incognito one (and the session,
 * so its next request reuses what it has read instead of reading it all
 * again), while incognito and when `url` is OAIY's.
 */
export function incognitoHeaderFor(url: string): Record<string, string> {
  if (!incognito) return {};
  try {
    if (!oaiyOrigins.has(new URL(url).origin)) return {};
    return session ? { 'X-OAIY-Incognito': '1', 'X-OAIY-Session': session } : { 'X-OAIY-Incognito': '1' };
  } catch {
    return {};
  }
}
