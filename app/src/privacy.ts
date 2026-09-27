/**
 * Incognito: while an incognito project is open, requests to nrob carry
 * `X-NROB-Incognito: 1`, so nrob keeps nothing of them either (no media
 * saved, no job list, no prompt cache, no logs). Only nrob is told: another
 * service may refuse a request with a header it does not expect.
 */
let incognito = false;
/** The incognito session (the incognito project): nrob keeps its prompt state in memory for its next request, and nowhere else. */
let session: string | null = null;
const nrobOrigins = new Set<string>();

export function setIncognito(on: boolean, id: string | null = null): void {
  incognito = on;
  session = on ? id : null;
}

export function isIncognito(): boolean {
  return incognito;
}

/** An origin nrob answers at (its media service, or a chat provider on it). */
export function addNrobOrigin(url: string | undefined): void {
  try {
    if (url) nrobOrigins.add(new URL(url).origin);
  } catch {
    /* not a URL */
  }
}

/**
 * The headers for a request to `url`: nrob's incognito one (and the session,
 * so its next request reuses what it has read instead of reading it all
 * again), while incognito and when `url` is nrob's.
 */
export function incognitoHeaderFor(url: string): Record<string, string> {
  if (!incognito) return {};
  try {
    if (!nrobOrigins.has(new URL(url).origin)) return {};
    return session ? { 'X-NROB-Incognito': '1', 'X-NROB-Session': session } : { 'X-NROB-Incognito': '1' };
  } catch {
    return {};
  }
}
