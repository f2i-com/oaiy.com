/**
 * Incognito: while an incognito project is open, requests to nrob carry
 * `X-NROB-Incognito: 1`, so nrob keeps nothing of them either (no media
 * saved, no job list, no prompt cache, no logs). Only nrob is told: another
 * service may refuse a request with a header it does not expect.
 */
let incognito = false;
const nrobOrigins = new Set<string>();

export function setIncognito(on: boolean): void {
  incognito = on;
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

/** The header for a request to `url`: nrob's incognito one, while incognito and when `url` is nrob's. */
export function incognitoHeaderFor(url: string): Record<string, string> {
  if (!incognito) return {};
  try {
    return nrobOrigins.has(new URL(url).origin) ? { 'X-NROB-Incognito': '1' } : {};
  } catch {
    return {};
  }
}
