/**
 * The desktop page's library of ready-made services.
 *
 * The list is fetched from the PHP API (`/api/service-library`), which serves a
 * folder of service templates. Only a host that runs that API has one: the
 * standalone release build is static files and has no `/api`, so the answer to
 * the request there is a 404, or the host's front page. That is not a failure
 * to try again, and the page says so instead of offering a retry that cannot
 * work.
 */

/** One template in the library, as the API lists it. */
export interface LibraryItem {
  file: string;
  name: string;
  description: string;
  icon?: string;
  category?: string;
  count?: number;
  size?: number;
  downloadUrl: string;
}

/** What asking for the library came to. */
export type LibraryResult =
  | { state: 'ok'; items: LibraryItem[] }
  /** This copy of the site has no library behind it: asking again will not help. */
  | { state: 'unavailable' }
  /** The library should be there and did not answer properly: asking again may help. */
  | { state: 'error' };

/** Statuses that mean "there is no such thing here", not "it broke". */
const ABSENT = new Set([404, 405, 410, 501]);

function isLibraryItem(item: unknown): item is LibraryItem {
  if (!item || typeof item !== 'object') return false;
  const entry = item as Record<string, unknown>;
  return (
    ['file', 'name', 'description', 'downloadUrl'].every((key) => typeof entry[key] === 'string') &&
    (entry.category === undefined || typeof entry.category === 'string') &&
    // A download is one of the API's own paths, never another site or a path with a backslash.
    /^\/api\//.test(entry.downloadUrl as string) &&
    !(entry.downloadUrl as string).includes('\\')
  );
}

/**
 * Ask the API at `apiBase` (empty for the site's own origin) for the library.
 * It never rejects: whatever goes wrong is a state to show.
 *
 *   - a 404 (or 405, 410, 501), or a 200 that is a page and not JSON (a host
 *     that answers every path with its front page): `unavailable`;
 *   - no answer at all, a timeout, any other status, JSON that is not a library:
 *     `error`.
 */
export async function loadServiceLibrary(
  apiBase: string,
  signal?: AbortSignal,
  fetchFn: typeof fetch = fetch,
): Promise<LibraryResult> {
  try {
    const res = await fetchFn(`${apiBase}/api/service-library`, { headers: { Accept: 'application/json' }, signal });
    if (ABSENT.has(res.status)) return { state: 'unavailable' };
    if (!res.ok) return { state: 'error' };
    const type = res.headers.get('content-type');
    if (type && !/json/i.test(type)) return { state: 'unavailable' };
    const data = (await res.json()) as { services?: unknown } | null;
    if (!Array.isArray(data?.services) || !data.services.every(isLibraryItem)) return { state: 'error' };
    return { state: 'ok', items: data.services };
  } catch {
    return { state: 'error' };
  }
}
