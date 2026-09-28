import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Stale-while-revalidate for the panels.
 *
 * Switching views remounts the whole content subtree (App keys it on `view`, so
 * each panel starts at the top rather than mid-scroll). That is the behaviour we
 * want, but it means a revisited panel used to throw away everything it knew and
 * sit on "Loading…" until a fresh round trip landed — every single time.
 *
 * So the last good value for a key lives here, OUTSIDE React, and survives the
 * remount: a returning panel paints instantly from cache and refreshes in the
 * background. The cache is deliberately tiny and process-local — this is a
 * desktop app talking to its own loopback API, not a CDN.
 */

interface Entry {
  value: unknown;
  at: number;
}

const store = new Map<string, Entry>();

/** Read a cached value without subscribing (for a first paint). */
export function peek<T>(key: string): T | undefined {
  return store.get(key)?.value as T | undefined;
}

/** Seed the cache — used by a panel that fetched through another path. */
export function put(key: string, value: unknown): void {
  store.set(key, { value, at: Date.now() });
}

/** Drop one key (or everything), so the next read is authoritative. */
export function invalidate(key?: string): void {
  if (key === undefined) store.clear();
  else store.delete(key);
}

export interface CachedResult<T> {
  /** Last known value: present immediately on a revisit. */
  data: T | undefined;
  /** True only when there is nothing to show yet — i.e. the real first load. */
  loading: boolean;
  error: string | null;
  /** Force a refresh (after a mutation). */
  refresh: () => Promise<void>;
}

/**
 * Keep `key` fresh by calling `fetcher`, returning the cached value immediately.
 *
 * `intervalMs` polls while mounted. A failed refresh keeps the last good value
 * and surfaces `error` — a transient blip should not blank a panel that was
 * showing something useful a second ago.
 */
export function useCached<T>(
  key: string,
  fetcher: () => Promise<T>,
  intervalMs?: number,
): CachedResult<T> {
  const [data, setData] = useState<T | undefined>(() => peek<T>(key));
  const [error, setError] = useState<string | null>(null);
  // Keep the latest fetcher without making it a dependency: panels declare it
  // inline, so a new identity every render would restart the interval endlessly.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      const next = await fetcherRef.current();
      put(key, next);
      setData(next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [key]);

  useEffect(() => {
    let cancelled = false;
    // Adopt whatever another panel may have cached for this key since mount.
    const cached = peek<T>(key);
    if (cached !== undefined) setData(cached);

    const tick = async () => {
      if (cancelled) return;
      await refresh();
    };
    void tick();
    if (!intervalMs) return () => { cancelled = true; };
    const id = window.setInterval(() => void tick(), intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [key, intervalMs, refresh]);

  return { data, loading: data === undefined && error === null, error, refresh };
}
