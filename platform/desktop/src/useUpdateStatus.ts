import { useCallback, useState } from 'react';
import { updates, type UpdateStatus } from './api';
import { peek, put } from './useCached';
import { useVisiblePoll } from './useVisiblePoll';

const KEY = 'updateStatus';

/** How often the status is read: quickly while something is moving (a check, a download, an install), slowly otherwise. */
export const POLL_FAST_MS = 1000;
export const POLL_SLOW_MS = 5000;

export function pollInterval(status: UpdateStatus | null | undefined): number {
  return status && (status.state === 'checking' || status.state === 'downloading' || status.state === 'installing') ? POLL_FAST_MS : POLL_SLOW_MS;
}

export interface UpdateStatusResult {
  /** The last status read (from the cache on a return to the page); null before the first. */
  status: UpdateStatus | null;
  /** The last read failed (the local API is not answering). */
  error: string | null;
  refresh: () => Promise<void>;
  /** Take a status a command answered, without waiting for the next read. */
  apply: (status: UpdateStatus) => void;
}

/**
 * What OAIY knows about a newer release, read while the window is visible. Shared by Settings and the
 * Overview banner: the last value is kept outside React, so returning to a page paints at once.
 */
export function useUpdateStatus(): UpdateStatusResult {
  const [status, setStatus] = useState<UpdateStatus | null>(() => peek<UpdateStatus>(KEY) ?? null);
  const [error, setError] = useState<string | null>(null);

  const apply = useCallback((next: UpdateStatus) => {
    put(KEY, next);
    setStatus(next);
    setError(null);
  }, []);

  const refresh = useCallback(async () => {
    try {
      apply(await updates.status());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [apply]);

  useVisiblePoll(() => void refresh(), pollInterval(status));

  return { status, error, refresh, apply };
}
