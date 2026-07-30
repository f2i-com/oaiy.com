/**
 * storageQuota — surface QuotaExceededError to the user.
 *
 * The web build autosaves the whole project (flows, edges, run history,
 * constants) to `localStorage`. When the user accumulates enough big
 * inputs (data-URL images, long execution logs, packaged macros) the
 * 5–10 MB quota fills up and `localStorage.setItem` throws
 * `QuotaExceededError`. Existing call sites already wrap that in
 * try/catch and log a dev-only warning, so production users get no
 * feedback at all — they keep editing for an hour, refresh, and lose
 * everything since the last successful save.
 *
 * This module turns that silent failure into a single sticky toast.
 * Call sites import `notifyStorageQuotaExceeded(area, error)` from
 * their catch handler. The helper:
 *
 *   1. Sniffs the error to confirm it's actually a quota event
 *      (browsers expose it under several names — see `isQuotaError`).
 *   2. Dispatches a `oaiy:storage-quota` CustomEvent on `window`.
 *   3. Throttles itself so re-entrant saves during the same session
 *      only fire one event.
 *
 * OAIYApp subscribes to that event and renders a sticky warning toast
 * with guidance ("Clear run history in Settings → defaults"). The
 * subscription is the only consumer; nothing else listens.
 */

/** Browsers report a full quota under different names. Cover the lot. */
function isQuotaError(error: unknown): boolean {
  if (!(error instanceof DOMException)) return false;
  // Standard: QuotaExceededError. Firefox: NS_ERROR_DOM_QUOTA_REACHED.
  // Older WebKit reports code 22. Some embedded browsers blanket-throw
  // "QuotaExceededError" as the message instead of the name; check both.
  return (
    error.name === 'QuotaExceededError'
    || error.name === 'NS_ERROR_DOM_QUOTA_REACHED'
    || error.code === 22
    || /quota.*exceeded/i.test(error.message ?? '')
  );
}

const STORAGE_QUOTA_EVENT = 'oaiy:storage-quota';

export interface StorageQuotaEventDetail {
  /** Which save path tripped — e.g. "project", "workflow-autosave". */
  area: string;
  /** The original DOMException, for debug surfaces. */
  error: DOMException;
}

/**
 * Notify listeners of a quota-exceeded write. No-op if the error
 * isn't actually a quota error — call sites can pass any error
 * unconditionally without sniffing first.
 *
 * Throttled: fires once, then stays armed-off until a write SUCCEEDS again
 * (markStorageWriteSucceeded re-arms it). Without the re-arm a user who
 * DISMISSES the sticky toast, frees space, then later re-fills the quota would
 * get NO further warning and silently lose autosaves — the exact failure this
 * guard exists to surface.
 */
let fired = false;
export function notifyStorageQuotaExceeded(area: string, error: unknown): void {
  if (fired) return;
  if (!isQuotaError(error)) return;
  fired = true;
  try {
    window.dispatchEvent(
      new CustomEvent<StorageQuotaEventDetail>(STORAGE_QUOTA_EVENT, {
        detail: { area, error: error as DOMException },
      }),
    );
  } catch {
    // CustomEvent unavailable (extremely old browser) — nothing useful
    // to do; the silent-fail behaviour is the existing baseline.
  }
}

/**
 * Subscribe to quota events. Returns an unsubscribe function suitable
 * for a React `useEffect` cleanup.
 */
export function subscribeStorageQuota(
  handler: (detail: StorageQuotaEventDetail) => void,
): () => void {
  const onEvent = (e: Event) => {
    const ce = e as CustomEvent<StorageQuotaEventDetail>;
    handler(ce.detail);
  };
  window.addEventListener(STORAGE_QUOTA_EVENT, onEvent);
  return () => window.removeEventListener(STORAGE_QUOTA_EVENT, onEvent);
}

/**
 * Re-arm the throttle after a write SUCCEEDS, so a fresh out-of-space episode
 * (e.g. after the user dismissed the toast + freed space) can warn again. Call
 * after a successful localStorage.setItem on the autosave paths. No-op when not
 * already fired.
 */
export function markStorageWriteSucceeded(): void {
  fired = false;
}

/**
 * Test-only: reset the once-per-session throttle. Production code
 * doesn't need this — exposed so unit tests can re-fire the event.
 */
export function _resetStorageQuotaForTests(): void {
  fired = false;
}
