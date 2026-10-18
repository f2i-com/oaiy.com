/**
 * Which failed runs the person has seen: they opened Run history from Overview's warning, dismissed the warning, or
 * looked at the failed runs in Run history. Overview warns only of the runs that failed since (a run that failed
 * last week is not news every time it opens). The runs are failed and timed-out ones, the ones both count.
 *
 * Kept as when the newest of them ended (`at`, the desktop's `lastFailedAt`) and how many there were (`count`). The
 * time is what says whether one failed since: by the count alone, a failure after some were cleared, or after the
 * oldest were let go, left the count where it was and went unsaid. The count says how many more, while it can.
 */
export const FAILED_SEEN = 'oaiy.overview.failedSeen';

export interface FailedSeen {
  count: number;
  /** When the newest failure seen ended; null when only a count was kept (before the time was). */
  at: string | null;
}

const when = (at: unknown): string | null => (typeof at === 'string' && Number.isFinite(Date.parse(at)) ? at : null);
const atLeastZero = (n: unknown): number => (typeof n === 'number' && Number.isFinite(n) && n > 0 ? n : 0);

export function readFailedSeen(): FailedSeen {
  try {
    const raw = localStorage.getItem(FAILED_SEEN) ?? '';
    if (raw.trim().startsWith('{')) {
      const saved = JSON.parse(raw) as { count?: unknown; at?: unknown };
      return { count: atLeastZero(saved.count), at: when(saved.at) };
    }
    // (a count alone, as it was kept before)
    return { count: atLeastZero(Number(raw || 0)), at: null };
  } catch {
    return { count: 0, at: null };
  }
}

/** Seen: `count` failed runs, the newest of them ended `at` (never taken back to an earlier time than one seen). */
export function writeFailedSeen(count: number, at?: string | null): void {
  const before = readFailedSeen().at;
  const newest = when(at);
  const kept = newest && (!before || Date.parse(newest) > Date.parse(before)) ? newest : before;
  try {
    localStorage.setItem(FAILED_SEEN, JSON.stringify({ count: Math.max(0, count), at: kept }));
  } catch {
    /* storage can be unavailable: the warning comes back next time */
  }
}

/** When the newest of `runs` that failed or timed out ended (asked, for one that never started). */
export function lastFailedOf(runs: ReadonlyArray<{ status: string; finishedAt?: string; reservedAt: string }>): string | null {
  let newest: string | null = null;
  for (const r of runs) {
    if (r.status !== 'failed' && r.status !== 'timed_out') continue;
    const at = when(r.finishedAt) ?? when(r.reservedAt);
    if (at && (!newest || Date.parse(at) > Date.parse(newest))) newest = at;
  }
  return newest;
}

/**
 * The runs that failed since those seen: whether any did, and how many when that can be told (null: some did, but
 * the count cannot say how many, as when failed runs were cleared meanwhile). By the time when both the desktop
 * (`lastFailedAt`) and what was seen have one; by the count otherwise.
 */
export function failedSince(seen: FailedSeen, failed: number, lastFailedAt: string | null | undefined): { any: boolean; count: number | null } {
  if (failed <= 0) return { any: false, count: 0 };
  const last = when(lastFailedAt);
  if (last && seen.at) {
    if (Date.parse(last) <= Date.parse(seen.at)) return { any: false, count: 0 };
    return { any: true, count: failed > seen.count ? failed - seen.count : null };
  }
  const more = failed - Math.min(seen.count, failed);
  return { any: more > 0, count: more };
}
