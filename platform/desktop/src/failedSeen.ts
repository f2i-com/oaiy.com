/**
 * How many failed runs the person has seen: they opened Run history from Overview's warning, dismissed the warning,
 * or looked at the failed runs in Run history. Overview warns only of the runs that failed since (a run that failed
 * last week is not news every time it opens). The count is of failed and timed-out runs, the ones both count.
 */
export const FAILED_SEEN = 'oaiy.overview.failedSeen';

export function readFailedSeen(): number {
  try {
    const n = Number(localStorage.getItem(FAILED_SEEN) ?? 0);
    return Number.isFinite(n) && n > 0 ? n : 0;
  } catch {
    return 0;
  }
}

export function writeFailedSeen(n: number): void {
  try {
    localStorage.setItem(FAILED_SEEN, String(Math.max(0, n)));
  } catch {
    /* storage can be unavailable: the warning comes back next time */
  }
}
