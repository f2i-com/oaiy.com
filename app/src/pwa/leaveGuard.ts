/**
 * The page's own guard against being left by accident (main.ts: a `beforeunload`
 * handler that objects while the agent works or changes are still being saved).
 *
 * The browser shows its leave prompt only for a real navigation, and gives no
 * way to ask what the person answered before the page is gone. Updating has to
 * know before it does anything that cannot be undone (the new worker takes
 * over, and the running version's files start to go), so it asks the guard
 * first: the event goes to the page's handlers without any navigation, and if
 * one objects the person is asked in the app's own dialog. A person who agrees
 * has agreed once: the guard keeps quiet for a moment, so the browser does not
 * ask the same question again as the page reloads.
 */

/** Whether the page's leave guard (its `beforeunload` handlers) objects to leaving now. Nothing is left. */
export function leaveGuardBlocks(target: EventTarget): boolean {
  const event = new Event('beforeunload', { cancelable: true });
  target.dispatchEvent(event);
  return event.defaultPrevented;
}

/** How long an agreement to leave lasts: the reload that follows it, and no more. */
export const AGREEMENT_MS = 15_000;

let agreedAt = Number.NEGATIVE_INFINITY;

/** The person agreed to leave (by choosing Reload): the guard need not object for the next few seconds. */
export function agreeToLeave(now: number = Date.now()): void {
  agreedAt = now;
}

/** Whether the person agreed to leave a moment ago. */
export function agreedToLeave(now: number = Date.now()): boolean {
  return now - agreedAt < AGREEMENT_MS;
}
