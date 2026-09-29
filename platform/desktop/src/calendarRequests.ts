import { useEffect, useSyncExternalStore } from 'react';
import { calendar, type Appointment } from './api';
import { ymd } from './calendarModel';

/**
 * The requests waiting for someone to confirm them (asked for on a call, by
 * text, or by the agent), from today on: one list for the sidebar's count
 * and the Calendar page's strip, so a request confirmed on the page leaves
 * the count at once. Asked every ten seconds while the window is visible and
 * something shows it, and only while the calendar is on. `null` until known.
 */

const POLL_MS = 10_000;

let requests: Appointment[] | null = null;
let inflight: Promise<void> | null = null;
let timer: number | undefined;
let enabled = false;
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

async function fetchNow(): Promise<void> {
  if (!enabled) return;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const got = await calendar.get(ymd(new Date()));
      const next = got.appointments.filter((a) => a.status === 'requested').sort((a, b) => a.start.localeCompare(b.start));
      if (JSON.stringify(next) !== JSON.stringify(requests)) {
        requests = next;
        emit();
      }
    } catch {
      /* the desktop is away: what was known stays */
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

function schedule(): void {
  if (timer !== undefined) window.clearInterval(timer);
  timer = undefined;
  if (enabled && listeners.size && !document.hidden) timer = window.setInterval(() => void fetchNow(), POLL_MS);
}

function onVisibility(): void {
  if (!document.hidden) void fetchNow();
  schedule();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (listeners.size === 1) {
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('focus', onVisibility);
    void fetchNow();
    schedule();
  }
  return () => {
    listeners.delete(listener);
    if (!listeners.size) {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('focus', onVisibility);
      schedule();
    }
  };
}

/** Turn the asking on or off (the calendar module is on, or not). Off forgets what was known. */
export function setRequestsEnabled(on: boolean): void {
  if (on === enabled) return;
  enabled = on;
  if (!on && requests !== null) {
    requests = null;
    emit();
  }
  if (on) void fetchNow();
  schedule();
}

/** The waiting requests, soonest first: `null` until known (or while the calendar is off). */
export function useWaitingRequests(on: boolean): Appointment[] | null {
  useEffect(() => setRequestsEnabled(on), [on]);
  return useSyncExternalStore(subscribe, () => requests);
}

/** Ask again now (a request was confirmed, declined or changed here). */
export async function refetchRequests(): Promise<void> {
  await inflight;
  await fetchNow();
}

/** Tests: forget what is known. */
export function resetRequests(): void {
  requests = null;
  inflight = null;
  enabled = false;
  if (timer !== undefined) window.clearInterval(timer);
  timer = undefined;
}
