import { useSyncExternalStore } from 'react';
import { setup as setupApi, type SetupState } from './api';

/**
 * The setup wizard's record on the desktop (`GET /api/setup`), shared by every
 * page that asks: the first-run wizard's place, and each plugin's setup
 * version last finished. Asked every five seconds while the window is visible
 * and something listens, and at once after a change. Three-state: `null`
 * until the desktop has answered, so nothing opens or nudges on a guess.
 */

const POLL_MS = 5000;

let snapshot: SetupState | null = null;
let inflight: Promise<SetupState | null> | null = null;
let timer: number | undefined;
const listeners = new Set<() => void>();

function publish(next: SetupState): void {
  snapshot = next;
  for (const l of listeners) l();
}

function fetchNow(): Promise<SetupState | null> {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      // Inside the try: a page that runs without the desktop's setup routes
      // (an older desktop, a test's stand-in API) just has no record.
      const got = await setupApi.get();
      if (got && typeof got === 'object' && got.firstRun) publish(got);
      return snapshot;
    } catch {
      return snapshot;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

function schedule(): void {
  if (timer !== undefined) window.clearInterval(timer);
  timer = undefined;
  if (listeners.size && !document.hidden) timer = window.setInterval(() => void fetchNow(), POLL_MS);
}

function onVisibility(): void {
  if (!document.hidden) void fetchNow();
  schedule();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (listeners.size === 1) {
    document.addEventListener('visibilitychange', onVisibility);
    void fetchNow();
    schedule();
  }
  return () => {
    listeners.delete(listener);
    if (!listeners.size) {
      document.removeEventListener('visibilitychange', onVisibility);
      schedule();
    }
  };
}

/** The record now: `null` until the desktop has answered. */
export function useSetupState(): SetupState | null {
  return useSyncExternalStore(subscribe, () => snapshot);
}

/** Ask again now. */
export function refetchSetup(): Promise<SetupState | null> {
  return fetchNow();
}

/** A write answered with the new record: everyone sees it at once. */
export function setSetupState(next: SetupState): void {
  publish(next);
}

/** For tests: forget the record. */
export function resetSetupState(): void {
  snapshot = null;
  inflight = null;
}

// ---------------------------------------------------------------------------
// Opening the wizard from anywhere (a card, a button, after an install)
// ---------------------------------------------------------------------------

const OPEN_EVENT = 'oaiy:open-setup';

export interface OpenSetup {
  /** A plugin's own wizard; none: the first-run wizard. */
  plugin?: string;
  /** Open at this step (the Agent's `plugin_setup_open`); none: where it was left. */
  step?: string;
}

/** Open the setup page (App listens). */
export function openSetup(target: OpenSetup = {}): void {
  window.dispatchEvent(new CustomEvent<OpenSetup>(OPEN_EVENT, { detail: target }));
}

/** Called with each request to open the setup page. */
export function onOpenSetup(cb: (target: OpenSetup) => void): () => void {
  const handler = (e: Event) => cb((e as CustomEvent<OpenSetup>).detail ?? {});
  window.addEventListener(OPEN_EVENT, handler);
  return () => window.removeEventListener(OPEN_EVENT, handler);
}

// ---------------------------------------------------------------------------
// The old guide's dismissal, carried over once
// ---------------------------------------------------------------------------

const GUIDE_DISMISSED = 'oaiy.setupGuide.dismissed';
const GUIDE_CARRIED = 'oaiy.setup.guideCarried';

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * Someone who dismissed the old setup guide has set this desktop up: their
 * first-run wizard is recorded as finished, once. `true` when it was.
 * (The desktop does the same for a desktop already in use when it first
 * makes its record; this covers the one sign only the window keeps.)
 */
export async function carryGuideDismissal(state: SetupState): Promise<boolean> {
  if (read(GUIDE_CARRIED) === '1') return false;
  const dismissed = read(GUIDE_DISMISSED) === '1';
  try {
    if (dismissed && !state.firstRun.finished) {
      const { migrated: _migrated, ...firstRun } = state.firstRun;
      publish(await setupApi.putFirstRun({ ...firstRun, finished: true }));
    }
    localStorage.setItem(GUIDE_CARRIED, '1');
    return dismissed && !state.firstRun.finished;
  } catch {
    return false;
  }
}

/** The guide was dismissed in this window (the first-run wizard never opens by itself then). */
export function guideDismissed(): boolean {
  return read(GUIDE_DISMISSED) === '1';
}

/** "Run setup again": forget the dismissal and its carry-over, so it does not finish itself again. */
export function forgetGuideDismissal(): void {
  try {
    localStorage.removeItem(GUIDE_DISMISSED);
    localStorage.removeItem(GUIDE_CARRIED);
  } catch {
    /* storage can be unavailable */
  }
}
