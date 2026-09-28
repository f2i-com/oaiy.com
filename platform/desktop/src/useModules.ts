import { useSyncExternalStore } from 'react';
import { modules as modulesApi, type ModulesSnapshot } from './api';

/**
 * The desktop's modules (the phone, the calendar: on only while a plugin
 * provides them), shared by every page that asks. Asked every five seconds
 * while the window is visible, with the last ETag so an unchanged answer is a
 * bare 304, and at once after a plugin is installed, removed, or turned on or
 * off. Three-state: `null` until the desktop has said, so nothing is hidden or
 * shown on a guess.
 */

const POLL_MS = 5000;

let snapshot: ModulesSnapshot | null = null;
let etag: string | null = null;
let inflight: Promise<void> | null = null;
let timer: number | undefined;
const listeners = new Set<() => void>();

function fetchNow(): Promise<void> {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const got = await modulesApi.list(etag);
      if (got) {
        snapshot = got.snapshot;
        etag = got.etag;
        for (const listener of listeners) listener();
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

/** The modules now: `null` until the desktop has said. */
export function useModules(): ModulesSnapshot | null {
  return useSyncExternalStore(subscribe, () => snapshot);
}

/** Whether `id` is on: `null` until known. */
export function moduleOn(modules: ModulesSnapshot | null, id: string): boolean | null {
  if (!modules) return null;
  return modules.modules.some((m) => m.id === id && m.enabled);
}

/** Ask again now (a plugin was installed, removed, or turned on or off). */
export async function refetchModules(): Promise<void> {
  await inflight;
  await fetchNow();
}

/** Tests: forget what is known. */
export function resetModules(): void {
  snapshot = null;
  etag = null;
  inflight = null;
}
