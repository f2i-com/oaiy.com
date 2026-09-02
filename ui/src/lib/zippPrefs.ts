/**
 * The Zipp-sandbox toggle, and the Worker factory it gates.
 *
 * Untrusted package workflows run in a Web Worker: a separate realm, with the
 * dangerous globals `defineProperty`'d away and the source scanned for escape
 * patterns. Both layers say in their own comments that this is best-effort — a
 * full browser realm has every capability, and the defence is removing them one
 * name at a time.
 *
 * Zipp is a JavaScript engine compiled to WebAssembly whose guest global is a
 * positive allowlist that never held a host object. A script that successfully
 * reconstructs `globalThis` there finds no `fetch`, no `Worker`, no
 * `importScripts` — not hidden, absent. It also enforces instruction and heap
 * budgets, so a runaway loop or allocation stops on its own.
 *
 * It is therefore the DEFAULT engine for package workflows. The trade is a
 * ~1.2 MB (compressed) download on the first package run, an interpreter rather
 * than a JIT, and a ~16 MiB ceiling on any single value crossing the boundary;
 * the toggle exists so a user who hits one of those can fall back to the plain
 * Worker for a run without editing anything.
 *
 * Only package (hardened) workflows are affected either way. A user's own flows
 * are trusted and run in-thread on the browser's engine — see `executeScript`
 * in oaiy-core's runtime.ts.
 *
 * # Read per run, not per engine
 *
 * The JobManager is built once (`useState` lazy init in JobQueueContext) and
 * lives for the page. If this module handed it a factory-or-undefined at that
 * moment, the toggle would only apply after a reload — which is exactly what an
 * earlier version did while its comment claimed otherwise. So the factory is
 * ALWAYS supplied, and it is the factory that consults the flag each time it is
 * asked for a Worker: Zipp when on, oaiy-core's own Blob-URL Worker when off.
 */

import { STORAGE_KEYS } from './storageKeys';
import {
  spawnUntrustedWorker,
  type UntrustedWorkerFactory,
} from 'oaiy-core/src/untrusted-executor';

/** Default when the user has expressed no preference. */
const DEFAULT_ENABLED = true;

/** Whether the browser can host the Zipp engine at all. */
export function zippSandboxSupported(): boolean {
  return (
    typeof Worker !== 'undefined' &&
    typeof WebAssembly !== 'undefined' &&
    typeof WebAssembly.instantiate === 'function'
  );
}

export function zippSandboxEnabled(): boolean {
  if (!zippSandboxSupported()) return false;
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.ZIPP_SANDBOX_ENABLED);
    if (raw === null) return DEFAULT_ENABLED;
    return raw === 'true';
  } catch {
    // Private mode, or storage disabled. Fall back to the default rather than
    // failing the run.
    return DEFAULT_ENABLED;
  }
}

export function setZippSandboxEnabled(enabled: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEYS.ZIPP_SANDBOX_ENABLED, String(enabled));
  } catch {
    /* storage unavailable; the toggle simply does not persist */
  }
}

/**
 * A Worker running the Zipp engine, speaking oaiy-core's untrusted-workflow
 * protocol. Constructed per run and terminated by `runInWorker` on finish,
 * error or abort — which is what makes a WebAssembly trap survivable: a trapped
 * Engine cannot be disposed, so the whole instance is discarded instead.
 */
function spawnZippWorker(): ReturnType<UntrustedWorkerFactory> {
  const worker = new Worker(
    new URL('../workers/zipp-untrusted-worker.ts', import.meta.url),
    { type: 'module', name: 'oaiy-zipp-sandbox' },
  );
  return {
    worker,
    cleanup: () => worker.terminate(),
  };
}

/**
 * The factory handed to the JobManager. Decides per invocation — i.e. per
 * untrusted run — so a change in Settings applies to the next run without a
 * reload. Falls back to oaiy-core's default Worker when the toggle is off or
 * the browser cannot host WebAssembly, so the untrusted path never silently
 * loses its isolation.
 */
export const untrustedWorkerFactory: UntrustedWorkerFactory = () =>
  zippSandboxEnabled() ? spawnZippWorker() : spawnUntrustedWorker();
