/**
 * The Zipp-sandbox toggle, and the Worker factory it gates.
 *
 * Untrusted package workflows currently run as JavaScript in a Web Worker: a
 * separate realm, with the dangerous globals `defineProperty`'d away and the
 * source scanned for escape patterns. Both layers say in their own comments
 * that this is best-effort — a full browser realm has every capability, and the
 * defence is removing them one name at a time.
 *
 * Zipp is a JavaScript engine compiled to WebAssembly whose guest global is a
 * positive allowlist that never held a host object. A script that successfully
 * reconstructs `globalThis` there finds no `fetch`, no `Worker`, no
 * `importScripts` — not hidden, absent. It also enforces an instruction budget,
 * so a runaway loop stops on its own.
 *
 * The trade is a ~1.2 MB (compressed) engine downloaded on the first untrusted
 * run, an interpreter rather than a JIT, and a ~16 MiB ceiling on any single
 * value crossing the boundary. Hence a toggle: OFF by default while the
 * compatibility tail is shaken out, and a per-user opt-in until it isn't.
 *
 * The flag is deliberately read at *job* time rather than cached at module load
 * so flipping it in Settings applies to the next run without a reload.
 */

import { STORAGE_KEYS } from './storageKeys';
import type { UntrustedWorkerFactory } from 'oaiy-core/src/untrusted-executor';

/** Default when the user has expressed no preference. */
const DEFAULT_ENABLED = false;

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
 * protocol. Returns `null` when the toggle is off or the browser cannot host
 * it, which is the signal for oaiy-core to keep its default Worker.
 *
 * The Worker is constructed per run and terminated by `runInWorker` on finish,
 * error or abort. That is what makes a WebAssembly trap survivable: a trapped
 * Engine cannot be disposed, so the instance is discarded wholesale.
 */
export function zippWorkerFactory(): UntrustedWorkerFactory | undefined {
  if (!zippSandboxEnabled()) return undefined;
  return () => {
    const worker = new Worker(
      new URL('../workers/zipp-untrusted-worker.ts', import.meta.url),
      { type: 'module', name: 'oaiy-zipp-sandbox' },
    );
    return {
      worker,
      cleanup: () => worker.terminate(),
    };
  };
}
