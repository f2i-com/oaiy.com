/**
 * Host-side client for the custom-node COMPILER sandbox worker.
 *
 * A package's compiler bundle is UNTRUSTED. It used to run via `new Function` in the main
 * app realm (full localStorage/fetch/DOM/__TAURI_INTERNALS__ access); it now runs inside
 * compiler-sandbox-worker.ts — a Worker realm with no Window APIs and neutralized egress.
 *
 * Robustness against HOSTILE / BROKEN packages (the worker is a single shared thread):
 *  - Every register + compile has a TIMEOUT. A compiler that hangs (infinite loop) or a
 *    worker that wedges can't block flow compilation forever — the op rejects, and the
 *    worker is terminated + recreated so it can't stay wedged.
 *  - On recreate, all known bundles are re-registered from `registeredBundles`.
 *  - The host validates message ids/keys and ignores forged ones.
 */
const TIMEOUT_MS = 8000;

let worker: Worker | null = null;
let nextId = 1;

const pending = new Map<
  number,
  { resolve: (code: string) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }
>();
// Bundles we've been asked to register, kept so we can re-register on a fresh worker.
const registeredBundles = new Map<string, { bundle: string; globalName: string }>();
// In-flight/resolved registration promises (per worker instance).
const registrations = new Map<string, Promise<boolean>>();
const registrationResolvers = new Map<string, (ok: boolean) => void>();
// Keys whose bundle HUNG the worker at load (register never acked). A hostile/broken package
// must not be retried — re-registering it on a fresh worker just re-wedges the shared thread
// (the compile-timeout → resetWorker → re-register loop was a permanent CPU-spin DoS that denied
// compilation to EVERY package). Quarantined keys are skipped until the page reloads.
const quarantined = new Set<string>();
// Backstop: if the worker keeps needing resets, stop recreating it rather than spinning forever.
// Compilation then fails fast (the deferred-compile fallback upstream keeps one failed node from
// failing the whole flow).
let resetCount = 0;
const MAX_RESETS = 6;
let disabled = false;

function resetWorker(reason: string): void {
  if (worker) {
    try {
      worker.terminate();
    } catch {
      /* ignore */
    }
    worker = null;
  }
  // Fail all in-flight compiles — they fall back to null output upstream.
  for (const p of pending.values()) {
    clearTimeout(p.timer);
    p.reject(new Error(`compiler worker reset: ${reason}`));
  }
  pending.clear();
  // Resolve (to false) any in-flight registrations BEFORE clearing, so a caller awaiting one
  // fast-fails instead of hanging on a promise whose resolver we'd otherwise drop. (Also stops a
  // key queued behind a hang from later quarantining ITSELF — its resolver is already gone.)
  for (const resolve of registrationResolvers.values()) resolve(false);
  registrations.clear();
  registrationResolvers.clear();
  if (++resetCount > MAX_RESETS) disabled = true;
}

function sendRegister(key: string, bundle: string, globalName: string): void {
  const w = worker;
  if (!w) return;
  const p = new Promise<boolean>((resolve) => {
    registrationResolvers.set(key, resolve);
    // If the worker never acks (e.g. the bundle hangs at load), fail the registration so
    // compiles fall back to null instead of awaiting forever.
    setTimeout(() => {
      if (registrationResolvers.get(key)) {
        registrationResolvers.delete(key);
        // No ack within the budget → this bundle hung the worker at load. Quarantine it so it's
        // never re-registered (which would re-wedge the shared worker), and terminate the wedged
        // thread to reclaim the pinned CPU core. resetWorker clears the other in-flight resolvers,
        // so keys merely queued behind this hang are NOT mis-quarantined.
        quarantined.add(key);
        resolve(false);
        resetWorker(`register timeout: quarantined ${key}`);
      }
    }, TIMEOUT_MS);
  });
  registrations.set(key, p);
  w.postMessage({ type: 'register', key, bundle, globalName });
}

function getWorker(): Worker | null {
  if (disabled) return null;
  if (!worker) {
    worker = new Worker(new URL('../sandbox/compiler-sandbox-worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.addEventListener('message', (ev: MessageEvent) => {
      const msg = ev.data as
        | { type: 'result'; id: number; code: string }
        | { type: 'error'; id: number; message: string }
        | { type: 'registered'; key: string; ok: boolean }
        | undefined;
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'result' && typeof msg.id === 'number') {
        const p = pending.get(msg.id);
        if (p) {
          clearTimeout(p.timer);
          pending.delete(msg.id);
          // A healthy compile resets the reset budget, so MAX_RESETS only trips on a SUSTAINED
          // storm (consecutive resets with no successful compile between) rather than on unrelated
          // transients accumulated over a long session. Reset on 'result' (not 'registered'): a
          // bundle that hangs only at COMPILE time still re-registers cleanly, so a register-ack
          // reset would let it evade the cap.
          resetCount = 0;
          p.resolve(String(msg.code));
        }
      } else if (msg.type === 'error' && typeof msg.id === 'number') {
        const p = pending.get(msg.id);
        if (p) {
          clearTimeout(p.timer);
          pending.delete(msg.id);
          p.reject(new Error(String(msg.message)));
        }
      } else if (msg.type === 'registered' && typeof msg.key === 'string') {
        registrationResolvers.get(msg.key)?.(!!msg.ok);
        registrationResolvers.delete(msg.key);
      }
    });
    worker.addEventListener('error', (e) => resetWorker(`error: ${e?.message ?? 'worker error'}`));
    worker.addEventListener('messageerror', () => resetWorker('messageerror'));
    // Re-register everything we know about on the fresh worker — EXCEPT quarantined keys, whose
    // bundles hung at load (re-registering one would just re-wedge this fresh worker).
    for (const [key, b] of registeredBundles) {
      if (!quarantined.has(key)) sendRegister(key, b.bundle, b.globalName);
    }
  }
  return worker;
}

/** Load a compiler bundle into the sandbox. Idempotent per key. */
export function registerCompiler(key: string, bundle: string, globalName: string): Promise<boolean> {
  // A bundle that hung at load (or a disabled sandbox) is never (re)registered.
  if (quarantined.has(key) || disabled) return Promise.resolve(false);
  registeredBundles.set(key, { bundle, globalName });
  if (!getWorker()) return Promise.resolve(false);
  if (!registrations.has(key)) sendRegister(key, bundle, globalName);
  return registrations.get(key) ?? Promise.resolve(false);
}

/** Run a registered compiler for one node. Rejects (→ null fallback upstream) on failure/timeout. */
export async function compileInSandbox(key: string, ctx: Record<string, unknown>): Promise<string> {
  if (!registeredBundles.has(key)) throw new Error(`compileInSandbox: no compiler registered for ${key}`);
  if (quarantined.has(key)) throw new Error(`compileInSandbox: compiler quarantined (hung at load) for ${key}`);
  if (!getWorker()) throw new Error(`compileInSandbox: compiler sandbox disabled for ${key}`); // ensures the worker exists + re-registers after a reset
  const reg = registrations.get(key);
  const ok = reg ? await reg : false;
  if (!ok) throw new Error(`compileInSandbox: compiler failed to load for ${key}`);
  const w = getWorker();
  if (!w) throw new Error(`compileInSandbox: compiler sandbox disabled for ${key}`);
  const id = nextId++;
  return new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      // A wedged single-threaded worker stays wedged — terminate + recreate it.
      resetWorker(`compile timeout for ${key}`);
      reject(new Error(`compileInSandbox: timeout for ${key}`));
    }, TIMEOUT_MS);
    pending.set(id, { resolve, reject, timer });
    try {
      // ctx carries node/inputs(Map)/outputVar/sanitizedId/skipVarDeclaration — all
      // structured-cloneable. A non-cloneable field throws here (loud, not silent).
      w.postMessage({ type: 'compile', id, key, ctx });
    } catch (e) {
      clearTimeout(timer);
      pending.delete(id);
      reject(e instanceof Error ? e : new Error(String(e)));
    }
  });
}

/** Drop a compiler's registration (on package unload). Tells the worker to forget it too. */
export function unregisterCompiler(key: string): void {
  registeredBundles.delete(key);
  registrations.delete(key);
  registrationResolvers.delete(key);
  // An explicit unload clears the quarantine so a re-imported package gets a fresh chance.
  quarantined.delete(key);
  if (worker) worker.postMessage({ type: 'unregister', key });
  // Free the worker entirely when nothing is registered and nothing is in flight.
  if (registeredBundles.size === 0 && pending.size === 0 && worker) {
    try {
      worker.terminate();
    } catch {
      /* ignore */
    }
    worker = null;
  }
}
