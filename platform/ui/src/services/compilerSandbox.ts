/**
 * Host-side client for the custom-node COMPILER sandbox worker.
 *
 * A package's compiler bundle is UNTRUSTED. It runs inside compiler-sandbox-worker.ts —
 * a fresh, ephemeral Worker realm per compile with no Window APIs and neutralized egress.
 *
 * Per-compile worker isolation prevents cross-package / cross-compile prototype poisoning
 * (one package mutating Object.prototype/Array.prototype/etc. can never affect another
 * package or subsequent compiles) AND keeps a wedged/hostile compiler from blocking
 * other packages.
 */
const TIMEOUT_MS = 8000;

let nextId = 1;

// Bundles we've been asked to register, kept so each compile spawns an isolated fresh worker.
const registeredBundles = new Map<string, { bundle: string; globalName: string }>();
// Keys whose bundle HUNG a worker at load (register never acked). A hostile/broken package
// is quarantined until explicitly unloaded or the page reloads.
const quarantined = new Set<string>();

/** Register a compiler bundle for isolated per-compile execution. Idempotent per key. */
export function registerCompiler(key: string, bundle: string, globalName: string): Promise<boolean> {
  if (quarantined.has(key)) return Promise.resolve(false);
  if (!bundle.includes(globalName)) return Promise.resolve(false);
  registeredBundles.set(key, { bundle, globalName });
  return Promise.resolve(true);
}

/**
 * Run a registered compiler for one node inside a dedicated fresh Worker, then
 * immediately terminate the Worker so no state or prototype mutations survive.
 * Rejects (→ null fallback upstream) on failure/timeout.
 */
export async function compileInSandbox(key: string, ctx: Record<string, unknown>): Promise<string> {
  const entry = registeredBundles.get(key);
  if (!entry) throw new Error(`compileInSandbox: no compiler registered for ${key}`);
  if (quarantined.has(key)) throw new Error(`compileInSandbox: compiler quarantined (hung at load) for ${key}`);

  const id = nextId++;
  const w = new Worker(new URL('../sandbox/compiler-sandbox-worker.ts', import.meta.url), {
    type: 'module',
  });

  return new Promise<string>((resolve, reject) => {
    let settled = false;
    let registered = false;

    const cleanup = () => {
      clearTimeout(timer);
      try {
        w.terminate();
      } catch {
        /* ignore */
      }
    };

    const fail = (err: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };

    const succeed = (code: string) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(code);
    };

    const timer = setTimeout(() => {
      if (!registered) {
        quarantined.add(key);
      }
      fail(new Error(`compileInSandbox: timeout for ${key}`));
    }, TIMEOUT_MS);

    w.addEventListener('message', (ev: MessageEvent) => {
      if (settled) return;
      const msg = ev.data as
        | { type: 'result'; id: number; code: string }
        | { type: 'error'; id: number; message: string }
        | { type: 'registered'; key: string; ok: boolean; message?: string }
        | undefined;
      if (!msg || typeof msg !== 'object') return;

      if (msg.type === 'registered' && msg.key === key) {
        if (registered) return;
        registered = true;
        if (!msg.ok) {
          fail(new Error(`compileInSandbox: compiler failed to load for ${key}${msg.message ? `: ${msg.message}` : ''}`));
          return;
        }
        try {
          w.postMessage({ type: 'compile', id, key, ctx });
        } catch (e) {
          fail(e instanceof Error ? e : new Error(String(e)));
        }
      } else if (msg.type === 'result' && msg.id === id) {
        succeed(String(msg.code));
      } else if (msg.type === 'error' && msg.id === id) {
        fail(new Error(String(msg.message)));
      }
    });

    w.addEventListener('error', (e) => fail(new Error(`compiler worker error: ${e?.message ?? 'worker error'}`)));
    w.addEventListener('messageerror', () => fail(new Error('compiler worker messageerror')));

    try {
      w.postMessage({ type: 'register', key, bundle: entry.bundle, globalName: entry.globalName });
    } catch (e) {
      fail(e instanceof Error ? e : new Error(String(e)));
    }
  });
}

/** Drop a compiler's registration (on package unload). */
export function unregisterCompiler(key: string): void {
  registeredBundles.delete(key);
  // An explicit unload clears the quarantine so a re-imported package gets a fresh chance.
  quarantined.delete(key);
}
