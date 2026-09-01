/**
 * Web-Worker-based execution of untrusted package workflows.
 *
 * This module is wired into `OAIYRuntime.executeScript`: when a flow runs
 * in hardened mode (a package context is set) and the host environment
 * provides `Worker`, `Blob`, and `URL.createObjectURL`, the compiled
 * script is shipped to a worker via `spawnUntrustedWorker` and run in
 * the worker's own realm. Callers can opt out with `RuntimeConfig.
 * useWorkerForUntrusted = false` (mainly for Node test envs without
 * `Worker`); in that case the runtime falls back to the in-thread path
 * that still applies `HARDENED_SHADOW_PREAMBLE` + `detectSandboxEscape`.
 *
 * # Why a Worker?
 *
 * The in-thread isolation (see `HARDENED_SHADOW_PREAMBLE` and
 * `detectSandboxEscape` in `runtime.ts`) layers four defences but cannot
 * prevent a determined adversary from reconstructing the realm's global
 * via prototype chains. A Worker runs in its own JavaScript realm, so a
 * recovered `globalThis` only points at the worker's pristine globals —
 * it does not have access to the main thread's Tauri `invoke`, file
 * handles, or DOM. Any capability the worker needs has to be brokered
 * through `postMessage`, which is exactly the existing host-call model.
 *
 * # Protocol
 *
 * The main thread spawns a worker, posts an `init` message containing
 * the compiled script, and then services `host_call` messages by
 * forwarding them to the same module-function broker the in-thread path
 * uses. The worker is otherwise identical to the in-thread runtime.
 *
 *   main → worker:  { type: 'init',  script: string }
 *   worker → main:  { type: 'host_call',     id, kind, args }
 *   main → worker:  { type: 'host_result',   id, result }
 *   worker → main:  { type: 'console',       level, message }
 *   worker → main:  { type: 'finish',        value }
 *   worker → main:  { type: 'finish_error',  message }
 *   main → worker:  { type: 'abort' }
 *
 * Every host call from the worker carries a monotonic `id` so the main
 * thread's reply is matched to the right pending callback. Aborts are
 * propagated by terminating the worker and rejecting the promise.
 *
 * Integration tests for this path live in
 * `__tests__/worker-isolation.test.ts` and cover: realm isolation,
 * capability denial, abort propagation, and the in-thread fallback.
 */

/**
 * Worker-side bootstrap source. Embedded as a string so the runtime can
 * spawn a Worker from a Blob URL without a separate bundling step. The
 * code uses only ES2018 syntax + `postMessage` / `addEventListener` — both
 * available in every Worker context (browser, Tauri webview, Node 22+).
 */
export const UNTRUSTED_WORKER_SOURCE = `
'use strict';

// SECURITY: neutralize the worker realm's ambient network/egress globals so
// untrusted workflow code can't bypass the host's permission-gated + DNS-pinned
// network broker — neither via fetch(...) nor a RECOVERED globalThis.fetch(...).
// In a worker globalThis === self, so redefining these (non-configurable) on self
// AND every object up its prototype chain (DedicatedWorkerGlobalScope.prototype,
// WorkerGlobalScope.prototype where fetch/importScripts actually live, …) closes the
// reconstruction path the var-shadow + escape-scanner can't (those are best-effort only).
// The host bridge uses only postMessage/addEventListener, so this doesn't affect it.
(function () {
  // Worker/SharedWorker are included so a recovered realm can't spawn a NESTED worker —
  // a FRESH realm WITHOUT this bootstrap, where fetch() is live again — and exfiltrate
  // the flow's data past the permission-gated/DNS-pinned host broker. (We deliberately
  // do NOT shadow Blob/URL.createObjectURL: legit flow code uses Blob for data sizing,
  // and severing the worker constructors already closes the fresh-realm spawn.)
  var egress = ['fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'BroadcastChannel', 'importScripts', 'Request', 'Response', 'Headers', 'navigator', 'Worker', 'SharedWorker'];
  // Walk the FULL prototype chain, not just [self, parent]: per WebIDL, [Global] interface
  // members are installed on the interface PROTOTYPE objects, not flattened onto the global
  // instance. In a real DedicatedWorkerGlobalScope the WindowOrWorkerGlobalScope mixin members —
  // notably fetch / importScripts / the navigator getter — live on WorkerGlobalScope.prototype,
  // the GRANDPARENT (getPrototypeOf(getPrototypeOf(self))), which a 2-element loop skipped, so a
  // recovered global could still call getPrototypeOf(getPrototypeOf(g)).fetch. Stop before
  // Object.prototype so we don't clobber it globally.
  for (var p = self; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
    for (var ei = 0; ei < egress.length; ei++) {
      try { Object.defineProperty(p, egress[ei], { value: undefined, configurable: false, writable: false }); } catch (_) {}
    }
  }
})();

// Pending host-call callbacks keyed by monotonic id.
let __nextId = 1;
const __pending = new Map();

self.addEventListener('message', function (ev) {
  const msg = ev.data;
  if (!msg || typeof msg !== 'object') return;
  if (msg.type === 'init') {
    runScript(msg.script);
    return;
  }
  if (msg.type === 'host_result') {
    const cb = __pending.get(msg.id);
    if (cb) {
      __pending.delete(msg.id);
      try { cb(msg.result); } catch (_) {}
    }
    return;
  }
});

function runScript(script) {
  const host = {
    call: function (kind, args, callback) {
      if (kind === '__system.finish') {
        let value = null;
        try { value = typeof args[0] === 'string' && args[0] ? JSON.parse(args[0]) : args[0]; } catch (_) { value = args[0]; }
        self.postMessage({ type: 'finish', value: value });
        return;
      }
      if (kind === '__system.finish_error') {
        self.postMessage({ type: 'finish_error', message: String(args[0]) });
        return;
      }
      const id = __nextId++;
      __pending.set(id, callback);
      self.postMessage({ type: 'host_call', id: id, kind: kind, args: args });
    },
  };

  const consoleProxy = {
    log: function () { self.postMessage({ type: 'console', level: 'log', args: Array.from(arguments).map(String) }); },
    warn: function () { self.postMessage({ type: 'console', level: 'warn', args: Array.from(arguments).map(String) }); },
    error: function () { self.postMessage({ type: 'console', level: 'error', args: Array.from(arguments).map(String) }); },
    info: function () { self.postMessage({ type: 'console', level: 'info', args: Array.from(arguments).map(String) }); },
    debug: function () {},
  };

  try {
    // The script string already embeds the global-shadow preamble, the
    // sandbox-escape scan happens main-side before posting, and the
    // wrapper is built so user code never reaches the global object
    // through \`this\`.
    const fn = new Function('host', 'console', script);
    fn.call(Object.create(null), host, consoleProxy);
  } catch (e) {
    self.postMessage({ type: 'finish_error', message: 'Worker bootstrap failed: ' + (e && e.message ? e.message : String(e)) });
  }
}
`;

/**
 * Builds the Worker an untrusted workflow runs in, plus the teardown that
 * releases whatever backs it.
 *
 * `OAIYRuntime` accepts one of these via `RuntimeConfig.untrustedWorkerFactory`
 * so a host app can substitute a different engine — `ui/` supplies a Zipp-backed
 * Worker — without oaiy-core needing bundler-resolved URLs. Any implementation
 * must speak the protocol documented at the top of this file.
 */
export type UntrustedWorkerFactory = () => { worker: Worker; cleanup: () => void };

/**
 * Construct a Worker from the bootstrap source above. Returns the Worker
 * plus a cleanup function that revokes the Blob URL.
 *
 * Callers are responsible for posting the `init` message and listening
 * for `host_call`, `console`, `finish`, and `finish_error` events.
 */
export function spawnUntrustedWorker(): { worker: Worker; cleanup: () => void } {
  const blob = new Blob([UNTRUSTED_WORKER_SOURCE], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url);
  return {
    worker,
    cleanup: () => {
      worker.terminate();
      URL.revokeObjectURL(url);
    },
  };
}
