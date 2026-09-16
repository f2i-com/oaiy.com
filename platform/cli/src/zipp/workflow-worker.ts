/**
 * The `worker_threads` shell that runs one workflow on the ZIPP engine — the
 * Node counterpart of `ui/src/workers/zipp-untrusted-worker.ts`, serving the
 * same `serveZippWorkflow` loop over `parentPort` instead of `self`.
 *
 * Built by `cli/esbuild.mjs` as its own entry (`dist/oaiy-zipp-worker.mjs`),
 * with the release's glue (`zipp_wasm.js`) bundled in statically: the build
 * verified the install those bytes came from, so there is no second file to
 * check at run time. The WebAssembly module arrives compiled, through
 * `workerData` (a `WebAssembly.Module` survives the structured clone), and
 * `initSync` instantiates it — the glue's URL/fetch path is never taken.
 *
 * What the flow sees from in here is the ZIPP VM and nothing else. This
 * thread's own realm is already thin — `createZippThreadExecutor` spawns it
 * with `env: {}` — but the point is that flow code does not run in this realm
 * at all: `process`, `require`, `Buffer` and `globalThis.__TAURI__` are
 * `undefined` to it. Every host call goes back over the port to the main
 * thread's broker, as it does in the browser.
 *
 * stdout belongs to the CLI's machine-readable result, so anything this thread
 * says — the glue's own `console.warn`, a stray log — goes to stderr, and the
 * spawner pipes both of this thread's streams there as well.
 */
import { parentPort, workerData } from 'node:worker_threads';
import { format } from 'node:util';
import { Engine, initSync } from '../../../ui/vendor/zipp-wasm-python/zipp_wasm.js';
import type { ZippEngine } from 'oaiy-core/src/zipp-executor';
import { serveZippWorkflow } from 'oaiy-core/src/zipp-worker-loop';

for (const level of ['log', 'info', 'debug', 'warn', 'error'] as const) {
  console[level] = (...args: unknown[]) => {
    process.stderr.write(format(...args) + '\n');
  };
}

const port = parentPort;
if (!port) {
  throw new Error('oaiy-zipp-worker: not running as a worker_threads Worker');
}

const { wasmModule, instructionBudgetSteps } = workerData as {
  wasmModule?: unknown;
  instructionBudgetSteps?: number;
};
if (!(wasmModule instanceof WebAssembly.Module)) {
  throw new Error('oaiy-zipp-worker: workerData carries no compiled WebAssembly.Module');
}
initSync({ module: wasmModule });

// One workflow per worker; the main thread terminates this thread when it is
// done with it (see `zipp-worker-loop.ts`). `Engine` is created on `init`.
serveZippWorkflow(
  {
    post: (message) => port.postMessage(message),
    onMessage: (callback) => {
      port.on('message', callback);
    },
  },
  () => new Engine() as unknown as ZippEngine,
  { instructionBudgetSteps },
);
