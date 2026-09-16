/**
 * The Worker that runs workflows on the Zipp engine.
 *
 * Every flow lands here while the Flow sandbox toggle is on: package flows
 * hardened (strict mode, null-prototype `this`, the shadow preamble), a user's
 * own flows as-is. The `hardened` flag on the init message is the only
 * difference the engine sees; see `buildZippScript`.
 *
 * It is a shell. Everything interesting lives in oaiy-core, so it can be
 * tested under Node against the real WebAssembly module without a browser
 * (see `ui/tests/zipp-executor.mjs`): the script wrapper and the pump loop
 * Zipp leaves to its embedder in `zipp-executor.ts`, and the worker side of
 * the message protocol — init once, `host_call`/`host_result` bookkeeping,
 * abort, `ready` — in `zipp-worker-loop.ts`, which the Node CLI's
 * `worker_threads` shell serves over the same loop. What this file adds is
 * the two things only the bundler can provide: the module URL of the engine
 * glue and the URL of the `.wasm` beside it.
 *
 * The protocol is `untrusted-executor.ts`'s, unchanged, so the main thread does
 * not know or care which engine answered:
 *
 *   main → worker:  { type: 'init', script, hardened }
 *   worker → main:  { type: 'host_call',    id, kind, args }
 *   main → worker:  { type: 'host_result',  id, result }
 *   worker → main:  { type: 'console',      level, args }
 *   worker → main:  { type: 'finish',       value }
 *   worker → main:  { type: 'finish_error', message }
 *   main → worker:  { type: 'abort' }
 *
 * # One workflow per Worker, and why that is not just tidiness
 *
 * A WebAssembly trap leaves an Engine that cannot even be disposed —
 * `dispose()` throws "recursive use of an object detected" — so it would leak.
 * The WebAssembly *module* survives fine, but the only way to be sure nothing
 * is left behind is to discard the whole Worker after each run. The main thread
 * already does exactly that: `runInWorker` terminates on `finish`,
 * `finish_error`, abort and error. So a trap costs one Worker, which was going
 * away regardless.
 *
 * Guest code could once force that trap on purpose — the heap ceiling was
 * re-checked on an instruction stride, and one instruction can commit
 * megabytes, so a loop of large allocations reached the module's linked memory
 * maximum before the budget noticed. Fixed in zipp.org `833680d8`, which every
 * ZIPP release since v0.0.11 includes (the engine is installed from a release,
 * scripts/fetch-zipp-release.mjs); the ceiling now reports a catchable
 * RangeError for every allocation shape.
 *
 * That is also why the module is instantiated per Worker rather than shared:
 * a fresh linear memory per workflow is the isolation, and `new Engine()` costs
 * ~0.02 ms against a ~20 ms one-time instantiation the browser caches.
 */

import init, { Engine } from '../../vendor/zipp-wasm/zipp_wasm.js';
import wasmUrl from '../../vendor/zipp-wasm/zipp_wasm_bg.wasm?url';
import { type ZippEngine } from 'oaiy-core/src/zipp-executor';
import { serveZippWorkflow } from 'oaiy-core/src/zipp-worker-loop';

// The handler is registered synchronously, here at module evaluation: the
// main thread posts `init` the moment the Worker exists and nothing waits for
// `ready`, so the WebAssembly module is loaded inside `createEngine` — after
// `init` has arrived — rather than behind a top-level await in front of the
// listener. No instruction budget is passed: the browser runs on the engine's
// default, as it always has.
serveZippWorkflow(
  {
    post: message => (self as unknown as Worker).postMessage(message),
    onMessage: callback => self.addEventListener('message', event => callback(event.data)),
  },
  async () => {
    await init({ module_or_path: wasmUrl });
    return new Engine() as unknown as ZippEngine;
  },
);
