/**
 * The Worker that runs untrusted package workflows on the Zipp engine.
 *
 * It is a shell. Everything interesting — the script wrapper and the pump loop
 * Zipp leaves to its embedder — lives in `oaiy-core/src/zipp-executor.ts`, so
 * it can be tested under Node against the real WebAssembly module without a
 * browser (see `ui/tests/zipp-executor.mjs`). What this file adds is the two
 * things only the bundler can provide: the module URL of the engine glue and
 * the URL of the `.wasm` beside it.
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
 * maximum before the budget noticed. Fixed in zipp.org `833680d8`, which the
 * vendored artifact includes; the ceiling now reports a catchable RangeError
 * for every allocation shape.
 *
 * That is also why the module is instantiated per Worker rather than shared:
 * a fresh linear memory per workflow is the isolation, and `new Engine()` costs
 * ~0.02 ms against a ~20 ms one-time instantiation the browser caches.
 */

import init, { Engine } from '../../vendor/zipp-wasm/zipp_wasm.js';
import wasmUrl from '../../vendor/zipp-wasm/zipp_wasm_bg.wasm?url';
import {
  buildZippScript,
  ZippSession,
  type ZippConsoleLevel,
  type ZippEngine,
} from 'oaiy-core/src/zipp-executor';

interface InitMessage {
  type: 'init';
  script: string;
  hardened?: boolean;
}

type Incoming =
  | InitMessage
  | { type: 'host_result'; id: number; result: unknown }
  | { type: 'abort' };

/** Resolvers for `host_call`s awaiting a `host_result` from the main thread. */
const pending = new Map<number, (result: unknown) => void>();
let nextCallId = 1;
let session: ZippSession | null = null;
let started = false;

function post(message: unknown): void {
  (self as unknown as Worker).postMessage(message);
}

/**
 * Round-trip one module call to the main thread's broker. Never rejects: a
 * broker failure comes back as `{ __error__ }`, which `__step` turns into a
 * throw at the guest's `await` site. Rejecting here instead would surface as a
 * trapped workflow rather than a catchable error inside the script.
 */
function callHost(kind: string, args: string[]): Promise<unknown> {
  return new Promise(resolve => {
    const id = nextCallId++;
    pending.set(id, resolve);
    post({ type: 'host_call', id, kind, args });
  });
}

self.addEventListener('message', (event: MessageEvent<Incoming>) => {
  const message = event.data;
  if (!message || typeof message !== 'object') return;

  if (message.type === 'host_result') {
    const resolve = pending.get(message.id);
    if (resolve) {
      pending.delete(message.id);
      resolve(message.result);
    }
    return;
  }

  if (message.type === 'abort') {
    session?.abort();
    return;
  }

  if (message.type === 'init') {
    // The main thread posts `init` exactly once per Worker. Guard anyway: a
    // second compile would run a second workflow against the same host bridge
    // and interleave two sets of results into one flow.
    if (started) return;
    started = true;
    void run(message);
  }
});

async function run(message: InitMessage): Promise<void> {
  let engine: Engine | null = null;
  try {
    await init({ module_or_path: wasmUrl });
    engine = new Engine();

    session = new ZippSession(engine as unknown as ZippEngine, {
      onHostCall: callHost,
      onConsole: (level: ZippConsoleLevel, args: string[]) =>
        post({ type: 'console', level, args }),
      onFinish: (value: unknown) => post({ type: 'finish', value }),
      onError: (msg: string) => post({ type: 'finish_error', message: msg }),
    });

    await session.run(buildZippScript(message.script, message.hardened !== false));
  } catch (e) {
    post({
      type: 'finish_error',
      message: `Zipp worker failed: ${e instanceof Error ? e.message : String(e)}`,
    });
  } finally {
    // Best-effort: a trapped Engine throws here, and there is nothing useful to
    // do about it — the main thread terminates this Worker either way.
    try {
      engine?.dispose();
    } catch {
      /* trapped instance; the Worker is discarded next */
    }
  }
}

post({ type: 'ready' });
