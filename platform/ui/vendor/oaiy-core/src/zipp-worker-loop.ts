/**
 * The worker side of the untrusted-workflow protocol, on the Zipp engine.
 *
 * `runInWorker` (runtime.ts) speaks one protocol to whatever runs a script for
 * it, and does not know or care which engine answers:
 *
 *   main → worker:  { type: 'init', script, hardened }
 *   worker → main:  { type: 'host_call',    id, kind, args }
 *   main → worker:  { type: 'host_result',  id, result }
 *   worker → main:  { type: 'console',      level, args }
 *   worker → main:  { type: 'finish',       value }
 *   worker → main:  { type: 'finish_error', message }
 *   main → worker:  { type: 'abort' }
 *
 * This module is the bookkeeping that protocol needs on the worker side —
 * init once, the `host_call` → `host_result` map, abort, `ready` — written
 * against a two-method `port` rather than a browser `Worker` global, so the
 * same loop serves a Web Worker (`self.postMessage` / `addEventListener`) and
 * a Node `worker_threads` port (`parentPort.postMessage` / `.on('message')`).
 * It is host-neutral by construction: nothing here touches `self`, `window`,
 * `process` or `console`. What each shell adds is the one thing only it can:
 * how the WebAssembly module is loaded, which is why the engine arrives
 * through `createEngine` and may be asynchronous.
 *
 * # One workflow per worker
 *
 * The main thread posts `init` exactly once per worker and terminates it on
 * `finish`, `finish_error`, abort and error. A WebAssembly trap leaves an
 * Engine that cannot even be disposed — `dispose()` throws "recursive use of
 * an object detected" — so discarding the whole worker after each run is the
 * only way to be sure nothing is left behind; a trap then costs one worker
 * that was going away regardless. The loop guards the init-once rule anyway:
 * a second compile would run a second workflow against the same host bridge
 * and interleave two sets of results into one flow.
 */

// `.ts`, not the `.js` the rest of oaiy-core writes: this module is imported
// straight into Node by `ui/tests/zipp-executor.mjs` (native type stripping,
// which does not rewrite extensions), exactly as `zipp-executor.ts` is. tsc
// (`allowImportingTsExtensions`), Vite and esbuild all resolve it as well.
import {
  buildZippScript,
  ZippSession,
  type ZippConsoleLevel,
  type ZippEngine,
  type ZippSessionOptions,
} from './zipp-executor.ts';

/** The two things a message port has to do for the loop. */
export interface ZippWorkerPort {
  post(message: unknown): void;
  onMessage(callback: (message: unknown) => void): void;
}

export interface ZippWorkflowInit {
  type: 'init';
  script: string;
  hardened?: boolean;
}

export type ZippWorkflowIncoming =
  | ZippWorkflowInit
  | { type: 'host_result'; id: number; result: unknown }
  | { type: 'abort' };

export interface ZippWorkflowServeOptions extends ZippSessionOptions {
  /**
   * A script profile's preamble (`zipp-script.ts`), placed at program top
   * level by `buildZippScript` for a TRUSTED workflow. A hardened `init` with
   * a preamble set here is refused by `buildZippScript` and the run fails.
   */
  preamble?: string;
}

/**
 * Serve one workflow over `port`. Registers the message handler, posts
 * `ready`, and returns; the run itself happens when `init` arrives.
 *
 * `createEngine` is called once, on `init`. A browser shell loads the
 * WebAssembly module inside it (so the handler is registered synchronously at
 * module evaluation and no early `init` is lost); a Node shell that already
 * ran `initSync` returns `new Engine()` directly. Either way the engine is
 * disposed after the run, best-effort.
 */
export function serveZippWorkflow(
  port: ZippWorkerPort,
  createEngine: () => ZippEngine | Promise<ZippEngine>,
  opts: ZippWorkflowServeOptions = {},
): void {
  const { preamble, ...sessionOptions } = opts;
  /** Resolvers for `host_call`s awaiting a `host_result` from the main thread. */
  const pending = new Map<number, (result: unknown) => void>();
  let nextCallId = 1;
  let session: ZippSession | null = null;
  let started = false;

  /**
   * Round-trip one module call to the main thread's broker. Never rejects: a
   * broker failure comes back as `{ __error__ }`, which `__step` turns into a
   * throw at the guest's `await` site. Rejecting here instead would surface
   * as a trapped workflow rather than a catchable error inside the script.
   */
  function callHost(kind: string, args: string[]): Promise<unknown> {
    return new Promise(resolve => {
      const id = nextCallId++;
      pending.set(id, resolve);
      port.post({ type: 'host_call', id, kind, args });
    });
  }

  async function run(message: ZippWorkflowInit): Promise<void> {
    let engine: ZippEngine | null = null;
    try {
      engine = await createEngine();

      session = new ZippSession(
        engine,
        {
          onHostCall: callHost,
          onConsole: (level: ZippConsoleLevel, args: string[]) =>
            port.post({ type: 'console', level, args }),
          onFinish: (value: unknown) => port.post({ type: 'finish', value }),
          onError: (msg: string) => port.post({ type: 'finish_error', message: msg }),
        },
        sessionOptions,
      );

      await session.run(buildZippScript(message.script, message.hardened !== false, { preamble }));
    } catch (e) {
      port.post({
        type: 'finish_error',
        message: `Zipp worker failed: ${e instanceof Error ? e.message : String(e)}`,
      });
    } finally {
      // Best-effort: a trapped Engine throws here, and there is nothing useful
      // to do about it — the main thread terminates this worker either way.
      try {
        engine?.dispose();
      } catch {
        /* trapped instance; the worker is discarded next */
      }
    }
  }

  port.onMessage((raw: unknown) => {
    const message = raw as ZippWorkflowIncoming | null | undefined;
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
      if (started) return;
      started = true;
      void run(message);
    }
  });

  port.post({ type: 'ready' });
}
