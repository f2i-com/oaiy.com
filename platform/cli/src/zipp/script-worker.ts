/**
 * The `worker_threads` shell behind `oaiy script`: ONE leaf job per message on
 * the ZIPP engine, through `zipp-script.ts`'s `runScriptJob`.
 *
 * Built by `cli/esbuild.mjs` as its own entry (`dist/oaiy-script-worker.mjs`)
 * exactly like the workflow shell (`workflow-worker.ts`): the release's glue
 * bundled in, the compiled `WebAssembly.Module` arriving through `workerData`,
 * `initSync`, console to stderr. What differs is the loop. A workflow runs
 * once and the worker is discarded; this worker stays and answers job after
 * job, each on a fresh Engine, until the main thread's `ScriptHost` retires
 * it — after a resource/host error or a trap, when the instance's retained
 * dynamic-definition bytes pass the budget, after enough jobs, or when a job
 * runs past its watchdog and the thread is terminated mid-job.
 *
 * Protocol with `script-host.ts` (internal, never on the wire):
 *
 *   worker → main:  { type: 'ready', engine }                       once, at start
 *   main → worker:  { type: 'job', seq, job, profile? }             one validated job
 *   worker → main:  { type: 'done', seq, result, usage }            its result and the
 *                                                                   instance's usage
 *
 * `engine` is `scriptEngineIdentity(zippProfile(), …)`: the running engine's
 * own version and revision, plus the release tag and the .wasm digest the main
 * thread read from the build's define — never a literal. `usage.retainedBytes`
 * is what `zippInstanceUsage()` says the disposed engines left behind in this
 * WASM instance (functions + classes compiled at run time), the measurement the
 * host recycles on; `usage.trapped` is set once a WebAssembly trap has poisoned
 * the instance, after which nothing more should run here.
 *
 * This shell also supplies the one host-side check the envelope cannot make
 * for itself: `parseExpression`, which decides mode `parse` with a real parser
 * rather than a string wrapper (`parseOneExpression`).
 *
 * The request was validated on the main thread; a job that reaches this thread
 * is well-formed. Whatever else goes wrong — the engine throwing where the
 * envelope did not expect it, a message this shell does not understand — is
 * answered as a `host` result rather than left unanswered: the main thread
 * must never wait on a job that produced no reply.
 */
import { parentPort, workerData } from 'node:worker_threads';
import { format } from 'node:util';
import { parseExpressionAt } from 'acorn';
import { Engine, initSync, zippInstanceUsage, zippProfile } from '../../../ui/vendor/zipp-wasm-python/zipp_wasm.js';
import {
  runScriptJob,
  scriptEngineIdentity,
  type ScriptEngineCtor,
  type ScriptJob,
  type ScriptJobResult,
  type ScriptProfile,
} from 'oaiy-core/src/zipp-script';

for (const level of ['log', 'info', 'debug', 'warn', 'error'] as const) {
  console[level] = (...args: unknown[]) => {
    process.stderr.write(format(...args) + '\n');
  };
}

const port = parentPort;
if (!port) {
  throw new Error('oaiy-script-worker: not running as a worker_threads Worker');
}

const { wasmModule, release, wasmSha256 } = workerData as {
  wasmModule?: unknown;
  release?: unknown;
  wasmSha256?: unknown;
};
if (!(wasmModule instanceof WebAssembly.Module)) {
  throw new Error('oaiy-script-worker: workerData carries no compiled WebAssembly.Module');
}
if (typeof release !== 'string' || typeof wasmSha256 !== 'string') {
  throw new Error('oaiy-script-worker: workerData carries no engine identity (release, wasmSha256)');
}
initSync({ module: wasmModule });

const engine = scriptEngineIdentity(zippProfile(), { release, wasmSha256 });
const EngineCtor = Engine as unknown as ScriptEngineCtor;

let trapped = false;

export interface ScriptWorkerUsage {
  /** Bytes of run-time-compiled functions and classes the instance still holds after every dispose. */
  retainedBytes: number;
  /** A WebAssembly trap has poisoned this instance; it must be retired. */
  trapped: boolean;
}

function usage(): ScriptWorkerUsage {
  try {
    const u = zippInstanceUsage() as { retainedFunctionBytes?: unknown; retainedClassBytes?: unknown } | null;
    const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
    return { retainedBytes: n(u?.retainedFunctionBytes) + n(u?.retainedClassBytes), trapped };
  } catch {
    // An instance that cannot even report on itself is not one to keep.
    return { retainedBytes: Number.MAX_SAFE_INTEGER, trapped: true };
  }
}

const errText = (e: unknown): string => (typeof e === 'string' ? e : e instanceof Error ? e.message || String(e) : String(e));

/**
 * Mode `parse`'s real check: is `source` ONE expression and nothing else?
 *
 * The engine-side wrapper cannot answer that — it concatenates the source into
 * `return (` + source + `)`, so `1); globalThis.pwned = (1` closes the
 * parenthesis, adds a statement and compiles, and the job came back
 * `{ok: true, value: null}` for source that is not an expression at all. No
 * textual wrapper fixes it (a comma survives every bracket shape), so this
 * parses.
 *
 * `parseExpressionAt` at offset 0, then everything after `node.end` must be
 * blank: that is what rejects a trailing `; statement`, a stray `)` and a
 * second comma-joined expression, while leaving a legitimate expression with
 * trailing whitespace or a newline alone. Acorn parses; it never evaluates,
 * and Guard A scans the shell it lands in.
 */
function parseOneExpression(source: string): void {
  // `preserveParens`, or `node.end` stops short of a wrapping `)` and a
  // perfectly ordinary `(function () {})` is reported as two things.
  const node = parseExpressionAt(source, 0, { ecmaVersion: 'latest', preserveParens: true }) as { end: number };
  const rest = source.slice(node.end);
  if (rest.trim() !== '') {
    throw new SyntaxError(`expected one expression, and ${JSON.stringify(rest.trim().slice(0, 40))} follows it`);
  }
}

port.on('message', (raw: unknown) => {
  const message = raw as { type?: unknown; seq?: unknown; job?: ScriptJob; profile?: ScriptProfile } | null;
  if (!message || typeof message !== 'object' || message.type !== 'job') return;
  const seq = typeof message.seq === 'number' ? message.seq : -1;
  const id = typeof message.job?.id === 'string' ? message.job.id : '';
  let result: ScriptJobResult;
  if (trapped) {
    result = { id, ok: false, errorKind: 'host', error: 'Not attempted: this Zipp instance trapped on an earlier job and awaits retirement' };
  } else if (!message.job) {
    result = { id, ok: false, errorKind: 'host', error: 'The script worker received a job message with no job' };
  } else {
    try {
      result = runScriptJob(EngineCtor, message.job, {
        profile: message.profile,
        languages: engine.languages,
        parseExpression: parseOneExpression,
        onEngineTrap: () => {
          trapped = true;
        },
      });
    } catch (e) {
      result = { id, ok: false, errorKind: 'host', error: `The script worker failed: ${errText(e)}` };
    }
  }
  port.postMessage({ type: 'done', seq, result, usage: usage() });
});

port.postMessage({ type: 'ready', engine });
