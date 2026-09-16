/**
 * The CLI's `ScriptExecutor`: one `worker_threads` Worker per script, running
 * the ZIPP engine, presented to the runtime's `runInWorker` as a `WorkerLike`.
 *
 * `runInWorker` speaks the browser Worker idiom — `addEventListener` for
 * `message` / `error` / `messageerror`, `postMessage`, and a `cleanup` it
 * calls on finish, error and abort. This adapts a Node `Worker` to that:
 *
 *   * `message`      → `{ data }`, the shape a `MessageEvent` has;
 *   * `error`        → `{ message }`, from the Worker's `error` event AND from
 *                      any `exit` with a non-zero code. A thread that dies
 *                      (a top-level throw, `process.exit`, an out-of-memory
 *                      kill) emits only `exit`; without this the runtime would
 *                      wait forever for a `finish` that is never coming. The
 *                      normal `terminate()` also exits non-zero, but by then
 *                      `runInWorker` has torn down and ignores it;
 *   * `messageerror` → `{ message }`;
 *   * `cleanup`      → `terminate()`. It is the only stop signal on this path:
 *                      an abort (`--timeout`, a job abort) reaches the runtime's
 *                      `abortSignal`, which tears down and terminates; the loop
 *                      inside a busy VM is interrupted, not asked.
 *
 * Hygiene: the thread starts with an EMPTY environment (`env: {}`) — a token
 * the Desktop puts in the CLI's environment is not in the worker's, before
 * even considering that flow code cannot reach `process` at all — and both
 * of its output streams are piped to this process's stderr, so nothing it
 * writes can land in the CLI's machine-readable stdout.
 */
import { Worker } from 'node:worker_threads';
import type { ScriptExecutor, WorkerLike } from 'oaiy-core/src/script-executor';
import { ZIPP_MAX_INSTRUCTION_BUDGET_STEPS } from 'oaiy-core/src/zipp-executor';
import type { ZippArtifact } from './artifact';

export interface ZippThreadExecutorOptions {
  /**
   * Instruction budget for the top level and every re-entry, in steps
   * (`ZippSessionOptions.instructionBudgetSteps`). Unset leaves the engine's
   * default. Validated here, once, so a bad value is refused before any
   * worker is spawned rather than failing every job.
   */
  instructionBudgetSteps?: number;
  /**
   * A script profile's preamble (`run --profile`), placed at program top level
   * by `buildZippScript` in the worker (`ZippWorkflowServeOptions.preamble`).
   * Already validated by `loadScriptProfile`; carried, never inspected, here.
   */
  preamble?: string;
}

type EventKind = 'message' | 'error' | 'messageerror';

export function createZippThreadExecutor(
  artifact: ZippArtifact,
  opts: ZippThreadExecutorOptions = {},
): ScriptExecutor {
  const steps = opts.instructionBudgetSteps;
  if (steps !== undefined && (!Number.isInteger(steps) || steps < 1 || steps > ZIPP_MAX_INSTRUCTION_BUDGET_STEPS)) {
    throw new RangeError(
      `instructionBudgetSteps must be an integer from 1 to ${ZIPP_MAX_INSTRUCTION_BUDGET_STEPS}, not ${String(steps)}`,
    );
  }

  return () => {
    const worker = new Worker(artifact.workerEntry, {
      workerData: { wasmModule: artifact.module, instructionBudgetSteps: steps, preamble: opts.preamble },
      env: {},
      stdout: true,
      stderr: true,
    });
    // `end: false`: the first worker to exit must not close this process's stderr.
    worker.stdout.pipe(process.stderr, { end: false });
    worker.stderr.pipe(process.stderr, { end: false });

    const listeners: Record<EventKind, Array<(ev: unknown) => void>> = {
      message: [],
      error: [],
      messageerror: [],
    };
    const emit = (kind: EventKind, ev: unknown) => {
      for (const fn of listeners[kind]) fn(ev);
    };

    worker.on('message', (data) => emit('message', { data }));
    worker.on('messageerror', (err) => emit('messageerror', { message: err.message }));
    // Always registered: an `error` with no listener throws in this thread.
    worker.on('error', (err) => emit('error', { message: err instanceof Error ? err.message : String(err) }));
    worker.on('exit', (code) => {
      if (code !== 0) emit('error', { message: `ZIPP worker exited with code ${code}` });
    });

    const adapter: WorkerLike = {
      addEventListener(kind, fn) {
        listeners[kind].push(fn);
      },
      postMessage(msg) {
        worker.postMessage(msg);
      },
    };

    return {
      worker: adapter,
      cleanup: () => {
        void worker.terminate();
      },
    };
  };
}
