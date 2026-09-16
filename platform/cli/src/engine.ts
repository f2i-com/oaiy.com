/**
 * Headless flow runner. Wires the SHARED, host-agnostic engine
 * (`ui/src/engine/createEngine`) to the Node host and drives a single job to a
 * terminal state — the Node analogue of the browser's JobQueueProvider +
 * CliWorkflowRunner, minus React.
 *
 * Every flow the CLI runs — `run`, `worker`, the tests — runs on the ZIPP VM
 * in a worker thread, through `createCliEngine`. That is the only way this
 * package builds a `JobManager`, and it cannot be talked out of it.
 */
// Side effect: install window.__TAURI__ before any shared code reads it.
import './node-host/core';
import { createEngine, type CreateEngineOptions } from '../../ui/src/engine/createEngine';
import type { Flow, JobManager, WorkflowGraph } from 'oaiy-core';
import type { WorkflowInputs } from 'oaiy-core';
import { loadNodeBundledModules } from './generated/bundled-modules';
import { closeAll as closeBrowserSessions } from './node-host/browser';
import { loadZippArtifact, EngineUnavailableError } from './zipp/artifact';
import { createZippThreadExecutor } from './zipp/thread-executor';

export interface CliEngineOptions extends CreateEngineOptions {
  /**
   * Instruction budget for a script's top level and every re-entry, in steps
   * (1 to 2e9). Unset runs on the engine's own default, as the browser does.
   */
  instructionBudgetSteps?: number;
}

/**
 * The CLI's engine: the shared `createEngine`, with its script engine fixed.
 *
 * Loads (and, once per process, verifies and compiles) the staged ZIPP
 * artifact FIRST — no engine, no `JobManager`, nothing to submit to — then
 * hands the runtime a `ScriptExecutor` that runs each script on a fresh
 * `worker_threads` Worker and sets `requireScriptExecutor`. Callers pass
 * whatever `createEngine` options they need (`config`, `tauriInvoke`, …)
 * through the spread; the two engine fields come LAST, so no caller can hand
 * the runtime a different engine, or none, and reach its in-thread path.
 *
 * Rejects with `EngineUnavailableError` when the artifact is missing or is
 * not the one this build was made from, and with `RangeError` for a budget
 * outside the engine's range — both before any worker exists.
 */
export async function createCliEngine(opts: CliEngineOptions = {}): Promise<JobManager> {
  const { instructionBudgetSteps, ...engineOptions } = opts;
  const artifact = await loadZippArtifact();
  return createEngine({
    ...engineOptions,
    scriptExecutor: createZippThreadExecutor(artifact, { instructionBudgetSteps }),
    requireScriptExecutor: true,
  });
}

export interface RunOptions {
  inputs?: Record<string, unknown>;
  /** Constant key→value (API keys etc.), resolved by the runtime's getConstant. */
  constants?: Record<string, string>;
  timeoutMs?: number;
  /** Auto-allow local-network (private IP) requests. Default true for headless. */
  allowLocalNetwork?: boolean;
  flowName?: string;
  /**
   * Every flow available for subflow/macro resolution, including the entry flow.
   * The compiler looks its subflow and macro targets up by id in this list, so
   * without it those nodes silently fall through to the unknown-type passthrough.
   */
  availableFlows?: Flow[];
  /**
   * Path to a connector config (`--connector`). Its node types are built into a
   * module and registered before the flow compiles, so a graph from the linked
   * provider resolves. Absent, provider node types are unknown and the compiler
   * refuses the flow — which is the correct outcome, not a silent skip.
   */
  connectorPath?: string;
  /** See `CliEngineOptions.instructionBudgetSteps`. */
  instructionBudgetSteps?: number;
}

export interface RunResult {
  success: boolean;
  status: string;
  jobId: string;
  /** Per-node outputs (the user-facing result). */
  results: Record<string, unknown>;
  /** Full workflow context (everything, incl. internal `__` keys). */
  result: unknown;
  /**
   * What the flow's OUTPUT node declared, which is the flow's answer.
   *
   * Reported separately from `results` because a caller wants the flow's
   * result, not a map of every node's working. A host that hands the whole
   * per-node map back to whoever queued the run leaks this engine's internals
   * into their data model, and anything they wrote against "the flow's result"
   * then addresses the wrong object. `undefined` for a flow with no output
   * node — an absent answer, which is not the same as an empty one.
   */
  output: unknown;
  error?: string;
  /**
   * Why a run did not complete, when the reason is the host's rather than the
   * flow's: `timeout` when `timeoutMs` elapsed and the job was aborted;
   * `engine_unavailable` when the ZIPP artifact could not be had, in which
   * case nothing ran (`jobId` is empty). A flow's own failure has no code.
   */
  errorCode?: 'timeout' | 'engine_unavailable';
  /** The engine the flow ran on. The CLI has exactly one. */
  engine: 'zipp';
  logs: unknown[];
}

/** The context slot the compiled output node assigns. */
const OUTPUT_SLOT = '__output__';

const TERMINAL = new Set(['completed', 'failed', 'aborted']);
const DEFAULT_TIMEOUT_MS = 45 * 60 * 1000;

export async function runFlow(graph: WorkflowGraph, opts: RunOptions = {}): Promise<RunResult> {
  // Populate the shared module registry (the singleton createEngine reads).
  await loadNodeBundledModules();

  // A linked provider's node types, if one was supplied. Loaded AFTER the
  // bundled modules so a node-type collision is caught (loadModule refuses it)
  // instead of shadowing a built-in node.
  if (opts.connectorPath) {
    const { loadConnectorModule } = await import('./connector');
    await loadConnectorModule(opts.connectorPath);
  }

  let engine: JobManager;
  try {
    engine = await createCliEngine({
      // Subflow and macro nodes resolve their target by id out of this list. It was
      // never passed, so those nodes could not execute headlessly at all.
      availableFlows: opts.availableFlows,
      projectConstants: opts.constants,
      networkPermissionHandler: async () => ({
        allowed: opts.allowLocalNetwork !== false,
        remember: false,
      }),
      instructionBudgetSteps: opts.instructionBudgetSteps,
    });
  } catch (e) {
    // No engine: the run did not happen. Reported, not thrown, so `run` and
    // `worker` deliver it the way they deliver any other failed result.
    if (e instanceof EngineUnavailableError) {
      return {
        success: false,
        status: 'failed',
        jobId: '',
        results: {},
        result: undefined,
        output: undefined,
        error: e.message,
        errorCode: e.code,
        engine: 'zipp',
        logs: [],
      };
    }
    throw e;
  }

  try {
    const flowName = opts.flowName ?? 'cli-run';
    const jobId = engine.submit('cli-run', flowName, graph, opts.inputs as WorkflowInputs | undefined);
    let timedOut = false;

    await new Promise<void>((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        unsub();
        resolve();
      };
      const check = () => {
        const job = engine.getJob(jobId);
        if (job && TERMINAL.has(job.status)) finish();
      };
      const unsub = engine.onStateChange(() => check());
      const timer = setTimeout(() => {
        timedOut = true;
        try {
          engine.abort(jobId);
        } catch {
          /* ignore */
        }
        finish();
      }, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      check(); // in case the job already finished synchronously
    });

    const job = engine.getJob(jobId);
    const status = job?.status ?? 'failed';
    const success = status === 'completed';
    return {
      success,
      status,
      jobId,
      results: (job?.nodeOutputs ?? {}) as Record<string, unknown>,
      result: job?.result,
      output:
        job?.result && typeof job.result === 'object'
          ? (job.result as Record<string, unknown>)[OUTPUT_SLOT]
          : undefined,
      error: success
        ? undefined
        : job?.error ?? (status === 'aborted' ? 'Run aborted (timed out?)' : 'Workflow failed'),
      errorCode: !success && timedOut ? 'timeout' : undefined,
      engine: 'zipp',
      logs: job?.logs ?? [],
    };
  } finally {
    // Best-effort teardown of any Playwright/Chromium sessions the flow opened
    // but didn't explicitly close — closeAll() is a no-op when none exist. Without
    // it a browser flow orphans a Chromium child (run exits via process.exit) and
    // `oaiy worker` accumulates leaked sessions across queued runs.
    try {
      await closeBrowserSessions();
    } catch {
      /* best effort */
    }
  }
}
