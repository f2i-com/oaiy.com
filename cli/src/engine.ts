/**
 * Headless flow runner. Wires the SHARED, host-agnostic engine
 * (`ui/src/engine/createEngine`) to the Node host and drives a single job to a
 * terminal state — the Node analogue of the browser's JobQueueProvider +
 * CliWorkflowRunner, minus React.
 */
// Side effect: install window.__TAURI__ before any shared code reads it.
import './node-host/core';
import { createEngine } from '../../ui/src/engine/createEngine';
import type { Flow, WorkflowGraph } from 'oaiy-core';
import { loadNodeBundledModules } from './generated/bundled-modules';
import { closeAll as closeBrowserSessions } from './node-host/browser';

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
}

export interface RunResult {
  success: boolean;
  status: string;
  jobId: string;
  /** Per-node outputs (the user-facing result). */
  results: Record<string, unknown>;
  /** Full workflow context (everything, incl. internal `__` keys). */
  result: unknown;
  error?: string;
  logs: unknown[];
}

const TERMINAL = new Set(['completed', 'failed', 'aborted']);
const DEFAULT_TIMEOUT_MS = 45 * 60 * 1000;

export async function runFlow(graph: WorkflowGraph, opts: RunOptions = {}): Promise<RunResult> {
  // Populate the shared module registry (the singleton createEngine reads).
  await loadNodeBundledModules();

  const engine = createEngine({
    // Subflow and macro nodes resolve their target by id out of this list. It was
    // never passed, so those nodes could not execute headlessly at all.
    availableFlows: opts.availableFlows,
    projectConstants: opts.constants,
    networkPermissionHandler: async () => ({
      allowed: opts.allowLocalNetwork !== false,
      remember: false,
    }),
  });

  try {
    const flowName = opts.flowName ?? 'cli-run';
    const jobId = engine.submit('cli-run', flowName, graph, opts.inputs);

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
      error: success
        ? undefined
        : job?.error ?? (status === 'aborted' ? 'Run aborted (timed out?)' : 'Workflow failed'),
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
