/**
 * The main-thread side of `oaiy script`: one warm `worker_threads` Worker
 * running `script-worker.ts`, driven a job at a time, watched by a clock, and
 * replaced whenever it can no longer be trusted.
 *
 * # Why a job at a time
 *
 * `runScriptRequest` (zipp-script.ts) runs a whole batch synchronously. A
 * host that hands the batch over has no way to enforce a per-job `budgetMs`:
 * the only lever against a busy VM is `terminate()`, and terminating mid-batch
 * loses every job after the one that overran. So the host validates the whole
 * request first (the same `validateScriptRequest`, with a real sha256 for a
 * profile's digest — a refusal runs nothing) and then posts the jobs one by
 * one. Each gets its own watchdog; a job that overruns is answered `timeout`,
 * its worker is terminated and replaced, and the NEXT job runs on the
 * replacement. A batch never loses a job to another job's fault.
 *
 * # When the worker is replaced (decision 5-2)
 *
 *   (a) a job ends `resource` or `host`, or the worker reports a trap — the
 *       instance may be poisoned (a trapped Engine cannot even be disposed);
 *   (b) the instance's retained dynamic-definition bytes (`zippInstanceUsage`:
 *       functions + classes compiled at run time, which `dispose()` cannot
 *       reclaim) reach `SCRIPT_INSTANCE_RETAINED_BUDGET_BYTES`;
 *   (c) `SCRIPT_INSTANCE_MAX_JOBS` jobs have run on it;
 *   (d) the watchdog fired, or the worker died (an `error`, or an `exit` the
 *       host did not ask for) — a job in flight is answered `host`.
 *
 * `instance` counts workers spawned so far (`pong.instance` on the wire), so a
 * caller can see a recycle happen; `jobsOnInstance` restarts at zero.
 *
 * # The watchdog
 *
 * `budgetMs ?? SCRIPT_DEFAULT_BUDGET_MS` plus `SCRIPT_WATCHDOG_GRACE_MS`, the
 * FormLogic engine's figures. The budget is the requester's statement of how
 * long the job should take; the grace absorbs a cold Engine and message
 * latency, so a job is killed for running long, not for starting late.
 *
 * Hygiene is the workflow executor's: `env: {}`, both streams piped to this
 * process's stderr (stdout is the protocol's), terminate on any doubt.
 */
import { Worker } from 'node:worker_threads';
import {
  validateScriptRequest,
  type ScriptEngineIdentity,
  type ScriptJob,
  type ScriptJobResult,
  type ScriptProfile,
  type ScriptRefusal,
  type ScriptResponse,
} from 'oaiy-core/src/zipp-script';
import type { ZippArtifact } from './artifact';
import { sha256Hex } from './profile';
import type { ScriptWorkerUsage } from './script-worker';

/** Watchdog budget for a job that names none, in ms. */
export const SCRIPT_DEFAULT_BUDGET_MS = 1000;
/** Added to every job's budget before the worker is terminated. */
export const SCRIPT_WATCHDOG_GRACE_MS = 1500;
/** Retained dynamic-definition bytes in one worker's WASM instance before it is recycled. */
export const SCRIPT_INSTANCE_RETAINED_BUDGET_BYTES = 48 * 1024 * 1024;
/** Jobs one worker may serve before it is recycled regardless of measurement. */
export const SCRIPT_INSTANCE_MAX_JOBS = 5000;

export interface ScriptHostOptions {
  /**
   * Spawn a replacement as soon as a worker is retired, so the next job finds
   * one warm (`--serve`). Off, the replacement is spawned by the next job.
   */
  keepWarm?: boolean;
}

/** Why a worker was retired, for the stderr log. */
export type RetireReason = 'error' | 'trap' | 'retained' | 'jobs' | 'timeout' | 'death' | 'shutdown';

interface Pending {
  seq: number;
  jobId: string;
  timer: ReturnType<typeof setTimeout>;
  settle: (outcome: Outcome) => void;
}

interface Outcome {
  result: ScriptJobResult;
  usage?: ScriptWorkerUsage;
  /** The worker is already gone or must go (timeout, death). */
  retire?: RetireReason;
}

/** The identity the response carries when no worker ever reported one: the build's own record, the same fields. */
function identityFromBuild(artifact: ZippArtifact): ScriptEngineIdentity {
  const { release, version, revision, wasmSha256, languages } = artifact.identity;
  return { name: 'zipp', release, version, revision, wasmSha256, languages };
}

export class ScriptHost {
  /** Workers spawned so far; `pong.instance`. */
  instance = 0;
  /** Jobs served by the current worker. */
  jobsOnInstance = 0;
  /** The running engine's identity, once a worker has reported it. */
  engine: ScriptEngineIdentity;

  private worker: Worker | null = null;
  private ready: Promise<ScriptEngineIdentity> | null = null;
  private pending: Pending | null = null;
  private seq = 0;
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;
  private readonly keepWarm: boolean;

  constructor(private readonly artifact: ZippArtifact, opts: ScriptHostOptions = {}) {
    this.keepWarm = opts.keepWarm === true;
    this.engine = identityFromBuild(artifact);
  }

  /** Spawn the worker now (if none) and resolve once it has reported the engine. */
  warm(): Promise<ScriptEngineIdentity> {
    if (this.closed) return Promise.reject(new Error('the script host is shut down'));
    return this.ensureWorker();
  }

  /**
   * Validate and run one request. Requests are served one at a time in
   * arrival order; a refusal runs nothing. Never rejects for anything the
   * request did — every failure is a per-job result or the refusal object.
   */
  run(input: unknown): Promise<ScriptResponse | ScriptRefusal> {
    const task = this.queue.then(() => this.runChecked(input));
    this.queue = task.catch(() => undefined);
    return task;
  }

  /** Terminate the worker and refuse further work. Resolves when the thread has exited. */
  async shutdown(): Promise<void> {
    this.closed = true;
    const worker = this.worker;
    this.retire('shutdown');
    if (worker) await worker.terminate().catch(() => undefined);
  }

  private async runChecked(input: unknown): Promise<ScriptResponse | ScriptRefusal> {
    const checked = validateScriptRequest(input, { sha256: sha256Hex });
    if (!checked.ok) return { v: 1, error: checked.error };
    const { profile, jobs } = checked.request;
    const results: ScriptJobResult[] = [];
    for (const job of jobs) {
      results.push(await this.runJob(job, profile));
    }
    return { v: 1, engine: this.engine, results };
  }

  private async runJob(job: ScriptJob, profile: ScriptProfile | undefined): Promise<ScriptJobResult> {
    if (this.closed) {
      return { id: job.id, ok: false, errorKind: 'host', error: 'The script host is shutting down' };
    }
    try {
      await this.ensureWorker();
    } catch (e) {
      return { id: job.id, ok: false, errorKind: 'host', error: `The script worker could not start: ${e instanceof Error ? e.message : String(e)}` };
    }
    const worker = this.worker;
    if (!worker) {
      return { id: job.id, ok: false, errorKind: 'host', error: 'The script worker is gone' };
    }

    const seq = ++this.seq;
    const budgetMs = job.budgetMs ?? SCRIPT_DEFAULT_BUDGET_MS;
    const outcome = await new Promise<Outcome>((settle) => {
      const timer = setTimeout(() => {
        if (this.pending?.seq !== seq) return;
        this.pending = null;
        settle({
          result: {
            id: job.id,
            ok: false,
            errorKind: 'timeout',
            error: `The job ran past its ${budgetMs} ms budget (plus ${SCRIPT_WATCHDOG_GRACE_MS} ms grace); the engine worker was terminated`,
          },
          retire: 'timeout',
        });
      }, budgetMs + SCRIPT_WATCHDOG_GRACE_MS);
      this.pending = { seq, jobId: job.id, timer, settle };
      worker.postMessage({ type: 'job', seq, job, profile });
    });
    // Credited to the worker that ran it: a death handler may already have
    // spawned the replacement (count reset), which ran nothing.
    if (this.worker === worker) this.jobsOnInstance += 1;

    const retire = outcome.retire ?? this.recyclePolicy(outcome);
    if (retire) {
      // Only the worker this job ran on: a death handler may already have
      // replaced it, and the replacement is not the one at fault.
      if (this.worker === worker) this.retire(retire);
      if (this.keepWarm && !this.closed) void this.ensureWorker().catch(() => undefined);
    }
    return outcome.result;
  }

  /** (a), (b), (c) of decision 5-2, read off a job that completed. */
  private recyclePolicy(outcome: Outcome): RetireReason | null {
    const { result, usage } = outcome;
    if (usage?.trapped) return 'trap';
    if (!result.ok && (result.errorKind === 'resource' || result.errorKind === 'host')) return 'error';
    if ((usage?.retainedBytes ?? 0) >= SCRIPT_INSTANCE_RETAINED_BUDGET_BYTES) return 'retained';
    if (this.jobsOnInstance >= SCRIPT_INSTANCE_MAX_JOBS) return 'jobs';
    return null;
  }

  private ensureWorker(): Promise<ScriptEngineIdentity> {
    if (this.ready) return this.ready;
    const { identity } = this.artifact;
    const worker = new Worker(this.artifact.scriptWorkerEntry, {
      workerData: { wasmModule: this.artifact.module, release: identity.release, wasmSha256: identity.wasmSha256 },
      env: {},
      stdout: true,
      stderr: true,
    });
    // `end: false`: a retired worker must not close this process's stderr.
    worker.stdout.pipe(process.stderr, { end: false });
    worker.stderr.pipe(process.stderr, { end: false });
    this.worker = worker;
    this.instance += 1;
    this.jobsOnInstance = 0;

    this.ready = new Promise<ScriptEngineIdentity>((resolve, reject) => {
      worker.on('message', (raw: unknown) => {
        const message = raw as { type?: unknown } | null;
        if (!message || typeof message !== 'object') return;
        if (message.type === 'ready') {
          const { engine } = message as { engine?: ScriptEngineIdentity };
          if (engine && engine.name === 'zipp') this.engine = engine;
          resolve(this.engine);
        } else if (message.type === 'done') {
          this.onDone(worker, message as { seq?: unknown; result?: ScriptJobResult; usage?: ScriptWorkerUsage });
        }
      });
      // Always registered: an `error` with no listener throws in this thread.
      worker.on('error', (err) => {
        this.onDeath(worker, `the script worker failed: ${err instanceof Error ? err.message : String(err)}`);
        reject(err instanceof Error ? err : new Error(String(err)));
      });
      worker.on('exit', (code) => {
        this.onDeath(worker, `the script worker exited with code ${code}`);
        reject(new Error(`the script worker exited with code ${code} before it was ready`));
      });
    });
    // A spawn failure is reported to the job that meets it; nothing to do here.
    this.ready.catch(() => undefined);
    return this.ready;
  }

  private onDone(worker: Worker, message: { seq?: unknown; result?: ScriptJobResult; usage?: ScriptWorkerUsage }): void {
    const pending = this.pending;
    // A reply from a retired worker, or for a job the watchdog already answered, is dropped.
    if (!pending || this.worker !== worker || pending.seq !== message.seq) return;
    clearTimeout(pending.timer);
    this.pending = null;
    const result: ScriptJobResult = message.result && typeof message.result === 'object' && message.result.id === pending.jobId
      ? message.result
      : { id: pending.jobId, ok: false, errorKind: 'host', error: 'The script worker replied without a result for the job' };
    pending.settle({ result, usage: message.usage });
  }

  /** The worker died on its own (a top-level throw, an out-of-memory kill, an exit the host did not ask for). */
  private onDeath(worker: Worker, what: string): void {
    if (this.worker !== worker) return; // already retired by the host: expected
    this.worker = null;
    this.ready = null;
    const pending = this.pending;
    if (pending) {
      clearTimeout(pending.timer);
      this.pending = null;
      pending.settle({
        result: { id: pending.jobId, ok: false, errorKind: 'host', error: `${what[0].toUpperCase()}${what.slice(1)} while running the job` },
        retire: 'death',
      });
    }
    if (this.keepWarm && !this.closed) void this.ensureWorker().catch(() => undefined);
  }

  /** Terminate the current worker. Its `exit` is then expected and ignored. */
  private retire(reason: RetireReason): void {
    const worker = this.worker;
    if (!worker) return;
    this.worker = null;
    this.ready = null;
    const pending = this.pending;
    if (pending) {
      // Only a shutdown can find a job in flight here; a watchdog or a death
      // has already answered its own. The result is the honest one.
      clearTimeout(pending.timer);
      this.pending = null;
      pending.settle({
        result: { id: pending.jobId, ok: false, errorKind: 'host', error: 'The script host shut down while the job was running' },
        retire: reason,
      });
    }
    if (reason !== 'shutdown') {
      process.stderr.write(`oaiy script: recycling engine worker #${this.instance} after ${this.jobsOnInstance} job(s) (${reason})\n`);
    }
    void worker.terminate().catch(() => undefined);
  }
}
