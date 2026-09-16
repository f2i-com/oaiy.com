/**
 * The host's script engine — what every compiled flow, trusted or hardened,
 * runs on when a host supplies one (`RuntimeConfig.scriptExecutor`).
 *
 * Unlike `UntrustedWorkerFactory` this does not require a real `Worker`, or a
 * `Worker` global at all, so a Node host can hand the runtime a
 * `worker_threads`-backed engine. Whatever is returned must speak the
 * `init` / `host_call` / `host_result` / `console` / `finish` /
 * `finish_error` protocol documented in `untrusted-executor.ts`.
 */
export interface WorkerLike {
  // Loose callback parameter on purpose: `runInWorker` annotates its
  // listeners as `(ev: MessageEvent)` / `(ev: ErrorEvent)`, and under
  // `strict` a narrower parameter type here would be a contravariance error.
  addEventListener(type: 'message' | 'error' | 'messageerror', fn: (ev: any) => void): void;
  postMessage(msg: unknown): void;
}

/** Spawns the engine for one script run; `cleanup` tears it down. */
export type ScriptExecutor = () => { worker: WorkerLike; cleanup: () => void };
