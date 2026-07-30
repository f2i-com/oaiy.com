/**
 * Backend dispatcher — talks to the optional oaiy-api sharing/remote-run
 * backend (see ../../api/). All calls are no-ops when `VITE_API_BASE`
 * is unset, so the UI remains fully functional standalone.
 *
 * Three roles in this module:
 *   1. CRUD helpers (`createFlow`, `updateFlow`, `readFlow`, …) — used
 *      by the share dialog and the auto-save bridge in useProject.
 *   2. The DISPATCHER — a long-polling loop that picks up queued runs
 *      from external callers (e.g. ChatGPT POSTing to /runs/) and
 *      hands them to the consumer's `runHandler` callback. The handler
 *      typically calls back into oaiy-core's workflow runtime with the
 *      supplied inputs.
 *   3. HEARTBEAT — a 30s setInterval that pings /heartbeat so the
 *      backend's `client_connected` flag stays true while the tab is
 *      open. The dispatcher itself implicitly heartbeats every poll
 *      but the explicit beat keeps things alive when the queue is
 *      empty for long stretches.
 *
 * Hash semantics mirror the backend:
 *   - `hash_view`  — read-only share
 *   - `hash_edit`  — read-write + run-trigger; required for everything
 *                    in this module except `readFlow`/`getStatus`.
 *
 * No assumption is made about the consumer's UI layer. The dispatcher
 * fires `onRunRequested` with a per-run approval callback; whether
 * that hits a toast, a modal, or auto-approves is up to the caller.
 */

// ---------------------------------------------------------------------------
// Config + types
// ---------------------------------------------------------------------------

import {
  backendBaseUrl,
  isSharingEnabled,
  getFlowPassword,
} from './sharingPrefs';
import {
  encryptPayload,
  decryptEnvelope,
  isEncryptedEnvelope,
  type EncryptedEnvelope,
} from './flowCrypto';

const API_BASE: string = backendBaseUrl();

/** Service-portable snapshot of a flow — the same shape ProjectIO exports. */
export interface FlowSnapshot {
  flows?: unknown[];
  services?: unknown[];
  settings?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface CreatedFlow {
  id: number;
  hash_view: string;
  hash_edit: string;
  owner_token: string;
  urls: { view: string; edit: string } | null;
}

export interface ReadFlow {
  id: number;
  title: string;
  flow_json: FlowSnapshot;
  created_at: string;
  updated_at: string;
  hash_view: string;
  /** Null when the caller used the view-only hash. */
  hash_edit: string | null;
}

export interface FlowStatus {
  client_connected: boolean;
  last_seen: string | null;
}

export interface QueuedRun {
  id: number;
  inputs: Record<string, unknown>;
  reason: string;
}

/** What the dispatcher hands back when reporting a run outcome. */
export type RunOutcome =
  | { status: 'done'; result?: unknown }
  | { status: 'error'; error: string }
  | { status: 'cancelled' };

export interface DispatcherOptions {
  /** Edit hash for the flow we're listening on. */
  hashEdit: string;
  /**
   * Called when a queued run lands. Return the outcome to POST back to
   * the backend. Throwing is equivalent to `{status: 'error', error: e}`.
   *
   * Implementations typically:
   *   1. Show a per-run approval toast (skip if `autoApprove` is set
   *      in localStorage for this hash).
   *   2. On approval, materialize the flow + execute it via
   *      oaiy-core's workflow runtime with the supplied `inputs`.
   *   3. Return `{status:'done', result: <node output map>}`.
   */
  runHandler: (run: QueuedRun) => Promise<RunOutcome>;
  /** Called whenever a poll iteration completes — handy for status pills. */
  onPoll?: (state: 'idle' | 'busy' | 'error', detail?: string) => void;
  /** Defaults to console.warn — wire to your logger if you have one. */
  onError?: (err: unknown) => void;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/**
 * True if backend sharing is ACTIVE — both a backend URL is configured
 * AND the user hasn't disabled the toggle in Settings. The dispatcher
 * loop short-circuits when this is false, and the share-dialog UI hides
 * itself. The build-time URL alone is necessary but not sufficient.
 */
export function isBackendEnabled(): boolean {
  return API_BASE !== '' && isSharingEnabled();
}

async function apiJson<T>(
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  path: string,
  body?: unknown,
  init: RequestInit = {},
): Promise<T> {
  if (!isBackendEnabled()) {
    throw new Error('backend disabled (VITE_API_BASE is empty)');
  }
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(init.headers as Record<string, string> | undefined),
  };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  // Spread `init` FIRST so its `signal` (long-poll abort) flows through, then let
  // the merged headers/method/body/credentials win — putting `...init` LAST would
  // re-spread init.headers and clobber the merged Accept/Content-Type.
  const resp = await fetch(API_BASE + path, {
    ...init,
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: 'include',
  });
  // 204 = empty success (heartbeat); 200/201/202 = JSON body.
  if (resp.status === 204) return undefined as T;
  const text = await resp.text();
  let parsed: unknown = null;
  try { parsed = text ? JSON.parse(text) : null; }
  catch { /* fall through with text */ }
  if (!resp.ok) {
    // The backend's error body may be a flat `{error: "msg"}` (older) or a
    // nested `{error: {message, type}}` (current). Unwrap both so the thrown
    // message is readable instead of "[object Object]".
    const errField = parsed && typeof parsed === 'object' ? (parsed as { error?: unknown }).error : null;
    const message =
      (errField && typeof errField === 'object' && 'message' in errField
        ? String((errField as { message: unknown }).message)
        : errField != null
          ? String(errField)
          : null) ?? text ?? `HTTP ${resp.status}`;
    throw new Error(`[backend] ${method} ${path} → ${resp.status}: ${message}`);
  }
  return (parsed as T) ?? (undefined as T);
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

/**
 * Create a flow on the backend. If `password` is set, the flow body is
 * AES-GCM encrypted client-side before being sent — the backend just
 * stores opaque ciphertext + IV + salt.
 */
export async function createFlow(
  snapshot: FlowSnapshot,
  opts: { title?: string; password?: string } = {},
): Promise<CreatedFlow> {
  const body: unknown = opts.password
    ? await encryptPayload(snapshot, opts.password)
    : snapshot;
  return apiJson<CreatedFlow>('POST', '/api/flows', {
    title: opts.title ?? '',
    flow_json: body,
  });
}

/**
 * Read a flow. If the body is an encrypted envelope:
 *   - If `password` is provided, decrypt with it (throws on wrong pw).
 *   - Else: throws `PasswordRequiredError` so the caller can prompt
 *     and re-call. The raw envelope is attached to the error for
 *     deferred decryption.
 */
export class PasswordRequiredError extends Error {
  public readonly envelope: EncryptedEnvelope;
  public readonly meta: ReadFlow;
  constructor(envelope: EncryptedEnvelope, meta: ReadFlow) {
    super('flow is encrypted — password required');
    this.envelope = envelope;
    this.meta = meta;
  }
}

export async function readFlow(
  hash: string,
  opts: { password?: string } = {},
): Promise<ReadFlow> {
  const raw = await apiJson<ReadFlow>('GET', `/api/flows/${encodeURIComponent(hash)}`);
  const body = raw.flow_json as unknown;
  if (isEncryptedEnvelope(body)) {
    if (!opts.password) {
      throw new PasswordRequiredError(body, raw);
    }
    const plain = (await decryptEnvelope(body, opts.password)) as FlowSnapshot;
    return { ...raw, flow_json: plain };
  }
  return raw;
}

export async function updateFlow(
  hashEdit: string,
  patch: { flow_json?: FlowSnapshot; title?: string; password?: string },
): Promise<{ ok: true }> {
  const wire: { flow_json?: unknown; title?: string } = {};
  if (patch.title !== undefined) wire.title = patch.title;
  if (patch.flow_json !== undefined) {
    wire.flow_json = patch.password
      ? await encryptPayload(patch.flow_json, patch.password)
      : patch.flow_json;
  }
  return apiJson('PUT', `/api/flows/${encodeURIComponent(hashEdit)}`, wire);
}

export function deleteFlow(hashEdit: string, ownerToken: string): Promise<{ ok: true }> {
  return apiJson('DELETE', `/api/flows/${encodeURIComponent(hashEdit)}`, undefined, {
    headers: { 'X-Owner-Token': ownerToken },
  });
}

export function getStatus(hash: string): Promise<FlowStatus> {
  return apiJson<FlowStatus>('GET', `/api/flows/${encodeURIComponent(hash)}/status`);
}

// ---------------------------------------------------------------------------
// Run report (called by the dispatcher AFTER runHandler resolves)
// ---------------------------------------------------------------------------

const REPORT_RETRIES = 4;
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Retry transient/5xx + network failures; 4xx (404 wrong hash, 409 not
 *  running) are permanent and shouldn't be retried. */
function isRetryable(err: unknown): boolean {
  const m = err instanceof Error ? err.message : String(err);
  const status = m.match(/→ (\d{3}):/);
  if (!status) return true; // fetch/network error — worth retrying
  return Number(status[1]) >= 500;
}

async function reportRunResult(hashEdit: string, runId: number, outcome: RunOutcome): Promise<{ ok: true }> {
  const base =
    outcome.status === 'error'
      ? { status: 'error', error: outcome.error }
      : outcome.status === 'cancelled'
        ? { status: 'cancelled' }
        : { status: 'done', result: outcome.result ?? null };
  // The edit hash authorises the result POST — the backend scopes the
  // update to the flow that owns this run, so a sequential run id alone
  // can't be used to forge another flow's result.
  const body = { ...base, hash: hashEdit };
  // The local run already spent the user's compute, so a transient network
  // blip or a backend restart shouldn't silently drop the result. Retry with
  // exponential backoff; bail immediately on a permanent 4xx.
  for (let attempt = 0; ; attempt++) {
    try {
      return await apiJson<{ ok: true }>('POST', `/api/runs/${runId}/result`, body);
    } catch (err) {
      if (attempt >= REPORT_RETRIES - 1 || !isRetryable(err)) throw err;
      await sleep(Math.min(8_000, 500 * 2 ** attempt));
    }
  }
}

function heartbeat(hashEdit: string): Promise<{ ok: true }> {
  return apiJson('POST', `/api/flows/${encodeURIComponent(hashEdit)}/heartbeat`);
}

// ---------------------------------------------------------------------------
// Long-poll dispatcher
// ---------------------------------------------------------------------------

/**
 * Start a polling loop that picks up queued runs and hands each to
 * `runHandler` in turn. Returns a `stop()` callback — call it when the
 * user closes the flow or navigates away.
 *
 * The loop is sequential: one run at a time, no overlapping execution. This is a
 * deliberate UX/diagnostics choice (the browser is doing real local work per run, and
 * one-at-a-time keeps logs legible) — NOT a safety requirement: the runtime now handles
 * concurrent jobs safely via per-job context isolation (see JobManager.processQueue).
 */
export function startBackendDispatcher(opts: DispatcherOptions): () => void {
  if (!isBackendEnabled()) {
    opts.onError?.(new Error('backend disabled'));
    return () => {};
  }

  let stopped = false;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  const onError = opts.onError ?? ((e) => console.warn('[dispatcher]', e));

  // Independent heartbeat — covers the long stretches between polls
  // when the queue is empty.
  heartbeatTimer = setInterval(() => {
    if (stopped) return;
    heartbeat(opts.hashEdit).catch(onError);
  }, 30_000);

  // Track the in-flight long-poll so stop() can tear it down at once and a client
  // timeout can bound it (the server long-poll holds ~20s; a stalled/proxy-buffered
  // connection would otherwise hang the loop forever).
  let activePoll: AbortController | null = null;
  (async function loop() {
    while (!stopped) {
      try {
        opts.onPoll?.('idle');
        const ctrl = new AbortController();
        activePoll = ctrl;
        const pollTimeout = setTimeout(() => ctrl.abort(), 30_000);
        let next: QueuedRun | null;
        try {
          next = await apiJson<QueuedRun | null>(
            'GET',
            `/api/flows/${encodeURIComponent(opts.hashEdit)}/runs/pending`,
            undefined,
            { signal: ctrl.signal },
          );
        } finally {
          clearTimeout(pollTimeout);
          activePoll = null;
        }
        if (stopped) break;
        if (!next) continue; // long-poll timed out; loop immediately

        opts.onPoll?.('busy', `run #${next.id}`);

        // If the flow is password-protected, the inputs in the queue
        // were encrypted by the AI client with the same password.
        // Decrypt before handing to the run handler. Wrong/missing
        // password → report an error back so the caller knows the run
        // didn't execute.
        let resolvedRun: QueuedRun = next;
        const password = getFlowPassword(opts.hashEdit);
        // Fail closed: encrypted inputs with no password in this browser
        // can't be decrypted — report an error rather than passing raw
        // ciphertext to the run handler (matches readFlow's PasswordRequiredError).
        if (isEncryptedEnvelope(next.inputs as unknown) && !password) {
          await reportRunResult(opts.hashEdit, next.id, {
            status: 'error',
            error: 'inputs are encrypted but no flow password is set in this browser',
          }).catch(onError);
          continue;
        }
        if (password && isEncryptedEnvelope(next.inputs as unknown)) {
          try {
            resolvedRun = {
              ...next,
              inputs: (await decryptEnvelope(
                next.inputs as unknown as EncryptedEnvelope,
                password,
              )) as Record<string, unknown>,
            };
          } catch (err) {
            await reportRunResult(opts.hashEdit, next.id, {
              status: 'error',
              error: 'inputs decrypt failed: ' + String((err as Error).message ?? err),
            }).catch(onError);
            continue;
          }
        }

        // By design, possession of the edit hash IS the run-trigger capability —
        // the owner hands it to whatever drives the flow (including an AI), so runs
        // execute with no per-run gate. (A per-run approval toggle was scaffolded in
        // sharingPrefs but never wired; the dead read here was removed.)
        let outcome: RunOutcome;
        try {
          outcome = await opts.runHandler(resolvedRun);
        } catch (err) {
          outcome = { status: 'error', error: String((err as Error)?.message ?? err) };
        }

        // Encrypt outgoing result the same way if we have a password —
        // keeps the result confidential between user and AI client even
        // though it transits the backend.
        if (password && outcome.status === 'done' && outcome.result !== undefined) {
          try {
            outcome = {
              status: 'done',
              result: await encryptPayload(outcome.result, password),
            };
          } catch (err) {
            outcome = { status: 'error', error: 'result encrypt failed: ' + String((err as Error).message ?? err) };
          }
        }
        try {
          await reportRunResult(opts.hashEdit, next.id, outcome);
        } catch (err) {
          // The run executed but we couldn't report it after retries (compute
          // already spent). Surface it visibly rather than swallowing — the
          // backend's stale-run reaper will mark it errored for the caller.
          opts.onPoll?.('error', `couldn't report result for run #${next.id}`);
          onError(err);
        }
      } catch (err) {
        // An aborted long-poll (our client timeout, or stop()) is not an error —
        // just reconnect, or exit if we're stopping. No onError/backoff.
        if ((err as Error)?.name === 'AbortError') {
          if (stopped) break;
          continue;
        }
        opts.onPoll?.('error', String((err as Error)?.message ?? err));
        onError(err);
        // Backoff before retrying so a flapping backend doesn't get
        // hammered. 5s is plenty for transient network hiccups.
        await new Promise((r) => setTimeout(r, 5_000));
      }
    }
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  })();

  return () => {
    stopped = true;
    activePoll?.abort(); // tear down an in-flight long-poll immediately
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  };
}
