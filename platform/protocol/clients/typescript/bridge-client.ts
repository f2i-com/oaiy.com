/**
 * OAIY Bridge Protocol v1 — reference client.
 *
 * The consumer half of `protocol/README.md`: what a product imports to drive an
 * OAIY runtime. FormLogic is the first consumer, but nothing here is
 * FormLogic-specific — that is the whole point of the seam. It lives in the OAIY
 * repo, next to the schemas it implements, so a consumer vendors the client
 * rather than re-deriving the protocol and drifting from it.
 *
 * Zero dependencies. Uses the global `fetch`, so it runs unchanged in a browser
 * (oaiy.com's own web app), in Node 18+ (a co-located worker, FormLogic Cloud's
 * relay leg), and in a Tauri webview.
 *
 * # Design
 *
 * Every method maps one protocol operation and surfaces its outcome as the
 * protocol defines it — not as a uniform 200-with-a-field. A lost claim, a
 * loop-guard refusal and a duplicate are distinct results a caller must branch
 * on, so they are distinct return shapes or distinct thrown errors, never
 * flattened. The single most important line in the file is the health assertion:
 * a 200 on `127.0.0.1:17972` is not proof you reached OAIY, and every other
 * method is worthless if the first one trusted a squatter.
 */

// --- wire types (mirrors protocol/v1/*.schema.json) -----------------------

export type RunStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'timed_out'
  | 'cancelled';

export type BridgeErrorCode =
  | 'invalid_request'
  | 'invalid_flow'
  | 'flow_not_found'
  | 'capability_denied'
  | 'capability_unavailable'
  | 'connection_missing'
  | 'node_failed'
  | 'timeout'
  | 'cancelled'
  | 'runtime_unavailable'
  | 'internal';

/** Who is asking. Only `product` is read by OAIY; the rest is opaque to it. */
export interface Caller {
  product: string;
  tenantId?: string;
  scopeId?: string;
  label?: string;
}

export interface Health {
  status: 'ok' | 'degraded';
  product: string;
  protocol: string;
  version: string;
  runtime?: string;
  deviceId?: string;
  detail?: string;
}

export interface RunError {
  code: BridgeErrorCode;
  message: string;
  detail?: string;
  nodeId?: string;
  capability?: string;
  retryable?: boolean;
}

export interface RunRecord {
  runId: string;
  status: RunStatus;
  output?: unknown;
  error?: RunError;
  /** True when this reservation matched an existing run and nothing new ran. */
  idempotent?: boolean;
  runtime?: string;
  claimedBy?: string;
  correlationId?: string;
  idempotencyKey?: string;
  reservedAt?: string;
  startedAt?: string;
  finishedAt?: string;
  [k: string]: unknown;
}

export interface Capability {
  id: string;
  available: boolean;
  reason?: string;
  detail?: string;
  connections?: string[];
  pluginId?: string;
  title?: string;
}

export interface CapabilityManifest {
  protocol: string;
  runtime: string;
  deviceId?: string;
  capabilities: Capability[];
}

export interface FlowDescriptor {
  flowId: string;
  name: string;
  description?: string;
  enabled?: boolean;
  revision?: number;
  inputs?: Array<{ name: string; required?: boolean; type?: string }>;
  capabilities?: string[];
  runtimes?: string[];
}

export interface ReceivedEvent {
  seq: number;
  receivedAtMs: number;
  envelope: Record<string, unknown>;
  outcomes: string[];
}

// --- errors ---------------------------------------------------------------

/**
 * A typed failure from the bridge. Carries the protocol's `code` so a caller
 * branches on the closed taxonomy rather than substring-matching a message —
 * which is the whole reason the taxonomy is closed.
 */
export class BridgeError extends Error {
  readonly code: BridgeErrorCode | 'unreachable' | 'wrong_product' | 'protocol_mismatch';
  readonly httpStatus?: number;
  readonly detail?: string;

  constructor(
    code: BridgeError['code'],
    message: string,
    opts: { httpStatus?: number; detail?: string; cause?: unknown } = {},
  ) {
    super(message, opts.cause !== undefined ? { cause: opts.cause } : undefined);
    this.name = 'BridgeError';
    this.code = code;
    this.httpStatus = opts.httpStatus;
    this.detail = opts.detail;
  }
}

// --- client ---------------------------------------------------------------

export interface BridgeClientOptions {
  /** Defaults to the standard loopback endpoint. */
  baseUrl?: string;
  /** This consumer's identity. `product` is required by the protocol. */
  caller: Caller;
  /**
   * Bearer token for a non-browser client. A browser sends an `Origin` the
   * runtime trusts (oaiy.com / its own webview); a Node consumer has no Origin
   * and needs this to reach exec routes.
   */
  token?: string;
  /** Inject a fetch (tests, a proxied transport). Defaults to the global. */
  fetch?: typeof fetch;
  /** Per-request timeout for everything except a deliberate long-poll. */
  requestTimeoutMs?: number;
}

export interface RunFlowOptions {
  flowId?: string;
  /** Inline graph, mutually exclusive with `flowId`. */
  graph?: Record<string, unknown>;
  input?: Record<string, unknown>;
  capabilities?: string[];
  /**
   * `sync` waits for the terminal result (the runtime long-polls server-side).
   * `async` returns as soon as the run is reserved. `queued` reserves without
   * starting, for a separate claimer.
   */
  mode?: 'sync' | 'async' | 'queued';
  timeoutMs?: number;
  correlationId?: string;
  /**
   * REQUIRED and load-bearing. It must be STABLE for the same logical event —
   * `binding:<id>:<eventId>`, never a fresh uuid — or at-least-once delivery
   * runs the flow more than once. See the protocol's idempotency note.
   */
  idempotencyKey: string;
  /** Overrides the client's default caller for this one call. */
  caller?: Caller;
}

const DEFAULT_BASE = 'http://127.0.0.1:17972';
const PROTOCOL = 'oaiy-bridge/1';
const PRODUCT = 'oaiy-desktop';
const DEFAULT_TIMEOUT_MS = 15_000;

export class OaiyBridgeClient {
  private readonly base: string;
  private readonly caller: Caller;
  private readonly token?: string;
  private readonly fetchImpl: typeof fetch;
  private readonly requestTimeoutMs: number;

  constructor(opts: BridgeClientOptions) {
    if (!opts.caller || !opts.caller.product) {
      // Failing here rather than at the first request means a misconfigured
      // consumer is caught at construction, not mid-flow.
      throw new BridgeError('invalid_request', 'a caller.product is required');
    }
    this.base = (opts.baseUrl ?? DEFAULT_BASE).replace(/\/+$/, '');
    this.caller = opts.caller;
    this.token = opts.token;
    const f = opts.fetch ?? globalThis.fetch;
    if (typeof f !== 'function') {
      throw new BridgeError(
        'runtime_unavailable',
        'no fetch available; pass options.fetch on a runtime without a global fetch',
      );
    }
    this.fetchImpl = f;
    this.requestTimeoutMs = opts.requestTimeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  // --- handshake ----------------------------------------------------------

  /**
   * The handshake. Asserts this is actually OAIY and speaks our protocol major.
   *
   * A 200 on a fixed loopback port is not proof — a different app has answered
   * `/api/health` with a compatible shape before. So `product` is matched
   * exactly and a foreign protocol major is refused rather than best-effort
   * parsed. Every other method assumes this passed.
   */
  async health(): Promise<Health> {
    const body = await this.get<Health>('/api/health', { open: true });
    if (body.product !== PRODUCT) {
      throw new BridgeError(
        'wrong_product',
        `expected ${PRODUCT} at ${this.base}, but it identifies as ${JSON.stringify(
          body.product,
        )}`,
      );
    }
    if (majorOf(body.protocol) !== majorOf(PROTOCOL)) {
      throw new BridgeError(
        'protocol_mismatch',
        `this client speaks ${PROTOCOL}; the runtime speaks ${body.protocol}`,
      );
    }
    return body;
  }

  /** A soft check: resolves false instead of throwing when OAIY is absent. */
  async isAvailable(): Promise<boolean> {
    try {
      await this.health();
      return true;
    } catch {
      return false;
    }
  }

  // --- discovery ----------------------------------------------------------

  async capabilities(): Promise<CapabilityManifest> {
    return this.get<CapabilityManifest>('/api/bridge/capabilities', { open: true });
  }

  /** Does the runtime currently offer `capabilityId`, and is it available? */
  async hasCapability(capabilityId: string): Promise<boolean> {
    const m = await this.capabilities();
    return m.capabilities.some((c) => c.id === capabilityId && c.available);
  }

  async listFlows(): Promise<FlowDescriptor[]> {
    const r = await this.get<{ flows: FlowDescriptor[] }>('/api/bridge/flows');
    return r.flows ?? [];
  }

  // --- flow execution -----------------------------------------------------

  /**
   * Invoke a flow. This is the `oaiy.runFlow` operation from the architecture
   * notes: hand a flowId + input, get back a run outcome.
   *
   * In `sync` mode this resolves to a TERMINAL record (or throws on a failed
   * run). In `async`/`queued` it resolves as soon as the run is reserved —
   * poll `getRun` or use `runToCompletion` to wait client-side.
   *
   * A duplicate (same idempotencyKey) is NOT an error: it returns the existing
   * run with `idempotent: true`, so a retried trigger is a no-op that still
   * hands back the original runId.
   */
  async runFlow(opts: RunFlowOptions): Promise<RunRecord> {
    if (!opts.flowId && !opts.graph) {
      throw new BridgeError('invalid_request', 'runFlow needs a flowId or an inline graph');
    }
    if (opts.flowId && opts.graph) {
      throw new BridgeError('invalid_request', 'flowId and graph are mutually exclusive');
    }
    if (!opts.idempotencyKey || !opts.idempotencyKey.trim()) {
      throw new BridgeError(
        'invalid_request',
        'idempotencyKey is required and must be stable for the logical event',
      );
    }
    const body = {
      protocol: PROTOCOL,
      caller: opts.caller ?? this.caller,
      flowId: opts.flowId,
      graph: opts.graph,
      input: opts.input,
      capabilities: opts.capabilities,
      mode: opts.mode ?? 'async',
      timeoutMs: opts.timeoutMs,
      correlationId: opts.correlationId ?? opts.idempotencyKey,
      idempotencyKey: opts.idempotencyKey,
    };
    // A sync run legitimately takes as long as the flow's own budget, so its
    // request timeout must exceed it — otherwise the client gives up on a run
    // the server is still faithfully executing.
    const timeout =
      opts.mode === 'sync'
        ? (opts.timeoutMs ?? 30_000) + this.requestTimeoutMs
        : this.requestTimeoutMs;
    return this.send<RunRecord>('POST', '/api/bridge/runs', body, { timeoutMs: timeout });
  }

  async getRun(runId: string): Promise<RunRecord> {
    return this.get<RunRecord>(`/api/bridge/runs/${encodeURIComponent(runId)}`);
  }

  /**
   * Poll an async run to a terminal state.
   *
   * Polling, not a socket, because a dropped connection must never lose the
   * outcome — the run is always durably readable. Throws `timeout` (client-side)
   * if the run does not finish within `waitMs`; the run itself keeps going, and
   * the runId stays valid to poll later.
   */
  async runToCompletion(
    runId: string,
    opts: { waitMs?: number; pollMs?: number } = {},
  ): Promise<RunRecord> {
    const waitMs = opts.waitMs ?? 120_000;
    const pollMs = opts.pollMs ?? 500;
    const deadline = nowMs() + waitMs;
    for (;;) {
      const run = await this.getRun(runId);
      if (isTerminal(run.status)) return run;
      if (nowMs() >= deadline) {
        throw new BridgeError(
          'timeout',
          `run ${runId} did not finish within ${waitMs}ms (it is still ${run.status}; poll it later)`,
        );
      }
      await sleep(pollMs);
    }
  }

  /**
   * Run a flow and wait for its result, throwing if the run FAILED.
   *
   * The convenience most callers actually want: `runFlow(sync)` then unwrap.
   * A failed run is thrown as a BridgeError carrying the flow's own error code —
   * "succeeded with wrong output" is impossible because a failed run never
   * reaches the success path here.
   */
  async runFlowForResult(opts: RunFlowOptions): Promise<unknown> {
    let run = await this.runFlow({ ...opts, mode: 'sync' });
    if (!isTerminal(run.status)) {
      // The server answered 202 (still running past its budget). Fall back to
      // client-side polling rather than pretending it finished.
      run = await this.runToCompletion(run.runId, {
        waitMs: (opts.timeoutMs ?? 30_000) + 10_000,
      });
    }
    if (run.status === 'succeeded') return run.output;
    throw new BridgeError(
      run.error?.code ?? 'node_failed',
      run.error?.message ?? `flow run ${run.status}`,
      { detail: run.error?.detail },
    );
  }

  /** Request cancellation. A running run stops when the runtime notices. */
  async cancelRun(runId: string): Promise<{ status: RunStatus; cancelRequested?: boolean }> {
    return this.send('POST', `/api/bridge/runs/${encodeURIComponent(runId)}/cancel`, undefined);
  }

  // --- connectors ---------------------------------------------------------

  /**
   * Call a plugin connector command, gated against the plugin's manifest before
   * it is forwarded. `idempotencyKey` is required for commands the plugin marks
   * journalled (anything with a side effect a retry must not repeat).
   */
  async connector(
    connectorId: string,
    command: string,
    opts: { payload?: unknown; idempotencyKey?: string } = {},
  ): Promise<unknown> {
    const r = await this.send<{ ok: boolean; result?: unknown }>(
      'POST',
      `/api/bridge/connectors/${encodeURIComponent(connectorId)}/request`,
      { command, payload: opts.payload, idempotencyKey: opts.idempotencyKey },
    );
    return r.result;
  }

  // --- events -------------------------------------------------------------

  /** One page of events after `since` (a sequence number, not a timestamp). */
  async pollEvents(
    since = 0,
    limit = 100,
  ): Promise<{ events: ReceivedEvent[]; next: number }> {
    return this.get(`/api/bridge/events?since=${since}&limit=${limit}`);
  }

  /**
   * Stream events as an async iterable, polling under the hood.
   *
   * `for await (const ev of client.streamEvents()) …`. Sequence-based, so no
   * event is skipped and none is replayed across polls. Stop by breaking the
   * loop or aborting `opts.signal`.
   */
  async *streamEvents(
    opts: { since?: number; pollMs?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<ReceivedEvent> {
    let since = opts.since ?? 0;
    const pollMs = opts.pollMs ?? 1000;
    while (!opts.signal?.aborted) {
      const { events, next } = await this.pollEvents(since, 200);
      for (const ev of events) yield ev;
      since = next;
      if (!events.length) await sleep(pollMs, opts.signal);
    }
  }

  // --- transport ----------------------------------------------------------

  private async get<T>(path: string, o: { open?: boolean } = {}): Promise<T> {
    return this.request<T>('GET', path, undefined, o);
  }

  private async send<T>(
    method: string,
    path: string,
    body: unknown,
    o: { timeoutMs?: number } = {},
  ): Promise<T> {
    return this.request<T>(method, path, body, o);
  }

  private async request<T>(
    method: string,
    path: string,
    body: unknown,
    o: { open?: boolean; timeoutMs?: number } = {},
  ): Promise<T> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    // The bearer is sent on every request, not just exec ones: a restricted
    // READ (events, runs) needs it too from a non-browser client. Harmless on
    // an open route.
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), o.timeoutMs ?? this.requestTimeoutMs);
    let resp: Response;
    try {
      resp = await this.fetchImpl(`${this.base}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        // The API sets no Access-Control-Allow-Credentials, so a credentialed
        // request would fail the CORS check outright in a browser.
        credentials: 'omit',
        signal: controller.signal,
      });
    } catch (e) {
      throw new BridgeError(
        'unreachable',
        `could not reach OAIY at ${this.base}${path}: ${errText(e)}`,
        { cause: e },
      );
    } finally {
      clearTimeout(timer);
    }

    const text = await resp.text();
    const json = text ? safeJson(text) : undefined;

    if (!resp.ok) {
      const err = (json as { error?: { code?: string; message?: string; detail?: string } })
        ?.error;
      throw new BridgeError(
        (err?.code as BridgeErrorCode) ?? httpToCode(resp.status),
        err?.message ?? `${method} ${path} → HTTP ${resp.status}`,
        { httpStatus: resp.status, detail: err?.detail },
      );
    }
    return json as T;
  }
}

// --- helpers --------------------------------------------------------------

export function isTerminal(status: RunStatus): boolean {
  return (
    status === 'succeeded' ||
    status === 'failed' ||
    status === 'timed_out' ||
    status === 'cancelled'
  );
}

function majorOf(protocol: string): string {
  // `oaiy-bridge/1` → `oaiy-bridge/1`; `oaiy-bridge/1.2` → `oaiy-bridge/1`.
  const [name, ver = ''] = protocol.split('/');
  return `${name}/${ver.split('.')[0]}`;
}

function httpToCode(status: number): BridgeErrorCode {
  if (status === 403) return 'capability_denied';
  if (status === 404) return 'flow_not_found';
  if (status === 503 || status === 502) return 'capability_unavailable';
  if (status === 504) return 'timeout';
  if (status >= 500) return 'internal';
  return 'invalid_request';
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function errText(e: unknown): string {
  if (e && typeof e === 'object' && 'name' in e && (e as { name?: string }).name === 'AbortError') {
    return 'request timed out';
  }
  return e instanceof Error ? e.message : String(e);
}

function nowMs(): number {
  return Date.now();
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
}
