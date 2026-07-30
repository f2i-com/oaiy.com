/**
 * `oaiy worker` — drive an oaiy-api run queue from a server. The Node analogue of
 * the browser's backend dispatcher (ui/src/lib/backendDispatcher.ts): long-poll
 * `/runs/pending`, execute the flow headlessly, report the result back. This is
 * how a hosted oaiy-web (or an AI client POSTing runs) gets its flows executed on
 * a box with no browser open.
 */
import { runFlow } from './engine';
import type { WorkflowGraph } from 'oaiy-core';

interface QueuedRun {
  id: number;
  inputs?: Record<string, unknown>;
  reason?: string;
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

// Retries for reporting a finished run's result (mirrors the browser dispatcher's
// REPORT_RETRIES): the compute is already spent, so a transient blip at report
// time shouldn't silently discard it. Capped well under the backend's run TTL.
const REPORT_RETRIES = 4;
// Floor between empty long-poll responses, so a backend that doesn't actually hold
// the connection (POLL_TIMEOUT=0 / a buffering proxy) can't make the worker busy-spin.
const EMPTY_POLL_FLOOR_MS = 1000;
// Heartbeat cadence while a run executes. The worker is otherwise silent mid-run (it only
// bumps last_seen when polling /pending between runs), so without this the backend's
// liveness-gated stale-run reaper would mark a long run as timed-out. Matches the browser
// dispatcher's ~30s beat — well inside the backend's REAP_LIVENESS_SECONDS (default 120s).
const HEARTBEAT_MS = 30_000;

/** Parse the `→ <status>:` we encode in apiJson errors; 0 = network/no-status. */
const httpStatusOf = (msg: string): number => Number((msg.match(/→ (\d{3}):/) || [])[1]) || 0;

/**
 * Detect the browser's encrypted envelope.
 *
 * Must match what actually produces one — `encryptPayload` in
 * ui/src/lib/flowCrypto.ts emits `{ $enc: 'aes-gcm/pbkdf2', kdf, iter, salt, iv,
 * ct }`. This used to look for a `ciphertext` field, which nothing in the system
 * has ever emitted, so the guard never fired: an encrypted flow fell through to
 * extractGraph(), yielded no nodes, and the worker "ran" it and reported the run
 * as done. The `$enc` tag is the canonical marker; ct/iv is a shape fallback.
 */
const ENC_ENVELOPE_TAG = 'aes-gcm/pbkdf2';

function isEncrypted(x: unknown): boolean {
  if (!x || typeof x !== 'object') return false;
  const o = x as Record<string, unknown>;
  if (o.$enc === ENC_ENVELOPE_TAG) return true;
  return typeof o.ct === 'string' && typeof o.iv === 'string';
}

async function apiJson(
  method: string,
  base: string,
  path: string,
  body?: unknown,
): Promise<unknown> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const resp = await fetch(base + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (resp.status === 204) return undefined;
  const text = await resp.text();
  let parsed: unknown = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    /* leave as text */
  }
  if (!resp.ok) {
    const errField =
      parsed && typeof parsed === 'object' ? (parsed as { error?: unknown }).error : null;
    const msg =
      typeof errField === 'string' ? errField : text || `HTTP ${resp.status}`;
    throw new Error(`${method} ${path} → ${resp.status}: ${msg}`);
  }
  return parsed;
}

/** Pull the executable {nodes, edges} graph out of a stored flow_json snapshot. */
function extractGraph(flowJson: unknown): WorkflowGraph {
  const f = flowJson as Record<string, any>;
  if (Array.isArray(f?.nodes) && Array.isArray(f?.edges)) return { nodes: f.nodes, edges: f.edges };
  if (Array.isArray(f?.flows) && Array.isArray(f.flows[0]?.graph?.nodes)) {
    const g = f.flows[0].graph;
    return { nodes: g.nodes, edges: Array.isArray(g.edges) ? g.edges : [] };
  }
  if (Array.isArray(f?.graph?.nodes)) {
    return { nodes: f.graph.nodes, edges: Array.isArray(f.graph.edges) ? f.graph.edges : [] };
  }
  throw new Error('could not find a {nodes, edges} graph in the flow_json');
}

export interface WorkerOptions {
  backend: string;
  hashEdit: string;
  constants?: Record<string, string>;
  once?: boolean;
  log?: (msg: string) => void;
}

export async function runWorker(opts: WorkerOptions): Promise<{ fatal: boolean }> {
  const base = opts.backend.replace(/\/+$/, '');
  const log = opts.log ?? ((m: string) => process.stderr.write(`oaiy worker: ${m}\n`));

  // Fetch the flow once — the queue only carries {id, inputs}, not the graph.
  const flow = (await apiJson('GET', base, `/api/flows/${encodeURIComponent(opts.hashEdit)}`)) as {
    flow_json?: unknown;
    title?: string;
  };
  if (isEncrypted(flow?.flow_json)) {
    throw new Error('flow is password-encrypted; the CLI worker does not support encrypted flows yet');
  }
  const graph = extractGraph(flow.flow_json);
  const flowName = flow.title || 'queued-run';
  log(`listening on ${base} · flow ${opts.hashEdit} (${graph.nodes.length} nodes)`);

  let stopped = false;
  // Graceful/force two-stage: the first signal lets the in-flight run finish +
  // report (then the loop exits); a second forces exit, so a long run isn't an
  // un-interruptible wait. Logs so the operator knows the signal registered.
  const onSignal = (sig: string) => {
    if (stopped) {
      log(`second ${sig} — forcing exit`);
      process.exit(130);
    }
    stopped = true;
    log(`${sig} received — stopping after the current run (Ctrl-C again to force-quit)`);
  };
  process.on('SIGINT', () => onSignal('SIGINT'));
  process.on('SIGTERM', () => onSignal('SIGTERM'));

  // Set when the loop bails on a permanent client error (bad hash / auth / gone)
  // so the caller can exit non-zero — a supervisor must see the worker died.
  let fatal = false;
  while (!stopped) {
    let next: QueuedRun | null = null;
    const polledAt = Date.now();
    try {
      next = (await apiJson(
        'GET',
        base,
        `/api/flows/${encodeURIComponent(opts.hashEdit)}/runs/pending`,
      )) as QueuedRun | null;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // Permanent client errors (bad hash, auth, gone) are fatal — don't spin.
      // Transient throttles/timeouts (408/429) and any 5xx back off + retry, so
      // a fronting proxy's rate limit doesn't kill the worker.
      const status = httpStatusOf(msg);
      const FATAL = new Set([400, 401, 403, 404, 410]);
      if (FATAL.has(status)) {
        log(`fatal: ${msg}`);
        fatal = true;
        break;
      }
      const backoff = status === 429 ? 30000 : 5000;
      log(`poll error: ${msg} (retry in ${backoff / 1000}s)`);
      await sleep(backoff);
      continue;
    }
    if (!next) {
      if (opts.once) break;
      // The backend is expected to long-poll (~20s). If pending came back empty far
      // faster, it isn't holding the connection — floor the re-poll so the worker
      // can't busy-spin / flood the endpoint (the error path already backs off).
      // Costs nothing when the server long-polls correctly (a ~20s return >> floor).
      const elapsed = Date.now() - polledAt;
      if (elapsed < EMPTY_POLL_FLOOR_MS) await sleep(EMPTY_POLL_FLOOR_MS - elapsed);
      continue;
    }

    log(`run #${next.id} claimed (${next.reason || 'no reason'})`);
    let body: Record<string, unknown>;
    if (isEncrypted(next.inputs)) {
      body = { status: 'error', error: 'encrypted inputs not supported by the CLI worker', hash: opts.hashEdit };
    } else {
      // Keep last_seen fresh while the run executes so the backend's liveness-gated reaper
      // doesn't time out a still-running job (the worker is otherwise silent mid-run).
      const heartbeat = setInterval(() => {
        void apiJson('POST', base, `/api/flows/${encodeURIComponent(opts.hashEdit)}/heartbeat`, {}).catch(
          () => {},
        );
      }, HEARTBEAT_MS);
      try {
        const res = await runFlow(graph, {
          inputs: next.inputs || {},
          constants: opts.constants,
          flowName,
          // Untrusted queue inputs: deny local-network access by default on the engine's
          // policyFetch path (the Node host independently SSRF-guards direct http_request).
          // Operators with trusted local-service flows opt in with OAIY_HTTP_ALLOW_LOCAL=1.
          allowLocalNetwork: process.env.OAIY_HTTP_ALLOW_LOCAL === '1',
        });
        body = res.success
          ? { status: 'done', result: res.results, hash: opts.hashEdit }
          : { status: 'error', error: res.error || 'run failed', hash: opts.hashEdit };
      } catch (e) {
        body = { status: 'error', error: e instanceof Error ? e.message : String(e), hash: opts.hashEdit };
      } finally {
        clearInterval(heartbeat);
      }
    }

    // Report with bounded retry/backoff: the run already spent its compute, so a
    // transient network blip or a backend restart at report time shouldn't drop the
    // result (the backend's stale-run reaper would then mark a successful run errored).
    // Retry network/5xx; bail immediately on a permanent 4xx (wrong hash / not running).
    for (let attempt = 0; ; attempt++) {
      try {
        await apiJson('POST', base, `/api/runs/${next.id}/result`, body);
        log(`run #${next.id} → ${body.status}`);
        break;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        const status = httpStatusOf(msg);
        const retryable = status === 0 || status >= 500;
        if (stopped || attempt >= REPORT_RETRIES - 1 || !retryable) {
          log(`could not report run #${next.id}: ${msg}`);
          break;
        }
        await sleep(Math.min(8000, 500 * 2 ** attempt));
      }
    }

    if (opts.once) break;
  }
  log('stopped');
  return { fatal };
}
