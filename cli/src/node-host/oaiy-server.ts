/**
 * Delegation client for `oaiy-server` — the heavy "lifecycle" ops the Node host
 * doesn't do natively. When a flow needs a companion-managed AI service
 * (Ollama, llama.cpp, LTX2, …) started/ensured, the runtime calls
 * `ensure_service_ready(_by_port)`; here we forward that to oaiy-server's HTTP
 * API, which actually spawns/supervises the service. The flow then HTTP-calls
 * the service's own port directly.
 *
 *   OAIY_SERVER_URL    base URL of oaiy-server   [http://127.0.0.1:17972]
 *   OAIY_SERVER_TOKEN  bearer token (only needed for privileged routes)
 */
import { warnIfPlaintextRemoteToken } from '../insecure-token';

const BASE = (process.env.OAIY_SERVER_URL || 'http://127.0.0.1:17972').replace(/\/+$/, '');
const TOKEN = process.env.OAIY_SERVER_TOKEN || '';
warnIfPlaintextRemoteToken(BASE, !!TOKEN);

interface EnsureResult {
  success: boolean;
  already_running: boolean;
  port?: number;
  error?: string;
}

async function call(method: string, path: string, body?: unknown): Promise<Response> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (TOKEN) headers.Authorization = `Bearer ${TOKEN}`;
  // Bound the delegation call so a hung oaiy-server fails fast + legibly instead of
  // hanging on undici's opaque ~300s default — these routes return quickly (the
  // runtime polls separately while the service warms). Env-tunable.
  const ms = Number(process.env.OAIY_SERVER_TIMEOUT_MS) || 60000;
  try {
    return await fetch(BASE + path, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(ms),
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'TimeoutError') {
      throw new Error(`oaiy-server timed out after ${ms}ms (${method} ${path})`);
    }
    throw e;
  }
}

/** `ensure_service_ready_by_port`: start whatever service owns `port`. */
export async function ensureByPort(args: Record<string, unknown>): Promise<EnsureResult> {
  const port = Number(args.port ?? 0);
  try {
    const resp = await call('POST', '/api/services/ensure-by-port', { port });
    if (!resp.ok) {
      return { success: false, already_running: false, error: `oaiy-server HTTP ${resp.status}` };
    }
    // oaiy-server's registry returns { found, alreadyRunning, started, id, error };
    // adapt it to the runtime's EnsureServiceResult shape (success/port/...).
    // "started" counts as success — the runtime retries while the service warms.
    const r = (await resp.json()) as {
      found?: boolean;
      alreadyRunning?: boolean;
      started?: boolean;
      error?: string | null;
    };
    return {
      success: !!(r.found && (r.alreadyRunning || r.started)),
      already_running: !!r.alreadyRunning,
      port: r.found ? port : undefined,
      error: r.error ?? (r.found ? undefined : `no oaiy-server service owns port ${port}`),
    };
  } catch (e) {
    return {
      success: false,
      already_running: false,
      error: `oaiy-server unreachable at ${BASE}: ${e instanceof Error ? e.message : String(e)}`,
    };
  }
}

/** Look up a service's bound port from the registry snapshot. */
async function lookupPort(id: string): Promise<number | undefined> {
  try {
    const resp = await call('GET', '/api/services');
    if (!resp.ok) return undefined;
    const j = (await resp.json()) as { services?: Array<{ id?: string; port?: number; defaultPort?: number }> };
    const svc = (j.services ?? []).find((s) => s.id === id);
    return svc ? (svc.port ?? svc.defaultPort) : undefined;
  } catch {
    return undefined;
  }
}

/** `ensure_service_ready`: best-effort start of a service by id. */
export async function ensureService(args: Record<string, unknown>): Promise<EnsureResult> {
  const id = String(args.service ?? args.id ?? args.serviceId ?? '');
  if (!id) {
    return { success: false, already_running: false, error: 'ensure_service_ready: no service id' };
  }
  try {
    const resp = await call('POST', `/api/services/${encodeURIComponent(id)}/start`);
    if (resp.ok || resp.status === 204) {
      // /start has no body; resolve the port so the runtime can build the URL.
      // If the port can't be resolved, report failure with the real cause rather
      // than success-with-undefined-port (which the consumer silently discards).
      const port = await lookupPort(id);
      if (port === undefined) {
        return {
          success: false,
          already_running: false,
          error: `service "${id}" started but its port could not be resolved from oaiy-server`,
        };
      }
      return { success: true, already_running: false, port };
    }
    return { success: false, already_running: false, error: `oaiy-server HTTP ${resp.status}` };
  } catch (e) {
    return {
      success: false,
      already_running: false,
      error: `oaiy-server unreachable at ${BASE}: ${e instanceof Error ? e.message : String(e)}`,
    };
  }
}
