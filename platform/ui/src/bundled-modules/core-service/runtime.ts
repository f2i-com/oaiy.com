/**
 * Core Service Module Runtime
 *
 * The compiler emits `await Service.call(...)`. The runtime renders the
 * request body + headers (substituting {{input}} / {{inputRaw}} / {{apiKey}}),
 * issues an HTTP request (preferring `ctx.tauri.invoke('http_request')` so
 * desktop dodges CORS and web reuses the shim's `fetch`), parses the
 * response, and extracts a value via dot/bracket path.
 *
 * Mirrors the templating helpers in core-ai's custom-provider branch so
 * the two stay easy to evolve together; duplication is intentional — each
 * module is meant to be a self-contained unit.
 */

import type { RuntimeModule, RuntimeContext, RuntimeMethod } from 'oaiy-core/src/module-types';
import type { CustomService } from './examples';
import { desktopAccess, desktopCredential, runContract } from './contractRuntime';

// Per-job method factory: methods close over THIS job's ctx (no module-level singleton).
function createServiceMethods(ctx: RuntimeContext): Record<string, RuntimeMethod> {

function jsonEscape(value: unknown): string {
  return JSON.stringify(value ?? '');
}

function stringify(value: unknown): string {
  if (value === null || value === undefined) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function renderTemplate(template: string, vars: Record<string, unknown>): string {
  if (!template) return template;
  return template.replace(/\{\{(\w+)\}\}/g, (match, name: string) => {
    const isRaw = name.endsWith('Raw');
    const key = isRaw ? name.slice(0, -3) : name;
    if (!(key in vars)) return match;
    const value = vars[key];
    // API keys are ALWAYS embedded inside a quoted string (a header value like
    // "Bearer {{apiKey}}"), never as a bare JSON slot — so treat `apiKey` as
    // raw whether or not the author used the `Raw` suffix. Without this,
    // `{{apiKey}}` JSON-escapes (adds quotes) and corrupts the surrounding
    // string into `"Bearer "<key>""` → invalid JSON. This also repairs
    // services saved before the presets were corrected to use {{apiKeyRaw}}.
    const raw = isRaw || key === 'apiKey';
    return raw ? stringify(value) : jsonEscape(value);
  });
}

function extractJsonPath(root: unknown, path: string): unknown {
  if (!path) return root;
  const parts = path.split(/[.\[\]]+/).filter(Boolean);
  let cur: unknown = root;
  for (const part of parts) {
    if (cur === null || cur === undefined) return undefined;
    const idx = /^\d+$/.test(part) ? Number(part) : part;
    cur = (cur as Record<string | number, unknown>)[idx as never];
  }
  return cur;
}

function resolveApiKey(apiKeyConstant: string): string {
  if (!apiKeyConstant) return '';
  // Literal-key heuristic — mirrors core-ai resolveApiKey so pasting a
  // raw key into apiKeyConstant still works without going through the
  // project constants table.
  if (
    apiKeyConstant.startsWith('sk-') ||
    apiKeyConstant.startsWith('anthropic-') ||
    apiKeyConstant.startsWith('gsk_') ||
    apiKeyConstant.length > 40
  ) {
    return apiKeyConstant;
  }
  const getConstant = (ctx as unknown as { getConstant?: (n: string) => string | undefined })
    ?.getConstant;
  if (getConstant) {
    const k = getConstant(apiKeyConstant);
    if (k) return k;
  }
  const getSetting = (ctx as unknown as { getModuleSetting?: (n: string) => unknown })
    ?.getModuleSetting;
  if (getSetting) {
    const k = getSetting(apiKeyConstant);
    if (k) return String(k);
  }
  // Nothing resolved. If the value looks like an ENV-VAR-style constant NAME
  // (ALL_CAPS_WITH_UNDERSCORES, e.g. OPENAI_API_KEY) it is almost certainly a
  // reference to a project constant that is empty / not yet loaded — fail
  // loudly instead of shipping the literal name as `Authorization: Bearer
  // OPENAI_API_KEY` (a confusing upstream 401 that also discloses the constant
  // name to the endpoint). A real key pasted directly never has this shape, so
  // the directly-pasted-literal-key path below still works.
  if (/^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$/.test(apiKeyConstant)) {
    throw new Error(
      `[Service] API key constant "${apiKeyConstant}" is empty or not loaded yet — set it in Settings → API Keys.`,
    );
  }
  return apiKeyConstant; // last-resort: treat as a directly-pasted literal key
}

interface HttpResult {
  status: number;
  ok?: boolean;
  headers?: Record<string, string>;
  body: string;
  statusText?: string;
}

function sanitizeUrlForLog(rawUrl: string): string {
  try {
    const u = new URL(rawUrl);
    if (u.username || u.password) {
      u.username = '';
      u.password = '';
    }
    for (const key of Array.from(u.searchParams.keys())) {
      if (/(key|token|secret|password|auth|credential)/i.test(key)) {
        u.searchParams.set(key, '[redacted]');
      }
    }
    return u.toString();
  } catch {
    return rawUrl.replace(/([?&][^=&#\s]*?(?:key|token|secret|password|auth|credential)[^=&#\s]*=)[^&#\s]*/gi, '$1[redacted]');
  }
}

async function httpCall(
  url: string,
  method: string,
  headers: Record<string, string>,
  body: string | undefined,
  nodeId?: string,
): Promise<HttpResult> {
  // Always route through the runtime's policy-enforced fetch (`ctx.secureFetch`) so
  // `isLocalNetworkUrl` / `localNetworkWhitelist` / `onLocalNetworkPermission` are
  // enforced consistently (and desktop still proxies through native `http_request`
  // after policy approval).
  const fetcher = ctx?.secureFetch ?? ctx?.fetch;
  const resp = fetcher
    ? await fetcher(url, {
        method,
        headers,
        body,
        signal: ctx?.abortSignal,
        ...(nodeId ? { nodeId } : {}),
        purpose: 'Service HTTP call',
      })
    : await fetch(url, { method, headers, body, signal: ctx?.abortSignal });
  const text = await resp.text();
  const respHeaders: Record<string, string> = {};
  if (resp.headers && typeof resp.headers.forEach === 'function') {
    resp.headers.forEach((v, k) => { respHeaders[k] = v; });
  }
  return { status: resp.status, ok: resp.ok, headers: respHeaders, body: text, statusText: resp.statusText };
}

async function call(
  varsOrInput: unknown,
  endpoint: string,
  method: string,
  headersTemplate: string,
  bodyTemplate: string,
  responseType: string,
  responsePath: string,
  apiKeyConstant: string,
  nodeId: string,
): Promise<unknown> {
  if (!endpoint) {
    throw new Error(
      '[Service] Endpoint URL is empty — pick a Service Preset or fill the Endpoint field.',
    );
  }
  const apiKey = resolveApiKey(apiKeyConstant);
  // Accept either a vars dict {prompt, image, …} (services with declared
  // inputs[]) or a single legacy value (everything else, wrapped to
  // {input: value}). Compiler always emits dict shape now; this branch
  // catches edge cases and stale saved-flow shapes defensively.
  let vars: Record<string, unknown>;
  if (varsOrInput && typeof varsOrInput === 'object' && !Array.isArray(varsOrInput)) {
    vars = { ...(varsOrInput as Record<string, unknown>) };
  } else {
    vars = { input: varsOrInput };
  }
  vars.apiKey = apiKey;
  const upMethod = (method || 'POST').toUpperCase();

  // Headers
  let headers: Record<string, string> = {};
  const rawHeaders = (headersTemplate || '').trim();
  if (rawHeaders) {
    const rendered = renderTemplate(rawHeaders, vars);
    try {
      const parsed = JSON.parse(rendered);
      if (parsed && typeof parsed === 'object') {
        headers = parsed as Record<string, string>;
      }
    } catch (e) {
      const rawErr = e instanceof Error ? e.message : String(e);
      const safeErr = apiKey ? rawErr.split(apiKey).join('[redacted]') : rawErr;
      throw new Error(
        `[Service] Headers template did not render to valid JSON: ${safeErr}\nTemplate: ${rawHeaders.slice(0, 200)}`,
      );
    }
  }
  if (!Object.keys(headers).some((k) => k.toLowerCase() === 'content-type')) {
    headers['Content-Type'] = 'application/json';
  }
  if (apiKey && !Object.keys(headers).some((k) => k.toLowerCase() === 'authorization')) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }
  // OAIY Desktop's own engine routes take its credential, as a contract service's call does (contractRuntime): the
  // engine's language model picked here (a service, not a contract) was refused "authentication required".
  // Never attach the desktop credential for a packaged/untrusted run lacking `network:local`.
  const canUseDesktopToken = !ctx.currentPackageId || Boolean(ctx.hasPermission?.('network:local'));
  if (canUseDesktopToken && !Object.keys(headers).some((k) => k.toLowerCase() === 'authorization')) {
    const credential = desktopCredential(endpoint, desktopAccess());
    if (credential) headers['Authorization'] = `Bearer ${credential}`;
  }

  // Body
  let body: string | undefined;
  if (upMethod !== 'GET' && upMethod !== 'HEAD') {
    const rawBody = (bodyTemplate || '').trim();
    if (rawBody) {
      body = renderTemplate(rawBody, vars);
    } else {
      // No body template → fall back to the legacy `input` slot (the only
      // var that's set in both legacy single-value and multi-var shapes).
      const legacy = vars.input;
      if (legacy !== undefined && legacy !== null) {
        body = typeof legacy === 'string' ? legacy : JSON.stringify(legacy);
      }
    }
  }

  // If this endpoint targets an OAIY Desktop-managed local port, ask OAIY Desktop
  // to start the matching service (if stopped) before firing the request — so
  // picking a stopped companion service in a flow and running it "just works"
  // (Phase 3.5). Same cross-host handshake core-ai uses: on web ctx.tauri is the
  // shim, whose ensure_service_ready_by_port proxies to OAIY Desktop; on
  // desktop/CLI it's the real command. Fire-and-forget — any failure (no
  // companion, or a non-companion local server) falls through to the direct
  // request. A heavy service still loading its model may not answer this first
  // request; it has been started, so a re-run succeeds once it's warm.
  const ensureInvoke = ctx?.tauri?.invoke as
    | (<T>(cmd: string, args?: unknown) => Promise<T>)
    | undefined;
  let parsedEndpoint: URL | null = null;
  try {
    parsedEndpoint = new URL(endpoint);
  } catch {
    parsedEndpoint = null;
  }
  const isLoopbackHost =
    parsedEndpoint !== null &&
    (parsedEndpoint.hostname === '127.0.0.1' || parsedEndpoint.hostname === 'localhost');
  if (ensureInvoke && isLoopbackHost && parsedEndpoint && parsedEndpoint.port) {
    const port = parseInt(parsedEndpoint.port, 10);
    if (Number.isFinite(port) && port > 0) {
      try {
        const r = await ensureInvoke<{ success?: boolean; port?: number }>(
          'ensure_service_ready_by_port',
          { port },
        );
        if (r?.success && r.port && r.port !== port) {
          parsedEndpoint.port = String(r.port);
          endpoint = parsedEndpoint.toString();
        }
      } catch {
        /* no companion / not reachable — fall through to the direct request */
      }
    }
  }

  ctx?.log?.('info', `[Service] ${upMethod} ${sanitizeUrlForLog(endpoint)} (node ${nodeId})`);
  const res = await httpCall(endpoint, upMethod, headers, body, nodeId);
  const ok = res.ok ?? (res.status >= 200 && res.status < 300);
  if (!ok) {
    throw new Error(`[Service] HTTP ${res.status}: ${(res.body || '').slice(0, 400)}`);
  }

  if ((responseType || 'json') === 'text') {
    return responsePath ? extractJsonPath(res.body, responsePath) : res.body;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(res.body);
  } catch {
    // Tolerate non-JSON responses when no path was requested — return
    // the raw text so users see SOMETHING rather than an exception.
    if (!responsePath) return res.body;
    throw new Error(
      `[Service] Expected JSON but response was not parseable: ${res.body.slice(0, 200)}`,
    );
  }
  return responsePath ? extractJsonPath(parsed, responsePath) : parsed;
}

/**
 * A contract service (see contract.ts): an OAIY engine model, OAIY Voice, or
 * a rig that declares what it makes. What it gives back: a picture as a
 * data: URL, audio / video / a 3D model as `{audio|video|model, path}`, text
 * as text, anything else as the answer at its response path.
 */
async function callContract(
  contract: CustomService | string,
  vars: Record<string, unknown>,
  nodeId: string,
): Promise<unknown> {
  ctx.onNodeStatus?.(nodeId, 'running');
  try {
    const r = await runContract(ctx, contract, vars, { nodeId, label: typeof contract === 'object' ? contract.name : 'Service' });
    ctx.onNodeStatus?.(nodeId, 'completed');
    switch (r.output) {
      case 'image':
        ctx.onImage?.(nodeId, r.image!);
        return r.image;
      case 'audio':
        return { audio: r.url, path: r.path };
      case 'video':
        return { video: r.url, path: r.path };
      case 'model3d':
        return { model: r.url, path: r.path };
      case 'text':
        return r.text;
      default:
        return r.value;
    }
  } catch (e) {
    ctx.onNodeStatus?.(nodeId, 'error');
    throw e;
  }
}

  return {
    call,
    callContract: callContract as RuntimeMethod,
  };
}

const CoreServiceRuntime: RuntimeModule = {
  name: 'Service',
  createMethods: createServiceMethods,
  methods: {},
};

export default CoreServiceRuntime;
