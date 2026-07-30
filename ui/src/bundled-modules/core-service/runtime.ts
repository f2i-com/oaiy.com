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

async function httpCall(
  url: string,
  method: string,
  headers: Record<string, string>,
  body: string | undefined,
): Promise<HttpResult> {
  // Prefer the shared HTTP proxy — desktop reaches it via Tauri reqwest
  // (dodges CORS); the web shim routes it to `fetch` directly. Same
  // result shape, same call site.
  const invoke = ctx?.tauri?.invoke as
    | (<T>(cmd: string, args?: unknown) => Promise<T>)
    | undefined;
  if (invoke) {
    return await invoke<HttpResult>('http_request', { url, method, headers, body });
  }
  const resp = await fetch(url, { method, headers, body });
  const text = await resp.text();
  return { status: resp.status, ok: resp.ok, body: text };
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
      throw new Error(
        `[Service] Headers template did not render to valid JSON: ${(e as Error).message}\nRendered: ${rendered.slice(0, 200)}`,
      );
    }
  }
  if (!Object.keys(headers).some((k) => k.toLowerCase() === 'content-type')) {
    headers['Content-Type'] = 'application/json';
  }
  if (apiKey && !Object.keys(headers).some((k) => k.toLowerCase() === 'authorization')) {
    headers['Authorization'] = `Bearer ${apiKey}`;
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
  if (ensureInvoke && /(?:127\.0\.0\.1|localhost)/.test(endpoint)) {
    const portMatch = endpoint.match(/:(\d+)/);
    if (portMatch) {
      const port = parseInt(portMatch[1], 10);
      try {
        const r = await ensureInvoke<{ success?: boolean; port?: number }>(
          'ensure_service_ready_by_port',
          { port },
        );
        if (r?.success && r.port && r.port !== port) {
          endpoint = endpoint.replace(`:${port}`, `:${r.port}`);
        }
      } catch {
        /* no companion / not reachable — fall through to the direct request */
      }
    }
  }

  ctx?.log?.('info', `[Service] ${upMethod} ${endpoint} (node ${nodeId})`);
  const res = await httpCall(endpoint, upMethod, headers, body);
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

  return {
    call,
  };
}

const CoreServiceRuntime: RuntimeModule = {
  name: 'Service',
  createMethods: createServiceMethods,
  methods: {},
};

export default CoreServiceRuntime;
