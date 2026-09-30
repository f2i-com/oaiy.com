/**
 * How much the model can read at once (its context window), found out from the server where it says (design 3.2, `probe`).
 * The Agent's own detection (app/src/agent/context.ts) asks the same servers the same way; this asks for a record and its key,
 * through the holder's guarded fetch, so a server that answers the Agent answers here.
 *
 * Servers report the window differently: Ollama's /api/ps (the size the model is LOADED with) and /api/show; LM Studio's
 * /api/v0/models/<id>; OAIY's /v1/discovery; llama.cpp's /props; and the model list's own entries (vLLM, OpenRouter and others).
 *
 * Only a server on this computer or network is asked (`kind: 'local-server'`): a service on the internet is never sent a probe. The key
 * is for the record's API base, so it goes ONLY to an address under that base (`/v1/models`, `/v1/discovery`); the servers' own
 * information endpoints are outside it (`/api/ps`, `/props`) and are asked WITHOUT the key. The holder chose every address, not the page.
 */
import { providerEndpoints, recordHeaders } from '@oaiy/shared/providers/endpoints';
import { providerTypeOf } from '@oaiy/shared/providers/adapters';
import type { ProviderRecord } from '@oaiy/shared/providers/types';

export interface ProbeResult {
  /** The window in tokens, or null when the server does not say. */
  contextTokens: number | null;
  /** Where the number came from, in words. */
  how: string | null;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v >= 512 ? Math.floor(v) : typeof v === 'string' && /^\d+$/.test(v) && Number(v) >= 512 ? Number(v) : null);

export async function probeContext(record: ProviderRecord, key: string, fetchImpl: typeof fetch): Promise<ProbeResult> {
  const none: ProbeResult = { contextTokens: null, how: null };
  const model = record.model?.trim();
  if (!model || record.dialect === 'anthropic' || record.kind !== 'local-server') return none;
  const endpoints = providerEndpoints({ type: providerTypeOf(record), baseUrl: record.baseUrl, serverKind: record.serverKind });
  const withKey = recordHeaders(record, key);
  const withoutKey = recordHeaders(record, '');
  const base = new URL(record.baseUrl);
  const origin = base.origin;
  /** The key goes only where the record's API base says it may: under the base, on its origin. */
  const underBase = (url: string): boolean => {
    const u = new URL(url);
    return u.origin === origin && (u.pathname === base.pathname || u.pathname.startsWith(`${base.pathname}/`));
  };
  const getJson = async (url: string, init: RequestInit = {}): Promise<unknown> => {
    const response = await fetchImpl(url, { ...init, headers: { ...(underBase(url) ? withKey : withoutKey), ...(init.method === 'POST' ? { 'content-type': 'application/json' } : {}) }, signal: AbortSignal.timeout(6000) });
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  };
  const post = (url: string): Promise<unknown> => getJson(url, { method: 'POST', body: JSON.stringify({ model }) });

  const attempts: Array<() => Promise<ProbeResult | null>> = [];
  const isOllama = record.serverKind === 'ollama' || /:11434$/.test(origin);
  if (isOllama) {
    attempts.push(async () => {
      const body = (await getJson(`${origin}/api/ps`)) as { models?: Array<Record<string, unknown>> };
      const loaded = body.models?.find((m) => m.name === model || m.model === model || String(m.name).startsWith(`${model}:`));
      const tokens = num(loaded?.context_length);
      return tokens ? { contextTokens: tokens, how: 'the size Ollama loaded the model with (/api/ps)' } : null;
    });
    attempts.push(async () => {
      const body = (await post(`${origin}/api/show`)) as Record<string, unknown>;
      const params = typeof body.parameters === 'string' ? /num_ctx\s+(\d+)/.exec(body.parameters)?.[1] : undefined;
      return params ? { contextTokens: Number(params), how: 'the num_ctx the model is set up with (/api/show)' } : null;
    });
  }
  if (record.serverKind === 'lmstudio' || /:1234$/.test(origin)) {
    attempts.push(async () => {
      const body = (await getJson(`${origin}/api/v0/models/${encodeURIComponent(model)}`)) as Record<string, unknown>;
      const tokens = num(body.loaded_context_length) ?? num(body.max_context_length);
      return tokens ? { contextTokens: tokens, how: 'LM Studio (/api/v0/models)' } : null;
    });
  }
  if (record.serverKind === 'oaiy') {
    attempts.push(async () => {
      // Under the API base (`/v1`) when the record's base is the usual one: the key goes with it there.
      const body = (await getJson(`${origin}/v1/discovery`)) as { llm?: { context_tokens?: unknown } };
      const tokens = num(body.llm?.context_tokens);
      return tokens ? { contextTokens: tokens, how: 'OAIY (/v1/discovery)' } : null;
    });
  }
  // Any OpenAI-compatible server: the models list, then llama.cpp's /props.
  attempts.push(async () => {
    const body = (await getJson(endpoints.models)) as { data?: Array<Record<string, unknown>> };
    const entry = body.data?.find((m) => m.id === model) ?? (body.data?.length === 1 ? body.data[0] : undefined);
    if (!entry) return null;
    const meta = (entry.meta ?? {}) as Record<string, unknown>;
    const tokens = num(entry.max_model_len) ?? num(entry.context_length) ?? num(entry.context_window) ?? num(entry.max_context_length) ?? num(meta.n_ctx) ?? num((entry.top_provider as Record<string, unknown> | undefined)?.context_length) ?? num(meta.n_ctx_train);
    return tokens ? { contextTokens: tokens, how: 'the server’s model list' } : null;
  });
  attempts.push(async () => {
    const body = (await getJson(`${origin}/props`)) as Record<string, unknown>;
    const settings = (body.default_generation_settings ?? {}) as Record<string, unknown>;
    const tokens = num(settings.n_ctx) ?? num(body.n_ctx);
    return tokens ? { contextTokens: tokens, how: 'llama.cpp (/props)' } : null;
  });
  for (const attempt of attempts) {
    try {
      const found = await attempt();
      if (found) return found;
    } catch {
      /* this server does not answer that one */
    }
  }
  return none;
}
