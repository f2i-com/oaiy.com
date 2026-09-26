/**
 * How much the model can read at once (its context window), found out from
 * the server where it says, and the arithmetic of fitting a conversation into
 * it: what the system prompt and tools cost, what to keep free for the reply,
 * and when to compact.
 *
 * Servers report the window differently:
 *   - Ollama: /api/ps gives the size the model is LOADED with (often 4096,
 *     whatever the model supports), which is what actually applies; /api/show
 *     gives the model's maximum;
 *   - LM Studio: /api/v0/models/<id> gives loaded_context_length;
 *   - llama.cpp: /props gives n_ctx;
 *   - vLLM, OpenRouter and others: /v1/models entries carry max_model_len,
 *     context_length or meta.n_ctx;
 *   - cloud models: a table of known windows.
 * A person's own number in Settings wins over all of them.
 */
import { providerEndpoints, providerHeaders } from './providers/providerConnection';
import type { ProviderConfig } from './providers/types';

/** Known windows by model name (checked in order; the first match wins). */
const KNOWN: Array<[RegExp, number]> = [
  [/claude-(opus|sonnet)-4-[56]|claude-sonnet-4-5|claude-(opus|sonnet)-5|claude-fable/i, 200_000],
  [/claude/i, 200_000],
  [/gpt-4\.1/i, 1_047_576],
  [/gpt-5/i, 400_000],
  [/gpt-4o|gpt-4-turbo|o1-mini/i, 128_000],
  [/\bo[134](-|$)/i, 200_000],
  [/gemini-(1\.5|2|2\.5|3)/i, 1_048_576],
  [/llama-?3\.[1-3]|llama-?4/i, 131_072],
  [/deepseek/i, 131_072],
  [/qwen-?3|qwen3/i, 32_768],
  [/qwen-?2\.5/i, 32_768],
  [/mistral-large|mixtral/i, 131_072],
];

/** When nothing better is known: generous for a cloud API, cautious for a local server. */
export const DEFAULT_CLOUD_WINDOW = 128_000;
export const DEFAULT_LOCAL_WINDOW = 8_192;

export function knownWindow(provider: Pick<ProviderConfig, 'type' | 'modelId'>): number | null {
  const id = provider.modelId ?? '';
  // A local server runs a model with whatever window it was loaded with, not its maximum.
  if (provider.type === 'local') return null;
  for (const [pattern, tokens] of KNOWN) if (pattern.test(id)) return tokens;
  return null;
}

/** The window to plan for, and where the number came from. */
export function contextWindow(provider: ProviderConfig): { tokens: number; source: 'yours' | 'server' | 'known' | 'default' } {
  if (provider.contextTokens && provider.contextTokens >= 1024) return { tokens: provider.contextTokens, source: 'yours' };
  if (provider.detectedContext && provider.detectedContext.model === provider.modelId && provider.detectedContext.tokens >= 1024) return { tokens: provider.detectedContext.tokens, source: 'server' };
  const known = knownWindow(provider);
  if (known) return { tokens: known, source: 'known' };
  return { tokens: provider.type === 'local' ? DEFAULT_LOCAL_WINDOW : DEFAULT_CLOUD_WINDOW, source: 'default' };
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v >= 512 ? Math.floor(v) : typeof v === 'string' && /^\d+$/.test(v) && Number(v) >= 512 ? Number(v) : null);

async function getJson(url: string, init: RequestInit, signal?: AbortSignal): Promise<unknown> {
  const timeout = AbortSignal.timeout(6000);
  const response = await fetch(url, { ...init, signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  if (!response.ok) throw new Error(`${response.status}`);
  return response.json();
}

/** Ask the server how big the model's window is. Null when it does not say. */
export async function detectContextWindow(provider: ProviderConfig, signal?: AbortSignal): Promise<{ tokens: number; how: string } | null> {
  const model = provider.modelId?.trim();
  if (!model || provider.type === 'anthropic') return null;
  const endpoints = providerEndpoints(provider);
  const headers = providerHeaders(provider);
  const origin = new URL(endpoints.chat).origin;
  const attempts: Array<() => Promise<{ tokens: number; how: string } | null>> = [];
  const isOllama = provider.serverKind === 'ollama' || /:11434$/.test(origin);
  if (isOllama) {
    attempts.push(async () => {
      const body = (await getJson(`${origin}/api/ps`, { headers }, signal)) as { models?: Array<Record<string, unknown>> };
      const loaded = body.models?.find((m) => m.name === model || m.model === model || String(m.name).startsWith(`${model}:`));
      const tokens = num(loaded?.context_length);
      return tokens ? { tokens, how: 'the size Ollama loaded the model with (/api/ps)' } : null;
    });
    attempts.push(async () => {
      const body = (await getJson(`${origin}/api/show`, { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) }, signal)) as Record<string, unknown>;
      // num_ctx in the model's parameters is what Ollama runs it with; the model's maximum is only a ceiling.
      const params = typeof body.parameters === 'string' ? /num_ctx\s+(\d+)/.exec(body.parameters)?.[1] : undefined;
      if (params) return { tokens: Number(params), how: 'the num_ctx the model is set up with (/api/show)' };
      return null;
    });
  }
  if (provider.serverKind === 'lmstudio' || /:1234$/.test(origin)) {
    attempts.push(async () => {
      const body = (await getJson(`${origin}/api/v0/models/${encodeURIComponent(model)}`, { headers }, signal)) as Record<string, unknown>;
      const tokens = num(body.loaded_context_length) ?? num(body.max_context_length);
      return tokens ? { tokens, how: 'LM Studio (/api/v0/models)' } : null;
    });
  }
  // Any OpenAI-compatible server: the models list, then llama.cpp's /props.
  attempts.push(async () => {
    const body = (await getJson(endpoints.models, { headers }, signal)) as { data?: Array<Record<string, unknown>> };
    const entry = body.data?.find((m) => m.id === model) ?? (body.data?.length === 1 ? body.data[0] : undefined);
    if (!entry) return null;
    const meta = (entry.meta ?? {}) as Record<string, unknown>;
    const tokens = num(entry.max_model_len) ?? num(entry.context_length) ?? num(entry.context_window) ?? num(entry.max_context_length) ?? num(meta.n_ctx) ?? num((entry.top_provider as Record<string, unknown> | undefined)?.context_length) ?? num(meta.n_ctx_train);
    return tokens ? { tokens, how: 'the server\'s model list' } : null;
  });
  attempts.push(async () => {
    const body = (await getJson(`${origin}/props`, { headers }, signal)) as Record<string, unknown>;
    const settings = (body.default_generation_settings ?? {}) as Record<string, unknown>;
    const tokens = num(settings.n_ctx) ?? num(body.n_ctx);
    return tokens ? { tokens, how: 'llama.cpp (/props)' } : null;
  });
  if (isOllama) attempts.push(async () => {
    const body = (await getJson(`${origin}/api/show`, { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) }, signal)) as { model_info?: Record<string, unknown> };
    const key = Object.keys(body.model_info ?? {}).find((k) => k.endsWith('.context_length'));
    // The model's maximum, but Ollama's default load is smaller: capped at its usual default.
    const tokens = key ? num(body.model_info![key]) : null;
    return tokens ? { tokens: Math.min(tokens, 4096), how: 'Ollama\'s default load size (the model allows more: set OLLAMA_CONTEXT_LENGTH or num_ctx, then detect again)' } : null;
  });
  for (const attempt of attempts) {
    try {
      const found = await attempt();
      if (found) return found;
    } catch {
      /* this server does not answer that one */
    }
  }
  return null;
}

/** A context-overflow error from the server, with the window it states when it does. */
export function overflowWindow(message: string): { overflow: boolean; tokens: number | null } {
  const overflow = /context (length|size|window)|maximum context|too many tokens|exceeds? (the )?(available )?context|prompt is too long|reduce the length|n_ctx|input.*too long/i.test(message);
  if (!overflow) return { overflow: false, tokens: null };
  const stated =
    /maximum context length is (\d+)/i.exec(message) ??
    /context (?:size|length|window)(?: of| is|:)? \(?(\d{3,7})\)?/i.exec(message) ??
    /available context size \((\d+)/i.exec(message) ??
    /n_ctx(?:_slot)?\s*[=:]\s*(\d+)/i.exec(message) ??
    /(\d{4,7}) tokens? (?:maximum|limit)/i.exec(message);
  return { overflow: true, tokens: stated ? Number(stated[1]) : null };
}

/** An error about max_tokens being too high, with the limit it states. */
export function outputLimit(message: string): number | null {
  if (!/max_tokens|max_completion_tokens|output tokens|completion tokens|max_new_tokens/i.test(message)) return null;
  const stated =
    /valid range of max_tokens is \[\d+,\s*(\d+)\]/i.exec(message) ??
    /(?:at most|maximum(?: allowed)?(?: number of)?(?: output| completion)? tokens?(?: is| of)?|less than or equal to|<=?)\s*:?\s*(\d{3,7})/i.exec(message) ??
    /max_tokens:?\s*\d+\s*>\s*(\d{3,7})/i.exec(message) ??
    /(\d{3,7}) (?:output|completion) tokens/i.exec(message);
  const n = stated ? Number(stated[1]) : NaN;
  return Number.isFinite(n) && n >= 256 ? n : null;
}

/** The share of the window the prompt may reach before compacting. */
export const DEFAULT_COMPACT_AT = 0.75;

/**
 * The room there is: what to keep free for the reply (a quarter of the window
 * at most, and never more than the output limit) and what the prompt may use.
 */
export function budgetFor(window: number, fixedTokens: number, maxOutput = 16_384): { reply: number; prompt: number } {
  // A small window keeps less for the reply: the instructions and tools already take a lot of it.
  const reply = Math.max(512, Math.min(maxOutput, Math.floor(window * (window < 16_384 ? 0.2 : 0.25))));
  return { reply, prompt: Math.max(0, window - reply - fixedTokens) };
}

/** Round a token count for people: 950, 12.3k, 262k. */
export function formatTokens(n: number): string {
  if (n < 1000) return String(Math.round(n));
  if (n < 100_000) return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`;
  return `${Math.round(n / 1000)}k`;
}
