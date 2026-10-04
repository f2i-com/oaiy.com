/**
 * What the Agent runs on: the person's choice, kept on OAIY Desktop
 * (`GET /api/agent/preferences`, set in its setup wizard and Settings →
 * Agent). `engine` is the model chosen in OAIY's Engines, reached as the app
 * always has (the OAIY provider that follows Engines); `chatgpt` is the
 * person's ChatGPT sign-in, through OAIY's Codex connector; `provider` is one
 * of the desktop's AI providers (LM Studio, Ollama, an API key), through its
 * gateway, with the model chosen there.
 *
 * On ChatGPT a live phone call keeps its own route: one of the connector's
 * live-call aliases, which pins a fast model and effort (and still takes the
 * call's tools, streaming). Every other conversation uses the generic route,
 * with the model the choice names, or Codex's own default.
 */
import { AIProviderError } from '../agent/providers/aiProvider';
import { CHATGPT_SIGN_IN, signInNeeded } from '../agent/providers/chatgpt';
import type { ProviderConfig } from '../agent/providers/types';

/** The person's choice, as the desktop keeps it. */
export type AgentModel = { source: 'engine' } | { source: 'chatgpt'; model?: string } | ProviderChoice;

/** One of the desktop's AI providers, by id, and the model the Agent runs on there (`name`: the provider's, as the desktop answers it beside the choice). */
export type ProviderChoice = { source: 'provider'; provider: string; model: string; name?: string };

export const ENGINE: AgentModel = { source: 'engine' };

/** Every kind of conversation the app holds: the person's own (a project, "Set up OAIY", the Front desk's runner) and the phone's. */
export type AgentKind = 'project' | 'setup' | 'runner' | 'call' | 'sms' | 'task';

/** The connector's generic route: the model is the caller's to name. */
export const CODEX_ROUTE = 'openai-codex-agent';
/** The live-call alias a call takes by default: GPT-5.5 with reasoning off, pinned by the desktop. */
export const CALL_ROUTE = 'openai-codex-agent-none';
/** The live-call alias for Luna: low reasoning, in Fast mode (a caller cannot wait). */
export const LUNA_CALL_ROUTE = 'openai-codex-agent-luna-low-fast';
/**
 * The window the app works in on ChatGPT. The connector reports no usage, so
 * the app cannot learn it; this keeps prompts well inside what Codex's models
 * take, and compacts before a turn gets slow.
 */
export const CHATGPT_CONTEXT_TOKENS = 128_000;

type Send = (input: string, init: RequestInit) => Promise<Response>;
const send: Send = (input, init) => fetch(input, init);
const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** The choice in the desktop's answer (`{model: {source, provider?, model?}, providerName?}`), or null when it is not one. */
export function parseAgentModel(body: unknown): AgentModel | null {
  const model = isRecord(body) ? body.model : null;
  if (!isRecord(model)) return null;
  if (model.source === 'engine') return ENGINE;
  const name = typeof model.model === 'string' ? model.model.trim() : '';
  if (model.source === 'provider') {
    const provider = typeof model.provider === 'string' ? model.provider.trim() : '';
    if (!provider || !name) return null;
    const label = isRecord(body) && typeof body.providerName === 'string' ? body.providerName.trim() : '';
    return label ? { source: 'provider', provider, model: name, name: label } : { source: 'provider', provider, model: name };
  }
  if (model.source !== 'chatgpt') return null;
  return name ? { source: 'chatgpt', model: name } : { source: 'chatgpt' };
}

/**
 * The desktop's choice now. An older desktop, without the route (404), has
 * none: the engine. Null when it could not be asked (keep what was known).
 */
export async function readAgentModel(desktop: { origin: string; token: string }, signal?: AbortSignal, fetchImpl: Send = send): Promise<AgentModel | null> {
  try {
    const resp = await fetchImpl(`${desktop.origin}/api/agent/preferences`, { headers: { authorization: `Bearer ${desktop.token}` }, signal });
    if (resp.status === 404) return ENGINE;
    if (!resp.ok) return null;
    return parseAgentModel(await resp.json()) ?? ENGINE;
  } catch {
    return null;
  }
}

/** The same choice? (A provider renamed is the same one.) */
export function sameModel(a: AgentModel, b: AgentModel): boolean {
  if (a.source !== b.source) return false;
  if (a.source === 'engine') return true;
  if (a.source === 'provider') return a.provider === (b as ProviderChoice).provider && a.model === (b as ProviderChoice).model;
  return a.model === (b as { model?: string }).model;
}

/** The live-call alias for a call on ChatGPT: Luna's when the choice names Luna, else the default one. */
export function callRoute(model?: string): string {
  return model && /luna/i.test(model) ? LUNA_CALL_ROUTE : CALL_ROUTE;
}

/** Where one of the connector's routes answers (an OpenAI-style API base). */
export function codexBase(origin: string, route: string): string {
  return `${origin.replace(/\/+$/, '')}/api/ai/providers/${route}/v1`;
}

/**
 * ChatGPT as a provider for a conversation of `kind`, on the desktop at
 * `desktop` (its token is the key). `model`: the one chosen, or Codex's
 * default once looked up; empty until then (the run looks it up first). A
 * call's route pins its own model, so its `modelId` only names it.
 */
export function chatgptProvider(desktop: { origin: string; token: string }, kind: AgentKind, model: string | null | undefined): ProviderConfig {
  const route = kind === 'call' ? callRoute(model ?? undefined) : CODEX_ROUTE;
  return {
    id: `oaiy-chatgpt${kind === 'call' ? '-call' : ''}`,
    // Not `local` (that warms the model and sends llama.cpp's template switches) nor `openai` (its own host rules).
    type: 'custom',
    serverKind: 'other',
    name: 'ChatGPT',
    apiKey: desktop.token,
    baseUrl: codexBase(desktop.origin, route),
    modelId: model || (kind === 'call' ? route : ''),
    contextTokens: CHATGPT_CONTEXT_TOKENS,
    // The connector runs one turn at a time: sub-agents take turns too.
    parallelAgents: 1,
  };
}

/**
 * One of the desktop's AI providers, through its gateway (the same route as
 * ChatGPT's, by the provider's id; the desktop's token is the key, and the
 * desktop adds the provider's own). Every conversation, a call's too, runs on
 * the model chosen for it.
 */
export function desktopProvider(desktop: { origin: string; token: string }, choice: ProviderChoice): ProviderConfig {
  return {
    id: `oaiy-provider-${choice.provider}`,
    // As ChatGPT's: the desktop's gateway speaks OpenAI's API, whatever the server behind it is.
    type: 'custom',
    serverKind: 'other',
    name: choice.name || choice.provider,
    apiKey: desktop.token,
    baseUrl: codexBase(desktop.origin, choice.provider),
    modelId: choice.model,
    // A local server (LM Studio, Ollama) answers one request at a time.
    parallelAgents: 1,
  };
}

/** The provider a conversation of `kind` runs on: the engine's (the app's own, as chosen in Settings), ChatGPT's, or a desktop AI provider's. */
export function providerFor(kind: AgentKind, choice: AgentModel, engine: ProviderConfig | null, desktop: { origin: string; token: string } | null, codexDefault: string | null): ProviderConfig | null {
  if (choice.source === 'engine' || !desktop) return engine;
  if (choice.source === 'provider') return desktopProvider(desktop, choice);
  return chatgptProvider(desktop, kind, choice.model ?? codexDefault);
}

/** The model Codex runs when none is named: the catalogue's `isDefault` row, else its first. */
export function pickCodexModel(body: unknown): string | null {
  const rows = isRecord(body) && Array.isArray(body.data) ? body.data.filter(isRecord) : [];
  const ids = rows.filter((r) => typeof r.id === 'string' && r.id.trim());
  const chosen = ids.find((r) => r.isDefault === true) ?? ids[0];
  return chosen ? String(chosen.id).trim() : null;
}

/**
 * Codex's own default model, from the connector's model list. Signed out, it
 * answers 428: the error says where to sign in (and nothing else is used).
 */
export async function codexDefaultModel(desktop: { origin: string; token: string }, signal?: AbortSignal, fetchImpl: Send = send): Promise<string> {
  let resp: Response;
  try {
    resp = await fetchImpl(`${codexBase(desktop.origin, CODEX_ROUTE)}/models`, { headers: { authorization: `Bearer ${desktop.token}` }, signal });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(`OAIY Desktop did not answer for ChatGPT (${(error as Error).message}).`);
  }
  const text = await resp.text();
  if (!resp.ok) {
    if (signInNeeded(text)) throw new AIProviderError('http', CHATGPT_SIGN_IN, { status: resp.status, detail: text.slice(0, 240) });
    let said = '';
    try {
      const body = JSON.parse(text) as unknown;
      const error = isRecord(body) ? body.error : null;
      said = isRecord(error) && typeof error.message === 'string' ? error.message : typeof error === 'string' ? error : '';
    } catch {
      said = text.slice(0, 200);
    }
    throw new Error(`ChatGPT in OAIY did not list its models (HTTP ${resp.status})${said ? `: ${said}` : ''}.`);
  }
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    /* not a list */
  }
  const model = pickCodexModel(body);
  if (!model) throw new Error('ChatGPT in OAIY lists no models to run the Agent on.');
  return model;
}

/** What the header's model chip says: which provider and model the Agent is on ("ChatGPT · gpt-5.5", "OAIY · Qwen3-8B"). */
export function modelChipText(provider: ProviderConfig | null, problem = ''): string {
  if (!provider) return 'Set up AI…';
  if (problem) return `${provider.name} · ${problem}`;
  return `${provider.name} · ${provider.modelId || (provider.name === 'ChatGPT' ? 'default model' : 'no model')}`;
}
