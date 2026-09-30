/*
 * The shapes a provider has, in every place OAIY keeps one.
 *
 * `ProviderConfig` and its neighbours (the top half) are the Agent's own record: adapted from softn.com
 * (apps/softn-studio/src/types/studio.ts), Copyright f2i-com, licensed under the Apache License, Version 2.0.
 * They stay as the Agent has always had them; app/src/agent/providers/types.ts re-exports them.
 *
 * `ProviderRecord` (the bottom half) is the one record of the web app's providers origin (design 4.1): two
 * dialects, one meaning for "base URL", and no key. The key is not a field of any record: the store keeps it
 * as the secret named `key:<id>` (see `providerKeyName`), and nothing that leaves the providers origin carries it.
 *
 * This file is plain TypeScript with no imports, so the Agent, the flow editor and the providers origin can all
 * use it (`@oaiy/shared`).
 */

// --- The Agent's provider (as it is saved on this device) ----------------------

export type ProviderType = 'anthropic' | 'openai' | 'local' | 'custom';

export type LocalServerKind = 'ollama' | 'lmstudio' | 'oaiy' | 'other';

export interface ProviderConfig {
  id: string;
  type: ProviderType;
  name: string;
  apiKey: string;
  /** The server's address or API base (`http://localhost:11434`, `https://example.com/v1`). */
  baseUrl?: string;
  /** The model every request uses. There is no default. */
  modelId?: string;
  /**
   * OAIY only: use the model chosen in OAIY's Engines (its default), whichever
   * that is now. Requests name no model, so the engine answers with its own
   * choice; `modelId` shows which that was when OAIY was last asked.
   */
  followEngine?: boolean;
  orgId?: string;
  /** For a `local` provider: which server, for its address and its advice. */
  serverKind?: LocalServerKind;
  /** The context window in tokens, as the person set it: wins over anything detected. */
  contextTokens?: number;
  /** The window the server reported for `model` (or stated in an overflow error). */
  detectedContext?: { model: string; tokens: number; how: string; at: number };
  /** How many agents may use this provider at once (default 1 for a local server, 3 for an API). */
  parallelAgents?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

// --- The shared record --------------------------------------------------------

/** How a provider talks. Gemini, Ollama, LM Studio, llama.cpp, vLLM, OpenRouter, Groq, Mistral, DeepSeek, xAI and Together speak `openai`; Anthropic is the other. */
export type Dialect = 'openai' | 'anthropic';

/** How the key is attached to a request: `Authorization: Bearer`, Anthropic's `x-api-key`, or not at all. */
export type ProviderAuth = 'bearer' | 'x-api-key' | 'none';

export type ProviderCap = 'chat' | 'tools' | 'vision' | 'image' | 'speech' | 'transcription' | 'embeddings';

export const PROVIDER_CAPS: readonly ProviderCap[] = ['chat', 'tools', 'vision', 'image', 'speech', 'transcription', 'embeddings'];

/** Where the provider is: a service on the internet, a server on this computer or network, or the model that runs inside the providers origin itself. */
export type ProviderKind = 'external' | 'local-server' | 'browser-engine';

/** Who makes the calls: the providers origin (`broker`), or the person's own OAIY server or desktop, which holds the key (`gateway`). */
export type ProviderVia = 'broker' | 'gateway';

export type ServerKind = LocalServerKind;

export interface ExtraHeader {
  name: string;
  value: string;
}

/**
 * The record. `baseUrl` is the API base INCLUDING its version segment, defined once for every place that has one:
 * `https://api.openai.com/v1`, `https://api.anthropic.com/v1`, `https://openrouter.ai/api/v1`,
 * `https://generativelanguage.googleapis.com/v1beta/openai`, `http://localhost:11434/v1`. It has no query string, no
 * fragment and no user name; a request is `baseUrl` plus one path of a fixed list (`API_PATHS` in endpoints.ts).
 */
export interface ProviderRecord {
  v: 1;
  id: string;
  name: string;
  dialect: Dialect;
  baseUrl: string;
  auth: ProviderAuth;
  /** Only the names of `EXTRA_HEADER_NAMES`. The holder attaches them itself; a page cannot add one. */
  extraHeaders?: ExtraHeader[];
  model?: string;
  caps: ProviderCap[];
  contextTokens?: number;
  kind: ProviderKind;
  preset?: string;
  serverKind?: ServerKind;
  via: ProviderVia;
  parallelAgents?: number;
}

/** What a page is told about a record: everything but the secret. `hasKey` says whether one is stored; `locked` whether the vault could open it. */
export interface ProviderSummary {
  id: string;
  name: string;
  dialect: Dialect;
  /** The host of the base URL (`api.openai.com`, `localhost:11434`): where a key would go. */
  host: string;
  caps: ProviderCap[];
  model: string | null;
  hasKey: boolean;
  kind: ProviderKind;
  locked: boolean;
}

/** The headers a record may carry beyond the ones its dialect fixes (lower case). */
export const EXTRA_HEADER_NAMES: readonly string[] = ['openai-organization', 'http-referer', 'x-title', 'anthropic-beta'];

/** The name a provider's key has in the secret store. */
export function providerKeyName(id: string): string {
  return `key:${id}`;
}
