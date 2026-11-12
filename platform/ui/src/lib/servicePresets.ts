/**
 * The providers the welcome wizard offers, and how a provider's models are listed for it.
 *
 * Nothing here names a model. A preset says where a provider is and how it is spoken to; which model it runs is
 * asked of the provider (its own list) and chosen by the person, and the service that is made carries that choice
 * in its `model`, which the body template takes as `{{model}}`. The wizard used to write a model into each body
 * (one per provider, picked when the wizard was written), and the starter flow it built asked the person's server
 * for a model it did not have.
 *
 * No React here: the wizard draws these, and tests/starter-models.mjs runs them against a stand-in server.
 */
import type { CustomService } from 'oaiy-core/modules/core-service/examples';
import { chatModels, listModels, type ListModelsOptions, type ModelInfo } from '@oaiy/shared/providers/models';
import { ProviderConnectionError } from '@oaiy/shared/providers/errors';
import type { ProviderConfig } from '@oaiy/shared/providers/types';

/** What a starter flow makes with the provider: a reply to a prompt, or a picture from one. */
export type PresetUse = 'chat' | 'image';

export interface ServicePreset {
  id: string;
  label: string;
  /** One line under the tile's form: what it is, in the person's words. */
  hint: string;
  /** Which line icon the tile shows (the wizard maps these to its icons). */
  icon: 'cloud' | 'server' | 'plug' | 'monitor';
  needsApiKey: boolean;
  apiKeyConstant?: string;
  apiKeyDocsUrl?: string;
  /** How the provider's list is asked for, and how its errors are worded. */
  provider: Pick<ProviderConfig, 'type' | 'serverKind'>;
  /** What can be made with it, each with the address it is called at. The first is the default. */
  uses: Array<{ use: PresetUse; endpoint: string }>;
  /** What to do when the provider lists no models, after "<label> has no models yet: ". */
  whenEmpty: string;
}

export const SERVICE_PRESETS: ServicePreset[] = [
  {
    id: 'openai',
    label: 'OpenAI',
    // (its key is kept in this browser and goes only to api.openai.com, which answers a browser's requests)
    hint: 'api.openai.com, with your own API key. What it can make, replies or pictures, comes from the models your key lists.',
    icon: 'cloud',
    needsApiKey: true,
    apiKeyConstant: 'OPENAI_API_KEY',
    apiKeyDocsUrl: 'https://platform.openai.com/api-keys',
    provider: { type: 'openai' },
    uses: [
      { use: 'chat', endpoint: 'https://api.openai.com/v1/chat/completions' },
      { use: 'image', endpoint: 'https://api.openai.com/v1/images/generations' },
    ],
    whenEmpty: 'check the key and its project',
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    hint: 'api.anthropic.com, with your own API key.',
    icon: 'cloud',
    needsApiKey: true,
    apiKeyConstant: 'ANTHROPIC_API_KEY',
    apiKeyDocsUrl: 'https://console.anthropic.com/settings/keys',
    provider: { type: 'anthropic' },
    uses: [{ use: 'chat', endpoint: 'https://api.anthropic.com/v1/messages' }],
    whenEmpty: 'check the key',
  },
  {
    id: 'ollama',
    label: 'Ollama',
    hint: 'Ollama on this computer. No key needed.',
    icon: 'server',
    needsApiKey: false,
    provider: { type: 'local', serverKind: 'ollama' },
    uses: [{ use: 'chat', endpoint: 'http://localhost:11434/api/chat' }],
    whenEmpty: 'pull one',
  },
  {
    id: 'openai-compatible',
    label: 'OpenAI-compatible',
    hint: 'Any server that speaks OpenAI’s chat API: vLLM, llama-server, LM Studio and others. No key for one on this computer.',
    icon: 'plug',
    needsApiKey: false,
    provider: { type: 'local', serverKind: 'other' },
    uses: [{ use: 'chat', endpoint: 'http://localhost:1234/v1/chat/completions' }],
    whenEmpty: 'load one in the server',
  },
  {
    id: 'lm-studio',
    label: 'LM Studio',
    hint: 'LM Studio on this computer, with its server started. No key needed.',
    icon: 'monitor',
    needsApiKey: false,
    provider: { type: 'local', serverKind: 'lmstudio' },
    uses: [{ use: 'chat', endpoint: 'http://localhost:1234/api/v1/chat' }],
    whenEmpty: 'load one in LM Studio',
  },
];

/**
 * The service a preset makes for `use`, running `model`: everything but its id, name and address. The body names no
 * model itself: `{{model}}` is the service's `model`, quoted, so changing the model later changes what is asked for.
 */
export function presetFields(preset: ServicePreset, use: PresetUse, model: string): Omit<CustomService, 'id' | 'name' | 'endpoint'> {
  const key = preset.apiKeyConstant ? { apiKeyConstant: preset.apiKeyConstant } : {};
  if (preset.id === 'openai' && use === 'image') {
    return {
      method: 'POST',
      // {{apiKeyRaw}} (not {{apiKey}}): inside a quoted JSON string the value goes in as it is.
      headers: '{"Authorization": "Bearer {{apiKeyRaw}}"}',
      bodyTemplate: '{\n  "model": {{model}},\n  "prompt": {{input}},\n  "n": 1,\n  "size": "1024x1024"\n}',
      responseType: 'json',
      // (base64, which the output node shows as a picture)
      responsePath: 'data.0.b64_json',
      apiFormat: 'openai',
      model,
      nodeTypes: ['image_gen', 'service_call'],
      ...key,
    };
  }
  if (preset.id === 'anthropic') {
    return {
      method: 'POST',
      // `anthropic-dangerous-direct-browser-access` is Anthropic's own opt-in for calls from a browser.
      headers: '{"anthropic-version": "2023-06-01", "x-api-key": "{{apiKeyRaw}}", "anthropic-dangerous-direct-browser-access": "true"}',
      bodyTemplate: '{\n  "model": {{model}},\n  "max_tokens": 4096,\n  "messages": [{"role": "user", "content": {{input}}}]\n}',
      responseType: 'json',
      responsePath: 'content.0.text',
      apiFormat: 'anthropic',
      model,
      nodeTypes: ['ai_llm', 'service_call'],
      ...key,
    };
  }
  if (preset.id === 'ollama') {
    return {
      method: 'POST',
      headers: '{}',
      bodyTemplate: '{\n  "model": {{model}},\n  "messages": [{"role": "user", "content": {{input}}}],\n  "stream": false\n}',
      responseType: 'json',
      responsePath: 'message.content',
      apiFormat: 'ollama',
      model,
      nodeTypes: ['ai_llm', 'service_call'],
    };
  }
  if (preset.id === 'lm-studio') {
    return {
      method: 'POST',
      headers: '{}',
      bodyTemplate: '{\n  "model": {{model}},\n  "input": {{input}}\n}',
      responseType: 'json',
      responsePath: '',
      apiFormat: 'openai',
      model,
      nodeTypes: ['ai_llm', 'service_call'],
    };
  }
  return {
    method: 'POST',
    headers: preset.needsApiKey ? '{"Authorization": "Bearer {{apiKeyRaw}}"}' : '{}',
    bodyTemplate: '{\n  "model": {{model}},\n  "messages": [{"role": "user", "content": {{input}}}]\n}',
    responseType: 'json',
    responsePath: 'choices.0.message.content',
    apiFormat: 'openai',
    model,
    nodeTypes: ['ai_llm', 'service_call'],
    ...key,
  };
}

/** Where a provider's list of models is asked for, from the address its service is called at. */
export function providerOf(preset: ServicePreset, endpoint: string, apiKey: string): Pick<ProviderConfig, 'type' | 'apiKey' | 'baseUrl' | 'serverKind'> {
  let baseUrl = endpoint.trim();
  try {
    const url = new URL(baseUrl);
    // Ollama's and LM Studio's own chat routes are not under the API base their lists are: the server's address is.
    if (preset.provider.serverKind === 'ollama') baseUrl = url.origin;
    else if (preset.provider.serverKind === 'lmstudio') baseUrl = `${url.origin}/v1`;
    // (OpenAI's pictures route is beside its chat route, under the same base)
    else baseUrl = `${url.origin}${url.pathname.replace(/\/+$/, '').replace(/\/images\/generations$/, '')}`;
  } catch {
    // Not an address: the request says so.
  }
  return { ...preset.provider, apiKey, baseUrl };
}

/** Whether a model's id says it makes pictures (a capability's word, not a model's name). */
const makesPictures = (id: string): boolean => id.toLowerCase().includes('image');

export interface ListedModels {
  /** The models for each thing the preset can make, as the provider names them, newest first where it says. */
  byUse: Partial<Record<PresetUse, string[]>>;
  /** How many the provider listed in all. */
  total: number;
}

/** The provider's own list of models, sorted into what each can be used for here. Throws a sentence a person can act on. */
export async function listPresetModels(preset: ServicePreset, endpoint: string, apiKey: string, options: ListModelsOptions = {}): Promise<ListedModels> {
  const all: ModelInfo[] = await listModels(providerOf(preset, endpoint, apiKey), options);
  const byUse: ListedModels['byUse'] = {};
  for (const { use } of preset.uses) {
    byUse[use] = use === 'image' ? all.filter((m) => makesPictures(m.id)).map((m) => m.id) : chatModels(all).models.map((m) => m.id);
  }
  return { byUse, total: all.length };
}

/** The ends of an address that are a route of the API, not part of its base: the list of models is beside them. */
const ROUTE_TAILS = /\/(?:chat\/completions|completions|images\/generations|images\/edits|audio\/speech|audio\/transcriptions|embeddings|messages)$/;

/**
 * The provider behind any service, from how it is spoken to and where it is: for asking it which models it has
 * (Settings → Services' List models, the way the wizard's Test asks). A service on Ollama's or LM Studio's own
 * routes is asked at the server's address; any other at the base its route is under, and at Ollama's own list if
 * that fails (the shared code tries both for a server that is not one of the two big APIs).
 */
export function providerOfService(service: Pick<CustomService, 'apiFormat' | 'endpoint'>, apiKey: string): Pick<ProviderConfig, 'type' | 'apiKey' | 'baseUrl' | 'serverKind'> {
  const endpoint = service.endpoint.trim();
  let url: URL | null = null;
  try {
    url = new URL(endpoint);
  } catch {
    // Not an address: the request says so.
  }
  if (!url) return { type: 'custom', apiKey, baseUrl: endpoint };
  const path = url.pathname.replace(/\/+$/, '');
  if (service.apiFormat === 'anthropic' || url.hostname === 'api.anthropic.com') return { type: 'anthropic', apiKey, baseUrl: `${url.origin}${path.replace(ROUTE_TAILS, '')}` };
  if (service.apiFormat === 'ollama' || /^\/api\/(?:chat|generate)$/.test(path)) return { type: 'local', serverKind: 'ollama', apiKey, baseUrl: url.origin };
  if (service.apiFormat === 'lmstudio' || /^\/api\/v\d+\//.test(path)) return { type: 'local', serverKind: 'lmstudio', apiKey, baseUrl: `${url.origin}/v1` };
  const base = ROUTE_TAILS.test(path) ? `${url.origin}${path.replace(ROUTE_TAILS, '')}` : url.origin;
  return { type: url.hostname === 'api.openai.com' ? 'openai' : 'custom', apiKey, baseUrl: base };
}

/** Every model the service's provider lists, by name, as it lists them. Throws a sentence a person can act on. */
export async function listServiceModels(service: Pick<CustomService, 'apiFormat' | 'endpoint'>, apiKey: string, options: ListModelsOptions = {}): Promise<string[]> {
  return (await listModels(providerOfService(service, apiKey), options)).map((m) => m.id);
}

/** What the wizard says when the provider answered and has nothing to run, e.g. "Ollama has no models yet: pull one, then Test again." */
export function noModelsWords(preset: ServicePreset): string {
  return `${preset.label} has no models yet: ${preset.whenEmpty}, then Test again.`;
}

/** A listing's failure as a sentence (the shared code words its own; anything else is said plainly). */
export function listErrorWords(error: unknown, endpoint: string): string {
  if (error instanceof ProviderConnectionError) return error.message;
  return `Could not ask ${endpoint} for its models: ${error instanceof Error ? error.message : String(error)}`;
}
