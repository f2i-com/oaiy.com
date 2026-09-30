/*
 * Which models a provider offers, and one GET with its failures turned into sentences.
 *
 * Adapted from softn.com (apps/softn-studio/src/lib/providerConnection.ts),
 * Copyright f2i-com, licensed under the Apache License, Version 2.0.
 * app/src/agent/providers/providerConnection.ts re-exports it.
 *
 * Nothing here names a model. Model names change faster than Studio ships,
 * so the list always comes from the provider itself and the person picks
 * from it; a provider with no chosen model is refused, not guessed for.
 *
 * `listModels` is the Agent's (a `ProviderConfig`); `listRecordModels` is the same walk for a `ProviderRecord` and its
 * key, which the providers origin runs with its own `fetch`.
 */
import { providerEndpoints, providerHeaders, recordHeaders, type ProviderEndpoints } from './endpoints';
import { ProviderConnectionError, describeConnectionError, isBlockedMixedContent, kindForStatus, redactSecret, type ErrorContext } from './errors';
import { providerTypeOf } from './adapters';
import type { LocalServerKind, ProviderConfig, ProviderRecord, ProviderType } from './types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/** A short excerpt of the provider's error body, if it has a message. */
async function errorDetail(resp: Response): Promise<string | undefined> {
  try {
    const text = (await resp.text()).slice(0, 2000);
    try {
      const body: unknown = JSON.parse(text);
      const error = isRecord(body) ? body.error : undefined;
      const message = isRecord(error) ? error.message : typeof error === 'string' ? error : isRecord(body) ? body.message : undefined;
      if (typeof message === 'string' && message.trim()) return message.trim().slice(0, 240);
    } catch {
      // Not JSON; a short plain-text body is still worth showing.
    }
    const plain = text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    return plain && plain.length <= 240 ? plain : undefined;
  } catch {
    return undefined;
  }
}

// --- Model lists ------------------------------------------------------------

export interface ModelInfo {
  id: string;
  /** A friendlier name, when the provider gives one. */
  label?: string;
  /** When the provider says the model was published or last changed, in ms. */
  created?: number;
}

export interface ListModelsOptions {
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  /** This page's protocol and origin; default the real page's. */
  page?: { protocol: string; origin: string };
  /**
   * Whether the provider's own words go into an error message: `include` (the default: the Agent's own use, where the text is the person's)
   * or `omit` (the message is fixed wording that depends only on the status, so it can be handed to a page that must not be able to read
   * the key out of it). `scrub` is applied to the text when it is included.
   */
  providerText?: 'include' | 'omit';
  scrub?: (text: string) => string;
}

const LIST_TIMEOUT_MS = 15_000;
/** A provider that keeps saying "more" is not followed forever. */
const MAX_PAGES = 20;

function currentPage(): { protocol: string; origin: string } {
  if (typeof window === 'undefined') return { protocol: 'http:', origin: 'this site' };
  return { protocol: window.location.protocol, origin: window.location.origin };
}

/** A time in ms from seconds, milliseconds or an ISO string, if it is one. */
function toMs(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value < 1e12 ? value * 1000 : value;
  if (typeof value === 'string') {
    const at = Date.parse(value);
    return Number.isFinite(at) ? at : undefined;
  }
  return undefined;
}

/** Newest first when the provider says when; otherwise, and on ties, by name. */
export function sortModels(models: ModelInfo[]): ModelInfo[] {
  return [...models].sort((a, b) => {
    if (a.created !== undefined && b.created !== undefined && a.created !== b.created) return b.created - a.created;
    if (a.created !== undefined && b.created === undefined) return -1;
    if (b.created !== undefined && a.created === undefined) return 1;
    return a.id.localeCompare(b.id);
  });
}

/**
 * Words in a model id that mean it does something other than chat: turn
 * text into vectors, speech or pictures, or check content. The list is of
 * capabilities, not of models, so a new chat model is never hidden by it.
 */
const NON_CHAT_WORDS = ['embed', 'tts', 'whisper', 'transcribe', 'speech', 'audio', 'realtime', 'dall-e', 'image', 'video', 'moderation', 'rerank'];

/** Whether an id looks like a model that can hold a conversation. */
export function isLikelyChatModel(id: string): boolean {
  const lower = id.toLowerCase();
  return !NON_CHAT_WORDS.some((word) => lower.includes(word));
}

/** The models worth offering for chat, and how many were left out. */
export function chatModels(models: ModelInfo[]): { models: ModelInfo[]; hidden: number } {
  const kept = models.filter((model) => isLikelyChatModel(model.id));
  return { models: kept, hidden: models.length - kept.length };
}

function readModelList(body: unknown): ModelInfo[] | null {
  if (!isRecord(body) || !Array.isArray(body.data)) return null;
  const models: ModelInfo[] = [];
  for (const item of body.data) {
    if (!isRecord(item) || typeof item.id !== 'string' || !item.id.trim()) continue;
    // OAIY lists its image, video, speech, music, sound effects and 3D models here too, typed: they cannot chat.
    if (item.type === 'image' || item.type === 'video' || item.type === 'speech' || item.type === 'music' || item.type === 'sound' || item.type === 'model3d') continue;
    models.push({
      id: item.id,
      label: typeof item.display_name === 'string' && item.display_name !== item.id ? item.display_name : undefined,
      created: toMs(item.created_at ?? item.created),
    });
  }
  return models;
}

function readOllamaTags(body: unknown): ModelInfo[] | null {
  if (!isRecord(body) || !Array.isArray(body.models)) return null;
  const models: ModelInfo[] = [];
  for (const item of body.models) {
    if (!isRecord(item)) continue;
    const id = typeof item.model === 'string' ? item.model : typeof item.name === 'string' ? item.name : '';
    if (id.trim()) models.push({ id, created: toMs(item.modified_at) });
  }
  return models;
}

/**
 * One GET, with the failures turned into `ProviderConnectionError`s that
 * say what to do. `read` turns the JSON into a list, or null when the JSON
 * is not a list.
 */
async function getJson(
  url: string,
  headers: Record<string, string>,
  context: Omit<ErrorContext, 'url'>,
  options: ListModelsOptions,
): Promise<unknown> {
  const page = options.page ?? currentPage();
  if (isBlockedMixedContent(url, page.protocol)) {
    throw new ProviderConnectionError('mixed-content', describeConnectionError('mixed-content', { ...context, url, pageOrigin: page.origin }));
  }
  const doFetch = options.fetchImpl ?? fetch;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, options.timeoutMs ?? LIST_TIMEOUT_MS);
  const onAbort = () => controller.abort();
  options.signal?.addEventListener('abort', onAbort, { once: true });
  try {
    if (options.signal?.aborted) controller.abort();
    controller.signal.throwIfAborted();
    const resp = await doFetch(url, { method: 'GET', headers, signal: controller.signal });
    if (!resp.ok) {
      const kind = kindForStatus(resp.status);
      const found = options.providerText === 'omit' ? undefined : await errorDetail(resp);
      const detail = found !== undefined && options.scrub ? options.scrub(found) : found;
      throw new ProviderConnectionError(kind, describeConnectionError(kind, { ...context, url, pageOrigin: page.origin, detail }, resp.status), resp.status);
    }
    const text = await resp.text();
    try {
      return JSON.parse(text);
    } catch {
      throw new ProviderConnectionError('invalid-response', describeConnectionError('invalid-response', { ...context, url }));
    }
  } catch (err) {
    if (err instanceof ProviderConnectionError) throw err;
    if (options.signal?.aborted) throw new ProviderConnectionError('cancelled', describeConnectionError('cancelled', { ...context, url }));
    if (timedOut) throw new ProviderConnectionError('timeout', describeConnectionError('timeout', { ...context, url }));
    // fetch says only "Failed to fetch" for a server that is down and for a
    // CORS refusal alike; the message covers both.
    throw new ProviderConnectionError('network', describeConnectionError('network', { ...context, url, pageOrigin: page.origin }));
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', onAbort);
  }
}

/** Where a provider is asked for its models, and how. */
interface ListTarget {
  type: ProviderType;
  serverKind?: LocalServerKind;
  endpoints: ProviderEndpoints;
  headers: Record<string, string>;
  /** The headers for Ollama's own list, which is outside the API base: for a record, without the key. */
  tagsHeaders?: Record<string, string>;
}

async function listFrom(target: ListTarget, options: ListModelsOptions): Promise<ModelInfo[]> {
  const { endpoints, headers } = target;
  const context = { type: target.type, serverKind: target.serverKind };

  if (target.type === 'anthropic') {
    const all: ModelInfo[] = [];
    const seen = new Set<string>();
    let after: string | null = null;
    for (let page = 0; page < MAX_PAGES; page++) {
      const url = new URL(endpoints.models);
      url.searchParams.set('limit', '1000');
      if (after) url.searchParams.set('after_id', after);
      const body = await getJson(url.toString(), headers, context, options);
      const models = readModelList(body);
      if (!models) throw new ProviderConnectionError('invalid-response', describeConnectionError('invalid-response', { ...context, url: endpoints.models }));
      for (const model of models) {
        if (!seen.has(model.id)) {
          seen.add(model.id);
          all.push(model);
        }
      }
      const record = body as Record<string, unknown>;
      const next = typeof record.last_id === 'string' ? record.last_id : null;
      if (record.has_more !== true || !next || next === after) break;
      after = next;
    }
    return sortModels(all);
  }

  try {
    const body = await getJson(endpoints.models, headers, context, options);
    const models = readModelList(body) ?? readOllamaTags(body);
    if (!models) throw new ProviderConnectionError('invalid-response', describeConnectionError('invalid-response', { ...context, url: endpoints.models }));
    return sortModels(models);
  } catch (err) {
    const first = err instanceof ProviderConnectionError ? err : null;
    const retryable = first && !['cancelled', 'mixed-content', 'auth'].includes(first.kind);
    if (!endpoints.ollamaTags || !retryable) throw err;
    try {
      const body = await getJson(endpoints.ollamaTags, target.tagsHeaders ?? headers, context, options);
      const models = readOllamaTags(body);
      if (!models) throw first;
      return sortModels(models);
    } catch {
      throw first;
    }
  }
}

/**
 * Every model the provider offers, newest first where it says when.
 *
 * - OpenAI and OpenAI-compatible servers: `GET {base}/models`.
 * - Anthropic: `GET {base}/models`, following `has_more`/`last_id` pages.
 * - Local and custom servers: when `/models` fails, Ollama's `/api/tags`
 *   is tried before giving up, and the first error is the one reported.
 *
 * The list is returned whole; `chatModels` decides what to offer.
 */
export async function listModels(
  provider: Pick<ProviderConfig, 'type' | 'apiKey' | 'baseUrl' | 'orgId' | 'serverKind'>,
  options: ListModelsOptions = {},
): Promise<ModelInfo[]> {
  return listFrom(
    { type: provider.type, serverKind: provider.serverKind, endpoints: providerEndpoints(provider), headers: providerHeaders(provider) },
    options,
  );
}

/**
 * The same for a record and its key: what the providers origin runs, with the key it holds. The provider's own words are OMITTED from an
 * error's message unless the caller says `providerText: 'include'` (a page of the holder's own, whose DOM no app can read); then they are
 * scrubbed of the key first.
 */
export async function listRecordModels(
  record: Pick<ProviderRecord, 'dialect' | 'baseUrl' | 'auth' | 'extraHeaders' | 'kind' | 'preset' | 'serverKind'>,
  key: string,
  options: ListModelsOptions = {},
): Promise<ModelInfo[]> {
  const type = providerTypeOf(record);
  const providerText = options.providerText ?? 'omit';
  const endpoints = providerEndpoints({ type, baseUrl: record.baseUrl, serverKind: record.serverKind });
  // Ollama's own list is at the ORIGIN, outside the API base the key is for: only a server on this computer or network is asked, and
  // without the key (Ollama takes none).
  if (record.kind !== 'local-server') delete endpoints.ollamaTags;
  return listFrom(
    { type, serverKind: record.serverKind, endpoints, headers: recordHeaders(record, key), tagsHeaders: recordHeaders(record, '') },
    { ...options, providerText, scrub: providerText === 'include' ? (text) => redactSecret(text, key) : undefined },
  );
}
