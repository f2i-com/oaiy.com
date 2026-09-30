/*
 * Where a provider's endpoints are, and how a request to one is made.
 *
 * The first half (`LOCAL_SERVERS` to `providerHeaders`) is the Agent's own, moved here unchanged: adapted from softn.com
 * (apps/softn-studio/src/lib/providerConnection.ts), Copyright f2i-com, licensed under the Apache License, Version 2.0.
 * app/src/agent/providers/providerConnection.ts re-exports it.
 *
 * The second half is the record's (design 3.2 and 4.1): a `ProviderRecord` has one API base, and a request to it is that
 * base plus ONE path of a fixed list, with the query and the headers built here, never taken from a page. Everything a
 * page may choose is checked by `buildRequestUrl` and `recordHeaders`, and refused (`RequestRefused`) when it is not one of
 * the few things it may choose.
 */
import type { Dialect, LocalServerKind, ProviderConfig, ProviderRecord, ProviderType } from './types';
import { EXTRA_HEADER_NAMES } from './types';

/** Where each local server listens out of the box. */
export const LOCAL_SERVERS: Record<LocalServerKind, { label: string; baseUrl: string; help: string }> = {
  ollama: {
    label: 'Ollama',
    baseUrl: 'http://localhost:11434',
    help: 'Start Ollama and pull at least one model. Ollama has to allow this page: set OLLAMA_ORIGINS to include this site before starting it.',
  },
  lmstudio: {
    label: 'LM Studio',
    baseUrl: 'http://localhost:1234',
    help: 'Load a model, then start the server in LM Studio’s Developer tab with “Enable CORS” switched on.',
  },
  oaiy: {
    label: 'OAIY',
    baseUrl: 'http://127.0.0.1:8080',
    help: 'OAIY serves chat, images and video. Without an API key it answers only the pages in gateway.cors_origins in its config: add this site there (or set an API key and enter it here). Settings → Images and video uses the same server.',
  },
  other: {
    label: 'Another server',
    baseUrl: 'http://localhost:8080',
    help: 'Any server that speaks the OpenAI chat completions API. It has to allow requests from this page (CORS).',
  },
};

export const CLOUD_BASE: Partial<Record<ProviderType, string>> = {
  anthropic: 'https://api.anthropic.com',
  openai: 'https://api.openai.com',
};

/** The address a provider of this type uses when none is given. */
export function defaultBaseUrl(type: ProviderType, serverKind?: LocalServerKind): string {
  if (type === 'local') return LOCAL_SERVERS[serverKind ?? 'ollama'].baseUrl;
  if (type === 'custom') return LOCAL_SERVERS.ollama.baseUrl;
  return CLOUD_BASE[type] ?? '';
}

export interface ProviderEndpoints {
  /** Where replies are requested: chat completions, or Anthropic's messages. */
  chat: string;
  /** The OpenAI-style (or Anthropic) model list. */
  models: string;
  /** Ollama's own model list, tried when the OpenAI-style one fails. Local and custom servers only. */
  ollamaTags?: string;
}

/**
 * The endpoints for a provider, from whatever address it was saved with.
 *
 * Providers saved before this file existed stored the full completion URL
 * (`…/v1/chat/completions`, `…/v1/messages`); the setup now asks for the
 * server's address (`http://localhost:11434`) or an API base (`…/v1`).
 * All three shapes work, so nothing stored has to be migrated.
 */
export function providerEndpoints(provider: Pick<ProviderConfig, 'type' | 'baseUrl' | 'serverKind'>): ProviderEndpoints {
  const raw = provider.baseUrl?.trim() || defaultBaseUrl(provider.type, provider.serverKind);
  const isAnthropic = provider.type === 'anthropic';
  const tail = isAnthropic ? '/messages' : '/chat/completions';
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    // An unparseable address is reported by fetch; keep it recognisable.
    return { chat: raw, models: raw };
  }
  let path = url.pathname.replace(/\/+$/, '');
  if (path.endsWith(tail)) path = path.slice(0, -tail.length);
  else if (path.endsWith('/models')) path = path.slice(0, -'/models'.length);
  if (path === '') path = '/v1';
  const base = `${url.origin}${path}`;
  const endpoints: ProviderEndpoints = {
    chat: `${base}${tail}${url.search}`,
    models: `${base}/models${url.search}`,
  };
  if (provider.type === 'local' || provider.type === 'custom') endpoints.ollamaTags = `${url.origin}/api/tags`;
  return endpoints;
}

/** Headers for a provider: its key in the form it expects, plus the fixed ones. */
export function providerHeaders(provider: Pick<ProviderConfig, 'type' | 'apiKey' | 'orgId'>, json = false): Record<string, string> {
  const headers: Record<string, string> = json ? { 'Content-Type': 'application/json' } : {};
  if (provider.type === 'anthropic') {
    headers['x-api-key'] = provider.apiKey;
    headers['anthropic-version'] = '2023-06-01';
    // Anthropic refuses browser calls without this; Studio has no server of
    // its own, so the browser is the only place a request can come from.
    headers['anthropic-dangerous-direct-browser-access'] = 'true';
    return headers;
  }
  if (provider.apiKey) headers.Authorization = `Bearer ${provider.apiKey}`;
  if (provider.type === 'openai' && provider.orgId?.trim()) headers['OpenAI-Organization'] = provider.orgId.trim();
  return headers;
}

// --- The record's API base and the paths a page may ask for --------------------

/**
 * The paths of a provider's API that a page may ask the holder for, and what each may be. Fixed: no path outside this
 * list is ever sent, whatever the page says (design 3.2). `/models` is the same for both dialects.
 */
export const API_PATHS: Readonly<Record<string, { dialects: readonly Dialect[]; method: 'GET' | 'POST' }>> = Object.freeze({
  '/chat/completions': { dialects: ['openai'], method: 'POST' },
  '/messages': { dialects: ['anthropic'], method: 'POST' },
  '/models': { dialects: ['openai', 'anthropic'], method: 'GET' },
  '/images/generations': { dialects: ['openai'], method: 'POST' },
  '/audio/speech': { dialects: ['openai'], method: 'POST' },
  '/audio/transcriptions': { dialects: ['openai'], method: 'POST' },
  '/embeddings': { dialects: ['openai'], method: 'POST' },
});

const API_PATH_SET: ReadonlySet<string> = new Set(Object.keys(API_PATHS));

export type RefusalCode = 'bad-path' | 'bad-method' | 'bad-query' | 'bad-base' | 'bad-url' | 'bad-headers';

/** A request that is not one a page may make. `code` is what the page is told; `message` says why. */
export class RequestRefused extends Error {
  readonly code: RefusalCode;

  constructor(code: RefusalCode, message: string) {
    super(message);
    this.name = 'RequestRefused';
    this.code = code;
  }
}

/** What may never appear in a path a page sends, even before it is compared with the list: `//`, `..`, `@`, `\`, `?`, `#`, any `%` (an encoded form of them), a space or a control character. */
const PATH_FORBIDDEN = /\/\/|\.\.|[@\\?#%\s\u0000-\u001f\u007f]/;

/**
 * The path a page asked for, if it is exactly one of `API_PATHS` for this dialect and `method` is the one it takes.
 * Anything else is refused: there is no pattern, no prefix and no normalising.
 */
export function checkApiPath(dialect: Dialect, path: unknown, method: unknown): string {
  if (typeof path !== 'string' || path.length === 0 || path.length > 64 || PATH_FORBIDDEN.test(path) || !path.startsWith('/')) {
    throw new RequestRefused('bad-path', 'That is not a path this provider can be asked for.');
  }
  if (!API_PATH_SET.has(path)) throw new RequestRefused('bad-path', 'That is not a path this provider can be asked for.');
  const entry = API_PATHS[path];
  if (!entry.dialects.includes(dialect)) throw new RequestRefused('bad-path', `${path} is not part of this provider's API.`);
  if (method !== entry.method) throw new RequestRefused('bad-method', `${path} takes ${entry.method}.`);
  return path;
}

/**
 * A record's API base from what a person typed or an older record held: an http(s) address with no user name, password,
 * query or fragment, no trailing slash, and its version segment (`/v1` when the address gave no path at all, as
 * `providerEndpoints` reads a bare server address). A pasted `…/chat/completions`, `…/messages` or `…/models` is cut off.
 * Null when it cannot be one.
 */
export function normalizeApiBase(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  // A parser quietly turns `\` into `/`, `%2e%2e` into `..` and `..` into a shorter path; what a person typed is not what it
  // becomes, so an address that needs any of that is refused rather than guessed at.
  if (/[\\%\s\u0000-\u001f\u007f]/.test(text) || /\/\.{1,2}(?:[/?#]|$)/.test(text)) return null;
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  if (url.username || url.password || url.search || url.hash) return null;
  if (!url.hostname) return null;
  let path = url.pathname.replace(/\/+$/, '');
  for (const tail of ['/chat/completions', '/messages', '/models']) {
    if (path.endsWith(tail)) {
      path = path.slice(0, -tail.length);
      break;
    }
  }
  if (path === '') path = '/v1';
  if (PATH_FORBIDDEN.test(path)) return null;
  return `${url.origin}${path}`;
}

/** `[name, value]` pairs a page may put in the query: they are checked, then written by `URLSearchParams`, so nothing a page sends is ever part of the path. */
export type QueryPairs = ReadonlyArray<readonly [string, string]>;

const QUERY_NAME = /^[A-Za-z0-9_.-]{1,64}$/;
const CONTROL = /[\u0000-\u001f\u007f]/;

function checkQuery(query: unknown): QueryPairs {
  if (query === undefined) return [];
  if (!Array.isArray(query) || query.length > 16) throw new RequestRefused('bad-query', 'The query is not a short list of name and value pairs.');
  const pairs: Array<readonly [string, string]> = [];
  for (const pair of query) {
    if (!Array.isArray(pair) || pair.length !== 2) throw new RequestRefused('bad-query', 'The query is not a short list of name and value pairs.');
    const [name, value] = pair as unknown[];
    if (typeof name !== 'string' || !QUERY_NAME.test(name) || typeof value !== 'string' || value.length > 512 || CONTROL.test(value)) {
      throw new RequestRefused('bad-query', 'A query parameter is not allowed.');
    }
    pairs.push([name, value]);
  }
  return pairs;
}

/**
 * The address a request goes to: the record's base and the path, then the query. The base is the record's, never the
 * page's. The result is parsed and must have the record's origin and exactly its path, with no user name, password or
 * fragment, or the request is refused; nothing a page sends can move it to another host, another path or another scheme.
 */
export function buildRequestUrl(record: Pick<ProviderRecord, 'dialect' | 'baseUrl'>, path: unknown, method: unknown, query?: unknown): string {
  const api = checkApiPath(record.dialect, path, method);
  const pairs = checkQuery(query);
  const base = normalizeApiBase(record.baseUrl);
  if (base === null || base !== record.baseUrl) throw new RequestRefused('bad-base', 'The provider’s address is not one that can be called.');
  const baseUrl = new URL(base);
  const written = new URLSearchParams();
  for (const [name, value] of pairs) written.append(name, value);
  const search = written.toString();
  const target = new URL(`${base}${api}${search ? `?${search}` : ''}`);
  if (target.origin !== baseUrl.origin || target.pathname !== `${baseUrl.pathname}${api}` || target.username || target.password || target.hash || target.protocol !== baseUrl.protocol) {
    throw new RequestRefused('bad-url', 'The request would not go to the provider’s own address.');
  }
  return target.toString();
}

const HEADER_VALUE_MAX = 512;

function headerValueOk(value: unknown): value is string {
  return typeof value === 'string' && value.length <= HEADER_VALUE_MAX && !CONTROL.test(value);
}

/**
 * The headers of a request to a record's provider. A page may choose only `content-type` and `accept` (anything else it
 * sends is left out, silently: it cannot add `Authorization`, `x-api-key` or any other header). The key, the dialect's
 * fixed headers and the record's own extra headers are put on here.
 */
export function recordHeaders(
  record: Pick<ProviderRecord, 'dialect' | 'auth' | 'extraHeaders'>,
  key: string,
  incoming: Readonly<Record<string, unknown>> | ReadonlyArray<readonly [string, unknown]> = {},
): Record<string, string> {
  const headers: Record<string, string> = {};
  const entries = Array.isArray(incoming) ? incoming : Object.entries(incoming);
  for (const [name, value] of entries as ReadonlyArray<readonly [unknown, unknown]>) {
    if (typeof name !== 'string') continue;
    const lower = name.toLowerCase();
    if (lower !== 'content-type' && lower !== 'accept') continue;
    if (!headerValueOk(value)) throw new RequestRefused('bad-headers', `The ${lower} header is not allowed.`);
    headers[lower] = value;
  }
  if (record.dialect === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    // Anthropic refuses a browser's call without this.
    headers['anthropic-dangerous-direct-browser-access'] = 'true';
  }
  if (key) {
    if (record.auth === 'bearer') headers.authorization = `Bearer ${key}`;
    else if (record.auth === 'x-api-key') headers['x-api-key'] = key;
  }
  for (const extra of record.extraHeaders ?? []) {
    if (!EXTRA_HEADER_NAMES.includes(extra.name.toLowerCase()) || !headerValueOk(extra.value) || !extra.value.trim()) continue;
    headers[extra.name.toLowerCase()] = extra.value.trim();
  }
  return headers;
}
