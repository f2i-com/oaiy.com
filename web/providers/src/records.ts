/**
 * What a provider record may be, checked (design 4.1): the one place a record is made, whether by the add form, the edit form or
 * an import. A record that passes is what `buildRequestUrl` and `recordHeaders` will accept, and nothing else is saved.
 */
import { isLoopbackHost } from '@oaiy/shared/providers/errors';
import { normalizeApiBase } from '@oaiy/shared/providers/endpoints';
import { EXTRA_HEADER_NAMES, PROVIDER_CAPS, type Dialect, type ExtraHeader, type ProviderCap, type ProviderKind, type ProviderRecord, type ServerKind } from '@oaiy/shared/providers/types';

/** What a form hands over: text and choices, not yet a record. Numbers may still be text. */
export interface RecordInput {
  id?: string;
  name?: unknown;
  dialect?: unknown;
  baseUrl?: unknown;
  model?: unknown;
  caps?: unknown;
  kind?: unknown;
  preset?: unknown;
  serverKind?: unknown;
  extraHeaders?: unknown;
  contextTokens?: unknown;
  parallelAgents?: unknown;
}

export type Validation = { ok: true; record: ProviderRecord } | { ok: false; errors: Record<string, string> };

const CONTROL = /[\u0000-\u001f\u007f]/;
const SERVER_KINDS: readonly ServerKind[] = ['ollama', 'lmstudio', 'oaiy', 'other'];
const CANONICAL_HEADER: Record<string, string> = { 'openai-organization': 'OpenAI-Organization', 'http-referer': 'HTTP-Referer', 'x-title': 'X-Title', 'anthropic-beta': 'anthropic-beta' };

/** A new record id: `p_` and twelve hex digits. */
export function newRecordId(random: (n: number) => Uint8Array): string {
  return `p_${Array.from(random(6), (b) => b.toString(16).padStart(2, '0')).join('')}`;
}

const text = (value: unknown): string | null => (typeof value === 'string' ? value.trim() : null);

function wholeNumber(value: unknown, min: number, max: number): number | null | undefined {
  if (value === undefined || value === null || value === '') return undefined;
  const n = typeof value === 'number' ? value : typeof value === 'string' && /^\d+$/.test(value.trim()) ? Number(value.trim()) : NaN;
  return Number.isInteger(n) && n >= min && n <= max ? n : null;
}

/**
 * The record for `input`, or what is wrong with it, by field. `id` is the record's own (a new one, or the one being edited).
 * `auth` is not asked: it follows from the dialect. `via` is always `broker`: a gateway's provider is a mirror and is never saved
 * here, and the on-device engine's record is made by the engine, not by a form.
 */
export function validateRecord(input: RecordInput, id: string): Validation {
  const errors: Record<string, string> = {};

  const name = text(input.name);
  if (name === null || name === '') errors.name = 'Give it a name.';
  else if (name.length > 80 || CONTROL.test(name)) errors.name = 'A name is up to 80 characters, on one line.';

  const dialect = input.dialect;
  if (dialect !== 'openai' && dialect !== 'anthropic') errors.dialect = 'Choose OpenAI-compatible or Anthropic.';

  const kind = input.kind;
  if (kind !== 'external' && kind !== 'local-server') errors.kind = 'Choose a service on the internet or a server on this computer.';

  const baseUrl = normalizeApiBase(input.baseUrl);
  if (baseUrl === null) errors.baseUrl = 'That is not an address a provider can be called at: use https://… (or http://… for a server on this computer), with no user name, no ?query and no #fragment.';
  else if (kind === 'external' && new URL(baseUrl).protocol === 'http:' && !isLoopbackHost(new URL(baseUrl).hostname)) {
    errors.baseUrl = 'A service on the internet has to be https: a key sent over plain http can be read on the way.';
  }

  const model = input.model === undefined || input.model === null || input.model === '' ? undefined : text(input.model);
  if (model === null || (model !== undefined && (model.length > 200 || CONTROL.test(model)))) errors.model = 'A model name is up to 200 characters, on one line.';

  let caps: ProviderCap[] = ['chat'];
  if (input.caps !== undefined) {
    if (!Array.isArray(input.caps) || input.caps.some((c) => !PROVIDER_CAPS.includes(c as ProviderCap))) errors.caps = 'Those are not things a provider can do.';
    else caps = PROVIDER_CAPS.filter((c) => (input.caps as string[]).includes(c));
  }

  let serverKind: ServerKind | undefined;
  if (input.serverKind !== undefined && input.serverKind !== null && input.serverKind !== '') {
    if (kind !== 'local-server' || !SERVER_KINDS.includes(input.serverKind as ServerKind)) errors.serverKind = 'That is not a kind of server.';
    else serverKind = input.serverKind as ServerKind;
  }

  let preset: string | undefined;
  if (input.preset !== undefined && input.preset !== null && input.preset !== '') {
    if (typeof input.preset !== 'string' || !/^[a-z0-9-]{1,40}$/.test(input.preset)) errors.preset = 'That is not a preset.';
    else preset = input.preset;
  }

  const extraHeaders: ExtraHeader[] = [];
  if (input.extraHeaders !== undefined && input.extraHeaders !== null) {
    if (!Array.isArray(input.extraHeaders)) errors.extraHeaders = 'Extra headers are a list.';
    else {
      const seen = new Set<string>();
      for (const item of input.extraHeaders as unknown[]) {
        const header = item as { name?: unknown; value?: unknown } | null;
        const headerName = typeof header?.name === 'string' ? header.name.trim().toLowerCase() : '';
        const value = text(header?.value);
        if (value === '' && headerName !== '') continue; // an empty row is nothing
        if (!EXTRA_HEADER_NAMES.includes(headerName) || value === null || value.length > 512 || CONTROL.test(value) || seen.has(headerName)) {
          errors.extraHeaders = `Only ${EXTRA_HEADER_NAMES.map((n) => CANONICAL_HEADER[n]).join(', ')} can be added, once each, with a short value.`;
          break;
        }
        seen.add(headerName);
        extraHeaders.push({ name: CANONICAL_HEADER[headerName], value });
      }
    }
  }

  const contextTokens = wholeNumber(input.contextTokens, 1024, 10_000_000);
  if (contextTokens === null) errors.contextTokens = 'The context window is a whole number of tokens, 1024 or more.';
  const parallelAgents = wholeNumber(input.parallelAgents, 1, 16);
  if (parallelAgents === null) errors.parallelAgents = 'How many agents may use it at once is a whole number from 1 to 16.';

  if (Object.keys(errors).length > 0 || baseUrl === null || name === null) return { ok: false, errors };

  const record: ProviderRecord = {
    v: 1,
    id,
    name,
    dialect: dialect as Dialect,
    baseUrl,
    auth: dialect === 'anthropic' ? 'x-api-key' : 'bearer',
    caps,
    kind: kind as ProviderKind,
    via: 'broker',
  };
  if (extraHeaders.length > 0) record.extraHeaders = extraHeaders;
  if (model !== undefined && model !== null) record.model = model;
  if (preset !== undefined) record.preset = preset;
  if (serverKind !== undefined) record.serverKind = serverKind;
  if (contextTokens !== undefined && contextTokens !== null) record.contextTokens = contextTokens;
  if (parallelAgents !== undefined && parallelAgents !== null) record.parallelAgents = parallelAgents;
  return { ok: true, record };
}

/**
 * Whether saving `next` over `prev` moves where the key goes: a different address or a different dialect. A keyed record can only
 * be moved by someone who types the key again (design 6, threat 2), so a stolen form cannot point an old key at a new place.
 */
export function movesKey(prev: ProviderRecord, next: ProviderRecord): boolean {
  return prev.baseUrl !== next.baseUrl || prev.dialect !== next.dialect;
}

/** The host a key would be sent to, for a summary a page may see (`api.openai.com`, `localhost:11434`). */
export function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).host;
  } catch {
    return '';
  }
}
