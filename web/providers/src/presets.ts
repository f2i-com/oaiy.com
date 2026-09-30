/**
 * The presets (design 4.5): six places to point a provider at, as data. Each says where it is, how it talks, and what a person is
 * told about it (where a key comes from, whether browsers are known to be let in). The `browser` field is data too, so it can be
 * flipped without a release when a vendor stops answering pages (design 4.3, open question 14).
 *
 * The file is checked when it is read (`validatePresets`): a preset that is malformed, repeats an id, points at an address a record
 * cannot have, or names a header outside the allowed few, is a mistake in the file and stops the page from starting.
 */
import raw from './presets.json';
import { normalizeApiBase } from '@oaiy/shared/providers/endpoints';
import { EXTRA_HEADER_NAMES, type ProviderKind, type ServerKind } from '@oaiy/shared/providers/types';
import type { RecordInput } from './records';

export type BrowserState = 'ok' | 'unverified' | 'no' | 'needs-setup' | 'unknown';

export interface ServerKindPreset {
  id: ServerKind;
  label: string;
  baseUrl: string;
  help: string;
}

export interface Preset {
  id: string;
  label: string;
  dialect: 'openai' | 'anthropic';
  kind: Exclude<ProviderKind, 'browser-engine'>;
  baseUrl: string;
  editableBase?: boolean;
  browser: BrowserState;
  keyUrl?: string;
  keyHint: string;
  note: string;
  extraHeaders?: string[];
  serverKinds?: ServerKindPreset[];
}

const BROWSER_STATES: readonly BrowserState[] = ['ok', 'unverified', 'no', 'needs-setup', 'unknown'];
const SERVER_KINDS: readonly string[] = ['ollama', 'lmstudio', 'oaiy', 'other'];

/** The problems in a presets file, in words; empty when it is sound. */
export function validatePresets(data: unknown): string[] {
  const problems: string[] = [];
  const file = data as { version?: unknown; presets?: unknown } | null;
  if (!file || file.version !== 1 || !Array.isArray(file.presets)) return ['the presets file is not version 1 with a list of presets'];
  const seen = new Set<string>();
  for (const [i, p] of (file.presets as Array<Record<string, unknown>>).entries()) {
    const at = `preset ${i}${typeof p?.id === 'string' ? ` (${p.id})` : ''}`;
    if (typeof p !== 'object' || p === null) {
      problems.push(`${at} is not an object`);
      continue;
    }
    if (typeof p.id !== 'string' || !/^[a-z0-9-]{1,40}$/.test(p.id)) problems.push(`${at} has no plain id`);
    else if (seen.has(p.id)) problems.push(`${at} repeats an id`);
    else seen.add(p.id);
    if (typeof p.label !== 'string' || !p.label.trim() || p.label.length > 60) problems.push(`${at} has no label`);
    if (p.dialect !== 'openai' && p.dialect !== 'anthropic') problems.push(`${at} has no dialect`);
    if (p.kind !== 'external' && p.kind !== 'local-server') problems.push(`${at} has no kind`);
    if (typeof p.baseUrl !== 'string') problems.push(`${at} has no baseUrl`);
    else if (p.baseUrl === '' ? p.editableBase !== true : normalizeApiBase(p.baseUrl) !== p.baseUrl) problems.push(`${at} has a baseUrl a record cannot have (or an empty one that is not editable)`);
    if (typeof p.baseUrl === 'string' && p.kind === 'external' && p.baseUrl !== '' && !p.baseUrl.startsWith('https://')) problems.push(`${at} is a service on the internet at an address that is not https`);
    if (!BROWSER_STATES.includes(p.browser as BrowserState)) problems.push(`${at} has no browser state`);
    if (typeof p.keyHint !== 'string' || typeof p.note !== 'string') problems.push(`${at} has no key hint or note`);
    if (p.keyUrl !== undefined && (typeof p.keyUrl !== 'string' || !p.keyUrl.startsWith('https://'))) problems.push(`${at} has a key link that is not https`);
    if (p.extraHeaders !== undefined) {
      if (!Array.isArray(p.extraHeaders) || p.extraHeaders.some((h) => typeof h !== 'string' || !EXTRA_HEADER_NAMES.includes(h.toLowerCase()))) problems.push(`${at} names a header outside the allowed ones`);
    }
    if (p.serverKinds !== undefined) {
      if (p.kind !== 'local-server' || !Array.isArray(p.serverKinds) || p.serverKinds.length === 0) problems.push(`${at} has server kinds but is not a local server`);
      else {
        for (const s of p.serverKinds as Array<Record<string, unknown>>) {
          if (typeof s?.id !== 'string' || !SERVER_KINDS.includes(s.id) || typeof s.label !== 'string' || typeof s.help !== 'string' || typeof s.baseUrl !== 'string' || normalizeApiBase(s.baseUrl) !== s.baseUrl) {
            problems.push(`${at} has a server kind that is not sound`);
          }
        }
      }
    }
    if (p.kind === 'local-server' && p.serverKinds === undefined) problems.push(`${at} is a local server with no server kinds`);
  }
  return problems;
}

/** The presets, checked. Throws when the file is not sound. */
export function loadPresets(data: unknown = raw): Preset[] {
  const problems = validatePresets(data);
  if (problems.length > 0) throw new Error(`presets.json: ${problems.join('; ')}`);
  return (data as { presets: Preset[] }).presets;
}

/** A preset's text with the origin the server has to allow filled in. */
export function withOrigin(text: string, origin: string): string {
  return text.split('{origin}').join(origin);
}

/** What the add form starts from for a preset (and, for a server, one of its kinds). */
export function inputFromPreset(preset: Preset, serverKind?: ServerKind): RecordInput {
  const server = serverKind ? preset.serverKinds?.find((s) => s.id === serverKind) : undefined;
  const input: RecordInput = {
    name: server ? server.label : preset.label,
    dialect: preset.dialect,
    kind: preset.kind,
    baseUrl: server ? server.baseUrl : preset.baseUrl,
    preset: preset.id,
  };
  if (preset.kind === 'local-server') input.serverKind = server?.id ?? preset.serverKinds?.[0]?.id;
  return input;
}
