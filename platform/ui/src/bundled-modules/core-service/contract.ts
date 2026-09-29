/**
 * Service lookup shared by every compiler, and the "contract" services.
 *
 * Three lists hold services, read synchronously from localStorage (the
 * compilers run in the renderer):
 *   - `oaiy.customServices`  — the user's own, from Settings → Services;
 *   - `oaiy.desktopServices` — what OAIY Desktop offers, published by
 *     lib/desktopServices.ts: its services (Python rigs, ComfyUI …,
 *     ids `companion:*`) and OAIY's engine's models and OAIY Voice (ids
 *     `engine:<kind>:<model>`, `voice:*`);
 *   - the built-in examples, for flows saved before the Services registry.
 *
 * A service that declares what it makes (`output`), a job to follow (`job`)
 * or a recording body (`requestFormat`) is a CONTRACT service: the typed nodes
 * (image_gen, music_gen, …) and Service Call run it through
 * `runContract` (contractRuntime.ts) instead of their own request code, so a
 * Python rig that declares the same fields plugs into those nodes just as the
 * engine's models do.
 *
 * `engine:<kind>` (no model) names the engine's default model of that kind.
 * An `engine:*` / `voice:*` id that is not listed where the flow is compiled
 * (the CLI has no localStorage) is looked up again when the flow runs.
 */
import { BUILT_IN_SERVICES, type CustomService, type ServiceNodeTag, type ServiceOutputKind } from './examples';

export const CUSTOM_SERVICE_STORAGE_KEY = 'oaiy.customServices';
export const DESKTOP_SERVICE_STORAGE_KEY = 'oaiy.desktopServices';

export function readServiceKey(key: string): CustomService[] {
  try {
    if (typeof localStorage === 'undefined') return [];
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CustomService[]) : [];
  } catch {
    return [];
  }
}

/** An id naming one of OAIY's own services (an engine model, OAIY Voice). */
export function isOaiyServiceId(id: string): boolean {
  return /^(engine|voice):[a-z0-9]/i.test(id);
}

/** `engine:<kind>` → the kind (`engine:music` → `music`); null for any other id. */
export function engineKindOf(id: string): string | null {
  const m = /^engine:([a-z0-9]+)$/i.exec(id);
  return m ? m[1].toLowerCase() : null;
}

/** The engine's default service of `kind` in `list` (else its first). */
export function engineDefault(list: CustomService[], kind: string): CustomService | null {
  const of = list.filter((s) => s.group === 'engine' && s.kind === kind);
  return of.find((s) => s.default) ?? of[0] ?? null;
}

/**
 * The service `id` names. The user's own shadow same-id built-ins; OAIY's
 * (`companion:*`, `engine:*`, `voice:*`) never collide with them.
 */
export function resolveService(
  id: string,
  lists: { custom?: CustomService[]; desktop?: CustomService[] } = {},
): CustomService | null {
  if (!id) return null;
  const custom = lists.custom ?? readServiceKey(CUSTOM_SERVICE_STORAGE_KEY);
  const desktop = lists.desktop ?? readServiceKey(DESKTOP_SERVICE_STORAGE_KEY);
  const kind = engineKindOf(id);
  if (kind) return engineDefault(desktop, kind);
  return [...custom, ...desktop, ...BUILT_IN_SERVICES].find((s) => s.id === id) ?? null;
}

/** Whether `s` runs as a contract (see the module docs). */
export function isContractService(s: CustomService | null | undefined): s is CustomService {
  return !!s && !!(s.output || s.job || s.requestFormat === 'wav16k');
}

/**
 * What a compiler inlines for the node's service: the contract itself when it
 * is listed here, the id when it is OAIY's and not listed here (looked up when
 * the flow runs), null when the node should use its own request code.
 */
export function contractLiteral(presetId: string, preset: CustomService | null): string | null {
  if (isContractService(preset)) return JSON.stringify(preset);
  if (!preset && isOaiyServiceId(presetId)) return JSON.stringify(presetId);
  return null;
}

/** The engine's kind of model each node runs (for `engine:<kind>` and the palette). */
export const NODE_ENGINE_KIND: Readonly<Record<string, string>> = {
  ai_llm: 'llm',
  image_gen: 'image',
  video_gen: 'video',
  text_to_speech: 'speech',
  music_gen: 'music',
  speech_to_text: 'transcription',
  sound_effect: 'sound',
  model_3d: 'model3d',
  background_removal: 'background',
  image_upscale: 'upscale',
};

/** What each node makes, for a plain HTTP service picked on it (run as a contract). */
export const NODE_OUTPUT: Readonly<Record<string, ServiceOutputKind>> = {
  image_gen: 'image',
  video_gen: 'video',
  text_to_speech: 'audio',
  music_gen: 'audio',
  speech_to_text: 'text',
  sound_effect: 'audio',
  model_3d: 'model3d',
  background_removal: 'image',
  image_upscale: 'image',
};

/**
 * A node's service as the literal its compiler inlines: a listed contract; an
 * OAIY id to look up when the flow runs; or, given `output` (the nodes whose
 * own request code does not take presets), any other service run as a
 * contract making that. Null: the node's own request code runs.
 */
export function nodeContract(serviceId: string, preset: CustomService | null, output?: ServiceOutputKind): string | null {
  const literal = contractLiteral(serviceId, preset);
  if (literal) return literal;
  if (preset && output) return JSON.stringify({ ...preset, output });
  return null;
}

/** The model a contract call names: an engine entry's own; otherwise the node's, else the preset's. */
export function contractModel(preset: CustomService | null, nodeModel: unknown): string {
  if (preset?.group === 'engine') return preset.model ?? '';
  return (typeof nodeModel === 'string' && nodeModel) || preset?.model || '';
}

/**
 * A contract's body: `{{name}}` is the value as JSON — `null` when the node has
 * none (or it is empty) — and `{{nameRaw}}` the value as text, for use inside a
 * string. A field left null at the top level is not sent, so the server applies
 * its own default rather than being told "null".
 */
export function renderContractBody(template: string, vars: Record<string, unknown>): string {
  const rendered = (template || '{}').replace(/\{\{(\w+)\}\}/g, (_match, name: string) => {
    const raw = name.endsWith('Raw');
    const key = raw ? name.slice(0, -3) : name;
    const value = vars[key];
    if (raw) return value == null ? '' : typeof value === 'string' ? value : JSON.stringify(value);
    if (value === undefined || value === '' || (typeof value === 'number' && Number.isNaN(value))) return 'null';
    return JSON.stringify(value);
  });
  try {
    const parsed = JSON.parse(rendered);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      for (const k of Object.keys(parsed)) if (parsed[k] === null) delete parsed[k];
      return JSON.stringify(parsed);
    }
  } catch {
    // Not JSON (a form body, a hand-written template): sent as rendered.
  }
  return rendered;
}

/** Dot/bracket path into a JSON value (`data.0.b64_json`, `error.message`). */
export function extractPath(root: unknown, path: string | undefined | null): unknown {
  if (!path) return root;
  let cur: unknown = root;
  for (const part of path.split(/[.[\]]+/).filter(Boolean)) {
    if (cur === null || cur === undefined) return undefined;
    cur = (cur as Record<string, unknown>)[/^\d+$/.test(part) ? Number(part) : part];
  }
  return cur;
}

/** Whether any of `services` is offered for `nodeType`. */
export function servesNode(services: CustomService[], nodeType: ServiceNodeTag): boolean {
  return services.some((s) => s.nodeTypes?.includes(nodeType));
}
