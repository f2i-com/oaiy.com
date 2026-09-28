/**
 * Which service-driven nodes the palette offers, and what a node on the
 * canvas says when the service it uses is not there.
 *
 * In OAIY's window the palette shows a media node (Image Gen, Music Gen, 3D
 * Model, …) only when something installed can run it: an OAIY engine model of
 * its kind whose files are here, an installed desktop service (a Python rig,
 * Ollama …) tagged for it, or one of the user's own services tagged for it.
 * No music model and no music service → no Music Gen in the palette. It asks
 * only once the desktop's list has answered, so nothing is hidden on a guess,
 * and it follows the list as it changes (installing a model shows its node).
 *
 * Outside OAIY (the plain web editor) the palette keeps its own rules: the
 * typed AI nodes stay folded into Service Call + a preset, as before, and the
 * new service nodes appear only when a service is tagged for them.
 *
 * A node already in a flow is never hidden or dropped: it is registered
 * either way, opens and compiles, and says what is missing on the node.
 */
import type { CustomService } from 'oaiy-core/modules/core-service/examples';
import {
  NODE_ENGINE_KIND,
  isOaiyServiceId,
  resolveService,
  servesNode,
} from 'oaiy-core/modules/core-service/contract';

/** Nodes that run on a service (their palette entry depends on one being there). */
export const SERVICE_NODE_TYPES: ReadonlySet<string> = new Set(Object.keys(NODE_ENGINE_KIND));

/**
 * Typed nodes the plain web editor keeps out of its palette (Service Call +
 * a preset stands in for them there). In OAIY they show when they can run.
 */
export const WEB_TYPED_NODE_IDS: ReadonlySet<string> = new Set([
  'ai_llm',
  'image_gen',
  'video_gen',
  'text_to_speech',
  'music_gen',
  'speech_to_text',
]);

/** The service a node uses when none is picked: the new nodes default to the engine. */
export const DEFAULT_SERVICE: Readonly<Record<string, string>> = {
  sound_effect: 'engine:sound',
  model_3d: 'engine:model3d',
  background_removal: 'engine:background',
  image_upscale: 'engine:upscale',
};

/** `data.service` values that are the node's own request code, not a preset. */
const LEGACY_SERVICE_VALUES: ReadonlySet<string> = new Set(['ace-step', 'heartmula', 'qwen3-tts', 'chatterbox-tts']);

const KIND_LABEL: Readonly<Record<string, string>> = {
  llm: 'language model',
  image: 'picture model',
  video: 'video model',
  speech: 'speech model',
  music: 'music model',
  sound: 'sound-effect model',
  model3d: '3D model',
  background: 'background-removal model',
  upscale: 'upscaler',
  transcription: 'transcription',
};

export interface AvailabilityEnv {
  /** In OAIY's window (`window.__OAIY_DESKTOP__`). */
  inOaiy: boolean;
  /** The desktop's list has answered at least once. */
  loaded: boolean;
  /** The user's own services (`oaiy.customServices`). */
  custom: CustomService[];
  /** The desktop's services and OAIY's engine (`oaiy.desktopServices`). */
  desktop: CustomService[];
}

/** Whether the palette offers node type `id`. */
export function paletteShowsNode(id: string, env: AvailabilityEnv): boolean {
  if (!SERVICE_NODE_TYPES.has(id)) return true;
  const all = [...env.custom, ...env.desktop];
  if (env.inOaiy) return env.loaded && servesNode(all, id as never);
  if (WEB_TYPED_NODE_IDS.has(id)) return false;
  return servesNode(all, id as never);
}

/** What `engine:<kind>[:<model>]` / `voice:*` names, for a message. */
function describeOaiyId(id: string): { kind: string; model: string } {
  const [, kind = '', ...rest] = id.split(':');
  return { kind: id.startsWith('voice:') ? 'transcription' : kind, model: rest.join(':') };
}

/**
 * What a node on the canvas says about its service, or null when all is well
 * (or when the desktop has not answered yet, so nothing is claimed missing on
 * a guess). Covers the service-driven nodes and Service Call.
 */
export function nodeNotice(nodeType: string, data: Record<string, unknown>, env: AvailabilityEnv): string | null {
  if (!SERVICE_NODE_TYPES.has(nodeType) && nodeType !== 'service_call') return null;
  const picked = typeof data.service === 'string' ? data.service.trim() : '';
  const id = picked || DEFAULT_SERVICE[nodeType] || '';
  if (!id || LEGACY_SERVICE_VALUES.has(id)) return null;
  if (resolveService(id, { custom: env.custom, desktop: env.desktop })) return null;
  if (isOaiyServiceId(id)) {
    if (!env.inOaiy) return 'This uses OAIY\'s own engine, which flows reach in the OAIY app. Open the flow there, or pick another service.';
    if (!env.loaded) return null;
    if (id.startsWith('voice:')) {
      return 'OAIY Voice is not installed, so nothing here writes out recordings. Install it in OAIY → Services, or pick another service.';
    }
    const { kind, model } = describeOaiyId(id);
    const what = KIND_LABEL[kind] || `${kind} model`;
    return model
      ? `The ${what} “${model}” is not installed in OAIY's engine (or its files are missing). Add it in OAIY → Engines, or pick another service.`
      : `OAIY's engine has no ${what} installed. Add one in OAIY → Engines, or pick another service (your own Python rigs are in OAIY → Services).`;
  }
  if (id.startsWith('companion:')) {
    if (!env.loaded) return null;
    return `The service “${id.slice('companion:'.length)}” is not installed in OAIY. Install it in OAIY → Services, or pick another service.`;
  }
  return `The service “${id}” is not in this editor's services. Add it in Settings → Services, or pick another service.`;
}

/**
 * The service a node dropped from the palette starts with (in OAIY): the
 * engine's default for its kind, else the first service offered for it, else
 * none (the node's own default).
 */
export function initialServiceFor(nodeType: string, env: AvailabilityEnv): string | null {
  if (!SERVICE_NODE_TYPES.has(nodeType)) return null;
  const all = [...env.desktop, ...env.custom].filter((s) => s.nodeTypes?.includes(nodeType as never));
  const engine = all.filter((s) => s.group === 'engine');
  // `engine:<kind>` follows the default set in OAIY → Engines.
  if (engine.length > 0 && engine[0].kind && engine[0].kind !== 'transcription') return `engine:${engine[0].kind}`;
  const pick = engine[0] ?? all[0];
  return pick ? pick.id : null;
}

/**
 * The node a service dropped from the palette becomes: the typed node it is
 * offered for first (Music Gen for an engine music model), when it runs as
 * a contract and that node is registered; otherwise Service Call.
 */
export function nodeTypeForService(service: CustomService | null, isRegistered: (type: string) => boolean): string {
  const typed = service?.nodeTypes?.find((t) => t !== 'service_call');
  if (service && typed && (service.group === 'engine' || service.output) && isRegistered(typed)) return typed;
  return 'service_call';
}
