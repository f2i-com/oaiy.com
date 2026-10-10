/**
 * Images, video, speech, music and 3D models for the agent, from an OpenAI-spec
 * media service: OAIY (found through its discovery document), or any server with
 * `/images/generations`, `/videos` and `/audio/speech`. The page calls it directly, as it
 * calls the AI provider: it is the person's own service, set up in Settings,
 * so these requests are not behind the network gate.
 */
import { incognitoHeaderFor } from '../privacy';

export interface ImageModelInfo {
  id: string;
  default?: boolean;
  architecture?: string;
  /** Takes reference images (edits, several pictures combined). */
  edits?: boolean;
  maxReferences?: number;
  /** Width and height must be multiples of this. */
  sizeStep?: number;
  defaultSize?: string;
  negativePrompt?: boolean;
}

export interface VideoModelInfo {
  id: string;
  default?: boolean;
  family?: string;
  fps?: number;
  maxFrames?: number;
  maxSeconds?: number;
  /** The longest side, in pixels. */
  maxSide?: number;
  /** Animates a start image. */
  startImage?: boolean;
  /** Speaks `say` in a saved voice with the picture, lips in sync (OAIY with ID-LoRA). */
  lipSync?: boolean;
}

export interface SpeechModelInfo {
  id: string;
  default?: boolean;
  /** Speaks in a voice described in words (`instructions`). */
  describedVoices?: boolean;
  /** Speaks in voices saved on the server. */
  savedVoices?: boolean;
  /** What speaks: OAIY's `qwen3-tts` or `breeze-tts-2`. */
  engine?: string;
  /** The terms of the model's weights and what it makes, when OAIY says. */
  license?: string;
  sampleRate?: number;
}

export interface MusicModelInfo {
  id: string;
  default?: boolean;
  maxSeconds?: number;
  sampleRate?: number;
  channels?: number;
}

/** A sound effects model (OAIY's MOSS-SoundEffect): a description in, a short sound out. */
export interface SoundModelInfo {
  id: string;
  default?: boolean;
  maxSeconds?: number;
  sampleRate?: number;
  channels?: number;
}

/** A 3D model maker (OAIY's Pixal3D): a picture of one object in, a GLB mesh out. */
export interface Model3dModelInfo {
  id: string;
  default?: boolean;
  /** The resolutions it makes models at (1024, 1536). */
  resolutions?: number[];
  /** The simplified mesh's triangle budget when none is asked for. */
  faces?: number;
  /** False while the service cannot make models with it yet. */
  ready?: boolean;
  /** The terms of the model's weights and what it makes, when OAIY says. */
  license?: string;
  /** The service removes a picture's background itself (any background will do). */
  removesBackground?: boolean;
  /** The service enlarges a small picture before it makes the model. */
  upscales?: boolean;
}

/** A picture tool the service has: background removal or upscaling. */
export interface PictureToolInfo {
  id: string;
  /** Upscaling: how much larger it makes a picture (2, 4). */
  scales?: number[];
  /** Upscaling: the largest picture it takes, in pixels. */
  maxPixels?: number;
  license?: string;
}

/** A voice saved on the server (designed once from a description). */
export interface VoiceInfo {
  name: string;
  description?: string;
  language?: string;
}

/**
 * A voice the service designed and handed back rather than kept (OAIY with
 * `keep: false`): the voice itself and its sample clip, for the project to
 * keep and send with each line spoken in it.
 */
export interface HandedVoice {
  voice: Record<string, unknown>;
  sample: Uint8Array;
}

/** A voice to speak in: a name the service knows, or a voice sent with the request. */
export type VoiceChoice = string | Record<string, unknown>;

export interface MediaSettings {
  /** The API base (`http://127.0.0.1:8080/v1`). Empty: there is no media service. */
  baseUrl: string;
  apiKey: string;
  /** The agent may use it. */
  enabled: boolean;
  imageModel?: string;
  videoModel?: string;
  imageModels: ImageModelInfo[];
  videoModels: VideoModelInfo[];
  speechModel?: string;
  musicModel?: string;
  soundModel?: string;
  model3dModel?: string;
  speechModels?: SpeechModelInfo[];
  musicModels?: MusicModelInfo[];
  soundModels?: SoundModelInfo[];
  model3dModels?: Model3dModelInfo[];
  /** Background removal and upscaling, when the service has them (OAIY's discovery). */
  backgroundModels?: PictureToolInfo[];
  upscaleModels?: PictureToolInfo[];
  /** Saved voices, and the OpenAI voice names the service also takes. */
  voices?: VoiceInfo[];
  openaiVoices?: string[];
  /** Full URLs from a discovery document (OAIY's routes are configurable). */
  endpoints?: { images?: string; edits?: string; videos?: string; speech?: string; voices?: string; music?: string; sound?: string; model3d?: string; background?: string; upscale?: string };
  /** Set when the details came from OAIY's discovery document. */
  discovered?: { service: string; version: string; origin: string; at: number };
}

export const EMPTY_MEDIA: MediaSettings = { baseUrl: '', apiKey: '', enabled: true, imageModels: [], videoModels: [] };

/** Where OAIY listens out of the box. */
export const OAIY_ORIGIN = 'http://127.0.0.1:8080';

/** What the agent can make with these settings. */
export function mediaReady(media: MediaSettings | null | undefined): { image: boolean; video: boolean; speech: boolean; music: boolean; sound: boolean; model3d: boolean; background: boolean; upscale: boolean } {
  const on = !!media && media.enabled && !!media.baseUrl.trim();
  return {
    image: on && !!(media!.imageModel || media!.imageModels.length),
    video: on && !!(media!.videoModel || media!.videoModels.length),
    speech: on && !!(media!.speechModel || media!.speechModels?.length),
    music: on && !!(media!.musicModel || media!.musicModels?.length),
    sound: on && !!(media!.soundModel || media!.soundModels?.length),
    model3d: on && !!(media!.model3dModel || media!.model3dModels?.length),
    background: on && !!media!.backgroundModels?.length && !!media!.endpoints?.background,
    upscale: on && !!media!.upscaleModels?.length && !!media!.endpoints?.upscale,
  };
}

/** What the service can make, in words ("images, video and speech"). */
export function mediaAbilities(media: MediaSettings | null | undefined): string {
  const ready = mediaReady(media);
  const can = [ready.image && 'images', ready.video && 'video', ready.speech && 'speech', ready.music && 'music', ready.sound && 'sound effects', ready.model3d && '3D models', ready.background && 'background removal', ready.upscale && 'upscaling'].filter(Boolean) as string[];
  return can.length > 1 ? `${can.slice(0, -1).join(', ')} and ${can[can.length - 1]}` : can[0] ?? '';
}

export class MediaError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = 'MediaError';
  }
}

/** The API base from whatever was typed: an origin, `…/v1`, or a full endpoint. */
export function mediaBase(address: string): string {
  const raw = /^[a-z]+:\/\//i.test(address.trim()) ? address.trim() : `http://${address.trim()}`;
  const url = new URL(raw);
  let path = url.pathname.replace(/\/+$/, '').replace(/\/(images\/generations|images\/edits|videos|models|discovery|chat\/completions)$/, '');
  if (path === '') path = '/v1';
  return `${url.origin}${path}`;
}

function endpointsOf(media: MediaSettings): { images: string; edits: string; videos: string; models: string; speech: string; voices: string; music: string; sound: string; model3d: string } {
  const base = mediaBase(media.baseUrl);
  const trim = (url: string) => url.replace(/\/+$/, '');
  return {
    images: media.endpoints?.images ?? `${base}/images/generations`,
    edits: media.endpoints?.edits ?? `${base}/images/edits`,
    videos: trim(media.endpoints?.videos ?? `${base}/videos`),
    models: `${base}/models`,
    speech: media.endpoints?.speech ?? `${base}/audio/speech`,
    voices: trim(media.endpoints?.voices ?? `${base}/audio/voices`),
    music: trim(media.endpoints?.music ?? `${base}/audio/music`),
    sound: trim(media.endpoints?.sound ?? `${base}/audio/sound_effects`),
    model3d: trim(media.endpoints?.model3d ?? `${base}/3d/models`),
  };
}

function authHeaders(apiKey: string): Record<string, string> {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function base64ToBytes(text: string): Uint8Array {
  const binary = atob(text.replace(/^data:[^,]*,/, '').replace(/\s+/g, ''));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

export interface MediaFile {
  bytes: Uint8Array;
  mime: string;
  name: string;
}

function dataUrl(file: MediaFile): string {
  return `data:${file.mime};base64,${bytesToBase64(file.bytes)}`;
}

/** The server's own words about a failed request, when it sent any. */
async function detailOf(resp: Response): Promise<string> {
  try {
    const text = await resp.text();
    try {
      const body = JSON.parse(text) as { error?: { message?: string } | string; message?: string };
      const detail = typeof body.error === 'string' ? body.error : body.error?.message ?? body.message;
      if (detail) return String(detail);
    } catch {
      /* not JSON */
    }
    return text.trim().slice(0, 400);
  } catch {
    return '';
  }
}

async function request(url: string, init: RequestInit & { apiKey: string; what: string }, signal?: AbortSignal): Promise<Response> {
  const { apiKey, what, ...rest } = init;
  let resp: Response;
  try {
    resp = await fetch(url, { ...rest, headers: { ...authHeaders(apiKey), ...incognitoHeaderFor(url), ...(rest.headers as Record<string, string> | undefined) }, signal });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new MediaError(`${what}: could not reach ${new URL(url).origin}. Is the service running, and does it allow requests from this page (CORS)?`);
  }
  if (resp.ok) return resp;
  const detail = await detailOf(resp);
  const hint = resp.status === 401 ? ' The service wants an API key: set it in Settings, under Images, video and audio.' : '';
  throw new MediaError(`${what} failed (HTTP ${resp.status})${detail ? `: ${detail}` : ''}.${hint}`, resp.status);
}

// --- Discovery -------------------------------------------------------------

export type Discovery =
  | {
      state: 'found';
      origin: string;
      service: string;
      version: string;
      media: MediaSettings;
      /** The chat side: its OpenAI base, models and window. */
      llm: { base: string; models: string[]; default?: string; contextTokens?: number; details: LlmModel[]; egpu: Egpu };
    }
  | { state: 'needs-key' | 'forbidden' | 'absent'; origin: string; message: string };

/** A language model the engine has: chosen in Engines (`default`), the one loaded, set to run on the eGPU. */
export type LlmModel = { id: string; default: boolean; loaded: boolean; egpu: boolean };

/** A Mac's eGPU (a card in a Thunderbolt enclosure), where OAIY offers it: switched on, which engine runs there, and
 * the model it holds or is loading (`state`: stopped, starting, ready, failed). */
export type Egpu = { available: boolean; enabled: boolean; engine?: string; state?: string; model?: string };

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v : undefined);
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const bool = (v: unknown): boolean | undefined => (typeof v === 'boolean' ? v : undefined);
const list = (v: unknown): Json[] => (Array.isArray(v) ? v.filter(isRecord) : []);

/** The origin of an address someone typed (`127.0.0.1:8080`, `http://host:8080/v1/discovery`). */
export function originOf(address: string): string {
  const raw = /^[a-z]+:\/\//i.test(address.trim()) ? address.trim() : `http://${address.trim()}`;
  return new URL(raw).origin;
}

/** Media settings and the chat side from an OAIY discovery document. */
export function readDiscovery(doc: Json, origin: string): Extract<Discovery, { state: 'found' }> {
  const endpoints = list(doc.endpoints);
  const urlOf = (name: string) => str(endpoints.find((e) => e.name === name)?.url);
  const models = isRecord(doc.models) ? doc.models : {};
  const defaults = isRecord(doc.defaults) ? doc.defaults : {};
  const imageModels: ImageModelInfo[] = list(models.image).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    default: bool(m.default),
    architecture: str(m.architecture),
    edits: bool(m.edits),
    maxReferences: num(m.max_references),
    sizeStep: num(m.size_step),
    defaultSize: str(m.default_size),
    negativePrompt: bool(m.negative_prompt),
  }));
  const videoModels: VideoModelInfo[] = list(models.video).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    default: bool(m.default),
    family: str(m.family),
    fps: num(m.fps),
    maxFrames: num(m.max_frames),
    maxSeconds: num(m.max_seconds),
    maxSide: num(m.max_side),
    startImage: bool(m.start_image),
    lipSync: bool(m.lip_sync),
  }));
  const speechModels: SpeechModelInfo[] = list(models.speech).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    default: bool(m.default),
    describedVoices: bool(m.described_voices),
    savedVoices: bool(m.saved_voices),
    engine: str(m.engine),
    license: str(m.license),
    sampleRate: num(m.sample_rate),
  }));
  const musicModels: MusicModelInfo[] = list(models.music).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    default: bool(m.default),
    maxSeconds: num(m.max_seconds),
    sampleRate: num(m.sample_rate),
    channels: num(m.channels),
  }));
  const soundModels: SoundModelInfo[] = list(models.sound).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    default: bool(m.default),
    maxSeconds: num(m.max_seconds),
    sampleRate: num(m.sample_rate),
    channels: num(m.channels),
  }));
  const model3dModels: Model3dModelInfo[] = list(models.model3d).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    default: bool(m.default),
    resolutions: Array.isArray(m.resolutions) ? m.resolutions.filter((r): r is number => typeof r === 'number') : undefined,
    faces: num(m.faces),
    ready: bool(m.ready),
    license: str(m.license),
    removesBackground: bool(m.removes_background),
    upscales: bool(m.upscales),
  }));
  const pictureTools = (kind: unknown): PictureToolInfo[] => list(kind).filter((m) => str(m.id)).map((m) => ({
    id: String(m.id),
    scales: Array.isArray(m.scales) ? m.scales.filter((s): s is number => typeof s === 'number') : undefined,
    maxPixels: num(m.max_pixels),
    license: str(m.license),
  }));
  const voiceDoc = isRecord(doc.voices) ? doc.voices : {};
  const voices: VoiceInfo[] = list(voiceDoc.saved).map((v) => ({ name: str(v.name) ?? str(v.id) ?? '', description: str(v.description), language: str(v.language) })).filter((v) => v.name);
  const openaiVoices = Array.isArray(voiceDoc.openai_names) ? voiceDoc.openai_names.filter((n): n is string => typeof n === 'string') : [];
  const base = str(doc.openai_base_url) ?? `${origin}/v1`;
  const llmModels = list(models.llm).map((m) => str(m.id)).filter((id): id is string => !!id);
  const llmDetails: LlmModel[] = list(models.llm)
    .filter((m) => !!str(m.id))
    .map((m) => ({ id: str(m.id) as string, default: bool(m.default) ?? false, loaded: bool(m.loaded) ?? false, egpu: bool(m.egpu) ?? false }));
  const egpuDoc = isRecord(doc.egpu) ? doc.egpu : {};
  const egpu: Egpu = { available: bool(egpuDoc.available) ?? false, enabled: bool(egpuDoc.enabled) ?? false, engine: str(egpuDoc.engine), state: str(egpuDoc.state), model: str(egpuDoc.model) };
  const llm = isRecord(doc.llm) ? doc.llm : {};
  const service = str(doc.service) ?? 'OAIY';
  const version = str(doc.version) ?? '';
  return {
    state: 'found',
    origin,
    service,
    version,
    media: {
      ...EMPTY_MEDIA,
      baseUrl: base,
      endpoints: { images: urlOf('images'), edits: urlOf('edits'), videos: urlOf('videos'), speech: urlOf('speech'), voices: urlOf('voices'), music: urlOf('music'), sound: urlOf('sound'), model3d: urlOf('model3d'), background: urlOf('background'), upscale: urlOf('upscale') },
      imageModels,
      videoModels,
      speechModels,
      musicModels,
      soundModels,
      model3dModels,
      backgroundModels: pictureTools(models.background),
      upscaleModels: pictureTools(models.upscale),
      voices,
      openaiVoices,
      imageModel: str(defaults.image) ?? imageModels.find((m) => m.default)?.id ?? imageModels[0]?.id,
      videoModel: str(defaults.video) ?? videoModels.find((m) => m.default)?.id ?? videoModels[0]?.id,
      speechModel: str(defaults.speech) ?? speechModels.find((m) => m.default)?.id ?? speechModels[0]?.id,
      musicModel: str(defaults.music) ?? musicModels.find((m) => m.default)?.id ?? musicModels[0]?.id,
      soundModel: str(defaults.sound) ?? soundModels.find((m) => m.default)?.id ?? soundModels[0]?.id,
      model3dModel: str(defaults.model3d) ?? model3dModels.find((m) => m.default)?.id ?? model3dModels[0]?.id,
      discovered: { service, version, origin, at: Date.now() },
    },
    llm: {
      base,
      models: llmModels,
      default: str(defaults.llm) ?? llmModels[0],
      contextTokens: num(llm.context_tokens),
      details: llmDetails,
      egpu,
    },
  };
}

/**
 * Ask a server for OAIY's discovery document: `/v1/discovery`, then the
 * fixed `/.well-known/oaiy.json`. Says why when it cannot: nothing there,
 * the page's origin not allowed (OAIY's own message says how to allow it),
 * or an API key needed.
 */
export async function discoverOaiy(address = OAIY_ORIGIN, apiKey = '', signal?: AbortSignal, timeoutMs = 5000): Promise<Discovery> {
  let origin: string;
  try {
    origin = originOf(address);
  } catch {
    return { state: 'absent', origin: address, message: `${address} is not an address` };
  }
  const deadline = AbortSignal.timeout(timeoutMs);
  const both = signal ? AbortSignal.any([signal, deadline]) : deadline;
  for (const path of ['/v1/discovery', '/.well-known/oaiy.json']) {
    let resp: Response;
    try {
      resp = await fetch(`${origin}${path}`, { headers: authHeaders(apiKey), signal: both });
    } catch (error) {
      if (signal?.aborted) throw error;
      return { state: 'absent', origin, message: `Nothing answered at ${origin}. Start OAIY, or give the address it listens on.` };
    }
    if (resp.status === 404) continue;
    if (resp.status === 403) return { state: 'forbidden', origin, message: `OAIY at ${origin} refused this page: ${(await detailOf(resp)) || 'forbidden'}. ${typeof location === 'undefined' ? '' : `The origin to allow is ${location.origin}.`}` };
    if (resp.status === 401) return { state: 'needs-key', origin, message: `OAIY at ${origin} needs its API key: enter it under Images, video and audio, then Find OAIY again.` };
    if (!resp.ok) return { state: 'absent', origin, message: `${origin}${path} answered HTTP ${resp.status}.` };
    let doc: unknown;
    try {
      doc = await resp.json();
    } catch {
      return { state: 'absent', origin, message: `${origin}${path} did not answer with JSON: it is not OAIY.` };
    }
    if (!isRecord(doc) || !(str(doc.service)?.startsWith('oaiy') || Array.isArray(doc.endpoints))) {
      return { state: 'absent', origin, message: `${origin} answered, but not as OAIY.` };
    }
    const auth = isRecord(doc.auth) ? doc.auth : {};
    if (auth.required === true && !Array.isArray(doc.endpoints)) {
      return { state: 'needs-key', origin, message: `OAIY at ${origin} needs its API key${apiKey ? ' (the one given was not accepted)' : ''}: enter it under Images, video and audio, then Find OAIY again.` };
    }
    return readDiscovery(doc, origin);
  }
  return { state: 'absent', origin, message: `${origin} has no discovery document: it is not OAIY, or an older version.` };
}

/** Found settings over the current ones: the key stays, chosen models stay while they still exist. */
export function mergeDiscovered(current: MediaSettings, found: MediaSettings): MediaSettings {
  const keepImage = current.imageModel && found.imageModels.some((m) => m.id === current.imageModel) ? current.imageModel : found.imageModel;
  const keepVideo = current.videoModel && found.videoModels.some((m) => m.id === current.videoModel) ? current.videoModel : found.videoModel;
  const keepSpeech = current.speechModel && found.speechModels?.some((m) => m.id === current.speechModel) ? current.speechModel : found.speechModel;
  const keepMusic = current.musicModel && found.musicModels?.some((m) => m.id === current.musicModel) ? current.musicModel : found.musicModel;
  const keepSound = current.soundModel && found.soundModels?.some((m) => m.id === current.soundModel) ? current.soundModel : found.soundModel;
  const keepModel3d = current.model3dModel && found.model3dModels?.some((m) => m.id === current.model3dModel) ? current.model3dModel : found.model3dModel;
  return { ...found, apiKey: current.apiKey, enabled: current.baseUrl ? current.enabled : true, imageModel: keepImage, videoModel: keepVideo, speechModel: keepSpeech, musicModel: keepMusic, soundModel: keepSound, model3dModel: keepModel3d };
}

/** The image, video, speech, music, sound effects and 3D models a server lists: typed (OAIY), else guessed from their names. */
export async function listMediaModels(media: MediaSettings, signal?: AbortSignal): Promise<{ image: string[]; video: string[]; speech: string[]; music: string[]; sound: string[]; model3d: string[] }> {
  const resp = await request(endpointsOf(media).models, { apiKey: media.apiKey, what: 'Listing the models' }, signal ?? AbortSignal.timeout(10_000));
  const body = (await resp.json()) as Json;
  const found = { image: [] as string[], video: [] as string[], speech: [] as string[], music: [] as string[], sound: [] as string[], model3d: [] as string[] };
  for (const m of list(body.data)) {
    const id = str(m.id);
    if (!id) continue;
    const type = str(m.type);
    // 3D first: a name like "hunyuan3d" or "image-to-3d" is not an image model.
    if (type === 'model3d' || (!type && /3d|pixal|trellis|triposr/i.test(id))) found.model3d.push(id);
    else if (type === 'image' || (!type && /dall-e|gpt-image|image|flux|sdxl|stable-diffusion/i.test(id))) found.image.push(id);
    else if (type === 'video' || (!type && /sora|video|veo|ltx|wan/i.test(id))) found.video.push(id);
    else if (type === 'speech' || (!type && /tts|speech/i.test(id))) found.speech.push(id);
    else if (type === 'music' || (!type && /music|song/i.test(id))) found.music.push(id);
    else if (type === 'sound' || (!type && /sound.?effect|sfx|foley/i.test(id))) found.sound.push(id);
  }
  return found;
}

// --- Jobs ------------------------------------------------------------------

/**
 * Follow an asynchronous job (a video, a song) until it is done: the status as
 * it goes, the service's reason when it fails, and the job stopped on the
 * server if the person stops the agent.
 */
async function followJob(
  media: MediaSettings,
  base: string,
  first: Json,
  what: string,
  onProgress: (message: string) => void,
  signal: AbortSignal | undefined,
  cancel: (id: string) => Promise<unknown>,
): Promise<{ id: string; job: Json }> {
  let job = first;
  const id = str(job.id);
  if (!id) throw new MediaError(`The ${what} service did not return a job id.`);
  const started = Date.now();
  try {
    while (job.status !== 'completed') {
      if (job.status === 'failed' || job.status === 'cancelled') {
        const error = isRecord(job.error) ? str(job.error.message) : str(job.error);
        throw new MediaError(`The ${what} failed: ${error ?? 'the service gave no reason'}.`);
      }
      const progress = num(job.progress);
      onProgress(job.status === 'queued' ? `${what} queued, waiting for the GPU…` : `making the ${what}${progress !== undefined ? `: ${Math.floor(progress)}%` : '…'}`);
      if (Date.now() - started > MAX_VIDEO_MS) throw new MediaError(`The ${what} took over an hour; stopped waiting for it.`);
      await sleep(POLL_MS, signal);
      job = (await (await request(`${base}/${encodeURIComponent(id)}`, { apiKey: media.apiKey, what: `Checking the ${what}` }, signal)).json()) as Json;
    }
    return { id, job };
  } catch (error) {
    if (signal?.aborted) void cancel(id).catch(() => undefined);
    throw error;
  }
}

// --- Speech and voices -----------------------------------------------------

export const SPEECH_FORMATS = ['mp3', 'wav', 'opus', 'aac', 'flac'] as const;
export type SpeechFormat = (typeof SPEECH_FORMATS)[number];

export interface SpeechRequest {
  /** What to say. */
  input: string;
  model?: string;
  /** A saved voice's name, an OpenAI voice name, or a voice sent with the request. */
  voice?: VoiceChoice;
  /** A voice described in words (or how to say it). */
  instructions?: string;
  language?: string;
  speed?: number;
  seed?: number;
  format?: SpeechFormat;
}

/** The speech as audio bytes (the whole clip: the service answers when it is made). */
/** Whether a voice sent with a request was made by Breeze TTS 2 (it has no speaker embedding, which Qwen3-TTS needs). */
function madeByBreeze(voice: VoiceChoice | undefined): boolean {
  return isRecord(voice) && Array.isArray(voice.speaker) && voice.speaker.length === 0;
}

/** The speech model for a request: the chosen one, unless the voice was made by Breeze TTS 2 and the chosen model is not Breeze. */
export function speechModelFor(media: MediaSettings, voice?: VoiceChoice): string | undefined {
  const chosen = media.speechModel;
  if (!madeByBreeze(voice)) return chosen;
  const models = media.speechModels ?? [];
  if (models.find((m) => m.id === chosen)?.engine === 'breeze-tts-2') return chosen;
  return models.find((m) => m.engine === 'breeze-tts-2')?.id ?? chosen;
}

export async function generateSpeech(media: MediaSettings, req: SpeechRequest, signal?: AbortSignal): Promise<{ bytes: Uint8Array; mime: string }> {
  const body: Json = { input: req.input, model: req.model || speechModelFor(media, req.voice), response_format: req.format ?? 'mp3' };
  if (req.voice) body.voice = req.voice;
  if (req.instructions) body.instructions = req.instructions;
  if (req.language) body.language = req.language;
  if (req.speed !== undefined) body.speed = req.speed;
  if (req.seed !== undefined) body.seed = req.seed;
  const resp = await request(endpointsOf(media).speech, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Speaking' }, signal);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  if (!bytes.length) throw new MediaError('The speech service answered without audio.');
  return { bytes, mime: resp.headers.get('content-type') ?? 'audio/mpeg' };
}

/** Design a voice from a description and save it on the server under `name`. */
/**
 * Design a voice. With `keep: false` a service that can (OAIY) hands the voice
 * back (`handed`) instead of keeping it; otherwise it is saved on the server
 * under `name`.
 */
export async function createVoice(media: MediaSettings, req: { name: string; description: string; sampleText?: string; language?: string; seed?: number; keep?: boolean }, signal?: AbortSignal): Promise<VoiceInfo & { handed?: HandedVoice }> {
  const body: Json = { name: req.name, description: req.description };
  // Designed by the chosen speech model, which then speaks it best.
  if (media.speechModel) body.model = media.speechModel;
  if (req.sampleText) body.sample_text = req.sampleText;
  if (req.language) body.language = req.language;
  if (req.seed !== undefined) body.seed = req.seed;
  if (req.keep === false) body.keep = false;
  const resp = await request(endpointsOf(media).voices, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Designing the voice' }, signal);
  const voice = (await resp.json().catch(() => ({}))) as Json;
  const info = { name: str(voice.name) ?? req.name, description: str(voice.description) ?? req.description, language: str(voice.language) ?? req.language };
  const handed = isRecord(voice.voice) && isRecord(voice.sample) && typeof voice.sample.data === 'string'
    ? { voice: voice.voice as Record<string, unknown>, sample: base64ToBytes(voice.sample.data) }
    : undefined;
  return handed ? { ...info, handed } : info;
}

/** The voices saved on the server. */
export async function listVoices(media: MediaSettings, signal?: AbortSignal): Promise<VoiceInfo[]> {
  const resp = await request(endpointsOf(media).voices, { apiKey: media.apiKey, what: 'Listing the voices' }, signal ?? AbortSignal.timeout(10_000));
  const body = (await resp.json()) as Json;
  return list(Array.isArray(body) ? body : body.data ?? body.voices).map((v) => ({ name: str(v.name) ?? str(v.id) ?? '', description: str(v.description), language: str(v.language) })).filter((v) => v.name);
}

// --- Music -----------------------------------------------------------------

export interface MusicRequest {
  /** The style: genre, instruments, mood, tempo, singer. */
  prompt: string;
  /** Lyrics, with [Verse] / [Chorus] sections; none with `instrumental`. */
  lyrics?: string;
  instrumental?: boolean;
  seconds?: number;
  model?: string;
  seed?: number;
  format?: 'wav' | 'mp3' | 'opus' | 'aac' | 'flac';
}

/** Start a song job, follow it, and download the song. */
export async function generateMusic(media: MediaSettings, req: MusicRequest, onProgress: (message: string) => void, signal?: AbortSignal): Promise<{ bytes: Uint8Array; seconds?: number; model: string }> {
  const ep = endpointsOf(media);
  const model = req.model || media.musicModel;
  const body: Json = { prompt: req.prompt, model };
  if (req.instrumental) body.instrumental = true;
  if (req.lyrics && !req.instrumental) body.lyrics = req.lyrics;
  if (req.seconds !== undefined) body.duration = req.seconds;
  if (req.seed !== undefined) body.seed = req.seed;
  const created = await request(ep.music, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Starting the song' }, signal);
  const { id, job } = await followJob(media, ep.music, (await created.json()) as Json, 'song', onProgress, signal, (jobId) =>
    fetch(`${ep.music}/${encodeURIComponent(jobId)}/cancel`, { method: 'POST', headers: authHeaders(media.apiKey) }));
  onProgress('downloading the song…');
  const format = req.format ?? 'wav';
  const content = await request(`${ep.music}/${encodeURIComponent(id)}/content${format === 'wav' ? '' : `?format=${format}`}`, { apiKey: media.apiKey, what: 'Downloading the song' }, signal);
  const seconds = num(job.seconds) ?? (str(job.seconds) ? Number(job.seconds) : undefined);
  return { bytes: new Uint8Array(await content.arrayBuffer()), seconds, model: str(job.model) ?? model ?? 'default' };
}

// --- Sound effects -----------------------------------------------------------

export interface SoundEffectRequest {
  /** What makes the sound, where, and how it sounds. */
  prompt: string;
  seconds?: number;
  model?: string;
  seed?: number;
  format?: 'wav' | 'mp3' | 'opus' | 'aac' | 'flac';
}

/** Start a sound effect job, follow it, and download the sound. */
export async function generateSoundEffect(media: MediaSettings, req: SoundEffectRequest, onProgress: (message: string) => void, signal?: AbortSignal): Promise<{ bytes: Uint8Array; seconds?: number; model: string }> {
  const ep = endpointsOf(media);
  const model = req.model || media.soundModel;
  const body: Json = { prompt: req.prompt, model };
  if (req.seconds !== undefined) body.seconds = req.seconds;
  if (req.seed !== undefined) body.seed = req.seed;
  const created = await request(ep.sound, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Starting the sound effect' }, signal);
  const { id, job } = await followJob(media, ep.sound, (await created.json()) as Json, 'sound effect', onProgress, signal, (jobId) =>
    fetch(`${ep.sound}/${encodeURIComponent(jobId)}/cancel`, { method: 'POST', headers: authHeaders(media.apiKey) }));
  onProgress('downloading the sound…');
  const format = req.format ?? 'wav';
  const content = await request(`${ep.sound}/${encodeURIComponent(id)}/content${format === 'wav' ? '' : `?format=${format}`}`, { apiKey: media.apiKey, what: 'Downloading the sound effect' }, signal);
  const seconds = num(job.seconds) ?? (str(job.seconds) ? Number(job.seconds) : undefined);
  return { bytes: new Uint8Array(await content.arrayBuffer()), seconds, model: str(job.model) ?? model ?? 'default' };
}

// --- 3D models ---------------------------------------------------------------

export interface Model3dRequest {
  /** A picture of one object, whole and centred, on a plain or transparent background. */
  image: MediaFile;
  /** 1024 or 1536: finer detail at 1536, and slower. */
  resolution?: number;
  /** The simplified mesh's triangle budget. */
  faces?: number;
  /** The camera the picture was taken with, in degrees (30 when not given). */
  fovDegrees?: number;
  seed?: number;
  model?: string;
}

export interface Model3dResult {
  /** The model: glTF 2.0 binary, Y up, its front facing +Z, fitted in a unit cube. */
  glb: Uint8Array;
  /** The object as the service cut it out of the picture (PNG, its background transparent), when it gave it. */
  cutout?: Uint8Array;
  /** How the object was cut out: "alpha" (the picture's own transparency), "birefnet" (its background removed) or "background" (a plain background cut away). */
  matte?: string;
  /** The object's crop enlarged from and to (pixels), when the service upscaled it. */
  upscaled?: [number, number];
  faces?: number;
  vertices?: number;
  /** How long the service took to make it. */
  seconds?: number;
  model: string;
}

/** Start a 3D model job from a picture, follow it, and download the GLB and the cut-out picture. */
export async function generate3dModel(media: MediaSettings, req: Model3dRequest, onProgress: (message: string) => void, signal?: AbortSignal): Promise<Model3dResult> {
  const ep = endpointsOf(media);
  const model = req.model || media.model3dModel;
  const body: Json = { image: dataUrl(req.image), model };
  if (req.resolution !== undefined) body.resolution = req.resolution;
  if (req.faces !== undefined) body.faces = req.faces;
  if (req.fovDegrees !== undefined) body.fov_degrees = req.fovDegrees;
  if (req.seed !== undefined) body.seed = req.seed;
  const created = await request(ep.model3d, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Starting the 3D model' }, signal);
  const { id, job } = await followJob(media, ep.model3d, (await created.json()) as Json, '3D model', onProgress, signal, (jobId) =>
    fetch(`${ep.model3d}/${encodeURIComponent(jobId)}/cancel`, { method: 'POST', headers: authHeaders(media.apiKey) }));
  onProgress('downloading the 3D model…');
  const content = await request(`${ep.model3d}/${encodeURIComponent(id)}/content`, { apiKey: media.apiKey, what: 'Downloading the 3D model' }, signal);
  const glb = new Uint8Array(await content.arrayBuffer());
  if (!glb.length) throw new MediaError('The 3D model service answered without a model.');
  // The cut-out shows what the model was made from; a service without it still made the model.
  let cutout: Uint8Array | undefined;
  try {
    const input = await request(`${ep.model3d}/${encodeURIComponent(id)}/input`, { apiKey: media.apiKey, what: 'Downloading the cut-out picture' }, signal);
    cutout = new Uint8Array(await input.arrayBuffer());
  } catch (error) {
    if (signal?.aborted || !(error instanceof MediaError)) throw error;
  }
  const up = Array.isArray(job.upscaled) && job.upscaled.length === 2 && job.upscaled.every((v) => typeof v === 'number') ? (job.upscaled as [number, number]) : undefined;
  return { glb, cutout: cutout?.length ? cutout : undefined, faces: num(job.faces), vertices: num(job.vertices), seconds: num(job.seconds_taken), model: str(job.model) ?? model ?? 'default', matte: str(job.matte), upscaled: up };
}

// --- Images ----------------------------------------------------------------

export interface ImageRequest {
  prompt: string;
  model?: string;
  size?: string;
  n?: number;
  negativePrompt?: string;
  seed?: number;
  /** Pictures to edit or combine. */
  references?: MediaFile[];
}

export interface ImageResult {
  images: Uint8Array[];
  model: string;
  size?: string;
  revisedPrompt?: string;
}

export async function generateImage(media: MediaSettings, req: ImageRequest, signal?: AbortSignal): Promise<ImageResult> {
  const ep = endpointsOf(media);
  const model = req.model || media.imageModel;
  const refs = req.references ?? [];
  let resp: Response;
  if (refs.length && !media.discovered) {
    // OpenAI's edits take the pictures as files.
    const form = new FormData();
    form.set('prompt', req.prompt);
    if (model) form.set('model', model);
    if (req.size) form.set('size', req.size);
    if (req.n) form.set('n', String(req.n));
    for (const ref of refs) form.append(refs.length > 1 ? 'image[]' : 'image', new Blob([ref.bytes as BlobPart], { type: ref.mime }), ref.name);
    resp = await request(ep.edits, { method: 'POST', body: form, apiKey: media.apiKey, what: 'Editing the image' }, signal);
  } else {
    const body: Json = { prompt: req.prompt, model, n: req.n ?? 1, size: req.size };
    // gpt-image models always answer in base64 and refuse the field.
    if (!/^gpt-image/i.test(model ?? '')) body.response_format = 'b64_json';
    if (req.negativePrompt) body.negative_prompt = req.negativePrompt;
    if (req.seed !== undefined) body.seed = req.seed;
    // OAIY edits through generations: the pictures as data URLs.
    if (refs.length) body.images = refs.map((r) => ({ image_url: dataUrl(r) }));
    resp = await request(ep.images, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: refs.length ? 'Editing the image' : 'Generating the image' }, signal);
  }
  const json = (await resp.json()) as Json;
  const images: Uint8Array[] = [];
  for (const item of list(json.data)) {
    const b64 = str(item.b64_json);
    if (b64) images.push(base64ToBytes(b64));
    else if (str(item.url)) {
      const file = await request(String(item.url), { apiKey: media.apiKey, what: 'Downloading the image' }, signal);
      images.push(new Uint8Array(await file.arrayBuffer()));
    }
  }
  if (!images.length) throw new MediaError('The image service answered without an image.');
  return { images, model: str(json.model) ?? model ?? 'default', size: str(json.size), revisedPrompt: str(list(json.data)[0]?.revised_prompt) };
}

// --- Picture tools -----------------------------------------------------------

export interface PictureResult {
  /** A PNG. */
  bytes: Uint8Array;
  width?: number;
  height?: number;
  model: string;
  seconds?: number;
}

async function pictureTool(media: MediaSettings, url: string | undefined, body: Json, what: string, signal?: AbortSignal): Promise<PictureResult> {
  if (!url) throw new MediaError(`${what}: the service has no such tool.`);
  const resp = await request(url, { method: 'POST', body: JSON.stringify({ ...body, response_format: 'b64_json' }), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what }, signal);
  const json = (await resp.json()) as Json;
  const item = list(json.data)[0];
  let bytes: Uint8Array | undefined;
  if (str(item?.b64_json)) bytes = base64ToBytes(String(item!.b64_json));
  else if (str(item?.url)) bytes = new Uint8Array(await (await request(String(item!.url), { apiKey: media.apiKey, what: `${what}: downloading` }, signal)).arrayBuffer());
  if (!bytes?.length) throw new MediaError(`${what}: the service answered without a picture.`);
  return { bytes, width: num(json.width), height: num(json.height), model: str(json.model) ?? 'default', seconds: num(json.seconds_taken) };
}

/** The picture with its background removed: a PNG at its own size, the background transparent. */
export function removeBackground(media: MediaSettings, image: MediaFile, signal?: AbortSignal): Promise<PictureResult> {
  return pictureTool(media, media.endpoints?.background, { image: dataUrl(image) }, 'Removing the background', signal);
}

/** The picture two or four times larger, its detail restored: a PNG. */
export function upscaleImage(media: MediaSettings, image: MediaFile, scale: 2 | 4, signal?: AbortSignal): Promise<PictureResult> {
  return pictureTool(media, media.endpoints?.upscale, { image: dataUrl(image), scale }, 'Upscaling the picture', signal);
}

// --- Video -----------------------------------------------------------------

export interface VideoRequest {
  prompt: string;
  model?: string;
  seconds?: number;
  size?: string;
  /** A picture for the first frame: the video animates it. */
  startImage?: MediaFile;
  /** A picture for the last frame: the video moves from the start to it. */
  endImage?: MediaFile;
  /** Words the character says: the service speaks them and the clip's lips follow. */
  speech?: { input: string; voice?: VoiceChoice; instructions?: string; language?: string; seed?: number };
  /** A soundtrack (speech, a voice, any audio) the clip follows. */
  audio?: MediaFile;
  /** The words spoken in `audio`: the clip's lips follow them. */
  transcript?: string;
  /** What the video should not show (watermarks, text, extra limbs...). */
  negativePrompt?: string;
}

/** The audio formats a video soundtrack may come in. */
export const SOUNDTRACK_FORMATS = ['wav', 'mp3', 'ogg', 'opus', 'flac', 'm4a', 'aac', 'webm'];

export interface VideoResult {
  bytes: Uint8Array;
  id: string;
  model: string;
  seconds?: string;
  size?: string;
}

/** How long a video job may take before the tool gives up on it. */
const MAX_VIDEO_MS = 60 * 60_000;
const POLL_MS = 2000;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(signal.reason);
    }, { once: true });
  });
}

/** Start a video job, follow it (progress as it goes) and download the MP4. Stopping cancels the job. */
export async function generateVideo(media: MediaSettings, req: VideoRequest, onProgress: (message: string) => void, signal?: AbortSignal): Promise<VideoResult> {
  const ep = endpointsOf(media);
  const model = req.model || media.videoModel;
  if (req.speech && req.audio) throw new MediaError('Give the video speech to say or an audio file to follow, not both.');
  let created: Response;
  if (req.startImage && !media.discovered && !req.endImage && !req.speech && !req.audio) {
    const form = new FormData();
    form.set('prompt', req.prompt);
    if (model) form.set('model', model);
    if (req.seconds !== undefined) form.set('seconds', String(req.seconds));
    if (req.size) form.set('size', req.size);
    form.set('input_reference', new Blob([req.startImage.bytes as BlobPart], { type: req.startImage.mime }), req.startImage.name);
    created = await request(ep.videos, { method: 'POST', body: form, apiKey: media.apiKey, what: 'Starting the video' }, signal);
  } else {
    const body: Json = { prompt: req.prompt, model, size: req.size };
    if (req.seconds !== undefined) body.seconds = String(req.seconds);
    if (req.startImage) body.input_reference = { image_url: dataUrl(req.startImage) };
    if (req.endImage) body.end_image = { image_url: dataUrl(req.endImage) };
    if (req.negativePrompt) body.negative_prompt = req.negativePrompt;
    if (req.speech) body.speech = Object.fromEntries(Object.entries({ model: speechModelFor(media, req.speech.voice), ...req.speech }).filter(([, v]) => v !== undefined && v !== ''));
    if (req.audio) {
      const format = req.audio.name.split('.').pop()!.toLowerCase();
      if (!SOUNDTRACK_FORMATS.includes(format)) throw new MediaError(`A soundtrack must be ${SOUNDTRACK_FORMATS.join(', ')}; ${req.audio.name} is not.`);
      body.input_audio = { data: bytesToBase64(req.audio.bytes), format };
      if (req.transcript) body.transcript = req.transcript;
    }
    created = await request(ep.videos, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Starting the video' }, signal);
  }
  // Stopped by the person: the job is forgotten on the server rather than left using the GPU.
  const { id, job } = await followJob(media, ep.videos, (await created.json()) as Json, 'video', onProgress, signal, (jobId) =>
    fetch(`${ep.videos}/${encodeURIComponent(jobId)}`, { method: 'DELETE', headers: authHeaders(media.apiKey) }));
  onProgress('downloading the video…');
  const content = await request(`${ep.videos}/${encodeURIComponent(id)}/content`, { apiKey: media.apiKey, what: 'Downloading the video' }, signal);
  return { bytes: new Uint8Array(await content.arrayBuffer()), id, model: str(job.model) ?? model ?? 'default', seconds: str(job.seconds) ?? (num(job.seconds) !== undefined ? String(job.seconds) : undefined), size: str(job.size) };
}
