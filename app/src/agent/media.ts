/**
 * Images and video for the agent, from an OpenAI-spec media service: nrob
 * (found through its discovery document), or any server with
 * `/images/generations` and `/videos`. The page calls it directly, as it
 * calls the AI provider: it is the person's own service, set up in Settings,
 * so these requests are not behind the network gate.
 */

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
}

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
  /** Full URLs from a discovery document (nrob's routes are configurable). */
  endpoints?: { images?: string; edits?: string; videos?: string };
  /** Set when the details came from nrob's discovery document. */
  discovered?: { service: string; version: string; origin: string; at: number };
}

export const EMPTY_MEDIA: MediaSettings = { baseUrl: '', apiKey: '', enabled: true, imageModels: [], videoModels: [] };

/** Where nrob listens out of the box. */
export const NROB_ORIGIN = 'http://127.0.0.1:8080';

/** What the agent can make with these settings. */
export function mediaReady(media: MediaSettings | null | undefined): { image: boolean; video: boolean } {
  const on = !!media && media.enabled && !!media.baseUrl.trim();
  return {
    image: on && !!(media!.imageModel || media!.imageModels.length),
    video: on && !!(media!.videoModel || media!.videoModels.length),
  };
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

function endpointsOf(media: MediaSettings): { images: string; edits: string; videos: string; models: string } {
  const base = mediaBase(media.baseUrl);
  return {
    images: media.endpoints?.images ?? `${base}/images/generations`,
    edits: media.endpoints?.edits ?? `${base}/images/edits`,
    videos: (media.endpoints?.videos ?? `${base}/videos`).replace(/\/+$/, ''),
    models: `${base}/models`,
  };
}

function authHeaders(apiKey: string): Record<string, string> {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

function bytesToBase64(bytes: Uint8Array): string {
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
    resp = await fetch(url, { ...rest, headers: { ...authHeaders(apiKey), ...(rest.headers as Record<string, string> | undefined) }, signal });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new MediaError(`${what}: could not reach ${new URL(url).origin}. Is the service running, and does it allow requests from this page (CORS)?`);
  }
  if (resp.ok) return resp;
  const detail = await detailOf(resp);
  const hint = resp.status === 401 ? ' The service wants an API key: set it in Settings, under Images and video.' : '';
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
      llm: { base: string; models: string[]; default?: string; contextTokens?: number };
    }
  | { state: 'needs-key' | 'forbidden' | 'absent'; origin: string; message: string };

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

/** Media settings and the chat side from an nrob discovery document. */
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
  }));
  const base = str(doc.openai_base_url) ?? `${origin}/v1`;
  const llmModels = list(models.llm).map((m) => str(m.id)).filter((id): id is string => !!id);
  const llm = isRecord(doc.llm) ? doc.llm : {};
  const service = str(doc.service) ?? 'nrob';
  const version = str(doc.version) ?? '';
  return {
    state: 'found',
    origin,
    service,
    version,
    media: {
      ...EMPTY_MEDIA,
      baseUrl: base,
      endpoints: { images: urlOf('images'), edits: urlOf('edits'), videos: urlOf('videos') },
      imageModels,
      videoModels,
      imageModel: str(defaults.image) ?? imageModels.find((m) => m.default)?.id ?? imageModels[0]?.id,
      videoModel: str(defaults.video) ?? videoModels.find((m) => m.default)?.id ?? videoModels[0]?.id,
      discovered: { service, version, origin, at: Date.now() },
    },
    llm: {
      base,
      models: llmModels,
      default: str(defaults.llm) ?? llmModels[0],
      contextTokens: num(llm.context_tokens),
    },
  };
}

/**
 * Ask a server for nrob's discovery document: `/v1/discovery`, then the
 * fixed `/.well-known/nrob.json`. Says why when it cannot: nothing there,
 * the page's origin not allowed (nrob's own message says how to allow it),
 * or an API key needed.
 */
export async function discoverNrob(address = NROB_ORIGIN, apiKey = '', signal?: AbortSignal, timeoutMs = 5000): Promise<Discovery> {
  let origin: string;
  try {
    origin = originOf(address);
  } catch {
    return { state: 'absent', origin: address, message: `${address} is not an address` };
  }
  const deadline = AbortSignal.timeout(timeoutMs);
  const both = signal ? AbortSignal.any([signal, deadline]) : deadline;
  for (const path of ['/v1/discovery', '/.well-known/nrob.json']) {
    let resp: Response;
    try {
      resp = await fetch(`${origin}${path}`, { headers: authHeaders(apiKey), signal: both });
    } catch (error) {
      if (signal?.aborted) throw error;
      return { state: 'absent', origin, message: `Nothing answered at ${origin}. Start nrob, or give the address it listens on.` };
    }
    if (resp.status === 404) continue;
    if (resp.status === 403) return { state: 'forbidden', origin, message: `nrob at ${origin} refused this page: ${(await detailOf(resp)) || 'forbidden'}. ${typeof location === 'undefined' ? '' : `The origin to allow is ${location.origin}.`}` };
    if (resp.status === 401) return { state: 'needs-key', origin, message: `nrob at ${origin} needs its API key: enter it under Images and video, then Find nrob again.` };
    if (!resp.ok) return { state: 'absent', origin, message: `${origin}${path} answered HTTP ${resp.status}.` };
    let doc: unknown;
    try {
      doc = await resp.json();
    } catch {
      return { state: 'absent', origin, message: `${origin}${path} did not answer with JSON: it is not nrob.` };
    }
    if (!isRecord(doc) || !(str(doc.service)?.startsWith('nrob') || Array.isArray(doc.endpoints))) {
      return { state: 'absent', origin, message: `${origin} answered, but not as nrob.` };
    }
    const auth = isRecord(doc.auth) ? doc.auth : {};
    if (auth.required === true && !Array.isArray(doc.endpoints)) {
      return { state: 'needs-key', origin, message: `nrob at ${origin} needs its API key${apiKey ? ' (the one given was not accepted)' : ''}: enter it under Images and video, then Find nrob again.` };
    }
    return readDiscovery(doc, origin);
  }
  return { state: 'absent', origin, message: `${origin} has no discovery document: it is not nrob, or an older version.` };
}

/** Found settings over the current ones: the key stays, chosen models stay while they still exist. */
export function mergeDiscovered(current: MediaSettings, found: MediaSettings): MediaSettings {
  const keepImage = current.imageModel && found.imageModels.some((m) => m.id === current.imageModel) ? current.imageModel : found.imageModel;
  const keepVideo = current.videoModel && found.videoModels.some((m) => m.id === current.videoModel) ? current.videoModel : found.videoModel;
  return { ...found, apiKey: current.apiKey, enabled: current.baseUrl ? current.enabled : true, imageModel: keepImage, videoModel: keepVideo };
}

/** The image and video models a server lists: typed (nrob), else guessed from their names. */
export async function listMediaModels(media: MediaSettings, signal?: AbortSignal): Promise<{ image: string[]; video: string[] }> {
  const resp = await request(endpointsOf(media).models, { apiKey: media.apiKey, what: 'Listing the models' }, signal ?? AbortSignal.timeout(10_000));
  const body = (await resp.json()) as Json;
  const image: string[] = [];
  const video: string[] = [];
  for (const m of list(body.data)) {
    const id = str(m.id);
    if (!id) continue;
    const type = str(m.type);
    if (type === 'image' || (!type && /dall-e|gpt-image|image|flux|sdxl|stable-diffusion/i.test(id))) image.push(id);
    else if (type === 'video' || (!type && /sora|video|veo|ltx|wan/i.test(id))) video.push(id);
  }
  return { image, video };
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
    // nrob edits through generations: the pictures as data URLs.
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

// --- Video -----------------------------------------------------------------

export interface VideoRequest {
  prompt: string;
  model?: string;
  seconds?: number;
  size?: string;
  /** A picture for the first frame: the video animates it. */
  startImage?: MediaFile;
}

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
  let created: Response;
  if (req.startImage && !media.discovered) {
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
    created = await request(ep.videos, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, apiKey: media.apiKey, what: 'Starting the video' }, signal);
  }
  let job = (await created.json()) as Json;
  const id = str(job.id);
  if (!id) throw new MediaError('The video service did not return a job id.');
  const started = Date.now();
  try {
    while (job.status !== 'completed') {
      if (job.status === 'failed' || job.status === 'cancelled') {
        const error = isRecord(job.error) ? str(job.error.message) : str(job.error);
        throw new MediaError(`The video failed: ${error ?? 'the service gave no reason'}.`);
      }
      const progress = num(job.progress);
      onProgress(job.status === 'queued' ? 'video queued, waiting for the GPU…' : `making the video${progress !== undefined ? `: ${Math.floor(progress)}%` : '…'}`);
      if (Date.now() - started > MAX_VIDEO_MS) throw new MediaError('The video took over an hour; stopped waiting for it.');
      await sleep(POLL_MS, signal);
      job = (await (await request(`${ep.videos}/${encodeURIComponent(id)}`, { apiKey: media.apiKey, what: 'Checking the video' }, signal)).json()) as Json;
    }
    onProgress('downloading the video…');
    const content = await request(`${ep.videos}/${encodeURIComponent(id)}/content`, { apiKey: media.apiKey, what: 'Downloading the video' }, signal);
    return { bytes: new Uint8Array(await content.arrayBuffer()), id, model: str(job.model) ?? model ?? 'default', seconds: str(job.seconds) ?? (num(job.seconds) !== undefined ? String(job.seconds) : undefined), size: str(job.size) };
  } catch (error) {
    // Stopped by the person: cancel the job rather than leave the GPU busy.
    if (signal?.aborted) void fetch(`${ep.videos}/${encodeURIComponent(id)}`, { method: 'DELETE', headers: authHeaders(media.apiKey) }).catch(() => undefined);
    throw error;
  }
}
