/**
 * Running a contract service (see contract.ts) from a node's runtime.
 *
 *   1. The body is the contract's template filled from the node's values
 *      (or, for `requestFormat: wav16k`, the recording as a 16 kHz mono WAV).
 *   2. The call is sent. A service on OAIY Desktop (a URL starting with `/`,
 *      or on the desktop's origin) is called with the desktop's credential:
 *      in OAIY's window `window.__OAIY_DESKTOP__`, in a flow the desktop runs
 *      `OAIY_SERVER_URL` / `OAIY_SERVER_TOKEN`; in a plain browser tab the
 *      desktop is at the engine address in Settings (this machine unless the
 *      person pointed the editor at another), with no credential. Anything
 *      else goes through the runtime's own fetch, with its local-network
 *      permission.
 *   3. A job is followed until it is done (its progress logged, and cancelled
 *      if the flow is stopped), then what it made is fetched.
 *   4. The answer becomes what the nodes pass on: a picture as a data: URL;
 *      audio, video or a 3D model as a file (in the temp folder, so the web
 *      build does not download it) with a URL to show it; text as text.
 *
 * Only OAIY's own engine routes go to the desktop with its credential: a
 * flow cannot use this to reach the rest of the desktop's API.
 */
import type { RuntimeContext } from 'oaiy-core/src/module-types';
import type { CustomService } from './examples';
import { engineDefault, engineKindOf, extractPath, renderContractBody } from './contract';
import { getEngineBase } from '../../lib/engineEndpoint';

/** What a contract call gives back. */
export interface ContractResult {
  output: string;
  /** A picture: a data: URL (or the service's own URL for it). */
  image?: string;
  /** A file it made: where it is, and a URL a player or viewer can load. */
  path?: string;
  url?: string;
  text?: string;
  /** The answer at the response path, when it is none of the above. */
  value?: unknown;
}

export interface ContractOptions {
  nodeId: string;
  /** Base name for a file it makes. */
  filename?: string;
  /** Shown in the log: "Music", "3D model", … */
  label?: string;
}

interface Desktop {
  origin: string;
  token: string | null;
}

/** The desktop routes a contract may call with the desktop's credential. */
const DESKTOP_ROUTES = ['/api/ai/engine/', '/api/voice/transcribe', '/api/ai/providers/oaiy-engine/'];
/** How often a job is polled (tests shorten it). */
export const contractTiming = { pollMs: 2000 };
const JOB_LIMIT_MS = 3 * 60 * 60 * 1000;

function trimSlash(s: string): string {
  return s.replace(/\/+$/, '');
}

/**
 * OAIY Desktop, as this run reaches it: where OAIY's window or the desktop's own
 * flow run says it is, else the engine address in Settings (this machine's
 * loopback unless the person pointed the editor elsewhere), with no credential.
 * Read each time, so a change of address is followed without a reload.
 */
export function desktopAccess(): Desktop {
  const given = (globalThis as { __OAIY_DESKTOP__?: { origin?: unknown; token?: unknown } }).__OAIY_DESKTOP__;
  if (given && typeof given.origin === 'string' && given.origin) {
    return { origin: trimSlash(given.origin), token: typeof given.token === 'string' && given.token ? given.token : null };
  }
  const env = typeof process !== 'undefined' ? (process.env as Record<string, string | undefined> | undefined) : undefined;
  if (env?.OAIY_SERVER_URL) return { origin: trimSlash(env.OAIY_SERVER_URL), token: env.OAIY_SERVER_TOKEN || null };
  return { origin: trimSlash(getEngineBase()), token: null };
}

/** A contract URL made absolute: `/…` is on the desktop. */
export function absoluteUrl(url: string, desktop: Desktop = desktopAccess()): string {
  return url.startsWith('/') ? `${desktop.origin}${url}` : url;
}

/** The desktop's credential, when `url` is one of its engine routes. */
function desktopCredential(url: string, desktop: Desktop): string | null | undefined {
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return undefined;
  }
  if (u.origin !== new URL(desktop.origin).origin) return undefined;
  if (!DESKTOP_ROUTES.some((r) => u.pathname.startsWith(r))) return undefined;
  return desktop.token;
}

type Body = string | Uint8Array | undefined;

interface Reply {
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  text(): Promise<string>;
  json(): Promise<unknown>;
  arrayBuffer(): Promise<ArrayBuffer>;
}

async function send(ctx: RuntimeContext, url: string, method: string, headers: Record<string, string>, body?: Body): Promise<Reply> {
  const desktop = desktopAccess();
  const credential = desktopCredential(url, desktop);
  if (credential !== undefined) {
    // OAIY's own engine: straight to the desktop (no browser-side timeout,
    // no local-network prompt for OAIY itself), with its credential.
    const h: Record<string, string> = { ...headers };
    if (credential) h.Authorization = `Bearer ${credential}`;
    return (await fetch(url, {
      method,
      headers: h,
      body: body as BodyInit | undefined,
      signal: ctx.abortSignal,
    })) as unknown as Reply;
  }
  if (body instanceof Uint8Array) throw new Error(`${url}: a recording can only be sent to OAIY Voice`);
  return (await ctx.fetch(url, { method, headers, body })) as unknown as Reply;
}

async function failure(reply: Reply, what: string): Promise<Error> {
  const text = await reply.text().catch(() => '');
  let message = text.slice(0, 400);
  try {
    const j = JSON.parse(text) as { error?: { message?: string } | string; message?: string };
    message = (typeof j.error === 'object' ? j.error?.message : j.error) || j.message || message;
  } catch {
    /* not JSON */
  }
  return new Error(`${what}: HTTP ${reply.status}${message ? ` — ${message}` : ''}`);
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error('Operation aborted by user'));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(t);
      reject(new Error('Operation aborted by user'));
    }, { once: true });
  });
}

// ---------------------------------------------------------------------------
// OAIY's services, looked up when the flow runs (the CLI compiles without the
// lists; `engine:<kind>` is the default model of a kind).
// ---------------------------------------------------------------------------

let listed: { at: number; services: CustomService[] } | null = null;

/** OAIY's engine services, as the desktop lists them now (cached for 30 s). */
export async function listOaiyServices(ctx: RuntimeContext): Promise<CustomService[]> {
  if (listed && Date.now() - listed.at < 30_000) return listed.services;
  const desktop = desktopAccess();
  const url = `${desktop.origin}/api/ai/engine/services`;
  const headers: Record<string, string> = {};
  if (desktop.token) headers.Authorization = `Bearer ${desktop.token}`;
  const reply = await fetch(url, { headers, signal: ctx.abortSignal });
  if (!reply.ok) throw new Error(`OAIY Desktop did not list its engine's services (HTTP ${reply.status})`);
  const body = (await reply.json()) as { services?: CustomService[] };
  const services = (body.services ?? []).map((s) => ({ ...s, group: s.group ?? 'engine', headers: s.headers ?? '{}' }) as CustomService);
  listed = { at: Date.now(), services };
  return services;
}

async function resolveAtRun(ctx: RuntimeContext, id: string): Promise<CustomService> {
  let services: CustomService[] = [];
  let why = '';
  try {
    services = await listOaiyServices(ctx);
  } catch (e) {
    why = ` (${e instanceof Error ? e.message : String(e)})`;
  }
  const kind = engineKindOf(id);
  const found = kind ? engineDefault(services, kind) : services.find((s) => s.id === id);
  if (!found) {
    throw new Error(
      `The service "${id}" is not available${why}: its model is not installed in OAIY's Engines, or the engines are not running. ` +
        'Pick another service, or add the model in Engines.',
    );
  }
  return found;
}

// ---------------------------------------------------------------------------
// Values in: pictures as data: URLs, recordings as bytes.
// ---------------------------------------------------------------------------

function toBase64(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

function fromBase64(b64: string): Uint8Array {
  const bin = atob(b64.replace(/\s+/g, ''));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const MIME: Record<string, string> = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif',
  mp3: 'audio/mpeg', wav: 'audio/wav', ogg: 'audio/ogg', flac: 'audio/flac', m4a: 'audio/mp4',
  mp4: 'video/mp4', webm: 'video/webm', mov: 'video/quicktime', glb: 'model/gltf-binary',
};

function mimeOf(pathOrExt: string, fallback = 'application/octet-stream'): string {
  const ext = (pathOrExt.split(/[?#]/)[0].split('.').pop() || '').toLowerCase();
  return MIME[ext] ?? fallback;
}

/** The first usable reference in a node value: a string, or an object's dataUrl / url / path / image / audio / video. */
function reference(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() || null;
  if (value && typeof value === 'object') {
    const o = value as Record<string, unknown>;
    for (const k of ['dataUrl', 'url', 'image', 'audio', 'video', 'model', 'path']) {
      if (typeof o[k] === 'string' && (o[k] as string).trim()) return (o[k] as string).trim();
    }
  }
  return null;
}

/** A node value's bytes: a data:, blob: or http(s) URL, or a file path. */
export async function valueBytes(ctx: RuntimeContext, value: unknown): Promise<{ bytes: Uint8Array; mime: string } | null> {
  const ref = reference(value);
  if (!ref) return null;
  if (ref.startsWith('data:')) {
    const comma = ref.indexOf(',');
    const head = ref.slice(5, comma);
    const data = ref.slice(comma + 1);
    const bytes = head.includes(';base64') ? fromBase64(data) : new TextEncoder().encode(decodeURIComponent(data));
    return { bytes, mime: head.split(';')[0] || 'application/octet-stream' };
  }
  if (ref.startsWith('blob:')) {
    const r = await fetch(ref);
    return { bytes: new Uint8Array(await r.arrayBuffer()), mime: r.headers.get('content-type') || mimeOf(ref) };
  }
  if (/^https?:\/\//.test(ref)) {
    const r = (await ctx.fetch(ref)) as unknown as Reply;
    if (!r.ok) throw new Error(`could not read ${ref}: HTTP ${r.status}`);
    return { bytes: new Uint8Array(await r.arrayBuffer()), mime: r.headers.get('content-type') || mimeOf(ref) };
  }
  if (!ctx.tauri) throw new Error(`cannot read the file ${ref} here`);
  const res = await ctx.tauri.invoke<{ content: string }>('plugin:oaiy-filesystem|read_file', { path: ref, readAs: 'base64' });
  return { bytes: fromBase64(res?.content ?? ''), mime: mimeOf(ref) };
}

/** A picture value as a data: URL (what the engine takes). */
export async function imageDataUrl(ctx: RuntimeContext, value: unknown): Promise<string | null> {
  const ref = reference(value);
  if (!ref) return null;
  if (ref.startsWith('data:')) return ref;
  const got = await valueBytes(ctx, value);
  if (!got || got.bytes.length === 0) return null;
  const mime = got.mime.startsWith('image/') ? got.mime : mimeOf(ref, 'image/png');
  return `data:${mime};base64,${toBase64(got.bytes)}`;
}

async function tempDir(ctx: RuntimeContext): Promise<string> {
  try {
    const d = await ctx.tauri?.invoke<string>('plugin:oaiy-filesystem|get_temp_dir');
    if (d) return trimSlash(String(d).replace(/\\/g, '/'));
  } catch {
    /* fall through */
  }
  return '/tmp';
}

/** A recording as a 16 kHz mono 16-bit WAV (FFmpeg: the real one, or ffmpeg.wasm in the browser). */
export async function wav16k(ctx: RuntimeContext, value: unknown): Promise<Uint8Array> {
  if (!ctx.tauri) throw new Error('converting the recording needs FFmpeg, which this host lacks');
  const ref = reference(value);
  if (!ref) throw new Error('no recording was given');
  const dir = `${await tempDir(ctx)}/oaiy-media`;
  const stamp = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  let input = ref;
  const made: string[] = [];
  if (/^(data:|blob:|https?:\/\/)/.test(ref)) {
    const got = await valueBytes(ctx, value);
    if (!got) throw new Error('the recording is empty');
    const ext = (Object.entries(MIME).find(([, m]) => m === got.mime)?.[0]) || 'bin';
    input = `${dir}/stt_in_${stamp}.${ext}`;
    await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', { path: input, content: toBase64(got.bytes), contentType: 'base64', createDirs: true });
    made.push(input);
  }
  const output = `${dir}/stt_${stamp}.wav`;
  made.push(output);
  try {
    const r = await ctx.tauri.invoke<{ code: number; stderr?: string }>('plugin:oaiy-filesystem|run_command', {
      command: 'ffmpeg',
      args: ['-hide_banner', '-loglevel', 'error', '-y', '-i', input, '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-f', 'wav', output],
      cwd: null,
    });
    if (r?.code !== 0) throw new Error(`FFmpeg could not convert the recording: ${r?.stderr || 'unknown error'}`);
    const res = await ctx.tauri.invoke<{ content: string }>('plugin:oaiy-filesystem|read_file', { path: output, readAs: 'base64' });
    return fromBase64(res?.content ?? '');
  } finally {
    for (const p of made) await ctx.tauri.invoke('plugin:oaiy-filesystem|delete_file', { path: p }).catch(() => {});
  }
}

/** Keep what a service made as a file in the temp folder; its path and a URL to show it. */
export async function keepFile(ctx: RuntimeContext, bytes: Uint8Array, ext: string, base: string): Promise<{ path: string; url: string }> {
  if (!ctx.tauri) {
    // No filesystem (a bare runtime): a data: URL stands in for both.
    const url = `data:${mimeOf(ext)};base64,${toBase64(bytes)}`;
    return { path: url, url };
  }
  const name = (base || 'output').replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 60) || 'output';
  const path = `${await tempDir(ctx)}/oaiy-media/${name}_${Date.now()}.${ext}`;
  await ctx.tauri.invoke('plugin:oaiy-filesystem|write_file', { path, content: toBase64(bytes), contentType: 'base64', createDirs: true });
  let url = path;
  try {
    url = (await ctx.tauri.invoke<string>('get_media_url', { filePath: path })) || path;
  } catch {
    /* the path itself */
  }
  return { path, url };
}

// ---------------------------------------------------------------------------
// The call.
// ---------------------------------------------------------------------------

function headersOf(service: CustomService): Record<string, string> {
  try {
    const h = JSON.parse(service.headers || '{}');
    return h && typeof h === 'object' ? (h as Record<string, string>) : {};
  } catch {
    return {};
  }
}

async function followJob(ctx: RuntimeContext, service: CustomService, started: unknown, label: string): Promise<string> {
  const job = service.job!;
  const id = String(extractPath(started, job.idPath) ?? '');
  if (!id) throw new Error(`${label}: the service started no job (no ${job.idPath} in its answer)`);
  const at = (u: string) => absoluteUrl(u.replace(/\{id\}/g, encodeURIComponent(id)));
  const cancel = () => {
    if (job.cancelUrl) void send(ctx, at(job.cancelUrl), 'POST', { 'Content-Type': 'application/json' }, '{}').catch(() => {});
  };
  ctx.log('info', `[${label}] job ${id} started`);
  const began = Date.now();
  let shown = -1;
  let state = String(extractPath(started, job.statusPath) ?? '');
  let last: unknown = started;
  for (;;) {
    if (job.done.includes(state)) return id;
    if (job.failed.includes(state)) {
      const why = extractPath(last, job.errorPath);
      throw new Error(`${label}: the job ${state}${why ? ` — ${String(why)}` : ''}`);
    }
    if (Date.now() - began > JOB_LIMIT_MS) {
      cancel();
      throw new Error(`${label}: the job did not finish within ${JOB_LIMIT_MS / 3_600_000} hours`);
    }
    try {
      await sleep(contractTiming.pollMs, ctx.abortSignal);
    } catch (e) {
      cancel();
      throw e;
    }
    const reply = await send(ctx, at(job.statusUrl), 'GET', {});
    if (!reply.ok) throw await failure(reply, `${label}: polling job ${id}`);
    last = await reply.json();
    state = String(extractPath(last, job.statusPath) ?? '');
    const progress = job.progressPath ? Number(extractPath(last, job.progressPath)) : NaN;
    if (Number.isFinite(progress) && progress >= shown + 10) {
      shown = progress;
      ctx.log('info', `[${label}] ${Math.round(progress)}%`);
    }
  }
}

async function jobContent(ctx: RuntimeContext, service: CustomService, id: string, label: string): Promise<{ bytes: Uint8Array; mime: string | null }> {
  const job = service.job!;
  const urls = [job.contentUrl, job.contentFallbackUrl].filter((u): u is string => !!u);
  let error: Error | null = null;
  for (const u of urls) {
    const reply = await send(ctx, absoluteUrl(u.replace(/\{id\}/g, encodeURIComponent(id))), 'GET', {});
    if (reply.ok) return { bytes: new Uint8Array(await reply.arrayBuffer()), mime: reply.headers.get('content-type') };
    error = await failure(reply, `${label}: fetching what job ${id} made`);
  }
  throw error ?? new Error(`${label}: the job has no content URL`);
}

function extFor(service: CustomService, mime: string | null, fallback: string): string {
  if (mime) {
    const hit = Object.entries(MIME).find(([, m]) => mime.startsWith(m));
    if (hit) return hit[0];
  }
  return service.outputFormat || fallback;
}

/**
 * Run `serviceOrId` with the node's values. `vars.model` defaults to the
 * service's model; everything else is the node's.
 */
export async function runContract(
  ctx: RuntimeContext,
  serviceOrId: CustomService | string,
  vars: Record<string, unknown>,
  opts: ContractOptions,
): Promise<ContractResult> {
  const service = typeof serviceOrId === 'string' ? await resolveAtRun(ctx, serviceOrId) : serviceOrId;
  const label = opts.label || service.name || 'Service';
  if (ctx.abortSignal?.aborted) throw new Error('Operation aborted by user');
  const endpoint = absoluteUrl(service.endpoint);
  const method = (service.method || 'POST').toUpperCase();
  let body: Body;
  let headers: Record<string, string>;
  if (service.requestFormat === 'wav16k') {
    body = await wav16k(ctx, vars.audio ?? vars.input);
    if (body.length > 8 * 1024 * 1024) throw new Error(`${label}: the recording is too long (over about 4 minutes at 16 kHz)`);
    headers = { 'Content-Type': 'audio/wav' };
  } else {
    // Pictures go as data: URLs: `image`, `images` and any input the service
    // declares as a picture, whatever form the node was given them in.
    const values: Record<string, unknown> = { ...vars, model: vars.model || service.model || undefined };
    const pictures = new Set(['image', ...(service.inputs ?? []).filter((i) => i.type === 'image').map((i) => i.id)]);
    for (const k of pictures) {
      if (values[k] != null && values[k] !== '') values[k] = await imageDataUrl(ctx, values[k]);
    }
    if (Array.isArray(values.images)) {
      const list = (await Promise.all(values.images.map((x) => imageDataUrl(ctx, x)))).filter((x): x is string => !!x);
      values.images = list.length ? list : undefined;
    }
    // Recordings likewise (a Whisper-style service takes `{{audio}}`).
    const recordings = new Set(['audio', ...(service.inputs ?? []).filter((i) => i.type === 'audio').map((i) => i.id)]);
    for (const k of recordings) {
      const ref = reference(values[k]);
      if (ref && !ref.startsWith('data:')) {
        const got = await valueBytes(ctx, values[k]);
        if (got) values[k] = `data:${got.mime};base64,${toBase64(got.bytes)}`;
      }
    }
    body = method === 'GET' ? undefined : renderContractBody(service.bodyTemplate, values);
    headers = { 'Content-Type': 'application/json', ...headersOf(service) };
  }
  ctx.log('info', `[${label}] ${method} ${endpoint}`);
  const first = await send(ctx, endpoint, method, headers, body);
  if (!first.ok) throw await failure(first, label);

  let bytes: Uint8Array | null = null;
  let mime: string | null = null;
  let value: unknown;
  if (service.job) {
    const id = await followJob(ctx, service, await first.json(), label);
    ({ bytes, mime } = await jobContent(ctx, service, id, label));
  } else if (service.responseType === 'binary') {
    mime = first.headers.get('content-type');
    bytes = new Uint8Array(await first.arrayBuffer());
  } else if (service.responseType === 'text') {
    value = await first.text();
  } else {
    value = extractPath(await first.json(), service.responsePath);
  }

  const output = service.output || '';
  if (output === 'image') {
    let image: string;
    if (bytes) image = `data:${mime || mimeOf(service.outputFormat || 'png')};base64,${toBase64(bytes)}`;
    else if (typeof value === 'string' && /^(data:|https?:\/\/|blob:)/.test(value)) image = value;
    else if (typeof value === 'string' && value) image = `data:${mimeOf(service.outputFormat || 'png')};base64,${value}`;
    else throw new Error(`${label}: no picture at "${service.responsePath}" in the answer`);
    return { output, image };
  }
  if (output === 'audio' || output === 'video' || output === 'model3d') {
    if (!bytes && typeof value === 'string' && value) {
      if (/^(data:|blob:|https?:\/\/)/.test(value)) {
        const got = await valueBytes(ctx, value);
        bytes = got?.bytes ?? null;
        mime = got?.mime ?? null;
      } else {
        // A path on this machine (a local rig's output file): passed on as is.
        return { output, path: value, url: value };
      }
    }
    if (!bytes || bytes.length === 0) throw new Error(`${label}: the service returned no file`);
    const fallback = output === 'audio' ? 'wav' : output === 'video' ? 'mp4' : 'glb';
    const kept = await keepFile(ctx, bytes, extFor(service, mime, fallback), opts.filename || service.kind || output);
    return { output, ...kept };
  }
  if (output === 'text') return { output, text: typeof value === 'string' ? value : JSON.stringify(value ?? '') };
  return { output: 'json', value };
}
