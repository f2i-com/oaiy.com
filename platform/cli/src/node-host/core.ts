/**
 * Node-host shim for `@tauri-apps/api/core` — the Node twin of
 * `ui/src/tauri-shim/core.ts`.
 *
 * This is the whole "host shell" swap for the headless CLI: it replaces Tauri's
 * `invoke` bridge with a router that satisfies each command in plain Node —
 * real `fetch` for HTTP, `node:fs` for the filesystem, native tooling
 * (ffmpeg / Playwright) and `oaiy-server` delegation for the heavy ops (wired in
 * the heavy-op phase). It also installs `window.__TAURI__`, because some shared
 * code (notably `ui/src/services/database.ts`) reaches the bridge through that
 * global rather than this import.
 *
 * Command contracts (arg/result shapes) match the desktop Rust handlers and the
 * browser shim exactly, so the shared engine + modules run unchanged.
 */
import fsp from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  DATA_DIR,
  TMP_ROOT,
  DOWNLOADS_DIR,
  ensureDir,
  resolvePath,
  getSecret,
  storeSecret,
  fsConfinementOn,
} from './context';
import { ssrfSafeFetch } from './ssrf-guard';
import * as ffmpeg from './ffmpeg';
import * as browser from './browser';
import * as oaiyServer from './oaiy-server';

type InvokeArgs = Record<string, unknown> | undefined;
const fwd = (p: string) => p.replace(/\\/g, '/');

function notYet(cap: string): never {
  throw new Error(
    `${cap} is not available in this CLI build yet (coming in the heavy-op phase). ` +
      `Run this flow in the desktop app for now.`,
  );
}

// Cap response buffering so an attacker-controlled URL can't OOM the single-threaded
// worker by streaming a multi-GB body inside the timeout window. Generous default for
// media flows; override with OAIY_HTTP_MAX_RESPONSE_BYTES.
const MAX_RESPONSE_BYTES = (() => {
  const n = Number(process.env.OAIY_HTTP_MAX_RESPONSE_BYTES);
  return Number.isFinite(n) && n > 0 ? n : 512 * 1024 * 1024; // 512 MB
})();

async function readBodyCapped(resp: Response): Promise<Buffer> {
  const cl = Number(resp.headers.get('content-length'));
  if (Number.isFinite(cl) && cl > MAX_RESPONSE_BYTES) {
    throw new Error(`http_request: response too large (${cl} > ${MAX_RESPONSE_BYTES} bytes)`);
  }
  if (!resp.body) return Buffer.alloc(0);
  const reader = (resp.body as ReadableStream<Uint8Array>).getReader();
  const chunks: Buffer[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      total += value.byteLength;
      if (total > MAX_RESPONSE_BYTES) {
        await reader.cancel().catch(() => {});
        throw new Error(`http_request: response exceeded ${MAX_RESPONSE_BYTES} bytes`);
      }
      chunks.push(Buffer.from(value));
    }
  }
  return Buffer.concat(chunks);
}

// --- http_request --------------------------------------------------------
// Two call shapes in the wild — match the desktop Rust handler exactly:
//   { url, method, headers, body }                   — flat (core-service)
//   { request: { url, method, headers, body, ... } } — wrapped (runtime, …)
async function httpRequest(args: InvokeArgs): Promise<unknown> {
  const a = (args ?? {}) as Record<string, any>;
  const r =
    a.request && typeof a.request === 'object' && !Array.isArray(a.request)
      ? (a.request as Record<string, any>)
      : a;
  const url: string = r.url ?? r.uri ?? '';
  if (!url) throw new Error('http_request: missing url');
  const method: string = (r.method ?? 'GET').toUpperCase();
  const headers: Record<string, string> = r.headers ?? {};
  let body: string | undefined;
  if (r.body !== undefined && r.body !== null && method !== 'GET' && method !== 'HEAD') {
    body = typeof r.body === 'string' ? r.body : JSON.stringify(r.body);
  }
  // Bound the request so a slow / slow-trickle endpoint can't wedge the
  // single-threaded worker for the whole 45-min job timeout. Honor a per-request
  // timeout, then an env override, then a generous default (these flows legitimately
  // call slow AI generation endpoints, so the default must not be tight).
  const toMs = (v: unknown) => (typeof v === 'number' && v > 0 ? v : undefined);
  // Cap the (input-controlled) timeout so a flow can't pin the single-threaded worker
  // far beyond the job's own ceiling on a slow/never-responding endpoint.
  const MAX_HTTP_TIMEOUT_MS = 30 * 60_000;
  const timeoutMs = Math.min(
    toMs(r.timeout_ms) ??
      (toMs(r.timeout_secs) !== undefined ? Number(r.timeout_secs) * 1000 : undefined) ??
      toMs(Number(process.env.OAIY_HTTP_TIMEOUT_MS)) ??
      600_000,
    MAX_HTTP_TIMEOUT_MS,
  );
  const ac = new AbortController();
  const timer = setTimeout(
    () => ac.abort(new Error(`http_request timed out after ${timeoutMs}ms`)),
    timeoutMs,
  );
  try {
    // SSRF guard under `oaiy worker` (untrusted URL inputs): block private/loopback/
    // link-local/metadata targets and re-validate every redirect hop. `oaiy run` (no
    // confinement) keeps full local access for trusted local flows; the runtime's
    // policyFetch sets allow_private_networks after a granted permission, and operators
    // can opt out with OAIY_HTTP_ALLOW_LOCAL=1.
    const allowLocal =
      r.allow_private_networks === true || process.env.OAIY_HTTP_ALLOW_LOCAL === '1';
    const confined = fsConfinementOn() && !allowLocal;
    const init: RequestInit = { method, headers, body, signal: ac.signal };
    let resp: Response;
    try {
      resp = confined ? await ssrfSafeFetch(url, init) : await fetch(url, init);
    } catch (e) {
      // Node's fetch throws a bare `TypeError: fetch failed` and hides the real
      // reason on `.cause`. Surfaced verbatim through the runtime that became
      // "Error: TypeError: fetch failed" in the job log — no URL, no reason, so a
      // flow pointing at a service that isn't running looked identical to a DNS
      // failure or a TLS error. Re-throw with the method, the URL, and whatever
      // the cause chain actually says.
      const cause = (e as { cause?: unknown }).cause;
      const causeMsg =
        cause instanceof Error
          ? `${(cause as NodeJS.ErrnoException).code ?? cause.name}: ${cause.message}`
          : cause
            ? String(cause)
            : e instanceof Error
              ? e.message
              : String(e);
      throw new Error(`${method} ${url} failed — ${causeMsg}`, { cause: e });
    }
    const respHeaders: Record<string, string> = {};
    resp.headers.forEach((v, k) => {
      respHeaders[k] = v;
    });

    // Match the desktop Rust http_request contract: binary bodies are base64-encoded
    // and flagged with `bodyIsBase64` so the runtime gateway (runtime.ts) decodes
    // them via arrayBuffer()/blob(). The old `await resp.text()` UTF-8-mangled every
    // image/audio/video download silently (Node, unlike the browser shim, has Buffer).
    const buf = await readBodyCapped(resp);
    const ct = (respHeaders['content-type'] ?? '').toLowerCase();
    const isBinaryCt =
      ct.startsWith('image/') ||
      ct.startsWith('audio/') ||
      ct.startsWith('video/') ||
      ct.startsWith('application/octet-stream') ||
      ct.startsWith('application/pdf') ||
      ct.startsWith('application/zip') ||
      ct.startsWith('application/gzip') ||
      ct.startsWith('application/x-tar') ||
      ct.startsWith('font/');
    let bodyStr: string;
    let bodyIsBase64: boolean;
    if (isBinaryCt) {
      bodyStr = buf.toString('base64');
      bodyIsBase64 = true;
    } else {
      // Strict UTF-8 (like Rust's String::from_utf8); fall back to base64 on invalid bytes.
      try {
        bodyStr = new TextDecoder('utf-8', { fatal: true }).decode(buf);
        bodyIsBase64 = false;
      } catch {
        bodyStr = buf.toString('base64');
        bodyIsBase64 = true;
      }
    }
    return {
      status: resp.status,
      statusText: resp.statusText,
      ok: resp.ok,
      headers: respHeaders,
      body: bodyStr,
      url: resp.url,
      bodyIsBase64,
    };
  } finally {
    clearTimeout(timer);
  }
}

// --- filesystem ----------------------------------------------------------
async function readFile(args: InvokeArgs): Promise<unknown> {
  const a = (args ?? {}) as Record<string, any>;
  const p = resolvePath(a.path);
  const readAs = a.readAs ?? a.contentType ?? 'base64';
  const buf = await fsp.readFile(p);
  if (readAs === 'utf-8' || readAs === 'utf8' || readAs === 'text') {
    return { content: buf.toString('utf-8'), contentType: 'utf-8' };
  }
  return { content: buf.toString('base64'), contentType: 'base64' };
}

async function writeFile(args: InvokeArgs): Promise<null> {
  const a = (args ?? {}) as Record<string, any>;
  const p = resolvePath(a.path);
  ensureDir(path.dirname(p));
  const ct = a.contentType ?? 'utf-8';
  const content = String(a.content ?? '');
  if (ct === 'base64') await fsp.writeFile(p, Buffer.from(content, 'base64'));
  else await fsp.writeFile(p, content, 'utf-8');
  return null;
}

async function deleteFile(args: InvokeArgs): Promise<null> {
  const a = (args ?? {}) as Record<string, any>;
  await fsp.rm(resolvePath(a.path), { force: true });
  return null;
}

async function nativeCopyFile(args: InvokeArgs): Promise<null> {
  const a = (args ?? {}) as Record<string, any>;
  const src = resolvePath(a.src ?? a.from ?? a.source);
  const dst = resolvePath(a.dst ?? a.to ?? a.destination);
  ensureDir(path.dirname(dst));
  await fsp.copyFile(src, dst);
  return null;
}

async function listFolder(args: InvokeArgs): Promise<unknown> {
  const a = (args ?? {}) as Record<string, any>;
  const p = resolvePath(a.path);
  const entries = await fsp.readdir(p, { withFileTypes: true });
  return entries.map((e) => ({
    name: e.name,
    path: fwd(path.join(p, e.name)),
    isDirectory: e.isDirectory(),
  }));
}

async function readChunkContent(args: InvokeArgs): Promise<string> {
  const a = (args ?? {}) as Record<string, any>;
  const p = resolvePath(String(a.path ?? ''));
  const start = Math.max(0, Number(a.start ?? 0));
  // Cap the requested length: Buffer.alloc(length) eagerly reserves the full size, so
  // an attacker-supplied length (e.g. 1e11) would OOM the worker before reading a byte.
  const MAX_CHUNK_BYTES = 128 * 1024 * 1024;
  const length = Math.min(Math.max(0, Number(a.length ?? 0)), MAX_CHUNK_BYTES);
  const readAs = a.readAs ?? 'text';
  const fd = await fsp.open(p, 'r');
  try {
    const buf = Buffer.alloc(length);
    const { bytesRead } = await fd.read(buf, 0, length, start);
    const slice = buf.subarray(0, bytesRead);
    return readAs === 'base64' ? slice.toString('base64') : slice.toString('utf-8');
  } finally {
    await fd.close();
  }
}

async function calculateFileChunks(args: InvokeArgs): Promise<unknown> {
  const a = (args ?? {}) as Record<string, any>;
  const orig = String(a.path ?? '');
  const size = (await fsp.stat(resolvePath(orig))).size;
  const chunkSize = Math.max(1, Number(a.chunkSize ?? 1024 * 1024));
  const overlap = Math.max(0, Number(a.overlap ?? 0));
  const step = Math.max(1, chunkSize - overlap);
  const starts: number[] = [];
  for (let s = 0; s < size; s += step) {
    starts.push(s);
    if (s + chunkSize >= size) break;
  }
  if (starts.length === 0) starts.push(0);
  const total = starts.length;
  // Keep the ORIGINAL (virtual) path so a follow-up read_chunk_content resolves it again.
  return starts.map((s, index) => ({
    path: orig,
    start: s,
    length: Math.min(chunkSize, size - s),
    index,
    total,
  }));
}

async function resizeImage(args: InvokeArgs): Promise<unknown> {
  const a = (args ?? {}) as Record<string, any>;
  const dataUrl = String(a.dataUrl ?? '');
  const maxDimension = Number(a.maxDimension ?? 0);
  const maxSizeKb = Number(a.maxSizeKb ?? 0);
  const fail = (error: string, origKb = 0): Record<string, unknown> => ({
    success: false,
    dataUrl: null,
    error,
    originalWidth: 0,
    originalHeight: 0,
    newWidth: 0,
    newHeight: 0,
    originalSizeKb: origKb,
    newSizeKb: 0,
  });

  const m = dataUrl.match(/^data:([^;]+);base64,(.*)$/);
  if (!m) return fail('resize_image: input is not a base64 data URL');
  const input = Buffer.from(m[2], 'base64');
  const origKb = Math.round(input.length / 1024);

  // `sharp` is an OPTIONAL dependency (native). When absent we degrade
  // gracefully — the caller's `if (result.success)` skips resize and uses the
  // original image (works, just not downscaled). With sharp installed, real
  // headless downscaling happens.
  let sharp: any;
  try {
    sharp = (await import('sharp')).default;
  } catch {
    return fail('resize_image needs the optional "sharp" dependency (npm i sharp in cli/)', origKb);
  }
  try {
    const meta = await sharp(input).metadata();
    const ow = meta.width ?? 0;
    const oh = meta.height ?? 0;
    const resizeOpts =
      maxDimension > 0 ? { fit: 'inside' as const, withoutEnlargement: true } : undefined;
    const dim = maxDimension > 0 ? maxDimension : undefined;

    let out = await sharp(input).resize(dim, dim, resizeOpts).toBuffer();
    // If a size cap is set and we're still over, recompress as JPEG, dropping quality.
    if (maxSizeKb > 0 && out.length / 1024 > maxSizeKb) {
      for (const quality of [80, 60, 45, 30, 20]) {
        out = await sharp(input).resize(dim, dim, resizeOpts).jpeg({ quality }).toBuffer();
        if (out.length / 1024 <= maxSizeKb) break;
      }
    }
    const outMeta = await sharp(out).metadata();
    const mime = outMeta.format === 'jpeg' ? 'image/jpeg' : `image/${outMeta.format ?? 'png'}`;
    return {
      success: true,
      dataUrl: `data:${mime};base64,${out.toString('base64')}`,
      originalWidth: ow,
      originalHeight: oh,
      newWidth: outMeta.width ?? ow,
      newHeight: outMeta.height ?? oh,
      originalSizeKb: origKb,
      newSizeKb: Math.round(out.length / 1024),
      error: null,
    };
  } catch (e) {
    return fail(`resize_image failed: ${e instanceof Error ? e.message : String(e)}`, origKb);
  }
}

function getMediaUrl(args: InvokeArgs): string {
  const a = (args ?? {}) as Record<string, any>;
  const fp: string = a.filePath ?? a.file_path ?? a.path ?? '';
  if (!fp) return '';
  if (/^(https?|data|blob|file):/i.test(fp)) return fp;
  return pathToFileURL(resolvePath(fp)).href;
}

async function cleanupTempDir(args: InvokeArgs): Promise<null> {
  const a = (args ?? {}) as Record<string, any>;
  const resolved = path.resolve(resolvePath(a.path ?? a.dir ?? '/tmp'));
  // Guard: never recursively delete outside the scratch root. Separator-boundary
  // check so a sibling dir sharing the name prefix can't match.
  const root = path.resolve(TMP_ROOT);
  if (resolved === root || resolved.startsWith(root + path.sep)) {
    await fsp.rm(resolved, { recursive: true, force: true });
  }
  return null;
}

function getSecrets(args: InvokeArgs): Record<string, string> {
  const keys: string[] = ((args ?? {}) as Record<string, any>).keys ?? [];
  const out: Record<string, string> = {};
  for (const k of keys) {
    const v = getSecret(k);
    if (v) out[k] = v;
  }
  return out;
}

// --- command router ------------------------------------------------------
export async function invoke<T = unknown>(cmd: string, args?: InvokeArgs): Promise<T> {
  switch (cmd) {
    case 'http_request':
      return (await httpRequest(args)) as T;

    case 'plugin:oaiy-filesystem|get_app_data_dir':
      return fwd(DATA_DIR) as T;
    case 'plugin:oaiy-filesystem|get_temp_dir':
      return fwd(TMP_ROOT) as T;
    case 'plugin:oaiy-filesystem|get_downloads_path':
    case 'get_downloads_path':
      return fwd(DOWNLOADS_DIR) as T;
    case 'plugin:oaiy-filesystem|read_file':
      return (await readFile(args)) as T;
    case 'plugin:oaiy-filesystem|write_file':
      return (await writeFile(args)) as T;
    case 'plugin:oaiy-filesystem|delete_file':
      return (await deleteFile(args)) as T;
    case 'plugin:oaiy-filesystem|native_copy_file':
      return (await nativeCopyFile(args)) as T;
    case 'plugin:oaiy-filesystem|list_folder':
      return (await listFolder(args)) as T;
    case 'plugin:oaiy-filesystem|read_chunk_content':
      return (await readChunkContent(args)) as T;
    case 'plugin:oaiy-filesystem|calculate_file_chunks':
      return (await calculateFileChunks(args)) as T;

    case 'get_media_url':
      return getMediaUrl(args) as T;
    case 'get_media_server_port':
    case 'get_media_token':
      return null as T;
    case 'cleanup_temp_dir':
      return (await cleanupTempDir(args)) as T;

    case 'get_secrets':
      return getSecrets(args) as T;
    case 'store_secret': {
      const a = (args ?? {}) as Record<string, any>;
      storeSecret(a.key, a.value ?? '');
      return null as T;
    }

    // Service lifecycle — delegated to oaiy-server (it spawns/supervises the
    // managed AI services; the flow then HTTP-calls the service port directly).
    case 'ensure_service_ready':
      return (await oaiyServer.ensureService((args ?? {}) as Record<string, unknown>)) as T;
    case 'ensure_service_ready_by_port':
      return (await oaiyServer.ensureByPort((args ?? {}) as Record<string, unknown>)) as T;
    case 'get_service_port':
      return null as T;

    // App/config stubs — the engine never depends on these to execute a flow.
    case 'get_cli_run_config':
      return { isRunMode: false } as T;
    case 'get_install_config':
      return {} as T;
    case 'get_api_status':
      return { allow_no_auth: true, running: false, port: null } as T;
    case 'get_api_port':
      return null as T;
    case 'load_all_macros':
      return [] as T;
    case 'get_bundled_plugins_dir':
      return null as T;
    case 'get_package_services':
      return [] as T;
    case 'extract_package_service':
      return '' as T;
    case 'start_package_service':
      return { id: '', running: false, healthy: false } as T;
    case 'stop_package_services':
      return null as T;
    case 'get_service_logs':
      return { logs: [] } as T;

    // Media ops — native ffmpeg/ffprobe (system binary on PATH).
    case 'plugin:oaiy-filesystem|run_command':
      return (await ffmpeg.runCommand((args ?? {}) as Record<string, unknown>)) as T;
    case 'get_video_info':
      return (await ffmpeg.getVideoInfo((args ?? {}) as Record<string, unknown>)) as T;
    case 'extract_video_frames': {
      const a = (args ?? {}) as Record<string, unknown>;
      return (await ffmpeg.extractVideoFrames(
        String(a.path ?? ''),
        (a.options ?? {}) as never,
      )) as T;
    }
    case 'extract_video_frames_batch':
      return (await ffmpeg.extractVideoFramesBatch((args ?? {}) as Record<string, unknown>)) as T;
    case 'resize_image':
      return (await resizeImage(args)) as T;

    // Browser automation — Playwright-backed chromium sessions.
    case 'plugin:oaiy-browser|browser_v2_create_session':
      return (await browser.createSession((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_goto':
      return (await browser.goto((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_evaluate_json':
      return (await browser.evaluateJson((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_get_html':
      return (await browser.getHtml((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_get_text':
      return (await browser.getText((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_get_title':
      return (await browser.getTitle((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_get_url':
      return (await browser.getUrl((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_wait_for_selector':
      return (await browser.waitForSelector((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_screenshot':
      return (await browser.screenshot((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_get_cookies':
      return (await browser.getCookies((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_set_cookies':
      return (await browser.setCookies((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_close_session':
      return (await browser.close((args ?? {}) as Record<string, never>)) as T;
    case 'plugin:oaiy-browser|browser_v2_list_sessions':
      return [] as T;
    case 'plugin:oaiy-browser|browser_v2_http_request':
      return (await httpRequest(args)) as T;
    case 'plugin:oaiy-browser|browser_engine_get_status':
      return {
        engine: 'chromium',
        source: 'system',
        binaryPath: '',
        platform: process.platform,
      } as T;
    case 'plugin:oaiy-browser|browser_engine_check_updates':
      return { updateAvailable: false } as T;
    case 'plugin:oaiy-browser|browser_engine_use_bundled':
    case 'plugin:oaiy-browser|browser_engine_set_custom_path':
      return null as T;

    default:
      // Heavy/native plugins with no Node implementation: throw a clear message
      // instead of returning null (callers that do `result.field` would
      // null-deref and surface a confusing TypeError otherwise).
      if (
        cmd.startsWith('plugin:oaiy-terminal|') ||
        cmd.startsWith('plugin:oaiy-agent|') ||
        cmd.startsWith('plugin:oaiy-tts|') ||
        cmd.startsWith('plugin:oaiy-llm|') ||
        cmd.startsWith('plugin:oaiy-diffusion|')
      ) {
        return notYet(`Native capability '${cmd}'`);
      }
      console.warn(`[node-host] unhandled invoke command: ${cmd} — returning null`);
      return null as T;
  }
}

export function convertFileSrc(filePath: string, _protocol = 'asset'): string {
  if (/^(https?|data|blob|file):/i.test(filePath)) return filePath;
  return pathToFileURL(resolvePath(filePath)).href;
}

// Install the globals some shared code reaches the bridge through (e.g.
// services/database.ts reads `window.__TAURI__.core.invoke` and registers a
// `beforeunload` flush listener at load). Backing `window` with a real
// EventTarget gives those addEventListener/removeEventListener calls a no-op
// home — the events simply never fire headless. Mirror the browser shim.
const g = globalThis as unknown as {
  window?: EventTarget & Record<string, unknown>;
  __TAURI__?: unknown;
  __OAIY_NODE_HOST__?: boolean;
};
if (typeof g.window === 'undefined') {
  g.window = new EventTarget() as EventTarget & Record<string, unknown>;
}
if (!g.window.__TAURI__) g.window.__TAURI__ = { core: { invoke, convertFileSrc } };
if (!g.__TAURI__) g.__TAURI__ = g.window.__TAURI__;
g.__OAIY_NODE_HOST__ = true;
