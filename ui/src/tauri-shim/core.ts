/**
 * Browser shim for `@tauri-apps/api/core`.
 *
 * This is the heart of the web host shell: it replaces Tauri's `invoke`
 * bridge with a router that decides, per command, how to satisfy it in a
 * plain browser — real `fetch` for the external-engine HTTP proxy, small
 * stubs for the handful of desktop-only commands, and a loud warning +
 * `null` for anything not yet mapped (so missing capability is visible in
 * the console without crashing the render).
 *
 * It also installs `window.__TAURI__` and `window.__TAURI_INTERNALS__`,
 * because some renderer code (e.g. services/database.ts and plugin UI
 * components) reach the bridge through those globals rather than this import.
 *
 * Importing this module for its side effect installs the globals; it is
 * imported first in main.tsx.
 */

import { pickFileViaInput, pickFolderViaInput, type OpenDialogOptions } from './dialog';

type InvokeArgs = Record<string, unknown> | undefined;

// ---------------------------------------------------------------------------
// Command router
// ---------------------------------------------------------------------------

async function httpRequest(args: InvokeArgs): Promise<unknown> {
  // The oaiy-core runtime proxies node HTTP through `http_request` (on desktop
  // this dodged CORS via the Rust backend). In the browser we issue a real
  // fetch — so the target engine must allow this origin (Ollama: OLLAMA_ORIGINS,
  // ComfyUI: --enable-cors-header), or be a same-origin packaged bundle.
  //
  // Two call shapes in the wild — match the desktop Rust handler exactly:
  //
  //   { url, method, headers, body }                        — flat (core-service)
  //   { request: { url, method, headers, body, ... } }      — wrapped (core
  //                                                            runtime,
  //                                                            core-utility,
  //                                                            core-browser)
  //
  // Unwrap `request` first so both lands in the same `r`. The original shim
  // only read the flat shape, so wrapped calls saw `a.method === undefined`
  // and fell through to the `'GET'` default — that's where the
  // "POST → GET" bug came from.
  const a = (args ?? {}) as Record<string, any>;
  const r = (a.request && typeof a.request === 'object' && !Array.isArray(a.request))
    ? (a.request as Record<string, any>)
    : a;
  const url: string = r.url ?? r.uri ?? '';
  if (!url) throw new Error('http_request: missing url');
  const method: string = (r.method ?? 'GET').toUpperCase();
  const headers: Record<string, string> = r.headers ?? {};
  let body: BodyInit | undefined;
  if (r.body !== undefined && r.body !== null && method !== 'GET' && method !== 'HEAD') {
    body = typeof r.body === 'string' ? r.body : JSON.stringify(r.body);
  }
  // Bound the request so a hung endpoint can't stall the run forever. Honor a
  // per-request timeout, else a generous default (these flows call slow AI
  // generation endpoints). No env override — process.env doesn't exist in browsers.
  const toMs = (v: unknown) => (typeof v === 'number' && v > 0 ? v : undefined);
  const timeoutMs =
    toMs(r.timeout_ms) ??
    (toMs(r.timeout_secs) !== undefined ? Number(r.timeout_secs) * 1000 : undefined) ??
    600_000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    let resp: Response;
    try {
      resp = await fetch(url, { method, headers, body, signal: ctrl.signal });
    } catch (e) {
      // Our timeout fired (DOMException 'AbortError') — distinct from a CORS block.
      if (e instanceof DOMException && e.name === 'AbortError') {
        throw new Error(
          `http_request: request to ${url} timed out after ${timeoutMs}ms (the endpoint did not respond). Raise the per-request timeout for slow generation endpoints, or verify the engine is reachable.`,
        );
      }
      // A bare TypeError from fetch means the request was blocked before any
      // response — almost always CORS (the target engine must allow this
      // origin), mixed content (an https page can't call an http:// endpoint),
      // or an unreachable host. Surface that instead of the opaque "Failed to
      // fetch" so the user knows it's a browser limitation, not a flow bug.
      if (e instanceof TypeError) {
        const pageHttps = typeof window !== 'undefined' && window.location.protocol === 'https:';
        let hint = `Browser blocked the request to ${url} — the endpoint must allow this origin (CORS)`;
        if (url.startsWith('http://') && pageHttps) {
          hint += '; also, an https page cannot call an http:// endpoint (mixed content)';
        }
        hint += '. Local engines: enable CORS (e.g. OLLAMA_ORIGINS=*, ComfyUI --enable-cors-header, LM Studio "CORS" toggle). For an endpoint that doesn\'t allow browser requests, route it through a server-side proxy or the desktop app.';
        throw new Error(hint);
      }
      throw e;
    }
    const respHeaders: Record<string, string> = {};
    resp.headers.forEach((v, k) => { respHeaders[k] = v; });

    // Match the desktop Rust / node-host http_request contract: binary bodies are
    // base64-encoded and flagged with bodyIsBase64 so the runtime gateway decodes
    // them via arrayBuffer()/blob(). The old `resp.text()` UTF-8-mangled every
    // image/audio/video download (the lone host that broke the contract).
    const buf = new Uint8Array(await resp.arrayBuffer());
    const ct = (respHeaders['content-type'] || '').toLowerCase();
    const isBinaryCt =
      ct.startsWith('image/') || ct.startsWith('audio/') || ct.startsWith('video/') ||
      ct.startsWith('application/octet-stream') || ct.startsWith('application/pdf') ||
      ct.startsWith('application/zip') || ct.startsWith('application/gzip') ||
      ct.startsWith('application/x-tar') || ct.startsWith('font/');
    const toBase64 = (b: Uint8Array): string => {
      // Chunk so btoa's binary string can't blow the call stack on large payloads.
      let bin = '';
      const CHUNK = 8192;
      for (let i = 0; i < b.length; i += CHUNK) {
        bin += String.fromCharCode.apply(null, Array.from(b.subarray(i, i + CHUNK)));
      }
      return btoa(bin);
    };
    let bodyStr: string;
    let bodyIsBase64: boolean;
    if (isBinaryCt) {
      bodyStr = toBase64(buf);
      bodyIsBase64 = true;
    } else {
      try {
        bodyStr = new TextDecoder('utf-8', { fatal: true }).decode(buf);
        bodyIsBase64 = false;
      } catch {
        bodyStr = toBase64(buf);
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

// ---------------------------------------------------------------------------
// Virtual filesystem for the web build
// ---------------------------------------------------------------------------
//
// Desktop's filesystem-plugin write_file+read_file is a real disk; the web
// build can't write to disk, but workflows still chain write→read for
// intermediates (extracted video frames, intermediate audio, etc.). We back
// them with an in-memory `Map<path, {content, contentType}>` so chains work.
//
// On TOP of that, paths that look like a user-facing output (not in a
// known temp/cache subtree) ALSO trigger a browser download — that's how
// "any output is a downloaded file" gets delivered in the web build.
//
// Paths starting with `data:` are decoded directly by read_file, so files
// the user uploads via the dialog shim flow through the same code path
// the desktop uses for reading a real disk file.

/**
 * webVfs entry shape.
 *
 * Small / textual entries stay as in-memory strings — fastest read,
 * smallest overhead. Large or binary entries are stored as Blob handles
 * because:
 *   - Blob is heap-cheap (browser-managed storage that can spill to
 *     disk above ~16-64 MB depending on browser).
 *   - base64-encoding a 500 MB output adds 33% AND keeps it in JS string
 *     heap, which V8's string-length cap on 32-bit ints starts brushing.
 *   - Blobs round-trip cleanly into ffmpeg.wasm via WORKERFS mount in
 *     subsequent nodes (zero copy).
 *
 * `read_file` materialises Blob entries to the `{ content, contentType }`
 * shape callers expect (it base64-encodes on demand); chained ffmpeg
 * nodes get the Blob directly via VfsBridge.getBlob.
 */
type WebVfsEntry =
  | { kind: 'string'; content: string; contentType: 'base64' | 'utf-8' }
  | { kind: 'blob'; blob: Blob; mime: string };

// Threshold above which writeBytes prefers Blob-backed storage. Tuned
// from the (rough) point where a base64 string starts eating noticeable
// JS heap. Adjust if you see real-world workflows OOMing on outputs
// below this size.
const VFS_BLOB_THRESHOLD = 4 * 1024 * 1024; // 4 MiB

const webVfs = new Map<string, WebVfsEntry>();

function isIntermediatePath(path: string): boolean {
  const p = path.toLowerCase();
  return (
    p.includes('/temp/') ||
    p.includes('/tmp/') ||
    p.includes('/cache/') ||
    p.includes('/intermediate/') ||
    // SQLite database storage (per-flow .sqlite files + the .dbdir /
    // .dbmarker placeholders the database service writes to force the
    // directory to exist). On desktop these are real disk files; in
    // the browser they live in the OPFS-backed VFS that the SQL shim
    // owns. They are never user-facing outputs — skip the download.
    p.includes('/databases/') ||
    p.endsWith('.sqlite') ||
    p.endsWith('.sqlite-journal') ||
    p.endsWith('/.dbdir') ||
    p.endsWith('/.dbmarker') ||
    /\/oaiy-[a-z]+-[a-z0-9]+\//.test(p) || // oaiy-* must be a temp DIRECTORY (trailing /), not an output filename like oaiy-render-001.png
    /[/\\]oaiy-llm-frame-/.test(p)
  );
}

function basename(path: string): string {
  const m = path.match(/[^/\\]+$/);
  return m ? m[0] : 'download';
}

// ---------------------------------------------------------------------------
// Persistent web storage for secrets + internal app-state files.
//
// The desktop build keeps API keys in an OS keyring and small app-state
// (user macros, built-in-macro overrides) in real files on disk. The browser
// has neither, so we back both with localStorage under dedicated keys. They
// are kept OUT of the exportable project blob — useProject redacts secret
// values before it serialises the project, and app-state lives under its own
// keys — so exporting or sharing a project never leaks API keys.
// ---------------------------------------------------------------------------
const SECRETS_KEY = 'oaiy_web_secrets';
const APPSTATE_PREFIX = 'oaiy_web_appstate:';

function readSecretMap(): Record<string, string> {
  try {
    const raw = localStorage.getItem(SECRETS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    // Reject arrays/null — typeof [] is 'object', and a stray array here
    // would surface as a bogus secret map.
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, string>)
      : {};
  } catch {
    return {};
  }
}

function writeSecretMap(map: Record<string, string>): void {
  try {
    localStorage.setItem(SECRETS_KEY, JSON.stringify(map));
  } catch (e) {
    console.warn('[oaiy-web] could not persist secret (storage full?)', e);
  }
}

// Known INTERNAL app-state files the app persists under the web app-data root
// ('/oaiy', from get_app_data_dir). These must survive a reload and must never
// trigger a browser download. Kept as an explicit allowlist (not a /oaiy/*.json
// glob) so a user's workflow OUTPUT written under /oaiy — e.g. a save node
// pointed at /oaiy/result.json — is NOT silently swallowed into localStorage
// instead of downloading. Add new internal filenames here as they appear.
const APPSTATE_FILES = new Set<string>(['user-macros.json']);

function isAppStatePath(path: string): boolean {
  // Root-level files in the app-data dir only (no subdirectories), matched
  // against the known-internal allowlist.
  const m = path.replace(/\\/g, '/').match(/^\/oaiy\/([^/]+)$/);
  return m !== null && APPSTATE_FILES.has(m[1]);
}

function mimeFromExt(filename: string): string {
  const ext = (filename.split('.').pop() || '').toLowerCase();
  switch (ext) {
    case 'png': return 'image/png';
    case 'jpg': case 'jpeg': return 'image/jpeg';
    case 'webp': return 'image/webp';
    case 'gif': return 'image/gif';
    case 'svg': return 'image/svg+xml';
    case 'mp4': return 'video/mp4';
    case 'webm': return 'video/webm';
    case 'mov': return 'video/quicktime';
    case 'mp3': return 'audio/mpeg';
    case 'wav': return 'audio/wav';
    case 'ogg': return 'audio/ogg';
    case 'flac': return 'audio/flac';
    case 'json': return 'application/json';
    case 'txt': case 'md': return 'text/plain';
    case 'html': return 'text/html';
    case 'csv': return 'text/csv';
    default: return 'application/octet-stream';
  }
}

function triggerDownloadFromBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Browser needs a beat to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/** Materialise a WebVfsEntry into a Blob for download / object-URL use. */
function entryToBlob(entry: WebVfsEntry, filename: string): Blob {
  if (entry.kind === 'blob') return entry.blob;
  if (entry.contentType === 'base64') {
    try {
      const bin = atob(entry.content);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new Blob([bytes], { type: mimeFromExt(filename) });
    } catch (e) {
      console.warn('[oaiy-web] entryToBlob: malformed base64, falling back to text', e);
      return new Blob([entry.content], { type: 'text/plain' });
    }
  }
  return new Blob([entry.content], { type: mimeFromExt(filename) || 'text/plain' });
}

/**
 * Read a Blob into the {content, contentType} shape desktop's read_file
 * returns. Used by the blob:/data: branch of read_file for both live
 * URLs (fetched out of the URL registry) and persisted blobs recovered
 * from IndexedDB after a page reload.
 */
/**
 * Materialise a Blob into a string-backed WebVfsEntry. This shape is
 * what `read_file` callers consume — they unwrap `.content` as a string,
 * so a Blob-kind entry here would surface as `undefined` and break the
 * pipeline.
 *
 * For write-side flows that DO want the zero-copy Blob carry (large
 * native_copy_file outputs, VfsBridge.writeBytes for ffmpeg outputs)
 * see `blobToEntryPreferBlob` below.
 */
async function blobToEntry(blob: Blob, requestedType: string): Promise<WebVfsEntry> {
  if (requestedType === 'base64') {
    const buf = await blob.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    const CHUNK = 8192;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      const slice = bytes.subarray(i, i + CHUNK);
      // eslint-disable-next-line prefer-spread
      bin += String.fromCharCode.apply(null, Array.from(slice));
    }
    return { kind: 'string', content: btoa(bin), contentType: 'base64' };
  }
  return { kind: 'string', content: await blob.text(), contentType: 'utf-8' };
}

/**
 * Variant of blobToEntry that prefers a Blob-kind entry for large
 * binary payloads. Only safe to use at WRITE sites — entries returned
 * from this function may have `kind: 'blob'` and no `content` string,
 * which breaks read_file consumers expecting `{ content }`.
 *
 * Used by native_copy_file (large file copies) so the destination
 * entry shares the source Blob handle without round-tripping bytes
 * through a base64 string.
 */
async function blobToEntryPreferBlob(blob: Blob, requestedType: string): Promise<WebVfsEntry> {
  if (requestedType === 'base64' && blob.size >= VFS_BLOB_THRESHOLD) {
    return { kind: 'blob', blob, mime: blob.type || 'application/octet-stream' };
  }
  return blobToEntry(blob, requestedType);
}

/**
 * Adapter from the renderer's in-memory webVfs / blob: URLs / IndexedDB
 * blob store into the byte-oriented interface the ffmpeg.wasm shim
 * expects. Built lazily per-invoke so the bridge always sees the latest
 * VFS state (matters for chains where one node writes a temp file and
 * the next reads it).
 *
 * readBytes resolution order for a path:
 *   1. webVfs key — fast path, no async work
 *   2. blob: URL — try a live fetch, then fall back to the IDB blob
 *      store recovery (uploaded in a previous session)
 *   3. data: URL — fetch directly
 *
 * writeBytes always stores as `base64` because ffmpeg outputs are
 * binary; the same round-trip the dialog shim uses for upload blobs.
 */
function buildVfsBridge(): import('./ffmpeg').VfsBridge {
  return {
    has: (p) => webVfs.has(p),
    keys: () => Array.from(webVfs.keys()),
    readBytes: async (p) => {
      const hit = webVfs.get(p);
      if (hit) {
        if (hit.kind === 'blob') {
          const buf = await hit.blob.arrayBuffer();
          return new Uint8Array(buf);
        }
        if (hit.contentType === 'base64') {
          try {
            const bin = atob(hit.content);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            return bytes;
          } catch {
            return null;
          }
        }
        return new TextEncoder().encode(hit.content);
      }
      // Path isn't a VFS key — try as a URL (file picker handed us a
      // blob: or data: URL, or a runtime synthesised one).
      if (p.startsWith('blob:') || p.startsWith('data:')) {
        try {
          const resp = await fetch(p);
          if (resp.ok) {
            const buf = await resp.arrayBuffer();
            return new Uint8Array(buf);
          }
        } catch {
          // fall through to IDB recovery
        }
        if (p.startsWith('blob:')) {
          try {
            const { getBlob } = await import('../utils/blobStore');
            const blob = await getBlob(p);
            if (blob) {
              const buf = await blob.arrayBuffer();
              return new Uint8Array(buf);
            }
          } catch {
            // ignored
          }
        }
      }
      return null;
    },
    getBlob: async (p) => {
      // First: webVfs entry that was written as a Blob (a previous
      // ffmpeg node's large output, an image hand-off, etc.) — hand
      // back the Blob handle directly. Zero copy for chained ffmpeg.
      const hit = webVfs.get(p);
      if (hit && hit.kind === 'blob') return hit.blob;
      // Only blob:/data: URLs can become a real disk-backed Blob without
      // materialising bytes. Small string-backed webVfs entries fall
      // through to the readBytes copy path.
      if (!p.startsWith('blob:') && !p.startsWith('data:')) return null;
      // Live URL first — Chrome/Firefox keep the underlying Blob in the
      // URL registry so fetch is a cheap reference-grab, no full read.
      try {
        const resp = await fetch(p);
        if (resp.ok) return await resp.blob();
      } catch {
        // dead URL — fall through to IDB recovery
      }
      if (p.startsWith('blob:')) {
        try {
          const { getBlob } = await import('../utils/blobStore');
          const blob = await getBlob(p);
          if (blob) return blob;
        } catch {
          // ignored
        }
      }
      return null;
    },
    writeBytes: (p, bytes, contentType) => {
      // For large outputs, store as a Blob — keeps the bytes out of the
      // base64 string heap (which adds 33% + caps near the JS string
      // length limit) and gives subsequent ffmpeg nodes a zero-copy
      // WORKERFS-mountable input.
      let entry: WebVfsEntry;
      if (bytes.length >= VFS_BLOB_THRESHOLD) {
        const mime = contentType && contentType !== 'base64'
          ? contentType
          : mimeFromExt(basename(p));
        // Slice into a new ArrayBuffer to detach from any callerheld
        // reference, then wrap in a Blob.
        const detached = new Uint8Array(bytes.length);
        detached.set(bytes);
        const blob = new Blob([detached], { type: mime });
        entry = { kind: 'blob', blob, mime };
      } else {
        let bin = '';
        const CHUNK = 8192;
        for (let i = 0; i < bytes.length; i += CHUNK) {
          const slice = bytes.subarray(i, i + CHUNK);
          bin += String.fromCharCode.apply(null, Array.from(slice));
        }
        entry = { kind: 'string', content: btoa(bin), contentType: 'base64' };
      }
      webVfs.set(p, entry);
      if (!isIntermediatePath(p)) {
        try {
          triggerDownloadFromBlob(basename(p), entryToBlob(entry, basename(p)));
        } catch (e) {
          console.warn('[oaiy-web] vfs-bridge: download failed', e);
        }
      }
    },
  };
}

const handlers: Record<string, (args: InvokeArgs) => Promise<unknown>> = {
  // The runtime's HTTP proxy → the external-engine path.
  http_request: httpRequest,

  // Desktop API server status. `allow_no_auth: true` is the desktop
  // bypass signal — the SplashScreen's own auto-onStart drives the
  // web-build transition instead, and SplashScreen.useEffect handles
  // the brief visible window before dismiss.
  get_api_status: async () => ({ allow_no_auth: true, running: false, port: null }),
  get_api_port: async () => null,

  // Desktop-only config/state commands → browser-safe shapes (callers expect
  // objects/arrays, not null; returning null caused ".isRunMode of null" /
  // "not iterable" crashes during startup).
  get_cli_run_config: async () => ({ isRunMode: false }),
  get_install_config: async () => ({}),
  get_default_app_data_dir: async () => '/oaiy',
  // Secrets — desktop uses an OS keyring; the web build persists them to
  // localStorage under a dedicated key so API keys survive a page reload.
  // Without these handlers `store_secret` was a silent no-op and every
  // secret constant was blanked by the project autosave's redaction, so
  // keys vanished on the next refresh.
  get_secrets: async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const keys = Array.isArray(a.keys) ? (a.keys as string[]) : [];
    const map = readSecretMap();
    const out: Record<string, string> = {};
    for (const k of keys) {
      const v = map[k];
      if (typeof v === 'string' && v.length > 0) out[k] = v;
    }
    return out;
  },
  store_secret: async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const key = String(a.key ?? '');
    if (!key) return null;
    const value = a.value == null ? '' : String(a.value);
    const map = readSecretMap();
    // Empty value = delete (deleteConstant overwrites with '' since there's
    // no delete_secret command).
    if (value === '') delete map[key];
    else map[key] = value;
    writeSecretMap(map);
    return null;
  },
  load_all_macros: async () => [],

  // Filesystem plugin commands. Backed by the in-memory VFS above, with
  // a side-effect download for paths that look like user-facing outputs.
  'plugin:oaiy-filesystem|get_app_data_dir': async () => '/oaiy',
  // Runtimes (audio_append, video_append, video_captions, …) write
  // intermediate files (concat lists, subtitle sidecars) into the temp
  // dir then point ffmpeg at them. webVfs is a single flat namespace, so
  // any unique prefix works — we use `/tmp` to match the Linux-ish feel.
  'plugin:oaiy-filesystem|get_temp_dir': async () => '/tmp',
  // Listing a real folder needs a filesystem the browser doesn't have. The
  // generic fallback returned null, which the listFolder runtime then iterated
  // → opaque NPE. Throw an actionable error instead (the runtime catches it
  // and logs it to the execution log).
  'plugin:oaiy-filesystem|list_folder': async () => {
    throw new Error('Listing folder contents is not supported in the browser build — pick a single file (File Read) instead, or use the desktop app for folder access.');
  },
  'plugin:oaiy-filesystem|write_file': async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const path = String(a.path ?? '');
    const content = String(a.content ?? '');
    const contentType = String(a.contentType ?? 'utf-8');
    if (!path) return null;
    // write_file from JS runtimes is always string content (base64 or
    // utf-8); large binary outputs flow through the VfsBridge.writeBytes
    // path which already picks Blob vs string. Keep this handler as
    // string-only for back-compat.
    const ct: 'base64' | 'utf-8' = contentType === 'base64' ? 'base64' : 'utf-8';
    const entry: WebVfsEntry = { kind: 'string', content, contentType: ct };
    webVfs.set(path, entry);
    if (isAppStatePath(path)) {
      // Internal app-state (e.g. /oaiy/user-macros.json) — persist across
      // reloads and NEVER download. Previously these writes both vanished
      // on refresh (in-memory VFS) and popped a browser download on every
      // ~2s macro autosave.
      try {
        localStorage.setItem(APPSTATE_PREFIX + path, JSON.stringify(entry));
      } catch (e) {
        console.warn('[oaiy-web] write_file: app-state persist failed (storage full?)', e);
        // Surface a quota toast so the user isn't silently losing macros —
        // dynamic import keeps this low-level shim free of a load-order dep.
        void import('../lib/storageQuota')
          .then((m) => m.notifyStorageQuotaExceeded('macros', e))
          .catch(() => { /* storageQuota unavailable — console.warn above stands */ });
      }
    } else if (!isIntermediatePath(path)) {
      try { triggerDownloadFromBlob(basename(path), entryToBlob(entry, basename(path))); }
      catch (e) { console.warn('[oaiy-web] write_file: download failed', e); }
    }
    return null;
  },
  'plugin:oaiy-filesystem|delete_file': async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const path = String(a.path ?? '');
    webVfs.delete(path);
    if (isAppStatePath(path)) {
      try { localStorage.removeItem(APPSTATE_PREFIX + path); } catch { /* ignore */ }
    }
    return null;
  },
  // native_copy_file: desktop uses a fast OS-level copy; we fake it by
  // doing read+write through the VFS. Returns the byte count (as a
  // sentinel — desktop returns u64; here it's whatever we wrote). Many
  // chain nodes (audio_fade output → save_audio, video_save → user
  // download) rely on this; without a shim they fail at runtime.
  'plugin:oaiy-filesystem|native_copy_file': async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const src = String(a.src ?? a.from ?? a.source ?? '');
    const dst = String(a.dst ?? a.to ?? a.destination ?? '');
    if (!src || !dst) throw new Error('native_copy_file: src and dst are required');
    // Source can be a blob:/data: URL or a VFS path. Read it via the
    // same path read_file uses so chained pipelines just work.
    let entry: WebVfsEntry | null = null;
    if (src.startsWith('blob:') || src.startsWith('data:')) {
      try {
        const resp = await fetch(src);
        if (resp.ok) entry = await blobToEntryPreferBlob(await resp.blob(), 'base64');
      } catch { /* fall through to IDB lookup */ }
      if (!entry && src.startsWith('blob:')) {
        try {
          const { getBlob } = await import('../utils/blobStore');
          const blob = await getBlob(src);
          if (blob) entry = await blobToEntryPreferBlob(blob, 'base64');
        } catch { /* ignore */ }
      }
    } else {
      entry = webVfs.get(src) ?? null;
    }
    if (!entry) throw new Error(`native_copy_file: source not readable (${src})`);
    webVfs.set(dst, entry);
    if (!isIntermediatePath(dst)) {
      try { triggerDownloadFromBlob(basename(dst), entryToBlob(entry, basename(dst))); }
      catch (e) { console.warn('[oaiy-web] native_copy_file: download failed', e); }
    }
    // Best-effort byte count for callers that log it. Blob.size or
    // string.length — both are roughly the source size on disk.
    return entry.kind === 'blob' ? entry.blob.size : entry.content.length;
  },
  // run_command: desktop spawns a sidecar binary. The only commands the
  // bundled core nodes ever pass are `ffmpeg` and `ffprobe` — both routed
  // to ffmpeg.wasm via the dedicated shim. Anything else returns a
  // structured failure so the calling runtime takes its error path
  // rather than throwing on undefined output.
  'plugin:oaiy-filesystem|run_command': async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const command = String(a.command ?? '');
    const cmdArgs = Array.isArray(a.args) ? (a.args as unknown[]).map((x) => String(x)) : [];

    if (command !== 'ffmpeg' && command !== 'ffprobe') {
      return {
        code: -1,
        stdout: '',
        stderr: `oaiy-web: run_command('${command}') is not supported in the browser build (only ffmpeg/ffprobe are routed to ffmpeg.wasm)`,
      };
    }

    // Bridge the in-memory VFS into the ffmpeg shim. The shim expects raw
    // bytes; webVfs stores either 'base64' content or 'utf-8' strings.
    const { runFFmpegCommand } = await import('./ffmpeg');
    return runFFmpegCommand(command, cmdArgs, buildVfsBridge());
  },

  // Frame extractor — desktop uses a dedicated `extract_video_frames`
  // Rust command (native/src/lib.rs). Here we route through the shared
  // ffmpeg.wasm singleton instead, returning the same FrameInfo[] shape
  // the desktop emits. Each emitted frame is also written to the VFS at
  // /tmp/frames_<ts>/frame_NNNNN.<ext> so downstream nodes can read it
  // back via the standard plugin:oaiy-filesystem|read_file path.
  extract_video_frames: async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const path = String(a.path ?? '');
    const options = (a.options ?? {}) as Record<string, unknown>;
    if (!path) throw new Error('extract_video_frames: path required');
    const { extractVideoFrames } = await import('./ffmpeg');
    return extractVideoFrames(
      path,
      {
        intervalSeconds: Number(options.intervalSeconds ?? 1),
        startTime: options.startTime as number | undefined,
        endTime: options.endTime as number | undefined,
        maxFrames: options.maxFrames as number | undefined,
        outputFormat: (options.outputFormat as string | undefined) ?? 'png',
        scaleWidth: options.scaleWidth as number | undefined,
        scaleHeight: options.scaleHeight as number | undefined,
      },
      buildVfsBridge(),
    );
  },
  // Batched variant — the desktop streams frames in groups for very long
  // videos. The browser path just delegates to extractVideoFrames using
  // the batch-specific time range, then synthesises the BatchResult
  // wrapper. `hasMore` is left false because we don't pre-probe duration
  // here; the runtime can call again with a higher batchIndex and we'll
  // emit zero frames once the window passes the video end.
  extract_video_frames_batch: async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const path = String(a.path ?? '');
    const intervalSeconds = Number(a.intervalSeconds ?? 1);
    const batchSize = Number(a.batchSize ?? 32);
    const batchIndex = Number(a.batchIndex ?? 0);
    const outputFormat = String(a.outputFormat ?? 'png');
    const scaleWidth = a.scaleWidth as number | undefined;
    const scaleHeight = a.scaleHeight as number | undefined;

    const startTime = batchIndex * batchSize * intervalSeconds;
    const endTime = (batchIndex + 1) * batchSize * intervalSeconds;

    const { extractVideoFrames } = await import('./ffmpeg');
    const frames = await extractVideoFrames(
      path,
      {
        intervalSeconds,
        startTime,
        endTime,
        maxFrames: batchSize,
        outputFormat,
        scaleWidth,
        scaleHeight,
      },
      buildVfsBridge(),
    );

    // Re-base frame timestamps + indices to the batch window so the
    // calling runtime sees globally-correct timestamps.
    for (let i = 0; i < frames.length; i++) {
      frames[i].index = batchIndex * batchSize + i;
      frames[i].timestamp = startTime + i * intervalSeconds;
    }

    return {
      frames,
      batchIndex,
      totalBatches: 0,         // unknown without probing duration; the
                               // runtime treats 0 as "indeterminate"
      totalFrames: frames.length,
      hasMore: frames.length === batchSize,
      nextStartTime: endTime,
    };
  },
  // Used by extractLastFrame in the video runtime — first calls
  // get_video_info to learn the duration, then asks for one frame near
  // the end. Re-implemented on top of the ffprobe shim's metadata parse.
  get_video_info: async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const path = String(a.path ?? '');
    if (!path) throw new Error('get_video_info: path required');
    const { getVideoInfoFromVfs } = await import('./ffmpeg');
    return getVideoInfoFromVfs(path, buildVfsBridge());
  },

  'plugin:oaiy-filesystem|read_file': async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const path = String(a.path ?? '');
    // The desktop Rust signature names this parameter `read_as`, which
    // Tauri's invoke converts to `readAs` on the JS side. Every caller
    // I've seen in core-* uses readAs. Accept the older `contentType`
    // alias too — the shim originally probed it (bug: blob inputs that
    // should have been base64-decoded came back as UTF-8 garbage for
    // binary files like images/audio/video). readAs wins when both
    // exist.
    const requestedType = String(a.readAs ?? a.contentType ?? 'utf-8');
    // blob: and data: URLs are both real fetchable URLs in the browser —
    // dialog.open() returns blob: URLs (memory-efficient by reference);
    // legacy projects + inline embeds use data: URLs. Fetch + decode to
    // the requested content type so callers see what the desktop would.
    if (path.startsWith('blob:') || path.startsWith('data:')) {
      // Try the live URL first. Survives ~all in-session reads.
      try {
        const resp = await fetch(path);
        if (resp.ok) return await blobToEntry(await resp.blob(), requestedType);
      } catch {
        // fall through to IndexedDB recovery below
      }
      // blob: URLs from a PREVIOUS session are dead in this tab. Look
      // the file up in the IndexedDB store the dialog shim wrote it to.
      if (path.startsWith('blob:')) {
        try {
          const { getBlob } = await import('../utils/blobStore');
          const blob = await getBlob(path);
          if (blob) return await blobToEntry(blob, requestedType);
        } catch (e) {
          console.warn('[oaiy-web] read_file: blobStore lookup failed', e);
        }
      }
      throw new Error(
        path.startsWith('blob:')
          ? 'read_file: blob URL is no longer valid (file may have been uploaded in a previous session and IndexedDB is unavailable or cleared)'
          : 'read_file: malformed data: URL',
      );
    }
    // Chained workflow intermediates land here via the VFS.
    const hit = webVfs.get(path);
    if (hit) {
      if (hit.kind === 'string') {
        return { content: hit.content, contentType: hit.contentType };
      }
      // Blob-backed entry — materialise to the {content, contentType}
      // shape the caller expects. The Blob may have lived on disk; this
      // is the point where the bytes come back into memory.
      return await blobToEntry(hit.blob, requestedType);
    }
    // Internal app-state files persist to localStorage — recover them after
    // a reload (the in-memory webVfs starts empty each session). This is what
    // makes user macros + macro overrides survive a refresh in the web build.
    if (isAppStatePath(path)) {
      try {
        const raw = localStorage.getItem(APPSTATE_PREFIX + path);
        if (raw) {
          const entry = JSON.parse(raw) as WebVfsEntry;
          webVfs.set(path, entry);
          if (entry.kind === 'string') {
            return { content: entry.content, contentType: entry.contentType };
          }
        }
      } catch (e) {
        console.warn('[oaiy-web] read_file: app-state recover failed', e);
      }
    }
    // Surface as "not found" so callers take their file-absent path
    // (localStorage fallback / empty list) rather than misparsing.
    throw new Error('read_file: not found (no filesystem in the web build)');
  },

  // Desktop-only media-server / lifecycle commands with no web equivalent.
  // Mapped explicitly so the generic router doesn't warn on every call.
  get_media_server_port: async () => null,
  get_media_token: async () => null,
  get_bundled_plugins_dir: async () => null,
  set_lifecycle_config: async () => null,

  // Translate a webVfs path (or blob:/data: URL) into something a <video>
  // / <audio> / <img> tag can `src=` to. On desktop this returns a
  // `http://127.0.0.1:PORT/media/TOKEN/...` URL served by the bundled
  // media server. Here we synthesise an Object URL backed by the
  // underlying Blob — works for the writeBytes Blob path AND the
  // base64-string entries (entryToBlob handles both).
  //
  // The URL is leaked (no revokeObjectURL) — for a workflow that emits a
  // single output it's fine; the browser frees the backing Blob when the
  // tab closes. If a chain produces hundreds of intermediates we'd want
  // a per-path cache + revoke-on-overwrite. Not worth the bookkeeping
  // until it becomes a problem.
  get_media_url: async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const filePath = String(a.filePath ?? a.file_path ?? a.path ?? '');
    if (!filePath) return null;
    // Already a fetchable URL? Hand back as-is so the same node can
    // round-trip outputs from external HTTP services or earlier blob:
    // entries without re-encoding.
    if (
      filePath.startsWith('blob:') ||
      filePath.startsWith('data:') ||
      filePath.startsWith('http://') ||
      filePath.startsWith('https://')
    ) {
      return filePath;
    }
    const hit = webVfs.get(filePath);
    if (!hit) {
      // No entry → caller will likely log "no media" — return the path
      // unchanged so the failure mode is "404 on src=" rather than
      // "undefined string crashed the JSX".
      return filePath;
    }
    try {
      const blob = entryToBlob(hit, basename(filePath));
      return URL.createObjectURL(blob);
    } catch (e) {
      console.warn(`[oaiy-web] get_media_url: blob URL synthesis failed for ${filePath}`, e);
      return filePath;
    }
  },

  // Desktop returns the user's Downloads folder. In the browser there
  // isn't one we can address; node code uses the value as a parent path
  // for an output, and the actual delivery happens via the in-shim
  // download trigger (triggerDownloadFromBlob). Returning a synthetic
  // `/downloads` is enough — the resulting path lives in the webVfs
  // namespace and gets downloaded via the user's browser settings.
  get_downloads_path: async () => '/downloads',
  'plugin:oaiy-filesystem|get_downloads_path': async () => '/downloads',

  // Service-spawning commands. Local AI services (Ollama, DramaBox,
  // ACE-Step, ComfyUI etc.) can't be started from a browser tab, so we
  // return a structured "not available" failure that matches the
  // EnsureServiceResult shape the calling runtimes destructure. Without
  // this, `result.success` would NPE on `null`.
  ensure_service_ready: async () => ({
    success: false,
    already_running: false,
    error: 'Local service spawn is not available in the browser build',
  }),
  // The OAIY Companion (fixed localhost API :17972) may own this port. Ask it to
  // start the matching service if stopped, so picking a companion service in a
  // flow and running it "just works" without a manual Start first (Phase 3.5).
  // Fail-safe: any error, a non-OK response, or an absent companion returns
  // success:false — exactly the old no-op — so the caller falls through to a
  // direct request. 2s cap: ensure-by-port returns as soon as the spawn kicks
  // off, so a healthy companion answers in <100ms; a wedged one can't hang the run.
  ensure_service_ready_by_port: async (args) => {
    const port = (args as { port?: number } | undefined)?.port;
    if (typeof port !== 'number') {
      return { success: false, already_running: false, error: 'no port' };
    }
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 2000);
      const r = await fetch('http://127.0.0.1:17972/api/services/ensure-by-port', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ port }),
        credentials: 'omit',
        signal: ctrl.signal,
      }).finally(() => clearTimeout(timer));
      if (!r.ok) return { success: false, already_running: false };
      // The companion serializes EnsureByPortResult in camelCase (alreadyRunning).
      const res = (await r.json().catch(() => null)) as
        | { found?: boolean; started?: boolean; alreadyRunning?: boolean }
        | null;
      return {
        success: !!(res?.started || res?.alreadyRunning),
        already_running: !!res?.alreadyRunning,
      };
    } catch {
      // Companion absent / not reachable, or a non-companion local server.
      return { success: false, already_running: false };
    }
  },

  // Batched frame extractor uses tempBatchFolders to track folders for
  // cleanup. Our extractor writes to webVfs, not a real disk dir, so
  // there's nothing to recursively delete — accept the call and no-op.
  cleanup_temp_dir: async () => null,

  // Package-service lifecycle (desktop spawns the package's sidecar
  // binaries). The browser has no process manager, so the panel + the
  // run-modal preflight both treat "empty services array" as "nothing
  // to start" and the rest of the UI works unchanged. Stubbed here so
  // the 3-second status poll doesn't log a warn per tick.
  get_package_services: async () => [],
  extract_package_service: async () => '',
  start_package_service: async () => ({ id: '', running: false, healthy: false }),
  stop_package_services: async () => null,
  get_service_logs: async () => ({ logs: [] }),

  // Install-config write — paired with `get_install_config` above. No
  // persistent install state in the browser; we accept the call and
  // ignore it so the SettingsPanel save handlers don't warn.
  set_install_config: async () => null,

  // File + folder pickers — wired to the dialog shim's HTML-input fallback
  // so the audio/video/folder node click handlers (which all reach Tauri
  // via `pick_file` / `pick_folder` first and only fall back to a hidden
  // <input> when the call fails) get a usable picker in the browser too.
  //
  // `pick_file` returns a `blob:` URL string the rest of the pipeline
  // already treats as a path (read_file in this file fetches the blob
  // back out). `pick_folder` returns a friendly folder name; the picked
  // files live in IDB under `folder://<name>/<relpath>` keys for a future
  // listFolder web runtime to enumerate.
  'plugin:oaiy-filesystem|pick_file': async (args) => {
    const a = (args ?? {}) as Record<string, unknown>;
    const options: OpenDialogOptions = {};
    if (Array.isArray(a.filters)) options.filters = a.filters as OpenDialogOptions['filters'];
    if (typeof a.multiple === 'boolean') options.multiple = a.multiple;
    const picked = await pickFileViaInput(options);
    // Tauri's pick_file always returns a single string-or-null, even with
    // multi-select. We mirror that contract: hand back the first url.
    if (Array.isArray(picked)) return picked[0] ?? null;
    return picked;
  },
  'plugin:oaiy-filesystem|pick_folder': pickFolderViaInput,
};

export async function invoke<T = unknown>(cmd: string, args?: InvokeArgs): Promise<T> {
  const handler = handlers[cmd];
  if (handler) return (await handler(args)) as T;
  // `list_*` commands (services, plugins, bundled plugins, …) are expected to
  // return arrays by their callers — default to empty so `for…of`/`.map`/`.length`
  // don't throw during startup.
  if (cmd.startsWith('list_')) return [] as T;
  // Unmapped command: don't crash the render — warn and return null. Features
  // that need this command degrade; we map them as the web build matures.
  console.warn(`[oaiy-web] invoke('${cmd}') is not available in the browser build — returning null.`, args);
  return null as T;
}

/** Tauri's asset-URL helper. In the browser, paths/URLs are used as-is. */
export function convertFileSrc(filePath: string): string {
  return filePath;
}

/** Minimal stand-in for Tauri's internal callback registration. */
function transformCallback(callback?: (response: unknown) => void): number {
  const id = Math.floor(Math.random() * 2 ** 31);
  const w = window as any;
  w.__TAURI_CB__ = w.__TAURI_CB__ || {};
  if (callback) w.__TAURI_CB__[id] = callback;
  return id;
}

// ---------------------------------------------------------------------------
// Install globals (side effect on import)
// ---------------------------------------------------------------------------

const w = window as any;
if (!w.__TAURI_INTERNALS__) {
  w.__TAURI_INTERNALS__ = { invoke, transformCallback, convertFileSrc };
}
if (!w.__TAURI__) {
  w.__TAURI__ = { core: { invoke, convertFileSrc } };
}
// Marker the rest of the app reads (via lib/platform.isWebBuild) to gate
// features that need native capabilities the browser can't provide — e.g.
// unzipping/verifying .oaiy packages or scanning the filesystem for them.
// A real desktop (Tauri) build never loads this shim, so the flag stays unset.
w.__OAIY_WEB_SHIM__ = true;
