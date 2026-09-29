/**
 * The flow editor's service worker: what it keeps, what it never touches, how it updates.
 *
 * The worker script (`sw.ts`) only hands the real worker scope to `installWorker`, so
 * everything here can be tested with a stand-in scope (tests/sw-core.mjs) and the same
 * limits are read by the build (scripts/serviceWorkerPlugin.ts) and by the page.
 *
 * What it is for. The editor works with no server of its own, so once a browser has the
 * page it can open it again with no network: the HTML and the JavaScript and CSS the page
 * needs to start (the "shell") are kept when the worker installs, and everything else the
 * page fetches from its own site is kept as it goes by. That is all it does. It adds no
 * headers (the editor runs without COOP and COEP on purpose: the SQLite OPFS pool and the
 * engine calls depend on it) and it never answers for anyone else.
 *
 * Rules, each with a test:
 *
 *   - It only ever answers a GET for its own origin. Everything else is left to the
 *     browser untouched: other origins (which is every engine on 127.0.0.1 or localhost,
 *     every provider and every CDN), `/api/`, a `?flow=` share link, a Range request, and
 *     the worker's own script.
 *   - Nothing over MAX_CACHED_BYTES is kept, measured on the bytes of the body as the page
 *     gets them (a host that compresses sends a smaller Content-Length than the file). The
 *     ffmpeg core (32 MB), the esbuild and ZIPP engines and the language workers are far
 *     over it: they come from the network every time they are needed. The build lists the
 *     files it knows are over it, and the worker does not even read those.
 *   - HTML is network-first (a new deploy shows at once, and with no network the kept shell
 *     opens); everything else is stale-while-revalidate.
 *   - The caches are named for the build. A new build installs beside the old one and waits
 *     until the person says so (a message from the page); the old caches are removed only when
 *     the new worker takes over, and only the ones this worker made (CACHE_PREFIX).
 */

/** The most one kept response may hold. One number, read by the worker, the build and the page. */
export const MAX_CACHED_BYTES = 4 * 1024 * 1024;

/** Every cache this worker makes starts with this, so it never removes anyone else's. */
export const CACHE_PREFIX = 'oaiy-web-';

/** The page the worker is for, and the one kept for every navigation it answers. */
export const SHELL_PATH = '/app.html';

/** The worker's own script: never kept. */
export const WORKER_PATH = '/sw.js';

/** How long a network-first navigation waits before it opens the kept shell instead. */
export const NAVIGATION_TIMEOUT_MS = 4000;

/** The most addresses one "keep these" message from the page is allowed to name. */
export const MAX_WARM_URLS = 300;

/** What the build gives the worker. */
export interface WorkerConfig {
  /** Names the caches; changes with every build that changes the shell. */
  buildId: string;
  /** The shell: paths from the site root, the page first. */
  precache: readonly string[];
  /** Files of the build known to be over MAX_CACHED_BYTES: left to the network, never read. */
  oversize: readonly string[];
}

/** The messages the page sends the worker. */
export type WorkerMessage =
  | { type: 'skip-waiting' }
  | { type: 'cache-urls'; urls: { url: string; size?: number }[] };

export interface CacheNames {
  shell: string;
  runtime: string;
}

export function cacheNames(buildId: string): CacheNames {
  return { shell: `${CACHE_PREFIX}${buildId}-shell`, runtime: `${CACHE_PREFIX}${buildId}-runtime` };
}

/** Why the worker leaves a request to the browser. */
export type BypassReason = 'method' | 'range' | 'cross-origin' | 'api' | 'share-link' | 'worker-script' | 'oversize';

export type Route = { kind: 'bypass'; reason: BypassReason } | { kind: 'navigation' } | { kind: 'asset' };

/** The part of a Request `route` reads. */
export interface RouteRequest {
  url: string;
  method: string;
  mode?: string;
  headers: { has(name: string): boolean };
}

/**
 * What to do with a request. `workerOrigin` is the origin the worker was loaded from:
 * "another origin" is decided against it, never against a list of host names, so a site
 * served from 127.0.0.1 (a test, a preview) is still cached while an engine on the same
 * machine (another port, so another origin) is not.
 */
export function route(req: RouteRequest, workerOrigin: string, oversize: readonly string[] = []): Route {
  const bypass = (reason: BypassReason): Route => ({ kind: 'bypass', reason });
  if (req.method !== 'GET') return bypass('method');
  if (req.headers.has('range')) return bypass('range');
  let url: URL;
  try {
    url = new URL(req.url);
  } catch {
    return bypass('cross-origin');
  }
  if (url.origin !== workerOrigin) return bypass('cross-origin');
  if (url.pathname === '/api' || url.pathname.startsWith('/api/')) return bypass('api');
  if (url.searchParams.has('flow')) return bypass('share-link');
  if (url.pathname === WORKER_PATH) return bypass('worker-script');
  if (oversize.includes(url.pathname)) return bypass('oversize');
  if (req.mode === 'navigate') return { kind: 'navigation' };
  return { kind: 'asset' };
}

/** A response the worker may keep: a plain 200 from its own origin. */
export function isStorable(res: Response): boolean {
  return res.status === 200 && (res.type === 'basic' || res.type === 'default');
}

/**
 * The body of `response` as bytes, or null when it is longer than `max`. Counted on the
 * bytes as they are read (decoded, as the page gets them), so a compressed file cannot pass
 * on the strength of a small Content-Length; a Content-Length already over the limit says
 * no without reading anything. Reading stops at the limit, so a huge file is never held.
 */
export async function readWithinLimit(response: Response, max: number = MAX_CACHED_BYTES): Promise<Uint8Array | null> {
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > max) {
    void response.body?.cancel().catch(() => undefined);
    return null;
  }
  if (!response.body) return new Uint8Array(0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > max) {
      // Not awaited: the copy of a response is one branch of a tee, and cancelling a branch
      // settles only when the other one is cancelled or read too (which the caller does next).
      void reader.cancel().catch(() => undefined);
      return null;
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.byteLength;
  }
  return bytes;
}

/**
 * Keep a copy of `response` under `key`, if its body is within `max`. The response itself is
 * left untouched for whoever asked for it (the copy is taken before this first waits), so it
 * can be handed to the page straight away. The copy holds the body as the page got it, so the
 * headers that described the encoding on the wire are dropped; everything else (the type, and
 * a Content-Security-Policy the host sent for a worker's script) is kept exactly.
 */
export async function putWithinLimit(cache: Cache, key: string, response: Response, max: number = MAX_CACHED_BYTES): Promise<boolean> {
  const copy = response.clone();
  const bytes = await readWithinLimit(copy, max);
  if (!bytes) return false;
  const headers = new Headers(response.headers);
  headers.delete('content-encoding');
  headers.delete('content-length');
  await cache.put(key, new Response(bytes as BlobPart, { status: response.status, statusText: response.statusText, headers }));
  return true;
}

/* -------------------------------------------------------------------------- */
/* The worker                                                                  */
/* -------------------------------------------------------------------------- */

/** The parts of ExtendableEvent, FetchEvent and ExtendableMessageEvent the worker uses. */
export interface ExtendableEventLike {
  waitUntil(promise: Promise<unknown>): void;
}
export interface FetchEventLike extends ExtendableEventLike {
  request: Request;
  respondWith(response: Response | Promise<Response>): void;
}
export interface MessageEventLike extends ExtendableEventLike {
  data: unknown;
}

/** The parts of the worker's global scope the worker uses. */
export interface WorkerScopeLike {
  location: { origin: string };
  caches: CacheStorage;
  clients: { claim(): Promise<void> };
  fetch: typeof fetch;
  skipWaiting(): Promise<void>;
  addEventListener(type: 'install' | 'activate', listener: (event: ExtendableEventLike) => void): void;
  addEventListener(type: 'fetch', listener: (event: FetchEventLike) => void): void;
  addEventListener(type: 'message', listener: (event: MessageEventLike) => void): void;
}

export interface WorkerDeps {
  /** How long a navigation waits on the network before it opens the kept shell. */
  navigationTimeoutMs?: number;
  /** Resolves after `ms` (a test replaces it). */
  wait?: (ms: number) => Promise<void>;
}

const NO_HEADERS = { has: () => false };
const MATCH = { ignoreVary: true } as const;

/** Extend the event's life, unless it has already ended (then the work simply carries on). */
function keepAlive(event: ExtendableEventLike, work: Promise<unknown>): void {
  try {
    event.waitUntil(work);
  } catch {
    /* the event is over; the work is not cancelled by that */
  }
}

/** Wire the worker's events. */
export function installWorker(scope: WorkerScopeLike, config: WorkerConfig, deps: WorkerDeps = {}): void {
  const names = cacheNames(config.buildId);
  const origin = scope.location.origin;
  const absolute = (path: string) => new URL(path, origin).href;
  const timeoutMs = deps.navigationTimeoutMs ?? NAVIGATION_TIMEOUT_MS;
  const wait = deps.wait ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const noop = () => undefined;

  /**
   * The shell: kept, or the install fails and the old worker carries on. The page is fetched
   * fresh (its name never changes); every other file has a hashed name, so what the browser
   * already holds under it is the file, and is not downloaded a second time.
   */
  async function precache(): Promise<void> {
    const cache = await scope.caches.open(names.shell);
    await Promise.all(
      config.precache.map(async (path) => {
        const res = await scope.fetch(new Request(absolute(path), path === SHELL_PATH ? { cache: 'reload' } : undefined));
        if (!isStorable(res)) throw new Error(`the shell file ${path} answered ${res.status}`);
        if (!(await putWithinLimit(cache, absolute(path), res))) throw new Error(`the shell file ${path} is over ${MAX_CACHED_BYTES} bytes`);
      }),
    );
  }

  /** Remove the caches of other builds (only ours, by prefix), then take over the open pages. */
  async function activate(): Promise<void> {
    const keep = new Set<string>([names.shell, names.runtime]);
    for (const name of await scope.caches.keys()) {
      if (name.startsWith(CACHE_PREFIX) && !keep.has(name)) await scope.caches.delete(name);
    }
    await scope.clients.claim();
  }

  /** Something the page already has, kept as the runtime cache would have kept it. */
  async function keepLoaded(items: { url: string; size?: number }[]): Promise<void> {
    const runtime = await scope.caches.open(names.runtime);
    const shell = await scope.caches.open(names.shell);
    for (const item of items.slice(0, MAX_WARM_URLS)) {
      try {
        if (!item || typeof item.url !== 'string') continue;
        const url = new URL(item.url, origin).href;
        const decision = route({ url, method: 'GET', mode: 'no-cors', headers: NO_HEADERS }, origin, config.oversize);
        if (decision.kind !== 'asset') continue;
        if (typeof item.size === 'number' && item.size > MAX_CACHED_BYTES) continue;
        if ((await shell.match(url, MATCH)) || (await runtime.match(url, MATCH))) continue;
        const res = await scope.fetch(url);
        if (isStorable(res)) await putWithinLimit(runtime, url, res);
        void res.body?.cancel().catch(noop);
      } catch {
        /* one that cannot be fetched is not kept; the rest are */
      }
    }
  }

  /** HTML: the network first, the kept shell when there is no network (or it is slow, or failing). */
  async function navigation(event: FetchEventLike): Promise<Response> {
    const shell = await scope.caches.open(names.shell);
    const kept = await shell.match(absolute(SHELL_PATH), { ignoreSearch: true, ...MATCH });
    const network = scope.fetch(event.request).then((res) => {
      if (res.ok && (res.headers.get('content-type') ?? '').includes('text/html')) {
        keepAlive(event, putWithinLimit(shell, absolute(SHELL_PATH), res).catch(() => false));
      }
      return res;
    });
    // Keep the worker alive until the network answers, even if the shell was served first.
    keepAlive(event, network.catch(noop));
    if (!kept) return network;
    const first = await Promise.race([
      network.then(
        (res) => ({ res }),
        () => ({ res: null }),
      ),
      wait(timeoutMs).then(() => ({ res: null, timedOut: true })),
    ]);
    if (first.res && (first.res.ok || first.res.status < 500)) return first.res;
    return kept;
  }

  /** Everything else of ours: the kept copy at once, refreshed behind it; the network if there is none. */
  async function asset(event: FetchEventLike): Promise<Response> {
    const request = event.request;
    const shell = await scope.caches.open(names.shell);
    const runtime = await scope.caches.open(names.runtime);
    const kept = (await shell.match(request.url, MATCH)) ?? (await runtime.match(request.url, MATCH));
    if (kept) {
      keepAlive(
        event,
        (async () => {
          const res = await scope.fetch(request);
          if (isStorable(res)) await putWithinLimit(runtime, request.url, res);
          void res.body?.cancel().catch(noop);
        })().catch(noop),
      );
      return kept;
    }
    const res = await scope.fetch(request);
    if (isStorable(res)) keepAlive(event, putWithinLimit(runtime, request.url, res).catch(() => false));
    return res;
  }

  // A first install is not given a head start on the page that is open: skipWaiting is never
  // called here. A new build waits for the person (the message below).
  scope.addEventListener('install', (event) => {
    event.waitUntil(precache());
  });

  scope.addEventListener('activate', (event) => {
    event.waitUntil(activate());
  });

  scope.addEventListener('message', (event) => {
    const data = event.data as Partial<WorkerMessage> | null;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'skip-waiting') {
      event.waitUntil(scope.skipWaiting());
    } else if (data.type === 'cache-urls' && Array.isArray(data.urls)) {
      event.waitUntil(keepLoaded(data.urls));
    }
  });

  scope.addEventListener('fetch', (event) => {
    const decision = route(event.request, origin, config.oversize);
    if (decision.kind === 'bypass') return; // not answered: the browser does what it would have done
    event.respondWith(decision.kind === 'navigation' ? navigation(event) : asset(event));
  });
}
