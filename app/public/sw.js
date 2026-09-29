/*
 * The OAIY Agent's service worker.
 *
 * 1. Offline: the app shell and the Zipp engine are cached; everything
 *    same-origin is served from the cache and refreshed in the background.
 * 2. Isolation: every response it serves carries COOP/COEP, so the page is
 *    cross-origin isolated (SharedArrayBuffer, which the sandbox blocks on)
 *    even on a static host that cannot set headers.
 *
 * Requests to other origins (AI providers, the sandbox's gated fetches) are
 * never cached or touched.
 */
// The build stamps its own name here (vite.config.ts), so a new version gets a
// fresh cache: the app, the Zipp engine and the rest always come from the same build.
const CACHE = 'oaiy-agent-v1';
// The caches this worker owns: its own, the ones of earlier builds, and the
// ones the app had before it was called OAIY. Nothing else on the origin is touched.
const OWNED = ['oaiy-agent-', 'bot.computer-'];
const SCOPE = new URL(self.registration.scope);
const PRECACHE = ['./', './index.html', './manifest.webmanifest', './icon.svg', './icon-192.png', './icon-512.png', './zipp/zipp_wasm.js', './zipp/zipp_wasm_bg.wasm'];
// The entry a worker writes into its own cache when it takes over: when it did, so the next one knows which build ran before it.
const TOOK_OVER = new URL('./__oaiy-took-over', SCOPE).href;

const owned = (name) => OWNED.some((prefix) => name.startsWith(prefix));

// A new version installs beside the running one and waits, and the page says a new one is ready.
// The worker takes over when the person chooses to reload (the 'skip-waiting' message below), or
// when no tab is left. The very first worker has nothing to wait for and starts at once.
// While it waits, the running version keeps its own cache. Once it has taken over, the tabs that are
// still on the earlier version find their files in that version's cache (see `fetch`), which is kept
// until the version after this one takes over.
// `addAll` is all or nothing: one file that cannot be fetched leaves this cache empty. The very first
// worker starts anyway (the page works online and caches what it loads). An upgrade must not: an empty
// cache would replace a full one, and after Reload the app would not open offline. So the install fails,
// the running worker stays in charge, no update is announced, and the browser tries again at the next visit.
self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      // A cache of this name that was already there is not this install's to remove (the same build under a changed worker).
      const existed = await caches.has(CACHE);
      try {
        await (await caches.open(CACHE)).addAll(PRECACHE.map((p) => new URL(p, SCOPE).href));
      } catch (error) {
        if (!self.registration.active) return;
        if (!existed) await caches.delete(CACHE);
        throw error;
      }
    })(),
  );
});

/** The caches of other builds this worker owns, oldest first. */
async function otherBuilds() {
  return (await caches.keys()).filter((name) => name !== CACHE && owned(name));
}

/** When the worker of this build took over (0: it never did, or wrote no record). */
async function tookOverAt(name) {
  try {
    const record = await (await caches.open(name)).match(TOOK_OVER);
    return record ? Number((await record.json()).at) || 0 : 0;
  } catch {
    return 0;
  }
}

/**
 * The build that ran before this one: the one that took over last. Builds that never took over (they
 * were replaced while they waited) are not it. With no record at all (the caches of the bot.computer
 * days, or of a build from before the records) it is the newest cache that holds anything.
 */
async function previousBuild(names) {
  let previous = null;
  let latest = 0;
  for (const name of names) {
    const at = await tookOverAt(name);
    if (at > latest) {
      previous = name;
      latest = at;
    }
  }
  if (previous) return previous;
  for (const name of [...names].reverse()) if ((await (await caches.open(name)).keys()).length) return name;
  return null;
}

// Taking over keeps the previous build's cache (tabs that have not reloaded still run that build and
// load their files from it) and deletes every other cache of this app: the ones older than that, and
// the ones of builds that were replaced while they waited.
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const others = await otherBuilds();
      const previous = await previousBuild(others);
      await Promise.all(others.filter((name) => name !== previous).map((name) => caches.delete(name)));
      await (await caches.open(CACHE)).put(TOOK_OVER, new Response(JSON.stringify({ at: Date.now() }), { headers: { 'content-type': 'application/json' } }));
      await self.clients.claim();
    })(),
  );
});

/** A response for the request from the cache of another build of this app, if one has it. */
async function fromEarlierBuild(key) {
  for (const name of (await otherBuilds()).reverse()) {
    const hit = await (await caches.open(name)).match(key);
    if (hit) return hit;
  }
  return undefined;
}

self.addEventListener('message', (event) => {
  // The person chose Reload (pwa/update.ts): take over from the version that is running.
  if (event.data?.type === 'skip-waiting') {
    self.skipWaiting();
    return;
  }
  // The page lists what it loaded before this worker controlled it, so the
  // hashed bundles are cached from the first visit.
  if (event.data?.type !== 'cache' || !Array.isArray(event.data.urls)) return;
  const urls = event.data.urls.filter((u) => {
    try {
      return new URL(u).origin === SCOPE.origin;
    } catch {
      return false;
    }
  });
  event.waitUntil(caches.open(CACHE).then((cache) => Promise.all(urls.map((u) => cache.add(u).catch(() => {})))));
});

function isolate(response) {
  if (!response || response.status === 0 || response.type === 'opaque') return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== SCOPE.origin) return;
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      // Only the app's own page is the app shell. Other pages opened on this site
      // (the SoftN runtime, an icon opened in a tab, the preview frame) are cached as themselves.
      const shell = request.mode === 'navigate' && request.destination === 'document' && (url.pathname === SCOPE.pathname || url.pathname === `${SCOPE.pathname}index.html`);
      const key = shell ? new URL('./index.html', SCOPE).href : request;
      const network = fetch(request)
        .then((response) => {
          if (response.ok && response.type === 'basic') cache.put(key, response.clone());
          return response;
        })
        .catch(() => null);
      // A page: fresh when online, cached when not.
      if (request.mode === 'navigate') return isolate((await network) ?? (await cache.match(key)) ?? (await fromEarlierBuild(key)) ?? new Response('offline', { status: 503 }));
      const cached = await cache.match(key);
      if (cached) {
        event.waitUntil(network);
        return isolate(cached);
      }
      const response = await network;
      if (response?.ok) return isolate(response);
      // Not on the host (or no host): a tab that is still on an earlier build finds its own files in that build's cache.
      return isolate((await fromEarlierBuild(key)) ?? response ?? new Response('offline', { status: 503 }));
    })(),
  );
});
