/*
 * bot.computer's service worker.
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
const CACHE = 'bot.computer-v1';
const SCOPE = new URL(self.registration.scope);
const PRECACHE = ['./', './index.html', './manifest.webmanifest', './icon.svg', './zipp/zipp_wasm.js', './zipp/zipp_wasm_bg.wasm'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(PRECACHE.map((p) => new URL(p, SCOPE).href)))
      .catch(() => {})
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

// The page lists what it loaded before this worker controlled it, so the
// hashed bundles are cached from the first visit.
self.addEventListener('message', (event) => {
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
      if (request.mode === 'navigate') return isolate((await network) ?? (await cache.match(key)) ?? new Response('offline', { status: 503 }));
      const cached = await cache.match(key);
      if (cached) {
        event.waitUntil(network);
        return isolate(cached);
      }
      return isolate((await network) ?? new Response('offline', { status: 503 }));
    })(),
  );
});
