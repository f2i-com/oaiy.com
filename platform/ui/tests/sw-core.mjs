/**
 * The service worker's rules: what it answers, what it never touches, and what it keeps.
 *
 *     npm run test:sw
 *
 * `installWorker` (src/pwa/swCore.ts) is given a stand-in for the worker's scope: an in-memory
 * CacheStorage that behaves like the real one where it matters here (a response is consumed by
 * `put`, `match` hands out a new one each time), a network that answers as each test says, and
 * the worker's own events fired by hand. The real browser side is tests/pwa-e2e.mjs.
 *
 *   - route(): only a GET for the worker's own origin is ever answered, decided against the origin the
 *     worker was loaded from (so a site served from 127.0.0.1 is kept and an engine on another port
 *     is not); /api/, share links, ranges, the worker's script and the files the build says are too
 *     big are left to the browser.
 *   - The size rule counts the bytes the page gets, not Content-Length: a compressed body with a
 *     small header is still refused, an oversize file is never held, and the page gets every byte
 *     either way.
 *   - Install keeps the shell (and fails, leaving the old worker, if it cannot); it never skips
 *     waiting. Activate removes the caches of other builds and only those.
 *   - HTML is network-first with the kept shell as the fallback; everything else is stale-while-revalidate.
 *   - A response kept and served again has the headers the host sent (a worker script's CSP).
 */
import assert from 'node:assert/strict';
import { loadTs, suite } from './support/loadTs.mjs';

const W = await loadTs('src/pwa/swCore.ts');
const { check, finish } = suite('service worker');

const ORIGIN = 'https://oaiy.com';
const MAX = W.MAX_CACHED_BYTES;
const bytes = (n, fill = 97) => new Uint8Array(n).fill(fill);
const text = async (res) => new TextDecoder().decode(await res.arrayBuffer());

/* ---------------------------- a stand-in scope ---------------------------- */

class FakeCache {
  constructor(origin) {
    this.origin = origin;
    this.entries = new Map();
  }
  key(input, ignoreSearch) {
    const url = new URL(typeof input === 'string' ? input : input.url, this.origin);
    if (ignoreSearch) url.search = '';
    return url.href;
  }
  async put(input, response) {
    // Like the real one, `put` reads the response: a caller that hands over the one it is about to return breaks here.
    const body = new Uint8Array(await response.arrayBuffer());
    this.entries.set(this.key(input, false), { body, status: response.status, statusText: response.statusText, headers: [...response.headers] });
  }
  async match(input, options = {}) {
    const want = this.key(input, options.ignoreSearch);
    for (const [url, entry] of this.entries) {
      if (this.key(url, options.ignoreSearch) === want) return new Response(entry.body, { status: entry.status, statusText: entry.statusText, headers: entry.headers });
    }
    return undefined;
  }
  async keys() {
    return [...this.entries.keys()].map((url) => ({ url }));
  }
  async delete(input) {
    return this.entries.delete(this.key(input, false));
  }
}

class FakeCaches {
  constructor(origin) {
    this.origin = origin;
    this.stores = new Map();
  }
  async open(name) {
    if (!this.stores.has(name)) this.stores.set(name, new FakeCache(this.origin));
    return this.stores.get(name);
  }
  async keys() {
    return [...this.stores.keys()];
  }
  async delete(name) {
    return this.stores.delete(name);
  }
  async has(name) {
    return this.stores.has(name);
  }
}

const noRange = new Headers();

/** A request as the worker's fetch event carries it. (Node cannot build one with mode "navigate".) */
function request(url, { method = 'GET', mode = 'no-cors', headers, destination = '' } = {}) {
  return { url: new URL(url, ORIGIN).href, method, mode, destination, headers: new Headers(headers ?? noRange) };
}

const CONFIG = {
  buildId: 'b1',
  precache: ['/app.html', '/assets/app-aaaa1111.js', '/assets/app-bbbb2222.css'],
  oversize: ['/assets/ffmpeg-core-cccc3333.wasm'],
};

function makeWorker({ origin = ORIGIN, config = CONFIG, timeoutMs = 20 } = {}) {
  const listeners = {};
  const caches = new FakeCaches(origin);
  const state = { fetched: [], skipped: 0, claimed: 0, online: true, routes: new Map(), gate: null };
  const scope = {
    location: { origin },
    caches,
    clients: { claim: async () => { state.claimed++; } },
    skipWaiting: async () => { state.skipped++; },
    addEventListener: (type, fn) => { listeners[type] = fn; },
    fetch: async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      const cache = typeof input === 'string' ? undefined : input.cache;
      state.fetched.push({ url, cache });
      if (state.gate) await state.gate;
      if (!state.online) throw new TypeError('Failed to fetch');
      const path = new URL(url).pathname;
      const make = state.routes.get(path);
      if (!make) return new Response('not found', { status: 404 });
      return make();
    },
  };
  W.installWorker(scope, config, { navigationTimeoutMs: timeoutMs, wait: (ms) => new Promise((resolve) => setTimeout(resolve, ms)) });

  const extendable = () => {
    const waits = [];
    return { waits, waitUntil: (p) => { waits.push(Promise.resolve(p)); } };
  };
  const worker = {
    scope,
    caches,
    state,
    listeners,
    ok(path, body = 'ok', headers = { 'content-type': 'text/javascript' }) {
      state.routes.set(path, () => new Response(body, { status: 200, headers }));
    },
    async install() {
      const event = extendable();
      listeners.install(event);
      await Promise.all(event.waits);
    },
    async activate() {
      const event = extendable();
      listeners.activate(event);
      await Promise.all(event.waits);
    },
    async message(data) {
      const event = { ...extendable(), data };
      listeners.message(event);
      await Promise.all(event.waits);
    },
    /** Fire a fetch event. `answered` is false when the worker did not respondWith (the browser goes to the network itself). */
    async fetch(req) {
      let answer;
      const event = { ...extendable(), request: req, respondWith: (r) => { answer = Promise.resolve(r); } };
      listeners.fetch(event);
      return {
        answered: answer !== undefined,
        response: answer,
        // Everything the worker kept running after answering (the write-behind of a cache): once the
        // answer is made (its writes are asked for on the way), and until they are done.
        settled: async () => {
          await answer?.catch(() => undefined);
          await Promise.all(event.waits);
        },
      };
    },
  };
  return worker;
}

/** A worker that has installed the shell of CONFIG from a network that has all of it. */
async function installedWorker(options) {
  const w = makeWorker(options);
  w.ok('/app.html', '<html>shell v1</html>', { 'content-type': 'text/html; charset=utf-8' });
  w.ok('/assets/app-aaaa1111.js', 'console.log(1)');
  w.ok('/assets/app-bbbb2222.css', 'a{}', { 'content-type': 'text/css' });
  await w.install();
  return w;
}

/* --------------------------------- route ---------------------------------- */

await check('route: a GET for the worker\'s own origin is an asset, a navigation is a navigation', () => {
  assert.deepEqual(W.route(request('/assets/x-12345678.js'), ORIGIN), { kind: 'asset' });
  assert.deepEqual(W.route(request('/app.html', { mode: 'navigate' }), ORIGIN), { kind: 'navigation' });
  assert.deepEqual(W.route(request('/fonts/inter.woff2'), ORIGIN), { kind: 'asset' });
});

await check('route: what is left to the browser, and why', () => {
  const bypass = (req, reason, origin = ORIGIN, oversize) => assert.deepEqual(W.route(req, origin, oversize), { kind: 'bypass', reason }, `${req.method} ${req.url}`);
  bypass(request('/api/health'), 'api');
  bypass(request('/api'), 'api');
  bypass(request('/api/flows/abc/runs', { method: 'POST' }), 'method');
  bypass(request('/x.js', { method: 'HEAD' }), 'method');
  bypass(request('/assets/movie.mp4', { headers: { range: 'bytes=0-1' } }), 'range');
  bypass(request('/app.html?flow=abc123', { mode: 'navigate' }), 'share-link');
  bypass(request('/assets/x.js?flow=1'), 'share-link');
  bypass(request('/sw.js'), 'worker-script');
  bypass(request('/assets/ffmpeg-core-cccc3333.wasm'), 'oversize', ORIGIN, CONFIG.oversize);
  bypass({ url: 'not a url', method: 'GET', headers: noRange }, 'cross-origin');
  // Not left out: a similar name, and a page whose query is something else.
  assert.equal(W.route(request('/assets/ffmpeg-core-dddd4444.wasm'), ORIGIN, CONFIG.oversize).kind, 'asset');
  assert.equal(W.route(request('/app.html?utm=1', { mode: 'navigate' }), ORIGIN).kind, 'navigation');
  assert.equal(W.route(request('/apis/x.js'), ORIGIN).kind, 'asset', '/apis is not /api/');
});

await check('route: other origins are never answered, every engine and provider among them (worker on https://oaiy.com)', () => {
  for (const url of [
    'http://127.0.0.1:17972/api/health',
    'http://localhost:17972/api/services',
    'http://127.0.0.1:8080/v1/models',
    'http://[::1]:17972/api/health',
    'http://192.168.1.50:17972/api/health',
    'https://api.openai.com/v1/chat/completions',
    'https://cdn.example/x.js',
    'https://oaiy.com:8443/x.js',
    'http://oaiy.com/x.js',
    'https://api.oaiy.com/api/flows',
  ]) {
    assert.deepEqual(W.route(request(url), ORIGIN), { kind: 'bypass', reason: 'cross-origin' }, url);
  }
});

await check('route: decided against the worker\'s own origin, not a list of hosts (a site served from 127.0.0.1 is kept, the engine beside it is not)', () => {
  const site = 'http://127.0.0.1:41234';
  assert.deepEqual(W.route(request(`${site}/assets/a-12345678.js`), site), { kind: 'asset' });
  assert.deepEqual(W.route(request(`${site}/app.html`, { mode: 'navigate' }), site), { kind: 'navigation' });
  assert.deepEqual(W.route(request(`${site}/api/health`), site), { kind: 'bypass', reason: 'api' });
  for (const url of ['http://127.0.0.1:41235/api/health', 'http://127.0.0.1:17972/api/health', 'http://localhost:41234/assets/a.js', 'http://localhost:17972/']) {
    assert.deepEqual(W.route(request(url), site), { kind: 'bypass', reason: 'cross-origin' }, url);
  }
});

/* ------------------------------ the size rule ------------------------------ */

await check('readWithinLimit: the limit is on the bytes read; the boundary is exact', async () => {
  assert.equal((await W.readWithinLimit(new Response(bytes(10)), 10)).byteLength, 10);
  assert.equal(await W.readWithinLimit(new Response(bytes(11)), 10), null);
  assert.equal((await W.readWithinLimit(new Response(null), 10)).byteLength, 0);
});

await check('readWithinLimit: a small Content-Length on a body that is really larger (a compressed file) is still refused', async () => {
  const res = new Response(bytes(100), { headers: { 'content-encoding': 'gzip', 'content-length': '12' } });
  assert.equal(await W.readWithinLimit(res, 50), null);
});

await check('readWithinLimit: a Content-Length over the limit refuses without reading, and a chunked body stops at the limit', async () => {
  let pulls = 0;
  const source = new ReadableStream({ pull(controller) { pulls++; controller.enqueue(bytes(8)); } });
  assert.equal(await W.readWithinLimit(new Response(source, { headers: { 'content-length': '999' } }), 50), null);
  assert.equal(pulls, 0, 'nothing was read');

  let chunks = 0;
  const endless = new ReadableStream({ pull(controller) { chunks++; controller.enqueue(bytes(8)); } });
  assert.equal(await W.readWithinLimit(new Response(endless), 50), null);
  assert.ok(chunks <= 10, `read ${chunks} chunks of an endless body`);
});

await check('putWithinLimit: the copy has the body as the page got it and the headers the host sent, minus the encoding\'s', async () => {
  const cache = await new FakeCaches(ORIGIN).open('c');
  const csp = "default-src 'none'; script-src 'self' 'unsafe-eval'; connect-src 'none'; worker-src 'none'";
  const res = new Response('worker code', { status: 200, headers: { 'content-type': 'text/javascript', 'content-security-policy': csp, 'content-encoding': 'br', 'content-length': '5', etag: '"x"' } });
  assert.equal(await W.putWithinLimit(cache, `${ORIGIN}/assets/compiler-sandbox-worker-abc.js`, res), true);
  assert.equal(await text(res), 'worker code', 'the response is still whole for whoever asked for it');
  const kept = await cache.match(`${ORIGIN}/assets/compiler-sandbox-worker-abc.js`);
  assert.equal(await text(kept), 'worker code');
  assert.equal(kept.headers.get('content-security-policy'), csp);
  assert.equal(kept.headers.get('content-type'), 'text/javascript');
  assert.equal(kept.headers.get('etag'), '"x"');
  assert.equal(kept.headers.get('content-encoding'), null);
  assert.equal(kept.headers.get('content-length'), null);
});

await check('putWithinLimit: an over-limit body is not kept, and the response is still whole', async () => {
  const cache = await new FakeCaches(ORIGIN).open('c');
  const res = new Response(bytes(80));
  assert.equal(await W.putWithinLimit(cache, `${ORIGIN}/big`, res, 50), false);
  assert.equal((await cache.keys()).length, 0);
  assert.equal((await res.arrayBuffer()).byteLength, 80);
});

await check('isStorable: a plain 200 only', () => {
  assert.equal(W.isStorable(new Response('x')), true);
  assert.equal(W.isStorable(new Response('x', { status: 404 })), false);
  assert.equal(W.isStorable(new Response(null, { status: 206 })), false);
  assert.equal(W.isStorable(Response.error()), false);
});

/* -------------------------------- install --------------------------------- */

await check('install: the shell is kept, under the build\'s own cache, and skipWaiting is never called', async () => {
  const w = await installedWorker();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  const kept = (await shell.keys()).map((k) => new URL(k.url).pathname).sort();
  assert.deepEqual(kept, [...CONFIG.precache].sort());
  assert.equal(await text(await shell.match(`${ORIGIN}/app.html`)), '<html>shell v1</html>');
  assert.equal(w.state.skipped, 0, 'a new build waits for the person');
  assert.equal(w.state.claimed, 0);
  // The page is fetched fresh; the hashed files are not fetched again if the browser has them.
  const page = w.state.fetched.find((f) => f.url.endsWith('/app.html'));
  assert.equal(page.cache, 'reload');
  assert.notEqual(w.state.fetched.find((f) => f.url.endsWith('.js')).cache, 'reload');
});

await check('install: a shell file that cannot be had fails the install, so the old worker carries on', async () => {
  const w = makeWorker();
  w.ok('/app.html', 'x', { 'content-type': 'text/html' });
  w.ok('/assets/app-aaaa1111.js', 'x');
  // the stylesheet is missing: the network answers 404
  await assert.rejects(w.install(), /app-bbbb2222\.css.*404/);
  w.state.routes.set('/assets/app-bbbb2222.css', () => new Response('a{}'));
  w.state.online = false;
  await assert.rejects(w.install(), /Failed to fetch/);
});

await check('install: a shell file over the limit fails the install rather than leave a shell that will not start', async () => {
  const w = makeWorker();
  w.ok('/app.html', 'x', { 'content-type': 'text/html' });
  w.ok('/assets/app-aaaa1111.js', 'x');
  w.state.routes.set('/assets/app-bbbb2222.css', () => new Response(bytes(MAX + 1), { headers: { 'content-type': 'text/css' } }));
  await assert.rejects(w.install(), /over 4194304 bytes/);
});

/* -------------------------------- activate -------------------------------- */

await check('activate: the caches of other builds go, only ours by prefix, and the pages are taken over', async () => {
  const w = await installedWorker();
  for (const name of ['oaiy-web-old1-shell', 'oaiy-web-old1-runtime', 'oaiy-web-old2-shell', 'bot.computer-v3', 'workbox-precache', W.cacheNames('b1').runtime]) await w.caches.open(name);
  await w.activate();
  assert.deepEqual((await w.caches.keys()).sort(), ['bot.computer-v3', 'oaiy-web-b1-runtime', 'oaiy-web-b1-shell', 'workbox-precache']);
  assert.equal(w.state.claimed, 1);
});

/* ------------------------------- navigation -------------------------------- */

await check('navigation: the network first, and the answer becomes the kept shell', async () => {
  const w = await installedWorker();
  w.ok('/app.html', '<html>shell v2</html>', { 'content-type': 'text/html' });
  const out = await w.fetch(request('/app.html', { mode: 'navigate' }));
  assert.ok(out.answered);
  assert.equal(await text(await out.response), '<html>shell v2</html>');
  await out.settled();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  assert.equal(await text(await shell.match(`${ORIGIN}/app.html`)), '<html>shell v2</html>');
});

await check('navigation: no network opens the kept shell, whatever the query (?utm=1), and with none kept it fails as it would have', async () => {
  const w = await installedWorker();
  w.state.online = false;
  const out = await w.fetch(request('/app.html?utm=1', { mode: 'navigate' }));
  assert.equal(await text(await out.response), '<html>shell v1</html>');
  const bare = makeWorker();
  bare.state.online = false;
  const none = await bare.fetch(request('/app.html', { mode: 'navigate' }));
  await assert.rejects(none.response, /Failed to fetch/);
});

await check('navigation: a slow network gives way to the kept shell, and a failing server too; a missing page is passed on', async () => {
  const w = await installedWorker({ timeoutMs: 15 });
  w.state.gate = new Promise((resolve) => setTimeout(resolve, 120));
  const slow = await w.fetch(request('/app.html', { mode: 'navigate' }));
  assert.equal(await text(await slow.response), '<html>shell v1</html>', 'the shell, after the wait');
  await slow.settled();
  w.state.gate = null;

  w.state.routes.set('/app.html', () => new Response('down', { status: 503, headers: { 'content-type': 'text/html' } }));
  const failing = await w.fetch(request('/app.html', { mode: 'navigate' }));
  assert.equal(await text(await failing.response), '<html>shell v1</html>', 'a 503 does not replace the shell');
  await failing.settled();

  w.state.routes.set('/app.html', () => new Response('gone', { status: 404, headers: { 'content-type': 'text/html' } }));
  const missing = await w.fetch(request('/app.html', { mode: 'navigate' }));
  assert.equal((await missing.response).status, 404);
  await missing.settled();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  assert.equal(await text(await shell.match(`${ORIGIN}/app.html`)), '<html>shell v1</html>', 'and neither was kept');
});

await check('navigation: something that is not HTML is not kept as the shell', async () => {
  const w = await installedWorker();
  w.state.routes.set('/app.html', () => new Response('{"portal":true}', { status: 200, headers: { 'content-type': 'application/json' } }));
  const out = await w.fetch(request('/app.html', { mode: 'navigate' }));
  await out.response;
  await out.settled();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  assert.equal(await text(await shell.match(`${ORIGIN}/app.html`)), '<html>shell v1</html>');
});

await check('navigation: a share link (?flow=) is left to the browser, online or not', async () => {
  const w = await installedWorker();
  const before = w.state.fetched.length;
  const out = await w.fetch(request('/app.html?flow=abcdefghijk', { mode: 'navigate' }));
  assert.equal(out.answered, false);
  assert.equal(w.state.fetched.length, before, 'the worker fetched nothing for it');
});

/* --------------------------------- assets ---------------------------------- */

await check('assets: a miss goes to the network and is kept behind the answer; the next time it comes from the cache', async () => {
  const w = await installedWorker();
  w.ok('/assets/chunk-abcd1234.js', 'chunk v1');
  const first = await w.fetch(request('/assets/chunk-abcd1234.js'));
  assert.equal(await text(await first.response), 'chunk v1');
  await first.settled();
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal(await text(await runtime.match(`${ORIGIN}/assets/chunk-abcd1234.js`)), 'chunk v1');

  w.state.online = false; // gone from the network: the cache answers
  const again = await w.fetch(request('/assets/chunk-abcd1234.js'));
  assert.equal(await text(await again.response), 'chunk v1');
  await again.settled();
});

await check('assets: a hit is answered at once from the cache and refreshed behind it (stale-while-revalidate)', async () => {
  const w = await installedWorker();
  w.ok('/assets/chunk-abcd1234.js', 'chunk v1');
  await (await w.fetch(request('/assets/chunk-abcd1234.js'))).settled();
  w.ok('/assets/chunk-abcd1234.js', 'chunk v2');
  const served = await w.fetch(request('/assets/chunk-abcd1234.js'));
  assert.equal(await text(await served.response), 'chunk v1', 'the copy it had');
  await served.settled();
  const next = await w.fetch(request('/assets/chunk-abcd1234.js'));
  assert.equal(await text(await next.response), 'chunk v2', 'the refreshed copy');
  await next.settled();
});

await check('assets: the shell files are served from the shell cache', async () => {
  const w = await installedWorker();
  w.state.online = false;
  const out = await w.fetch(request('/assets/app-aaaa1111.js'));
  assert.equal(await text(await out.response), 'console.log(1)');
});

await check('assets: no answer from the network and nothing kept is a failure, as without a worker; a 404 is passed on and not kept', async () => {
  const w = await installedWorker();
  w.state.online = false;
  const out = await w.fetch(request('/assets/never-seen-12345678.js'));
  await assert.rejects(out.response, /Failed to fetch/);
  w.state.online = true;
  const missing = await w.fetch(request('/assets/missing-12345678.js'));
  assert.equal((await missing.response).status, 404);
  await missing.settled();
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, 0);
});

await check('assets: a file over the limit reaches the page whole and is not kept, whatever its headers say', async () => {
  const w = await installedWorker();
  const big = bytes(MAX + 1000, 66);
  const cases = {
    '/assets/plain-11111111.js': { 'content-type': 'text/javascript' },
    '/assets/declared-22222222.wasm': { 'content-type': 'application/wasm', 'content-length': String(MAX + 1000) },
    '/assets/compressed-33333333.js': { 'content-type': 'text/javascript', 'content-encoding': 'gzip', 'content-length': '900000' }, // the wire size of a compressed body
  };
  for (const [path, headers] of Object.entries(cases)) {
    w.state.routes.set(path, () => new Response(big.slice(), { status: 200, headers }));
    const out = await w.fetch(request(path));
    const res = await out.response;
    const body = new Uint8Array(await res.arrayBuffer());
    assert.equal(body.byteLength, MAX + 1000, `${path}: the page got every byte`);
    assert.equal(body[MAX], 66);
    await out.settled();
  }
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, 0, 'none of them was kept');
});

await check('assets: a file just under the limit is kept', async () => {
  const w = await installedWorker();
  w.state.routes.set('/assets/edge-44444444.js', () => new Response(bytes(MAX), { status: 200, headers: { 'content-type': 'text/javascript' } }));
  const out = await w.fetch(request('/assets/edge-44444444.js'));
  assert.equal((await (await out.response).arrayBuffer()).byteLength, MAX);
  await out.settled();
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, 1);
});

await check('assets: the files the build lists as too big are never fetched by the worker at all', async () => {
  const w = await installedWorker();
  const before = w.state.fetched.length;
  const out = await w.fetch(request('/assets/ffmpeg-core-cccc3333.wasm'));
  assert.equal(out.answered, false);
  assert.equal(w.state.fetched.length, before);
});

await check('assets: a worker script comes back with the CSP the host sent (the compiler sandbox worker)', async () => {
  const w = await installedWorker();
  const csp = "default-src 'none'; script-src 'self' 'unsafe-eval'; connect-src 'none'; worker-src 'none'";
  w.ok('/assets/compiler-sandbox-worker-abcd1234.js', 'self.onmessage=1', { 'content-type': 'text/javascript', 'content-security-policy': csp });
  await (await w.fetch(request('/assets/compiler-sandbox-worker-abcd1234.js'))).settled();
  w.state.online = false;
  const out = await w.fetch(request('/assets/compiler-sandbox-worker-abcd1234.js'));
  const res = await out.response;
  assert.equal(res.headers.get('content-security-policy'), csp);
  assert.equal(await text(res), 'self.onmessage=1');
});

/* ------------------------------ what it never touches ------------------------------ */

await check('the worker answers nothing for /api/, another origin (the engines), a POST, a range or its own script', async () => {
  const w = await installedWorker();
  const before = w.state.fetched.length;
  for (const req of [
    request('/api/service-library'),
    request('http://127.0.0.1:17972/api/health'),
    request('http://localhost:8080/v1/models'),
    request('https://api.openai.com/v1/models'),
    request('/api/flows', { method: 'POST' }),
    request('/assets/clip.mp4', { headers: { range: 'bytes=0-' } }),
    request('/sw.js'),
  ]) {
    const out = await w.fetch(req);
    assert.equal(out.answered, false, `${req.method} ${req.url}`);
  }
  assert.equal(w.state.fetched.length, before);
  for (const name of await w.caches.keys()) {
    const cache = await w.caches.open(name);
    for (const key of await cache.keys()) assert.ok(!/\/api\/|127\.0\.0\.1|localhost|openai|sw\.js/.test(key.url), `kept ${key.url}`);
  }
});

await check('a site served from 127.0.0.1 is kept, and the engine on the same address is not', async () => {
  const site = 'http://127.0.0.1:41234';
  const w = makeWorker({ origin: site });
  w.ok('/app.html', 'shell', { 'content-type': 'text/html' });
  w.ok('/assets/app-aaaa1111.js', 'x');
  w.ok('/assets/app-bbbb2222.css', 'x', { 'content-type': 'text/css' });
  w.ok('/assets/lazy-12345678.js', 'lazy');
  await w.install();
  const lazy = await w.fetch(request(`${site}/assets/lazy-12345678.js`));
  assert.ok(lazy.answered);
  await (await lazy.response).arrayBuffer();
  await lazy.settled();
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, 1);
  assert.equal((await w.fetch(request('http://127.0.0.1:41235/api/health'))).answered, false);
  assert.equal((await w.fetch(request('http://127.0.0.1:17972/api/health'))).answered, false);
});

/* -------------------------------- messages --------------------------------- */

await check('message: skip-waiting makes the waiting worker take over; other messages do nothing', async () => {
  const w = await installedWorker();
  await w.message({ type: 'skip-waiting' });
  assert.equal(w.state.skipped, 1);
  for (const junk of [null, 'skip-waiting', 7, {}, { type: 'other' }, { type: 'cache-urls', urls: 'x' }]) await w.message(junk);
  assert.equal(w.state.skipped, 1);
});

await check('message: cache-urls keeps what the page already loaded, by the same rules as everything else', async () => {
  const w = await installedWorker();
  w.ok('/assets/lazy-a1111111.js', 'lazy a');
  w.ok('/assets/font-b2222222.woff2', 'font', { 'content-type': 'font/woff2' });
  w.ok('/assets/too-big-c3333333.js', 'unused');
  w.ok('/api/data', 'nope');
  await w.message({
    type: 'cache-urls',
    urls: [
      { url: `${ORIGIN}/assets/lazy-a1111111.js`, size: 900 },
      { url: '/assets/font-b2222222.woff2' },
      { url: `${ORIGIN}/assets/too-big-c3333333.js`, size: MAX + 1 },
      { url: `${ORIGIN}/assets/ffmpeg-core-cccc3333.wasm`, size: 10 },
      { url: `${ORIGIN}/assets/app-aaaa1111.js`, size: 10 },
      { url: `${ORIGIN}/api/data` },
      { url: 'http://127.0.0.1:17972/api/health' },
      { url: `${ORIGIN}/app.html?flow=abc` },
      { url: 'https://cdn.example/x.js' },
      { url: 12 },
      null,
    ],
  });
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.deepEqual((await runtime.keys()).map((k) => new URL(k.url).pathname).sort(), ['/assets/font-b2222222.woff2', '/assets/lazy-a1111111.js']);
  const asked = w.state.fetched.map((f) => new URL(f.url).pathname);
  for (const never of ['/assets/too-big-c3333333.js', '/assets/ffmpeg-core-cccc3333.wasm', '/api/data']) assert.ok(!asked.includes(never), `fetched ${never}`);
  assert.equal(asked.filter((p) => p === '/assets/app-aaaa1111.js').length, 1, 'a shell file is not fetched again');
  assert.ok(!w.state.fetched.some((f) => /127\.0\.0\.1|cdn\.example/.test(f.url)));
});

await check('message: at most MAX_WARM_URLS addresses are taken from one message', async () => {
  const w = await installedWorker();
  const urls = [];
  for (let i = 0; i < W.MAX_WARM_URLS + 50; i++) {
    const path = `/assets/many-${String(i).padStart(4, '0')}.js`;
    w.ok(path, 'x');
    urls.push({ url: path });
  }
  await w.message({ type: 'cache-urls', urls });
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, W.MAX_WARM_URLS);
});

await check('the limit is one number: 4 MiB, shared with the build and the page', () => {
  assert.equal(W.MAX_CACHED_BYTES, 4 * 1024 * 1024);
  assert.equal(W.CACHE_PREFIX, 'oaiy-web-');
  assert.equal(W.SHELL_PATH, '/app.html');
});

/* ------------- a shell file that is not what its name says (a host that answers with its front page) ------------- */

const PAGE = '<!doctype html><html><title>the landing page</title></html>';
const html = () => new Response(PAGE, { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } });
const bodyOf = async (res) => new TextDecoder().decode(await res.arrayBuffer());

await check('content: the page is HTML; a script, a style sheet and an image must be what their name and use say; nothing else may be HTML', () => {
  const type = (t) => new Response('x', { headers: t ? { 'content-type': t } : {} });
  const fits = (path, t, destination) => W.fitsName(path, W.contentTypeOf(type(t)), destination);
  assert.equal(fits('/app.html', 'text/html; charset=utf-8'), true);
  assert.equal(fits('/app.html', 'text/javascript'), false, 'the page is not a script');
  assert.equal(fits('/assets/a-11111111.js', 'text/javascript; charset=utf-8'), true);
  assert.equal(fits('/assets/a-11111111.js', 'application/javascript'), true);
  assert.equal(fits('/assets/a-11111111.mjs', 'application/x-javascript'), true);
  assert.equal(fits('/assets/a-11111111.js', 'text/html'), false, 'HTML under a script\'s name');
  assert.equal(fits('/assets/a-11111111.js', 'application/octet-stream'), false);
  assert.equal(fits('/assets/a-11111111.js', ''), false, 'a script with no type would not run');
  assert.equal(fits('/assets/a-11111111.css', 'text/css'), true);
  assert.equal(fits('/assets/a-11111111.css', 'text/html'), false);
  assert.equal(fits('/assets/a.png', 'image/png'), true);
  assert.equal(fits('/assets/a.svg', 'image/svg+xml'), true);
  assert.equal(fits('/assets/a.png', 'text/html'), false);
  assert.equal(fits('/assets/a.woff2', 'font/woff2'), true);
  assert.equal(fits('/assets/a.woff2', 'text/html'), false);
  assert.equal(fits('/assets/a.wasm', 'text/html'), false);
  assert.equal(fits('/assets/a.wasm', 'application/wasm'), true);
  assert.equal(fits('/assets/data', 'application/xhtml+xml'), false);
  assert.equal(fits('/manifest.webmanifest', 'application/manifest+json'), true);
  // what the request is for counts even when the name says nothing
  assert.equal(fits('/assets/chunk', 'text/javascript', 'script'), true);
  assert.equal(fits('/assets/chunk', 'text/plain', 'script'), false);
  assert.equal(fits('/assets/chunk', 'text/css', 'style'), true);
  assert.equal(fits('/assets/chunk', 'application/json', 'style'), false);
  assert.equal(fits('/assets/worker', 'text/javascript', 'worker'), true);
  assert.equal(fits('/assets/worker', 'text/html', 'worker'), false);
  assert.equal(fits('/assets/pic', 'image/webp', 'image'), true);
  assert.equal(fits('/assets/pic', 'text/html', 'image'), false);
});

await check('install: a shell script answered with the front page fails the install, and no half a shell is left', async () => {
  const w = makeWorker();
  w.ok('/app.html', 'shell', { 'content-type': 'text/html' });
  w.state.routes.set('/assets/app-aaaa1111.js', html); // 200, text/html: what a host does for a file it does not have yet
  w.ok('/assets/app-bbbb2222.css', 'a{}', { 'content-type': 'text/css' });
  await assert.rejects(w.install(), /app-aaaa1111\.js.*text\/html/);
  assert.ok(!(await w.caches.has(W.cacheNames('b1').shell)), 'the shell cache of the build that did not install is removed');
});

await check('install: the host heals and the next install keeps the right file (the failed one left nothing behind)', async () => {
  const w = makeWorker();
  w.ok('/app.html', 'shell', { 'content-type': 'text/html' });
  w.state.routes.set('/assets/app-aaaa1111.js', html);
  w.ok('/assets/app-bbbb2222.css', 'a{}', { 'content-type': 'text/css' });
  await assert.rejects(w.install());
  w.ok('/assets/app-aaaa1111.js', 'console.log("the script")');
  await w.install();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  assert.equal(await bodyOf(await shell.match(`${ORIGIN}/assets/app-aaaa1111.js`)), 'console.log("the script")');
  const served = await w.fetch(request('/assets/app-aaaa1111.js', { destination: 'script' }));
  assert.equal(await bodyOf(await served.response), 'console.log("the script")');
});

await check('install: a wrong style sheet, and the page served as something else, fail the install too; so does a redirected answer', async () => {
  for (const [path, make, why] of [
    ['/assets/app-bbbb2222.css', () => new Response('x', { headers: { 'content-type': 'text/plain' } }), /text\/plain/],
    ['/app.html', () => new Response('x', { headers: { 'content-type': 'application/json' } }), /application\/json/],
    ['/assets/app-aaaa1111.js', () => Object.defineProperty(new Response('x', { headers: { 'content-type': 'text/javascript' } }), 'redirected', { value: true }), /redirect/],
  ]) {
    const w = makeWorker();
    w.ok('/app.html', 'shell', { 'content-type': 'text/html' });
    w.ok('/assets/app-aaaa1111.js', 'js', { 'content-type': 'text/javascript' });
    w.ok('/assets/app-bbbb2222.css', 'a{}', { 'content-type': 'text/css' });
    w.state.routes.set(path, make);
    await assert.rejects(w.install(), why, path);
  }
});

await check('an update whose shell is wrong does not install, and the build in charge keeps its files and its caches', async () => {
  const old = await installedWorker();
  const before = (await (await old.caches.open(W.cacheNames('b1').shell)).keys()).length;
  // a second build (b2) in the same origin's caches
  const w2 = makeWorker({ config: { ...CONFIG, buildId: 'b2' } });
  w2.caches.stores = old.caches.stores;
  w2.ok('/app.html', 'shell v2', { 'content-type': 'text/html' });
  w2.state.routes.set('/assets/app-aaaa1111.js', html);
  w2.ok('/assets/app-bbbb2222.css', 'a{}', { 'content-type': 'text/css' });
  await assert.rejects(w2.install());
  assert.equal((await (await old.caches.open(W.cacheNames('b1').shell)).keys()).length, before, 'the old build is untouched');
  assert.ok(!(await w2.caches.has(W.cacheNames('b2').shell)));
});

await check('assets: the front page is not kept under a script\'s name, and the page still gets what the network said', async () => {
  const w = await installedWorker();
  w.state.routes.set('/assets/lazy-a1111111.js', html);
  const out = await w.fetch(request('/assets/lazy-a1111111.js', { destination: 'script' }));
  assert.equal(await bodyOf(await out.response), PAGE, 'the worker does not hide what the host answered');
  await out.settled();
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, 0);
});

await check('assets: a bad copy in the shell (from whatever cause) is never served, and is replaced by the right file, in the shell', async () => {
  const w = await installedWorker();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  await shell.put(`${ORIGIN}/assets/app-aaaa1111.js`, html()); // the reviewer's state: HTML under the script's name
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  const first = await w.fetch(request('/assets/app-aaaa1111.js', { destination: 'script' }));
  assert.equal(await bodyOf(await first.response), 'console.log(1)', 'the network\'s file, not the bad copy');
  await first.settled();
  assert.equal(await bodyOf(await shell.match(`${ORIGIN}/assets/app-aaaa1111.js`)), 'console.log(1)', 'healed where it was');
  assert.equal((await runtime.keys()).length, 0, 'not moved to the runtime cache');
  w.state.online = false; // and now it is the kept one, offline
  const again = await w.fetch(request('/assets/app-aaaa1111.js', { destination: 'script' }));
  assert.equal(await bodyOf(await again.response), 'console.log(1)');
});

await check('assets: with no network a bad copy is not served either: the script fails as it would have', async () => {
  const w = await installedWorker();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  await shell.put(`${ORIGIN}/assets/app-aaaa1111.js`, html());
  w.state.online = false;
  const out = await w.fetch(request('/assets/app-aaaa1111.js', { destination: 'script' }));
  await assert.rejects(out.response, /Failed to fetch/);
  await out.settled();
  assert.equal((await shell.keys()).some((k) => k.url.endsWith('/assets/app-aaaa1111.js')), false, 'and it is removed, not left to be found again');
});

await check('assets: what a request is for counts when its name says nothing (a chunk with no extension asked for as a script)', async () => {
  const w = await installedWorker();
  w.ok('/assets/chunk-abcd1234', 'not a script', { 'content-type': 'text/plain' });
  const asScript = await w.fetch(request('/assets/chunk-abcd1234', { destination: 'script' }));
  assert.equal(await bodyOf(await asScript.response), 'not a script', 'the page gets it, and decides');
  await asScript.settled();
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal((await runtime.keys()).length, 0, 'but it is not kept as a script');
  const asData = await w.fetch(request('/assets/chunk-abcd1234'));
  await (await asData.response).arrayBuffer();
  await asData.settled();
  assert.equal((await runtime.keys()).length, 1, 'asked for as data it is');
});

await check('assets: a copy is refreshed in the cache it came from (a shell file in the shell, not the runtime cache)', async () => {
  const w = await installedWorker();
  w.ok('/assets/app-aaaa1111.js', 'console.log(2)');
  const out = await w.fetch(request('/assets/app-aaaa1111.js', { destination: 'script' }));
  assert.equal(await bodyOf(await out.response), 'console.log(1)', 'the copy it had');
  await out.settled();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.equal(await bodyOf(await shell.match(`${ORIGIN}/assets/app-aaaa1111.js`)), 'console.log(2)');
  assert.equal((await runtime.keys()).length, 0);
});

await check('assets: a good copy is kept when the refresh comes back wrong (the front page)', async () => {
  const w = await installedWorker();
  w.state.routes.set('/assets/app-aaaa1111.js', html);
  const out = await w.fetch(request('/assets/app-aaaa1111.js', { destination: 'script' }));
  assert.equal(await bodyOf(await out.response), 'console.log(1)');
  await out.settled();
  const shell = await w.caches.open(W.cacheNames('b1').shell);
  assert.equal(await bodyOf(await shell.match(`${ORIGIN}/assets/app-aaaa1111.js`)), 'console.log(1)');
});

await check('message: what the page says it loaded is kept only if it is what its name says', async () => {
  const w = await installedWorker();
  w.state.routes.set('/assets/lazy-a1111111.js', html);
  w.ok('/assets/lazy-b2222222.js', 'ok', { 'content-type': 'text/javascript' });
  await w.message({ type: 'cache-urls', urls: [{ url: '/assets/lazy-a1111111.js' }, { url: '/assets/lazy-b2222222.js' }] });
  const runtime = await w.caches.open(W.cacheNames('b1').runtime);
  assert.deepEqual((await runtime.keys()).map((k) => new URL(k.url).pathname), ['/assets/lazy-b2222222.js']);
});

finish();
