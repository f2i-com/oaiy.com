/**
 * The build makes /sw.js from its own output, and stops when the shell cannot be kept.
 *
 *     npm run test:sw-build
 *
 * scripts/serviceWorkerPlugin.ts:
 *   - reads the shell from the page the build wrote (module script, modulepreload and stylesheet links,
 *     the page first), and nothing else on it;
 *   - fails the build, saying what to fix, for a file the page names that the build did not make and
 *     for a shell file over the size the worker keeps;
 *   - lists every file of the build over that size for the worker to leave alone;
 *   - stamps the caches with an id that is the same for the same build and different when the shell,
 *     the page or the worker's own code changes;
 *   - makes a script that is a working worker: run in a stand-in scope it wires the four events with the
 *     config it was given;
 *   - and, in a real (small) Vite build, is in the bundle in time to see the page.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import { build } from 'vite';
import { loadTs, suite, UI } from './support/loadTs.mjs';

const P = await loadTs('scripts/serviceWorkerPlugin.ts');
const W = await loadTs('src/pwa/swCore.ts');
const { check, finish } = suite('service worker build');

const HTML = `<!doctype html>
<html lang="en" class="dark oaiy-app">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
    <link rel="manifest" href="/manifest.webmanifest" />
    <script>(function () { var s = "/not-a-file.js"; })();</script>
    <script type="module" crossorigin src="/assets/app-DP8Ay2Yg.js"></script>
    <link rel="modulepreload" crossorigin href="/assets/react-vendor-4qtZs0mh.js">
    <link rel="modulepreload" crossorigin href="/assets/monaco-base-BuVA0Tet.js?v=2">
    <link crossorigin href="/assets/ThemeContext-BvIPjpIp.css" rel="stylesheet">
    <link rel='stylesheet' href='/assets/xyflow-BZV40eAE.css'>
    <link rel="preload" as="font" href="/assets/inter-latin.woff2">
    <link rel="modulepreload" crossorigin href="/assets/react-vendor-4qtZs0mh.js">
  </head>
  <body><div id="root"></div></body>
</html>`;

const sizes = new Map([
  ['/assets/app-DP8Ay2Yg.js', 663117],
  ['/assets/react-vendor-4qtZs0mh.js', 193168],
  ['/assets/monaco-base-BuVA0Tet.js', 1186217],
  ['/assets/ThemeContext-BvIPjpIp.css', 200497],
  ['/assets/xyflow-BZV40eAE.css', 15851],
  ['/assets/ffmpeg-core-CgUfceKH.wasm', 32232419],
  ['/assets/zipp_wasm_bg-Ds93TMx-.wasm', 5743065],
  ['/assets/small-12345678.js', 10],
]);

await check('the shell is what the page asks for to start: the page, then its script, its preloads and its stylesheets, each once', () => {
  assert.deepEqual(P.shellFromHtml(HTML), [
    '/app.html',
    '/assets/app-DP8Ay2Yg.js',
    '/assets/react-vendor-4qtZs0mh.js',
    '/assets/monaco-base-BuVA0Tet.js',
    '/assets/ThemeContext-BvIPjpIp.css',
    '/assets/xyflow-BZV40eAE.css',
  ]);
});

await check('the shell has no icons, fonts, manifest or inline script, and a page with no script is only the page', () => {
  const shell = P.shellFromHtml(HTML);
  for (const not of ['/favicon.svg', '/apple-touch-icon.png', '/manifest.webmanifest', '/assets/inter-latin.woff2', '/not-a-file.js']) assert.ok(!shell.includes(not), not);
  assert.deepEqual(P.shellFromHtml('<html><body>hi</body></html>'), ['/app.html']);
});

await check('a file that is not from this site\'s root is refused rather than guessed at', () => {
  assert.throws(() => P.shellFromHtml('<script type="module" src="https://cdn.example/x.js"></script>'), /not a path from the site root/);
  assert.throws(() => P.shellFromHtml('<script type="module" src="assets/x.js"></script>'), /not a path from the site root/);
  assert.throws(() => P.shellFromHtml('<link rel="stylesheet" href="//cdn.example/x.css">'), /not a path from the site root/);
});

await check('the worker is made, with the shell, the over-size files and an id', async () => {
  const built = await P.makeServiceWorker({ root: UI, html: HTML, sizes });
  assert.deepEqual(built.config.precache, P.shellFromHtml(HTML));
  assert.deepEqual(built.config.oversize, ['/assets/ffmpeg-core-CgUfceKH.wasm', '/assets/zipp_wasm_bg-Ds93TMx-.wasm']);
  assert.match(built.config.buildId, /^[0-9a-f]{12}$/);
  assert.ok(built.source.startsWith('var __OAIY_SW_CONFIG__={'));
  assert.ok(built.source.includes(W.CACHE_PREFIX), 'the worker\'s own code follows');
  assert.ok(!built.source.includes('import '), 'one script, no imports');
});

await check('the build stops for a shell file the build did not make, and for one over the limit, saying which', async () => {
  const missing = new Map(sizes);
  missing.delete('/assets/xyflow-BZV40eAE.css');
  await assert.rejects(P.makeServiceWorker({ root: UI, html: HTML, sizes: missing }), /xyflow-BZV40eAE\.css is named by \/app\.html but the build did not emit it/);
  const big = new Map(sizes);
  big.set('/assets/monaco-base-BuVA0Tet.js', W.MAX_CACHED_BYTES + 1);
  await assert.rejects(P.makeServiceWorker({ root: UI, html: HTML, sizes: big }), /monaco-base-BuVA0Tet\.js is 4194305 bytes, over the 4194304 the worker keeps: split it/);
  const exactly = new Map(sizes);
  exactly.set('/assets/monaco-base-BuVA0Tet.js', W.MAX_CACHED_BYTES);
  await P.makeServiceWorker({ root: UI, html: HTML, sizes: exactly }); // at the limit is allowed
  await assert.rejects(P.makeServiceWorker({ root: UI, html: HTML + ' '.repeat(W.MAX_CACHED_BYTES), sizes }), /app\.html is \d+ bytes: over/);
});

await check('the id is the same for the same build and changes with the shell, the page or the worker', async () => {
  const one = await P.makeServiceWorker({ root: UI, html: HTML, sizes });
  const again = await P.makeServiceWorker({ root: UI, html: HTML, sizes });
  assert.equal(one.config.buildId, again.config.buildId);
  assert.equal(one.source, again.source, 'the same bytes: a build can be reproduced');
  const otherFile = HTML.replace('app-DP8Ay2Yg.js', 'app-ZZZZZZZZ.js');
  const moved = new Map(sizes);
  moved.set('/assets/app-ZZZZZZZZ.js', 1);
  assert.notEqual((await P.makeServiceWorker({ root: UI, html: otherFile, sizes: moved })).config.buildId, one.config.buildId, 'another shell file');
  assert.notEqual((await P.makeServiceWorker({ root: UI, html: HTML.replace('class="dark', 'class="light'), sizes })).config.buildId, one.config.buildId, 'the page changed');
  assert.notEqual((await P.makeServiceWorker({ root: UI, entry: 'tests/support/otherWorker.ts', html: HTML, sizes })).config.buildId, one.config.buildId, 'the code of the worker changed');
});

await check('the script is a working worker: run in a scope it wires install, activate, fetch and message, with its config', async () => {
  const built = await P.makeServiceWorker({ root: UI, html: HTML, sizes });
  const listeners = {};
  const self = { location: { origin: 'https://oaiy.com' }, addEventListener: (type, fn) => { listeners[type] = fn; }, caches: {}, clients: {}, skipWaiting() {}, fetch() {} };
  vm.runInNewContext(built.source, { self, Request, Response, Headers, URL, setTimeout, Promise, Uint8Array, TypeError, Error, Set, Number, Object, Array, Math, JSON, console });
  assert.deepEqual(Object.keys(listeners).sort(), ['activate', 'fetch', 'install', 'message']);
  // The fetch handler is the real one: it leaves an engine alone.
  let answered = false;
  listeners.fetch({ request: { url: 'http://127.0.0.1:17972/api/health', method: 'GET', mode: 'cors', headers: new Headers() }, respondWith() { answered = true; }, waitUntil() {} });
  assert.equal(answered, false);
  // and the shell is the one the build found
  const fetched = [];
  self.caches = { open: async () => ({ put: async () => {} }) };
  self.fetch = async (request) => { fetched.push(new URL(request.url).pathname); return new Response('x', { status: 200 }); };
  let work;
  listeners.install({ waitUntil: (p) => { work = p; } });
  await work;
  assert.deepEqual(fetched.sort(), [...built.config.precache].sort());
});

await check('in a real Vite build the plugin sees the page and emits /sw.js beside it; a shell file over the limit fails that build', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-sw-build-'));
  try {
    const project = (name, extra = '') => {
      const root = path.join(dir, name);
      fs.mkdirSync(root, { recursive: true });
      fs.writeFileSync(path.join(root, 'app.html'), '<!doctype html><html><body><div id="root"></div><script type="module" src="/main.js"></script></body></html>');
      fs.writeFileSync(path.join(root, 'main.js'), `import { lazy } from './lazy.js'; document.title = 'x'; ${extra}\nexport { lazy };`);
      fs.writeFileSync(path.join(root, 'lazy.js'), 'export const lazy = 1;');
      return root;
    };
    const run = (root) => build({
      root,
      logLevel: 'silent',
      configFile: false,
      plugins: [P.serviceWorkerPlugin({ root: UI })],
      build: { outDir: path.join(root, 'dist'), emptyOutDir: true, rollupOptions: { input: path.join(root, 'app.html') }, minify: false },
    });

    const small = project('small');
    await run(small);
    const worker = fs.readFileSync(path.join(small, 'dist', 'sw.js'), 'utf8');
    const config = JSON.parse(/^var __OAIY_SW_CONFIG__=(\{.*?\});/.exec(worker)[1]);
    assert.equal(config.precache[0], '/app.html');
    assert.ok(config.precache.some((p) => /^\/assets\/app-[\w-]+\.js$/.test(p)), JSON.stringify(config.precache));
    for (const file of config.precache.slice(1)) assert.ok(fs.existsSync(path.join(small, 'dist', file)), `${file} is in the build`);
    assert.deepEqual(config.oversize, []);

    const huge = project('huge', `window.data = "${'x'.repeat(W.MAX_CACHED_BYTES + 100)}";`);
    await assert.rejects(run(huge), /the app shell cannot be kept offline[\s\S]*over the 4194304 the worker keeps/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

finish();
