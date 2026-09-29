/**
 * The flow editor as an installable app, in a real browser, against the production build.
 *
 *     npm run build
 *     npm run test:pwa-e2e
 *
 * It serves `dist/` itself, on a port the system picks, and runs its own headless Chromium with
 * a fresh temporary profile for each scenario (nothing shared with any browser you use). Nothing
 * here talks to OAIY Desktop or the engines: the editor's engine address is pointed at a stand-in
 * on another port of this machine, and any request to the product's own ports (17972, 17872,
 * 17973, 8080, 7860, 8783, 9333) is refused by the browser before it leaves.
 *
 * What it proves:
 *   - the worker registers on /app.html and NOT on / or /desktop.html (fresh profile: registrations are per origin),
 *     with the scope /app.html and no COOP/COEP added;
 *   - Chrome finds no installability error, and the manifest parses without warnings;
 *   - the shell opens offline after ONE visit (the server is shut down, the page reloaded);
 *   - a large asset is not kept: the build's ffmpeg core (32 MB), a chunked body of 5 MB, and a 5 MB body
 *     that is small on the wire (gzip) are all delivered whole and none is in any cache, while a small
 *     asset fetched the same way is (so "not kept" is not just "nothing is kept");
 *   - /api/, the engine on another port (127.0.0.1 and localhost), and a ?flow= share link are never
 *     answered by the worker, and are in no cache;
 *   - the compiler sandbox worker's CSP survives the cache;
 *   - the install button appears after the browser's offer, plays it once, and never appears (nor a worker)
 *     in OAIY's own window;
 *   - an update waits: the message appears, nothing reloads by itself, "Reload" takes the new worker and
 *     the old build's caches go; a first install shows no update message.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';
import { startEngine, startSite } from './support/staticSite.mjs';
import { UI } from './support/loadTs.mjs';

const DIST = path.join(UI, 'dist');
if (!fs.existsSync(path.join(DIST, 'sw.js')) || !fs.existsSync(path.join(DIST, 'app.html'))) {
  console.error('There is no build with a service worker: run `npm run build` first.');
  process.exit(2);
}
const builtConfig = JSON.parse(/^var __OAIY_SW_CONFIG__=(\{.*?\});/.exec(fs.readFileSync(path.join(DIST, 'sw.js'), 'utf8'))[1]);
const assets = fs.readdirSync(path.join(DIST, 'assets'));
const largest = (pattern) => assets.filter((f) => pattern.test(f)).sort((a, b) => fs.statSync(path.join(DIST, 'assets', b)).size - fs.statSync(path.join(DIST, 'assets', a)).size)[0];
const FFMPEG = `/assets/${largest(/^ffmpeg-core.*\.wasm$/)}`;
const FFMPEG_BYTES = fs.statSync(path.join(DIST, FFMPEG)).size;
const CSP_WORKER = `/assets/${assets.find((f) => /^compiler-sandbox-worker-.*\.js$/.test(f))}`;
// A small file of the build that is not part of the shell: it can only be kept by the runtime rule.
const SMALL = `/assets/${assets.find((f) => /\.js$/.test(f) && !builtConfig.precache.includes(`/assets/${f}`) && fs.statSync(path.join(DIST, 'assets', f)).size < 100 * 1024)}`;

let pass = 0;
const failures = [];
const ok = (name, cond, detail = '') => {
  if (cond) {
    pass++;
    console.log(`  ✓ ${name}`);
  } else {
    failures.push(name);
    console.log(`  ✗ ${name}${detail ? `  -> ${detail}` : ''}`);
  }
};
const section = (s) => console.log(`\n-- ${s} --`);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The product's own ports: never reached from here.
const OWN_PORTS = new Set(['17972', '17872', '17973', '8080', '7860', '8783', '9333']);
const blocked = [];

/** A fresh browser with its own temporary profile (a persistent context: an incognito one is not installable). */
async function launch({ engine, desktopWindow = false }) {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-pwa-e2e-'));
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: true,
    channel: process.env.PWA_E2E_CHANNEL || undefined,
    viewport: { width: 1440, height: 900 },
  });
  await context.route(
    (url) => ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) && OWN_PORTS.has(url.port),
    (route) => {
      blocked.push(route.request().url());
      return route.abort();
    },
  );
  await context.addInitScript(
    (a) => {
      try {
        localStorage.setItem('oaiy_theme', 'dark');
        localStorage.setItem('skipSplash', 'true');
        localStorage.setItem('oaiy.wizard.completed', 'true');
        localStorage.setItem('oaiy.engineBase', a.engine);
      } catch { /* storage blocked */ }
      if (a.desktopWindow) window.__OAIY_DESKTOP__ = { origin: a.engine, token: 'test', theme: 'dark' };
    },
    { engine: engine.origin, desktopWindow },
  );
  return {
    context,
    async close() {
      await context.close();
      for (let i = 0; i < 5; i++) {
        try {
          fs.rmSync(userDataDir, { recursive: true, force: true });
          return;
        } catch {
          await sleep(300);
        }
      }
    },
  };
}

/** Watch every response of a page: whether the worker answered it. */
function watch(page) {
  const seen = [];
  const errors = [];
  page.on('response', (r) => seen.push({ url: r.url(), status: r.status(), fromWorker: r.fromServiceWorker() }));
  page.on('pageerror', (e) => errors.push(e.message));
  return { seen, errors };
}

const workers = (page) =>
  page.evaluate(async () => {
    const regs = await navigator.serviceWorker.getRegistrations();
    return {
      scopes: regs.map((r) => r.scope),
      scripts: regs.map((r) => (r.active || r.waiting || r.installing)?.scriptURL ?? null),
      controlled: navigator.serviceWorker.controller?.scriptURL ?? null,
    };
  });

const cacheReport = (page) =>
  page.evaluate(async () => {
    const out = {};
    for (const name of await caches.keys()) {
      const cache = await caches.open(name);
      out[name] = [];
      for (const req of await cache.keys()) {
        const res = await cache.match(req);
        out[name].push({ url: req.url, size: (await res.blob()).size, csp: res.headers.get('content-security-policy') });
      }
    }
    return out;
  });

const waitControlled = (page) => page.waitForFunction(() => navigator.serviceWorker.controller !== null, null, { timeout: 30000 });

/** Wait until the runtime cache has stopped growing (the worker keeps what the first load fetched, a few seconds after it). */
async function settleCaches(page) {
  let last = -1;
  let steady = 0;
  for (let i = 0; i < 60 && steady < 3; i++) {
    await sleep(700);
    const report = await cacheReport(page);
    const count = Object.values(report).reduce((n, list) => n + list.length, 0);
    steady = count === last ? steady + 1 : 0;
    last = count;
  }
  return last;
}

const engine = await startEngine();

/* ------------------------------------------------------------------ */
/* Where the worker is registered                                      */
/* ------------------------------------------------------------------ */
section('the worker registers on /app.html and on no other page');
{
  const site = await startSite(DIST);
  const browser = await launch({ engine });
  const marketing = await browser.context.newPage();
  for (const p of ['/', '/desktop.html']) {
    await marketing.goto(site.origin + p, { waitUntil: 'networkidle' });
    await sleep(2500);
    const state = await workers(marketing);
    ok(`${p} registers no worker`, state.scopes.length === 0 && state.controlled === null, JSON.stringify(state));
  }
  ok('and never asked for /sw.js', site.requests((r) => r.path === '/sw.js').length === 0);

  const page = await browser.context.newPage();
  const net = watch(page);
  await page.goto(`${site.origin}/app.html`, { waitUntil: 'networkidle' });
  await waitControlled(page);
  const state = await workers(page);
  ok('/app.html registers /sw.js with the scope /app.html', state.scopes.length === 1 && state.scopes[0] === `${site.origin}/app.html` && state.scripts[0] === `${site.origin}/sw.js`, JSON.stringify(state));
  ok('and controls the page after the first visit', state.controlled === `${site.origin}/sw.js`);
  ok('no update message on a first install', (await page.locator('[data-testid="update-notice"]').count()) === 0);
  ok('no COOP or COEP is added: the page is not cross-origin isolated', (await page.evaluate(() => crossOriginIsolated)) === false);
  const nav = net.seen.find((r) => r.url === `${site.origin}/app.html`);
  ok('the editor renders', (await page.locator('.app-shell').count()) === 1);

  // The marketing pages of the same origin, now that a worker exists: still not controlled.
  await marketing.goto(`${site.origin}/desktop.html`, { waitUntil: 'networkidle' });
  ok('once the worker exists /desktop.html is still not controlled by it', (await workers(marketing)).controlled === null);
  void nav;

  /* --------------------------------- installability --------------------------------- */
  section('Chrome finds the editor installable');
  const cdp = await browser.context.newCDPSession(page);
  const installability = await cdp.send('Page.getInstallabilityErrors');
  ok('no installability errors', installability.installabilityErrors.length === 0, JSON.stringify(installability.installabilityErrors));
  const parsed = await cdp.send('Page.getAppManifest');
  ok('the manifest is read from /manifest.webmanifest with no errors', parsed.url === `${site.origin}/manifest.webmanifest` && (parsed.errors ?? []).length === 0, JSON.stringify(parsed.errors));
  const data = JSON.parse(parsed.data);
  ok('and it is the one in public/: id, start URL, scope and icons', data.id === '/app.html' && data.start_url === '/app.html' && data.scope === '/' && data.icons.length >= 4);

  /* --------------------------------- what is kept ------------------------------------ */
  section('what is kept, and what is not');
  const settled = await settleCaches(page);
  const report = await cacheReport(page);
  const names = Object.keys(report);
  ok('two caches, named for the build', names.length === 2 && names.every((n) => n.startsWith(`oaiy-web-${builtConfig.buildId}-`)), names.join(', '));
  const shellName = names.find((n) => n.endsWith('-shell'));
  const kept = (report[shellName] ?? []).map((e) => new URL(e.url).pathname).sort();
  ok('the shell cache holds exactly the shell the build listed', JSON.stringify(kept) === JSON.stringify([...builtConfig.precache].sort()), `${kept.length} of ${builtConfig.precache.length}`);
  const runtimeName = names.find((n) => n.endsWith('-runtime'));
  console.log(`    (runtime cache after one visit: ${(report[runtimeName] ?? []).map((e) => new URL(e.url).pathname.replace('/assets/', '')).join(', ')})`);
  ok('the runtime cache holds what the first load fetched besides the shell', (report[runtimeName] ?? []).length > 0, `${(report[runtimeName] ?? []).length} files`);
  const biggest = Math.max(...Object.values(report).flat().map((e) => e.size));
  ok('nothing kept is over 4 MiB', biggest <= 4 * 1024 * 1024, String(biggest));
  ok('the shell was not kept twice', kept.length === new Set(kept).size);
  void settled;

  const probe = (pathname) =>
    page.evaluate(async (p) => {
      const res = await fetch(p);
      const body = await res.arrayBuffer();
      return { status: res.status, bytes: body.byteLength, csp: res.headers.get('content-security-policy') };
    }, pathname);
  const isKept = async (pathname) => Object.values(await cacheReport(page)).flat().some((e) => new URL(e.url).pathname === pathname);
  const answeredByWorker = (pathname) => net.seen.filter((r) => new URL(r.url).pathname === pathname).some((r) => r.fromWorker);

  const small = await probe(SMALL);
  await sleep(600);
  ok('positive control: a small file of the build is kept when the page fetches it', small.status === 200 && (await isKept(SMALL)), `${SMALL} ${JSON.stringify(small)}`);
  const smallChunked = await probe('/__test__/small-chunked.bin');
  await sleep(600);
  ok('and so is a small body with no Content-Length', smallChunked.bytes === 100 * 1024 && (await isKept('/__test__/small-chunked.bin')));

  const ffmpeg = await probe(FFMPEG);
  await sleep(600);
  ok('the ffmpeg core (32 MB) arrives whole', ffmpeg.status === 200 && ffmpeg.bytes === FFMPEG_BYTES, `${ffmpeg.bytes} of ${FFMPEG_BYTES}`);
  ok('and the worker did not answer for it (the build listed it as too big), and it is in no cache', !answeredByWorker(FFMPEG) && !(await isKept(FFMPEG)));
  ok('the build listed the big files: ffmpeg, esbuild and ZIPP engines, the language worker', builtConfig.oversize.length >= 4 && builtConfig.oversize.includes(FFMPEG), builtConfig.oversize.join(' '));

  const big = await probe('/__test__/big-chunked.bin');
  await sleep(600);
  ok('a 5 MB body with no Content-Length arrives whole, is handled by the worker, and is not kept', big.bytes === 5 * 1024 * 1024 && answeredByWorker('/__test__/big-chunked.bin') && !(await isKept('/__test__/big-chunked.bin')), JSON.stringify(big));
  const gz = await probe('/__test__/big-gzip.bin');
  await sleep(600);
  ok('a 5 MB body that is 5 KB on the wire (gzip) arrives whole and is not kept either', gz.bytes === 5 * 1024 * 1024 && !(await isKept('/__test__/big-gzip.bin')), JSON.stringify(gz));

  section('what the worker never touches');
  const apiBefore = site.requests((r) => r.path === '/api/ping').length;
  const api = await page.evaluate(async () => (await fetch('/api/ping')).status);
  ok('/api/ is asked of the network, not answered by the worker, and kept nowhere', api === 404 && site.requests((r) => r.path === '/api/ping').length === apiBefore + 1 && !answeredByWorker('/api/ping') && !(await isKept('/api/ping')));
  const engineBefore = engine.log.length;
  const fromEngine = await page.evaluate(async (origins) => {
    const out = [];
    for (const origin of origins) out.push((await (await fetch(`${origin}/api/health`)).json()).status);
    return out;
  }, [engine.origin, engine.localhostOrigin]);
  ok('the engine on 127.0.0.1 and on localhost (another port) is reached directly', fromEngine.join() === 'ok,ok' && engine.log.length >= engineBefore + 2);
  ok('and the worker answered for neither, and neither is in a cache', !net.seen.filter((r) => r.url.startsWith(engine.origin) || r.url.startsWith(engine.localhostOrigin)).some((r) => r.fromWorker) && Object.values(await cacheReport(page)).flat().every((e) => new URL(e.url).origin === site.origin && !new URL(e.url).pathname.startsWith('/api/')));
  ok('the editor\'s own probe of the engine address went the same way', engine.log.some((r) => r.path === '/api/health'));

  const share = await browser.context.newPage();
  const shareNet = watch(share);
  const shareResponse = await share.goto(`${site.origin}/app.html?flow=abcdefghijklmnopqrstuv`, { waitUntil: 'domcontentloaded' });
  ok('a ?flow= share link is left to the network', shareResponse && !shareResponse.fromServiceWorker());
  await sleep(1500);
  ok('and is kept nowhere', !Object.values(await cacheReport(share)).flat().some((e) => e.url.includes('flow=')));
  void shareNet;
  await share.close();

  const cspFirst = await probe(CSP_WORKER);
  await sleep(600);
  const cspSecond = await probe(CSP_WORKER);
  ok('the compiler sandbox worker keeps its CSP, from the network and from the cache', !!cspFirst.csp && cspFirst.csp === cspSecond.csp && /connect-src 'none'/.test(cspSecond.csp) && (await isKept(CSP_WORKER)), `${cspFirst.csp} | ${cspSecond.csp}`);

  ok('nothing was sent to the product\'s own ports by the editor', site.log.length > 0 && blocked.length === 0, blocked.join(' '));
  ok('the page had no errors', net.errors.length === 0, net.errors.slice(0, 2).join(' | '));

  /* --------------------------------------- install button ---------------------------------------- */
  section('the install button');
  ok('no button until the browser offers', (await page.locator('button[aria-label="Install app"]').count()) === 0);
  const prevented = await page.evaluate(() => {
    window.__prompts = 0;
    const offer = Object.assign(new Event('beforeinstallprompt', { cancelable: true }), {
      prompt: async () => { window.__prompts++; },
      userChoice: Promise.resolve({ outcome: 'dismissed' }),
    });
    window.dispatchEvent(offer);
    return offer.defaultPrevented;
  });
  ok('the browser\'s own bar is cancelled', prevented === true);
  const button = page.locator('button[aria-label="Install app"]');
  await button.waitFor({ timeout: 5000 });
  ok('then "Install app" is in the sidebar', (await button.count()) === 1 && /Install app/.test(await button.textContent()));
  await button.click();
  await sleep(300);
  ok('it plays the offer once and goes', (await page.evaluate(() => window.__prompts)) === 1 && (await button.count()) === 0);

  await browser.close();
  await site.close();
}

/* ------------------------------------------------------------------ */
/* Offline after one visit                                             */
/* ------------------------------------------------------------------ */
section('the shell opens offline after ONE visit');
{
  const site = await startSite(DIST);
  const browser = await launch({ engine });
  const page = await browser.context.newPage();
  const net = watch(page);
  await page.goto(`${site.origin}/app.html`, { waitUntil: 'networkidle' });
  await waitControlled(page);
  await settleCaches(page);
  await site.close(); // the site is gone: not a switch in the browser, the server itself
  const mark = net.seen.length;
  const consoleErrors = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  await page.reload({ waitUntil: 'domcontentloaded' });
  let rendered = true;
  try {
    await page.locator('.app-shell').waitFor({ timeout: 20000 });
  } catch {
    rendered = false;
  }
  ok('reloading with the server down still renders the editor', rendered);
  ok('served by the worker', net.seen.filter((r) => r.url === `${site.origin}/app.html`).some((r) => r.fromWorker));
  ok('the sidebar and canvas are there', rendered && (await page.locator('.oaiy-nav button').count()) === 4);
  // The fonts are fetched while the worker is still installing on a first visit: the page tells the worker
  // what it has loaded, so they come from the cache too.
  const fonts = await page.evaluate(async () => { await document.fonts.ready; return document.fonts.check('16px "Inter Variable"') && document.fonts.check('16px "JetBrains Mono Variable"'); });
  const offlineFonts = net.seen.slice(mark).filter((r) => /\.woff2$/.test(r.url));
  ok('and the self-hosted fonts are there too, from the cache', fonts && offlineFonts.length > 0 && offlineFonts.every((r) => r.status === 200 && r.fromWorker), JSON.stringify(offlineFonts.map((r) => [r.status, r.fromWorker])));
  await sleep(1500);
  ok('with no page errors', net.errors.length === 0, net.errors.slice(0, 3).join(' | '));
  console.log(`    (offline console errors: ${consoleErrors.length}${consoleErrors.length ? `, for example: ${consoleErrors.slice(0, 2).join(' | ').slice(0, 200)}` : ''})`);
  await browser.close();
}

/* ------------------------------------------------------------------ */
/* OAIY's own window                                                    */
/* ------------------------------------------------------------------ */
section("OAIY's own window gets no worker and no install button");
{
  const site = await startSite(DIST);
  const browser = await launch({ engine, desktopWindow: true });
  const page = await browser.context.newPage();
  const net = watch(page);
  await page.goto(`${site.origin}/app.html`, { waitUntil: 'networkidle' });
  await sleep(4000);
  const state = await workers(page);
  ok('no worker is registered where __OAIY_DESKTOP__ is set', state.scopes.length === 0 && state.controlled === null && site.requests((r) => r.path === '/sw.js').length === 0, JSON.stringify(state));
  const prevented = await page.evaluate(() => {
    const offer = Object.assign(new Event('beforeinstallprompt', { cancelable: true }), { prompt: async () => {}, userChoice: Promise.resolve({ outcome: 'accepted' }) });
    window.dispatchEvent(offer);
    return offer.defaultPrevented;
  });
  ok('and the install offer is not even taken', prevented === false && (await page.locator('button[aria-label="Install app"]').count()) === 0);
  ok('nothing was sent to the product\'s own ports', blocked.length === 0, blocked.join(' '));
  void net;
  await browser.close();
  await site.close();
}

/* ------------------------------------------------------------------ */
/* A host that answers a file it has not deployed yet with its front page */
/* ------------------------------------------------------------------ */
section('a shell file answered with the front page is never kept, and the editor still opens');
{
  // The worker fetches its shell with fetch() (Sec-Fetch-Dest: empty); the page loads the same files as scripts
  // (Sec-Fetch-Dest: script). Only the worker's own requests are answered wrongly, once, as the reviewer saw it.
  const SCRIPT = builtConfig.precache.find((p) => /^\/assets\/oaiy-ui-.*\.js$/.test(p)) ?? builtConfig.precache.find((p) => /\.js$/.test(p) && p !== builtConfig.precache[1]);
  const front = fs.readFileSync(path.join(DIST, 'index.html'));
  const wrong = (bad) => (request) => (bad.now && request.path === SCRIPT && request.headers['sec-fetch-dest'] === 'empty'
    ? { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' }, body: front }
    : undefined);
  const scriptBytes = fs.statSync(path.join(DIST, SCRIPT)).size;

  // 1. the first install meets it
  {
    const bad = { now: true };
    const site = await startSite(DIST, { override: (request) => wrong(bad)(request) });
    const browser = await launch({ engine });
    const page = await browser.context.newPage();
    const net = watch(page);
    await page.goto(`${site.origin}/app.html`, { waitUntil: 'networkidle' });
    await sleep(4000);
    ok('the editor opens (the page loaded its own script)', (await page.locator('.app-shell').count()) === 1);
    const state = await workers(page);
    ok('no worker took over the page: its install failed', state.controlled === null, JSON.stringify(state));
    ok('and no half a shell is left in a cache', !Object.keys(await cacheReport(page)).some((n) => n.endsWith('-shell')), Object.keys(await cacheReport(page)).join(', '));

    bad.now = false; // the host has deployed the file
    await page.reload({ waitUntil: 'networkidle' });
    await waitControlled(page);
    await settleCaches(page);
    const kept = Object.values(await cacheReport(page)).flat().find((e) => new URL(e.url).pathname === SCRIPT);
    ok('the next visit installs, and keeps the script itself (not the front page)', !!kept && kept.size === scriptBytes, JSON.stringify(kept));
    for (const reload of [1, 2]) {
      await page.reload({ waitUntil: 'networkidle' });
      await page.locator('.app-shell').waitFor({ timeout: 20000 }).catch(() => undefined);
      ok(`reload ${reload} after it healed opens the editor`, (await page.locator('.app-shell').count()) === 1);
    }
    ok('with no page errors (no "Expected a JavaScript-or-Wasm module script")', net.errors.length === 0, net.errors.slice(0, 2).join(' | '));
    await browser.close();
    await site.close();
  }

  // 2. an update meets it while the build in charge is healthy
  {
    const bad = { now: false };
    const deploy = { v2: false };
    const site = await startSite(DIST, {
      transform: (pathname, body) => (pathname === '/sw.js' && deploy.v2 ? Buffer.from(body.toString('utf8').replace(/"buildId":"[0-9a-f]+"/, '"buildId":"v2bad0000001"')) : body),
      override: (request) => wrong(bad)(request),
    });
    const browser = await launch({ engine });
    const page = await browser.context.newPage();
    await page.goto(`${site.origin}/app.html`, { waitUntil: 'networkidle' });
    await waitControlled(page);
    await settleCaches(page);
    const before = Object.keys(await cacheReport(page)).sort();
    bad.now = true;
    deploy.v2 = true;
    await page.evaluate(async () => { const [reg] = await navigator.serviceWorker.getRegistrations(); await reg.update().catch(() => undefined); });
    await sleep(4000);
    const after = Object.keys(await cacheReport(page)).sort();
    ok('the update did not install: no message, the build in charge keeps its caches and none of the new build is left', (await page.locator('[data-testid="update-notice"]').count()) === 0 && JSON.stringify(after) === JSON.stringify(before), `${before.join(',')} -> ${after.join(',')}`);
    ok('and the page is still under the build in charge', (await workers(page)).controlled === `${site.origin}/sw.js`);
    await page.reload({ waitUntil: 'networkidle' });
    await page.locator('.app-shell').waitFor({ timeout: 20000 }).catch(() => undefined);
    ok('the editor opens after a reload, from the kept shell', (await page.locator('.app-shell').count()) === 1);
    await browser.close();
    await site.close();
  }
}

/* ------------------------------------------------------------------ */
/* An update waits for the person                                       */
/* ------------------------------------------------------------------ */
section('a new build waits, says so, and takes over when the person reloads');
{
  const deploy = { v2: false };
  const site = await startSite(DIST, {
    transform: (pathname, body) => (pathname === '/sw.js' && deploy.v2 ? Buffer.from(body.toString('utf8').replace(/"buildId":"[0-9a-f]+"/, '"buildId":"v2build00001"')) : body),
  });
  const browser = await launch({ engine });
  const page = await browser.context.newPage();
  await page.goto(`${site.origin}/app.html`, { waitUntil: 'networkidle' });
  await waitControlled(page);
  await settleCaches(page);
  const before = Object.keys(await cacheReport(page));
  ok('the first build is in charge', before.every((n) => n.includes(builtConfig.buildId)));
  await page.evaluate(() => { window.__samePage = true; });

  deploy.v2 = true;
  await page.evaluate(async () => { const [reg] = await navigator.serviceWorker.getRegistrations(); await reg.update(); });
  const notice = page.locator('[data-testid="update-notice"]');
  await notice.waitFor({ timeout: 30000 });
  ok('the message says a new version is ready', /A new version is ready\./.test(await notice.textContent()));
  await sleep(1500);
  ok('the page did not reload by itself', (await page.evaluate(() => window.__samePage)) === true);
  const waiting = await page.evaluate(async () => { const [reg] = await navigator.serviceWorker.getRegistrations(); return { waiting: !!reg.waiting, active: !!reg.active }; });
  ok('the new worker is installed and waiting, the old one still in charge', waiting.waiting && waiting.active);
  const during = Object.keys(await cacheReport(page));
  ok('both builds\' caches exist meanwhile', during.some((n) => n.includes('v2build00001')) && during.some((n) => n.includes(builtConfig.buildId)), during.join(', '));

  await notice.getByRole('button', { name: 'Reload' }).click();
  await page.waitForFunction(() => !window.__samePage, null, { timeout: 30000 });
  await page.locator('.app-shell').waitFor({ timeout: 30000 });
  await waitControlled(page);
  await sleep(1000);
  const after = Object.keys(await cacheReport(page));
  ok('after "Reload" the page is the new build\'s and the old build\'s caches are gone', after.length >= 1 && after.every((n) => n.includes('v2build00001')), after.join(', '));
  ok('and no message is left', (await page.locator('[data-testid="update-notice"]').count()) === 0);
  await browser.close();
  await site.close();
}

await engine.close();
console.log(`\n${'-'.repeat(60)}`);
console.log(`pwa e2e: ${pass} passed, ${failures.length} failed`);
if (failures.length) console.log(`failed:\n  - ${failures.join('\n  - ')}`);
process.exit(failures.length ? 1 : 0);
