// The Agent as an installable, updatable app: the production build on a plain static server that sends
// NO isolation headers, in a browser of its own (a temporary profile).
//   node tests/e2e/pwa.mjs [--build]
//
// It proves that
//  - the manifest names OAIY, and every icon it (and the page) names is served, with the right type and size;
//  - Chrome reports NO installability errors for the served build (CDP Page.getInstallabilityErrors);
//  - the first service worker starts at once, and it removes the caches older than the previous build's (the
//    app's old name included) and nobody else's;
//  - "Install app" shows only where the browser offers it (never as an installed app, never in OAIY's window),
//    and iOS Safari gets one sentence;
//  - an upgrade whose precache cannot be fetched fails to install: the running worker stays and nothing is announced;
//  - a new version waits: the page says so and does not reload by itself; Reload asks first when the page would
//    object to leaving (the agent is at work) and sends nothing if the person stays; agreeing takes the update and
//    reloads once with no second prompt; when the browser's own leave prompt stops the reload, the next click works;
//  - a second tab that has not reloaded still finds its files (one the host deleted) after the first tab took the
//    update;
//  - the app still opens, isolated and with its sandbox, with the server gone.
//
// Settings (all optional): CHROME (the browser), PWA_E2E_PORT (the static server's port; default: any free
// one), PWA_E2E_DEBUG_PORT (the browser's remote-debugging port; default: any free one).
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { basename, extname, join, normalize } from 'node:path';
import { execSync } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const root = join(process.cwd(), 'dist');
if (!existsSync(join(root, 'index.html')) || process.argv.includes('--build')) execSync('npx vite build', { stdio: 'inherit' });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.png': 'image/png', '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.woff2': 'font/woff2' };

// What the server answers, changed by the tests: which service worker file it serves (the build's own, or a
// newer version of it), a hashed file that only the first version has (the host deleted it when it deployed
// the next), and paths that answer 404.
const OLD_CHUNK = '/assets/old-chunk-1a2b3c4d.js';
const serving = { workerVersion: 1, oldChunk: true, missing: new Set() };
// The cache of version 1 is the build's own; the later versions (other bytes of sw.js) have made-up names.
const cacheOf = (version) => (version === 1 ? builtCache : `oaiy-agent-${String(version).padStart(12, '0')}`);
const NEXT_CACHE = cacheOf(2);
const stamped = readFileSync(join(root, 'sw.js'), 'utf8');
const CACHE_LINE = /const CACHE = '(oaiy-agent-[0-9a-f]{12})';/;
const builtCache = CACHE_LINE.exec(stamped)?.[1];

// A model server that starts an answer and keeps it open: while it does, the agent is at work and the page's
// leave guard objects to leaving.
const heldReplies = [];
const model = createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  res.setHeader('Access-Control-Allow-Private-Network', 'true');
  if (req.method === 'OPTIONS') return res.end();
  if (req.url.endsWith('/models')) {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({ data: [{ id: 'mock-model', context_length: 131072 }] }));
  }
  if (req.method === 'GET') {
    res.statusCode = 404;
    return res.end();
  }
  res.setHeader('content-type', 'text/event-stream');
  res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'Working on it' } }] })}\n\n`);
  heldReplies.push(res);
});
await new Promise((r) => model.listen(0, '127.0.0.1', r));
const modelUrl = `http://127.0.0.1:${model.address().port}`;

const requests = [];
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  requests.push(path);
  // A blank page of the same origin that does not register the worker: where the old caches are put.
  if (path === '/__seed') {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><meta charset="utf-8"><title>seed</title>');
    return;
  }
  if (path === OLD_CHUNK && serving.oldChunk) {
    res.setHeader('content-type', 'text/javascript');
    res.end('export const chunk = "the first version";');
    return;
  }
  const relative = normalize(path).replace(/^([/\\])+/, '');
  let file = join(root, relative);
  const isPage = path === '/' || path === '/index.html';
  // A file that is not there is a 404 (not the page), so a wrong path in the manifest cannot pass.
  if (serving.missing.has(path) || !file.startsWith(root) || (!isPage && (!existsSync(file) || statSync(file).isDirectory()))) {
    res.statusCode = 404;
    res.setHeader('content-type', 'text/plain');
    res.end('not found');
    return;
  }
  if (isPage) file = join(root, 'index.html');
  res.setHeader('content-type', types[extname(file)] ?? 'application/octet-stream');
  if (path === '/sw.js' && serving.workerVersion >= 2) {
    // A next build: another cache name, so other bytes.
    res.end(stamped.replace(CACHE_LINE, `const CACHE = '${cacheOf(serving.workerVersion)}';`) + `\n// version ${serving.workerVersion}\n`);
    return;
  }
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(Number(process.env.PWA_E2E_PORT) || 0, '127.0.0.1', r));
const port = server.address().port;
const base = `http://127.0.0.1:${port}/`;

const executablePath = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].filter(Boolean).find((p) => existsSync(p));
if (!executablePath) {
  console.error('no Chrome found; set CHROME');
  process.exit(2);
}
// A browser of its own, with a new profile: an old profile would keep an old worker and old caches.
const PROFILE_PREFIX = 'oaiy-pwa-e2e-';
const profile = mkdtempSync(join(tmpdir(), PROFILE_PREFIX));
const args = ['--no-first-run', '--no-default-browser-check'];
if (process.env.PWA_E2E_DEBUG_PORT) args.push(`--remote-debugging-port=${process.env.PWA_E2E_DEBUG_PORT}`);
const browser = await puppeteer.launch({ executablePath, headless: true, userDataDir: profile, args });

let failures = 0;
const check = async (name, fn) => {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${name}\n     ${String(error.message).split('\n').join('\n     ')}`);
  }
};
const expect = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (actual, wanted, what) => expect(JSON.stringify(actual) === JSON.stringify(wanted), `${what}: wanted ${JSON.stringify(wanted)}, got ${JSON.stringify(actual)}`);

/** The size and kind of a PNG, from its header. */
function pngInfo(bytes) {
  expect(bytes.length > 33 && bytes.readUInt32BE(0) === 0x89504e47 && bytes.readUInt32BE(4) === 0x0d0a1a0a, 'not a PNG');
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20), colorType: bytes[25] };
}
async function get(path) {
  const response = await fetch(new URL(path, base));
  return { status: response.status, type: (response.headers.get('content-type') ?? '').split(';')[0], bytes: Buffer.from(await response.arrayBuffer()) };
}
async function terminal(page, command, waitFor) {
  // A tab in the background gets no frames, and a click waits for one.
  await page.bringToFront();
  await page.click('.term-input');
  await page.type('.term-input', command);
  await page.keyboard.press('Enter');
  try {
    await page.waitForFunction((w) => document.querySelector('.term-out')?.textContent.includes(w), { timeout: 30_000 }, waitFor);
  } catch {
    throw new Error(`the terminal never showed "${waitFor}":\n${(await page.$eval('.term-out', (el) => el.textContent)).slice(-400)}`);
  }
}
/** Opens the project menu and says whether "Install app" is in it, and what it says. */
async function installItem(page) {
  await page.waitForSelector('.menu-toggle', { timeout: 30_000 });
  await page.evaluate(() => document.querySelector('header.topbar')?.classList.add('menu-open'));
  const item = await page.evaluate(() => {
    const el = document.querySelector('.actions .install-app');
    return el && getComputedStyle(el).display !== 'none' ? el.textContent : null;
  });
  await page.evaluate(() => document.querySelector('header.topbar')?.classList.remove('menu-open'));
  return item;
}
const offer = (page, outcome = 'dismissed') =>
  page.evaluate((o) => {
    // What Chrome sends when the app can be installed.
    const event = new Event('beforeinstallprompt', { cancelable: true });
    event.prompt = () => {
      window.__prompts = (window.__prompts ?? 0) + 1;
      return Promise.resolve();
    };
    event.userChoice = Promise.resolve({ outcome: o });
    window.dispatchEvent(event);
    return event.defaultPrevented;
  }, outcome);
/** A browser that never offers to install: the events Chrome itself sends are dropped before the app hears them. */
const withoutBrowserOffer = (page) =>
  page.evaluateOnNewDocument(() => {
    window.addEventListener(
      'beforeinstallprompt',
      (event) => {
        if (event.isTrusted) event.stopImmediatePropagation();
      },
      true,
    );
  });
const cacheKeys = (page) => page.evaluate(() => caches.keys());
const isolated = (page) => page.waitForFunction(() => globalThis.crossOriginIsolated === true, { timeout: 30_000, polling: 250 });

try {
  const manifest = JSON.parse(readFileSync(join(root, 'manifest.webmanifest'), 'utf8'));
  const html = readFileSync(join(root, 'index.html'), 'utf8');

  // ---- the files of the build ----
  await check('the build stamped its service worker with a cache name of this app', () => {
    expect(builtCache, `dist/sw.js has no "const CACHE = 'oaiy-agent-<12 hex>';" line`);
    expect(!/bot\.computer-[0-9a-f]{6,}/.test(stamped), 'the worker still names a bot.computer cache');
    expect(stamped.includes("'bot.computer-'"), 'the worker no longer knows to remove the bot.computer caches');
  });

  await check('the manifest names OAIY and its colours are the app own', async () => {
    const served = await get('manifest.webmanifest');
    same([served.status, served.type], [200, 'application/manifest+json'], 'the manifest');
    same([manifest.name, manifest.short_name], ['OAIY', 'OAIY'], 'name and short name');
    expect(/OAIY/.test(manifest.description) && !/bot\.computer/i.test(JSON.stringify(manifest)), `the description: ${manifest.description}`);
    expect(typeof manifest.id === 'string' && manifest.id.length > 0, 'the manifest has no id');
    same([manifest.display, manifest.start_url, manifest.scope], ['standalone', './', './'], 'display, start_url and scope');
    // The app's dark background (styles.css --bg), which is also the page's theme-color.
    same([manifest.theme_color, manifest.background_color], ['#080d18', '#080d18'], 'theme and background colours');
    expect(html.includes('<meta name="theme-color" content="#080d18"') && html.includes('content="#f8f5ee"'), 'the page has no theme-color for both themes');
  });

  await check('every icon the manifest names is served with its type and size', async () => {
    const icons = manifest.icons ?? [];
    for (const icon of icons) {
      const served = await get(icon.src);
      expect(served.status === 200, `${icon.src}: HTTP ${served.status}`);
      expect(served.type === icon.type, `${icon.src}: served as ${served.type}, the manifest says ${icon.type}`);
      expect(icon.purpose === 'any' || icon.purpose === 'maskable', `${icon.src}: purpose "${icon.purpose}" (each icon is one or the other)`);
      if (icon.type === 'image/png') {
        const png = pngInfo(served.bytes);
        same([png.width, png.height], icon.sizes.split('x').map(Number), `${icon.src}: size`);
        if (icon.purpose === 'maskable') expect(png.colorType === 2, `${icon.src}: a maskable icon has no transparency (colour type ${png.colorType})`);
      } else {
        expect(icon.type === 'image/svg+xml' && served.bytes.toString('utf8').includes('<svg'), `${icon.src}: not an SVG`);
      }
    }
    const has = (size, purpose) => icons.some((i) => i.type === 'image/png' && i.sizes === `${size}x${size}` && i.purpose === purpose);
    expect(has(192, 'any') && has(512, 'any') && has(512, 'maskable'), 'the manifest lacks a PNG 192 and 512 (any) and a PNG 512 (maskable)');
  });

  await check('the page names a PNG favicon, an SVG favicon and an opaque apple-touch-icon that are served', async () => {
    const link = (rel) => [...html.matchAll(new RegExp(`<link rel="${rel}"[^>]*>`, 'g'))].map((m) => /href="([^"]+)"/.exec(m[0])?.[1]);
    const [png32] = link('icon').filter((h) => h?.endsWith('.png'));
    const [svg] = link('icon').filter((h) => h?.endsWith('.svg'));
    const [touch] = link('apple-touch-icon');
    expect(png32 && svg && touch, `icons in the page: ${JSON.stringify(link('icon'))} ${JSON.stringify(link('apple-touch-icon'))}`);
    const small = await get(png32);
    same([small.status, small.type, pngInfo(small.bytes).width], [200, 'image/png', 32], 'the PNG favicon');
    const vector = await get(svg);
    same([vector.status, vector.type], [200, 'image/svg+xml'], 'the SVG favicon');
    const ios = await get(touch);
    const info = pngInfo(ios.bytes);
    same([ios.status, ios.type, info.width, info.height], [200, 'image/png', 180, 180], 'the apple-touch-icon');
    expect(info.colorType === 2, 'the apple-touch-icon has transparency (iOS would fill it with black)');
  });

  // ---- a first visit: old caches, the first worker, installability ----
  // Caches that an earlier build, the app's old name and an unrelated user of this origin left.
  const seed = await browser.newPage();
  await seed.goto(`${base}__seed`);
  await seed.evaluate(async () => {
    // In this order: the newest of the app's caches is the one the first worker keeps (it has no record of a build that ran).
    for (const name of ['oaiy-agent-000000000001', 'bot.computer-deadbeef', 'unrelated-cache']) await (await caches.open(name)).put('/old', new Response('old'));
  });
  await seed.close();

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  // What the browser itself offered (headless may offer nothing), noted before the app can hear it.
  await page.evaluateOnNewDocument(() => {
    window.__realOffers = 0;
    window.addEventListener('beforeinstallprompt', () => window.__realOffers++);
  });
  let tabB = null;
  let navigations = 0;
  page.on('framenavigated', (frame) => {
    if (frame === page.mainFrame()) navigations++;
  });
  await page.goto(base);
  await isolated(page);
  await page.waitForSelector('.tree-row', { timeout: 30_000 });

  await check('the first worker starts at once: nothing waits, and no update is announced', async () => {
    const state = await page.evaluate(async () => {
      const registration = await navigator.serviceWorker.getRegistration();
      return { active: !!registration?.active, waiting: !!registration?.waiting, notice: !!document.querySelector('.update-notice') };
    });
    same(state, { active: true, waiting: false, notice: false }, 'the worker and the notice');
  });

  await check("the caches older than the previous build are removed (that one is kept), and nobody else's", async () => {
    let keys = [];
    for (let i = 0; i < 40; i++) {
      keys = (await cacheKeys(page)).sort();
      if (!keys.includes('oaiy-agent-000000000001')) break;
      await new Promise((r) => setTimeout(r, 250));
    }
    // The bot.computer cache was the newest of the app's, so it is the "previous build" a tab of the old days still runs.
    same(keys, [builtCache, 'bot.computer-deadbeef', 'unrelated-cache'].sort(), 'the caches');
  });

  await check('the cache holds every file the worker precaches', async () => {
    const list = /const PRECACHE = (\[[^\]]*\]);/.exec(stamped)?.[1];
    expect(list, 'no PRECACHE list in sw.js');
    const wanted = JSON.parse(list.replace(/'/g, '"'));
    const missing = await page.evaluate(async (names, cacheName) => {
      const cache = await caches.open(cacheName);
      const absent = [];
      for (const name of names) if (!(await cache.match(new URL(name, location.href)))) absent.push(name);
      return absent;
    }, wanted, builtCache);
    same(missing, [], 'precached files missing from the cache');
  });

  await check('Chrome reports no installability errors for the served build', async () => {
    const client = await page.createCDPSession();
    let errors = [{ errorId: 'not checked' }];
    for (let i = 0; i < 40 && errors.length; i++) {
      errors = (await client.send('Page.getInstallabilityErrors')).installabilityErrors;
      if (errors.length) await new Promise((r) => setTimeout(r, 250));
    }
    same(errors, [], 'installability errors');
    const { errors: manifestErrors } = await client.send('Page.getAppManifest');
    same(manifestErrors, [], 'manifest errors');
    const { appId } = await client.send('Page.getAppId').catch(() => ({ appId: null }));
    console.log(`     manifest id: ${manifest.id}; Chrome's app id: ${appId}`);
    if (appId) expect(appId === new URL(manifest.id, base).href, `the app id ${appId} is not made from the manifest id ${manifest.id}`);
    // The manifest had no id before the app was called OAIY, which makes the id the start URL's origin: the same
    // id keeps a copy that was installed then as the same app, so it takes the new name and icons.
    expect(appId === base, `the app id ${appId} is not the origin (${base}), where an earlier install has it`);
    await client.detach();
  });

  // ---- the install item ----
  const realOffers = await page.evaluate(() => window.__realOffers);
  console.log(`     Chrome itself offered the install ${realOffers} time(s) in this browser`);
  await check('"Install app" is in the menu when the browser offers it (Chrome\'s own offer, when it makes one)', async () => {
    const before = await installItem(page);
    if (realOffers > 0) expect(before === 'Install app', `Chrome offered, but the menu says ${before}`);
    expect((await offer(page)) === true, 'the page did not keep the browser event (preventDefault)');
    expect((await installItem(page)) === 'Install app', 'no "Install app" item after the browser offered');
    await page.evaluate(() => document.querySelector('header.topbar')?.classList.add('menu-open'));
    await page.click('.actions .install-app');
    same(await page.evaluate(() => window.__prompts), 1, 'times the browser prompt was opened');
    // Used once; the answer was "dismissed", so the offer is gone until the browser sends another.
    expect(await installItem(page) === null, 'the item is still there after the prompt was used');
  });

  await check('"Install app" is not there when the browser never offers it', async () => {
    const quiet = await browser.newPage();
    await withoutBrowserOffer(quiet);
    await quiet.goto(base);
    await quiet.waitForSelector('.tree-row', { timeout: 30_000 });
    same(await installItem(quiet), null, 'the item in a browser that offers nothing');
    // ...and it is there as soon as the browser does.
    await offer(quiet);
    same(await installItem(quiet), 'Install app', 'the item once the browser offers');
    await quiet.close();
  });

  await check('an app that is already installed is not offered "Install app"', async () => {
    const installed = await browser.newPage();
    // What a window of an installed app answers to (headless Chrome cannot open one).
    await installed.evaluateOnNewDocument(() => {
      const ask = window.matchMedia.bind(window);
      window.matchMedia = (query) => (/display-mode:\s*standalone/.test(query) ? { matches: true, media: query, onchange: null, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false } : ask(query));
    });
    await installed.goto(base);
    await installed.waitForSelector('.tree-row', { timeout: 30_000 });
    await offer(installed);
    same(await installItem(installed), null, 'the item in a standalone window');
    await installed.close();
  });

  await check('iOS Safari gets one sentence: Share, then Add to Home Screen', async () => {
    const ios = await browser.newPage();
    await withoutBrowserOffer(ios);
    await ios.setUserAgent('Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1');
    await ios.goto(base);
    await ios.waitForSelector('.tree-row', { timeout: 30_000 });
    same(await installItem(ios), 'Install app', 'the item on iOS Safari');
    await ios.evaluate(() => document.querySelector('header.topbar')?.classList.add('menu-open'));
    await ios.click('.actions .install-app');
    await ios.waitForSelector('dialog.modal[open]', { timeout: 5000 });
    const said = await ios.$eval('dialog.modal', (d) => d.textContent);
    expect(/Share/.test(said) && /Add to Home Screen/.test(said), `the dialog says: ${said}`);
    await ios.close();
  });

  await check('in OAIY own window nothing is offered, the browser event is left alone and no service worker is registered', async () => {
    const window_ = await browser.newPage();
    // What OAIY gives its window before the page runs: the desktop's address and a token. (The page is served from this test's own server,
    // so it is the desktop that says what it is: a name and a port of oaiy.localhost do not make a window. The desktop given is the test's
    // model server, which answers nothing of a desktop.)
    await window_.evaluateOnNewDocument((origin) => {
      window.__OAIY_DESKTOP__ = Object.freeze({ origin, token: 'window-token', theme: 'dark' });
    }, modelUrl);
    await window_.goto(`http://oaiy.localhost:${port}/`);
    await window_.waitForSelector('.menu-toggle', { timeout: 30_000 });
    expect(await offer(window_) === false, 'the page took the browser event inside OAIY window');
    same(await installItem(window_), null, 'the item in OAIY window');
    same(await window_.evaluate(async () => (await navigator.serviceWorker.getRegistrations()).length), 0, 'service workers registered');
    await window_.close();
  });

  await check('a page at oaiy.localhost on a port, given no desktop, is a tab and not OAIY\'s window: the browser event is kept and "Install app" is offered', async () => {
    const tab = await browser.newPage();
    await withoutBrowserOffer(tab);
    await tab.goto(`http://oaiy.localhost:${port}/`);
    await tab.waitForSelector('.tree-row', { timeout: 30_000 });
    expect((await offer(tab)) === true, 'the page did not keep the browser event: it took itself for OAIY\'s window');
    same(await installItem(tab), 'Install app', 'the item at oaiy.localhost on a port');
    await tab.close();
  });

  // ---- an update ----
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const stateOf = () =>
    page.evaluate(async () => {
      const registration = await navigator.serviceWorker.getRegistration();
      return { active: registration.active?.state, waiting: registration.waiting?.state, notice: !!document.querySelector('.update-notice'), alive: window.__stillHere === true };
    });
  // A real click (the browser only asks before leaving a page that has had one).
  const clickReload = async () => {
    await page.bringToFront();
    await page.click('.update-notice button.primary');
  };
  const askUpdate = () =>
    page.evaluate(async () => {
      const registration = await navigator.serviceWorker.getRegistration();
      await registration.update().catch(() => {});
    });
  const cachesAre = async (wanted, what) => {
    let keys = [];
    for (let i = 0; i < 40; i++) {
      keys = (await cacheKeys(page)).sort();
      if (JSON.stringify(keys) === JSON.stringify([...wanted].sort())) break;
      await sleep(250);
    }
    same(keys, [...wanted].sort(), what);
  };
  let navigationsBefore = 0;

  await check('an upgrade whose precache cannot be fetched fails to install: the running worker stays, and no update is announced', async () => {
    await page.evaluate(() => {
      window.__stillHere = true;
    });
    // The host deploys the next version, but one of the files its worker precaches is not served.
    serving.workerVersion = 2;
    serving.missing.add('/icon-512.png');
    await askUpdate();
    for (let i = 0; i < 80 && (await page.evaluate(async () => !!(await navigator.serviceWorker.getRegistration()).installing)); i++) await sleep(250);
    await sleep(1500);
    same(await stateOf(), { active: 'activated', notice: false, alive: true }, 'the workers and the notice after the failed install');
    // No cache of the failed version is left behind (an empty one would be taken for the previous build).
    same((await cacheKeys(page)).includes(NEXT_CACHE), false, 'the failed version cache');
    same((await cacheKeys(page)).includes(builtCache), true, 'the running version cache');
    serving.missing.delete('/icon-512.png');
  });

  await check('a new version waits: the page says so, and does not reload by itself', async () => {
    await page.evaluate(() => {
      window.__stillHere = true;
    });
    // A second tab on the running version, which has loaded one of its hashed files.
    tabB = await browser.newPage();
    await tabB.goto(base);
    await isolated(tabB);
    await tabB.waitForSelector('.tree-row', { timeout: 30_000 });
    same(await tabB.evaluate(async (u) => (await fetch(u)).status, OLD_CHUNK), 200, "the first version's hashed file, before the update");
    for (let i = 0; i < 40 && !(await tabB.evaluate((u) => caches.match(u).then(Boolean), OLD_CHUNK)); i++) await sleep(250);
    // The host deploys the next version, and the first one's hashed file is gone from it.
    serving.oldChunk = false;
    navigationsBefore = navigations;
    serving.workerVersion = 2;
    await askUpdate();
    await page.waitForSelector('.update-notice', { timeout: 30_000 });
    const notice = await page.$eval('.update-notice', (n) => n.textContent);
    expect(/A new version is ready/.test(notice) && /Reload/.test(notice), `the notice says: ${notice}`);
    // Left alone, the page stays as it is, and the new worker only waits.
    await sleep(2000);
    same(await stateOf(), { active: 'activated', waiting: 'installed', notice: true, alive: true }, 'the page and its workers while a new one waits');
    same(navigations, navigationsBefore, 'navigations before Reload');
    same(await page.evaluate(async () => (await caches.keys()).includes('oaiy-agent-000000000002')), true, 'the new version cached its files while waiting');
    same((await cacheKeys(page)).includes(builtCache), true, 'the running version kept its cache while the new one waits');
  });

  await check('Reload asks first when the page would object to leaving (the agent is at work), and staying sends nothing', async () => {
    // A provider, and a question that the model keeps answering: the agent is at work.
    await page.bringToFront();
    await page.click('button[title="AI providers"]');
    await page.waitForSelector('dialog.settings[open]');
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent.includes('Add a provider')).click());
    await page.select('dialog.settings select', 'local-other');
    await page.evaluate((url) => {
      const input = [...document.querySelectorAll('dialog.settings label')].find((l) => l.firstChild.textContent === 'Address').querySelector('input');
      input.value = url;
      input.dispatchEvent(new Event('input'));
      [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'List models').click();
    }, modelUrl);
    await page.waitForFunction(() => [...document.querySelectorAll('.model-picker select option')].some((o) => o.value === 'mock-model'), { timeout: 15_000 });
    await page.select('.model-picker select', 'mock-model');
    await page.waitForFunction(() => document.querySelector('.provider-row.selected')?.textContent.includes('mock-model'));
    await page.evaluate(() => [...document.querySelectorAll('dialog.settings button')].find((b) => b.textContent === 'Save').click());
    await page.waitForFunction(() => document.querySelector('button[title="AI provider"]')?.textContent.includes('mock-model'));
    await page.type('.chat-input', 'Take your time.');
    await page.keyboard.press('Enter');
    for (let i = 0; i < 100 && !heldReplies.length; i++) await sleep(100);
    expect(heldReplies.length > 0, 'the agent never asked the model');

    await clickReload();
    await page.waitForSelector('dialog.modal[open]', { timeout: 10_000 });
    const said = await page.$eval('dialog.modal', (d) => d.textContent);
    expect(/Reload OAIY\?/.test(said), `the dialog says: ${said}`);
    // Stay.
    await page.evaluate(() => [...document.querySelectorAll('dialog.modal button')].find((b) => b.textContent === 'Cancel').click());
    await sleep(2000);
    // Nothing was sent to the waiting worker, so it still waits; the page did not move; the button works.
    same(await stateOf(), { active: 'activated', waiting: 'installed', notice: true, alive: true }, 'the page and its workers after staying');
    same(navigations, navigationsBefore, 'navigations after staying');
    same(await page.$eval('.update-notice button.primary', (b) => !b.disabled), true, 'the Reload button is usable');
  });

  await check('agreeing to leave takes the update and reloads once, with no second question from the browser', async () => {
    const dialogs = [];
    const onDialog = (dialog) => {
      dialogs.push(dialog.type());
      dialog.accept();
    };
    page.on('dialog', onDialog);
    await clickReload();
    await page.waitForSelector('dialog.modal[open]', { timeout: 10_000 });
    await page.evaluate(() => [...document.querySelectorAll('dialog.modal button')].find((b) => b.textContent === 'Reload').click());
    await page.waitForFunction(() => window.__stillHere === undefined, { timeout: 30_000, polling: 100 });
    await isolated(page);
    await page.waitForSelector('.tree-row', { timeout: 30_000 });
    await sleep(2000);
    page.off('dialog', onDialog);
    same(dialogs, [], "the browser's own leave prompts");
    same(navigations - navigationsBefore, 1, 'navigations after Reload');
    same(await page.evaluate(() => !!document.querySelector('.update-notice')), false, 'the notice after reloading');
    // The version before is kept for one more update; the bot.computer cache, older than that, is gone.
    await cachesAre([builtCache, NEXT_CACHE, 'unrelated-cache'], 'the caches after the update');
  });

  await check('a tab that has not reloaded still finds its files after another tab took the update', async () => {
    // Tab B still runs the first version, and the worker that serves it now is the new one. The host no longer has the
    // first version's hashed file; the new worker finds it in the first version's cache.
    same(await tabB.evaluate(async (u) => (await fetch(u)).status, OLD_CHUNK), 200, "the first version's hashed file, after the update");
    // A file that no version ever had is still a 404: the 200 above came from the cache.
    same(await tabB.evaluate(async () => (await fetch('/assets/never-was-1a2b3c4d.js')).status), 404, 'a file no version has');
    await tabB.close();
  });

  await check("when the browser's own leave prompt stops the reload, the page goes on and the next click on Reload works", async () => {
    // A third version, and a page that the browser will not leave once (a guard of its own: only for a real navigation).
    await page.bringToFront();
    await page.evaluate(() => {
      window.__stillHere = true;
      window.__stay = 1;
      window.addEventListener('beforeunload', (event) => {
        if (event.isTrusted && window.__stay > 0) {
          window.__stay--;
          event.preventDefault();
        }
      });
    });
    navigationsBefore = navigations;
    serving.workerVersion = 3;
    await askUpdate();
    await page.waitForSelector('.update-notice', { timeout: 30_000 });
    const dialogs = [];
    const onDialog = (dialog) => {
      dialogs.push(dialog.type());
      // The person stays the first time, and leaves the next.
      if (dialogs.length === 1) dialog.dismiss();
      else dialog.accept();
    };
    page.on('dialog', onDialog);
    try {
      await clickReload();
      // The worker takes over, the page tries to reload, the browser asks, the person stays.
      for (let i = 0; i < 100 && !dialogs.length; i++) await sleep(200);
      same(dialogs, ['beforeunload'], "the browser's leave prompts after the first click");
      await sleep(1000);
      same(await stateOf(), { active: 'activated', notice: true, alive: true }, 'the page and its workers after staying');
      same(navigations, navigationsBefore, 'navigations after staying');
      // The Reload button comes back, and works.
      await page.waitForFunction(() => !document.querySelector('.update-notice button.primary').disabled, { timeout: 30_000, polling: 250 });
      await clickReload();
      await page.waitForFunction(() => window.__stillHere === undefined, { timeout: 30_000, polling: 100 });
      await isolated(page);
      await page.waitForSelector('.tree-row', { timeout: 30_000 });
      await sleep(2000);
    } finally {
      page.off('dialog', onDialog);
    }
    same(dialogs, ['beforeunload'], "the browser's leave prompts in all");
    same(navigations - navigationsBefore, 1, 'navigations after the second click');
    same(await page.evaluate(() => !!document.querySelector('.update-notice')), false, 'the notice after reloading');
    await cachesAre([cacheOf(2), cacheOf(3), 'unrelated-cache'], 'the caches after the second update');
  });

  // ---- offline ----
  await check('with the server gone, the app opens from the cache, isolated, and the sandbox runs', async () => {
    await terminal(page, 'python hello.py', 'mean temperature: 19.8');
    await new Promise((r) => setTimeout(r, 1500));
    server.closeAllConnections();
    server.close();
    await page.close();
    const offline = await browser.newPage();
    await offline.setViewport({ width: 1400, height: 900 });
    await offline.goto(base);
    await offline.waitForSelector('.tree-row', { timeout: 30_000 });
    expect(await offline.evaluate(() => globalThis.crossOriginIsolated), 'not isolated offline');
    await terminal(offline, 'node hello.js', 'warmest: Cairo');
  });
} finally {
  for (const reply of heldReplies) reply.end();
  await browser.close().catch(() => {});
  server.close();
  model.close();
  // The profile is ours (made above, named for this test): remove it.
  if (basename(profile).startsWith(PROFILE_PREFIX) && profile.startsWith(tmpdir())) {
    try {
      rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    } catch (error) {
      console.log(`  (could not remove ${profile}: ${error.message})`);
    }
  }
}
console.log(failures ? `\n${failures} failed` : '\nall passed');
process.exit(failures ? 1 : 0);
