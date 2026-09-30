/**
 * When the editor looks for OAIY Desktop, and when it does not.
 *
 *     npm run test:desktop-probes
 *
 * OAIY's own window looks for the desktop the moment the editor opens, as it always has. A tab in a browser must not: on a public
 * address that request goes to 127.0.0.1 or the LAN, which is a permission prompt for the visitor (Chrome's Local Network Access)
 * and shows the site what is on their network. A tab looks when its person presses Connect, and, once the desktop has answered,
 * when the editor opens from then on (the link); Disconnect forgets the link. Each scenario here is a page of its own, with
 * its own copy of the modules, its own storage and a fetch that records every request and a clock that never ticks by itself.
 *
 *   - a tab with no link: no request, no timer, and the services a session with a desktop left in storage are cleared;
 *   - OAIY's window, a tab with a link, a tab with an address of its own: the two probes of the page opening, as before;
 *   - Connect: ONE health request, the link kept, the poll and the service list started without asking again;
 *   - Connect with nothing there: one request, nothing kept, nothing keeps asking;
 *   - Disconnect: nothing kept, nothing asking, the palette's desktop services gone.
 *
 * The modules are TypeScript with aliases only the bundler resolves, so they are bundled for Node with esbuild (as
 * engine-endpoint.mjs does); oaiy-ui-components is stubbed.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';
import { UI, suite } from './support/loadTs.mjs';

const { check, finish } = suite('desktop probes');

const stub = path.join(os.tmpdir(), `oaiy-desktop-probes-stub-${process.pid}.mjs`);
fs.writeFileSync(stub, 'export function invalidateDynamicOptions() {}\nexport function subscribeToDynamicOptionsInvalidation() { return () => {}; }\n');
const bundlePath = path.join(os.tmpdir(), `oaiy-desktop-probes-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export * as detection from './src/lib/desktopDetection.ts';
export * as services from './src/lib/desktopServices.ts';
export * as link from './src/lib/desktopLink.ts';
export * as connect from './src/lib/desktopConnect.ts';
export * as endpoint from './src/lib/engineEndpoint.ts';
export * as words from './src/lib/connectWords.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  plugins: [{ name: 'oaiy-ui-components', setup: (build) => build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: stub })) }],
});

const DEFAULT = 'http://127.0.0.1:17972';
const LAN = 'http://192.168.1.50:17972';
const KEYS = { services: 'oaiy.desktopServices', linked: 'oaiy.desktopLinked', engine: 'oaiy.engineBase' };

let n = 0;
/**
 * A page of its own: the globals a browser gives it (its address, storage, a fetch that records, an interval that only records),
 * and a fresh copy of the modules. `desktop` is what OAIY's window gives its pages before they run.
 */
async function page({ hostname = 'flows.example.org', protocol = 'https:', port = '', desktop = null, tauri = false, shim = false, storage = {}, health = 'answers' } = {}) {
  const store = new Map(Object.entries(storage));
  const requests = [];
  const timers = new Map();
  let timerId = 0;
  for (const key of ['__OAIY_DESKTOP__', '__OAIY_WEB_SHIM__', '__TAURI_INTERNALS__']) delete globalThis[key];
  globalThis.window = globalThis;
  globalThis.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
  const origin = `${protocol}//${hostname}${port ? `:${port}` : ''}`;
  Object.defineProperty(globalThis, 'location', { value: { hostname, protocol, port, origin, href: `${origin}/app.html` }, configurable: true, writable: true });
  // Node's own setInterval is replaced by one that records: nothing ticks by itself.
  globalThis.setInterval = (fn, ms) => {
    timers.set(++timerId, { fn, ms });
    return timerId;
  };
  globalThis.clearInterval = (id) => timers.delete(id);
  if (desktop) globalThis.__OAIY_DESKTOP__ = desktop;
  // Tauri's global and the flow editor's shim marker, as the page has them when it loads: the window is read as the modules load.
  if (tauri) globalThis.__TAURI_INTERNALS__ = {};
  if (shim) globalThis.__OAIY_WEB_SHIM__ = true;
  const json = (body) => ({ ok: true, status: 200, headers: { get: () => 'application/json' }, json: async () => body, text: async () => JSON.stringify(body) });
  globalThis.fetch = async (url, init) => {
    const u = String(url);
    requests.push({ url: u, method: init?.method ?? 'GET' });
    if (u.endsWith('/api/health')) {
      if (health === 'nothing') throw new TypeError('Failed to fetch');
      return json({ status: 'ok', product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: '1.2.3' });
    }
    if (u.endsWith('/api/services')) return json({ services: [{ id: 'py-rig', name: 'Python rig', description: 'x', category: 'llm', status: 'running', port: 8123, defaultPort: 8123, installed: true }] });
    if (u.endsWith('/api/ai/engine/services')) return json({ services: [] });
    return { ok: false, status: 404, json: async () => ({}), text: async () => '' };
  };
  const modules = await import(`${pathToFileURL(bundlePath).href}?page=${++n}`);
  return {
    ...modules,
    store,
    requests,
    timers,
    health: () => requests.filter((r) => r.url.endsWith('/api/health')),
    /** Let the requests that were started finish (all answers are immediate). */
    settle: () => new Promise((resolve) => setTimeout(resolve, 20)),
  };
}

const STALE = JSON.stringify([{ id: 'companion:py-rig', name: 'Python rig (running)', endpoint: `${DEFAULT.replace('17972', '8123')}/v1/chat/completions` }]);

// ---------------------------------------------------------------------------
// A tab with no link
// ---------------------------------------------------------------------------
for (const [name, opts] of [
  ['a public host', { hostname: 'flows.example.org' }],
  ['a dev server', { hostname: 'localhost', protocol: 'http:' }],
  ['a local copy on a loopback address', { hostname: '127.0.0.1', protocol: 'http:' }],
  ['a host of the web app (flows.web.localhost)', { hostname: 'flows.web.localhost', protocol: 'http:' }],
]) {
  await check(`a tab on ${name} that has no link sends nothing when the editor opens, and nothing keeps asking`, async () => {
    const p = await page({ ...opts, storage: { [KEYS.services]: STALE } });
    p.detection.startDesktopDetection();
    p.services.startDesktopServiceSync();
    await p.settle();
    assert.deepEqual(p.requests, [], 'no request at all: not to 127.0.0.1, not to the LAN, not anywhere');
    assert.equal(p.timers.size, 0, 'no poll is running');
    assert.equal(p.detection.getDesktopInfo().checked, false, 'the desktop has not been asked, so it is neither there nor not there');
    assert.equal(p.detection.getDesktopInfo().available, false);
    assert.equal(p.store.get(KEYS.services), undefined, "the services a session with a desktop left behind are not this tab's");
    assert.equal(p.detection.mayLookOnLoad(), false);
  });
}

await check('a tab with no link and storage that throws still opens: nothing to clear, nothing sent', async () => {
  const p = await page();
  globalThis.localStorage = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); }, removeItem: () => { throw new Error('blocked'); } };
  assert.doesNotThrow(() => {
    p.detection.startDesktopDetection();
    p.services.startDesktopServiceSync();
  });
  await p.settle();
  assert.deepEqual(p.requests, []);
});

await check("a script that sets __OAIY_DESKTOP__ or Tauri's global after the page has loaded does not turn the tab into OAIY's window: the editor still looks at nothing (the review's F8)", async () => {
  const p = await page({ storage: { [KEYS.services]: STALE } });
  // What a flow's code, a model-written page or an injected script can do once the page is running.
  globalThis.__OAIY_DESKTOP__ = Object.freeze({ origin: 'http://192.168.1.1:17972', token: 'x', theme: 'dark' });
  globalThis.__TAURI_INTERNALS__ = {};
  assert.equal(p.detection.mayLookOnLoad(), false);
  p.detection.startDesktopDetection();
  p.services.startDesktopServiceSync();
  await p.settle();
  assert.deepEqual(p.requests, []);
  assert.equal(p.timers.size, 0);
});

await check('starting the detection twice is one poll, and asking on purpose (refreshDesktopStatus) is always allowed: it is the person asking', async () => {
  const p = await page();
  p.detection.startDesktopDetection();
  p.detection.startDesktopDetection();
  assert.equal(p.timers.size, 0);
  const info = await p.detection.refreshDesktopStatus();
  assert.equal(info.available, true);
  assert.equal(p.health().length, 1);
  assert.equal(p.timers.size, 0, 'and asking once starts no poll');
});

// ---------------------------------------------------------------------------
// Where the editor looks as it opens, as before
// ---------------------------------------------------------------------------
const given = { origin: 'http://127.0.0.1:18100', token: 'tok', theme: 'light' };

await check("OAIY's own window looks for the desktop as it opens, as it always has: two health requests, the poll, the service lists", async () => {
  const p = await page({ hostname: 'oaiyflows.localhost', protocol: 'http:', desktop: given });
  p.detection.startDesktopDetection();
  p.services.startDesktopServiceSync();
  await p.settle();
  assert.equal(p.health().length, 2, 'the detection probes at once, and so does the service sync');
  assert.ok(p.health().every((r) => r.url === `${DEFAULT}/api/health`), p.health().map((r) => r.url).join(', '));
  assert.ok(p.timers.size >= 1, 'the poll runs');
  assert.ok([...p.timers.values()].some((t) => t.ms === 10_000));
  const urls = p.requests.map((r) => r.url);
  assert.ok(urls.includes(`${given.origin}/api/services`) && urls.includes(`${given.origin}/api/ai/engine/services`), `and the desktop's service lists: ${urls.join(', ')}`);
  assert.equal(p.detection.getDesktopInfo().available, true);
  assert.equal(p.detection.getDesktopInfo().checked, true);
});

await check('OAIY\'s own window looks even with nothing saved and a stale list in storage: the list is kept until the first probe has answered', async () => {
  const p = await page({ hostname: 'oaiyflows.localhost', protocol: 'http:', desktop: given, storage: { [KEYS.services]: STALE } });
  p.detection.startDesktopDetection();
  p.services.startDesktopServiceSync();
  assert.equal(p.store.get(KEYS.services), STALE, 'not cleared as the page opens');
});

await check('a tab with a link (Connect was pressed and answered) looks as it opens, like the own window: two health requests and the poll', async () => {
  const p = await page({ storage: { [KEYS.linked]: '1' } });
  assert.equal(p.detection.mayLookOnLoad(), true);
  p.detection.startDesktopDetection();
  p.services.startDesktopServiceSync();
  await p.settle();
  assert.equal(p.health().length, 2);
  assert.ok(p.health().every((r) => r.url === `${DEFAULT}/api/health`));
  assert.ok(p.timers.size >= 1);
});

await check('a tab whose engine has an address of its own (Settings) looks there as it opens, and only there', async () => {
  const p = await page({ storage: { [KEYS.engine]: LAN } });
  assert.equal(p.link.hasSavedDesktopLink(), true);
  assert.equal(p.link.linkedByConnect(), false, 'an address is a link, but Disconnect has nothing to forget');
  p.detection.startDesktopDetection();
  p.services.startDesktopServiceSync();
  await p.settle();
  assert.equal(p.health().length, 2);
  assert.ok(p.requests.every((r) => r.url.startsWith(LAN)), p.requests.map((r) => r.url).join(', '));
});

await check('the desktop shell is one of OAIY\'s own places: it looks as it opens (a page of the Tauri shell has Tauri behind it)', async () => {
  const p = await page({ hostname: 'localhost', protocol: 'http:', tauri: true });
  p.detection.startDesktopDetection();
  await p.settle();
  assert.equal(p.health().length, 1);
});

await check("the editor's browser shim defines Tauri's global as well, and that does not make a tab into a window that looks", async () => {
  const p = await page({ tauri: true, shim: true });
  p.detection.startDesktopDetection();
  p.services.startDesktopServiceSync();
  await p.settle();
  assert.deepEqual(p.requests, []);
});

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------
await check('Connect makes exactly one health request; the desktop answers, the link is kept, the poll starts and the list is fetched once', async () => {
  const p = await page();
  const info = await p.connect.connectDesktop();
  await p.settle();
  assert.equal(info.available, true);
  assert.equal(p.health().length, 1, 'ONE health request');
  assert.equal(p.health()[0].url, `${DEFAULT}/api/health`);
  assert.equal(p.health()[0].method, 'GET');
  assert.equal(p.store.get(KEYS.linked), '1', 'the link is kept');
  assert.equal(p.requests.filter((r) => r.url === `${DEFAULT}/api/services`).length, 1, "and the desktop's services asked for once");
  assert.equal(p.requests.length, 2, `nothing else went out: ${p.requests.map((r) => r.url).join(', ')}`);
  assert.equal([...p.timers.values()].filter((t) => t.ms === 10_000).length, 2, 'the health poll and the service poll, one each');
  assert.equal(JSON.parse(p.store.get(KEYS.services)).length, 1, 'the palette has the desktop service');
  assert.equal(p.detection.getDesktopInfo().checked, true);
});

await check('the next time the editor opens with that link, it looks as it opens', async () => {
  const first = await page();
  await first.connect.connectDesktop();
  await first.settle();
  const again = await page({ storage: Object.fromEntries(first.store) });
  assert.equal(again.detection.mayLookOnLoad(), true);
  again.detection.startDesktopDetection();
  again.services.startDesktopServiceSync();
  await again.settle();
  assert.equal(again.health().length, 2);
});

await check('Connect with nothing there makes one request and keeps nothing: no link, no poll, no list', async () => {
  const p = await page({ health: 'nothing', storage: { [KEYS.services]: STALE } });
  const info = await p.connect.connectDesktop();
  await p.settle();
  assert.equal(info.available, false);
  assert.equal(info.checked, true, 'it was asked, and nothing answered');
  assert.equal(p.requests.length, 1);
  assert.equal(p.store.get(KEYS.linked), undefined);
  assert.equal(p.timers.size, 0);
  assert.equal(p.link.hasSavedDesktopLink(), false);
});

await check('Connect pressed again asks once more each time, and starts no second poll', async () => {
  const p = await page();
  await p.connect.connectDesktop();
  await p.connect.connectDesktop();
  await p.settle();
  assert.equal(p.health().length, 2, 'one request per press');
  assert.equal([...p.timers.values()].filter((t) => t.ms === 10_000).length, 2, 'the same two polls');
  assert.equal(p.requests.filter((r) => r.url === `${DEFAULT}/api/services`).length, 1, 'and the list is not asked twice');
});

await check('Connect asks the address the engine has now, and a change of address while a tab has no link is asked once (the person pressed Save)', async () => {
  const p = await page();
  p.endpoint.setEngineBase('192.168.1.50:17972');
  await p.settle();
  assert.equal(p.health().length, 1);
  assert.equal(p.health()[0].url, `${LAN}/api/health`);
  assert.equal(p.timers.size, 0, 'the address change starts no poll by itself (the settings card does)');
  p.requests.length = 0;
  await p.connect.connectDesktop();
  assert.equal(p.health().length, 1);
  assert.equal(p.health()[0].url, `${LAN}/api/health`);
});

// ---------------------------------------------------------------------------
// Disconnect
// ---------------------------------------------------------------------------
await check('Disconnect forgets the link, stops the polls, empties the palette of the desktop, and the next opening sends nothing', async () => {
  const p = await page();
  await p.connect.connectDesktop();
  await p.settle();
  assert.ok(p.timers.size > 0);
  p.connect.disconnectDesktop();
  assert.equal(p.store.get(KEYS.linked), undefined);
  assert.equal(p.timers.size, 0);
  assert.equal(p.store.get(KEYS.services), undefined, "the desktop's services are gone from the palette");
  assert.equal(p.detection.getDesktopInfo().checked, false);
  assert.equal(p.detection.getDesktopInfo().available, false);
  const after = await page({ storage: Object.fromEntries(p.store) });
  after.detection.startDesktopDetection();
  after.services.startDesktopServiceSync();
  await after.settle();
  assert.deepEqual(after.requests, []);
});

await check('Reset in Settings gives the default address back: with no Connect link kept, the tab has no link, and what the address started stops (the poll, the services, and no request after)', async () => {
  const p = await page();
  p.connect.keepToTheLink();
  // The person saves an address: the settings card asks once (the address change) and starts keeping up with the desktop there.
  p.endpoint.setEngineBase('192.168.1.50:17972');
  p.detection.startDesktopDetection({ probeNow: false });
  p.services.startDesktopServiceSync({ probeNow: false });
  await p.settle();
  assert.equal([...p.timers.values()].filter((t) => t.ms === 10_000).length, 2, 'the health poll and the service poll');
  assert.equal(JSON.parse(p.store.get(KEYS.services)).length, 1, "the desktop's service is in the palette's list");
  assert.ok(p.requests.every((r) => r.url.startsWith(LAN)), p.requests.map((r) => r.url).join(', '));
  const before = p.requests.length;
  p.endpoint.setEngineBase(null);
  await p.settle();
  assert.equal(p.timers.size, 0, 'nothing polls any more');
  assert.equal(p.store.get(KEYS.services), undefined, "the desktop's services leave the palette");
  assert.equal(p.detection.getDesktopInfo().checked, false, 'and the desktop is neither there nor not there again');
  assert.equal(p.requests.length, before, 'and not one request went out after the Reset, to the default address or any other');
  assert.equal(p.link.hasSavedDesktopLink(), false);
});

await check('Reset keeps everything where the tab is still linked by Connect, and in OAIY\'s own window, which needs no link', async () => {
  const linked = await page();
  linked.connect.keepToTheLink();
  await linked.connect.connectDesktop();
  linked.endpoint.setEngineBase('192.168.1.50:17972');
  await linked.settle();
  linked.endpoint.setEngineBase(null);
  await linked.settle();
  assert.equal(linked.link.linkedByConnect(), true);
  assert.equal([...linked.timers.values()].filter((t) => t.ms === 10_000).length, 2, 'still polling');
  assert.ok(linked.store.get(KEYS.services), 'and the list kept');

  const window = await page({ hostname: 'oaiyflows.localhost', protocol: 'http:', desktop: given });
  window.connect.keepToTheLink();
  window.detection.startDesktopDetection();
  window.services.startDesktopServiceSync();
  await window.settle();
  const polling = window.timers.size;
  assert.ok(polling >= 1);
  window.endpoint.setEngineBase('192.168.1.50:17972');
  window.endpoint.setEngineBase(null);
  await window.settle();
  assert.equal(window.timers.size, polling, 'OAIY\'s own window never stops looking');
});

// ---------------------------------------------------------------------------
// Disconnect while the desktop is being asked (the review's F7)
// ---------------------------------------------------------------------------
/** Hold the answers to health requests until `release()`; the rest of the fake desktop answers as it did. */
function holdHealth() {
  const real = globalThis.fetch;
  const waiting = [];
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/health')) await new Promise((resolve) => waiting.push(resolve));
    return real(url, init);
  };
  return { release: () => waiting.splice(0).forEach((resolve) => resolve()), restore: () => { globalThis.fetch = real; } };
}

await check('a probe that was on its way when Disconnect was pressed does not bring the desktop back: its answer is about a link that is gone', async () => {
  const p = await page();
  await p.connect.connectDesktop();
  await p.settle();
  const hold = holdHealth();
  try {
    const late = p.detection.refreshDesktopStatus();
    p.connect.disconnectDesktop();
    assert.equal(p.detection.getDesktopInfo().available, false);
    hold.release();
    await late;
    await p.settle();
    assert.equal(p.detection.getDesktopInfo().available, false, 'the late answer did not say the desktop is there');
    assert.equal(p.detection.getDesktopInfo().checked, false, 'nor that it was asked');
    assert.equal(p.store.get(KEYS.linked), undefined);
    assert.equal(p.timers.size, 0);
    assert.equal(p.store.get(KEYS.services), undefined, 'and no service list came with it');
  } finally {
    hold.restore();
  }
});

await check('a Connect that was on its way when Disconnect was pressed keeps nothing: no link, no poll, no list', async () => {
  const p = await page();
  const hold = holdHealth();
  try {
    const pending = p.connect.connectDesktop();
    p.connect.disconnectDesktop();
    hold.release();
    const info = await pending;
    await p.settle();
    assert.equal(info.available, false);
    assert.equal(p.store.get(KEYS.linked), undefined, 'the answer was not kept as a link');
    assert.equal(p.timers.size, 0);
    assert.equal(p.store.get(KEYS.services), undefined);
    assert.equal(p.health().length, 1);
    assert.equal(p.detection.getDesktopInfo().checked, false);
  } finally {
    hold.restore();
  }
});

// ---------------------------------------------------------------------------
// What the button and its card say, for each state (the review's F2 and F12)
// ---------------------------------------------------------------------------
await check('the words of each state: never connected sends nothing until pressed; linked (by Connect, or an address) asks each time the editor opens and every ten seconds; disconnected is the first again', async () => {
  const p = await page();
  const W = p.words;
  const base = DEFAULT;
  const never = W.connectState({ linkedByConnect: false, addressGiven: false, base, available: false });
  const connected = W.connectState({ linkedByConnect: true, addressGiven: false, base, available: true });
  const offline = W.connectState({ linkedByConnect: true, addressGiven: false, base, available: false });
  const address = W.connectState({ linkedByConnect: false, addressGiven: true, base: LAN, available: false });
  const addressUp = W.connectState({ linkedByConnect: false, addressGiven: true, base: LAN, available: true });
  assert.deepEqual([never.kind, connected.kind, offline.kind, address.kind, addressUp.kind], ['never', 'connected', 'connected', 'address', 'address']);

  const neverWords = W.connectWords(never);
  assert.match(neverWords, /^Connect sends one request to http:\/\/127\.0\.0\.1:17972 \(GET \/api\/health\) to ask whether OAIY Desktop is running\. Until you press it this page sends nothing to your computer or your network\./);
  assert.match(neverWords, /may ask whether this site may connect to devices on your network/);
  assert.match(neverWords, /keeps the link: from then on the editor asks it again each time it opens and every 10 seconds while it is open, and reads its services\. Disconnect ends that\./, 'what one request leads to is said, not hidden');
  // What is refused until then, and what the check cannot see (F4): the address as written, not where a redirect or a name leads.
  assert.match(neverWords, /Until you connect, the editor also loads no picture, video or sound from an address on your computer or your network, judging by the address as written: a link that redirects there, or a name that points there, is left to your browser's own question\./);

  for (const [name, state] of [['connected', connected], ['offline', offline], ['address', address], ['address that answers', addressUp]]) {
    const words = W.connectWords(state);
    assert.match(words, /each time it opens and every 10 seconds while it is open/, name);
    assert.doesNotMatch(words, /sends nothing|Until you press it|loads no picture/, `${name}: what is true of a tab that has never connected is not said of this one`);
    assert.ok(words.includes(state.base), `${name}: it names where it asks`);
  }
  assert.match(W.connectWords(connected), /^Connected to http:\/\/127\.0\.0\.1:17972\. This browser keeps the link: the editor asks OAIY Desktop whether it is running, and reads its services, each time it opens and every 10 seconds while it is open\. Disconnect ends that\.$/);
  assert.match(W.connectWords(offline), /nothing answers now\. Connect asks again now; Disconnect forgets the link and ends that\.$/);
  assert.match(W.connectWords(address), /^The engine's address in Settings is http:\/\/192\.168\.1\.50:17972, so the editor asks there .*; nothing answers now\. Reset above gives the default address back/);
  assert.doesNotMatch(W.connectWords(addressUp), /nothing answers/);
  assert.equal(new Set([neverWords, W.connectWords(connected), W.connectWords(offline), W.connectWords(address)]).size, 4, 'each says something the others do not');
  assert.match(W.connectLead(never), /Already running OAIY Desktop\?/);
  assert.match(W.connectLead(connected), /^Linked to OAIY Desktop\. $/);
  assert.match(W.connectLead(offline), /does not answer now/);
});

await check('the state follows what the tab has: Connect made a link (it wins over an address), an address alone is a link, and neither is never', async () => {
  const p = await page();
  const S = (linkedByConnect, addressGiven) => p.words.connectState({ linkedByConnect, addressGiven, base: DEFAULT, available: false }).kind;
  assert.equal(S(false, false), 'never');
  assert.equal(S(true, false), 'connected');
  assert.equal(S(true, true), 'connected');
  assert.equal(S(false, true), 'address');
  assert.equal(p.words.POLL_SECONDS, 10, 'and the seconds it says are the seconds the poll waits');
  await p.connect.connectDesktop();
  assert.deepEqual([...p.timers.values()].map((t) => t.ms / 1000).filter((s) => s === p.words.POLL_SECONDS).length, 2);
});

await check('listeners are told when the link is made or forgotten, so what depends on it is drawn again', async () => {
  const p = await page();
  const told = [];
  const unsubscribe = p.link.subscribeDesktopLink(() => told.push(p.link.hasSavedDesktopLink()));
  await p.connect.connectDesktop();
  p.connect.disconnectDesktop();
  p.endpoint.setEngineBase('192.168.1.50:17972');
  p.endpoint.setEngineBase(null);
  unsubscribe();
  p.endpoint.setEngineBase('192.168.1.50:17972');
  assert.deepEqual(told, [true, false, true, false]);
});

fs.rmSync(bundlePath, { force: true });
fs.rmSync(stub, { force: true });
finish();
