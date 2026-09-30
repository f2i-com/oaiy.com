/**
 * What the flow editor says when OAIY Desktop's Playwright service does not answer it.
 *
 *     npm run test:browser-service
 *
 * The service drives a real browser, so it answers only OAIY Desktop's own windows (the Agent and
 * the Flows page) and programs that send no Origin; any other web page gets no answer. A refused
 * request comes back without CORS headers, so the page cannot read the 403: all it sees is
 * `TypeError: Failed to fetch`, which is also what a service that is still starting looks like.
 * core-browser's runtime tells them apart by where the page is and how long the service has been up:
 *
 *   - outside OAIY's window (a hosted or paired editor at oaiy.com, a browser tab), a network error
 *     from a service that has been running for a while is a refusal: an error that says so, at once;
 *   - a service that has only just started is given time to launch its browser, as before;
 *   - inside OAIY's window nothing changes: it waits for the browser, and a service that never
 *     answers ends as "did not become ready", not as a refusal;
 *   - an answer that is an HTTP error (not a network error) is never taken for a refusal.
 *
 * The runtime is TypeScript with aliases only the bundler resolves, so it is bundled for Node with
 * esbuild (as engine-endpoint.mjs does), and the desktop's answers are a stub `fetch`.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

let pass = 0;
const failures = [];
async function check(name, fn) {
  try {
    await fn();
    pass++;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failures.push(`${name} — ${e.message}`);
    console.log(`  FAIL ${name} — ${e.stack || e.message}`);
  }
}

const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
/** The page the editor runs in: the hosted editor by default. */
function pageAt(origin, { inOaiy = false } = {}) {
  globalThis.window = {
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    addEventListener() {},
    location: { protocol: origin.split(':')[0] + ':', origin },
    __OAIY_WEB_SHIM__: true,
    ...(inOaiy ? { __OAIY_DESKTOP__: { origin: 'http://127.0.0.1:17972', token: 'tok' } } : {}),
  };
}

const json = (body, status = 200) => ({
  ok: status < 400,
  status,
  headers: { get: () => 'application/json' },
  json: async () => body,
  text: async () => JSON.stringify(body),
});
const HOUR = 3_600_000;
const iso = (msAgo) => new Date(Date.now() - msAgo).toISOString();

/**
 * A desktop and its Playwright service. `health` decides each answer of the service's /health
 * (call number from 1): 'ok', 'network' (a TypeError, what a page sees of a refusal or of a port
 * nobody listens on yet) or 'http503'.
 */
function desktop({ status = 'running', startedAt = iso(HOUR), health }) {
  const seen = { health: 0, ensured: 0 };
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.endsWith('/api/services/ensure-by-port')) {
      seen.ensured++;
      return json({ found: true, started: true, alreadyRunning: false, name: 'svc' });
    }
    if (u.endsWith('/api/services')) return json({ services: [{ id: 'playwright-browser', port: 17880, status, startedAt }] });
    if (u.endsWith(':17880/health')) {
      seen.health++;
      const outcome = health(seen.health);
      if (outcome === 'network') throw new TypeError('Failed to fetch');
      if (outcome === 'http503') return json({ error: 'browser not ready' }, 503);
      return json({ ok: true });
    }
    if (u.endsWith(':17880/session')) return json({ sessionId: 's-1' });
    return json({}, 404);
  };
  return seen;
}

// Bundle the runtime for Node.
const aliases = {
  name: 'oaiy-aliases',
  setup(build) {
    build.onResolve({ filter: /^oaiy-core\// }, (a) => {
      const p = path.join(UI, 'vendor/oaiy-core', a.path.slice('oaiy-core/'.length));
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
  },
};
const bundlePath = path.join(os.tmpdir(), `oaiy-browser-service-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { default as BrowserRuntime } from './src/bundled-modules/core-browser/runtime.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  plugins: [aliases],
});
pageAt('https://oaiy.com');
const { BrowserRuntime } = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

const methodsFor = () =>
  BrowserRuntime.createMethods({ log() {}, onNodeStatus() {}, abortSignal: undefined, secureFetch: async () => ({ status: 200, ok: true, text: async () => '{}' }), tauri: { invoke: async () => null } });
const open = () => methodsFor().createSessionV2({ mode: 'chromium' });

/** Run `fn` with the runtime's 750 ms pauses (and every other timer) taken out, so a 60-attempt wait takes no time. */
async function withoutWaiting(fn) {
  const real = globalThis.setTimeout;
  globalThis.setTimeout = (f, _ms, ...rest) => real(f, 0, ...rest);
  try {
    return await fn();
  } finally {
    globalThis.setTimeout = real;
  }
}

const REFUSAL = /did not answer this page at https:\/\/oaiy\.com.*Flows page/s;

// ---------------------------------------------------------------------------
// A hosted editor, outside OAIY's window
// ---------------------------------------------------------------------------
await check('a hosted editor is told at once that the service only answers OAIY Desktop\'s windows, when a service that has been up for an hour does not answer it', async () => {
  pageAt('https://oaiy.com');
  const seen = desktop({ health: () => 'network' });
  await assert.rejects(open(), (e) => REFUSAL.test(e.message) && /OAIY Desktop's own windows/.test(e.message));
  assert.equal(seen.health, 1, 'one attempt, not the whole wait');
});

await check('the message names the page it is about, wherever the editor is', async () => {
  pageAt('http://localhost:5173');
  desktop({ health: () => 'network' });
  await assert.rejects(open(), /did not answer this page at http:\/\/localhost:5173/);
});

await check('a hosted editor waits for a service that has only just started, and is served when it answers', async () => {
  pageAt('https://oaiy.com');
  const seen = desktop({ status: 'running', startedAt: iso(2_000), health: (n) => (n < 3 ? 'network' : 'ok') });
  const session = await open();
  assert.equal(session.id, 's-1');
  assert.equal(seen.health, 3, 'it kept asking while the browser launched');
});

await check('a service that was not running is started, and given the time to launch from then', async () => {
  pageAt('https://oaiy.com');
  // Its last start was an hour ago, but it was stopped: that says nothing about how long this start has been.
  const seen = desktop({ status: 'stopped', startedAt: iso(HOUR), health: (n) => (n < 2 ? 'network' : 'ok') });
  const session = await open();
  assert.equal(session.id, 's-1');
  assert.equal(seen.ensured, 1);
  assert.equal(seen.health, 2);
});

await check('a hosted editor is served by a desktop whose service does not check the caller (an older desktop)', async () => {
  pageAt('https://oaiy.com');
  const seen = desktop({ health: () => 'ok' });
  assert.equal((await open()).id, 's-1');
  assert.equal(seen.health, 1);
});

await check('a service that never answers a page outside the window ends as the refusal after the wait, not as "did not become ready"', async () => {
  pageAt('https://oaiy.com');
  const seen = desktop({ startedAt: iso(2_000), health: () => 'network' });
  await withoutWaiting(() => assert.rejects(open(), REFUSAL));
  assert.equal(seen.health, 60, 'the same wait as before');
});

await check('an HTTP error from the service is not a refusal: the wait goes on and the answer is shown', async () => {
  pageAt('https://oaiy.com');
  const seen = desktop({ health: (n) => (n < 3 ? 'http503' : 'ok') });
  assert.equal((await open()).id, 's-1');
  assert.equal(seen.health, 3);
  desktop({ health: () => 'http503' });
  await withoutWaiting(() => assert.rejects(open(), /did not become ready: browser not ready/));
});

// ---------------------------------------------------------------------------
// OAIY's own window: exactly as before
// ---------------------------------------------------------------------------
for (const origin of ['http://oaiyflows.localhost', 'oaiyflows://localhost']) {
  await check(`in OAIY's window (${origin}) the wait for the browser is as it was, however long the service has been up`, async () => {
    pageAt(origin, { inOaiy: true });
    const seen = desktop({ health: (n) => (n < 4 ? 'network' : 'ok') });
    assert.equal((await open()).id, 's-1');
    assert.equal(seen.health, 4, 'it did not give up on the first network error');
  });

  await check(`in OAIY's window (${origin}) a service that never answers ends as "did not become ready", not as a refusal`, async () => {
    pageAt(origin, { inOaiy: true });
    const seen = desktop({ health: () => 'network' });
    await withoutWaiting(() => assert.rejects(open(), (e) => /did not become ready: Failed to fetch/.test(e.message) && !/did not answer this page/.test(e.message)));
    assert.equal(seen.health, 60);
  });
}

await check("the desktop that is not running at all is still the message about starting OAIY Desktop", async () => {
  pageAt('https://oaiy.com');
  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch');
  };
  await assert.rejects(open(), /the OAIY Desktop is not running/);
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
