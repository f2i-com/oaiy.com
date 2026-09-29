/**
 * Every place the editor reaches OAIY's engine follows the address in Settings.
 *
 *     npm run test:engine-endpoint
 *
 * The address is a setting (lib/engineEndpoint.ts; `http://127.0.0.1:17972`
 * until it is changed) so the engine can run on a desktop and the editor on a
 * phone on the same network. The health probe followed it; the calls that start
 * a service, drive the browser service, run an engine model and list the
 * desktop's services each had the loopback address written in, so on a phone
 * they went to the phone itself. Here each one is called with the setting at its
 * default, then changed WITHOUT reloading anything (the modules are loaded once),
 * then reset:
 *
 *   - the shim's `ensure_service_ready_by_port` (what a flow run in the browser
 *     calls: the module broker hands it the shim's invoke);
 *   - core-ai's own `ensure-by-port` call, for a run with no broker;
 *   - core-browser's Playwright service lookup and start;
 *   - the engine contract calls (`desktopAccess`, `absoluteUrl`);
 *   - the desktop service list the palette polls (`refreshDesktopServices`);
 *   - the health probe and the `baseUrl` the dock and the settings card show
 *     (which stayed at the old address until availability changed).
 * The desktop's own window still wins where it says where the desktop is
 * (`__OAIY_DESKTOP__`, `OAIY_SERVER_URL`); the setting is what is used after
 * them. No source file may write the address into a request again.
 *
 * The modules are TypeScript with aliases only the bundler resolves, so they
 * are bundled for Node with esbuild (as flow-services.mjs does); the shim's
 * ffmpeg (Vite-only imports) is left out, and oaiy-ui-components is stubbed.
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

// A plain browser tab: storage, a window with no OAIY around it, and a fetch
// that records what it is asked and answers as a desktop would.
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
globalThis.window = {
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  addEventListener() {},
  location: { protocol: 'https:', origin: 'https://oaiy.com' },
};

const requests = [];
const json = (body, status = 200) => ({
  ok: status < 400,
  status,
  headers: { get: () => 'application/json' },
  json: async () => body,
  text: async () => JSON.stringify(body),
});
/** What a desktop (and its Playwright service) answers, whatever address it is asked at. */
globalThis.fetch = async (url, init) => {
  const u = String(url);
  requests.push({ url: u, method: init?.method ?? 'GET' });
  if (u.endsWith('/api/health')) return json({ status: 'ok', product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: '1.2.3' });
  if (u.endsWith('/api/services/ensure-by-port')) return json({ found: true, started: true, alreadyRunning: false, name: 'svc' });
  if (u.endsWith('/api/services')) return json({ services: [{ id: 'playwright-browser', port: 17880, status: 'running' }] });
  if (u.endsWith('/api/ai/engine/services')) return json({ services: [] });
  if (u.endsWith('/health')) return json({ ok: true });
  if (u.endsWith('/session')) return json({ sessionId: 's-1' });
  return json({}, 404);
};
const asked = () => requests.splice(0).map((r) => r.url);

const stub = path.join(os.tmpdir(), `oaiy-ui-stub-${process.pid}.mjs`);
fs.writeFileSync(stub, 'export function invalidateDynamicOptions() {}\nexport function subscribeToDynamicOptionsInvalidation() { return () => {}; }\n');
const aliases = {
  name: 'oaiy-aliases',
  setup(build) {
    build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: stub }));
    build.onResolve({ filter: /^oaiy-core\/modules\// }, (a) => ({ path: path.join(UI, 'src/bundled-modules', `${a.path.slice('oaiy-core/modules/'.length)}.ts`) }));
    build.onResolve({ filter: /^oaiy-core$/ }, () => ({ path: path.join(UI, 'vendor/oaiy-core/src/index.ts') }));
    build.onResolve({ filter: /^oaiy-core\// }, (a) => {
      const rest = a.path.slice('oaiy-core/'.length);
      const p = path.join(UI, 'vendor/oaiy-core', rest);
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
    // Only loaded when a flow runs ffmpeg, which none of this does; its imports are Vite's.
    build.onResolve({ filter: /^\.\/ffmpeg$/ }, () => ({ path: './ffmpeg', external: true }));
  },
};
const bundlePath = path.join(os.tmpdir(), `oaiy-engine-endpoint-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { setEngineBase, getEngineBase, DEFAULT_ENGINE_BASE } from './src/lib/engineEndpoint.ts';
export { invoke } from './src/tauri-shim/core.ts';
export { desktopAccess, absoluteUrl } from './src/bundled-modules/core-service/contractRuntime.ts';
export { default as AIRuntime } from './src/bundled-modules/core-ai/runtime.ts';
export { default as BrowserRuntime } from './src/bundled-modules/core-browser/runtime.ts';
export { refreshDesktopServices } from './src/lib/desktopServices.ts';
import * as detection from './src/lib/desktopDetection.ts';
export { detection };`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
  plugins: [aliases],
});
const M = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });
fs.rmSync(stub, { force: true });

const DEFAULT = 'http://127.0.0.1:17972';
const LAN = 'http://192.168.1.50:17972';

// ---------------------------------------------------------------------------
// The reaches: each does its work at whatever the setting is, and returns the
// address(es) it asked for the desktop's own API.
// ---------------------------------------------------------------------------
const ctxFor = (over = {}) => ({
  log() {},
  onNodeStatus() {},
  abortSignal: undefined,
  secureFetch: async () => ({ status: 200, ok: true, text: async () => JSON.stringify({ choices: [{ message: { content: 'hello' } }] }) }),
  ...over,
});

/** The API calls (not the service's own) a request list made. */
const api = (urls, base) => urls.filter((u) => u.startsWith(`${base}/api/`));

const reaches = {
  'the shim starts a service by port (ensure_service_ready_by_port)': async (base) => {
    const r = await M.invoke('ensure_service_ready_by_port', { port: 8123 });
    assert.deepEqual(r, { success: true, already_running: false });
    return api(asked(), base);
  },
  "core-ai starts the endpoint's service itself when a run has no broker": async (base) => {
    const methods = M.AIRuntime.createMethods(ctxFor());
    const out = await methods.chat('', 'hi', null, 'http://127.0.0.1:8123/v1/chat/completions', 'm', '', false, 16, 0, 'text', false, 'auto', 'n1');
    assert.equal(out, 'hello');
    return api(asked(), base);
  },
  "core-browser finds and starts the desktop's Playwright service": async (base) => {
    const methods = M.BrowserRuntime.createMethods(ctxFor({ tauri: { invoke: async () => null } }));
    const session = await methods.createSessionV2({ mode: 'chromium' });
    assert.equal(session.id, 's-1');
    return api(asked(), base);
  },
  'a contract call to the engine is made absolute on the desktop': async (base) => {
    assert.equal(M.desktopAccess().origin, base);
    assert.equal(M.absoluteUrl('/api/ai/engine/gateway/v1/images/generations'), `${base}/api/ai/engine/gateway/v1/images/generations`);
    return [`${M.desktopAccess().origin}/api/x`];
  },
  'the palette polls the desktop for its services': async (base) => {
    await M.refreshDesktopServices();
    return api(asked(), base);
  },
  'the health probe asks the desktop whether it is there': async (base) => {
    const info = await M.detection.refreshDesktopStatus();
    assert.equal(info.available, true);
    return api(asked(), base);
  },
};

async function withSetting(value, fn) {
  M.setEngineBase(value);
  asked(); // what the change itself set going (a re-probe) is not what is measured
  try {
    return await fn();
  } finally {
    M.setEngineBase(null);
    asked();
  }
}

await check('the setting starts at the loopback address, and every reach asks it', async () => {
  assert.equal(M.DEFAULT_ENGINE_BASE, DEFAULT);
  assert.equal(M.getEngineBase(), DEFAULT);
  for (const [name, reach] of Object.entries(reaches)) {
    const urls = await reach(DEFAULT);
    assert.ok(urls.length > 0, `${name}: asked the desktop at ${DEFAULT}`);
    assert.ok(urls.every((u) => u.startsWith(`${DEFAULT}/api/`)), `${name}: ${urls.join(', ')}`);
  }
});

for (const [name, reach] of Object.entries(reaches)) {
  await check(`${name} follows the setting, and comes back to the default`, async () => {
    // Loaded and used at the default first, so a value read at load would be caught.
    await reach(DEFAULT);
    await withSetting('192.168.1.50:17972', async () => {
      assert.equal(M.getEngineBase(), LAN);
      const urls = await reach(LAN);
      assert.ok(urls.length > 0, `asked the desktop at ${LAN}`);
      assert.ok(urls.every((u) => u.startsWith(`${LAN}/api/`)), urls.join(', '));
      assert.ok(!urls.some((u) => u.includes('127.0.0.1:17972')), 'and never the loopback address');
    });
    const back = await reach(DEFAULT);
    assert.ok(back.length > 0 && back.every((u) => u.startsWith(`${DEFAULT}/api/`)), back.join(', '));
  });
}

await check('a Playwright service is looked up on the desktop the setting names (the service itself stays on its own port)', async () => {
  await withSetting('192.168.1.50:17972', async () => {
    const methods = M.BrowserRuntime.createMethods(ctxFor({ tauri: { invoke: async () => null } }));
    await methods.createSessionV2({ mode: 'chromium' });
    const urls = asked();
    assert.equal(urls[0], `${LAN}/api/services`);
    assert.equal(urls[1], `${LAN}/api/services/ensure-by-port`);
  });
});

await check("OAIY's own window still says where the desktop is: its origin wins over the setting", async () => {
  const injected = 'http://127.0.0.1:18100';
  globalThis.__OAIY_DESKTOP__ = { origin: injected, token: 'tok' };
  window.__OAIY_DESKTOP__ = globalThis.__OAIY_DESKTOP__;
  try {
    await withSetting('192.168.1.50:17972', async () => {
      assert.equal(M.desktopAccess().origin, injected);
      assert.equal(M.desktopAccess().token, 'tok');
      await M.refreshDesktopServices();
      const urls = asked();
      assert.ok(urls.some((u) => u === `${injected}/api/services`), urls.join(', '));
      assert.ok(urls.some((u) => u === `${injected}/api/ai/engine/services`), 'and the engine list, as in OAIY');
      assert.ok(!urls.some((u) => u.startsWith(LAN)), 'nothing went to the setting');
    });
  } finally {
    delete globalThis.__OAIY_DESKTOP__;
    delete window.__OAIY_DESKTOP__;
  }
});

await check("a flow the desktop runs (OAIY_SERVER_URL) still reaches that desktop, not the setting", async () => {
  const before = { url: process.env.OAIY_SERVER_URL, token: process.env.OAIY_SERVER_TOKEN };
  process.env.OAIY_SERVER_URL = 'http://127.0.0.1:18200/';
  process.env.OAIY_SERVER_TOKEN = 'server-token';
  try {
    await withSetting('192.168.1.50:17972', async () => {
      assert.deepEqual(M.desktopAccess(), { origin: 'http://127.0.0.1:18200', token: 'server-token' });
    });
  } finally {
    for (const [k, v] of [['OAIY_SERVER_URL', before.url], ['OAIY_SERVER_TOKEN', before.token]]) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
  assert.deepEqual(M.desktopAccess(), { origin: DEFAULT, token: null });
});

await check('what the dock and the settings card show follows the setting even when the desktop answers the same', async () => {
  const seen = [];
  const unsubscribe = M.detection.subscribeDesktopStatus((info) => seen.push(info.baseUrl));
  await M.detection.refreshDesktopStatus();
  assert.equal(M.detection.getDesktopInfo().baseUrl, DEFAULT);
  seen.length = 0;
  // A desktop at the new address answers just as this one did (available, same version).
  M.setEngineBase('192.168.1.50:17972');
  await M.detection.refreshDesktopStatus();
  assert.equal(M.detection.getDesktopInfo().available, true);
  assert.equal(M.detection.getDesktopInfo().baseUrl, LAN, 'not left at the old address');
  assert.ok(seen.includes(LAN), 'and whoever shows it was told');
  M.setEngineBase(null);
  await M.detection.refreshDesktopStatus();
  assert.equal(M.detection.getDesktopInfo().baseUrl, DEFAULT);
  unsubscribe();
  asked();
  // The load-time snapshot is gone: nothing can read a value that does not follow the setting.
  assert.equal('DESKTOP_API_BASE' in M.detection, false);
});

await check('a probe that answers after the address was changed is about the old address, and is dropped', async () => {
  const realFetch = globalThis.fetch;
  try {
    for (const outcome of ['answers', 'fails']) {
      await M.detection.refreshDesktopStatus();
      assert.equal(M.detection.getDesktopInfo().baseUrl, DEFAULT);
      // The desktop at the old address is slow to reply.
      const release = [];
      globalThis.fetch = async (url, init) => {
        if (String(url) === `${DEFAULT}/api/health`) {
          await new Promise((resolve) => release.push(resolve));
          if (outcome === 'fails') throw new TypeError('Failed to fetch');
        }
        return realFetch(url, init);
      };
      const late = M.detection.refreshDesktopStatus();
      // The person points the editor at another desktop, which answers at once.
      M.setEngineBase('192.168.1.50:17972');
      await M.detection.refreshDesktopStatus();
      assert.equal(M.detection.getDesktopInfo().baseUrl, LAN);
      assert.equal(M.detection.getDesktopInfo().available, true);
      // Now the old one answers (or does not).
      release.forEach((resolve) => resolve());
      await late;
      assert.equal(M.detection.getDesktopInfo().baseUrl, LAN, `the ${outcome} probe of the old address did not overwrite it`);
      assert.equal(M.detection.getDesktopInfo().available, true, `and did not say the new one is ${outcome === 'fails' ? 'not there' : 'the old one'}`);
      globalThis.fetch = realFetch;
      M.setEngineBase(null);
      await M.detection.refreshDesktopStatus();
    }
  } finally {
    globalThis.fetch = realFetch;
    M.setEngineBase(null);
    asked();
  }
});

// ---------------------------------------------------------------------------
// No request writes the address in again.
// ---------------------------------------------------------------------------
await check('no source writes the loopback engine address into a request', () => {
  // Where the address is named on purpose: its definition, the CLI's default for a connector host,
  // the marketing pages that print it, and the settings card's example of a LAN address.
  const allowed = new Set([
    'lib/engineEndpoint.ts',
    'connector-module/types.ts',
    'landing/DesktopPage.tsx',
    'landing/LandingPage.tsx',
    'components/panels/settings/EngineEndpointCard.tsx',
  ]);
  const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]));
  const sources = [...walk(path.join(UI, 'src')), ...walk(path.join(UI, 'vendor', 'oaiy-core'))].filter((f) => /\.(tsx?|json)$/.test(f));
  const hits = [];
  for (const file of sources) {
    const rel = path.relative(path.join(UI, 'src'), file).replace(/\\/g, '/');
    if (allowed.has(rel)) continue;
    fs.readFileSync(file, 'utf8').split('\n').forEach((line, i) => {
      if (/17972/.test(line) && !/^\s*(\/\/|\*|\/\*)/.test(line)) hits.push(`${rel}:${i + 1}: ${line.trim().slice(0, 100)}`);
    });
  }
  assert.deepEqual(hits, [], 'go through getEngineBase() (lib/engineEndpoint.ts)');
});

await check("the dock shows the address the setting names, not the one read when the page loaded", () => {
  const app = fs.readFileSync(path.join(UI, 'src/components/OAIYApp.tsx'), 'utf8');
  assert.doesNotMatch(app, /DESKTOP_API_BASE/);
  assert.match(app, /endpointUrl=\{backend\.share \? backend\.share\.editUrl : companion\.baseUrl\}/);
  assert.match(app, /const url = backend\.share \? backend\.share\.editUrl : companion\.baseUrl;/);
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
