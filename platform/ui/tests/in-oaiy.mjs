/**
 * The editor inside OAIY's window.
 *
 *     npm run test:in-oaiy
 *
 * - Its flows are OAIY Desktop's (`useDesktopFlows`): what goes to the desktop,
 *   what counts as a change, how a desktop flow becomes the editor's, and which
 *   side's flow wins when one or both changed (`reconcile`).
 * - Its theme is OAIY's (`ThemeContext`'s `desktopTheme`): what the desktop said
 *   while the page was loading, then what it said before a reload, then what it
 *   said at the start.
 *
 * The modules are TypeScript with imports only a bundler resolves, so they are
 * bundled for Node with esbuild (as zipp-routing.mjs does).
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
function check(name, fn) {
  try {
    fn();
    pass++;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failures.push(`${name} — ${e.message}`);
    console.log(`  FAIL ${name} — ${e.message}`);
  }
}

// A window as OAIY's gives the page: the desktop's address, token and theme.
const session = new Map();
globalThis.window = { __OAIY_DESKTOP__: { origin: 'http://127.0.0.1:17972', token: 't', theme: 'light' } };
globalThis.sessionStorage = { getItem: (k) => session.get(k) ?? null, setItem: (k, v) => session.set(k, String(v)) };

const bundlePath = path.join(os.tmpdir(), `oaiy-in-oaiy-${process.pid}.mjs`);
// The palette's environment reads the desktop's service list, which reaches oaiy-ui-components (the flow canvas's library): stubbed, as engine-endpoint.mjs does.
const uiStub = path.join(os.tmpdir(), `oaiy-in-oaiy-stub-${process.pid}.mjs`);
fs.writeFileSync(uiStub, 'export function invalidateDynamicOptions() {}\nexport function subscribeToDynamicOptionsInvalidation() { return () => {}; }\n');
await esbuild.build({
  stdin: {
    contents: `export { flowKey, fromDoc, reconcile, shared } from './src/hooks/useDesktopFlows.ts';
export { desktopTheme } from './src/contexts/ThemeContext.tsx';
export { taskText } from './src/bundled-modules/core-agent/runtime.ts';
export { shellNavItems } from './src/components/chrome/ShellChrome.tsx';
export { currentCaps } from './src/lib/caps.ts';
export { currentAvailabilityEnv } from './src/lib/availabilityEnv.ts';
export { paletteShowsNode, nodeNotice } from './src/lib/nodeAvailability.ts';
export { hasSavedDesktopLink } from './src/lib/desktopLink.ts';
export { forgetPageHost } from '@oaiy/shared/capabilities/host';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
  plugins: [{ name: 'oaiy-ui-components', setup: (build) => build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: uiStub })) }],
});
fs.rmSync(uiStub, { force: true });
const { flowKey, fromDoc, reconcile, shared, desktopTheme, taskText, shellNavItems, currentCaps, currentAvailabilityEnv, paletteShowsNode, nodeNotice, hasSavedDesktopLink, forgetPageHost } = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

const flow = (extra = {}) => ({ id: 'f1', name: 'Greeting', createdAt: '', updatedAt: '', graph: { nodes: [], edges: [] }, ...extra });

check("a desktop flow becomes the editor's, with its nodes and edges", () => {
  const f = fromDoc('welcome-note', { name: 'Welcome note', nodes: [{ id: 'a', type: 'input_text', data: {} }], edges: [] }, '2026-09-28T00:00:00Z');
  assert.equal(f.id, 'welcome-note');
  assert.equal(f.name, 'Welcome note');
  assert.equal(f.graph.nodes.length, 1);
  assert.equal(fromDoc('x', { name: 'no nodes' }, ''), null);
});

check('what is sent is the name, the nodes and the edges: a move of the view is not a change', () => {
  const a = flow();
  assert.equal(flowKey(a), flowKey({ ...a, updatedAt: 'later' }));
  assert.notEqual(flowKey(a), flowKey({ ...a, name: 'Other' }));
});

check("macros, demos, built-in and local-only flows stay the editor's own", () => {
  assert.equal(shared(flow()), true);
  for (const extra of [{ isMacro: true }, { isDemo: true }, { isBuiltIn: true }, { localOnly: true }]) assert.equal(shared(flow(extra)), false, JSON.stringify(extra));
});

const named = (id, name, nodes = []) => flow({ id, name, graph: { nodes, edges: [] } });
const key = (f) => flowKey(f);

check('a flow new on the desktop comes here, and one the two agree on stays', () => {
  const a = named('a', 'A');
  const r = reconcile(new Map([['a', a], ['b', named('b', 'B')]]), [a], new Map([['a', key(a)]]));
  assert.deepEqual(r.put.map((f) => f.id), ['b']);
  assert.deepEqual(r.remove, []);
  assert.equal(r.synced.get('b'), key(named('b', 'B')));
});

check("a flow changed there (the agent rewrote it) is replaced here, even at a start with nothing agreed", () => {
  const old = named('a', 'A');
  const rewritten = named('a', 'A', [{ id: 'n' }]);
  assert.deepEqual(reconcile(new Map([['a', rewritten]]), [old], new Map([['a', key(old)]])).put[0].graph, rewritten.graph);
  assert.equal(reconcile(new Map([['a', rewritten]]), [old], new Map()).put.length, 1);
});

check('a flow changed here, or on both sides, is left for the push to send', () => {
  const agreed = named('a', 'A');
  const edited = named('a', 'A edited');
  const r = reconcile(new Map([['a', agreed]]), [edited], new Map([['a', key(agreed)]]));
  assert.deepEqual(r.put, []);
  assert.equal(r.synced.get('a'), key(agreed), 'still differs from here, so it is sent');
  const both = reconcile(new Map([['a', named('a', 'A there')]]), [edited], new Map([['a', key(agreed)]]));
  assert.deepEqual(both.put, []);
});

check('a flow deleted there goes here unless changed here; one deleted here is not brought back', () => {
  const a = named('a', 'A');
  const gone = reconcile(new Map(), [a], new Map([['a', key(a)]]));
  assert.deepEqual(gone.remove, ['a']);
  assert.equal(gone.synced.has('a'), false);
  const kept = reconcile(new Map(), [named('a', 'A edited')], new Map([['a', key(a)]]));
  assert.deepEqual(kept.remove, []);
  const deletedHere = reconcile(new Map([['a', a]]), [], new Map([['a', key(a)]]));
  assert.deepEqual(deletedHere.put, [], 'the push removes it there');
});

check('"Ask the Agent" words its task from the input, the setting and the context', () => {
  assert.equal(taskText('Summarise this', '', null), 'Summarise this');
  assert.equal(taskText('Sam', 'Welcome {{ input }} to Green Lawns.', null), 'Welcome Sam to Green Lawns.');
  assert.equal(taskText(null, 'Check the calendar for tomorrow.', { booked: 3 }), 'Check the calendar for tomorrow.\n\nWhat the flow gives with it:\n{\n "booked": 3\n}');
  assert.equal(taskText('', '', ''), '');
});

check('the theme is the one OAIY said last', () => {
  assert.equal(desktopTheme(), 'light', 'at the start');
  session.set('oaiy-desktop-theme', 'dark');
  assert.equal(desktopTheme(), 'dark', 'said before a reload');
  window.__OAIY_THEME__ = 'light';
  assert.equal(desktopTheme(), 'light', 'said while loading');
  window.__OAIY_THEME__ = 'sepia';
  assert.equal(desktopTheme(), 'dark', 'not a theme');
});

// ---------------------------------------------------------------------------
// What the editor shows where it is (lib/caps.ts, shared/capabilities): in OAIY's window every control it had is still there.
// ---------------------------------------------------------------------------
// In a browser `window` is the page's global object; here it is a stand-in, so the page's globals get the same desktop (readHost reads them).
globalThis.__OAIY_DESKTOP__ = window.__OAIY_DESKTOP__;
forgetPageHost();
const store = new Map();
globalThis.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
const SECTIONS = ['workflows', 'data', 'queue', 'packages'];
const DESKTOP_NODES = ['browser_session', 'browser_page', 'browser_extract', 'browser_action', 'ask_agent', 'input_folder'];
const ids = (items) => items.map((i) => i.id);

/** The page as a tab in a browser on a public host, for `fn`: no desktop given, and the address of a site. */
function asTab(fn) {
  const given = window.__OAIY_DESKTOP__;
  delete window.__OAIY_DESKTOP__;
  delete globalThis.__OAIY_DESKTOP__;
  globalThis.location = { hostname: 'flows.example.org', protocol: 'https:', port: '', origin: 'https://flows.example.org', href: 'https://flows.example.org/app.html' };
  // The window is read once per page; this page becomes another one, so it is read again.
  forgetPageHost();
  try {
    return fn();
  } finally {
    window.__OAIY_DESKTOP__ = given;
    globalThis.__OAIY_DESKTOP__ = given;
    delete globalThis.location;
    forgetPageHost();
    store.clear();
  }
}

check("in OAIY's window every feature of the desktop's is on: nothing that was there is hidden", () => {
  const caps = currentCaps();
  assert.equal(caps.host, 'oaiy-window');
  assert.equal(caps.mode, 'paired');
  assert.equal(caps.looksOnLoad, true);
  for (const [id, on] of Object.entries(caps.features)) assert.equal(on, true, id);
  for (const id of ['phone', 'calendar', 'flowTools', 'control', 'setup', 'browserNodes', 'askAgent', 'inputFolder', 'packages', 'dock']) assert.ok(id in caps.features, `${id} is a feature the layer knows`);
});

check("the sections in OAIY's window are Workflows, Data, Queue and Packages, in that order, with the features given or not", () => {
  const noop = () => {};
  assert.deepEqual(ids(shellNavItems('workflows', noop)), SECTIONS, 'no features given: as before the editor knew where it was');
  assert.deepEqual(ids(shellNavItems('workflows', noop, {}, currentCaps().features)), SECTIONS, 'the features of this window');
  const items = shellNavItems('data', noop, { queue: 'badge' }, currentCaps().features);
  assert.deepEqual(items.map((i) => [i.id, i.active, i.badge]), [['workflows', false, undefined], ['data', true, undefined], ['queue', false, 'badge'], ['packages', false, undefined]]);
  const picked = [];
  items.find((i) => i.id === 'packages').onClick();
  shellNavItems('workflows', (s) => picked.push(s), {}, currentCaps().features).find((i) => i.id === 'packages').onClick();
  assert.deepEqual(picked, ['packages'], 'and Packages still opens');
});

check("the palette in OAIY's window offers the desktop's nodes (the browser nodes, Ask the Agent, Folder input) and says nothing is missing on them", () => {
  const env = currentAvailabilityEnv();
  assert.equal(env.inOaiy, true);
  for (const type of DESKTOP_NODES) {
    assert.equal(paletteShowsNode(type, env), true, type);
    assert.equal(nodeNotice(type, {}, env), null, `${type}: nothing missing`);
  }
});

check('a tab that is not linked to a desktop has no Packages section, no dock and none of the desktop\'s nodes in the palette', () => {
  asTab(() => {
    const caps = currentCaps();
    assert.equal(caps.host, 'browser');
    assert.equal(caps.mode, 'standalone');
    assert.equal(caps.looksOnLoad, false);
    assert.equal(hasSavedDesktopLink(), false);
    assert.deepEqual(ids(shellNavItems('workflows', () => {}, {}, caps.features)), ['workflows', 'data', 'queue']);
    assert.equal(caps.features.dock, false);
    const env = currentAvailabilityEnv();
    assert.equal(env.inOaiy, false);
    for (const type of DESKTOP_NODES) {
      assert.equal(paletteShowsNode(type, env), false, type);
      assert.match(nodeNotice(type, {}, env), /needs? OAIY Desktop or an OAIY server.*Connect it in Settings/, `${type}: says what it needs`);
    }
    assert.equal(paletteShowsNode('browser_request', env), true, 'a plain HTTP request is still offered');
    assert.equal(paletteShowsNode('input_text', env), true);
  });
});

check('a tab linked to a desktop (Connect answered) has the desktop\'s nodes and the dock, and still no Packages: no link gives a tab the desktop\'s own commands', () => {
  asTab(() => {
    store.set('oaiy.desktopLinked', '1');
    const caps = currentCaps();
    assert.equal(caps.mode, 'paired');
    assert.equal(caps.host, 'browser');
    assert.deepEqual(ids(shellNavItems('workflows', () => {}, {}, caps.features)), ['workflows', 'data', 'queue']);
    assert.equal(caps.features.dock, true);
    const env = currentAvailabilityEnv();
    for (const type of DESKTOP_NODES) {
      assert.equal(paletteShowsNode(type, env), true, type);
      assert.equal(nodeNotice(type, {}, env), null, type);
    }
  });
});

check('the capabilities are the same object until the link changes, so a component that follows them is not drawn again for nothing', () => {
  asTab(() => {
    const a = currentCaps();
    assert.equal(currentCaps(), a);
    store.set('oaiy.desktopLinked', '1');
    const b = currentCaps();
    assert.notEqual(b, a);
    assert.equal(currentCaps(), b);
  });
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
