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
await esbuild.build({
  stdin: {
    contents: `export { flowKey, fromDoc, reconcile, shared } from './src/hooks/useDesktopFlows.ts';
export { desktopTheme } from './src/contexts/ThemeContext.tsx';
export { taskText } from './src/bundled-modules/core-agent/runtime.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
});
const { flowKey, fromDoc, reconcile, shared, desktopTheme, taskText } = await import(pathToFileURL(bundlePath).href);
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

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
