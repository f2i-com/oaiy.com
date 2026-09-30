/**
 * What the flow editor's nodes load a picture, a video or a sound from (the review's F4).
 *
 *     npm run test:local-media
 *
 * A flow can carry the address of what its nodes show, and a flow can come from someone else. Opening one made the tab GET whatever
 * address its author wrote, with no run and no Connect: 127.0.0.1, the LAN. So:
 *
 *   - a tab that is not linked to OAIY Desktop does not load media from an address on this computer or its network (lib/localMedia.ts),
 *     and its nodes say so; OAIY's own window, the desktop shell and a tab that IS linked load them as they always did;
 *   - a flow that comes in from outside arrives without the run outputs those nodes show (utils/runOutputs.ts): an imported project, an
 *     imported workflow, a flow file.
 *
 * The modules are TypeScript with aliases only the bundler resolves, so they are bundled for Node with esbuild (as engine-endpoint.mjs).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';
import { UI, suite } from './support/loadTs.mjs';

const { check, finish } = suite('local media');

const stub = path.join(os.tmpdir(), `oaiy-local-media-stub-${process.pid}.mjs`);
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
    build.onResolve({ filter: /^\.\/ffmpeg$/ }, () => ({ path: './ffmpeg', external: true }));
  },
};
const bundlePath = path.join(os.tmpdir(), `oaiy-local-media-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { mediaUrlBlocked, BLOCKED_MEDIA_WORDS } from './src/lib/localMedia.ts';
export { RUN_OUTPUT_FIELDS, withoutRunOutputs, flowWithoutRunOutputs } from './src/utils/runOutputs.ts';
export { parseImportedProject } from './src/utils/ProjectIO.ts';
export { parseWorkflowJson } from './src/utils/WorkflowIO.ts';
export { normalizeFlowData } from './src/hooks/usePackageManager.ts';
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
  define: { 'import.meta.env': '{"DEV":false,"PROD":true,"MODE":"test"}' },
  plugins: [aliases],
});
// usePackageManager.ts (normalizeFlowData lives there) pulls in browser libraries that read `window` as they load.
globalThis.window ??= globalThis;
const M = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });
fs.rmSync(stub, { force: true });

// A page: its address, storage, and (for OAIY's window) the desktop the window gave it. The window is read once per page, so each is read afresh.
const store = new Map();
globalThis.localStorage = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
function onPage({ url = 'https://flows.example.org/app.html', desktop = null, linked = false, address = null }, fn) {
  const u = new URL(url);
  globalThis.location = { hostname: u.hostname, protocol: u.protocol, port: u.port, origin: u.origin, href: u.href };
  delete globalThis.__OAIY_DESKTOP__;
  if (desktop) globalThis.__OAIY_DESKTOP__ = desktop;
  store.clear();
  if (linked) store.set('oaiy.desktopLinked', '1');
  if (address) store.set('oaiy.engineBase', address);
  M.forgetPageHost();
  try {
    return fn();
  } finally {
    delete globalThis.location;
    delete globalThis.__OAIY_DESKTOP__;
    store.clear();
    M.forgetPageHost();
  }
}
const tab = (fn) => onPage({}, fn);

const LOCAL = [
  'http://127.0.0.1:8188/view?filename=stale.png',
  'http://127.0.0.1:8188/view?filename=stale.mp4',
  'http://localhost/x.png',
  'http://[::1]:8080/a.wav',
  'http://192.168.77.8/pic.png',
  'http://10.0.0.5/a.mp4',
  'http://172.16.0.9/a.mp3',
  'http://172.31.255.255/a.png',
  'http://169.254.169.254/latest/meta-data',
  'http://100.100.100.100/a.png',
  'http://printer.local/a.png',
  'http://foo.localhost/a.png',
  'https://192.168.1.1/x.png',
  'http://0.0.0.0/x',
  'HTTP://127.0.0.1/x.png',
  '   http://127.0.0.1/x.png',
  'http://user:pass@192.168.1.5/x.png',
  'http://[::ffff:192.168.1.5]/x.png',
  'http://[fd12:3456::1]/x.png',
];
const FINE = [
  'data:image/png;base64,iVBORw0KGgo=',
  'data:video/mp4;base64,AAAA',
  'blob:https://flows.example.org/0d1e2f',
  '/images/a.png',
  'a.png',
  './pics/a.png',
  'https://flows.example.org/x.png',
  'https://example.org/a.png',
  'http://8.8.8.8/x.png',
  'https://cdn.example.org:8443/a.mp4',
  'http://172.32.0.1/a.png',
  'http://192.169.1.1/a.png',
  'not an address at all',
  '',
  'C:\\Users\\me\\a.png',
  'file:///C:/a.png',
];

// ---------------------------------------------------------------------------
// Where a tab that is not linked will not load media from
// ---------------------------------------------------------------------------
await check('a tab that is not linked to a desktop does not load media from loopback, the private ranges, link-local, Tailscale or a local name', () => {
  tab(() => {
    for (const url of LOCAL) assert.equal(M.mediaUrlBlocked(url), true, url);
  });
});

await check('it loads what is not on this computer or its network, what is data or a blob, and what is the page\'s own (a relative address, its own origin)', () => {
  tab(() => {
    for (const url of FINE) assert.equal(M.mediaUrlBlocked(url), false, JSON.stringify(url));
    for (const value of [undefined, null, 7, {}, [], true]) assert.equal(M.mediaUrlBlocked(value), false, String(value));
  });
});

await check('a local copy of the editor loads its own files, and only its own origin: another port of the same computer is another address', () => {
  onPage({ url: 'http://localhost:5173/app.html' }, () => {
    assert.equal(M.mediaUrlBlocked('http://localhost:5173/assets/a.png'), false, 'its own origin');
    assert.equal(M.mediaUrlBlocked('/assets/a.png'), false);
    assert.equal(M.mediaUrlBlocked('http://localhost:8188/view?filename=x.png'), true, 'ComfyUI on the same computer');
    assert.equal(M.mediaUrlBlocked('http://127.0.0.1:5173/assets/a.png'), true, 'another name of it is another origin');
  });
});

await check('OAIY\'s own window (the desktop is given), the desktop shell and a tab that is linked load them all, as they always did', () => {
  const given = { origin: 'http://127.0.0.1:17972', token: 'window-token', theme: 'dark' };
  onPage({ url: 'http://oaiyflows.localhost/app.html', desktop: given }, () => {
    for (const url of LOCAL) assert.equal(M.mediaUrlBlocked(url), false, `window: ${url}`);
  });
  onPage({ url: 'oaiyflows://localhost/app.html' }, () => {
    for (const url of LOCAL) assert.equal(M.mediaUrlBlocked(url), false, `oaiyflows://: ${url}`);
  });
  onPage({ linked: true }, () => {
    for (const url of LOCAL) assert.equal(M.mediaUrlBlocked(url), false, `linked by Connect: ${url}`);
  });
  onPage({ address: 'http://192.168.1.50:17972' }, () => {
    for (const url of LOCAL) assert.equal(M.mediaUrlBlocked(url), false, `an engine address of its own: ${url}`);
  });
});

await check('a page that is at oaiy.localhost on a port is a tab, not OAIY\'s window: it does not load them', () => {
  onPage({ url: 'http://oaiyflows.localhost:8000/app.html' }, () => {
    assert.equal(M.mediaUrlBlocked('http://192.168.77.8/pic.png'), true);
  });
});

await check('what it says in place of the media names why and what to do', () => {
  assert.match(M.BLOCKED_MEDIA_WORDS, /^Blocked: this address is on your computer or network, and this page is not linked to OAIY Desktop\. Connect it in Settings → Services to load it\.$/);
});

// ---------------------------------------------------------------------------
// A flow that comes in from outside arrives without run outputs
// ---------------------------------------------------------------------------
const node = (id, type, data) => ({ id, type, position: { x: 0, y: 0 }, data });

await check('run outputs are what a run writes into a node, and only those (Output\'s result, an image\'s address, a video\'s address)', () => {
  assert.deepEqual([...M.RUN_OUTPUT_FIELDS].sort(), ['imageUrl', 'outputValue', 'videoUrl']);
});

await check('withoutRunOutputs leaves out those fields, keeps every other field and every other node as it was, and changes nothing it was given', () => {
  const nodes = [
    node('a', 'output', { label: 'Result', outputValue: 'http://192.168.77.7:8188/view?filename=stale.png' }),
    node('b', 'image_view', { imageUrl: 'http://192.168.77.8/pic.png', url: 'http://192.168.77.8/pic.png', label: 'View' }),
    node('c', 'video_save', { videoUrl: 'http://127.0.0.1:8188/x.mp4', outputValue: 'x', filename: 'clip' }),
    node('d', 'service_call', { endpoint: 'http://192.168.77.11:9000/x', method: 'GET' }),
    { id: 'e', type: 'input_text', position: { x: 0, y: 0 } },
    { id: 'f', type: 'input_text', position: { x: 0, y: 0 }, data: null },
  ];
  const before = JSON.stringify(nodes);
  const out = M.withoutRunOutputs(nodes);
  assert.equal(JSON.stringify(nodes), before, 'the input is not changed');
  assert.deepEqual(out[0].data, { label: 'Result' });
  assert.deepEqual(out[1].data, { url: 'http://192.168.77.8/pic.png', label: 'View' });
  assert.deepEqual(out[2].data, { filename: 'clip' });
  assert.equal(out[3], nodes[3], 'a node with none is the same node');
  assert.equal(out[4], nodes[4]);
  assert.equal(out[5], nodes[5]);
  assert.deepEqual(out.map((n) => n.id), ['a', 'b', 'c', 'd', 'e', 'f']);
});

const STALE_FLOW = { id: 'shared-flow', name: 'Shared flow', createdAt: '', updatedAt: '', graph: { nodes: [node('out', 'output', { label: 'Out', outputValue: 'http://192.168.77.7:8188/view?filename=stale.png' }), node('view', 'image_view', { imageUrl: 'http://192.168.77.8/pic.png' }), node('svc', 'service_call', { endpoint: 'http://192.168.77.11:9000/x' })], edges: [] } };

await check('an imported project arrives without them, whatever else it carries', () => {
  const project = M.parseImportedProject(JSON.stringify({ version: '1.0', name: 'shared', flows: [STALE_FLOW, { id: 'broken', name: 'Broken', graph: null }] }));
  const [flow, broken] = project.flows;
  assert.deepEqual(flow.graph.nodes.map((n) => n.data), [{ label: 'Out' }, {}, { endpoint: 'http://192.168.77.11:9000/x' }]);
  assert.equal(flow.name, 'Shared flow');
  assert.deepEqual(broken.graph, { nodes: [], edges: [] });
});

await check('an imported workflow file arrives without them', () => {
  const parsed = M.parseWorkflowJson(JSON.stringify({ name: 'wf', nodes: STALE_FLOW.graph.nodes, edges: [] }));
  assert.deepEqual(parsed.nodes.map((n) => n.data), [{ label: 'Out' }, {}, { endpoint: 'http://192.168.77.11:9000/x' }]);
  assert.equal(parsed.name, 'wf');
});

await check('a flow file added with "Import flows…" arrives without them, in either shape it may have (nodes and edges at the top, or a graph)', () => {
  const clean = [{ label: 'Out' }, {}, { endpoint: 'http://192.168.77.11:9000/x' }];
  const flat = M.normalizeFlowData({ name: 'flat', nodes: STALE_FLOW.graph.nodes, edges: [] });
  assert.deepEqual(flat.graph.nodes.map((n) => n.data), clean);
  const graphed = M.normalizeFlowData({ name: 'graphed', graph: STALE_FLOW.graph });
  assert.deepEqual(graphed.graph.nodes.map((n) => n.data), clean);
  assert.equal(graphed.name, 'graphed');
  assert.deepEqual(STALE_FLOW.graph.nodes[0].data, { label: 'Out', outputValue: 'http://192.168.77.7:8188/view?filename=stale.png' }, 'what it was given is unchanged');
  assert.equal(M.normalizeFlowData({ name: 'neither' }), null);
});

await check('a flow that a shared link brings (App.tsx keeps its project) arrives without them, and what is not a flow with a graph is left as it is', () => {
  const shared = M.flowWithoutRunOutputs(STALE_FLOW);
  assert.deepEqual(shared.graph.nodes.map((n) => n.data), [{ label: 'Out' }, {}, { endpoint: 'http://192.168.77.11:9000/x' }]);
  assert.equal(shared.name, 'Shared flow');
  assert.deepEqual(shared.graph.edges, []);
  assert.deepEqual(STALE_FLOW.graph.nodes[0].data, { label: 'Out', outputValue: 'http://192.168.77.7:8188/view?filename=stale.png' }, 'what it was given is unchanged');
  for (const other of [null, undefined, 7, 'text', {}, { graph: null }, { graph: {} }, { graph: { nodes: 'no' } }]) assert.equal(M.flowWithoutRunOutputs(other), other, JSON.stringify(other));
});

fs.rmSync(stub, { force: true });
finish();
