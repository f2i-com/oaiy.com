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
    contents: `export { flowKey, fromDoc, reconcile, shared, toDoc } from './src/hooks/useDesktopFlows.ts';
export { resolveNodeService, withNodeServices } from './src/bundled-modules/core-service/contract.ts';
export { listenForFlowsIntents } from './src/lib/flowsIntents.ts';
export { desktopTheme } from './src/contexts/ThemeContext.tsx';
export { taskText } from './src/bundled-modules/core-agent/runtime.ts';
export { shellNavItems } from './src/components/chrome/ShellChrome.tsx';
export { currentCaps } from './src/lib/caps.ts';
export { currentAvailabilityEnv } from './src/lib/availabilityEnv.ts';
export { paletteShowsNode, nodeNotice } from './src/lib/nodeAvailability.ts';
export { hasSavedDesktopLink } from './src/lib/desktopLink.ts';
export { forgetPageHost } from '@oaiy/shared/capabilities/host';
export { withNewIds, rewriteFlowRefs } from './src/lib/flowIds.ts';
export { reachesUntrusted } from './src/lib/flowIds.ts';
export { UndoHistory, restoreOnto, canvasKey } from './src/lib/undoHistory.ts';
export { carriesProviderSeeds, refusalText, saveStateOf } from './src/hooks/useDesktopFlows.ts';
export { savedLabelOf } from './src/lib/savedLabel.ts';`,
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
const M = await import(pathToFileURL(bundlePath).href);
const { flowKey, fromDoc, reconcile, shared, toDoc, desktopTheme, taskText, shellNavItems, currentCaps, currentAvailabilityEnv, paletteShowsNode, nodeNotice, hasSavedDesktopLink, forgetPageHost } = M;
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

check("a desktop flow's description, tags and sharing come with it, and go back into its document", () => {
  const doc = { name: 'Pictures', description: 'Makes one', tags: ['image', 3], oaiyRemote: true, oaiyTool: { name: 'make' }, nodes: [], edges: [] };
  const f = fromDoc('pictures', doc, '');
  assert.equal(f.description, 'Makes one');
  assert.deepEqual(f.tags, ['image']);
  assert.equal(f.remote, true);
  assert.equal(fromDoc('plain', { name: 'Plain', nodes: [] }, '').remote, undefined);
  // Back: the editor's part put in, the rest of the document (the agent's tool) kept, a sharing switched off gone.
  const back = JSON.parse(JSON.stringify(toDoc({ ...f, remote: false, name: 'Pictures 2' }, doc)));
  assert.deepEqual(back, { name: 'Pictures 2', description: 'Makes one', tags: ['image'], oaiyTool: { name: 'make' }, nodes: [], edges: [] });
  // A change to any of them is one to send.
  for (const change of [{ description: 'Other' }, { tags: ['x'] }, { remote: false }]) assert.notEqual(flowKey(f), flowKey({ ...f, ...change }), JSON.stringify(change));
});

check("demos, built-in (macros among them) and local-only flows stay the editor's own; the person's macros go", () => {
  assert.equal(shared(flow()), true);
  assert.equal(shared(flow({ isMacro: true })), true, "a macro of the person's own: the flows using it run on the desktop with it");
  for (const extra of [{ isDemo: true }, { isBuiltIn: true }, { isMacro: true, isBuiltIn: true }, { localOnly: true }]) assert.equal(shared(flow(extra)), false, JSON.stringify(extra));
});

check('a macro goes to the desktop marked as one, and comes back one, with its inputs and outputs', () => {
  const nodes = [
    { id: 'mi', type: 'macro_input', data: { name: 'text', inputType: 'string' } },
    { id: 'mo', type: 'macro_output', data: { name: 'result' } },
  ];
  const macro = flow({ id: 'user-macro-1', name: 'Shout', isMacro: true, graph: { nodes, edges: [] } });
  const doc = JSON.parse(JSON.stringify(toDoc(macro)));
  assert.equal(doc.isMacro, true);
  assert.equal(JSON.parse(JSON.stringify(toDoc(flow()))).isMacro, undefined, 'an ordinary flow is not marked');
  const back = fromDoc('user-macro-1', doc, '');
  assert.equal(back.isMacro, true);
  assert.deepEqual(back.macroMetadata.inputs.map((i) => [i.id, i.name, i.type]), [['mi', 'text', 'string']]);
  assert.deepEqual(back.macroMetadata.outputs.map((o) => [o.id, o.name]), [['mo', 'result']]);
  // Made a macro, or made a flow again: a change to send.
  assert.notEqual(flowKey(macro), flowKey({ ...macro, isMacro: false }));
  // Changed there: it stays a macro here, its inputs read again.
  const rewritten = fromDoc('user-macro-1', { ...doc, nodes: [...nodes, { id: 'mi2', type: 'macro_input', data: { name: 'more' } }] }, '');
  const r = reconcile(new Map([['user-macro-1', rewritten]]), [macro], new Map([['user-macro-1', flowKey(macro)]]));
  assert.equal(r.put[0].isMacro, true);
  assert.deepEqual(r.put[0].macroMetadata.inputs.map((i) => i.name), ['text', 'more']);
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
  const deletedHere = reconcile(new Map([['a', a]]), [], new Map([['a', key(a)]]), new Set(['a']));
  assert.deepEqual(deletedHere.put, [], 'the push removes it there');
});

check('a flow missing here but not deleted here (a new project, an import) comes back rather than being deleted there', () => {
  const a = named('a', 'A');
  const back = reconcile(new Map([['a', a]]), [], new Map([['a', key(a)]]));
  assert.deepEqual(back.put.map((f) => f.id), ['a']);
  assert.equal(back.synced.get('a'), key(a));
});

check('a flow the desktop listed but could not send is left as it is here, not taken for deleted there', () => {
  const a = named('a', 'A');
  const r = reconcile(new Map(), [a], new Map([['a', key(a)]]), new Set(), new Set(['a']));
  assert.deepEqual(r.remove, []);
  assert.equal(r.synced.get('a'), key(a));
});

check("a node naming the person's own service, or the desktop's, carries how to call it there, never a key", () => {
  const { resolveNodeService, withNodeServices } = M;
  const mine = {
    id: 'my-rig', name: 'My rig', description: 'long words', icon: '🧩', endpoint: 'http://192.168.1.9:7000/run?key=sk-abcdefghijklmnopqrstuv', method: 'POST',
    headers: JSON.stringify({ Authorization: 'Bearer {{apiKey}}', 'X-Api-Key': 'sk-literalkey0123456789abcdef', 'Content-Type': 'application/json', 'X-Note': 'plain' }),
    bodyTemplate: '{"prompt": {{input}}, "token": "hf_abcdefghijklmnopqrstu"}', responseType: 'json', responsePath: 'out', apiKeyConstant: 'RIG_KEY', model: 'm1', installHint: 'pip install',
  };
  const companion = { id: 'companion:voice', name: 'Voice (running)', endpoint: 'http://127.0.0.1:9100/v1/audio', method: 'POST', headers: '{}', bodyTemplate: '{{inputRaw}}', responseType: 'json', responsePath: '', group: 'desktop' };
  const lists = { custom: [mine], desktop: [companion, { id: 'engine:image:flux', name: 'Flux', group: 'engine', kind: 'image', endpoint: '/api/x', method: 'POST', headers: '{}', bodyTemplate: '', responseType: 'json', responsePath: '' }] };
  const nodes = [
    { id: 'a', type: 'service_call', data: { service: 'my-rig' } },
    { id: 'b', type: 'model_3d', data: { service: 'engine:model3d', pictureService: 'companion:voice' } },
    { id: 'c', type: 'image_gen', data: { service: 'engine:image:flux' } },
    { id: 'd', type: 'service_call', data: { service: 'ollama-chat' } },
    { id: 'e', type: 'service_call', data: { service: 'gone', _services: { gone: { endpoint: 'x' } } } },
  ];
  const kept = withNodeServices(nodes, lists);
  const rig = kept[0].data._services['my-rig'];
  assert.equal(rig.model, 'm1');
  assert.equal(rig.apiKeyConstant, 'RIG_KEY', 'the constant is named, its value is not there');
  assert.equal(rig.description, undefined);
  assert.equal(rig.installHint, undefined);
  assert.deepEqual(JSON.parse(rig.headers), { Authorization: 'Bearer {{apiKey}}', 'Content-Type': 'application/json', 'X-Note': 'plain' }, 'a literal key in a header is not copied');
  assert.doesNotMatch(JSON.stringify(rig), /sk-abc|sk-literal|hf_abc/, 'no key, in any field');
  assert.equal(kept[1].data._services['companion:voice'].endpoint, 'http://127.0.0.1:9100/v1/audio');
  assert.equal(Object.keys(kept[1].data._services).length, 1, "OAIY's engine is looked up when the flow runs, not copied");
  assert.equal(kept[2].data._services, undefined);
  assert.equal(kept[3].data._services, undefined, 'a built-in example is in the code');
  assert.equal(kept[4].data._services, undefined, 'a copy of a service no longer listed is not kept on');
  // Where the lists are not (the CLI), the copy is what the node runs; where they are, the list's own wins.
  assert.equal(resolveNodeService('my-rig', kept[0].data).endpoint, rig.endpoint);
  assert.equal(resolveNodeService('my-rig', {}), null);
  // The editor's own flow never holds the copies: they come off as the desktop's flow comes in.
  const store = new Map([['oaiy.customServices', JSON.stringify([mine])]]);
  globalThis.localStorage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };
  const doc = JSON.parse(JSON.stringify(toDoc(flow({ graph: { nodes: [nodes[0]], edges: [] } }))));
  assert.ok(doc.nodes[0].data._services['my-rig'], 'saved to the desktop with it');
  const back = fromDoc('f1', doc, '');
  assert.equal(back.graph.nodes[0].data._services, undefined);
  assert.equal(flowKey(back), flowKey(flow({ graph: { nodes: [nodes[0]], edges: [] } })), 'so the two sides still agree');
  assert.equal(resolveNodeService('my-rig', {}).endpoint, mine.endpoint, "the editor's own list first");
  delete globalThis.localStorage;
});

check("the dashboard's \"open this flow\" is taken, asked before the editor listened or after, by a flow id alone", () => {
  const opened = [];
  // (asked while the page was starting: waiting where the desktop left it)
  window.__OAIY_FLOWS_INTENTS__ = [['openFlow', 'morning-greeting'], ['openFlow', 'bad id"); alert(1)'], ['closeAll', 'x']];
  const stop = M.listenForFlowsIntents((id) => opened.push(id));
  assert.deepEqual(opened, ['morning-greeting']);
  assert.equal(window.__OAIY_FLOWS_INTENTS__, undefined, 'taken, not taken again');
  window.__oaiyFlowsIntent('openFlow', 'tidy_up-2');
  window.__oaiyFlowsIntent('openFlow', 42);
  assert.deepEqual(opened, ['morning-greeting', 'tidy_up-2']);
  stop();
  assert.equal(window.__oaiyFlowsIntent, undefined);
  // Outside OAIY's window nothing listens.
  const given = window.__OAIY_DESKTOP__;
  delete window.__OAIY_DESKTOP__;
  M.listenForFlowsIntents(() => opened.push('no'));
  assert.equal(window.__oaiyFlowsIntent, undefined);
  window.__OAIY_DESKTOP__ = given;
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

check("the sections in OAIY's window are Flows, Data, Queue and Packages, in that order, with the features given or not", () => {
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

check('flows brought in (a shared link, Import flows…) get ids of their own, and the subflow and macro nodes that call each other follow', () => {
  const flows = [
    { id: 'a', name: 'A', graph: { nodes: [{ id: 'n1', type: 'subflow', data: { flowId: 'b' } }, { id: 'n2', type: 'macro', data: { _macroWorkflowId: 'm' } }], edges: [] } },
    { id: 'b', name: 'B', graph: { nodes: [{ id: 'n3', type: 'subflow', data: { flowId: 'elsewhere' } }], edges: [] } },
    { id: 'm', name: 'M', isMacro: true, graph: { nodes: [], edges: [] } },
  ];
  let n = 0;
  const { flows: out, ids } = M.withNewIds(flows, () => `new-${++n}`);
  assert.deepEqual(out.map((f) => f.id), ['new-1', 'new-2', 'new-3']);
  assert.equal(ids.get('b'), 'new-2');
  assert.equal(out[0].graph.nodes[0].data.flowId, 'new-2');
  assert.equal(out[0].graph.nodes[1].data._macroWorkflowId, 'new-3');
  assert.equal(out[1].graph.nodes[0].data.flowId, 'elsewhere', 'a flow not brought in is still named as it was');
  assert.equal(flows[0].graph.nodes[0].data.flowId, 'b', 'what it was given is unchanged');
  assert.equal(out[2], M.rewriteFlowRefs([out[2]], ids)[0], 'a flow with nothing to rename is the same object');
});

check('a shared flow runs sandboxed, and so does a flow that calls one; the rest of the project does not', () => {
  const flows = [
    { id: 'mine', graph: { nodes: [], edges: [] } },
    { id: 'shared', untrusted: true, graph: { nodes: [], edges: [] } },
    { id: 'calls', graph: { nodes: [{ id: 'x', type: 'subflow', data: { flowId: 'mid' } }], edges: [] } },
    { id: 'mid', graph: { nodes: [{ id: 'y', type: 'macro', data: { _macroWorkflowId: 'shared' } }], edges: [] } },
  ];
  const graph = (id) => flows.find((f) => f.id === id).graph;
  assert.equal(M.reachesUntrusted('mine', graph('mine'), flows), false);
  assert.equal(M.reachesUntrusted('shared', graph('shared'), flows), true);
  assert.equal(M.reachesUntrusted('calls', graph('calls'), flows), true, 'through a flow it calls');
});

check('undo goes back over adds, deletes, moves and wires, keeps what a run put on the nodes, and redo comes forward again', () => {
  const node = (id, x = 0, data = {}) => ({ id, type: 'input_text', position: { x, y: 0 }, data });
  const h = new M.UndoHistory();
  const a = { nodes: [node('a')], edges: [] };
  assert.equal(h.record(a), null, 'the first state is where it starts');
  const b = { nodes: [node('a'), node('b', 200)], edges: [] };
  assert.deepEqual(h.record(b).nodes, a.nodes, 'a node added is a step');
  assert.equal(h.record({ nodes: b.nodes.map((n) => ({ ...n, selected: true, data: { ...n.data, outputValue: 'ran', _status: 'completed' } })), edges: [] }), null, 'a selection or a run is not');
  const c = { nodes: b.nodes, edges: [{ id: 'e1', source: 'a', target: 'b' }] };
  h.record(c);
  const moved = { nodes: [node('a', 50), node('b', 200)], edges: c.edges };
  // an undo straight after a change takes it in first (it had not settled)
  const ran = { nodes: moved.nodes.map((n) => ({ ...n, data: { ...n.data, outputValue: 'ran' } })), edges: moved.edges };
  const back = h.undo(ran);
  assert.equal(M.canvasKey(back.nodes, back.edges), M.canvasKey(c.nodes, c.edges), 'the move is undone');
  const shown = M.restoreOnto(back, ran);
  assert.equal(shown.nodes[0].data.outputValue, 'ran', "a run's output stays on the node");
  const unwired = M.restoreOnto(h.undo(shown), shown);
  assert.equal(unwired.edges.length, 0, 'then the wire');
  const first = M.restoreOnto(h.undo(unwired), unwired);
  assert.equal(first.nodes.length, 1, 'then the node added');
  assert.equal(h.undo(first), null, 'and no further');
  assert.equal(h.redo(a).nodes.length, 2);
  assert.ok(h.canRedo);
  // a change after an undo ends what could be redone
  h.record({ nodes: [node('a'), node('b', 200), node('c', 400)], edges: [] });
  assert.equal(h.canRedo, false);
  // a picture dropped on a node is compared by its length and start, not whole
  assert.ok(M.canvasKey([node('p', 0, { fileContent: `data:image/png;base64,${'A'.repeat(100000)}` })], []).length < 400);
});

// From the flow editor's review (2026-10-19): an AI LLM node saved with Ollama's defaults under its service, and what
// the header says while a flow is or is not on the desktop.
const SEEDED = { provider: 'ollama', model: 'llama3.2', endpoint: 'http://localhost:11434/v1/chat/completions', requestFormat: 'openai' };

check("a desktop flow's AI LLM nodes come without the Ollama fields they were given under their service, and say they had them", () => {
  const doc = { name: 'Lighthouse', nodes: [
    { id: 'llm', type: 'ai_llm', data: { service: 'engine:llm', ...SEEDED, systemPrompt: 'Be brief.' } },
    { id: 'own', type: 'ai_llm', data: { service: '', ...SEEDED } },
    { id: 'mine', type: 'ai_llm', data: { service: 'engine:llm', provider: 'lmstudio', model: 'my-model' } },
  ], edges: [] };
  const f = fromDoc('lighthouse', doc, '');
  assert.deepEqual(f.graph.nodes[0].data, { service: 'engine:llm', systemPrompt: 'Be brief.' });
  assert.deepEqual(f.graph.nodes[1].data, { service: '', ...SEEDED }, 'with no service they are the node\'s own');
  assert.deepEqual(f.graph.nodes[2].data, { service: 'engine:llm', provider: 'lmstudio', model: 'my-model' }, 'what is not Ollama\'s default is kept');
  assert.equal(M.carriesProviderSeeds(doc), true, 'sent back without them');
  assert.equal(M.carriesProviderSeeds({ nodes: f.graph.nodes }), false);
  assert.equal(M.carriesProviderSeeds({ name: 'none' }), false);
});

check('a save the desktop refused says its words, or that it did not take it: never its status', () => {
  assert.equal(M.refusalText(500, { error: 'disk full' }), 'OAIY Desktop said: disk full');
  assert.equal(M.refusalText(500, { error: { message: 'no room' } }), 'OAIY Desktop said: no room');
  assert.equal(M.refusalText(500, null), 'OAIY Desktop did not take it');
  assert.equal(M.refusalText(403, { other: 1 }), 'OAIY Desktop did not take it');
  assert.equal(M.refusalText(null, null), 'OAIY Desktop could not be reached');
  assert.ok(!/\d{3}/.test(M.refusalText(503, {})));
});

check("the header says where the flow is from the save's result: Saving…, Not saved yet, Saved in OAIY", () => {
  const f = flow({ graph: { nodes: [{ id: 'a', type: 'input_text', data: {} }], edges: [] } });
  assert.equal(M.saveStateOf(f, flowKey(f), false), 'saved');
  assert.equal(M.saveStateOf(f, undefined, false), 'saving', 'changed here, not sent yet');
  assert.equal(M.saveStateOf(f, 'older', true), 'not-saved', 'the desktop did not take it');
  assert.equal(M.saveStateOf(f, flowKey(f), true), 'saved', 'it has it now');
  const label = (o) => M.savedLabelOf({ inOaiy: true, shared: true, storage: 'saved', desktop: 'saved', ...o });
  assert.deepEqual(label({}), { label: 'Saved in OAIY', problem: false });
  assert.deepEqual(label({ desktop: 'saving' }), { label: 'Saving…', problem: false });
  assert.deepEqual(label({ desktop: 'not-saved' }), { label: 'Not saved yet', problem: true });
  assert.deepEqual(label({ storage: 'not-saved', desktop: 'saved' }), { label: 'Not saved: storage full', problem: true });
  assert.deepEqual(label({ shared: false, desktop: 'not-saved' }), { label: 'Saved in this window', problem: false }, 'one kept in the window is not the desktop\'s');
  assert.deepEqual(label({ inOaiy: false }), { label: 'Saved in this browser', problem: false });
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
