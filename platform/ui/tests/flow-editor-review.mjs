/**
 * What a hands-on review of the flow editor found (2026-10-19), and keeps fixed.
 *
 *     npm run test:flow-editor-review
 *
 * - An AI LLM node added from the palette in OAIY's window was given Ollama's
 *   address and llama3.2 under the service it showed, and they won when the
 *   flow ran (utils/NodeDefaults.ts; the compiler's side is in
 *   flow-services.mjs, the desktop's flows in in-oaiy.mjs).
 * - A Logic Block of one line that is a statement (`throw new Error("no")`)
 *   was taken for an expression and the flow did not compile; a block that
 *   fails says which block, and its card is marked (core-utility/compiler.ts,
 *   oaiy-core's runtime). Run here on oaiy-core's JobManager, in this thread.
 * - A run that starts clears the last one's results; one that fails or is
 *   stopped marks them (hooks/useWorkflowExecution.ts).
 * - ⌘A and ⌘V after a click on an empty part of the canvas (lib/canvasTarget.ts).
 * - "Import a project…" says what a file is not before asking to replace the
 *   project (utils/ProjectIO.ts).
 * - A toast says a step's error without its HTTP status (lib/readableError.ts).
 * - An example on the models this computer has (lib/exampleFlows.ts), and cards
 *   kept clear of one a run made taller (lib/respace.ts).
 *
 * The modules are TypeScript with aliases only the bundler resolves, so they are
 * bundled for Node with esbuild (as flow-services.mjs does).
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

// A plain page: storage, and no OAIY around it.
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
globalThis.window = globalThis.window ?? {
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  addEventListener() {},
  removeEventListener() {},
  dispatchEvent() { return true; },
  location: { protocol: 'https:', origin: 'https://oaiy.com' },
};
// Nothing here reaches the network, OAIY's own ports least of all.
globalThis.fetch = async (url) => { throw new Error(`this test may not reach ${url}`); };

const stub = path.join(os.tmpdir(), `oaiy-review-stub-${process.pid}.mjs`);
fs.writeFileSync(stub, 'export function invalidateDynamicOptions() {}\nexport function subscribeToDynamicOptionsInvalidation() { return () => {}; }\nexport function getNodeNotice() { return null; }\n');
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
    build.onResolve({ filter: /^@oaiy\/shared\// }, (a) => {
      const p = path.join(UI, '..', '..', 'shared', a.path.slice('@oaiy/shared/'.length));
      return { path: fs.existsSync(p) ? p : `${p}.ts` };
    });
  },
};
const MODS = ['core-input', 'core-utility', 'core-flow-control'];
const bundlePath = path.join(os.tmpdir(), `oaiy-review-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { getDefaultNodeData, withoutProviderSeeds } from './src/utils/NodeDefaults.ts';
export { beforeRun, afterUnfinishedRun } from './src/lib/runResults.ts';
export { isCanvasTarget } from './src/lib/canvasTarget.ts';
export { parseImportedProject, parseJsonFile } from './src/utils/ProjectIO.ts';
export { toastError, readableError } from './src/lib/readableError.ts';
export { withModelsHere, exampleFrom, flowFromExample } from './src/lib/exampleFlows.ts';
export { clearBelow } from './src/lib/respace.ts';
export { takeNodeMark, nodeErrorPrefix } from './vendor/oaiy-core/src/node-errors.ts';
export { JobManager, registerBundledModules, getModuleLoader, loadBundledModules } from 'oaiy-core';
${MODS.map((m, i) => `import manifest${i} from './src/bundled-modules/${m}/module.json';
import runtime${i} from './src/bundled-modules/${m}/runtime.ts';
import compiler${i} from './src/bundled-modules/${m}/compiler.ts';
${fs.readdirSync(path.join(UI, 'src/bundled-modules', m, 'nodes')).map((n, j) => `import node${i}_${j} from './src/bundled-modules/${m}/nodes/${n}';`).join('\n')}`).join('\n')}
export const MODULES = [${MODS.map((m, i) => `{ manifest: manifest${i}, runtime: runtime${i}, compiler: compiler${i}, nodes: [${fs.readdirSync(path.join(UI, 'src/bundled-modules', m, 'nodes')).map((_, j) => `node${i}_${j}`).join(', ')}] }`).join(', ')}];`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  jsx: 'automatic',
  // (vite's `import.meta.glob`, which the example list uses, is not esbuild's: no examples are read here)
  define: { 'import.meta.glob': '__oaiyNoGlob' },
  banner: { js: "import { createRequire as __oaiyRequire } from 'node:module'; const require = __oaiyRequire(import.meta.url); const __oaiyNoGlob = () => ({});" },
  plugins: [aliases],
});
const M = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });
fs.rmSync(stub, { force: true });

// ---------------------------------------------------------------------------
// 1. An AI LLM node's provider fields under a service.
// ---------------------------------------------------------------------------

const OLLAMA = { provider: 'ollama', model: 'llama3.2', endpoint: 'http://localhost:11434/v1/chat/completions' };

await check('an AI LLM node added on a service is not given the Ollama provider, model and address', () => {
  const settings = { defaultAIProvider: 'ollama', defaultAIEndpoint: OLLAMA.endpoint, defaultAIModel: '' };
  const onService = M.getDefaultNodeData('ai_llm', settings, { service: 'engine:llm' });
  for (const key of ['provider', 'model', 'endpoint', 'requestFormat', 'apiKeyConstant']) assert.equal(key in onService, false, key);
  assert.equal(onService.systemPrompt, 'You are a helpful assistant.');
  // with no service, its provider's fields are what it runs on, and it is given them as before
  const own = M.getDefaultNodeData('ai_llm', settings);
  assert.equal(own.provider, 'ollama');
  assert.equal(own.endpoint, OLLAMA.endpoint);
  assert.equal(M.getDefaultNodeData('ai_llm', settings, { service: '  ' }).provider, 'ollama', 'a blank service is none');
});

await check("a saved AI LLM node's Ollama fields under its service are left out when it comes in; its own are kept", () => {
  const nodes = [
    { id: 'a', type: 'ai_llm', data: { service: 'engine:llm', ...OLLAMA, requestFormat: 'openai', systemPrompt: 'Hi' } },
    { id: 'b', type: 'ai_llm', data: { service: 'engine:llm', provider: 'lmstudio', model: 'mine', endpoint: OLLAMA.endpoint } },
    { id: 'c', type: 'ai_llm', data: { ...OLLAMA } },
    { id: 'd', type: 'image_gen', data: { service: 'engine:image', model: 'llama3.2' } },
  ];
  const out = M.withoutProviderSeeds(nodes);
  assert.deepEqual(out[0].data, { service: 'engine:llm', systemPrompt: 'Hi' });
  assert.deepEqual(out[1].data, { service: 'engine:llm', provider: 'lmstudio', model: 'mine' }, "the person's provider and model stay");
  assert.equal(out[2], nodes[2], 'no service: untouched');
  assert.equal(out[3], nodes[3], 'not an AI LLM node: untouched');
});

// ---------------------------------------------------------------------------
// 3 and 5. Logic Blocks of one line, and a block that fails.
// ---------------------------------------------------------------------------

M.registerBundledModules(M.MODULES);
const loader = M.getModuleLoader();
await M.loadBundledModules(loader, M.MODULES);
const engine = new M.JobManager({ moduleRegistry: loader });

/** Run a Text Input (`text`) into a Logic Block named "Check" with `code`; its result, error and its node's statuses. */
async function runBlock(code, text = 'lighthouse', label = 'Check') {
  const graph = {
    nodes: [
      { id: 'in', type: 'input_text', position: { x: 0, y: 0 }, data: { value: text } },
      { id: 'blk', type: 'logic_block', position: { x: 200, y: 0 }, data: { code, label } },
    ],
    edges: [{ id: 'e1', source: 'in', sourceHandle: 'text', target: 'blk', targetHandle: 'input' }],
  };
  const statuses = [];
  const off = engine.onNodeStatus((_job, nodeId, status) => statuses.push(`${nodeId}:${status}`));
  const id = engine.submit('review', 'Review', graph);
  await new Promise((resolve) => {
    const done = () => ['completed', 'failed', 'aborted'].includes(engine.getJob(id)?.status);
    if (done()) return resolve();
    const stop = engine.onStateChange(() => { if (done()) { stop(); resolve(); } });
  });
  off();
  const job = engine.getJob(id);
  return { status: job.status, error: job.error, failed: job.failedNodeId, value: job.nodeOutputs?.blk, statuses, logs: job.logs.map((l) => l.message) };
}

await check('one-line Logic Blocks compile and run, statements and expressions alike', async () => {
  const cases = [
    ['input.toUpperCase()', 'LIGHTHOUSE'],
    ['input.toUpperCase();', 'LIGHTHOUSE'],
    ['input.length', 10],
    ['return input + "!"', 'lighthouse!'],
    ['`${input} at dusk`', 'lighthouse at dusk'],
    ['// the length\ninput.length', 10],
    ['let n = input.length', null],
    ['if (input) input.slice(0, 5)', null],
  ];
  for (const [code, want] of cases) {
    const r = await runBlock(code);
    assert.equal(r.status, 'completed', `${JSON.stringify(code)}: ${r.error}`);
    assert.deepEqual(r.value ?? null, want, JSON.stringify(code));
  }
});

await check("a one-line throw fails the run as the block's own error, named, and marks its node", async () => {
  for (const code of ['throw new Error("nope")', 'throw new Error("nope");', 'if (input) throw new Error("nope")']) {
    const r = await runBlock(code);
    assert.equal(r.status, 'failed', code);
    assert.equal(r.error, "Logic Block 'Check': nope", code);
    assert.ok(r.statuses.includes('blk:error'), `${code}: ${r.statuses}`);
    assert.ok(!r.logs.some((l) => /oaiy-node:|Zipp/.test(l)), r.logs.join('\n'));
  }
});

await check("a block's name is said as it is written, quotes and all; one with none is the Logic Block", async () => {
  assert.equal((await runBlock('throw new Error("nope")', 'x', 'Say "hi"\u2028now')).error, 'Logic Block \'Say "hi"\u2028now\': nope');
  assert.equal((await runBlock('throw new Error("nope")', 'x', '')).error, 'Logic Block: nope');
});

await check('a block that throws a TypeError says so, of the block', async () => {
  const r = await runBlock('return input.missing.call();');
  assert.equal(r.status, 'failed');
  assert.match(r.error, /^Logic Block 'Check': TypeError: /);
  assert.ok(!/in <script>/.test(r.error));
  assert.ok(r.statuses.includes('blk:error'));
  assert.ok(r.logs.some((l) => l.startsWith("The flow failed: Logic Block 'Check': TypeError")), r.logs.join('\n'));
});

await check('code that does not parse is a syntax error of the block, with its line, and marks its node', async () => {
  const r = await runBlock('return input +;');
  assert.equal(r.status, 'failed');
  assert.match(r.error, /^Logic Block 'Check': syntax error on line 1: Unexpected token$/);
  assert.ok(r.statuses.includes('blk:error'), String(r.statuses));
  assert.equal(r.failed, 'blk', 'and the job says which, for a canvas that hears of it after');
  const later = await runBlock('\n\nconst a = 1;\nreturn a +;');
  assert.match(later.error, /syntax error on line 4:/, 'the line as written, its blank lines counted');
});

await check('a mark is taken off a message wherever it is wrapped', () => {
  assert.deepEqual(M.takeNodeMark(`Error: Error: ${M.nodeErrorPrefix('n-1')}Logic Block: no`), { nodeId: 'n-1', message: 'Logic Block: no' });
  assert.deepEqual(M.takeNodeMark('Plain failure'), { nodeId: null, message: 'Plain failure' });
});

// ---------------------------------------------------------------------------
// 5. A run's results, as it starts and after it fails.
// ---------------------------------------------------------------------------

await check("a run starting clears the last run's results and marks; one that fails or stops marks its results stale", () => {
  const out = { id: 'o', type: 'output', data: { label: 'Result', outputValue: 'HELLO', _status: 'completed' } };
  const view = { id: 'v', type: 'image_view', data: { imageUrl: 'data:image/png;base64,AA', _stale: true } };
  const text = { id: 't', type: 'input_text', data: { value: 'Hello' } };
  assert.deepEqual(M.beforeRun(out).data, { label: 'Result' });
  assert.deepEqual(M.beforeRun(view).data, {});
  assert.equal(M.beforeRun(text), text, 'an input keeps its value');
  const running = { id: 'g', type: 'image_gen', data: { _status: 'running' } };
  assert.equal(M.afterUnfinishedRun(running, true).data._status, 'stopped', 'where Stop found it');
  assert.equal(M.afterUnfinishedRun(running, false).data._status, undefined, 'a failure elsewhere: not running any more');
  assert.equal(M.afterUnfinishedRun({ ...running, data: { _status: 'error' } }, false).data._status, 'error', 'the one that failed keeps its mark');
  assert.equal(M.afterUnfinishedRun({ id: 'o', type: 'output', data: { label: 'Result' } }, false).data._stale, true);
  assert.equal(M.afterUnfinishedRun(text, false), text);
  assert.equal(M.afterUnfinishedRun(text, false, 't').data._status, 'error', 'the node the run said failed is marked, heard or not');
});

// ---------------------------------------------------------------------------
// 11. ⌘A and ⌘V on the canvas.
// ---------------------------------------------------------------------------

await check('a key is the canvas\'s on the canvas and on the builder around it (an empty part of the canvas clicked)', () => {
  const el = (inCanvas) => ({ closest: (sel) => (sel === '.react-flow' && inCanvas ? {} : null) });
  const builder = el(false);
  assert.equal(M.isCanvasTarget(builder, builder), true, 'the builder itself');
  assert.equal(M.isCanvasTarget(el(true), builder), true, 'inside the canvas');
  assert.equal(M.isCanvasTarget(el(false), builder), false, 'the log or the properties: their own copy');
  assert.equal(M.isCanvasTarget(null, builder), false);
});

// ---------------------------------------------------------------------------
// 14. A project file, checked before anything is asked.
// ---------------------------------------------------------------------------

await check("a file that is not a project is said so in words, not the parser's", () => {
  assert.throws(() => M.parseImportedProject('hello there'), { message: "This file isn't an OAIY project (it isn't valid JSON)." });
  assert.throws(() => M.parseImportedProject('{"name": "x"}'), { message: "This file isn't an OAIY project (it has no flows)." });
  assert.throws(() => M.parseImportedProject('null'), { message: "This file isn't an OAIY project (it has no flows)." });
  assert.throws(() => M.parseImportedProject('{"nodes": [], "edges": []}'), /a flow, not a project: add it with Import flows/);
  assert.throws(() => M.parseJsonFile('{', 'flow'), { message: "This file isn't an OAIY flow (it isn't valid JSON)." });
  const p = M.parseImportedProject(JSON.stringify({ name: 'Mine', flows: [{ id: 'f', name: 'F', graph: { nodes: [{ id: 'l', type: 'ai_llm', data: { service: 'engine:llm', ...OLLAMA } }], edges: [] } }] }));
  assert.equal(p.name, 'Mine');
  assert.deepEqual(p.flows[0].graph.nodes[0].data, { service: 'engine:llm' }, "and an imported AI LLM node's Ollama fields under its service are left out");
});

// ---------------------------------------------------------------------------
// 15. A toast says a step's error without its status.
// ---------------------------------------------------------------------------

await check('a toast says the message, or a sentence for a status alone; the status stays in the log', () => {
  assert.equal(M.toastError('Image Gen: HTTP 503 — The picture engine is not running'), 'Image Gen: The picture engine is not running');
  assert.equal(M.toastError('Error: HTTP 500: {"error":{"message":"model failed to load"}}'), 'model failed to load');
  assert.equal(M.toastError('Image Gen: HTTP 503'), 'Image Gen: the service is not ready');
  assert.equal(M.toastError('Music: HTTP 404'), 'Music: the service has nothing at that address');
  assert.equal(M.toastError('OAIY Desktop did not list its engine\'s services (HTTP 502)'), "OAIY Desktop did not list its engine's services");
  assert.equal(M.toastError('Nothing about status here'), 'Nothing about status here');
  assert.match(M.readableError('Image Gen: HTTP 503 — down'), /HTTP 503/, 'the queue and the log keep it');
});

// ---------------------------------------------------------------------------
// 8. An example on the models here; 16. cards kept clear of one that grew.
// ---------------------------------------------------------------------------

await check("an example's step whose model is not here takes the default of its kind, its name without the model", () => {
  const example = M.exampleFrom('../../../examples/flows/demo.flow.json', {
    name: 'Demo', nodes: [
      { id: 'make', type: 'image_gen', data: { label: 'Make it (Some model, 8 steps)', service: 'engine:image:not-here' } },
      { id: 'llm', type: 'ai_llm', data: { label: 'Enhance', service: 'engine:llm' } },
      { id: 'kept', type: 'image_gen', data: { label: 'Kept (here)', service: 'engine:image:here' } },
    ], edges: [],
  });
  const desktop = [
    { id: 'engine:image:here', kind: 'image', model: 'here', group: 'engine', default: true, nodeTypes: ['image_gen'], endpoint: 'x' },
    { id: 'engine:llm:x', kind: 'llm', model: 'x', group: 'engine', default: true, nodeTypes: ['ai_llm'], endpoint: 'y' },
  ];
  const env = { inOaiy: true, loaded: true, custom: [], desktop };
  const { flow, swapped } = M.withModelsHere(M.flowFromExample(example), env);
  assert.deepEqual(flow.graph.nodes[0].data, { label: 'Make it', service: 'engine:image' });
  assert.deepEqual(swapped, [{ step: 'Make it', kind: 'picture model' }]);
  assert.equal(flow.graph.nodes[2].data.service, 'engine:image:here', 'a model that is here stays');
  // nothing known yet, or outside OAIY: as it is
  assert.equal(M.withModelsHere(M.flowFromExample(example), { ...env, loaded: false }).swapped.length, 0);
  assert.equal(M.withModelsHere(M.flowFromExample(example), { ...env, inOaiy: false }).swapped.length, 0);
  // no picture model at all: as it is (its card says what is missing)
  assert.equal(M.withModelsHere(M.flowFromExample(example), { ...env, desktop: [desktop[1]] }).swapped.length, 0);
});

await check('a card that grew pushes the cards below it in its column down, and nothing else', () => {
  const cards = [
    { id: 'view', x: 0, y: 0, width: 230, height: 200 },
    { id: 'out', x: 10, y: 140, width: 230, height: 90 },
    { id: 'after', x: 0, y: 260, width: 230, height: 90 },
    { id: 'beside', x: 400, y: 140, width: 230, height: 90 },
  ];
  const moves = M.clearBelow(cards, 'view', 100);
  assert.deepEqual([...moves], [['out', 224], ['after', 344]]);
  assert.equal(M.clearBelow(cards, 'view', 200).size, 0, 'it did not grow');
  assert.equal(M.clearBelow([{ id: 'view', x: 0, y: 0, width: 230, height: 120 }, { id: 'far', x: 0, y: 400, width: 230, height: 90 }], 'view', 100).size, 0, 'still clear');
});

engine.destroy?.();
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
