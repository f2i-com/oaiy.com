/**
 * Which engine a workflow actually reaches.
 *
 *     npm run test:zipp-routing
 *
 * `zipp-executor.mjs` proves the Zipp engine can run OAIY's compiled script.
 * This proves the runtime HANDS it that script — that a package (hardened)
 * workflow is routed to the injected Worker factory with the hardening flags
 * intact, that a trusted local flow is never sent there, and that the two ways
 * the Worker path can be unavailable both fail closed rather than quietly
 * downgrading to in-thread execution in the host realm.
 *
 * It drives the real `OAIYRuntime`. oaiy-core is TypeScript with `.js`-suffixed
 * imports that only a bundler resolves, so the runtime is bundled for Node with
 * esbuild (already a dependency via Vite) into a temp file and imported from
 * there. The Worker itself is a fake that speaks `untrusted-executor.ts`'s
 * protocol — what is under test is the routing, not the engine.
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
function check(name, ok, detail = '') {
  if (ok) {
    pass++;
    console.log(`  ok  ${name}`);
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`);
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

// ---------------------------------------------------------------------------
// Bundle the real runtime for Node.
// ---------------------------------------------------------------------------
const bundlePath = path.join(os.tmpdir(), `oaiy-zipp-routing-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { createRuntime } from './vendor/oaiy-core/src/runtime.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  alias: { 'oaiy-core': path.join(UI, 'vendor/oaiy-core/src/index.ts') },
});
const { createRuntime } = await import(pathToFileURL(bundlePath).href);

// ---------------------------------------------------------------------------
// A Worker that speaks the untrusted-workflow protocol, and records what it
// was handed.
// ---------------------------------------------------------------------------
class FakeWorker {
  constructor(log) {
    this.log = log;
    this.listeners = new Map();
    this.terminated = false;
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  removeEventListener() {}
  emit(type, data) {
    for (const fn of this.listeners.get(type) ?? []) fn({ data });
  }
  postMessage(msg) {
    this.log.posted.push(msg);
    if (msg?.type === 'init') {
      // Answer the way a real engine would once the script has run: with a
      // finish carrying whatever the flow returned. Asynchronously, like a
      // real Worker, so the runtime's listener is attached first.
      queueMicrotask(() => this.emit('message', { type: 'finish', value: { engine: 'fake-worker' } }));
    }
  }
  terminate() {
    this.terminated = true;
  }
}

/** A factory that records every call and hands out a fresh FakeWorker. */
function spyFactory() {
  const log = { calls: 0, posted: [], cleanedUp: 0, workers: [] };
  const factory = () => {
    log.calls++;
    const worker = new FakeWorker(log);
    log.workers.push(worker);
    return { worker, cleanup: () => { log.cleanedUp++; worker.terminate(); } };
  };
  return { factory, log };
}

/**
 * The runtime only takes the Worker path when the host has `Worker`, `Blob`
 * and `URL.createObjectURL`. Node 24 has the latter two; give it the first.
 */
function withWorkerGlobal(present, fn) {
  const had = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  if (present) globalThis.Worker = class {};
  else delete globalThis.Worker;
  return fn().finally(() => {
    if (had) Object.defineProperty(globalThis, 'Worker', had);
    else delete globalThis.Worker;
  });
}

const run = (runtime, script) => runtime.executeScript(script);

// ---------------------------------------------------------------------------
// 1. A package workflow goes to the injected factory, hardened.
// ---------------------------------------------------------------------------
console.log('\npackage (hardened) workflow');
await withWorkerGlobal(true, async () => {
  const { factory, log } = spyFactory();
  const runtime = createRuntime({ untrustedWorkerFactory: factory });
  runtime.setFlowContext('flow-1', 'pkg-1');

  const result = await run(runtime, 'let workflow_context = { from: "script" };');
  const init = log.posted.find(m => m?.type === 'init');

  check('the injected factory was used exactly once', log.calls === 1, `calls=${log.calls}`);
  check('the result came back from the Worker', result?.engine === 'fake-worker', JSON.stringify(result));
  check('init message carries hardened=true', init?.hardened === true, JSON.stringify(init && { hardened: init.hardened }));
  check("init script keeps 'use strict' at the top", typeof init?.script === 'string' && /^\s*'use strict';/.test(init.script));
  check('init script carries the shadow preamble (fetch)', /var fetch = undefined;/.test(init?.script ?? ''));
  check('init script carries the shadow preamble (postMessage)', /var postMessage = undefined;/.test(init?.script ?? ''));
  check('init script carries the generator trampoline', /function\* __run_workflow\(\)/.test(init?.script ?? '') && /__stepThrow/.test(init?.script ?? ''));
  check('the Worker was cleaned up after finish', log.cleanedUp === 1 && log.workers[0]?.terminated === true, `cleanedUp=${log.cleanedUp}`);
});

// ---------------------------------------------------------------------------
// 2. A trusted local flow never touches the factory.
// ---------------------------------------------------------------------------
console.log('\ntrusted local workflow');
await withWorkerGlobal(true, async () => {
  const { factory, log } = spyFactory();
  const runtime = createRuntime({ untrustedWorkerFactory: factory });
  runtime.setFlowContext('flow-2', null);

  const result = await run(runtime, 'let workflow_context = { engine: "in-thread" };');

  check('the factory was never called', log.calls === 0, `calls=${log.calls}`);
  check('the flow ran in-thread on the host engine', result?.engine === 'in-thread', JSON.stringify(result));
});

// ---------------------------------------------------------------------------
// 3. Fail closed: no Worker in this host.
// ---------------------------------------------------------------------------
// A hardened flow that cannot get a Worker must be refused, never run
// in-thread — on a Node host a recovered realm global is `process`/`require`.
console.log('\nfail closed');
await withWorkerGlobal(false, async () => {
  const { factory, log } = spyFactory();
  const runtime = createRuntime({ untrustedWorkerFactory: factory });
  runtime.setFlowContext('flow-3', 'pkg-3');

  let error = null;
  try { await run(runtime, 'let workflow_context = { leaked: true };'); }
  catch (e) { error = e; }

  check('a hardened flow without a Worker is refused', !!error && /Refusing to run untrusted package code in-thread/.test(String(error?.message ?? error)), String(error?.message ?? error).slice(0, 80));
  check('the factory was not consulted', log.calls === 0, `calls=${log.calls}`);
});

// ---------------------------------------------------------------------------
// 4. Fail closed: the factory itself breaks.
// ---------------------------------------------------------------------------
await withWorkerGlobal(true, async () => {
  const runtime = createRuntime({
    untrustedWorkerFactory: () => { throw new Error('engine unavailable'); },
  });
  runtime.setFlowContext('flow-4', 'pkg-4');

  let error = null;
  try { await run(runtime, 'let workflow_context = { leaked: true };'); }
  catch (e) { error = e; }

  check('a factory that throws fails the run instead of downgrading', !!error && /engine unavailable/.test(String(error?.message ?? error)), String(error?.message ?? error).slice(0, 80));
});

// ---------------------------------------------------------------------------
fs.rmSync(bundlePath, { force: true });
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
assert.equal(failures.length, 0);
