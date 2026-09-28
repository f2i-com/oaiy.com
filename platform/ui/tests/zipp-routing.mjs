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
    contents: `export { createRuntime } from './vendor/oaiy-core/src/runtime.ts';
export { JobManager } from './vendor/oaiy-core/src/queue/JobManager.ts';`,
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
const { createRuntime, JobManager } = await import(pathToFileURL(bundlePath).href);

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
// 2b. With runTrustedFlowsInWorker, a trusted flow goes to the factory too —
//     unhardened, so the engine applies no shadow preamble and no strict mode.
// ---------------------------------------------------------------------------
console.log('\ntrusted workflow, Worker requested');
await withWorkerGlobal(true, async () => {
  const { factory, log } = spyFactory();
  const runtime = createRuntime({ untrustedWorkerFactory: factory, runTrustedFlowsInWorker: true });
  runtime.setFlowContext('flow-2b', null);

  const result = await run(runtime, 'let workflow_context = { from: "script" };');
  const init = log.posted.find(m => m?.type === 'init');

  check('the factory was used for the trusted flow', log.calls === 1, `calls=${log.calls}`);
  check('the result came back from the Worker', result?.engine === 'fake-worker', JSON.stringify(result));
  check('init message carries hardened=false', init?.hardened === false, JSON.stringify(init && { hardened: init.hardened }));
  check("init script has no 'use strict' (matches the in-thread path)", !/^\s*'use strict';/.test(init?.script ?? ''));
  check('init script has no shadow preamble', !/var fetch = undefined;/.test(init?.script ?? ''));
  check('init script still carries the trampoline', /function\* __run_workflow\(\)/.test(init?.script ?? ''));
});
await withWorkerGlobal(true, async () => {
  // The flag without an engine is a no-op: never the default Blob Worker.
  const runtime = createRuntime({ runTrustedFlowsInWorker: true });
  runtime.setFlowContext('flow-2c', null);
  const result = await run(runtime, 'let workflow_context = { engine: "in-thread" };');
  check('the flag alone (no factory) keeps a trusted flow in-thread', result?.engine === 'in-thread', JSON.stringify(result));
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
// 5. The host's script engine (`scriptExecutor`): every flow, trusted or
//    hardened, runs on it — and it does not need a `Worker` global, so a Node
//    host can supply one. `requireScriptExecutor` refuses to run flow code
//    when none is set instead of falling back to in-thread `new Function`.
// ---------------------------------------------------------------------------
console.log('\nhost script engine (scriptExecutor)');

// 5a. A trusted flow goes to the executor, unhardened, with no Worker global.
await withWorkerGlobal(false, async () => {
  const { factory: scriptExecutor, log } = spyFactory();
  const runtime = createRuntime({ scriptExecutor });
  runtime.setFlowContext('flow-5a', null);

  const result = await run(runtime, 'let workflow_context = { engine: "v8" };');
  const init = log.posted.find(m => m?.type === 'init');

  check('a trusted flow runs on the executor without a Worker global', result?.engine === 'fake-worker', JSON.stringify(result));
  check('the executor was used exactly once', log.calls === 1, `calls=${log.calls}`);
  check('init message carries hardened=false', init?.hardened === false, JSON.stringify(init && { hardened: init.hardened }));
  check("init script has no 'use strict'", !/^\s*'use strict';/.test(init?.script ?? ''));
  check('the executor was cleaned up after finish', log.cleanedUp === 1 && log.workers[0]?.terminated === true, `cleanedUp=${log.cleanedUp}`);
});

// 5b. A hardened flow goes to the executor with hardened=true, still with no
//     Worker global; the escape scan still runs before the executor is asked.
await withWorkerGlobal(false, async () => {
  const { factory: scriptExecutor, log } = spyFactory();
  const runtime = createRuntime({ scriptExecutor });
  runtime.setFlowContext('flow-5b', 'pkg-5b');

  let result = null;
  let error = null;
  try { result = await run(runtime, 'let workflow_context = { from: "script" };'); }
  catch (e) { error = e; }
  const init = log.posted.find(m => m?.type === 'init');

  check('a hardened flow runs on the executor without a Worker global', !error && result?.engine === 'fake-worker', error ? String(error?.message ?? error).slice(0, 80) : JSON.stringify(result));
  check('init message carries hardened=true', init?.hardened === true, JSON.stringify(init && { hardened: init.hardened }));
  check("init script keeps 'use strict' at the top", typeof init?.script === 'string' && /^\s*'use strict';/.test(init.script));
});
await withWorkerGlobal(false, async () => {
  const { factory: scriptExecutor, log } = spyFactory();
  const runtime = createRuntime({ scriptExecutor });
  runtime.setFlowContext('flow-5b2', 'pkg-5b');

  let error = null;
  try { await run(runtime, 'let workflow_context = globalThis;'); }
  catch (e) { error = e; }

  check('a hardened escape probe is still refused before the executor', !!error && /sandbox-escape pattern/.test(String(error?.message ?? error)), String(error?.message ?? error).slice(0, 80));
  check('the executor was not consulted for the refused probe', log.calls === 0, `calls=${log.calls}`);
});

// 5c. requireScriptExecutor without an executor refuses to run at all.
await withWorkerGlobal(false, async () => {
  const runtime = createRuntime({ requireScriptExecutor: true });
  runtime.setFlowContext('flow-5c', null);

  let error = null;
  let result = null;
  try { result = await run(runtime, 'let workflow_context = { leaked: true };'); }
  catch (e) { error = e; }

  check('a required-but-missing executor refuses the run', !!error && /Refusing to run flow code/.test(String(error?.message ?? error)), String(error?.message ?? error).slice(0, 80));
  check('the probe never ran in-thread', result?.leaked !== true, JSON.stringify(result));
});
await withWorkerGlobal(false, async () => {
  // Same for a hardened flow: the refusal comes before any Worker decision.
  const runtime = createRuntime({ requireScriptExecutor: true });
  runtime.setFlowContext('flow-5c2', 'pkg-5c');

  let error = null;
  try { await run(runtime, 'let workflow_context = { leaked: true };'); }
  catch (e) { error = e; }

  check('a hardened flow is refused with the executor message, not the Worker one', !!error && /Refusing to run flow code/.test(String(error?.message ?? error)), String(error?.message ?? error).slice(0, 80));
});

// 5d. An executor that throws fails the run and never downgrades.
await withWorkerGlobal(false, async () => {
  const runtime = createRuntime({
    scriptExecutor: () => { throw new Error('engine unavailable'); },
    requireScriptExecutor: true,
  });
  runtime.setFlowContext('flow-5d', null);

  let error = null;
  let result = null;
  try { result = await run(runtime, 'let workflow_context = { leaked: true };'); }
  catch (e) { error = e; }

  check('an executor that throws fails the run instead of downgrading', !!error && /engine unavailable/.test(String(error?.message ?? error)), String(error?.message ?? error).slice(0, 80));
  check('the flow never ran in-thread after the executor failed', result?.leaked !== true, JSON.stringify(result));
});

// 5e. With both a Worker factory and an executor set, only the executor runs.
await withWorkerGlobal(true, async () => {
  const { factory, log: factoryLog } = spyFactory();
  const { factory: scriptExecutor, log: execLog } = spyFactory();
  const runtime = createRuntime({ untrustedWorkerFactory: factory, runTrustedFlowsInWorker: true, scriptExecutor });
  runtime.setFlowContext('flow-5e', null);

  const result = await run(runtime, 'let workflow_context = { from: "script" };');

  check('the Worker factory was not consulted when an executor is set', factoryLog.calls === 0, `calls=${factoryLog.calls}`);
  check('the executor was used exactly once', execLog.calls === 1, `calls=${execLog.calls}`);
  check('the result came back from the executor', result?.engine === 'fake-worker', JSON.stringify(result));
});

// 5f. JobManager applies the refusal to every job's runtime even when no
//     Worker factory was passed (the factory-gated block must not gate it).
await withWorkerGlobal(false, async () => {
  const manager = new JobManager({ databaseHandler: async () => ({}), requireScriptExecutor: true });
  const terminal = new Set(['completed', 'failed', 'aborted']);
  let jobId = null;
  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('job did not reach a terminal state in 5s')), 5000);
    const unsubscribe = manager.onStateChange(() => {
      const job = jobId ? manager.getJob(jobId) : null;
      if (job && terminal.has(job.status)) {
        clearTimeout(timer);
        unsubscribe();
        resolve(job);
      }
    });
  });
  jobId = manager.submit('flow-5f', 'flow 5f', { nodes: [], edges: [] });

  let job = null;
  let error = null;
  try { job = await done; }
  catch (e) { error = e; }
  if (!job) job = manager.getJob(jobId);

  check('the job reached a terminal state', !error, String(error?.message ?? ''));
  check('a JobManager job without an executor fails when one is required', job?.status === 'failed', `status=${job?.status}`);
  check('the job error is the executor refusal', /Refusing to run flow code/.test(job?.error ?? ''), String(job?.error ?? '').slice(0, 80));
});

// ---------------------------------------------------------------------------
fs.rmSync(bundlePath, { force: true });
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
assert.equal(failures.length, 0);
