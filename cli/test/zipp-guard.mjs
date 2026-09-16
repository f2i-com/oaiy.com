// Guards B, C and D: the built CLI runs flow code on the ZIPP VM and nowhere
// else, and can tell when it does not.
//
//     node test/zipp-guard.mjs [path/to/oaiy.mjs]
//
// Spawns dist/oaiy.mjs (build first), like zipp-cli.mjs and cli-http-exit.mjs.
//
//   B  The canary flow (fixtures/zipp-canary.json — every logic_block compile
//      branch, a condition expression, a loop, a subflow, a macro, the
//      Desktop's app-logic wrapper shape) runs to completion under
//      guards/host-eval-tripwire.mjs, which counts every Function / eval /
//      node:vm call in the main thread AND in every worker (`--import` is
//      inherited through execArgv; the report is keyed by threadId). Count: 0.
//   C  What the canary's `probe` block sees of the process: nothing — all five
//      typeofs 'undefined', including `Function('return this')().process`;
//      the condition `typeof process === 'undefined'` takes its true branch;
//      OAIY_SERVER_TOKEN, set in the CLI's environment, reaches neither stream
//      nor the result. A tight `while (true) {}` is stopped by the instruction
//      budget in under ten seconds; with the budget at its 2e9 ceiling,
//      `--timeout 3` aborts the run (errorCode `timeout`) in under six.
//   D  A missing artifact (OAIY_ZIPP_ASSET_DIR at an empty folder) and one
//      whose bytes differ by one bit are `engine_unavailable`, exit 1, before
//      any job exists. `capabilities --json` says `ready` and agrees with the
//      staged SOURCE.json and PROFILE.json (its figures are the staged file's,
//      not literals), says `unavailable` with a reason at the empty folder, and
//      `--instruction-budget` out of range is refused before the flow is read.
//
// Non-vacuity — the part that makes the rest evidence rather than a tautology.
// The tripwire is first shown to count (a script that calls new Function, eval
// and a recovered AsyncFunction constructor: 3). Then the CLI is built ONCE
// MORE from the same sources with the build-time define
// `__OAIY_TEST_ALLOW_V8__: 'true'` (src/engine.ts; `false` in every bundle
// cli/esbuild.mjs makes, and folded away — this file checks the production
// bundle carries neither the name nor `requireScriptExecutor: false`). That
// bundle runs flows on the host engine. The SAME canary on it reports
// `typeof process === 'object'`, READS the token, and trips the tripwire — so
// the production bundle's `'undefined'`, `no-process` and 0 are what the
// fixture and the tripwire say when a flow does run in the host realm, not
// what they say regardless.
import assert from 'node:assert/strict';
import esbuild from 'esbuild';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath, pathToFileURL } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const distDir = path.dirname(cli);
const distZipp = path.join(distDir, 'zipp');
const cliRoot = fileURLToPath(new URL('../', import.meta.url));
const tripwire = fileURLToPath(new URL('./guards/host-eval-tripwire.mjs', import.meta.url));
const canary = fileURLToPath(new URL('./fixtures/zipp-canary.json', import.meta.url));
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'oaiy-zipp-guard-'));

const CANARY = 'canary-secret';
const MARK = '__host_eval_tripwire__';

/** Spawn node with `args`; the child's environment is this one plus `env`. */
async function node(args, env = {}, killAfterMs = 30_000) {
  const child = spawn(process.execPath, args, {
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, ...env },
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d) => { stdout += d; });
  child.stderr.on('data', (d) => { stderr += d; });
  const started = Date.now();
  const timer = setTimeout(() => child.kill(), killAfterMs);
  let code, signal;
  try {
    [code, signal] = await once(child, 'close');
  } finally {
    clearTimeout(timer);
  }
  return { code, signal, stdout, stderr, elapsed: Date.now() - started };
}

/** `oaiy run <flow> --quiet …` on `bundle`, optionally under the tripwire. */
function run(bundle, flow, args, env, { tripwire: withTripwire = false, killAfterMs } = {}) {
  const pre = withTripwire ? ['--import', pathToFileURL(tripwire).href] : [];
  return node([...pre, bundle, 'run', flow, '--quiet', ...args], env, killAfterMs);
}

/** The tripwire's report lines in `stderr`, parsed. */
function tripwireReport(stderr) {
  const lines = stderr.split('\n').filter((l) => l.startsWith(MARK + ' '));
  const parse = (kind) => lines.filter((l) => l.startsWith(`${MARK} ${kind} `)).map((l) => JSON.parse(l.slice(MARK.length + kind.length + 2)));
  return { loaded: parse('loaded'), calls: parse('call'), summary: parse('summary') };
}

async function writeFlow(name, code) {
  const file = path.join(dir, name);
  await fs.writeFile(file, JSON.stringify({
    nodes: [
      { id: 'b', type: 'logic_block', position: { x: 0, y: 0 }, data: { code } },
      { id: 'out', type: 'output', position: { x: 200, y: 0 }, data: { value: '$nodes.b' } },
    ],
    edges: [{ id: 'edge', source: 'b', target: 'out' }],
  }));
  return file;
}

const UNDEFINED5 = ['undefined', 'undefined', 'undefined', 'undefined', 'undefined'];

/**
 * What every node of the canary must have produced. The engine-independent
 * values are the same on both engines; the condition `typeof process ===
 * 'undefined'` is the one node whose answer IS the engine.
 */
function assertCanaryValues(results, { onZipp }) {
  assert.equal(results.expr, 2, 'branch 1: a parenthesised expression');
  assert.equal(results.ret_await, 'dflt!', 'branch 2: return + await');
  assert.equal(results.ret, 42, 'branch 3: return, run as a function');
  assert.deepEqual(results.single, [2, 4, 6], 'branch 4: a single expression');
  assert.equal(results.await_noret, 'stored', 'branch 5: await, no return');
  assert.equal(results.plain, 'a-b', 'branch 6: statements, no return');
  assert.equal(results.cond, onZipp, `the condition expression (typeof process === 'undefined') answered ${results.cond}`);
  // Downstream of the true handle: runs when the condition holds, is skipped
  // (no output at all) when it does not.
  assert.equal(results.branch, onZipp ? 'took-true' : undefined, 'the node behind the true handle');
  assert.deepEqual(results.loop_end, [10, 20, 30], 'the count loop ran three times with loop_index');
  assert.equal(results.sub?.__output__, 'sub:2', 'the subflow ran on its own script and returned');
  assert.deepEqual(results.mac, { r: 43 }, 'the macro ran and mapped its output');
  assert.equal(results.applogic, 3, "the Desktop's (new Function('ctx', decodeURIComponent(…)))(…) shape runs");
}

try {
  // --- B0. The tripwire counts ------------------------------------------
  const trips = path.join(dir, 'trips.mjs');
  await fs.writeFile(trips, [
    "new Function('return 1')();",
    "(0, eval)('1 + 1');",
    "(async function () {}).constructor('return 2');",
    // The logger probe, as oaiy-core/src/logger.ts makes it (it throws outside a module): not counted.
    "try { new Function('try { return import.meta } catch(e) { return undefined }')(); } catch {}",
  ].join('\n') + '\n');
  const armed = await node(['--import', pathToFileURL(tripwire).href, trips]);
  assert.equal(armed.code, 0, `the tripwire broke the program it watched: ${armed.stderr}`);
  const armedReport = tripwireReport(armed.stderr);
  assert.deepEqual(armedReport.calls.map((c) => c.what), ['new Function', 'eval()', 'AsyncFunction()'],
    `the tripwire counted ${JSON.stringify(armedReport.calls)}`);
  assert.deepEqual(armedReport.summary, [{ threadId: 0, count: 3 }]);
  console.log('PASS: the tripwire counts new Function, eval and a recovered AsyncFunction, and skips the logger probe');

  // --- B0b. The two ways a string becomes code WITHOUT asking Function ----
  // A review walked both past this guard. A `worker_threads` Worker with
  // `{ eval: true }` compiles its source in another thread, which never asks
  // this process for `Function`; `process.binding('contextify')` hands back
  // the compiler itself. Both are now recorded — and an ordinary Worker, by
  // path, still is NOT, which is what keeps the canary's "0 calls" meaningful,
  // since the CLI spawns its own ZIPP and script workers exactly that way.
  const trips2 = path.join(dir, 'trips2.mjs');
  const plainWorker = path.join(dir, 'plain-worker.mjs');
  await fs.writeFile(plainWorker, 'process.exit(0);\n');
  await fs.writeFile(trips2, [
    "import { Worker } from 'node:worker_threads';",
    "await new Promise((r) => { const w = new Worker('1 + 1', { eval: true }); w.on('exit', r); });",
    `await new Promise((r) => { const w = new Worker(${JSON.stringify(plainWorker)}); w.on('exit', r); });`,
    "try { process.binding('contextify'); } catch {}",
  ].join('\n') + '\n');
  const armed2 = await node(['--import', pathToFileURL(tripwire).href, trips2]);
  assert.equal(armed2.code, 0, `the tripwire broke the program it watched: ${armed2.stderr}`);
  const report2 = tripwireReport(armed2.stderr);
  const main2 = report2.calls.filter((c) => c.threadId === 0).map((c) => c.what);
  assert.deepEqual(main2, ['new Worker({ eval: true })', 'process.binding()'],
    `the tripwire counted ${JSON.stringify(report2.calls)}`);
  console.log('PASS: the tripwire counts a worker_threads Worker with eval: true and process.binding, and not a Worker spawned by path');

  // --- B + C. The canary, under the tripwire, with the token in the env ---
  const outFile = path.join(dir, 'canary.json');
  const res = await run(cli, canary, ['--timeout', '30', '-o', outFile], { OAIY_SERVER_TOKEN: CANARY }, { tripwire: true, killAfterMs: 60_000 });
  assert.equal(res.signal, null, `the canary run did not finish: ${res.stderr}`);
  assert.equal(res.code, 0, `the canary run failed: ${res.stderr}`);
  const payload = JSON.parse(await fs.readFile(outFile, 'utf8'));
  assert.equal(payload.success, true);
  assert.equal(payload.status, 'completed');
  assert.equal(payload.engine, 'zipp', 'the payload names the engine');
  assert.equal(payload.errorCode, undefined);
  assertCanaryValues(payload.results, { onZipp: true });
  assert.deepEqual(payload.results.probe.types, UNDEFINED5, 'process, require, Buffer, __TAURI__ and Function("return this")().process are all undefined');
  assert.equal(payload.results.probe.token, 'no-process');
  assert.deepEqual(payload.output, payload.results.probe, 'the output node carries the probe');
  for (const [where, text] of [['stdout', res.stdout], ['stderr', res.stderr], ['the result file', JSON.stringify(payload)]]) {
    assert.doesNotMatch(text, new RegExp(CANARY), `the token reached ${where}`);
  }
  const report = tripwireReport(res.stderr);
  const threads = new Set(report.loaded.map((l) => l.threadId));
  assert.ok(threads.has(0), 'the tripwire loaded in the main thread');
  assert.ok([...threads].some((t) => t !== 0), `--import did not reach a worker thread (threads seen: ${[...threads].join(',')})`);
  assert.deepEqual(report.calls, [], `host code compilation during the canary: ${JSON.stringify(report.calls)}`);
  console.log(`PASS: the canary ran every compile branch, condition, loop, subflow, macro and app-logic shape on ZIPP with 0 host Function/eval/vm calls across ${threads.size} threads, and the token reached nothing`);

  // --- C. A tight loop is stopped by the budget -----------------------------
  // Two statements: a lone `while (true) {}` is one line without a `;`, which
  // the logic_block compiler takes for an expression and inlines as one.
  const spin = await writeFlow('spin.json', 'let n = 0;\nwhile (true) { n++; }');
  const spun = await run(cli, spin, ['--timeout', '60'], {}, { killAfterMs: 30_000 });
  assert.equal(spun.signal, null, `the spinning run did not finish: ${spun.stderr}`);
  assert.equal(spun.code, 1);
  const spunPayload = JSON.parse(spun.stdout);
  assert.equal(spunPayload.status, 'failed');
  assert.match(spunPayload.error ?? '', /budget/i, `not a budget failure: ${spunPayload.error}`);
  assert.equal(spunPayload.errorCode, undefined, "the flow's own failure has no host code");
  assert.ok(spun.elapsed < 10_000, `the budget took ${spun.elapsed} ms to stop the loop`);
  console.log(`PASS: while (true) {} is stopped by the instruction budget in ${spun.elapsed} ms`);

  // --- C. At the ceiling, --timeout still aborts a busy VM --------------------
  const timed = await run(cli, spin, ['--instruction-budget', '2000000000', '--timeout', '3'], {}, { killAfterMs: 20_000 });
  assert.equal(timed.signal, null, `the CLI did not exit on its own after the timeout: ${timed.stderr}`);
  assert.equal(timed.code, 1);
  const timedPayload = JSON.parse(timed.stdout);
  assert.equal(timedPayload.status, 'aborted');
  assert.equal(timedPayload.errorCode, 'timeout');
  assert.ok(timed.elapsed < 6_000, `the abort took ${timed.elapsed} ms`);
  assert.doesNotMatch(timed.stderr, /Assertion failed/);
  console.log(`PASS: --instruction-budget 2000000000 --timeout 3 aborts a busy VM in ${timed.elapsed} ms with errorCode timeout`);

  // --- D. No artifact / a corrupt artifact: engine_unavailable, nothing ran -------
  const trivial = await writeFlow('trivial.json', 'return 1;');
  const empty = path.join(dir, 'empty');
  await fs.mkdir(empty);
  const flipped = path.join(dir, 'flipped');
  await fs.mkdir(flipped);
  for (const name of await fs.readdir(distZipp)) {
    await fs.copyFile(path.join(distZipp, name), path.join(flipped, name));
  }
  const wasm = path.join(flipped, 'zipp_wasm_bg.wasm');
  const bytes = await fs.readFile(wasm);
  bytes[Math.floor(bytes.length / 3)] ^= 0x01;
  await fs.writeFile(wasm, bytes);
  for (const [label, assetDir, why] of [['a missing artifact', empty, /missing|cannot be read/], ['a corrupt artifact', flipped, /sha256/]]) {
    const r = await run(cli, trivial, [], { OAIY_ZIPP_ASSET_DIR: assetDir });
    assert.equal(r.signal, null, `${label}: the CLI did not finish: ${r.stderr}`);
    assert.equal(r.code, 1, `${label}: exit ${r.code}: ${r.stderr}`);
    const p = JSON.parse(r.stdout);
    assert.equal(p.success, false);
    assert.equal(p.status, 'failed');
    assert.equal(p.errorCode, 'engine_unavailable', `${label}: ${p.error}`);
    assert.equal(p.jobId, '', `${label}: a job was submitted`);
    assert.deepEqual(p.results, {}, `${label}: something ran`);
    assert.match(p.error ?? '', why, `${label}: ${p.error}`);
    assert.equal(p.engine, 'zipp');
  }
  console.log('PASS: a missing or altered engine artifact is engine_unavailable, exit 1, and no job exists');

  // --- D. capabilities --json: ready, and honest about the staged files ---------
  const stagedSource = JSON.parse(await fs.readFile(path.join(distZipp, 'SOURCE.json'), 'utf8'));
  const stagedProfile = JSON.parse(await fs.readFile(path.join(distZipp, 'PROFILE.json'), 'utf8'));
  const pkg = JSON.parse(await fs.readFile(path.join(cliRoot, 'package.json'), 'utf8'));
  const caps = await node([cli, 'capabilities', '--json']);
  assert.equal(caps.code, 0, `capabilities failed: ${caps.stderr}`);
  const report0 = JSON.parse(caps.stdout);
  assert.equal(report0.version, pkg.version, "version is package.json's");
  assert.deepEqual(report0.protocols, { run: 1, script: 1, profile: 1 });
  assert.deepEqual(report0.script.languages, stagedSource.languages, 'script.languages is what the leaf envelope runs on this engine');
  assert.ok(Number.isInteger(report0.script.defaultBudgetMs) && report0.script.defaultBudgetMs > 0);
  assert.ok(Number.isInteger(report0.script.maxBudgetMs) && report0.script.maxBudgetMs >= report0.script.defaultBudgetMs);
  assert.equal(report0.engine.status, 'ready');
  assert.equal(report0.engine.reason, undefined);
  assert.equal(report0.engine.name, 'zipp');
  assert.equal(report0.engine.wasmSha256, stagedSource.sha256, 'engine.wasmSha256 is the staged SOURCE.json digest');
  assert.equal(report0.engine.release, stagedSource.release);
  assert.equal(report0.engine.revision, stagedSource.revision);
  assert.deepEqual(report0.engine.languages, stagedSource.languages, 'engine.languages is what the bundle can run');
  assert.deepEqual(report0.run.languages, ['javascript'], 'run.languages is what THIS host runs');
  assert.equal(report0.run.defaultInstructionSteps, stagedProfile.limits.lifetimeSteps, 'the default budget is PROFILE lifetimeSteps');
  assert.equal(report0.run.maxInstructionSteps, stagedProfile.limits.maxInstructionBudgetSteps, 'the ceiling is PROFILE maxInstructionBudgetSteps');
  const capsMissing = await node([cli, 'capabilities', '--json'], { OAIY_ZIPP_ASSET_DIR: empty });
  assert.equal(capsMissing.code, 1, 'an unavailable engine is a failed probe');
  const reportMissing = JSON.parse(capsMissing.stdout);
  assert.equal(reportMissing.engine.status, 'unavailable');
  assert.match(reportMissing.engine.reason ?? '', /engine unavailable/i);
  assert.equal(reportMissing.engine.wasmSha256, stagedSource.sha256, "the identity is still the build's");
  assert.deepEqual(reportMissing.run, report0.run, 'what the host runs does not change with the artifact');
  console.log(`PASS: capabilities --json is ready and agrees with the staged SOURCE.json and PROFILE.json (${report0.run.defaultInstructionSteps} / ${report0.run.maxInstructionSteps}), and says unavailable with a reason at an empty folder`);

  // --- D. --instruction-budget out of range is refused before anything runs ------
  const ceiling = stagedProfile.limits.maxInstructionBudgetSteps;
  for (const bad of ['0', String(ceiling + 1), '1.5', 'abc']) {
    const refused = await node([cli, 'run', path.join(dir, 'does-not-exist.json'), '--instruction-budget', bad]);
    assert.equal(refused.code, 1, `--instruction-budget ${bad} was not refused`);
    assert.equal(refused.stdout, '', `--instruction-budget ${bad}: something was printed to stdout`);
    assert.match(refused.stderr, new RegExp(`--instruction-budget must be an integer from 1 to ${ceiling}`), `--instruction-budget ${bad}: ${refused.stderr}`);
    assert.doesNotMatch(refused.stderr, /ENOENT|does-not-exist/, `--instruction-budget ${bad}: the flow file was read before the option was refused`);
  }
  console.log('PASS: --instruction-budget 0, above the ceiling, fractional and non-numeric are refused as usage errors before the flow is read');

  // --- D (non-vacuity). The production bundle has no host-engine branch ---------
  const production = await fs.readFile(cli, 'utf8');
  assert.doesNotMatch(production, /__OAIY_TEST_ALLOW_V8__/, 'the production bundle still names the test-only define');
  assert.doesNotMatch(production, /requireScriptExecutor:\s*false/, 'the production bundle can build an un-required runtime');
  assert.match(production, /requireScriptExecutor:\s*true/, 'the production bundle requires the executor');

  // --- D (non-vacuity). The same canary on a host-engine build reads the token -----
  // Built here, from the same sources and build options, with the define
  // flipped at BUILD time. Nothing at run time selects this path.
  const { commonBuildOptions } = await import('../esbuild.mjs');
  const v8Bundle = path.join(distDir, '_v8_oaiy.mjs');
  await esbuild.build({
    ...commonBuildOptions,
    define: { ...commonBuildOptions.define, __OAIY_TEST_ALLOW_V8__: 'true' },
    entryPoints: [path.join(cliRoot, 'src', 'cli.ts')],
    outfile: v8Bundle,
    logLevel: 'warning',
  });
  const v8Text = await fs.readFile(v8Bundle, 'utf8');
  assert.match(v8Text, /requireScriptExecutor:\s*false/, 'the test bundle did not take the host-engine branch');
  const v8Out = path.join(dir, 'v8-canary.json');
  const v8 = await run(v8Bundle, canary, ['--timeout', '30', '-o', v8Out], { OAIY_SERVER_TOKEN: CANARY }, { tripwire: true, killAfterMs: 60_000 });
  assert.equal(v8.signal, null, `the host-engine canary did not finish: ${v8.stderr}`);
  assert.equal(v8.code, 0, `the host-engine canary failed: ${v8.stderr}`);
  const v8Payload = JSON.parse(await fs.readFile(v8Out, 'utf8'));
  assertCanaryValues(v8Payload.results, { onZipp: false });
  assert.equal(v8Payload.results.probe.types[0], 'object', `on the host engine the canary must see process; saw ${JSON.stringify(v8Payload.results.probe.types)}`);
  assert.equal(v8Payload.results.probe.token, CANARY, 'on the host engine the canary must read the token');
  const v8Report = tripwireReport(v8.stderr);
  // Three scripts compiled in the host realm (entry, subflow, macro), and the
  // flow's OWN `Function('return this')` and `new Function('ctx', …)` reaching
  // the host's Function — the same two calls that count 0 on ZIPP above,
  // because there they reach the guest's.
  const scripts = v8Report.calls.filter((c) => c.what === 'new Function' && c.head.startsWith('host,console,'));
  assert.equal(scripts.length, 3, `expected the entry, subflow and macro scripts to be compiled in the host realm: ${JSON.stringify(v8Report.calls)}`);
  assert.ok(v8Report.calls.some((c) => c.what === 'Function()' && c.head === 'return this'),
    `the probe's Function('return this') did not reach the host's Function: ${JSON.stringify(v8Report.calls)}`);
  assert.ok(v8Report.calls.some((c) => c.what === 'new Function' && c.head.startsWith('ctx,')),
    `the app-logic block's new Function('ctx', …) did not reach the host's Function: ${JSON.stringify(v8Report.calls)}`);
  console.log(`PASS: the same canary on a build with __OAIY_TEST_ALLOW_V8__ sees typeof process === 'object', reads the token, and trips the tripwire ${v8Report.calls.length}× (3 scripts + the flow's own Function calls) — the production result is load-bearing`);
} finally {
  await fs.rm(dir, { recursive: true, force: true });
  // The host-engine bundle must not outlive the test: it is a CLI that runs
  // flows on the host and still says engine: zipp. (Built in dist/ rather than
  // the temp dir because its eager `undici` import resolves from cli/node_modules.)
  for (const leftover of ['_v8_oaiy.mjs', '_v8_oaiy.mjs.map']) {
    await fs.rm(path.join(distDir, leftover), { force: true });
  }
}
