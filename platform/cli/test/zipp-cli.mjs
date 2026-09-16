// Exercise the built CLI at its process boundary: flow code runs on the ZIPP VM
// in a worker thread, so a token in the CLI's environment never reaches a flow
// or either output stream; a missing or corrupt engine artifact is refused
// before any job runs; a timed-out run is aborted and the process still exits
// on its own (no worker thread left behind).
//
//     node test/zipp-cli.mjs [path/to/oaiy.mjs]
//
// Like cli-http-exit.mjs this spawns dist/oaiy.mjs, so build first.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const distZipp = path.join(path.dirname(cli), 'zipp');
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'oaiy-zipp-cli-'));

const CANARY = 'canary-secret';

/** Run the CLI once; the child never inherits this process's environment as-is. */
async function run(flow, args, env, killAfterMs = 20_000) {
  const child = spawn(process.execPath, [cli, 'run', flow, '--quiet', ...args], {
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, ...env },
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (data) => { stdout += data; });
  child.stderr.on('data', (data) => { stderr += data; });
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

try {
  // --- 1. The canary: a token in the CLI's environment is unreachable -------
  const probe = fileURLToPath(new URL('./fixtures/realm-probe.json', import.meta.url));
  const canary = await run(probe, ['--timeout', '20'], { OAIY_SERVER_TOKEN: CANARY });
  assert.equal(canary.signal, null, `CLI failed to finish: ${canary.stderr}`);
  assert.equal(canary.code, 0, `CLI did not succeed: ${canary.stderr}`);
  const payload = JSON.parse(canary.stdout);
  assert.equal(payload.success, true);
  assert.deepEqual(payload.output.types, ['undefined', 'undefined', 'undefined', 'undefined']);
  assert.equal(payload.output.token, 'no-process');
  assert.doesNotMatch(canary.stdout, new RegExp(CANARY), 'the token reached stdout');
  assert.doesNotMatch(canary.stderr, new RegExp(CANARY), 'the token reached stderr');
  assert.equal(payload.engine, 'zipp', 'the payload names the engine the flow ran on');
  console.log('PASS: a flow cannot see the host realm or OAIY_SERVER_TOKEN, and the payload says engine: zipp');

  // --- 2. No artifact: refused before any job, exit 1 -----------------------
  const trivial = await writeFlow('trivial.json', 'return 1;');
  const empty = path.join(dir, 'empty');
  await fs.mkdir(empty);
  const missing = await run(trivial, [], { OAIY_ZIPP_ASSET_DIR: empty });
  assert.equal(missing.signal, null, `CLI failed to finish: ${missing.stderr}`);
  assert.equal(missing.code, 1, `CLI exit did not match its result: ${missing.stderr}`);
  const missingPayload = JSON.parse(missing.stdout);
  assert.equal(missingPayload.success, false);
  assert.equal(missingPayload.status, 'failed');
  assert.equal(missingPayload.errorCode, 'engine_unavailable');
  assert.equal(missingPayload.jobId, '', 'no job was submitted');
  console.log('PASS: a missing engine artifact is engine_unavailable and nothing runs');

  // --- 3. One flipped byte: the same refusal ---------------------------------
  const flipped = path.join(dir, 'flipped');
  await fs.mkdir(flipped);
  for (const name of await fs.readdir(distZipp)) {
    await fs.copyFile(path.join(distZipp, name), path.join(flipped, name));
  }
  const wasm = path.join(flipped, 'zipp_wasm_bg.wasm');
  const bytes = await fs.readFile(wasm);
  bytes[Math.floor(bytes.length / 2)] ^= 0x01;
  await fs.writeFile(wasm, bytes);
  const corrupt = await run(trivial, [], { OAIY_ZIPP_ASSET_DIR: flipped });
  assert.equal(corrupt.signal, null, `CLI failed to finish: ${corrupt.stderr}`);
  assert.equal(corrupt.code, 1, `CLI exit did not match its result: ${corrupt.stderr}`);
  const corruptPayload = JSON.parse(corrupt.stdout);
  assert.equal(corruptPayload.errorCode, 'engine_unavailable');
  assert.equal(corruptPayload.jobId, '');
  console.log('PASS: a corrupt engine artifact is engine_unavailable and nothing runs');

  // --- 4. --timeout aborts a run that renews its budget forever --------------
  // Every `await` renews the instruction budget, so only terminating the
  // worker stops this loop; the process must then exit by itself.
  const forever = await writeFlow('forever.json', "for (;;) { await Utility.memoryRead('k'); }");
  const timed = await run(forever, ['--timeout', '2'], {}, 15_000);
  assert.equal(timed.signal, null, `CLI did not exit on its own after the timeout: ${timed.stderr}`);
  assert.equal(timed.code, 1, `CLI exit did not match its result: ${timed.stderr}`);
  const timedPayload = JSON.parse(timed.stdout);
  assert.equal(timedPayload.status, 'aborted');
  assert.equal(timedPayload.errorCode, 'timeout');
  assert.ok(timed.elapsed < 10_000, `the abort took ${timed.elapsed} ms`);
  assert.doesNotMatch(timed.stderr, /Assertion failed/);
  console.log(`PASS: --timeout aborts the run (${timed.elapsed} ms) and the worker thread does not outlive the process`);
} finally {
  await fs.rm(dir, { recursive: true, force: true });
}
