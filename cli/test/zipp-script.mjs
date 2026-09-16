// `oaiy script` and `run --profile` at the process boundary: the leaf-script
// envelope served by one warm engine worker, its NDJSON protocol, its watchdog
// and recycle policy, and a script profile applied to a workflow run.
//
//     node test/zipp-script.mjs [path/to/oaiy.mjs]
//
// Like zipp-cli.mjs this spawns dist/oaiy.mjs, so build first.
//
//   * `--request`: a JavaScript job and a Python project run on the staged
//     web-python engine; the response validates against
//     protocol/v1/script-result.schema.json and names the staged engine
//     (dist/zipp/SOURCE.json — never a literal); a refused request is the
//     refusal object and exit 1; a non-JSON file and a missing engine write
//     NO output file.
//   * `--serve`: N batches over one worker; `ping` reports `instance` 1 and
//     the jobs served; a resource error (an exhausted instruction budget), a
//     watchdog `timeout` (a loop with the budget at its ceiling and a small
//     `budgetMs`) and the 5000-job count each recycle the worker — `instance`
//     goes up, `jobs` restarts — while the NEXT job in the same batch still
//     runs; a malformed line is one error line and the next request is served;
//     every stdout line is protocol JSON; `shutdown` and EOF exit 0 on their
//     own (no worker thread outlives the process); every result line
//     validates against the result schema, timeouts and host errors included.
//   * `run --profile`: a flow reads a name only the profile's preamble
//     defines; the profile's `instructionSteps` is the run's budget;
//     `instructionSteps` together with `--instruction-budget` is refused
//     BEFORE the flow file is read (no result file); a bad digest, a preamble
//     declaring an engine or envelope name (also by destructuring, which the
//     envelope's text scan cannot see), a `let` over a guest shim and a
//     preamble that does not parse are each refused.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const distZipp = path.join(path.dirname(cli), 'zipp');
const protocolDir = fileURLToPath(new URL('../../protocol/v1/', import.meta.url));
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'oaiy-zipp-script-'));

const sha256 = (text) => createHash('sha256').update(text, 'utf8').digest('hex');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// --- the result schema, checked with the same small validator ui/tests/zipp-script.mjs uses ---
const load = async (name) => JSON.parse(await fs.readFile(path.join(protocolDir, name), 'utf8'));
const schemas = [await load('script-request.schema.json'), await load('script-profile.schema.json'), await load('script-result.schema.json')];
const resultSchema = schemas[2];
const validate = makeValidator(schemas);
assert.ok(!validate(resultSchema, { v: 1, results: [] }), 'the validator rejects a known-bad result');
const validResult = (r) => validate(resultSchema, r);

/** Spawn the CLI with `args`; the child's environment is this one plus `env`. */
async function run(args, env = {}, killAfterMs = 60_000) {
  const child = spawn(process.execPath, [cli, ...args], {
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

async function writeJson(name, value) {
  const file = path.join(dir, name);
  await fs.writeFile(file, JSON.stringify(value));
  return file;
}

const PY_FILES = {
  'main.py': 'import lib\n\ndef run(ctx):\n    return {"v": lib.twice(ctx["n"]), "ctx": ctx}\n',
  'lib.py': 'def twice(n):\n    return n * 2\n',
};
const pyJob = (id, n) => ({ id, language: 'python', mode: 'python-project', files: PY_FILES, entry: 'main', call: 'run', args: [{ n }], budgetMs: 20_000 });
/** A loop only the wall clock stops: the instruction budget at its ceiling, a small `budgetMs`. */
const loopJob = (id, budgetMs) => ({ id, mode: 'program', source: 'while (true) {}', instructionSteps: 2_000_000_000, budgetMs });
/** A loop the instruction budget stops: `resource`. */
const budgetJob = (id) => ({ id, mode: 'program', source: 'while (true) {}', instructionSteps: 1000 });

/** Drive one `--serve` process: send lines, await the line matching `pred`. */
function serve(env = {}) {
  const child = spawn(process.execPath, [cli, 'script', '--serve'], {
    windowsHide: true,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, ...env },
  });
  const lines = [];
  const rawLines = [];
  const waiters = [];
  let buffer = '';
  let stderr = '';
  child.stderr.on('data', (d) => { stderr += d; });
  child.stdout.on('data', (d) => {
    buffer += d;
    let at;
    while ((at = buffer.indexOf('\n')) !== -1) {
      const raw = buffer.slice(0, at);
      buffer = buffer.slice(at + 1);
      rawLines.push(raw);
      let parsed;
      try { parsed = JSON.parse(raw); } catch { parsed = { __unparseable: raw }; }
      lines.push(parsed);
      for (const w of [...waiters]) {
        if (w.pred(parsed)) {
          waiters.splice(waiters.indexOf(w), 1);
          clearTimeout(w.timer);
          w.resolve(parsed);
        }
      }
    }
  });
  const closed = once(child, 'close');
  return {
    child,
    lines,
    rawLines,
    get stderr() { return stderr; },
    send(obj) { child.stdin.write((typeof obj === 'string' ? obj : JSON.stringify(obj)) + '\n'); },
    next(pred, timeoutMs = 30_000, what = 'a line') {
      const already = lines.find(pred);
      if (already) return Promise.resolve(already);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}; stderr: ${stderr}`)), timeoutMs);
        waiters.push({ pred, resolve, timer });
      });
    },
    async end() {
      child.stdin.end();
      return closed;
    },
    async close(waitMs = 15_000) {
      const timer = setTimeout(() => child.kill(), waitMs);
      try {
        const [code, signal] = await closed;
        return { code, signal };
      } finally {
        clearTimeout(timer);
      }
    },
  };
}

const stagedSource = JSON.parse(await fs.readFile(path.join(distZipp, 'SOURCE.json'), 'utf8'));
const stagedProfile = JSON.parse(await fs.readFile(path.join(distZipp, 'PROFILE.json'), 'utf8'));

try {
  // =========================================================================
  // 1. `script --request`: one request, one response.
  // =========================================================================
  const request = { v: 1, jobs: [
    { id: 'js', mode: 'program', source: '1 + 1' },
    { id: 'body', mode: 'body', source: 'return globalThis.k * 2', globals: { k: 21 } },
    pyJob('py', 21),
    { id: 'entry', mode: 'entry', source: 'function main(a, b) { return a + b; }', entry: 'main', args: [40, 2] },
    { id: 'throws', mode: 'program', source: 'throw new Error("boom")' },
  ] };
  const reqFile = await writeJson('request.json', request);
  const one = await run(['script', '--request', reqFile]);
  assert.equal(one.signal, null, `script did not finish: ${one.stderr}`);
  assert.equal(one.code, 0, `script failed: ${one.stderr}`);
  const response = JSON.parse(one.stdout);
  assert.ok(validResult(response), `the response does not validate: ${one.stdout}`);
  assert.equal(response.engine.name, 'zipp');
  assert.equal(response.engine.wasmSha256, stagedSource.sha256, 'engine.wasmSha256 is the staged SOURCE.json digest');
  assert.equal(response.engine.release, stagedSource.release);
  assert.equal(response.engine.revision, stagedSource.revision);
  assert.deepEqual(response.engine.languages, stagedSource.languages);
  const byId = Object.fromEntries(response.results.map((r) => [r.id, r]));
  assert.deepEqual(byId.js, { id: 'js', ok: true, value: 2 });
  assert.deepEqual(byId.body, { id: 'body', ok: true, value: 42 });
  assert.deepEqual(byId.py, { id: 'py', ok: true, value: { v: 42, ctx: { n: 21 } } }, 'the Python project ran on the staged web-python engine');
  assert.deepEqual(byId.entry, { id: 'entry', ok: true, value: 42 });
  assert.equal(byId.throws.ok, false);
  assert.equal(byId.throws.errorKind, 'guest');
  assert.match(byId.throws.error, /boom/);
  assert.deepEqual(response.results.map((r) => r.id), request.jobs.map((j) => j.id), 'results keep request order');
  console.log('PASS: script --request runs JavaScript and a Python project on the staged engine; the response validates and names the staged engine');

  // -o writes the same response and leaves stdout empty.
  const outFile = path.join(dir, 'response.json');
  const toFile = await run(['script', '--request', reqFile, '-o', outFile]);
  assert.equal(toFile.code, 0, toFile.stderr);
  assert.equal(toFile.stdout, '', 'stdout stays empty with -o');
  const written = JSON.parse(await fs.readFile(outFile, 'utf8'));
  assert.deepEqual(written.results, response.results);
  console.log('PASS: script --request -o writes the response and keeps stdout empty');

  // A refused request: the refusal object, exit 1, schema-valid.
  const badReq = await writeJson('bad-request.json', { v: 1, jobs: [{ id: 'x', mode: 'nope', source: '1' }] });
  const refused = await run(['script', '--request', badReq]);
  assert.equal(refused.code, 1, `a refused request must exit 1: ${refused.stderr}`);
  const refusal = JSON.parse(refused.stdout);
  assert.ok(validResult(refusal), `the refusal does not validate: ${refused.stdout}`);
  assert.equal(refusal.error.code, 'invalid_request');
  assert.match(refusal.error.message, /jobs\[0\] \(id "x"\): unknown mode "nope"/);
  console.log('PASS: a refused request is the refusal object, exit 1');

  // A file that is not JSON, and no engine: exit 1, NO output file.
  const notJson = path.join(dir, 'not.json');
  await fs.writeFile(notJson, '{ this is not json');
  const noOut = path.join(dir, 'never-written.json');
  const unparseable = await run(['script', '--request', notJson, '-o', noOut]);
  assert.equal(unparseable.code, 1);
  assert.match(unparseable.stderr, /not JSON/);
  const empty = path.join(dir, 'empty');
  await fs.mkdir(empty);
  const noEngine = await run(['script', '--request', reqFile, '-o', noOut], { OAIY_ZIPP_ASSET_DIR: empty });
  assert.equal(noEngine.code, 1);
  assert.equal(noEngine.stdout, '');
  assert.match(noEngine.stderr, /ZIPP engine unavailable/);
  await assert.rejects(fs.stat(noOut), 'no result file may exist after a refusal or a missing engine');
  console.log('PASS: a non-JSON request file and a missing engine exit 1 and write no output file');

  // `--serve` with no engine: one error line, exit 1.
  const deadServe = serve({ OAIY_ZIPP_ASSET_DIR: empty });
  const deadLine = await deadServe.next((l) => l.op === 'error', 20_000, 'the engine_unavailable line');
  assert.equal(deadLine.error.code, 'engine_unavailable');
  assert.equal(deadLine.id, null);
  const deadExit = await deadServe.close();
  assert.equal(deadExit.code, 1);
  assert.equal(deadServe.lines.length, 1, `exactly one line: ${deadServe.rawLines.join(' | ')}`);
  console.log('PASS: script --serve without an engine writes one engine_unavailable line and exits 1');

  // =========================================================================
  // 2. `script --serve`: the NDJSON protocol over one warm worker.
  // =========================================================================
  const s = serve();
  const resultLines = [];
  const onResult = (l) => { if (l.op === 'result') resultLines.push(l); };

  s.send({ op: 'ping' });
  const pong0 = await s.next((l) => l.op === 'pong', 30_000, 'the first pong');
  assert.deepEqual(pong0.engine, response.engine, 'pong.engine is the same identity the one-shot response carried');
  assert.equal(pong0.instance, 1);
  assert.equal(pong0.jobs, 0);

  // N batches, in order, on one worker.
  const N = 4;
  for (let i = 0; i < N; i++) {
    s.send({ op: 'batch', id: `b${i}`, request: { v: 1, jobs: [{ id: 'j', mode: 'program', source: `${i} * 10` }, pyJob('p', i)] } });
  }
  for (let i = 0; i < N; i++) {
    const line = await s.next((l) => l.op === 'result' && l.id === `b${i}`, 60_000, `result b${i}`);
    onResult(line);
    assert.ok(validResult(line.result), JSON.stringify(line));
    assert.deepEqual(line.result.results, [{ id: 'j', ok: true, value: i * 10 }, { id: 'p', ok: true, value: { v: i * 2, ctx: { n: i } } }]);
  }
  const order = s.lines.filter((l) => l.op === 'result').map((l) => l.id);
  assert.deepEqual(order, ['b0', 'b1', 'b2', 'b3'], 'batches are answered in arrival order');
  s.send({ op: 'ping', id: 'after-n' });
  const pongN = await s.next((l) => l.op === 'pong' && l.id === 'after-n', 30_000, 'pong after N');
  assert.equal(pongN.instance, 1, 'no recycle for successful jobs');
  assert.equal(pongN.jobs, N * 2, 'pong.jobs counts the jobs the worker served');
  console.log(`PASS: --serve answers ${N} batches (JS + Python) over one worker, in order; pong reports instance 1 and ${N * 2} jobs`);

  // (a) A resource error recycles the worker; the next job in the batch runs on the replacement.
  s.send({ op: 'batch', id: 'res', request: { v: 1, jobs: [budgetJob('burn'), { id: 'next', mode: 'program', source: '"alive"' }] } });
  const res = await s.next((l) => l.op === 'result' && l.id === 'res', 60_000, 'result res');
  onResult(res);
  assert.ok(validResult(res.result), JSON.stringify(res));
  assert.equal(res.result.results[0].ok, false);
  assert.equal(res.result.results[0].errorKind, 'resource', JSON.stringify(res.result.results[0]));
  assert.match(res.result.results[0].error, /instruction budget/);
  assert.deepEqual(res.result.results[1], { id: 'next', ok: true, value: 'alive' }, 'the job after the resource error still runs');
  s.send({ op: 'ping', id: 'after-res' });
  const pongRes = await s.next((l) => l.op === 'pong' && l.id === 'after-res', 30_000, 'pong after resource');
  assert.equal(pongRes.instance, 2, 'a resource error recycled the worker (instance 1 → 2)');
  assert.equal(pongRes.jobs, 1, 'the replacement served the one job after the error');
  console.log('PASS: a resource error recycles the worker (pong.instance 1 → 2) and the next job runs on the replacement');

  // The watchdog: a job past its budgetMs is `timeout`, the worker is replaced, the batch continues.
  const t0 = Date.now();
  s.send({ op: 'batch', id: 'slow', request: { v: 1, jobs: [loopJob('loop', 300), { id: 'after', mode: 'body', source: 'return "still served"' }] } });
  const slow = await s.next((l) => l.op === 'result' && l.id === 'slow', 60_000, 'result slow');
  const slowMs = Date.now() - t0;
  onResult(slow);
  assert.ok(validResult(slow.result), JSON.stringify(slow));
  assert.equal(slow.result.results[0].ok, false);
  assert.equal(slow.result.results[0].errorKind, 'timeout', JSON.stringify(slow.result.results[0]));
  assert.match(slow.result.results[0].error, /300 ms budget/);
  assert.deepEqual(slow.result.results[1], { id: 'after', ok: true, value: 'still served' }, 'the job after the timeout still runs');
  assert.ok(slowMs >= 1800 && slowMs < 15_000, `the watchdog fired after budget + grace (${slowMs} ms)`);
  s.send({ op: 'ping', id: 'after-slow' });
  const pongSlow = await s.next((l) => l.op === 'pong' && l.id === 'after-slow', 30_000, 'pong after slow');
  assert.equal(pongSlow.instance, 3, 'the watchdog recycled the worker (instance 2 → 3)');
  console.log(`PASS: a job past its budgetMs is 'timeout' (${slowMs} ms), the worker is replaced (instance 3) and the next job still runs`);

  // (c) The job count: 5000 jobs on one worker, then a replacement.
  const many = { v: 1, jobs: Array.from({ length: 5001 }, (_, i) => ({ id: `m${i}`, mode: 'program', source: String(i) })) };
  const tMany = Date.now();
  s.send({ op: 'batch', id: 'many', request: many });
  const manyLine = await s.next((l) => l.op === 'result' && l.id === 'many', 240_000, 'result many');
  onResult(manyLine);
  assert.ok(validResult(manyLine.result));
  assert.equal(manyLine.result.results.length, 5001);
  assert.ok(manyLine.result.results.every((r, i) => r.ok && r.value === i), 'every one of the 5001 jobs ran');
  s.send({ op: 'ping', id: 'after-many' });
  const pongMany = await s.next((l) => l.op === 'pong' && l.id === 'after-many', 30_000, 'pong after many');
  assert.equal(pongMany.instance, 4, 'the 5000th job recycled the worker (instance 3 → 4)');
  // Worker #3 had already served `pongSlow.jobs`; it took jobs up to its 5000th and the rest ran on #4.
  assert.equal(pongMany.jobs, 5001 - (5000 - pongSlow.jobs), 'the jobs past the 5000th ran on the replacement');
  console.log(`PASS: 5001 jobs in one batch (${Date.now() - tMany} ms): the 5000-job count recycles the worker (instance 4) and the last job runs on the replacement`);

  // Malformed lines: one error line each, the stream continues.
  s.send('this is not json');
  const notJsonLine = await s.next((l) => l.op === 'error' && /not JSON/.test(l.error?.message ?? ''), 10_000, 'the not-JSON error');
  assert.equal(notJsonLine.id, null);
  assert.equal(notJsonLine.error.code, 'invalid_line');
  s.send('[1, 2]');
  await s.next((l) => l.op === 'error' && /JSON object/.test(l.error?.message ?? ''), 10_000, 'the not-an-object error');
  s.send({ op: 'dance', id: 'd1' });
  const unknownOp = await s.next((l) => l.op === 'error' && l.id === 'd1', 10_000, 'the unknown-op error');
  assert.match(unknownOp.error.message, /unknown op "dance"/);
  s.send({ op: 'batch', request: { v: 1, jobs: [] } });
  await s.next((l) => l.op === 'error' && /needs an "id"/.test(l.error?.message ?? ''), 10_000, 'the no-id error');
  s.send({ op: 'batch', id: 'no-request' });
  await s.next((l) => l.op === 'error' && l.id === 'no-request' && /needs a "request"/.test(l.error?.message ?? ''), 10_000, 'the no-request error');
  // A refused request is the refusal object, as a result line.
  s.send({ op: 'batch', id: 'refused', request: { v: 1, jobs: [{ id: 'x', mode: 'program' }] } });
  const refusedLine = await s.next((l) => l.op === 'result' && l.id === 'refused', 10_000, 'the refusal');
  onResult(refusedLine);
  assert.ok(validResult(refusedLine.result));
  assert.equal(refusedLine.result.error.code, 'invalid_request');
  // And the next request is served.
  s.send({ op: 'batch', id: 'still', request: { v: 1, jobs: [{ id: 'ok', mode: 'program', source: '"served"' }] } });
  const still = await s.next((l) => l.op === 'result' && l.id === 'still', 30_000, 'result still');
  onResult(still);
  assert.deepEqual(still.result.results, [{ id: 'ok', ok: true, value: 'served' }]);
  console.log('PASS: malformed lines (not JSON, not an object, unknown op, batch without id/request) are one error line each; a refused request is the refusal; the next request is served');

  // Every stdout line was protocol JSON, and every result validated.
  assert.ok(s.lines.every((l) => !('__unparseable' in l) && typeof l.op === 'string'), `stdout carried a non-protocol line: ${s.rawLines.find((r) => { try { return typeof JSON.parse(r).op !== 'string'; } catch { return true; } })}`);
  assert.ok(resultLines.length >= N + 5 && resultLines.every((l) => validResult(l.result)), 'every result line validates against script-result.schema.json');

  // `shutdown`: exit 0 on its own, no more lines, no kill.
  const linesBefore = s.lines.length;
  s.send({ op: 'shutdown' });
  const exit = await s.close();
  assert.equal(exit.signal, null, `--serve did not exit on its own after shutdown: ${s.stderr}`);
  assert.equal(exit.code, 0, `--serve exit ${exit.code}: ${s.stderr}`);
  assert.equal(s.lines.length, linesBefore, 'shutdown writes no reply line');
  assert.doesNotMatch(s.stderr, /Assertion failed|Error:/);
  console.log(`PASS: {op:"shutdown"} exits 0 on its own (${s.lines.length} protocol lines, ${resultLines.length} results all schema-valid); no worker outlives the process`);

  // EOF on stdin does the same, after answering what it already received.
  const e = serve();
  e.send({ op: 'batch', id: 'last', request: { v: 1, jobs: [{ id: 'x', mode: 'program', source: '7' }] } });
  const [eofCode, eofSignal] = await e.end();
  assert.equal(eofSignal, null);
  assert.equal(eofCode, 0, `EOF exit ${eofCode}: ${e.stderr}`);
  const last = e.lines.find((l) => l.op === 'result' && l.id === 'last');
  assert.deepEqual(last?.result?.results, [{ id: 'x', ok: true, value: 7 }], 'a batch received before EOF is still answered');
  console.log('PASS: EOF on stdin answers what was received and exits 0');

  // =========================================================================
  // 3. `run --profile`.
  // =========================================================================
  const flow = (code) => ({
    nodes: [
      { id: 'b', type: 'logic_block', position: { x: 0, y: 0 }, data: { code } },
      { id: 'out', type: 'output', position: { x: 200, y: 0 }, data: { value: '$nodes.b' } },
    ],
    edges: [{ id: 'edge', source: 'b', target: 'out' }],
  });
  const readsHelper = await writeJson('reads-helper.json', flow("return typeof profileHelper === 'function' ? profileHelper(2) : 'absent';"));
  const preamble = 'function profileHelper(n) { return n * 21; }\nvar profileConst = 7;\n';
  const profileFile = await writeJson('profile.json', { v: 1, preamble, preambleSha256: sha256(preamble) });

  const without = await run(['run', readsHelper, '--quiet', '--timeout', '30']);
  assert.equal(without.code, 0, without.stderr);
  assert.equal(JSON.parse(without.stdout).output, 'absent', 'without a profile the name is undefined');
  const withProfile = await run(['run', readsHelper, '--quiet', '--timeout', '30', '--profile', profileFile]);
  assert.equal(withProfile.code, 0, withProfile.stderr);
  const applied = JSON.parse(withProfile.stdout);
  assert.equal(applied.output, 42, `the flow read the preamble's function: ${withProfile.stdout}`);
  assert.equal(applied.engine, 'zipp');
  console.log("PASS: run --profile places the preamble at program top level — a flow reads a function only the preamble defines (42, 'absent' without)");

  // The profile's instructionSteps is the run's budget.
  // Two statements, as zipp-guard.mjs writes it: a lone `while (true) {}` does not compile as a logic block.
  const forever = await writeJson('forever.json', flow('let n = 0;\nwhile (true) { n++; }'));
  const tinyBudget = await writeJson('profile-budget.json', { v: 1, preamble, preambleSha256: sha256(preamble), instructionSteps: 2000 });
  const budgeted = await run(['run', forever, '--quiet', '--timeout', '30', '--profile', tinyBudget]);
  assert.equal(budgeted.code, 1);
  const budgetedPayload = JSON.parse(budgeted.stdout);
  assert.equal(budgetedPayload.success, false);
  assert.notEqual(budgetedPayload.errorCode, 'timeout', 'the instruction budget, not the wall clock, ended the run');
  assert.match(budgetedPayload.error ?? '', /instruction budget/, budgeted.stdout);
  assert.ok(budgeted.elapsed < 20_000, `the budget stopped the loop in ${budgeted.elapsed} ms`);
  console.log(`PASS: the profile's instructionSteps is the run's budget (a loop stopped by the budget in ${budgeted.elapsed} ms)`);

  // Both budgets → refused before the flow is read; no result file.
  const missingFlow = path.join(dir, 'does-not-exist.json');
  const bothOut = path.join(dir, 'both.json');
  const both = await run(['run', missingFlow, '--quiet', '--profile', tinyBudget, '--instruction-budget', '5000', '-o', bothOut]);
  assert.equal(both.code, 1, 'two budgets must be refused');
  assert.equal(both.stdout, '');
  assert.match(both.stderr, /instructionSteps \(2000\) and --instruction-budget was also given \(5000\); pass one or the other/, both.stderr);
  assert.doesNotMatch(both.stderr, /ENOENT|does-not-exist/, 'the flow file was read before the budgets were refused');
  await assert.rejects(fs.stat(bothOut), 'no result file after a refusal');
  // A profile WITHOUT instructionSteps beside --instruction-budget is fine: the CLI figure applies.
  const cliBudget = await run(['run', forever, '--quiet', '--timeout', '30', '--profile', profileFile, '--instruction-budget', '2000']);
  assert.equal(cliBudget.code, 1);
  assert.match(JSON.parse(cliBudget.stdout).error ?? '', /instruction budget/);
  console.log('PASS: profile.instructionSteps together with --instruction-budget is refused before the flow is read (no result file); a profile without a budget takes the CLI figure');

  // Refusals: bad digest, reserved names (incl. destructuring), a lexical shim redeclaration, a parse error.
  const refusals = [
    ['a digest that does not match', { v: 1, preamble, preambleSha256: sha256('other') }, /preambleSha256 does not match/],
    ['a preamble declaring `host`', { v: 1, preamble: 'var host = 1;', preambleSha256: sha256('var host = 1;') }, /declares "host"/],
    ['a preamble redeclaring `__emit`', { v: 1, preamble: 'function __emit() {}', preambleSha256: sha256('function __emit() {}') }, /declares "__emit"/],
    ['a destructured `__replies` (only the parser sees it)', { v: 1, preamble: 'const { __replies } = {};', preambleSha256: sha256('const { __replies } = {};') }, /declares "__replies"/],
    ['`__oaiyConsole`, the wrapper\'s own name', { v: 1, preamble: 'let __oaiyConsole = 1;', preambleSha256: sha256('let __oaiyConsole = 1;') }, /declares "__oaiyConsole"/],
    ['a `let` over the guest shim `setTimeout`', { v: 1, preamble: 'let setTimeout = 1;', preambleSha256: sha256('let setTimeout = 1;') }, /redeclares the guest shim "setTimeout"/],
    ['a preamble that does not parse', { v: 1, preamble: 'function (', preambleSha256: sha256('function (') }, /does not parse/],
    ['a profile with an unknown field', { v: 1, preamble, preambleSha256: sha256(preamble), extra: 1 }, /unknown field "extra"/],
  ];
  for (const [i, [label, doc, why]] of refusals.entries()) {
    const file = await writeJson(`refuse-${i}.json`, doc);
    const r = await run(['run', missingFlow, '--quiet', '--profile', file, '-o', bothOut]);
    assert.equal(r.code, 1, `${label}: exit ${r.code}`);
    assert.equal(r.stdout, '', `${label}: something was printed`);
    assert.match(r.stderr, why, `${label}: ${r.stderr}`);
    assert.doesNotMatch(r.stderr, /ENOENT|does-not-exist/, `${label}: the flow was read first`);
  }
  await assert.rejects(fs.stat(bothOut));
  // A `var` over a shim is the requester's choice and is accepted.
  const varShim = 'var fetch = function () { return "mine"; };';
  const varShimFile = await writeJson('profile-var-shim.json', { v: 1, preamble: varShim, preambleSha256: sha256(varShim) });
  const readsFetch = await writeJson('reads-fetch.json', flow('return fetch();'));
  const replaced = await run(['run', readsFetch, '--quiet', '--timeout', '30', '--profile', varShimFile]);
  assert.equal(replaced.code, 0, replaced.stderr);
  assert.equal(JSON.parse(replaced.stdout).output, 'mine', 'a var in the preamble replaces the throwing shim');
  console.log(`PASS: ${refusals.length} profile refusals (digest, engine/envelope/wrapper names incl. destructuring, a let over a shim, a parse error, an unknown field) exit 1 before the flow is read; a var over a shim is accepted`);

  // =========================================================================
  // 4. capabilities: the script protocol and its languages.
  // =========================================================================
  const caps = await run(['capabilities', '--json']);
  assert.equal(caps.code, 0, caps.stderr);
  const report = JSON.parse(caps.stdout);
  assert.deepEqual(report.protocols, { run: 1, script: 1, profile: 1 });
  assert.deepEqual(report.run.languages, ['javascript'], 'run.languages: what a WORKFLOW may be written in');
  assert.deepEqual(report.script.languages, stagedSource.languages, 'script.languages: what a LEAF job may be written in (the engine list)');
  assert.ok(report.script.languages.includes('python'), 'the staged web-python engine runs Python leaf jobs, and says so');
  assert.equal(report.script.maxBudgetMs, 60_000);
  assert.equal(report.script.defaultBudgetMs, 1000);
  assert.equal(report.run.maxInstructionSteps, stagedProfile.limits.maxInstructionBudgetSteps);
  console.log('PASS: capabilities --json reports protocols {run, script, profile}, run.languages [javascript] and script.languages from the engine record');
} finally {
  await fs.rm(dir, { recursive: true, force: true });
}

/**
 * A validator for the subset of JSON Schema 2020-12 the protocol schemas use
 * (the same one `ui/tests/zipp-script.mjs` carries): type, const, enum,
 * properties, required, additionalProperties, propertyNames, min/max on
 * strings, numbers, arrays and objects, pattern, items, allOf/anyOf/oneOf/not,
 * if/then/else, $ref by `$id` and `#/` pointer.
 */
function makeValidator(schemas) {
  const byId = new Map(schemas.map((s) => [s.$id, s]));
  const typeOf = (v) => v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v === 'number' ? (Number.isInteger(v) ? 'integer' : 'number') : typeof v;
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  function resolve(ref, root) {
    const [id, fragment = ''] = ref.split('#');
    const doc = id ? byId.get(id) : root;
    if (!doc) throw new Error(`unresolved $ref ${ref}`);
    let node = doc;
    for (const seg of fragment.split('/').filter(Boolean)) node = node[seg.replace(/~1/g, '/').replace(/~0/g, '~')];
    if (node === undefined) throw new Error(`unresolved $ref ${ref}`);
    return { node, root: doc };
  }
  function valid(schema, v, root) {
    if (schema === true) return true;
    if (schema === false) return false;
    if (schema.$ref) {
      const r = resolve(schema.$ref, root);
      if (!valid(r.node, v, r.root)) return false;
    }
    if (schema.type) {
      const types = [].concat(schema.type);
      const t = typeOf(v);
      if (!types.includes(t) && !(t === 'integer' && types.includes('number'))) return false;
    }
    if (has(schema, 'const') && !same(schema.const, v)) return false;
    if (schema.enum && !schema.enum.some((e) => same(e, v))) return false;
    if (typeof v === 'string') {
      if (schema.minLength !== undefined && v.length < schema.minLength) return false;
      if (schema.maxLength !== undefined && v.length > schema.maxLength) return false;
      if (schema.pattern && !new RegExp(schema.pattern, 'u').test(v)) return false;
    }
    if (typeof v === 'number') {
      if (schema.minimum !== undefined && v < schema.minimum) return false;
      if (schema.maximum !== undefined && v > schema.maximum) return false;
    }
    if (Array.isArray(v)) {
      if (schema.minItems !== undefined && v.length < schema.minItems) return false;
      if (schema.items !== undefined && !v.every((x) => valid(schema.items, x, root))) return false;
    }
    if (typeOf(v) === 'object') {
      const keys = Object.keys(v);
      if (schema.required && !schema.required.every((k) => keys.includes(k))) return false;
      if (schema.minProperties !== undefined && keys.length < schema.minProperties) return false;
      if (schema.propertyNames && !keys.every((k) => valid(schema.propertyNames, k, root))) return false;
      for (const k of keys) {
        if (schema.properties && has(schema.properties, k)) {
          if (!valid(schema.properties[k], v[k], root)) return false;
        } else if (schema.additionalProperties !== undefined && !valid(schema.additionalProperties, v[k], root)) {
          return false;
        }
      }
    }
    if (schema.allOf && !schema.allOf.every((s) => valid(s, v, root))) return false;
    if (schema.anyOf && !schema.anyOf.some((s) => valid(s, v, root))) return false;
    if (schema.oneOf && schema.oneOf.filter((s) => valid(s, v, root)).length !== 1) return false;
    if (schema.not && valid(schema.not, v, root)) return false;
    if (schema.if) {
      const branch = valid(schema.if, v, root) ? schema.then : schema.else;
      if (branch !== undefined && !valid(branch, v, root)) return false;
    }
    return true;
  }
  return (schema, v) => valid(schema, v, schema);
}
