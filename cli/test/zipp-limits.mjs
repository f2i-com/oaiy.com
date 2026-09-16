// D13: what the ZIPP boundary does with large and host-typed values — MEASURED,
// not asserted. Every row is recorded; the one thing this test insists on is
// that no case hangs (each is bounded by the CLI's --timeout and a kill).
//
//     node test/zipp-limits.mjs [path/to/oaiy.mjs]
//
// Rows (the figures land in TESTING.md, "ZIPP boundary measurements"):
//
//   * a logic_block returning a 900 KiB, a 1.5 MiB and a 20 MiB string —
//     guest → host, through `__system.finish` (the whole workflow context as
//     one JSON string on the host-call queue; PROFILE `hostCallDrainStringBytes`
//     32 MiB) and structured clone to the main thread;
//   * a 2 MiB module result — host → guest: `Utility.memoryWrite` carries the
//     2 MiB out as a host-call argument, `Utility.memoryRead` brings it back
//     as a `host_result`, which the worker hands to `resolveHostCallback` as a
//     JS value (PROFILE `hostValueStringBytes` 16 MiB);
//   * a 2 MiB input literal — the compiler embeds `--inputs` into the script
//     source (`let __inputs = …`; PROFILE `initialSourceBytes` 16 MiB);
//   * Buffer, Uint8Array and Date host results — what a module method's
//     return value looks like to guest code after structured clone and ZIPP's
//     JS→guest conversion. These need a module method that returns such a
//     value, so they run through test/zipp-limits-probe.ts, an engine-level
//     script bundled here with the CLI's own build options, which registers a
//     `Probe` runtime module beside the bundled ones and runs blocks on
//     `createCliEngine`.
//
// Do NOT read a 1 MiB ceiling into this: PROFILE.json says 16 MiB for a host
// value's strings, and this file exists to replace that figure with a reading.
import esbuild from 'esbuild';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const distDir = path.dirname(cli);
const here = fileURLToPath(new URL('./', import.meta.url));
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'oaiy-zipp-limits-'));

const KiB = 1024;
const MiB = 1024 * KiB;
const TIMEOUT_S = 60;
const KILL_MS = 90_000;

async function node(args, killAfterMs = KILL_MS) {
  const child = spawn(process.execPath, args, { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env } });
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

/**
 * One block, and by default an output node. The output node matters to the
 * figures: the finish call carries the WHOLE workflow context as one JSON
 * string, and an output node holds the value twice more (`out`, `__output__`)
 * — three copies of a block's result cross, not one. `withOutput: false`
 * measures the single-copy ceiling.
 */
async function writeFlow(name, code, { withOutput = true } = {}) {
  const file = path.join(dir, name);
  const nodes = [{ id: 'b', type: 'logic_block', position: { x: 0, y: 0 }, data: { code } }];
  const edges = [];
  if (withOutput) {
    nodes.push({ id: 'out', type: 'output', position: { x: 200, y: 0 }, data: { value: '$nodes.b' } });
    edges.push({ id: 'edge', source: 'b', target: 'out' });
  }
  await fs.writeFile(file, JSON.stringify({ nodes, edges }));
  return file;
}

const rows = [];
let hung = 0;

/** Run one flow on the CLI and record what came back. */
async function measure(label, code, extraArgs = [], flowOpts = {}) {
  const flow = await writeFlow(`${rows.length}.json`, code, flowOpts);
  const outFile = path.join(dir, `${rows.length}.out.json`);
  const r = await node([cli, 'run', flow, '--quiet', '--timeout', String(TIMEOUT_S), '-o', outFile, ...extraArgs]);
  let payload = null;
  try {
    payload = JSON.parse(await fs.readFile(outFile, 'utf8'));
  } catch {
    /* no result file: the process was killed or died */
  }
  const value = payload?.results?.b;
  const row = {
    label,
    outcome: r.signal ? `KILLED after ${r.elapsed} ms (hang)` : payload ? payload.status : `exit ${r.code}, no result`,
    errorCode: payload?.errorCode ?? '',
    error: (payload?.error ?? '').replace(/\s+/g, ' ').slice(0, 140),
    value: typeof value === 'string' ? `string of ${value.length}` : value === undefined ? '' : JSON.stringify(value).slice(0, 80),
    ms: r.elapsed,
  };
  if (r.signal) hung++;
  rows.push(row);
  return row;
}

try {
  const str = (n) => `return 'x'.repeat(${n});`;
  await measure('900 KiB string result', str(900 * KiB));
  await measure('1.5 MiB string result', str(1.5 * MiB));
  await measure('20 MiB string result', str(20 * MiB));
  const single = { withOutput: false };
  await measure('1.5 MiB string result, no output node (one copy)', str(1.5 * MiB), [], single);
  await measure('3.9 MiB string result, no output node (one copy)', str(3.9 * MiB), [], single);
  await measure('4.5 MiB string result, no output node (one copy)', str(4.5 * MiB), [], single);
  await measure('2 MiB module result (memoryWrite → memoryRead, returns length)',
    `await Utility.memoryWrite('big', 'y'.repeat(${2 * MiB})); const v = await Utility.memoryRead('big'); return v.length;`);
  await measure('20 MiB module result (memoryWrite → memoryRead, returns length)',
    `await Utility.memoryWrite('big', 'y'.repeat(${20 * MiB})); const v = await Utility.memoryRead('big'); return v.length;`);
  const inputs = path.join(dir, 'inputs.json');
  await fs.writeFile(inputs, JSON.stringify({ big: 'z'.repeat(2 * MiB) }));
  await measure('2 MiB input literal (--inputs, returns length)', 'return inputs.big.length;', ['--inputs', inputs]);

  // --- host-typed results: an engine-level probe --------------------------
  const probeOut = path.join(distDir, '_zipp_limits_probe.mjs');
  const { commonBuildOptions } = await import('../esbuild.mjs');
  await esbuild.build({
    ...commonBuildOptions,
    entryPoints: [path.join(here, 'zipp-limits-probe.ts')],
    outfile: probeOut,
    logLevel: 'warning',
  });
  const probe = await node([probeOut], KILL_MS);
  if (probe.signal) {
    hung++;
    rows.push({ label: 'host-typed results probe', outcome: `KILLED after ${probe.elapsed} ms (hang)`, errorCode: '', error: probe.stderr.slice(-140), value: '', ms: probe.elapsed });
  } else {
    let recorded = [];
    try {
      recorded = JSON.parse(probe.stdout.split('\n').filter((l) => l.startsWith('{"rows":'))[0]).rows;
    } catch {
      rows.push({ label: 'host-typed results probe', outcome: `exit ${probe.code}, unreadable`, errorCode: '', error: probe.stderr.slice(-140), value: '', ms: probe.elapsed });
    }
    rows.push(...recorded);
  }

  // --- the table -----------------------------------------------------------
  const pad = (s, n) => String(s).padEnd(n);
  console.log('zipp-limits: measured (recorded, not asserted)');
  console.log(`  ${pad('case', 66)} ${pad('outcome', 10)} ${pad('errorCode', 12)} ${pad('ms', 6)} value / error`);
  for (const r of rows) {
    console.log(`  ${pad(r.label, 66)} ${pad(r.outcome, 10)} ${pad(r.errorCode, 12)} ${pad(r.ms, 6)} ${r.value}${r.error ? ` — ${r.error}` : ''}`);
  }
  if (hung) {
    console.error(`  FAIL ${hung} case(s) had to be killed: the boundary hung instead of failing`);
    process.exitCode = 1;
  } else {
    console.log(`PASS: ${rows.length} boundary cases completed or failed within their timeouts; none hung`);
  }
} finally {
  await fs.rm(dir, { recursive: true, force: true });
  for (const leftover of ['_zipp_limits_probe.mjs', '_zipp_limits_probe.mjs.map']) {
    await fs.rm(path.join(distDir, leftover), { force: true });
  }
}
