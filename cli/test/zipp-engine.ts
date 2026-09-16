/**
 * The CLI's engine is the ZIPP VM on a worker thread, and nothing else.
 *
 * `createCliEngine` is the only way the CLI builds a JobManager: it preloads
 * the staged WebAssembly artifact, checks it against the identity the build
 * recorded, and hands the runtime a `scriptExecutor` that spawns one
 * `worker_threads` Worker per script with `requireScriptExecutor: true`
 * forced last. This suite proves each link of that chain from the outside,
 * through the same `runFlow` the `run` and `worker` commands use:
 *
 *   * no artifact, or an artifact whose bytes differ by one bit, is
 *     `engine_unavailable` before any job is submitted;
 *   * flow code sees none of the host realm (`process`, `require`, `Buffer`,
 *     `__TAURI__`), and so cannot read `OAIY_SERVER_TOKEN` from an
 *     environment that holds it;
 *   * a caller cannot un-require the engine through the options spread;
 *   * a worker that dies is a failed run, not a hang;
 *   * a timed-out run is `aborted` and its worker is gone.
 *
 * Bundled and run by `build-engine-tests.mjs`, beside `dist/zipp/` and
 * `dist/oaiy-zipp-worker.mjs`, which `buildZippAssets()` stages first.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import type { WorkflowGraph } from 'oaiy-core';
import { runFlow, createCliEngine } from '../src/engine';
import { loadZippArtifact, EngineUnavailableError } from '../src/zipp/artifact';
import { createZippThreadExecutor } from '../src/zipp/thread-executor';
// The one raw `createEngine` in cli/test, and deliberately so: the worker-crash
// case needs an executor whose worker entry is a script that dies, which
// `createCliEngine` — correctly — gives no caller a way to supply.
import { createEngine } from '../../ui/src/engine/createEngine';
import { loadNodeBundledModules } from '../src/generated/bundled-modules';

let passed = 0;
let failed = 0;
function check(name: string, ok: boolean, detail?: string): void {
  if (ok) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

const TERMINAL = new Set(['completed', 'failed', 'aborted']);
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

function block(code: string): WorkflowGraph {
  return {
    nodes: [{ id: 'b', type: 'logic_block', position: { x: 0, y: 0 }, data: { code } }],
    edges: [],
  } as WorkflowGraph;
}

const REALM_PROBE = 'return [typeof process, typeof require, typeof Buffer, typeof globalThis.__TAURI__];';
const NO_REALM = JSON.stringify(['undefined', 'undefined', 'undefined', 'undefined']);

/** Drive one job on `engine` to a terminal state, bounded. */
function settle(engine: Awaited<ReturnType<typeof createCliEngine>>, jobId: string, boundMs: number) {
  return new Promise<{ status: string; error?: string; timedOut: boolean }>((resolve) => {
    let done = false;
    const finish = (timedOut: boolean) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      unsub();
      const job = engine.getJob(jobId);
      resolve({ status: job?.status ?? 'missing', error: job?.error, timedOut });
    };
    const check = () => {
      const job = engine.getJob(jobId);
      if (job && TERMINAL.has(job.status)) finish(false);
    };
    const unsub = engine.onStateChange(check);
    const timer = setTimeout(() => finish(true), boundMs);
    check();
  });
}

async function main(): Promise<void> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-zipp-engine-'));
  // A worker that lingers after the suite would keep this process alive: an
  // unref'd watchdog turns that hang into a bounded, named failure.
  setTimeout(() => {
    console.error('  FAIL a worker thread outlived the suite (the process did not exit on its own)');
    process.exit(1);
  }, 20_000).unref();

  try {
    // --- 1. The artifact, and its identity ---------------------------------
    const art = await loadZippArtifact();
    check('the staged artifact loads and compiles', art.module instanceof WebAssembly.Module);
    check('the identity names the engine', art.identity.name === 'zipp');
    check(
      'the identity is the sha256 of the bytes that were loaded',
      sha256(fs.readFileSync(art.wasmPath)) === art.identity.wasmSha256,
    );
    // The staged SOURCE.json is the installer's record; the identity was
    // taken from the same file at build time. Neither is a literal here.
    const staged = JSON.parse(fs.readFileSync(path.join(path.dirname(art.wasmPath), 'SOURCE.json'), 'utf8'));
    check(
      'the identity agrees with the staged SOURCE.json',
      art.identity.release === staged.release &&
        art.identity.revision === staged.revision &&
        art.identity.glueSha256 === staged.glueSha256 &&
        JSON.stringify(art.identity.languages) === JSON.stringify(staged.languages),
      `${JSON.stringify(art.identity)} vs ${JSON.stringify(staged)}`,
    );
    check('the same artifact is handed out for the process', (await loadZippArtifact()) === art);

    // --- 2. Fail closed: no artifact -------------------------------------
    const empty = path.join(tmp, 'empty');
    fs.mkdirSync(empty);
    await loadZippArtifact({ assetDir: empty }).then(
      () => check('an empty asset dir is refused', false, 'resolved'),
      (e) =>
        check(
          'an empty asset dir is refused as engine_unavailable',
          e instanceof EngineUnavailableError && e.code === 'engine_unavailable',
          String(e),
        ),
    );

    // --- 3. Fail closed: one flipped byte ----------------------------------
    const flipped = path.join(tmp, 'flipped');
    fs.mkdirSync(flipped);
    const bytes = fs.readFileSync(art.wasmPath);
    bytes[Math.floor(bytes.length / 2)] ^= 0x01;
    fs.writeFileSync(path.join(flipped, path.basename(art.wasmPath)), bytes);
    await loadZippArtifact({ assetDir: flipped }).then(
      () => check('a flipped byte is refused', false, 'resolved'),
      (e) =>
        check(
          'a flipped byte is refused as engine_unavailable',
          e instanceof EngineUnavailableError && e.code === 'engine_unavailable',
          String(e),
        ),
    );

    // --- 4. Through runFlow: engine_unavailable, and no job ---------------
    for (const [label, dir] of [
      ['a missing artifact', empty],
      ['a corrupt artifact', flipped],
    ] as const) {
      process.env.OAIY_ZIPP_ASSET_DIR = dir;
      try {
        const res = await runFlow(block('return 1;'), { timeoutMs: 10_000 });
        check(
          `${label} makes runFlow report engine_unavailable without running`,
          !res.success &&
            res.status === 'failed' &&
            res.errorCode === 'engine_unavailable' &&
            res.engine === 'zipp' &&
            res.jobId === '',
          JSON.stringify({ success: res.success, status: res.status, errorCode: res.errorCode, jobId: res.jobId, error: res.error }),
        );
      } finally {
        delete process.env.OAIY_ZIPP_ASSET_DIR;
      }
    }

    // --- 5. A real run: the host realm is out of reach --------------------
    process.env.OAIY_SERVER_TOKEN = 'canary-secret';
    try {
      const realm = await runFlow(block(REALM_PROBE), { timeoutMs: 30_000 });
      check(
        'a block sees no process, require, Buffer or __TAURI__',
        realm.status === 'completed' && JSON.stringify(realm.results?.['b']) === NO_REALM,
        `status=${realm.status} error=${realm.error ?? ''} got=${JSON.stringify(realm.results?.['b'])}`,
      );
      check('the result names the engine it ran on', realm.engine === 'zipp' && realm.errorCode === undefined);

      const token = await runFlow(
        block("return typeof process === 'undefined' ? 'no-process' : process.env.OAIY_SERVER_TOKEN;"),
        { timeoutMs: 30_000 },
      );
      check(
        'a flow cannot read OAIY_SERVER_TOKEN from the environment that holds it',
        token.status === 'completed' && token.results?.['b'] === 'no-process',
        `status=${token.status} got=${JSON.stringify(token.results?.['b'])}`,
      );
      check('and the token appears nowhere in the run result', !JSON.stringify(token).includes('canary-secret'));
    } finally {
      delete process.env.OAIY_SERVER_TOKEN;
    }

    // --- 5b. The Desktop's app-logic shape: a timer throws, the run goes on --
    // `app_logic.rs` runs each script as
    // `(new Function("ctx",decodeURIComponent(…)))(JSON.parse(…))` with the
    // try/catch INSIDE the encoded body. That body evaluates at the guest's
    // global scope, so it only sees the sandbox shims if they are top-level.
    // Before that was so, `setTimeout(fn, 10)` reached ZIPP's own timer, whose
    // callback never runs under the session's pump: the script reported
    // `ok:true` with its work silently dropped. Now it reports the TypeError,
    // and the next script still runs.
    const appLogic = (source: string, ctx: unknown) => {
      const pct = (s: string) =>
        Array.from(Buffer.from(s, 'utf8'), (b) => `%${b.toString(16).toUpperCase().padStart(2, '0')}`).join('');
      const body =
        `try {\n${source}\n;return JSON.stringify({ ok: true, value: run(ctx) });\n} ` +
        'catch (e) { return JSON.stringify({ ok: false, error: String((e && e.message) || e) }); }';
      return `(new Function("ctx",decodeURIComponent("${pct(body)}")))(JSON.parse(decodeURIComponent("${pct(JSON.stringify(ctx))}")))`;
    };
    const timerGraph = {
      nodes: [
        {
          id: 'timer',
          type: 'logic_block',
          position: { x: 0, y: 0 },
          data: { code: appLogic('function run(ctx) { setTimeout(function(){}, 10); return {}; }', { event: {} }) },
        },
        {
          id: 'next',
          type: 'logic_block',
          position: { x: 200, y: 0 },
          data: { code: appLogic("function run(ctx) { return { effects: [{ type: 'ui.toast', message: 'still here' }] }; }", {}) },
        },
      ],
      edges: [{ id: 'e', source: 'timer', target: 'next' }],
    } as WorkflowGraph;
    const timerRun = await runFlow(timerGraph, { timeoutMs: 30_000 });
    const parsed = (id: string) => {
      try {
        return JSON.parse(String(timerRun.results?.[id])) as { ok?: boolean; error?: string; value?: { effects?: { message?: string }[] } };
      } catch {
        return undefined;
      }
    };
    const timerOut = parsed('timer');
    const nextOut = parsed('next');
    check(
      'an app-logic script calling setTimeout reports ok:false with the sandbox TypeError',
      timerOut?.ok === false && /not available inside the Zipp sandbox/.test(timerOut?.error ?? ''),
      `status=${timerRun.status} error=${timerRun.error ?? ''} got=${JSON.stringify(timerRun.results?.['timer'])}`,
    );
    check(
      'and the NEXT script in the graph completes (one failing script does not end the run)',
      timerRun.status === 'completed' && nextOut?.ok === true && nextOut?.value?.effects?.[0]?.message === 'still here',
      `status=${timerRun.status} error=${timerRun.error ?? ''} got=${JSON.stringify(timerRun.results?.['next'])}`,
    );

    // --- 6. The spread cannot un-require the engine ------------------------
    await loadNodeBundledModules();
    const stubborn = await createCliEngine({
      scriptExecutor: null,
      requireScriptExecutor: false,
    } as never);
    const sid = stubborn.submit('test', 'un-require', block(REALM_PROBE), {});
    const sres = await settle(stubborn, sid, 30_000);
    check(
      'a caller passing scriptExecutor:null / requireScriptExecutor:false still runs on ZIPP',
      sres.status === 'completed' && JSON.stringify(stubborn.getJob(sid)?.nodeOutputs?.['b']) === NO_REALM,
      `status=${sres.status} error=${sres.error ?? ''} got=${JSON.stringify(stubborn.getJob(sid)?.nodeOutputs?.['b'])}`,
    );

    // --- 7. A worker that dies is a failed run, not a hang -----------------
    const crash = path.join(tmp, 'crash.mjs');
    fs.writeFileSync(crash, 'process.exit(3);\n');
    const crashing = createEngine({
      scriptExecutor: createZippThreadExecutor({ ...art, workerEntry: pathToFileURL(crash) }),
      requireScriptExecutor: true,
    });
    const cid = crashing.submit('test', 'crash', block('return 1;'), {});
    const cres = await settle(crashing, cid, 10_000);
    check(
      'a worker exiting non-zero fails the run',
      !cres.timedOut && cres.status === 'failed' && /exited with code 3/.test(cres.error ?? ''),
      `timedOut=${cres.timedOut} status=${cres.status} error=${cres.error ?? ''}`,
    );

    // --- 8. A timeout aborts the run and terminates the worker -------------
    // Each `await` renews the instruction budget, so this outlasts any timer;
    // only terminating the thread stops it.
    const t0 = Date.now();
    const timed = await runFlow(block("for (;;) { await Utility.memoryRead('k'); }"), {
      timeoutMs: 1_500,
      instructionBudgetSteps: 2_000_000_000,
    });
    check(
      'a run past its timeout is aborted with errorCode timeout',
      timed.status === 'aborted' && timed.errorCode === 'timeout' && !timed.success,
      `status=${timed.status} errorCode=${timed.errorCode} error=${timed.error ?? ''}`,
    );
    check('and the abort is prompt', Date.now() - t0 < 6_000, `${Date.now() - t0} ms`);

    // --- 9. The budget option is validated before any worker spawns --------
    await createCliEngine({ instructionBudgetSteps: 0 }).then(
      () => check('instructionBudgetSteps 0 is refused', false, 'resolved'),
      (e) => check('instructionBudgetSteps 0 is refused', e instanceof RangeError, String(e)),
    );
    await createCliEngine({ instructionBudgetSteps: 3_000_000_000 }).then(
      () => check('instructionBudgetSteps above the ceiling is refused', false, 'resolved'),
      (e) => check('instructionBudgetSteps above the ceiling is refused', e instanceof RangeError, String(e)),
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }

  console.log(`zipp-engine: ${passed} passed, ${failed} failed`);
  // No process.exit: the process must end on its own, or the watchdog above
  // reports the thread that kept it alive.
  process.exitCode = failed ? 1 : 0;
}

void main();
