/**
 * End-to-end parallel-jobs test.
 *
 * Proves the JobManager actually runs jobs CONCURRENTLY now that the capacity-1 clamp
 * is gone, and that two concurrent jobs keep isolated inputs/outputs. Run via `npm test`.
 *
 * Two checks:
 *  1. Capacity: submit 2 jobs in mode=parallel/maxConcurrency=2 and assert NEITHER is
 *     left 'queued' right after submit — i.e. both started (the old clamp would have
 *     queued the 2nd behind the 1st).
 *  2. Isolation: each job runs a distinct template flow; assert each job's output is its
 *     OWN, not the other's (no cross-contamination through the engine).
 */
import '../src/node-host/core'; // installs window.__TAURI__ before shared code reads it
import { createEngine } from '../../ui/src/engine/createEngine';
import { loadNodeBundledModules } from '../src/generated/bundled-modules';

/* eslint-disable @typescript-eslint/no-explicit-any */
let pass = 0;
let fail = 0;
const check = (name: string, cond: boolean) => {
  if (cond) pass++;
  else {
    fail++;
    console.log('  FAIL:', name);
  }
};

const TERMINAL = new Set(['completed', 'failed', 'aborted']);

function templateFlow(text: string): any {
  return {
    nodes: [
      { id: 'tpl', type: 'template', position: { x: 0, y: 0 }, data: { template: text } },
      { id: 'out', type: 'output', position: { x: 300, y: 0 }, data: { label: 'result', outputType: 'text' } },
    ],
    edges: [{ id: 'e1', source: 'tpl', target: 'out', sourceHandle: 'result', targetHandle: 'result' }],
  };
}

async function main(): Promise<void> {
  await loadNodeBundledModules();
  const engine = createEngine({
    config: { mode: 'parallel', maxConcurrency: 2 },
    networkPermissionHandler: async () => ({ allowed: true, remember: false }),
  });

  const idA = engine.submit('test', 'job-ALPHA', templateFlow('out-ALPHA'), {});
  const idB = engine.submit('test', 'job-BETA', templateFlow('out-BETA'), {});

  // (1) Concurrency: right after submitting both, BOTH should be active (started,
  // not terminal, not queued) at the same instant — i.e. genuinely running in
  // parallel. The old capacity-1 clamp would leave the 2nd parked in 'queued' until
  // the 1st finished. (Execution is async — both are admitted to 'running' here, then
  // their worker round-trips proceed concurrently.)
  const activeNow = [idA, idB].filter((id) => {
    const s = engine.getJob(id)?.status;
    return s != null && !TERMINAL.has(s) && s !== 'queued';
  }).length;
  const statusA0 = engine.getJob(idA)?.status;
  const statusB0 = engine.getJob(idB)?.status;
  check(`2 jobs concurrently active right after submit (A=${statusA0}, B=${statusB0})`, activeNow === 2);

  await new Promise<void>((resolve) => {
    const done = () => {
      if (TERMINAL.has(engine.getJob(idA)?.status ?? '') && TERMINAL.has(engine.getJob(idB)?.status ?? '')) {
        clearTimeout(timer);
        u();
        resolve();
      }
    };
    const u = engine.onStateChange(done);
    const timer = setTimeout(() => {
      u();
      resolve();
    }, 30000);
    done();
  });

  const jobA = engine.getJob(idA);
  const jobB = engine.getJob(idB);
  const outA = (jobA?.nodeOutputs as any)?.out;
  const outB = (jobB?.nodeOutputs as any)?.out;

  check(`A completed (status=${jobA?.status})`, jobA?.status === 'completed');
  check(`B completed (status=${jobB?.status})`, jobB?.status === 'completed');
  // (2) Isolation: each job's output is its own — not swapped/contaminated.
  check(`A output is its own ("${outA}")`, outA === 'out-ALPHA');
  check(`B output is its own ("${outB}")`, outB === 'out-BETA');

  console.log(`parallel-jobs: ${pass} passed, ${fail} failed (concurrently active at submit: ${activeNow})`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('parallel-jobs crashed:', e);
  process.exit(1);
});
