/**
 * Per-job module-context isolation test.
 *
 * The bundled module runtimes used to cache their RuntimeContext in a module-level
 * `let ctx` singleton, so two jobs running concurrently (maxConcurrency > 1) shared
 * one context — abort signals, streaming callbacks, and per-package permissions/
 * constants cross-contaminated, and JobManager had to clamp concurrency to 1.
 *
 * The fix: each module exports a `createMethods(ctx)` factory that returns FRESH
 * method closures bound to THIS job's context (+ any per-job state). This test
 * proves two instances built from the same module are fully isolated. As each
 * module is migrated, add it to MIGRATED below.
 *
 * Run: `npm test` (in cli/). No test runner needed — esbuild bundles + node runs it.
 */
import CoreUtility from '../../ui/src/bundled-modules/core-utility/runtime';
import CoreFilesystem from '../../ui/src/bundled-modules/core-filesystem/runtime';
import CoreDatabase from '../../ui/src/bundled-modules/core-database/runtime';
import CoreInput from '../../ui/src/bundled-modules/core-input/runtime';
import CoreService from '../../ui/src/bundled-modules/core-service/runtime';
import CoreAudio from '../../ui/src/bundled-modules/core-audio/runtime';
import CoreAI from '../../ui/src/bundled-modules/core-ai/runtime';
import CoreImage from '../../ui/src/bundled-modules/core-image/runtime';
import CoreFlowControl from '../../ui/src/bundled-modules/core-flow-control/runtime';
import CoreVideo from '../../ui/src/bundled-modules/core-video/runtime';
import CoreBrowser from '../../ui/src/bundled-modules/core-browser/runtime';

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyModule = { name: string; createMethods?: (ctx: any) => Record<string, any> };

const MIGRATED: AnyModule[] = [
  CoreUtility as unknown as AnyModule,
  CoreFilesystem as unknown as AnyModule,
  CoreDatabase as unknown as AnyModule,
  CoreInput as unknown as AnyModule,
  CoreService as unknown as AnyModule,
  CoreAudio as unknown as AnyModule,
  CoreAI as unknown as AnyModule,
  CoreImage as unknown as AnyModule,
  CoreFlowControl as unknown as AnyModule,
  CoreVideo as unknown as AnyModule,
  CoreBrowser as unknown as AnyModule,
];

let pass = 0;
let fail = 0;
const check = (name: string, cond: boolean) => {
  if (cond) pass++;
  else {
    fail++;
    console.log('  FAIL:', name);
  }
};

function mkCtx(tag: string, log: string[]) {
  return {
    log: (_lvl: string, msg: string) => log.push(`${tag}:${msg}`),
    hasPermission: () => true,
    getConstant: (n: string) => `${tag}-${n}`,
    abortSignal: { aborted: false },
  } as any;
}

for (const mod of MIGRATED) {
  if (!mod.createMethods) {
    check(`${mod.name} exposes createMethods`, false);
    continue;
  }
  const logA: string[] = [];
  const logB: string[] = [];
  const A = mod.createMethods(mkCtx('A', logA));
  const B = mod.createMethods(mkCtx('B', logB));
  // Distinct closures per instance (not shared singleton method refs).
  const firstKey = Object.keys(A)[0];
  check(`${mod.name}: distinct method closures`, A[firstKey] !== B[firstKey]);
}

// Core-Utility specifics: ctx + per-job memory isolation.
{
  const logA: string[] = [];
  const logB: string[] = [];
  const A = CoreUtility.createMethods!(mkCtx('A', logA) as any);
  const B = CoreUtility.createMethods!(mkCtx('B', logB) as any);
  (A.template as any)('{{x}}', { x: 'va' });
  (B.template as any)('{{y}}', { y: 'vb' });
  check('Utility: ctxA log not contaminated by B', logA.some((l) => l.startsWith('A:')) && !logA.some((l) => l.startsWith('B:')));
  check('Utility: ctxB log not contaminated by A', logB.some((l) => l.startsWith('B:')) && !logB.some((l) => l.startsWith('A:')));
  (A.memoryWrite as any)('k', 'value-A');
  (B.memoryWrite as any)('k', 'value-B');
  check('Utility: A.memory not contaminated by B', (A.memoryRead as any)('k') === 'value-A');
  check('Utility: B.memory not contaminated by A', (B.memoryRead as any)('k') === 'value-B');
}

console.log(`module-ctx-isolation: ${pass} passed, ${fail} failed (${MIGRATED.length}/11 modules migrated)`);
process.exit(fail ? 1 : 0);
