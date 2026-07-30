/**
 * Per-job CROSS-MODULE isolation test.
 *
 * Proves `ctx.modules.<Module>.<method>` and `ctx.callModule(...)` resolve through
 * THIS job's runtime (its private moduleFunctionHandlers), never the shared
 * `globalThis[name]` — the residual that pinned maxConcurrency=1.
 *
 * The killer case is INTERLEAVED registration: two runtimes (two "jobs") each register
 * a Browser module, with job B registering LAST (so the shared globalThis.Browser holds
 * B's methods), then each registers a Scraper module that does a module-internal cross-
 * call. With the old globalThis path, job A's scraper would see job B's Browser
 * (last-writer-wins). With ctx.modules it must see its own.
 *
 * Run via `npm test` (esbuild bundles + node runs; module-types is type-only so the
 * vendor import resolves without DOM/Tauri).
 */
import { createRuntime } from '../../ui/vendor/oaiy-core/src/runtime';
import type {
  RuntimeContext,
  RuntimeModule,
  LoadedModule,
} from '../../ui/vendor/oaiy-core/src/module-types';
import { MODULE_CLEANUP } from '../../ui/vendor/oaiy-core/src/module-types';

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

function asLoaded(mod: RuntimeModule): LoadedModule {
  return {
    manifest: { id: mod.name.toLowerCase(), name: mod.name } as any,
    runtime: mod,
    nodes: new Map(),
    enabled: true,
    path: '',
  } as any;
}

// Fake 'Browser': createSessionV2 returns the per-job TAG constant + is async (real
// module methods are async — proves the Proxy returns the raw handler so the caller's
// await sees the correct job's result).
const browserMod: RuntimeModule = {
  name: 'Browser',
  methods: {},
  createMethods: (ctx: RuntimeContext) => ({
    createSessionV2: async () => ctx.getConstant?.('TAG'),
    createSessionV2Sync: () => ctx.getConstant?.('TAG'),
  }),
};

// Fake 'Scraper': does MODULE-INTERNAL cross-calls via ctx.modules / ctx.callModule.
const capturedCtx: Record<string, RuntimeContext> = {};
const scraperMod: RuntimeModule = {
  name: 'Scraper',
  methods: {},
  createMethods: (ctx: RuntimeContext) => {
    capturedCtx[ctx.getConstant?.('TAG') as string] = ctx;
    return {
      scrapeViaModules: () => (ctx.modules!.Browser as any).createSessionV2Sync(),
      scrapeViaModulesAsync: () => (ctx.modules!.Browser as any).createSessionV2(),
      scrapeViaCall: () => ctx.callModule!('Browser', 'createSessionV2Sync'),
    };
  },
};

const lateMod: RuntimeModule = {
  name: 'Late',
  methods: {},
  createMethods: (ctx: RuntimeContext) => ({ ping: () => 'late-' + ctx.getConstant?.('TAG') }),
};

async function main(): Promise<void> {
  const A = createRuntime(() => {});
  A.setProjectConstants({ TAG: 'A' });
  const B = createRuntime(() => {});
  B.setProjectConstants({ TAG: 'B' });

  // Capture ctx.modules.Late BEFORE Late is registered, to prove call-time resolution.
  // (Register Browser+Scraper first so ctx is captured.)
  await A.registerDynamicModule(asLoaded(browserMod));
  await B.registerDynamicModule(asLoaded(browserMod)); // globalThis.Browser is now B's
  await A.registerDynamicModule(asLoaded(scraperMod));
  await B.registerDynamicModule(asLoaded(scraperMod));

  const ctxA = capturedCtx['A'];
  const ctxB = capturedCtx['B'];

  // --- The core invariant: cross-module calls route per-job despite B registering last.
  check('A.modules.Browser sync → A', (ctxA.modules!.Browser as any).createSessionV2Sync() === 'A');
  check('B.modules.Browser sync → B', (ctxB.modules!.Browser as any).createSessionV2Sync() === 'B');
  check('A.callModule Scraper.scrapeViaModules → A', (ctxA.callModule!('Scraper', 'scrapeViaModules')) === 'A');
  check('B.callModule Scraper.scrapeViaModules → B', (ctxB.callModule!('Scraper', 'scrapeViaModules')) === 'B');
  check('A.callModule Scraper.scrapeViaCall → A', (ctxA.callModule!('Scraper', 'scrapeViaCall')) === 'A');

  // --- Async method: the awaited result is the calling job's, not the last writer's.
  check('A async cross-call awaits A', (await (ctxA.callModule!('Scraper', 'scrapeViaModulesAsync') as Promise<unknown>)) === 'A');
  check('B async cross-call awaits B', (await (ctxB.callModule!('Scraper', 'scrapeViaModulesAsync') as Promise<unknown>)) === 'B');

  // --- callModule primitive routes per-job.
  check('A.callModule Browser.createSessionV2Sync → A', ctxA.callModule!('Browser', 'createSessionV2Sync') === 'A');

  // --- Proxy quality: stable method identity + introspection (ownKeys).
  check('method identity stable across reads', (ctxA.modules!.Browser as any).createSessionV2Sync === (ctxA.modules!.Browser as any).createSessionV2Sync);
  check('ownKeys introspection', Object.keys(ctxA.modules!.Browser).includes('createSessionV2Sync'));
  check('has-trap (in operator)', 'createSessionV2Sync' in ctxA.modules!.Browser);

  // --- Late registration resolves at call time (a snapshot would miss it).
  const lateProbe = (ctxA.modules as any).Late; // accessed before Late is registered
  await A.registerDynamicModule(asLoaded(lateMod));
  check('Proxy resolves late-registered module', lateProbe.ping() === 'late-A');

  // --- callModule throws clearly for a missing method.
  let threw = false;
  try {
    ctxA.callModule!('Browser', 'nonexistentMethod');
  } catch {
    threw = true;
  }
  check('callModule throws for missing method', threw);

  // --- ctx.database wiring + per-flow routing (parallel-safe) -----------------
  // Regression guard: ctx.database used to be undefined, so every Database node
  // returned "Database not available". Now it routes through onDatabase carrying
  // THIS job's flowId, so concurrent jobs hit their own per-flow database.
  {
    const reqs: Array<{ flowId?: string; op: string }> = [];
    const mockDb = async (r: any) => {
      reqs.push({ flowId: r.flowId, op: r.operation });
      return { success: true, insertedId: 'id-' + r.flowId, rowsAffected: 1, data: [] };
    };
    const RA = createRuntime(() => {}, undefined, undefined, undefined, undefined, mockDb);
    RA.setProjectConstants({ TAG: 'DA' });
    RA.setFlowContext('flowA');
    const RB = createRuntime(() => {}, undefined, undefined, undefined, undefined, mockDb);
    RB.setProjectConstants({ TAG: 'DB' });
    RB.setFlowContext('flowB');
    let dctxA: RuntimeContext | undefined;
    let dctxB: RuntimeContext | undefined;
    const dbProbe: RuntimeModule = {
      name: 'DbProbe',
      methods: {},
      createMethods: (ctx: RuntimeContext) => {
        if (ctx.getConstant?.('TAG') === 'DA') dctxA = ctx;
        else dctxB = ctx;
        return { noop: () => 1 };
      },
    };
    await RA.registerDynamicModule(asLoaded(dbProbe));
    await RB.registerDynamicModule(asLoaded(dbProbe));
    check('ctx.database is wired (not undefined)', !!dctxA!.database && !!dctxB!.database);
    const idA = await (dctxA!.database as any).insertDocument('items', { v: 1 });
    const idB = await (dctxB!.database as any).insertDocument('items', { v: 2 });
    check('A db op carries flowId=flowA', reqs.some((r) => r.flowId === 'flowA' && r.op === 'insert'));
    check('B db op carries flowId=flowB', reqs.some((r) => r.flowId === 'flowB' && r.op === 'insert'));
    check('per-flow insert ids not crossed', idA === 'id-flowA' && idB === 'id-flowB');

    // find/update/delete return-shape mapping (deterministic via a richer mock).
    const reqs2: any[] = [];
    const mockDb2 = async (r: any) => {
      reqs2.push(r);
      if (r.operation === 'query') return { success: true, data: [{ id: 'x1', _id: 'x1', _created: 't0', foo: 'bar' }] };
      if (r.operation === 'update' || r.operation === 'delete') return { success: true, rowsAffected: 1 };
      return { success: true, insertedId: 'i1' };
    };
    const RC = createRuntime(() => {}, undefined, undefined, undefined, undefined, mockDb2);
    RC.setProjectConstants({ TAG: 'DC' });
    RC.setFlowContext('flowC');
    let dctxC: RuntimeContext | undefined;
    const probe2: RuntimeModule = {
      name: 'DbProbe2',
      methods: {},
      createMethods: (ctx: RuntimeContext) => {
        dctxC = ctx;
        return { noop: () => 1 };
      },
    };
    await RC.registerDynamicModule(asLoaded(probe2));
    const found = await (dctxC!.database as any).findDocuments('items', {});
    check('findDocuments maps {id,data,created_at}', found.length === 1 && found[0].id === 'x1' && found[0].created_at === 't0' && found[0].data.foo === 'bar');
    check('findDocuments strips meta from data', !('_id' in found[0].data) && !('_created' in found[0].data) && !('id' in found[0].data));
    check('updateDocument true on rowsAffected>0', (await (dctxC!.database as any).updateDocument('x1', { foo: 'baz' })) === true);
    check('deleteDocument true on rowsAffected>0', (await (dctxC!.database as any).deleteDocument('x1')) === true);
    check('update routed with flowId=flowC', reqs2.some((r) => r.operation === 'update' && r.flowId === 'flowC'));
  }

  // --- per-job MODULE_CLEANUP runs via cleanupDynamicModules --------------------
  // Regression guard for the critical leak: cleanupDynamicModules() must actually
  // invoke each module's per-job cleanup (e.g. core-browser closing Chromium). The
  // JobManager finally now calls it per job.
  {
    let cleaned = 0;
    const cleanupMod: RuntimeModule = {
      name: 'CleanupProbe',
      methods: {},
      createMethods: (_ctx: RuntimeContext) => {
        const m: Record<string | symbol, unknown> = { noop: () => 1 };
        m[MODULE_CLEANUP] = async () => {
          cleaned += 1;
        };
        return m as any;
      },
    };
    const R = createRuntime(() => {});
    await R.registerDynamicModule(asLoaded(cleanupMod));
    check('cleanup not run before cleanupDynamicModules', cleaned === 0);
    await R.cleanupDynamicModules();
    check('MODULE_CLEANUP runs via cleanupDynamicModules', cleaned === 1);
    await R.cleanupDynamicModules();
    check('cleanupDynamicModules is idempotent (map cleared)', cleaned === 1);
  }

  // --- modules Proxy throws a clear error for a missing method ------------------
  {
    let threwClear = false;
    try {
      (ctxA.modules!.Browser as any).totallyMissingMethod();
    } catch (e) {
      threwClear = String((e as Error).message).includes('not registered');
    }
    check('ctx.modules.X.missing() throws a descriptive error', threwClear);
    check('feature-detect missing method via `in` returns false', !('totallyMissingMethod' in ctxA.modules!.Browser));
  }

  console.log(`cross-module-isolation: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('cross-module-isolation crashed:', e);
  process.exit(1);
});
