/**
 * The engine-level half of test/zipp-limits.mjs: what a module method's
 * Buffer, Uint8Array and Date return values look like to guest code once they
 * have crossed to the ZIPP worker (structured clone) and into the VM
 * (`resolveHostCallback`'s JS→guest conversion). Bundled and run by
 * zipp-limits.mjs; prints one JSON line `{"rows":[…]}` on stdout. Records,
 * never asserts — see the caller for why.
 *
 * A `Probe` runtime module is registered beside the bundled ones for the
 * duration of this process. Its methods return the host-typed values; a
 * logic_block calls them the way any module method is called.
 */
import { getModuleLoader, loadBundledModules } from 'oaiy-core';
import type { BundledModule, ModuleManifest, RuntimeModule, WorkflowGraph } from 'oaiy-core';
import { createCliEngine } from '../src/engine';
import { loadNodeBundledModules } from '../src/generated/bundled-modules';

const TERMINAL = new Set(['completed', 'failed', 'aborted']);

const probeRuntime: RuntimeModule = {
  name: 'Probe',
  methods: {
    buffer: async (n: number) => Buffer.alloc(Number(n), 7),
    bytes: async (n: number) => new Uint8Array(Number(n)).fill(9),
    date: async () => new Date(0),
    nested: async () => ({ when: new Date(0), raw: Buffer.from('hi'), list: [1, 'two', null] }),
  },
};

const probeModule: BundledModule = {
  manifest: {
    id: 'test-probe',
    name: 'Test Probe',
    version: '0.0.0',
    description: 'host-typed return values for test/zipp-limits.mjs',
    nodes: [],
  } as unknown as ModuleManifest,
  nodes: [],
  runtime: probeRuntime,
};

function block(code: string): WorkflowGraph {
  return {
    nodes: [{ id: 'b', type: 'logic_block', position: { x: 0, y: 0 }, data: { code } }],
    edges: [],
  } as WorkflowGraph;
}

/** Describe a guest value the way a reader of the table needs it. */
function describe(v: unknown): string {
  if (v === null || typeof v !== 'object') return `${typeof v}: ${String(v).slice(0, 60)}`;
  const json = JSON.stringify(v);
  return `${Array.isArray(v) ? 'array' : 'object'} keys=${Object.keys(v).length} json=${json.length > 80 ? json.slice(0, 77) + '…' : json}`;
}

async function main(): Promise<void> {
  await loadNodeBundledModules();
  const loaded = await loadBundledModules(getModuleLoader(), [probeModule]);
  if (loaded.errors.length) throw new Error(`Probe module did not load: ${loaded.errors.join('; ')}`);

  const engine = await createCliEngine();
  const rows: Array<Record<string, unknown>> = [];
  const cases: Array<[string, string]> = [
    ['Buffer(16) host result', 'const v = await Probe.buffer(16); return [typeof v, v === null ? "null" : Object.prototype.toString.call(v), Array.isArray(v), v && v.length, v && v.byteLength, JSON.stringify(v).slice(0, 60)];'],
    ['Uint8Array(16) host result', 'const v = await Probe.bytes(16); return [typeof v, v === null ? "null" : Object.prototype.toString.call(v), Array.isArray(v), v && v.length, v && v.byteLength, JSON.stringify(v).slice(0, 60)];'],
    ['Date host result', 'const v = await Probe.date(); return [typeof v, v === null ? "null" : Object.prototype.toString.call(v), v instanceof Date, JSON.stringify(v)];'],
    ['nested {Date, Buffer, list} host result', 'const v = await Probe.nested(); return [typeof v, JSON.stringify(v).slice(0, 80), v && typeof v.when, v && typeof v.raw];'],
    ['Buffer(2 MiB) host result (length seen by the guest)', 'const v = await Probe.buffer(2097152); return [typeof v, v && v.length, v && v.byteLength];'],
  ];

  for (const [label, code] of cases) {
    const started = Date.now();
    const jobId = engine.submit('zipp-limits', label, block(code), {});
    const result = await new Promise<{ status: string; error?: string; value?: unknown; timedOut: boolean }>((resolve) => {
      let done = false;
      const finish = (timedOut: boolean) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        unsub();
        const job = engine.getJob(jobId);
        if (timedOut) {
          try { engine.abort(jobId); } catch { /* ignore */ }
        }
        resolve({ status: job?.status ?? 'missing', error: job?.error, value: job?.nodeOutputs?.['b'], timedOut });
      };
      const check = () => {
        const job = engine.getJob(jobId);
        if (job && TERMINAL.has(job.status)) finish(false);
      };
      const unsub = engine.onStateChange(check);
      const timer = setTimeout(() => finish(true), 30_000);
      check();
    });
    rows.push({
      label,
      outcome: result.timedOut ? 'TIMED OUT (30 s, aborted)' : result.status,
      errorCode: '',
      error: (result.error ?? '').replace(/\s+/g, ' ').slice(0, 140),
      value: result.value === undefined ? '' : describe(result.value),
      ms: Date.now() - started,
    });
  }

  process.stdout.write(JSON.stringify({ rows }) + '\n');
  // Let the process end on its own: a lingering worker would show up as a hang.
}

void main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
