// A tripwire on every way this process could turn a string into code:
// `Function` (called or constructed, directly or through the constructor of an
// async, generator or async-generator function), indirect `eval`, the
// `node:vm` compilers, a `worker_threads` Worker asked to run a STRING
// (`{ eval: true }`), and `process.binding` / `process._linkedBinding` /
// `internalBinding`, which reach `contextify` under the public API.
//
// The last three were a review's finding: a proxy on `Function` and `eval`
// sees anything that RESOLVES to them, but a Worker with `eval: true` compiles
// its source in another thread that this process never asks `Function` for,
// and `process.binding('contextify')` hands back the compiler itself. Guard A
// flags all three in the bytes; this catches them if they ever run.
//
// `import('data:…')` is the one that stays uncovered here: a data: module is
// compiled by the loader, in C++, with no JS entry point to proxy. Guard A
// (test/no-host-eval.mjs) is the control for it, and `--experimental-loader`
// was not worth a permanent hook in a guard. Stated rather than implied.
//
// Loaded ahead of the program with
//
//     node --import <this file> dist/oaiy.mjs run <flow> …
//
// and, because `--import` is in `execArgv`, ahead of every `worker_threads`
// Worker the program spawns — the ZIPP worker included. Each counted call is
// written to stderr as one line, keyed by `threadId`, so the main thread's
// count and a worker's cannot be mistaken for each other and no thread needs
// an environment (the ZIPP worker starts with none) or a shared file to
// report. test/zipp-guard.mjs parses those lines.
//
//     __host_eval_tripwire__ loaded {"threadId":0}
//     __host_eval_tripwire__ call {"threadId":0,"what":"new Function","head":"host,console,\"use strict\";…"}
//     __host_eval_tripwire__ summary {"threadId":0,"count":1}
//
// The one call that is not counted: oaiy-core's logger asking whether
// `import.meta` exists, by its exact source text. It is the only permanent
// entry on test/no-host-eval.mjs's allowlist, and it runs when the bundle
// loads, engine or no engine, so counting it would tell the guard nothing.
import vm from 'node:vm';
import { syncBuiltinESMExports } from 'node:module';
import workerThreads, { threadId } from 'node:worker_threads';

export const MARK = '__host_eval_tripwire__';
const LOGGER_PROBE = 'try { return import.meta } catch(e) { return undefined }';

let count = 0;
const write = (kind, payload) => {
  process.stderr.write(`${MARK} ${kind} ${JSON.stringify({ threadId, ...payload })}\n`);
};
const record = (what, args) => {
  count++;
  const head = args.map((a) => (typeof a === 'string' ? a : String(a))).join(',').slice(0, 96);
  write('call', { what, head });
};
const isLoggerProbe = (args) => args.length === 1 && args[0] === LOGGER_PROBE;

// --- Function, and the constructors reachable from function objects --------
const RealFunction = globalThis.Function;
const trip = (label, Real) =>
  new Proxy(Real, {
    apply(target, thisArg, args) {
      if (!isLoggerProbe(args)) record(`${label}()`, args);
      return Reflect.apply(target, thisArg, args);
    },
    construct(target, args, newTarget) {
      if (!isLoggerProbe(args)) record(`new ${label}`, args);
      return Reflect.construct(target, args, newTarget);
    },
  });

const FunctionTrip = trip('Function', RealFunction);
globalThis.Function = FunctionTrip;
// `(function () {}).constructor(...)`: the prototype's own `constructor` slot.
Object.defineProperty(RealFunction.prototype, 'constructor', { value: FunctionTrip, writable: true, configurable: true });

for (const [label, sample] of [
  ['AsyncFunction', async function () {}],
  ['GeneratorFunction', function* () {}],
  ['AsyncGeneratorFunction', async function* () {}],
]) {
  const proto = Object.getPrototypeOf(sample);
  const Real = proto.constructor;
  Object.defineProperty(proto, 'constructor', { value: trip(label, Real), writable: false, configurable: true });
}

// --- eval (indirect; a direct `eval(...)` becomes an ordinary call to this) --
const RealEval = globalThis.eval;
globalThis.eval = (...args) => {
  record('eval()', args);
  return RealEval(...args);
};

// --- node:vm ------------------------------------------------------------------
for (const name of ['runInThisContext', 'runInContext', 'runInNewContext', 'compileFunction']) {
  const real = vm[name];
  vm[name] = function (...args) {
    record(`vm.${name}()`, args);
    return real.apply(this, args);
  };
}
vm.Script = trip('vm.Script', vm.Script);

// --- worker_threads: a Worker whose source is a STRING ----------------------
// Only `{ eval: true }` is recorded. The CLI spawns its own ZIPP and script
// workers through this same constructor, by path, and those are not host code
// compilation — recording them would make the canary's "0 calls" meaningless.
// `Reflect.construct` with `newTarget`, so a subclass of Worker still works.
{
  const RealWorker = workerThreads.Worker;
  workerThreads.Worker = new Proxy(RealWorker, {
    construct(target, args, newTarget) {
      if (args[1] && typeof args[1] === 'object' && args[1].eval === true) {
        record('new Worker({ eval: true })', [String(args[0]).slice(0, 96)]);
      }
      return Reflect.construct(target, args, newTarget);
    },
  });
}
syncBuiltinESMExports();

// --- process.binding: Node's own bindings, contextify among them ------------
// Deprecated but present on Node 24. `internalBinding` is not reachable
// without `--expose-internals`, so it is wrapped only if something has put it
// on the global — the gap is real and named here rather than papered over.
for (const name of ['binding', '_linkedBinding']) {
  const real = process[name];
  if (typeof real !== 'function') continue;
  process[name] = function (...args) {
    record(`process.${name}()`, args);
    return real.apply(this, args);
  };
}
if (typeof globalThis.internalBinding === 'function') {
  const realInternal = globalThis.internalBinding;
  globalThis.internalBinding = function (...args) {
    record('internalBinding()', args);
    return realInternal.apply(this, args);
  };
}

write('loaded', {});
process.on('exit', () => write('summary', { count }));
