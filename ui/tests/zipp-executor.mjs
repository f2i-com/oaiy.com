/**
 * The Zipp sandbox, exercised against the real vendored WebAssembly module.
 *
 *     npm run test:zipp
 *
 * This is not a mock. It loads `vendor/zipp-wasm/zipp_wasm_bg.wasm`, drives it
 * with the same `ZippSession` the Worker uses, and feeds it the exact program
 * shape `runtime.ts` emits — the `getModulesShim` output, `function*
 * __run_workflow`, the `__step` trampoline, and `HARDENED_SHADOW_PREAMBLE`.
 * The point is to catch the failures that only appear against a real engine:
 * a deadlocked pump, a lost `'use strict'`, a capability that turned out to be
 * reachable after all.
 *
 * Requires Node 22.18+ / 24+ — it imports `zipp-executor.ts` directly and
 * relies on native type stripping rather than a build step.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  buildZippScript,
  detectZippPreambleCollision,
  ZippSession,
} from '../vendor/oaiy-core/src/zipp-executor.ts';
// The REAL trampoline, not a copy. A copy is how the `__gen.throw` resume bug
// stayed invisible: the test agreed with itself.
import { WORKFLOW_TRAMPOLINE } from '../vendor/oaiy-core/src/workflow-trampoline.ts';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VENDOR = path.join(HERE, '..', 'vendor', 'zipp-wasm');

let pass = 0;
const failures = [];
function check(name, ok, detail = '') {
  if (ok) {
    pass++;
    console.log(`  ok  ${name}`);
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`);
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

// ---------------------------------------------------------------------------
// 1. The vendored artifact is the one we verified.
// ---------------------------------------------------------------------------
// The engine is committed bytes, so nothing else would notice a swap. The
// README records the SHA of a build that passed Zipp's own
// `check-wasm-memory.cjs` — the gate proving the module links a 1 GiB maximum,
// ABOVE the VM's 512 MiB accounting limit. Built the other way round the
// accounting limit can never fire, and guest heap exhaustion becomes an
// unrecoverable trap instead of a catchable RangeError. Pinning the SHA is how
// that property survives a careless refresh.
console.log('\nvendored artifact');
const wasmBytes = fs.readFileSync(path.join(VENDOR, 'zipp_wasm_bg.wasm'));
const actualSha = createHash('sha256').update(wasmBytes).digest('hex');
const readme = fs.readFileSync(path.join(VENDOR, 'README.md'), 'utf8');
const declaredSha = (readme.match(/`([0-9a-f]{64})`/) || [])[1];
check(
  'README SHA-256 matches the committed .wasm',
  declaredSha === actualSha,
  declaredSha ? `README ${declaredSha.slice(0, 16)}… vs file ${actualSha.slice(0, 16)}…` : 'no SHA in README',
);

const { default: init, Engine } = await import(
  pathToFileURL(path.join(VENDOR, 'zipp_wasm.js')).href
);
await init({ module_or_path: wasmBytes });

// ---------------------------------------------------------------------------
// Harness: the OAIY program shape, and a broker that answers it.
// ---------------------------------------------------------------------------

/** What `getModulesShim()` emits, for the modules a test needs. */
function modulesShim(names) {
  return names
    .map(
      ([mod, method]) => `let ${mod} = globalThis.${mod} || {};
${mod}["${method}"] = function(...args) {
  let jsonArgs = JSON.stringify(args);
  return { _kind: "${mod}.${method}", _args: [jsonArgs] };
};`,
    )
    .join('\n');
}

/** The wrapper `executeScript` puts around a compiled flow body, verbatim. */
function fullScript(body, { shim = [['Utility', 'httpRequest']], hardened = true } = {}) {
  const shadow = hardened
    ? `var fetch = undefined; var XMLHttpRequest = undefined; var WebSocket = undefined;
var document = undefined; var window = undefined; var self = undefined;
var localStorage = undefined; var navigator = undefined; var importScripts = undefined;
var Worker = undefined; var require = undefined; var process = undefined;
var setTimeout = undefined; var queueMicrotask = undefined;
var postMessage = undefined; var addEventListener = undefined; var close = undefined;`
    : '';
  return `${hardened ? `'use strict';\n` : ''}${modulesShim(shim)}
${shadow}
function* __run_workflow() {
   let __res = null;
   try {
${body}
       __res = workflow_context;
   } catch (e) {
       host.call("__system.finish_error", ["" + e], function(r){});
       return;
   }
   return __res;
}

${WORKFLOW_TRAMPOLINE}
`;
}

/**
 * Run one workflow the way the Worker does. `broker` receives (kind, args) and
 * may be async — the round trip is what proves the guest survives a real await.
 */
async function runFlow(body, { broker, hardened = true, shim, abortAfter } = {}) {
  const engine = new Engine();
  const logs = [];
  let value, error;
  let calls = 0;

  const session = new ZippSession(engine, {
    onHostCall: async (kind, args) => {
      calls++;
      if (abortAfter && calls >= abortAfter) session.abort();
      await new Promise(r => setTimeout(r, 1));
      return broker(kind, JSON.parse(args[0] ?? '[]'));
    },
    onConsole: (level, args) => logs.push(`${level}:${args.join(' ')}`),
    onFinish: v => { value = v; },
    onError: m => { error = m; },
  });

  try {
    await session.run(buildZippScript(fullScript(body, { shim, hardened }), hardened));
  } finally {
    try { engine.dispose(); } catch { /* trapped instance */ }
  }
  return { value, error, logs, calls };
}

const okBroker = () => ({ status: 200, body: 'alpha beta' });

// ---------------------------------------------------------------------------
// 2. The pump loop actually completes a workflow.
// ---------------------------------------------------------------------------
console.log('\npump loop');
{
  const { value, error, calls } = await runFlow(
    `       let workflow_context = {};
       const a = yield Utility.httpRequest("https://one.test");
       workflow_context.a = a.body;
       const words = [];
       for (const w of a.body.split(" ")) {
         const r = yield Utility.httpRequest(w);
         words.push(r.body.length);
       }
       workflow_context.total = words.reduce((x, y) => x + y, 0);`,
    { broker: okBroker },
  );
  check('workflow completes across real async host round-trips', !error, error);
  check('result is correct', value?.a === 'alpha beta' && value?.total === 20,
    JSON.stringify(value));
  check('every yield became a host call', calls === 3, `calls=${calls}`);
}

// ---------------------------------------------------------------------------
// 3. Console: level and ordering survive.
// ---------------------------------------------------------------------------
console.log('\nconsole');
{
  const { logs, error } = await runFlow(
    `       let workflow_context = {};
       console.log("before");
       yield Utility.httpRequest("x");
       console.warn("after");
       console.error("last", 42);`,
    { broker: okBroker },
  );
  check('no error', !error, error);
  check(
    'levels preserved and interleaved in order',
    logs.join('|') === 'log:before|warn:after|error:last 42',
    logs.join('|'),
  );
}

// ---------------------------------------------------------------------------
// 4. Broker errors reach the script as catchable exceptions.
// ---------------------------------------------------------------------------
console.log('\nerror propagation');
{
  const { value, error } = await runFlow(
    `       let workflow_context = {};
       try {
         yield Utility.httpRequest("boom");
         workflow_context.caught = "no";
       } catch (e) {
         workflow_context.caught = e.message;
       }`,
    { broker: () => ({ __error__: 'permission denied' }) },
  );
  check('__error__ becomes a catchable throw at the await site', !error, error);
  check('the script caught the right message', value?.caught === 'permission denied',
    JSON.stringify(value));
}
{
  const { error } = await runFlow(
    `       let workflow_context = {};
       null.boom;`,
    { broker: okBroker },
  );
  check('an uncaught guest error is reported', !!error && /TypeError/.test(error), error);
}

// ---------------------------------------------------------------------------
// 5. Hardened semantics match the in-thread path.
// ---------------------------------------------------------------------------
// The wrapper exists to make this true. Without it Zipp compiles one Program,
// `'use strict'` loses its prologue position, and the shadow preamble
// overwrites Zipp's own bridges instead of shadowing bindings.
console.log('\nhardened semantics');
{
  const { value, error } = await runFlow(
    `       let workflow_context = {};
       yield Utility.httpRequest("x");
       const recovered = Function("return this")();
       workflow_context.probes = {
         strictThis: (function(){ return typeof this; })(),
         frozenWriteThrows: (function(){ try { const o = Object.freeze({a:1}); o.a = 2; return false; } catch (e) { return true; } })(),
         shadowedFetch: typeof fetch,
         shadowedPostMessage: typeof postMessage,
         syncBridge: typeof __zippHostCall,
         recoveredFetch: typeof recovered.fetch,
         recoveredWorker: typeof recovered.Worker,
         recoveredProcess: typeof recovered.process,
         recoveredImportScripts: typeof recovered.importScripts,
       };`,
    { broker: okBroker },
  );
  check('no error', !error, error);
  const p = value?.probes ?? {};
  check('strict mode is active (this === undefined)', p.strictThis === 'undefined', p.strictThis);
  check('strict mode is active (frozen write throws)', p.frozenWriteThrows === true);
  check('shadow preamble applies (fetch)', p.shadowedFetch === 'undefined');
  check('shadow preamble applies (postMessage)', p.shadowedPostMessage === 'undefined');
  check("Zipp's sync host bridge is shadowed", p.syncBridge === 'undefined');
  check('a RECOVERED global has no fetch', p.recoveredFetch === 'undefined');
  check('a RECOVERED global has no Worker', p.recoveredWorker === 'undefined');
  check('a RECOVERED global has no process', p.recoveredProcess === 'undefined');
  check('a RECOVERED global has no importScripts', p.recoveredImportScripts === 'undefined');
}

// ---------------------------------------------------------------------------
// 5b. Trusted flows: host-realm shims, and timers that throw instead of trap.
// ---------------------------------------------------------------------------
// A user's own flow gets no shadow preamble, so this is the surface a code node
// actually sees under Zipp. The timer case is the one that matters: Zipp's own
// setTimeout with a delay tries to sleep a thread WebAssembly does not have and
// panics the instance, so the wrapper has to replace it before user code runs.
console.log('\ntrusted-flow shims');
{
  const { value, error } = await runFlow(
    `       let workflow_context = {};
       yield Utility.httpRequest("x");
       const bytes = new TextEncoder().encode("héllo 🌍");
       const text = new TextDecoder().decode(bytes);
       let micro = "not yet";
       queueMicrotask(() => { micro = "ran"; });
       yield Utility.httpRequest("y");            // re-entry drains microtasks
       const errs = {};
       for (const [name, fn] of [
         ["setTimeout", () => setTimeout(() => {}, 100)],
         ["setInterval", () => setInterval(() => {}, 100)],
         ["fetch", () => fetch("https://x.test")],
         ["crypto", () => crypto.randomUUID()],
       ]) { try { fn(); errs[name] = "no throw"; } catch (e) { errs[name] = e.message; } }
       workflow_context.probe = {
         utf8Bytes: bytes.length,
         roundTrip: text,
         b64: btoa("hi there"),
         b64Back: atob(btoa("hi there")),
         clone: structuredClone({ a: [1, { b: 2 }] }),
         perfIsNumber: typeof performance.now() === "number",
         micro,
         errs,
         strictThis: (function(){ return typeof this; })(),
       };`,
    { broker: okBroker, hardened: false },
  );
  check('trusted flow ran (no trap)', !error, error);
  const p = value?.probe ?? {};
  check('TextEncoder produces UTF-8 bytes', p.utf8Bytes === 11, `bytes=${p.utf8Bytes}`);
  check('TextDecoder round-trips multi-byte text and emoji', p.roundTrip === 'héllo 🌍', p.roundTrip);
  check('btoa matches the platform', p.b64 === 'aGkgdGhlcmU=', p.b64);
  check('atob inverts btoa', p.b64Back === 'hi there', p.b64Back);
  check('structuredClone clones plain data', JSON.stringify(p.clone) === '{"a":[1,{"b":2}]}', JSON.stringify(p.clone));
  check('performance.now works', p.perfIsNumber === true);
  check('queueMicrotask callbacks run at the next re-entry', p.micro === 'ran', p.micro);
  check('setTimeout throws a clear error instead of trapping', /not available inside the Zipp sandbox/.test(p.errs?.setTimeout ?? ''), p.errs?.setTimeout);
  check('setInterval throws a clear error', /not available/.test(p.errs?.setInterval ?? ''), p.errs?.setInterval);
  check('fetch throws a clear error naming the alternative', /HTTP Request node/.test(p.errs?.fetch ?? ''), p.errs?.fetch);
  check('crypto does not silently degrade to Math.random', /entropy/.test(p.errs?.crypto ?? ''), p.errs?.crypto);
  check('trusted flow is NOT strict (matches the in-thread path)', p.strictThis === 'object', p.strictThis);
}
{
  // The hardened surface is unchanged by the shims: the shadow preamble that
  // follows them wins, so a package flow still sees `undefined`, not a stub.
  const { value, error } = await runFlow(
    `       let workflow_context = {};
       yield Utility.httpRequest("x");
       workflow_context.probe = { fetch: typeof fetch, setTimeout: typeof setTimeout, TextEncoder: typeof TextEncoder };`,
    { broker: okBroker },
  );
  check('hardened: fetch is still undefined (preamble wins over shim)', !error && value?.probe?.fetch === 'undefined', JSON.stringify(value?.probe));
  check('hardened: setTimeout is still undefined', value?.probe?.setTimeout === 'undefined');
  check('hardened: pure helpers remain available', value?.probe?.TextEncoder === 'function');
}

// ---------------------------------------------------------------------------
// 6. A runaway script stops on its own.
// ---------------------------------------------------------------------------
// The V8 path has no equivalent: an infinite loop there pins the Worker until
// the user aborts. Zipp's instruction budget is renewed per re-entry, so this
// bounds one step rather than the whole workflow.
console.log('\ninstruction budget');
{
  const started = Date.now();
  const { error } = await runFlow(
    `       let workflow_context = {};
       yield Utility.httpRequest("x");
       let i = 0;
       while (true) { i++; }`,
    { broker: okBroker },
  );
  check('an infinite loop is stopped', !!error && /budget/i.test(error), error);
  check('stopped promptly', Date.now() - started < 30_000, `${Date.now() - started}ms`);
}

// ---------------------------------------------------------------------------
// 7. Guards.
// ---------------------------------------------------------------------------
console.log('\nguards');
check(
  'a module name colliding with the preamble is detected',
  detectZippPreambleCollision(['Utility', 'db']) === 'db',
);
check(
  'ordinary module names pass',
  detectZippPreambleCollision(['Utility', 'Agent', 'Abort']) === null,
);
{
  // A script that neither finishes nor calls out would hang the job forever.
  // Zipp has no event loop, so nothing can wake it — the session has to say so.
  const engine = new Engine();
  let error;
  const session = new ZippSession(engine, {
    onHostCall: async () => ({}),
    onConsole: () => {},
    onFinish: () => {},
    onError: m => { error = m; },
  });
  await session.run(buildZippScript('var idle = 1;', true));
  try { engine.dispose(); } catch { /* ignore */ }
  check('a script that never signals completion is reported, not hung',
    !!error && /stalled/i.test(error), error);
}

// ---------------------------------------------------------------------------
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
assert.equal(failures.length, 0);
