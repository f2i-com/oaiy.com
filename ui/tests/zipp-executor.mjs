/**
 * The Zipp sandbox, exercised against the real installed WebAssembly modules.
 *
 *     npm run test:zipp                               both installs, one process each
 *     ZIPP_VENDOR=zipp-wasm-python npm run test:zipp  one of them
 *
 * `scripts/fetch-zipp-release.mjs` installs both from one zipp.org release:
 * `vendor/zipp-wasm` (JavaScript only, the browser Worker's engine) and
 * `vendor/zipp-wasm-python` (for the CLI, Desktop and headless server). Both
 * must run OAIY's JavaScript the same way, so both get the whole suite.
 *
 * This is not a mock. It loads the install's `zipp_wasm_bg.wasm`, drives it
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
import { spawnSync } from 'node:child_process';
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
import { checkInstall, VARIANTS } from '../../scripts/fetch-zipp-release.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// The suite below drives one engine per process; with no ZIPP_VENDOR it runs
// once for each install and fails if either does.
if (!process.env.ZIPP_VENDOR) {
  let failed = false;
  for (const { folder } of VARIANTS) {
    console.log(`\n=== ${folder} ===`);
    const run = spawnSync(process.execPath, [...process.execArgv, fileURLToPath(import.meta.url)], { stdio: 'inherit', env: { ...process.env, ZIPP_VENDOR: folder } });
    if (run.status !== 0) failed = true;
  }
  process.exit(failed ? 1 : 0);
}
const VARIANT = VARIANTS.find(v => v.folder === process.env.ZIPP_VENDOR);
if (!VARIANT) {
  console.error(`ZIPP_VENDOR must name an install: ${VARIANTS.map(v => v.folder).join(' or ')}`);
  process.exit(1);
}
const VENDOR = path.join(HERE, '..', 'vendor', VARIANT.folder);

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
// 1. The installed artifact is the one the release verified.
// ---------------------------------------------------------------------------
// The engine is not committed: the installer takes it from a zipp.org release,
// checked against ZIPP's SHA256SUMS and the bundle's own, and records what it
// took in SOURCE.json. Every identity here comes from that record, never from
// a literal, so a new ZIPP release needs no edit to this file.
console.log(`\ninstalled artifact (${VARIANT.folder})`);
let source;
try {
  source = checkInstall(VENDOR, VARIANT);
  check('the install checks against the release it was taken from', true);
} catch (error) {
  check('the install checks against the release it was taken from', false, error.message);
  console.error(`\n${pass} passed, ${failures.length} failed (nothing else can run without the install)`);
  process.exit(1);
}
const wasmBytes = fs.readFileSync(path.join(VENDOR, 'zipp_wasm_bg.wasm'));
check('SOURCE.json records the .wasm beside it',
  createHash('sha256').update(wasmBytes).digest('hex') === source.sha256);

const checksums = new Map(fs.readFileSync(path.join(VENDOR, 'SHA256SUMS'), 'utf8')
  .trim().split(/\r?\n/).map(line => {
    const [digest, name] = line.trim().split(/\s+/, 2);
    return [name, digest];
  }));
for (const name of ['zipp_wasm.js', 'zipp_wasm.d.ts', 'zipp_wasm_bg.wasm', 'zipp_wasm_bg.wasm.d.ts', 'PROFILE.json', 'BUILD-INFO.txt']) {
  check(`${name} matches the release bundle's checksum`,
    createHash('sha256').update(fs.readFileSync(path.join(VENDOR, name))).digest('hex') === checksums.get(name));
}

const { default: init, Engine, zippProfile } = await import(
  pathToFileURL(path.join(VENDOR, 'zipp_wasm.js')).href
);
await init({ module_or_path: wasmBytes });
const profile = JSON.parse(zippProfile());
check('running engine matches the shipped release profile',
  JSON.stringify(profile) === JSON.stringify(JSON.parse(fs.readFileSync(path.join(VENDOR, 'PROFILE.json'), 'utf8'))));
check('running engine is the recorded release, commit and languages, with the sandbox profile',
  profile.version === source.version && profile.source?.sha === source.revision
    && JSON.stringify(profile.languages) === JSON.stringify(source.languages) && profile.features.includes('safe-sandbox'),
  `${profile.version} ${profile.source?.sha} ${JSON.stringify(profile.languages)} vs SOURCE.json ${source.version} ${source.revision} ${JSON.stringify(source.languages)}`);

// A module that links a 1 GiB maximum, ABOVE the VM's 512 MiB accounting
// limit, reports guest heap exhaustion as a catchable RangeError. Linked the
// other way round the accounting limit can never fire and exhaustion becomes an
// unrecoverable trap. Zipp's `check-wasm-memory.cjs` gates its builds on this;
// a release OAIY did not build is held to it here, from the module's own bytes.
const linkedMax = wasmMemoryMaxBytes(wasmBytes);
check('the module links a memory maximum above the heap accounting limit',
  linkedMax !== undefined && linkedMax === profile.limits?.linkedMemoryMaxBytes && linkedMax > profile.limits?.approxHeapBytes,
  `linked ${linkedMax}, profile linkedMemoryMaxBytes ${profile.limits?.linkedMemoryMaxBytes}, approxHeapBytes ${profile.limits?.approxHeapBytes}`);

/** The declared maximum of the module's linear memory, in bytes (defined or imported), or undefined. */
function wasmMemoryMaxBytes(bytes) {
  let at = 8;
  const leb = () => {
    let result = 0;
    let shift = 0;
    let byte;
    do {
      byte = bytes[at++];
      result += (byte & 0x7f) * 2 ** shift;
      shift += 7;
    } while (byte & 0x80);
    return result;
  };
  const limits = () => {
    const flags = bytes[at++];
    leb();
    return flags & 1 ? leb() * 65536 : undefined;
  };
  while (at < bytes.length) {
    const id = bytes[at++];
    const end = leb() + at;
    if (id === 5 && leb() > 0) return limits();
    if (id === 2) {
      for (let n = leb(); n > 0; n--) {
        at += leb();
        at += leb();
        const kind = bytes[at++];
        if (kind === 2) return limits();
        if (kind === 0) leb();
        else if (kind === 1) { at++; limits(); }
        else if (kind === 3) at += 2;
        else if (kind === 4) { at++; leb(); }
      }
    }
    at = end;
  }
  return undefined;
}

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
