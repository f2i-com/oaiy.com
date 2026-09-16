/**
 * The leaf-script envelope, against both installed Zipp engines.
 *
 *     npm run test:zipp-script                               both installs, one process each
 *     ZIPP_VENDOR=zipp-wasm-python npm run test:zipp-script  one of them
 *
 * `zipp-executor.mjs` proves the engine runs OAIY's compiled workflow. This
 * proves `zipp-script.ts` runs a LEAF — one body, one expression, one entry
 * function, one Python project — the way `oaiy script` and the Desktop's
 * script host will ask for it: the modes, the data boundary, every
 * `errorKind`, whole-request refusal, fresh-Engine isolation, and that the
 * JSON schemas under `protocol/v1` agree with the code on every fixture this
 * file sends. JavaScript jobs run identically on both installs; a Python job
 * runs on the python install and is `unsupported`, never attempted, on the
 * other.
 *
 * Requires Node 22.18+ / 24+ — it imports the `.ts` directly.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  buildScriptProgram,
  runScriptJob,
  runScriptRequest,
  mapAuthorLines,
  scriptEngineIdentity,
  validateScriptRequest,
  SCRIPT_ENVELOPE_GLOBALS,
  SCRIPT_ERROR_KINDS,
  SCRIPT_JS_MODES,
  SCRIPT_MAX_JOB_MODES,
  SCRIPT_MAX_LINE_OFFSET,
  SCRIPT_MAX_PROFILE_MODES,
  SCRIPT_REFUSED_GLOBALS,
} from '../vendor/oaiy-core/src/zipp-script.ts';
import { ZIPP_GUEST_SHIMS, ZIPP_MAX_INSTRUCTION_BUDGET_STEPS } from '../vendor/oaiy-core/src/zipp-executor.ts';
import { checkInstall, VARIANTS } from '../../scripts/fetch-zipp-release.mjs';
// The host-side parser mode `parse` needs; `oaiy script` supplies the same
// shape from the CLI's own acorn (cli/src/zipp/script-worker.ts).
import { parseExpressionAt as acornParseExpressionAt } from 'acorn';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROTOCOL = path.join(HERE, '..', '..', 'protocol', 'v1');

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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const show = (v) => JSON.stringify(v);

// ---------------------------------------------------------------------------
// The engine, and its identity from the install record — never a literal.
// ---------------------------------------------------------------------------
console.log(`\ninstalled engine (${VARIANT.folder})`);
const source = checkInstall(VENDOR, VARIANT);
const wasmBytes = fs.readFileSync(path.join(VENDOR, 'zipp_wasm_bg.wasm'));
const { default: init, Engine, zippProfile } = await import(pathToFileURL(path.join(VENDOR, 'zipp_wasm.js')).href);
await init({ module_or_path: wasmBytes });
const profile = JSON.parse(zippProfile());
const PROFILE_JSON = JSON.parse(fs.readFileSync(path.join(VENDOR, 'PROFILE.json'), 'utf8'));

const identity = scriptEngineIdentity(zippProfile(), { release: source.release, wasmSha256: source.sha256 });
check('identity carries the running engine\'s version, revision and languages',
  identity.name === 'zipp' && identity.version === profile.version && identity.revision === profile.source.sha && same(identity.languages, profile.languages));
check('identity agrees with SOURCE.json and PROFILE.json',
  identity.release === source.release && identity.version === source.version && identity.revision === source.revision
    && identity.wasmSha256 === createHash('sha256').update(wasmBytes).digest('hex') && same(identity.languages, PROFILE_JSON.languages));
const HAS_PYTHON = identity.languages.includes('python');
console.log(`  (python: ${HAS_PYTHON})`);

// Every Engine the runner builds is counted, so "never attempted" is a
// measured zero rather than a belief.
let constructions = 0;
class CountingEngine extends Engine {
  constructor() {
    super();
    constructions++;
  }
}
const sha256 = (text) => createHash('sha256').update(text).digest('hex');
const HOST = { engine: identity, sha256 };

// Every request this file sends, with the runner's verdict, for the schema
// section: valid ↔ accepted must hold both ways.
const fixtures = [];
const responses = [];
function send(request, { schemaBlind = false } = {}) {
  const before = constructions;
  const response = runScriptRequest(CountingEngine, request, HOST);
  fixtures.push({ request, accepted: !('error' in response), schemaBlind });
  responses.push(response);
  response.constructions = constructions - before;
  return response;
}
const job = (id, mode, source, extra = {}) => ({ id, mode, source, ...extra });
const one = (mode, source, extra = {}, request = {}) => {
  const response = send({ v: 1, jobs: [job('j', mode, source, extra)], ...request });
  if ('error' in response) return { refused: response.error };
  return response.results[0];
};
const value = (result) => ('value' in result ? result.value : undefined);

// ---------------------------------------------------------------------------
// 1. Modes.
// ---------------------------------------------------------------------------
console.log('\nprogram / body');
{
  const r = one('program', '1 + 1');
  check('program: the completion value', r.ok === true && r.value === 2, show(r));
  const ret = one('program', 'return 1');
  check('program: a top-level return is a guest SyntaxError', ret.ok === false && ret.errorKind === 'guest' && /return/i.test(ret.error), show(ret));
  const b = one('body', 'return 1 + 1');
  check('body: `return 1+1` gives 2', b.ok === true && b.value === 2, show(b));
  const none = one('body', '1 + 1');
  check('body without a return is ok with no value', none.ok === true && !('value' in none), show(none));
  const r2 = send({ v: 1, jobs: [job('a', 'program', '1'), job('b', 'body', 'return 2')] });
  check('results keep job order and ids', same(r2.results.map(x => [x.id, x.value]), [['a', 1], ['b', 2]]), show(r2.results));
  check('the response carries the engine identity', same(r2.engine, identity));
  const noJobs = send({ v: 1, jobs: [] });
  check('an empty batch is an empty result', same(noJobs.results, []) && noJobs.constructions === 0);
}

console.log('\nauto: the committed flow probe');
{
  // The flow probe's pinned expectations table, minus the requester's own prelude helpers.
  const CONTEXT = {
    inputs: { from: '+61491570156', durationSeconds: 12 },
    event: null,
    nodes: { customers: [{ id: 'r1', answers: { phone: '+61400000000', name: 'Other' } }, { id: 'r2', answers: { phone: '+61491570156', name: 'Ada' } }] },
    kv: { greeting: 'Hello' },
  };
  const auto = (source) => one('auto', source, { globals: CONTEXT });
  const guest = (source) => {
    const r = auto(source);
    return r.ok === false && r.errorKind === 'guest' ? r.error : `NOT A GUEST ERROR: ${show(r)}`;
  };
  check('auto: an expression', value(auto('inputs.durationSeconds > 5')) === true);
  check('auto: string building over the globals', value(auto('kv.greeting + ", " + nodes.customers[1].answers.name')) === 'Hello, Ada');
  check('auto: the completion value of multi-statement code',
    same(value(auto('const c = nodes.customers.find(r => r.answers.phone === inputs.from);\n({ found: !!c, name: c ? c.answers.name : null })')), { found: true, name: 'Ada' }));
  check('auto: a function body with a top-level return over the same globals',
    same(value(auto('if (!inputs.from) return { found: false };\nconst c = nodes.customers.find(r => r.answers.phone === inputs.from);\nreturn { found: !!c, name: c?.answers?.name, greeting: kv.greeting };')),
      { found: true, name: 'Ada', greeting: 'Hello' }));
  check("auto: 'return inputs.durationSeconds > 5;' → true", value(auto('return inputs.durationSeconds > 5;')) === true);
  const mixed = (limit) => `if (inputs.durationSeconds > ${limit}) return "big";\n"small"`;
  check('auto: mixed(5) → the return fires', value(auto(mixed(5))) === 'big');
  const m100 = auto(mixed(100));
  check('auto: mixed(100) → undefined (the trailing expression is dropped)', m100.ok === true && !('value' in m100), show(m100));
  check('auto: a return inside a nested function is not a function body (map)',
    same(value(auto('nodes.customers.map(function (r) { return r.answers.name; })')), ['Other', 'Ada']));
  check('auto: … (IIFE)', value(auto('(function () { return inputs.durationSeconds * 2; })()')) === 24);
  check('auto: … (declaration then call)', value(auto('function pick(r) { return r.id; }\npick(nodes.customers[0])')) === 'r1');
  check('auto: globalThis.runs is 1 — deciding the style executes nothing (script)', value(auto('globalThis.runs = (globalThis.runs || 0) + 1;\nglobalThis.runs')) === 1);
  check('auto: globalThis.runs is 1 (function body)', value(auto('globalThis.runs = (globalThis.runs || 0) + 1;\nreturn globalThis.runs;')) === 1);
  const plain = guest('1 +');
  const program = one('program', '1 +');
  check('auto: a genuine syntax error is a guest error with the program-mode message', program.ok === false && plain === program.error, `${plain} vs ${show(program)}`);
  for (const code of ['const a = ;\nreturn a;', 'return 1 +']) {
    const msg = guest(code);
    check(`auto: ${show(code)} is not blamed on the return`, !/NOT A GUEST/.test(msg) && !/'return' outside of a function/.test(msg), msg);
  }
  check('auto: "use strict" legacy octal in a function body', /legacy octal/.test(guest('"use strict";\nreturn 0123;')), guest('"use strict";\nreturn 0123;'));
  const runtime = guest("const raw = 'not json';\nJSON.parse(raw)");
  const runtimeProgram = one('program', "const raw = 'not json';\nJSON.parse(raw)", { globals: CONTEXT });
  check('auto: a runtime SyntaxError is the error it is, same as program mode', runtimeProgram.ok === false && runtime === runtimeProgram.error, `${runtime} vs ${show(runtimeProgram)}`);
  const asProgram = one('program', 'return inputs.durationSeconds > 5;', { globals: CONTEXT });
  check('program mode keeps refusing a top-level return', asProgram.ok === false && asProgram.errorKind === 'guest', show(asProgram));
}

console.log('\nentry / parse');
{
  const r = one('entry', 'function run(ctx) { return ctx.a; }', { entry: 'run', args: [{ a: 3 }] });
  check('entry: run({a:3}) → 3', r.ok === true && r.value === 3, show(r));
  const two = one('entry', 'function add(a, b) { return a + b; }', { entry: 'add', args: [2, 5] });
  check('entry: every arg is passed', two.ok === true && two.value === 7, show(two));
  const missing = one('entry', 'var x = 1;', { entry: 'run' });
  check('entry: a missing entry is ok with undefined', missing.ok === true && !('value' in missing), show(missing));
  const thrown = one('entry', 'function run() { throw new TypeError("bad ctx"); }', { entry: 'run' });
  check('entry: a throw is a guest error with its message', thrown.ok === false && thrown.errorKind === 'guest' && thrown.error === 'bad ctx', show(thrown));
  // Parsed standalone, the payload's `}` is unbalanced: a SyntaxError, not a
  // wrapper escape that runs `globalThis.x = 1`.
  const inert = one('parse', '1); })(); globalThis.x = 1; (function(){ return (1');
  check('parse: the wrapper-closing payload is a syntax error, not an escape', inert.ok === false && inert.errorKind === 'guest' && !/x is not defined/.test(inert.error), show(inert));
  const never = one('parse', '__emit({ok: true, value: "ran"})');
  check('parse: the source is never invoked', never.ok === true && never.value === null, show(never));
  const bad = one('parse', '1 +');
  check('parse: a syntax error is a guest error', bad.ok === false && bad.errorKind === 'guest', show(bad));
}

// ---------------------------------------------------------------------------
// 2. The data boundary.
// ---------------------------------------------------------------------------
console.log('\nglobals, args, prepare');
{
  const globals = JSON.parse('{"a":1,"__x":2,"a-b":3,"__proto__":{"polluted":true},"constructor":"c","prototype":"p"}');
  const r = one('program', '[typeof a, typeof __x, typeof constructor, typeof prototype, ({}).polluted === undefined, globalThis["a-b"] === undefined]', { globals });
  check('the globals filter drops __x, a-b, __proto__, constructor and prototype',
    same(r.value, ['number', 'undefined', 'function', 'undefined', true, true]), show(r));

  const ctx = { a: [1, 2.5, null, true, 'ü 日本   "quoted"'], b: { c: { d: 'e', f: false } }, s: 'x' };
  check('ctx round-trips as JSON through globals', same(value(one('program', 'ctx', { globals: { ctx } })), ctx));
  check('ctx round-trips as JSON through entry args', same(value(one('entry', 'function run(c) { return c; }', { entry: 'run', args: [ctx] })), ctx));
  const payload = '"; globalThis.pwned = 1; "';
  const program = buildScriptProgram(job('j', 'program', payload, { globals: { k: payload }, args: undefined }));
  check('source and globals appear in the program only as JSON literals',
    program.includes(JSON.stringify(payload)) && program.includes(JSON.stringify(JSON.stringify({ k: payload }))) && !program.includes(`(${payload})`));
  check('the guest shims sit at program top level, after the reply channel', program.indexOf('var __replies') < program.indexOf(ZIPP_GUEST_SHIMS) && program.indexOf(ZIPP_GUEST_SHIMS) < program.indexOf('var __out;'));

  // The other half of the shadowing fix, reachable only by going AROUND
  // validation: `buildScriptProgram` is the program builder, and a host that
  // built one without `validateScriptRequest` (or a profile preamble, or a
  // `prepare` hook, both of which run before the install loop) must not be able
  // to break the envelope either. So the bootstrap and the install loop hold
  // their own `JSON.parse`, `Array.isArray` and `hasOwnProperty`, snapshotted
  // in EMIT_PREAMBLE before anything else runs.
  {
    const shadowed = buildScriptProgram({ id: 'j', mode: 'program', source: 'b', globals: { Object: 1, b: 2 } });
    const e = new CountingEngine();
    let reply;
    try {
      e.initScript(shadowed);
      reply = e.evalInContext('__replies[__replies.length - 1]');
    } finally {
      try { e.dispose(); } catch { /* already gone */ }
    }
    check('the install loop survives a globals key that shadows Object, and still installs the rest',
      reply?.ok === true && reply.value === 2, show(reply));
    check('EMIT_PREAMBLE snapshots the intrinsics the envelope needs, before the shims',
      shadowed.indexOf('var __hasOwn = Object.prototype.hasOwnProperty;') > -1
        && shadowed.indexOf('var __jsonParse = JSON.parse;') > -1
        && shadowed.indexOf('var __isArray = Array.isArray;') > -1
        && shadowed.indexOf('var __hasOwn') < shadowed.indexOf(ZIPP_GUEST_SHIMS),
      shadowed.slice(0, 200));
    check('the install loop and the bootstrap use the snapshots, not the global names',
      shadowed.includes('__hasOwn.call(__ctx, __k)') && shadowed.includes('__ctx = __jsonParse(')
        && shadowed.includes('!__isArray(__args)') && !shadowed.includes('Object.prototype.hasOwnProperty.call(__ctx'),
      shadowed.slice(shadowed.indexOf('for (var __k'), shadowed.indexOf('for (var __k') + 120));
    check('SCRIPT_ENVELOPE_GLOBALS names every __ binding the envelope makes',
      ['__jsonParse', '__isArray', '__hasOwn'].every(n => SCRIPT_ENVELOPE_GLOBALS.includes(n)), show(SCRIPT_ENVELOPE_GLOBALS));
  }

  const preamble = 'function prep(c) { c.doubled = c.n * 2; return c; }\nfunction prepArgs(a) { return [a[0] + 1]; }\nfunction bad() { throw new Error("nope"); }\nvar notFn = 1;\n';
  const withProfile = { profile: { v: 1, preamble, preambleSha256: sha256(preamble) } };
  const prepared = one('program', 'doubled', { globals: { n: 21 }, prepare: 'prep' }, withProfile);
  check('prepare maps the globals before the source runs', prepared.ok === true && prepared.value === 42, show(prepared));
  const preparedArgs = one('entry', 'function run(x) { return x; }', { entry: 'run', args: [1], prepare: 'prepArgs' }, withProfile);
  check('prepare maps the args array in entry mode', preparedArgs.ok === true && preparedArgs.value === 2, show(preparedArgs));
  const noHook = one('program', '1', { prepare: 'absent' }, withProfile);
  check("errorKind 'prepare' when the hook is not a function", noHook.ok === false && noHook.errorKind === 'prepare' && /absent/.test(noHook.error), show(noHook));
  const notFn = one('program', '1', { prepare: 'notFn' }, withProfile);
  check("errorKind 'prepare' when the hook names a non-function", notFn.ok === false && notFn.errorKind === 'prepare', show(notFn));
  const threw = one('program', '1', { prepare: 'bad' }, withProfile);
  check("errorKind 'prepare' when the hook throws", threw.ok === false && threw.errorKind === 'prepare' && /nope/.test(threw.error), show(threw));
  const withoutProfile = one('program', '1', { prepare: 'prep' });
  check("prepare without a profile is 'prepare', not a crash", withoutProfile.ok === false && withoutProfile.errorKind === 'prepare', show(withoutProfile));
  const preambleSeen = one('program', 'typeof prep', {}, withProfile);
  check('the preamble is visible to the source', preambleSeen.value === 'function', show(preambleSeen));
  const preambleInFunction = one('body', 'return Function("return typeof prep")()', {}, withProfile);
  check('the preamble is visible inside a body compiled at run time', preambleInFunction.value === 'function', show(preambleInFunction));
  const profileBudget = one('program', 'for (var i = 0; i < 3e6; i++) {}; "done"', {}, { profile: { v: 1, preamble: '', preambleSha256: sha256(''), instructionSteps: 2_000_000 } });
  check('profile.instructionSteps bounds a job that sets none', profileBudget.ok === false && profileBudget.errorKind === 'resource', show(profileBudget));
  const jobWins = one('program', 'for (var i = 0; i < 3e6; i++) {}; "done"', { instructionSteps: 200_000_000 }, { profile: { v: 1, preamble: '', preambleSha256: sha256(''), instructionSteps: 2_000_000 } });
  check('a job\'s instructionSteps overrides the profile\'s', jobWins.ok === true && jobWins.value === 'done', show(jobWins));
}

console.log('\nshims and output');
{
  const timer = one('entry', 'function run() { try { setTimeout(function () {}, 0); return "ran"; } catch (e) { return e.message; } }', { entry: 'run' });
  check('a timer inside an entry-mode body meets the stub, not Zipp\'s intrinsic', /not available inside the Zipp sandbox/.test(timer.value), show(timer));
  const net = one('body', 'return Function("try { fetch(\\"x\\"); return \\"ran\\" } catch (e) { return e.message }")()');
  check('fetch inside a run-time Function meets the stub', /HTTP Request node/.test(net.value), show(net));
  const noisy = one('program', 'console.log("x"); print("y"); 7');
  check('console and print are no-ops that do not break the job', noisy.ok === true && noisy.value === 7, show(noisy));
  const deep = one('program', '(function () { var v = "leaf"; for (var i = 0; i < 12; i++) v = [v]; return v; })()');
  let depth = 0;
  let cursor = deep.value;
  while (Array.isArray(cursor) && cursor.length === 1) { cursor = cursor[0]; depth++; }
  check('output is cut at depth 8 (deeper levels become null)', depth === 8 && cursor === null, `depth ${depth}, leaf ${show(cursor)}`);
  const keys = one('program', 'JSON.parse(\'{"constructor":1,"prototype":2,"__proto__":3,"keep":4}\')');
  check('dangerous keys are dropped from output', same(keys.value, { keep: 4 }), show(keys));
  const nonJson = one('program', '({ f: function () {}, u: undefined, n: NaN, d: new Date(0), i: Infinity })');
  check('non-JSON values are null or dropped', same(nonJson.value, { n: null, d: '1970-01-01T00:00:00.000Z', i: null }), show(nonJson));
}

// ---------------------------------------------------------------------------
// 3. errorKind and isolation.
// ---------------------------------------------------------------------------
console.log('\nerrorKind');
{
  const thrown = one('program', 'throw new Error("boom")');
  check("a throw is 'guest'", thrown.ok === false && thrown.errorKind === 'guest' && thrown.error === 'boom', show(thrown));
  const loop = one('program', 'while (true) {}', { instructionSteps: 2_000_000 });
  check("while(true) under a budget is 'resource'", loop.ok === false && loop.errorKind === 'resource' && /budget/i.test(loop.error), show(loop));
  const uncatchable = one('program', 'try { while (true) {} } catch (e) { "caught" }', { instructionSteps: 2_000_000 });
  check('the budget cannot be caught in the guest', uncatchable.ok === false && uncatchable.errorKind === 'resource', show(uncatchable));
  const defaultLoop = one('program', 'while (true) {}');
  check("while(true) on the engine default is still 'resource'", defaultLoop.ok === false && defaultLoop.errorKind === 'resource', show(defaultLoop));
  const bounded = one('program', 'for (var i = 0; i < 3e6; i++) {}; "done"', { instructionSteps: 2_000_000 });
  check('a 3M-iteration loop stops under a 2M budget', bounded.ok === false && bounded.errorKind === 'resource', show(bounded));
  const unbounded = one('program', 'for (var i = 0; i < 3e6; i++) {}; "done"');
  check('the same loop completes on the engine default', unbounded.ok === true && unbounded.value === 'done', show(unbounded));
  const big = `var x = 0;\n// ${'a'.repeat(70 * 1024)}\nfunction run() { return 1; }`;
  check('the fixture is over the dynamic-code ceiling', big.length > profile.limits.dynamicCodeSourceBytes);
  const large = one('entry', big, { entry: 'run' });
  check("a 70 KiB entry source is 'resource'", large.ok === false && large.errorKind === 'resource', show(large));
  // Every JS mode compiles the source inside the guest (eval or Function), so
  // the same ceiling bounds a program too — a leaf source is at most 64 KiB.
  const largeProgram = one('program', big);
  check("the same source as a program is 'resource' too (indirect eval is dynamic compilation)", largeProgram.ok === false && largeProgram.errorKind === 'resource', show(largeProgram));
}

console.log('\nisolation');
{
  const r = send({ v: 1, jobs: [job('set', 'program', 'globalThis.leak = 1; typeof leak'), job('read', 'program', 'typeof leak')] });
  check('no state leaks between two consecutive jobs', r.results[0].value === 'number' && r.results[1].value === 'undefined', show(r.results));
  check('one fresh Engine per job', r.constructions === 2, String(r.constructions));
  const after = send({ v: 1, jobs: [job('exhaust', 'program', 'while (true) {}', { instructionSteps: 2_000_000 }), job('next', 'program', '"fine"')] });
  check('a job after a resource failure runs normally', after.results[0].errorKind === 'resource' && after.results[1].value === 'fine', show(after.results));
  const afterGuest = send({ v: 1, jobs: [job('g', 'program', 'globalThis.__replies = "forged"; throw 1'), job('n', 'program', '2')] });
  check('a job that tampered with its reply channel does not affect the next', afterGuest.results[1].value === 2, show(afterGuest.results));
  const brokenPreamble = { profile: { v: 1, preamble: 'var = ;', preambleSha256: sha256('var = ;') } };
  const src = one('program', '1', {}, brokenPreamble);
  check("a preamble that does not compile is 'source' (the engine's own kind), never a value", src.ok === false && src.errorKind === 'source', show(src));
}

console.log('\ntrap');
{
  // No real engine traps on demand; a stand-in that throws the one error
  // class a trap raises exercises the path the engine tests cannot.
  let built = 0;
  let disposed = 0;
  class TrappingEngine {
    constructor() { built++; }
    initScript() { throw new WebAssembly.RuntimeError('unreachable'); }
    evalInContext() { return null; }
    lastErrorKind() { return 'guest'; }
    dispose() { disposed++; }
  }
  const traps = [];
  const r = runScriptRequest(TrappingEngine, { v: 1, jobs: [job('t', 'program', '1'), job('after', 'program', '2')] }, { engine: identity, onEngineTrap: (e) => traps.push(e) });
  check("a WebAssembly trap is 'resource' and names itself", r.results[0].ok === false && r.results[0].errorKind === 'resource' && /trapped/.test(r.results[0].error), show(r.results[0]));
  check('the host is told once and the trapped engine is not disposed', traps.length === 1 && traps[0] instanceof WebAssembly.RuntimeError && disposed === 0, `${traps.length} ${disposed}`);
  check("the rest of the batch is not attempted on the poisoned instance ('host', no Engine built)",
    r.results[1].ok === false && r.results[1].errorKind === 'host' && /not attempted/i.test(r.results[1].error) && built === 1, show(r.results[1]));
  responses.push(r);
}

console.log('\nconsole, print and the output ceiling');
{
  // What the stubs DO, measured, because the comment used to claim more. The
  // envelope installs `globalThis.print` and `globalThis.console`: `print` is
  // replaced and writes nothing; `console` is not, whatever the assignment
  // looks like — reading `console.log` gives the stub back, and CALLING it still
  // reaches the engine's own output buffer.
  const e = new CountingEngine();
  let reply;
  let output;
  try {
    e.initScript(buildScriptProgram(job('j', 'program', 'console.log("LEAK-1"); print("LEAK-2"); String(console.log).replace(/\\s+/g, "")')));
    reply = e.evalInContext('__replies[__replies.length - 1]');
    output = e.takeOutput();
  } finally {
    try { e.dispose(); } catch { /* already gone */ }
  }
  const lines = Array.isArray(output) ? output : [];
  check('print IS stubbed: nothing it writes reaches the engine buffer',
    !lines.some(l => String(l).includes('LEAK-2')), show(output));
  check('console is NOT: the stub reads back, and the call still writes',
    reply?.value === 'function(){}' && lines.some(l => String(l).includes('LEAK-1')), `${show(reply)} ${show(output)}`);

  // …and the ceiling that follows from it: `lifetimeOutputBytes` (8 MiB on
  // v0.0.19) is a LIFETIME total a drain does not reset, so a chatty script
  // fails `resource` on its own account. Measured: nine 1 MiB rounds with a
  // drain after each still ended in the same RangeError. The engine per job is
  // fresh, so the next job is unaffected — the assertion below is that pair.
  check('the engine has an output ceiling and this fixture is over it',
    profile.limits.lifetimeOutputBytes > 0 && 9000 * 1000 > profile.limits.lifetimeOutputBytes,
    String(profile.limits.lifetimeOutputBytes));
  const chatty = 'for (var i = 0; i < 9000; i++) console.log("' + 'y'.repeat(1000) + '"); 42';
  const batch = send({ v: 1, jobs: [job('chatty', 'program', chatty, { instructionSteps: ZIPP_MAX_INSTRUCTION_BUDGET_STEPS }), job('next', 'program', '"fine"')] });
  check("a script past the output ceiling is 'resource', and says which budget",
    batch.results[0].ok === false && batch.results[0].errorKind === 'resource' && /output budget/.test(batch.results[0].error), show(batch.results[0]));
  check('the next job on its own fresh Engine is unaffected', batch.results[1].value === 'fine', show(batch.results[1]));

  // The drain itself: nothing a job wrote outlives the job. A stand-in Engine
  // records the call, because the real one empties a buffer nobody can see.
  let drains = 0;
  let evals = 0;
  class DrainSpyEngine {
    initScript() {}
    evalInContext() { evals++; return { ok: true, value: 1 }; }
    lastErrorKind() { return 'guest'; }
    takeOutput() { drains++; return []; }
    dispose() {}
  }
  const spied = runScriptRequest(DrainSpyEngine, { v: 1, jobs: [job('a', 'program', '1'), job('b', 'program', '2')] }, { engine: identity });
  check('every job drains the engine console once, after its source has run',
    drains === 2 && evals === 2 && spied.results.every(r => r.ok === true), `drains=${drains} evals=${evals} ${show(spied.results)}`);
  responses.push(spied);
}

console.log('\nrenewInstructionBudget');
{
  // The engine ANSWERS the renewal request — `false` for a budget already
  // spent, or a disposed engine — and the answer was dropped. The job would
  // then have run its entry on whatever was left and failed `resource`
  // somewhere in the user's code. A refused renewal is the HOST's problem.
  class RefusingEngine {
    initScript() {}
    evalInContext() { return { ok: true, value: 'should not be reached' }; }
    lastErrorKind() { return 'guest'; }
    setInstructionBudget() { return true; }
    renewInstructionBudget() { return false; }
    dispose() {}
  }
  const r = runScriptRequest(RefusingEngine, { v: 1, jobs: [job('r', 'program', '1')] }, { engine: identity });
  check("a refused renewal is 'host', not a value and not the guest's fault",
    r.results[0].ok === false && r.results[0].errorKind === 'host' && /renew the instruction budget/.test(r.results[0].error), show(r.results[0]));
  class RenewingEngine extends RefusingEngine {
    renewInstructionBudget() { return true; }
  }
  const ok = runScriptRequest(RenewingEngine, { v: 1, jobs: [job('r', 'program', '1')] }, { engine: identity });
  check('a granted renewal runs the entry as before', ok.results[0].ok === true, show(ok.results[0]));
  responses.push(r, ok);
  // And the real engine grants it, on both installs, which is why nothing else
  // in this file has to care.
  check('the installed engine grants the renewal the envelope asks for', one('program', '1 + 1').value === 2);
}

console.log('\nmode parse: one expression, checked by a parser when the host has one');
{
  // The engine-side wrapper compiles `return (` + source + `)`, so a source
  // that closes the parenthesis and adds a statement compiles: measured,
  // `1); globalThis.pwned = (1` came back `{ok: true, value: null}` for source
  // that is not an expression at all. Inert — the function is never invoked —
  // but a requester reading `ok` as "this expression is well-formed" was
  // reading something nobody had checked. `oaiy script` supplies acorn; here
  // the same shape of check is supplied directly.
  const INJECTION = '1); globalThis.pwned = (1';
  const bare = one('parse', INJECTION);
  check('without a parser the engine wrapper accepts an injected statement (the finding, recorded)',
    bare.ok === true && bare.value === null, show(bare));
  check('and it really is inert: nothing ran', one('program', 'typeof globalThis.pwned').value === 'undefined');

  const parseExpression = (source) => {
    const node = acornParseExpressionAt(source, 0, { ecmaVersion: 'latest', preserveParens: true });
    const rest = source.slice(node.end);
    if (rest.trim() !== '') throw new SyntaxError(`expected one expression, and ${JSON.stringify(rest.trim().slice(0, 40))} follows it`);
  };
  const withParser = (source) => runScriptRequest(CountingEngine, { v: 1, jobs: [job('p', 'parse', source)] }, { ...HOST, parseExpression }).results[0];
  const refusedByParser = withParser(INJECTION);
  check("with a parser the same source is 'guest' and says what followed",
    refusedByParser.ok === false && refusedByParser.errorKind === 'guest' && /follows it/.test(refusedByParser.error), show(refusedByParser));
  for (const bad of ['1; 2', '1), (2', 'a b', '}', '1) + (']) {
    const r = withParser(bad);
    check(`with a parser ${show(bad)} is not one expression`, r.ok === false && r.errorKind === 'guest', show(r));
  }
  for (const good of ['1 + 1', 'inputs.n > 5', '{ a: 1 }', '(function () { return 1; })', 'x\n', '  a ? b : c  ']) {
    const r = withParser(good);
    check(`with a parser ${show(good)} is one expression`, r.ok === true && r.value === null, show(r));
  }
  check('the parser never runs the expression', withParser('globalThis.parsedRan = 1').ok === true
    && one('program', 'typeof globalThis.parsedRan').value === 'undefined');
}

console.log('\nwhat a result is evidence of');
{
  // Documented in the module comment, and measured here so the document is
  // checkable. `__replies` is an ordinary array at the guest's global scope:
  // a script can push a success and then throw, and the host reports success.
  // That is a script lying about ITSELF, inside the sandbox it already owns,
  // so it is not a boundary being crossed — but a requester treating `ok:
  // false` as a trustworthy signal about anything else must know.
  // `__emit` too: the envelope's own emit runs after the throw is caught and
  // would push the real failure on top. A guest that means to lie silences it.
  const forged = one('program', '__emit = function () {}; __replies.push({ok: true, value: "forged"}); throw new Error("real failure")');
  check('a guest can forge its own reply, and the host reports it (the contract, recorded)',
    forged.ok === true && forged.value === 'forged', show(forged));
  // What it cannot forge: the parts the HOST writes.
  const r = send({ v: 1, jobs: [job('f', 'program', '__emit = function () {}; __replies.push({ok: false, kind: "host", error: "not really"}); 1')] });
  check('errorKind is the host\'s, never the guest\'s: a forged kind is read as guest',
    r.results[0].ok === false && r.results[0].errorKind === 'guest', show(r.results[0]));
  check('the engine identity is the host\'s and comes from the install record',
    r.engine.wasmSha256 === source.sha256 && r.engine.release === source.release, show(r.engine));
}

// ---------------------------------------------------------------------------
// 4. Whole-request refusal.
// ---------------------------------------------------------------------------
console.log('\nrefusal');
{
  const good = job('good', 'program', '1');
  const refuse = (name, request, expect, opts) => {
    const before = constructions;
    const r = send(request, opts);
    const ok = 'error' in r && r.error.code === 'invalid_request' && expect.test(r.error.message) && !('results' in r) && constructions === before;
    check(`refused: ${name}`, ok, show(r));
  };
  refuse('v: 2', { v: 2, jobs: [good] }, /v must be 1/);
  refuse('unknown mode', { v: 1, jobs: [good, job('x', 'nope', '1')] }, /jobs\[1\] \(id "x"\): unknown mode "nope"/);
  refuse('unknown language', { v: 1, jobs: [job('x', 'program', '1', { language: 'ruby' })] }, /jobs\[0\] \(id "x"\): unknown language "ruby"/);
  refuse('python language with a JS mode', { v: 1, jobs: [job('x', 'program', '1', { language: 'python' })] }, /"program" runs JavaScript/);
  refuse('python-project without a language', { v: 1, jobs: [{ id: 'x', mode: 'python-project', files: { 'main.py': '' }, entry: 'main', call: 'run' }] }, /needs language "python"/);
  refuse('python-project as javascript', { v: 1, jobs: [{ id: 'x', language: 'javascript', mode: 'python-project', files: { 'main.py': '' }, entry: 'main', call: 'run' }] }, /needs language "python"/);
  refuse('a non-identifier entry', { v: 1, jobs: [job('x', 'entry', '1', { entry: 'a-b' })] }, /entry must be an identifier/);
  refuse('a reserved word as entry', { v: 1, jobs: [job('x', 'entry', '1', { entry: 'return' })] }, /entry must be an identifier/);
  refuse('a non-identifier prepare', { v: 1, jobs: [job('x', 'program', '1', { prepare: '1x' })] }, /prepare must be an identifier/);
  refuse('globals that are an array', { v: 1, jobs: [job('x', 'program', '1', { globals: [1] })] }, /globals must be a plain object/);
  // A `globals` key that shadows a name the ENVELOPE uses broke the envelope,
  // not the script, and the job came back `guest` — which the contract says
  // means the script's own fault. Measured before this: `{JSON:1}` gave
  // "undefined is not a function"; `{Object:1,b:2}` killed the install loop
  // mid-loop so `b` never arrived, while `{b:2,Object:1}` ran, because
  // `Object` happened to be installed last. The request is what is malformed,
  // so the request is what is refused.
  for (const name of SCRIPT_REFUSED_GLOBALS) {
    refuse(`globals shadowing ${name}`, { v: 1, jobs: [job('x', 'program', '1', { globals: { [name]: 1 } })] },
      new RegExp(`globals may not shadow "${name}", which the guest program itself uses`));
  }
  refuse('globals shadowing a name beside an ordinary one', { v: 1, jobs: [job('x', 'program', 'b', { globals: { Object: 1, b: 2 } })] }, /globals may not shadow "Object"/);
  refuse('a missing id', { v: 1, jobs: [{ mode: 'program', source: '1' }] }, /jobs\[0\]: id must be a string/);
  refuse('an empty id', { v: 1, jobs: [job('', 'program', '1')] }, /id must not be empty/);
  refuse('a missing source', { v: 1, jobs: [{ id: 'x', mode: 'program' }] }, /source must be a string/);
  refuse('instructionSteps 0', { v: 1, jobs: [job('x', 'program', '1', { instructionSteps: 0 })] }, /instructionSteps must be an integer from 1 to/);
  refuse('instructionSteps above the engine maximum', { v: 1, jobs: [job('x', 'program', '1', { instructionSteps: profile.limits.maxInstructionBudgetSteps + 1 })] }, /instructionSteps must be an integer/);
  refuse('budgetMs 0', { v: 1, jobs: [job('x', 'program', '1', { budgetMs: 0 })] }, /budgetMs must be an integer from 1 to 60000/);
  refuse('an unknown job field', { v: 1, jobs: [job('x', 'program', '1', { foo: 1 })] }, /unknown field "foo"/);
  refuse('an unknown request field', { v: 1, jobs: [good], results: [] }, /request: unknown field "results"/);
  refuse('args on a non-entry mode', { v: 1, jobs: [job('x', 'program', '1', { args: [1] })] }, /args is for mode "entry" only/);
  refuse('entry on a non-entry mode', { v: 1, jobs: [job('x', 'body', '1', { entry: 'run' })] }, /entry is for mode "entry" only/);
  refuse('entry mode without entry', { v: 1, jobs: [job('x', 'entry', '1')] }, /entry must be a string/);
  refuse('prepare on parse', { v: 1, jobs: [job('x', 'parse', '1', { prepare: 'p' })] }, /prepare has no meaning/);
  refuse('jobs not an array', { v: 1, jobs: {} }, /jobs must be an array/);
  refuse('a job that is not an object', { v: 1, jobs: [good, 'x'] }, /jobs\[1\] must be an object/);
  refuse('python files empty', { v: 1, jobs: [{ id: 'p', language: 'python', mode: 'python-project', files: {}, entry: 'main', call: 'run' }] }, /files must be a non-empty object/);
  refuse('python file that is not a string', { v: 1, jobs: [{ id: 'p', language: 'python', mode: 'python-project', files: { 'main.py': 1 }, entry: 'main', call: 'run' }] }, /files\["main.py"\] must be a string/);
  refuse('python without call', { v: 1, jobs: [{ id: 'p', language: 'python', mode: 'python-project', files: { 'main.py': '' }, entry: 'main' }] }, /call must be a string/);
  refuse('python fallback with extra fields', { v: 1, jobs: [{ id: 'p', language: 'python', mode: 'python-project', files: { 'main.py': '' }, entry: 'main', call: 'run', fallbackOnSourceError: { files: { 'main.py': '' }, entry: 'x' } }] }, /fallbackOnSourceError: unknown field "entry"/);
  refuse('profile with a bad sha', { v: 1, profile: { v: 1, preamble: 'var a = 1;', preambleSha256: sha256('other') }, jobs: [good] }, /preambleSha256 does not match/, { schemaBlind: true });
  refuse('profile sha not hex', { v: 1, profile: { v: 1, preamble: '', preambleSha256: 'X'.repeat(64) }, jobs: [good] }, /64 lower-case hex/);
  refuse('profile preamble declaring `host`', { v: 1, profile: { v: 1, preamble: 'var host = 1;', preambleSha256: sha256('var host = 1;') }, jobs: [good] }, /declares "host"/, { schemaBlind: true });
  refuse('profile preamble redeclaring `__emit`', { v: 1, profile: { v: 1, preamble: 'function __emit() {}', preambleSha256: sha256('function __emit() {}') }, jobs: [good] }, /declares "__emit"/, { schemaBlind: true });
  refuse('profile hook that is not an identifier', { v: 1, profile: { v: 1, preamble: '', preambleSha256: sha256(''), hooks: { lane: { prepare: 'a b' } } }, jobs: [good] }, /hooks\["lane"\]: prepare must be an identifier/);
  refuse('profile.instructionSteps out of range', { v: 1, profile: { v: 1, preamble: '', preambleSha256: sha256(''), instructionSteps: 0 }, jobs: [good] }, /profile: instructionSteps must be an integer/);
  refuse('profile with an unknown field', { v: 1, profile: { v: 1, preamble: '', preambleSha256: sha256(''), extra: 1 }, jobs: [good] }, /profile: unknown field "extra"/);

  // Modes. A mode is SPENT by the runner without a second look — its files are
  // merged, its block written, its offset subtracted — so every part of one is
  // checked here, where a fault is still a refusal and not a wrong line number
  // in somebody's error message. `schemaBlind` marks the faults a JSON Schema
  // cannot state (a name used twice, a name that must exist elsewhere in the
  // document, a key that must NOT exist elsewhere).
  const MODE = {
    name: 'wrap',
    files: { 'main.py': 'import blk\n\n\ndef run(ctx):\n    return blk.v\n' },
    block: 'blk.py',
    before: 'v = (\n',
    after: '\n)\n',
    lineOffset: 1,
  };
  const withModes = (modes, python = {}) => ({
    v: 1,
    preamble: '',
    preambleSha256: sha256(''),
    python: { contract: 'c/1', files: { 'lib.py': 'x = 1\n' }, entry: 'main', call: 'run', modes, ...python },
  });
  const mode = (patch) => {
    const m = { ...MODE, ...patch };
    for (const [k, val] of Object.entries(patch)) if (val === undefined) delete m[k];
    return m;
  };
  const withMode = (patch, python) => withModes([mode(patch)], python);
  const modeJob = (patch = {}) => {
    const j = { id: 'm', language: 'python', mode: 'python-project', modes: ['wrap'], source: '1', ...patch };
    for (const [k, val] of Object.entries(patch)) if (val === undefined) delete j[k];
    return j;
  };
  const badProfile = (name, profile, expect, opts) => refuse(name, { v: 1, profile, jobs: [good] }, expect, opts);
  const badModeJob = (name, patch, expect, opts) => refuse(name, { v: 1, profile: withModes([MODE]), jobs: [modeJob(patch)] }, expect, opts);

  badProfile('profile modes that are not an array', withModes({}), /python\.modes must be a non-empty array of modes/);
  badProfile('profile modes that are empty', withModes([]), /python\.modes must be a non-empty array of modes/);
  badProfile('more modes than a profile may carry', withModes(Array.from({ length: 33 }, (_, i) => mode({ name: `m${i}` }))), /python\.modes has 33 modes; at most 32/);
  badProfile('a mode that is not an object', withModes(['wrap']), /python\.modes\[0\] must be an object/);
  badProfile('a mode with an unknown field', withMode({ wrapper: 1 }), /modes\[0\] \(name "wrap"\): unknown field "wrapper"/);
  badProfile('a mode with no before', withMode({ before: undefined }), /modes\[0\] \(name "wrap"\): before must be a string/);
  badProfile('a mode with no after', withMode({ after: undefined }), /modes\[0\] \(name "wrap"\): after must be a string/);
  badProfile('a mode with no block', withMode({ block: undefined }), /modes\[0\] \(name "wrap"\): block must be a string/);
  badProfile('a mode with no files', withMode({ files: undefined }), /modes\[0\] \(name "wrap"\): files must be a non-empty object/);
  badProfile('a mode with an empty block name', withMode({ block: '' }), /modes\[0\] \(name "wrap"\): block must not be empty/);
  badProfile('a mode with an empty name', withMode({ name: '' }), /modes\[0\]: name must not be empty/);
  badProfile('a mode whose name is not a name', withMode({ name: 'a b' }), /modes\[0\] \(name "a b"\): name must be a letter followed by/);
  badProfile('a mode with no lineOffset', withMode({ lineOffset: undefined }), /modes\[0\] \(name "wrap"\): lineOffset must be an integer from 0 to 1000000, not undefined/);
  badProfile('a mode whose lineOffset is a string', withMode({ lineOffset: '1' }), /lineOffset must be an integer from 0 to 1000000, not "1"/);
  badProfile('a mode whose lineOffset is fractional', withMode({ lineOffset: 1.5 }), /lineOffset must be an integer from 0 to 1000000, not 1.5/);
  badProfile('a mode whose lineOffset is negative', withMode({ lineOffset: -1 }), /lineOffset must be an integer from 0 to 1000000, not -1/);
  badProfile('a mode with an empty call', withMode({ call: '' }), /modes\[0\] \(name "wrap"\): call must not be empty/);
  badProfile('two modes of one name', withModes([MODE, mode({ block: 'other.py' })]), /python\.modes: two modes are named "wrap"/, { schemaBlind: true });
  badProfile('a mode whose block would replace a contract file', withMode({ block: 'lib.py' }), /block "lib.py" is already a file of this project/, { schemaBlind: true });
  badProfile('a mode whose block would replace its own file', withMode({ block: 'main.py' }), /block "main.py" is already a file of this project/, { schemaBlind: true });
  badProfile('a mode file that shadows a contract file', withMode({ files: { 'lib.py': '', 'main.py': '' } }), /files\["lib.py"\] would replace a file of profile\.python\.files/, { schemaBlind: true });
  badProfile('a mode whose project has no entry module', withMode({ files: { 'other.py': '' } }), /no file is the entry module "main" \(expected "main" or "main\.py"\)/, { schemaBlind: true });
  badProfile('a contract entry that is not a module name, with modes', withModes([MODE], { entry: 'main.py' }), /python: entry must be a Python module name to unfold a mode, not "main.py"/, { schemaBlind: true });

  badModeJob('a mode job that also carries files', { files: { 'a.py': '' } }, /modes and files are two ways to build one project; pass one or the other/);
  badModeJob('a mode job that also carries entry', { entry: 'main' }, /modes and entry are two ways to build one project/);
  badModeJob('a mode job that also carries call', { call: 'run' }, /modes and call are two ways to build one project/);
  badModeJob('a mode job that also carries a fallback file set', { fallbackOnSourceError: { files: { 'a.py': '' } } }, /modes and fallbackOnSourceError are two ways to build one project/);
  badModeJob('a mode job with no source', { source: undefined }, /jobs\[0\] \(id "m"\): source must be a string/);
  badModeJob('a mode job with an empty mode list', { modes: [] }, /modes must be a non-empty array of mode names/);
  badModeJob('a mode job listing more modes than it may', { modes: Array.from({ length: 9 }, () => 'wrap') }, /modes lists 9 modes; at most 8/);
  badModeJob('a mode job naming something that is not a mode name', { modes: ['a b'] }, /modes\[0\] must be a mode name, not "a b"/);
  badModeJob('a mode job naming a mode the profile does not define', { modes: ['wrap', 'nope'] }, /jobs\[0\] \(id "m"\): names the mode "nope", which the profile's python\.modes does not define/, { schemaBlind: true });
  badModeJob('a mode job with an unknown field', { emit: 1 }, /jobs\[0\] \(id "m"\): unknown field "emit"/);
  refuse('a mode job with no profile at all', { v: 1, jobs: [modeJob()] },
    /jobs\[0\] \(id "m"\): names the mode "wrap", and this request carries no profile python contract to define it/, { schemaBlind: true });
  refuse('a mode job whose profile carries no python contract', { v: 1, profile: { v: 1, preamble: '', preambleSha256: sha256('') }, jobs: [modeJob()] },
    /names the mode "wrap", and this request carries no profile python contract to define it/, { schemaBlind: true });
  refuse('a mode job whose profile defines no modes', { v: 1, profile: { v: 1, preamble: '', preambleSha256: sha256(''), python: { contract: 'c/1', files: { 'main.py': '' }, entry: 'main', call: 'run' } }, jobs: [modeJob()] },
    /names the mode "wrap", which the profile's python\.modes does not define/, { schemaBlind: true });
  refuse('source on a job that carries its own files', { v: 1, jobs: [{ id: 'p', language: 'python', mode: 'python-project', files: { 'main.py': '' }, entry: 'main', call: 'run', source: '1' }] },
    /source is for a job that names modes/);
  // The refusal names the JOB before it names anything of the profile's: a
  // malformed job is a malformed job whatever profile arrives with it.
  refuse('a malformed job beside a malformed profile is still the job',
    { v: 1, profile: withMode({ lineOffset: -1 }), jobs: [job('bad', 'program', 1)] }, /jobs\[0\] \(id "bad"\): source must be a string/);
  {
    const before = constructions;
    const r = runScriptRequest(CountingEngine, { v: 1, profile: { v: 1, preamble: '', preambleSha256: sha256('') }, jobs: [good] }, { engine: identity });
    check('refused: a profile on a host with no sha256', 'error' in r && /cannot verify a preamble/.test(r.error.message) && constructions === before, show(r));
  }
  const second = { v: 1, jobs: [good, job('bad', 'program', 1)] };
  const before = constructions;
  const r = send(second);
  check('one bad job refuses the whole request and runs nothing', 'error' in r && /jobs\[1\] \(id "bad"\)/.test(r.error.message) && constructions === before, show(r));
  check('validateScriptRequest is the same verdict', validateScriptRequest(second).ok === false && validateScriptRequest({ v: 1, jobs: [good] }).ok === true);
  const nonObject = send('nope');
  check('a non-object request is refused', 'error' in nonObject && /request must be an object/.test(nonObject.error.message));
}

// ---------------------------------------------------------------------------
// 5. Python.
// ---------------------------------------------------------------------------
console.log(`\npython (${HAS_PYTHON ? 'this install runs it' : 'this install does not'})`);
const PY_FILES = {
  'main.py': 'import lib\n\ndef run(ctx):\n    return {"v": lib.twice(ctx["n"]), "ctx": ctx}\n',
  'lib.py': 'def twice(n):\n    return n * 2\n',
};
const PY_CTX = { n: 21, s: 'ü 日本', l: [1, null, true, 2.5], nested: { a: { b: [] } }, f: false };
const py = (id, files, extra = {}) => ({ id, language: 'python', mode: 'python-project', files, entry: 'main', call: 'run', args: [PY_CTX], ...extra });
const pyOne = (files, extra = {}) => {
  const r = send({ v: 1, jobs: [py('p', files, extra)] });
  return 'error' in r ? { refused: r.error } : { ...r.results[0], constructions: r.constructions };
};
if (HAS_PYTHON) {
  const r = pyOne(PY_FILES);
  check('a two-file project returns a value', r.ok === true && r.value.v === 42, show(r));
  check('ctx round-trips as JSON into Python and back', same(r.value.ctx, PY_CTX), show(r.value));
  const stems = pyOne({ main: PY_FILES['main.py'], lib: PY_FILES['lib.py'] });
  check('module stems work as file names too', stems.ok === true && stems.value.v === 42, show(stems));
  const raised = pyOne({ 'main.py': 'def run(ctx):\n    raise ValueError("bad " + str(ctx["n"]))\n' });
  check("a raise is 'guest' with the Python message", raised.ok === false && raised.errorKind === 'guest' && /ValueError: bad 21/.test(raised.error), show(raised));
  const loop = pyOne({ 'main.py': 'def run(ctx):\n    while True:\n        pass\n' }, { instructionSteps: 2_000_000 });
  check("an infinite loop under a budget is 'resource'", loop.ok === false && loop.errorKind === 'resource', show(loop));
  const noCall = pyOne({ 'main.py': 'x = 1\n' });
  check("a missing call is 'guest'", noCall.ok === false && noCall.errorKind === 'guest' && /run/.test(noCall.error), show(noCall));
  const BROKEN = { 'main.py': 'def run(ctx:\n    return 1\n' };
  const source = pyOne(BROKEN);
  check("a syntax error with no fallback is 'source'", source.ok === false && source.errorKind === 'source' && /SyntaxError/.test(source.error), show(source));
  const FALLBACK = { files: { 'main.py': 'def run(ctx):\n    return "fallback"\n' } };
  const fell = pyOne(BROKEN, { fallbackOnSourceError: FALLBACK });
  check('a source error runs the fallback file set', fell.ok === true && fell.value === 'fallback', show(fell));
  check('the fallback is a second fresh Engine', fell.constructions === 2, String(fell.constructions));
  const topLevel = pyOne({ 'main.py': 'raise RuntimeError("top")\n' }, { fallbackOnSourceError: FALLBACK });
  check('a top-level raise during init also takes the fallback', topLevel.ok === true && topLevel.value === 'fallback', show(topLevel));
  const bothBroken = pyOne(BROKEN, { fallbackOnSourceError: { files: BROKEN } });
  check("a broken fallback is 'source', tried once", bothBroken.ok === false && bothBroken.errorKind === 'source' && bothBroken.constructions === 2, show(bothBroken));
  const guestNoRetry = pyOne({ 'main.py': 'def run(ctx):\n    raise ValueError("x")\n' }, { fallbackOnSourceError: FALLBACK });
  check('a guest error in the call phase does not retry', guestNoRetry.ok === false && guestNoRetry.errorKind === 'guest' && guestNoRetry.constructions === 1, show(guestNoRetry));
  const leak = send({ v: 1, jobs: [
    py('set', { 'main.py': 'import builtins\ndef run(ctx):\n    builtins.leaked = 1\n    return hasattr(builtins, "leaked")\n' }),
    py('read', { 'main.py': 'import builtins\ndef run(ctx):\n    return hasattr(builtins, "leaked")\n' }),
  ] });
  check('no state leaks between two consecutive Python jobs', leak.results[0].value === true && leak.results[1].value === false, show(leak.results));
  const mixed = send({ v: 1, jobs: [py('p', PY_FILES), job('j', 'program', 'typeof lib')] });
  check('a Python job and a JS job share nothing', mixed.results[0].value.v === 42 && mixed.results[1].value === 'undefined', show(mixed.results));
  const before = constructions;
  const denied = runScriptJob(CountingEngine, py('p', PY_FILES), { languages: ['javascript'] });
  check("a host whose engine profile lacks python answers 'unsupported' without building an Engine",
    denied.ok === false && denied.errorKind === 'unsupported' && constructions === before, show(denied));
} else {
  const r = pyOne(PY_FILES);
  check("a Python job on this install is 'unsupported'", r.ok === false && r.errorKind === 'unsupported' && /Python/.test(r.error), show(r));
  check('… and was never attempted (no Engine built)', r.constructions === 0, String(r.constructions));
  const mixed = send({ v: 1, jobs: [py('p', PY_FILES), job('j', 'program', '1 + 1')] });
  check('the JS job beside it still runs, on exactly one Engine',
    mixed.results[0].errorKind === 'unsupported' && mixed.results[1].value === 2 && mixed.constructions === 1, show(mixed.results));
  check('a broken fallback set is not consulted either', pyOne({ 'main.py': 'def run(ctx:\n' }, { fallbackOnSourceError: { files: PY_FILES } }).constructions === 0);
}

// ---------------------------------------------------------------------------
// 6. Python modes: the runner unfolds a wrapping the PROFILE carried, and
//    answers in the author's line numbers.
//
//    Everything here is a contract this file invents, except the last block,
//    which is a real one vendored from a consumer. Nothing in `zipp-script.ts`
//    knows either.
// ---------------------------------------------------------------------------
console.log('\npython modes');
const MODE_CONTRACT = {
  contract: 'oaiy-test/1',
  files: { 'shared.py': 'ctx = None\n\n\ndef bind(c):\n    global ctx\n    ctx = c\n\n\ndef twice(n):\n    return n * 2\n' },
  entry: 'main',
  call: 'run',
};
// `expr` wraps the source as one parenthesised expression (3 generated lines,
// the last of which OPENS the statement the source finishes); `stmt` runs it as
// a module (1 generated line). The pair is the two-phase shape: statements
// cannot be parenthesised, so `expr` fails to COMPILE and `stmt` answers.
const EXPR_MODE = {
  name: 'expr',
  files: { 'main.py': 'import shared\n\n\ndef run(c):\n    shared.bind(c)\n    import blk\n    return blk.value()\n' },
  block: 'blk.py',
  before: 'from shared import *\ndef value():\n    return (\n',
  after: '\n    )\n',
  lineOffset: 3,
};
const STMT_MODE = {
  name: 'stmt',
  files: { 'main.py': 'import shared\n\n\ndef run(c):\n    shared.bind(c)\n    import blk\n    return getattr(blk, "result", None)\n' },
  block: 'blk.py',
  before: 'from shared import *\n',
  after: '',
  lineOffset: 1,
};
// A wrapper whose FIRST line can fail on its own account: nothing the author
// wrote can be blamed for it.
const PROLOGUE_MODE = {
  name: 'prologue',
  files: { 'main.py': 'def run(ctx):\n    import blk\n    return blk.result\n' },
  block: 'blk.py',
  before: 'import nosuchmodule\nresult = None\n',
  after: '',
  lineOffset: 2,
};
const QUIET_MODE = {
  name: 'quiet',
  files: { 'main.py': 'def never():\n    import blk\n' },
  block: 'blk.py',
  before: '',
  after: '',
  lineOffset: 0,
  call: 'never',
};
const MODE_PROFILE = {
  v: 1,
  preamble: '',
  preambleSha256: sha256(''),
  python: { ...MODE_CONTRACT, modes: [EXPR_MODE, STMT_MODE, PROLOGUE_MODE, QUIET_MODE] },
};
const modeOne = (modes, source, extra = {}, profile = MODE_PROFILE) => {
  const r = send({ v: 1, profile, jobs: [{ id: 'm', language: 'python', mode: 'python-project', modes, source, args: [{ n: 21 }], ...extra }] });
  return 'error' in r ? { refused: r.error } : { ...r.results[0], constructions: r.constructions };
};
{
  // The pure function, so the arithmetic can be read (and mutated) on its own.
  const m = { ...EXPR_MODE };
  check('mapAuthorLines: a line below the wrapper is the author\'s',
    mapAuthorLines('KeyError (blk.py:5)', m, 'a\nb\nc\nd') === 'KeyError (blk.py:2)');
  check('mapAuthorLines: the splice line is the author\'s first line',
    mapAuthorLines('KeyError (blk.py:3)', m, 'a\nb') === 'KeyError (blk.py:1)');
  check('mapAuthorLines: a line past the author\'s last is capped at it',
    mapAuthorLines('SyntaxError (blk.py:5:4)', m, 'a') === 'SyntaxError (blk.py:1:4)');
  check('mapAuthorLines: a line above the splice is left byte-for-byte',
    mapAuthorLines('ImportError (blk.py:1)\n  File "blk.py", line 2, in <module>', m, 'a')
      === 'ImportError (blk.py:1)\n  File "blk.py", line 2, in <module>');
  check('mapAuthorLines: a traceback frame is renumbered, and only in the block file',
    mapAuthorLines('  File "main.py", line 5, in run\n  File "blk.py", line 5, in value', m, 'a\nb')
      === '  File "main.py", line 5, in run\n  File "blk.py", line 2, in value');
  check('mapAuthorLines: a character offset is rebased by the wrapper\'s length',
    mapAuthorLines(`blk.py: Python: nope (at offset ${m.before.length + 1})`, m, 'a\nb') === 'blk.py: Python: nope (at offset 1)');
  check('mapAuthorLines: an offset inside the wrapper is left alone',
    mapAuthorLines('blk.py: Python: nope (at offset 3)', m, 'a\nb') === 'blk.py: Python: nope (at offset 3)');
  check('mapAuthorLines: nothing it writes is line 0 or negative',
    [3, 4, 5, 6].every(n => /blk\.py:[1-9]\d*\b/.test(mapAuthorLines(`x (blk.py:${n})`, m, 'a'))));
  // A `before` that does NOT end in a newline: the author's first line shares
  // engine line lineOffset + 1, which the plain subtraction already reaches, so
  // line lineOffset is wrapper prologue like any line above it and is left
  // alone. Blaming the author for it is exactly what the rule is here to stop.
  const joined = { ...EXPR_MODE, before: 'import shared\nimport other\nvalue = (', lineOffset: 2 };
  check('mapAuthorLines: with a wrapper that does not end a line, the author\'s first line is lineOffset + 1',
    mapAuthorLines('x (blk.py:3)', joined, 'a\nb') === 'x (blk.py:1)' && mapAuthorLines('x (blk.py:4)', joined, 'a\nb') === 'x (blk.py:2)');
  check('… and its last line, which the author\'s code does NOT continue, is left alone',
    mapAuthorLines('ImportError (blk.py:2)', joined, 'a\nb') === 'ImportError (blk.py:2)');
  check('mapAuthorLines: a mode that generates nothing changes nothing',
    mapAuthorLines('x (blk.py:7)\n  File "blk.py", line 7, in f', QUIET_MODE, 'a\n'.repeat(9))
      === 'x (blk.py:7)\n  File "blk.py", line 7, in f');
}
if (HAS_PYTHON) {
  const expr = modeOne(['expr', 'stmt'], 'twice(ctx["n"]) if False else 7');
  check('a mode job runs the first mode and returns its value', expr.ok === true && expr.value === 7, show(expr));
  check('… on exactly one Engine', expr.constructions === 1, String(expr.constructions));
  const merged = modeOne(['expr'], 'twice(4)');
  check('the mode\'s files are merged over the contract\'s, so the entry sees both', merged.ok === true && merged.value === 8, show(merged));
  const fell = modeOne(['expr', 'stmt'], 'a = 1\nresult = a + 1');
  check('a source error moves to the next mode in the list', fell.ok === true && fell.value === 2, show(fell));
  check('… on a second fresh Engine', fell.constructions === 2, String(fell.constructions));
  const single = modeOne(['stmt'], 'result = 3');
  check('one mode is a list of one', single.ok === true && single.value === 3 && single.constructions === 1, show(single));
  const guestStop = modeOne(['expr', 'stmt'], 'ctx["missing"]');
  check('a guest error does NOT move to the next mode', guestStop.ok === false && guestStop.errorKind === 'guest' && guestStop.constructions === 1, show(guestStop));
  const bothBad = modeOne(['expr', 'stmt'], 'if');
  check('both phases failing reports the LAST one, tried once each', bothBad.ok === false && bothBad.errorKind === 'source' && bothBad.constructions === 2, show(bothBad));
  const ownCall = modeOne(['quiet'], 'x = 1\n', { args: [] });
  check('a mode may name its own call, so a compile-only wrapping needs no shared entry point', ownCall.ok === true && ownCall.value === null, show(ownCall));
  const quietBad = modeOne(['quiet'], 'def f(:\n', { args: [] });
  check('… and the block is still compiled: a syntax error is `source` at line 1', quietBad.ok === false && quietBad.errorKind === 'source' && /blk\.py:1:/.test(quietBad.error), show(quietBad));

  // Line attribution — the runner's promise.
  const atLine = (r) => (r.error ?? '').match(/blk\.py:(\d+)/)?.[1];
  const first = modeOne(['stmt'], 'result = ctx["missing"]');
  check('an error on the author\'s FIRST line says line 1', first.ok === false && atLine(first) === '1', show(first));
  const later = modeOne(['stmt'], 'a = 1\nb = 2\nresult = ctx["missing"]');
  check('an error on a LATER line says that line', atLine(later) === '3', show(later));
  const nested = modeOne(['stmt'], 'def f():\n    return ctx["missing"]\nresult = f()');
  check('a nested call renumbers every frame, in the traceback\'s order',
    nested.ok === false && /\(blk\.py:2\)/.test(nested.error) && /blk\.py", line 3[\s\S]*blk\.py", line 2/.test(nested.error), show(nested));
  const spliced = modeOne(['expr'], 'ctx["missing"]');
  check('the splice line is the author\'s first line, not line 0', atLine(spliced) === '1', show(spliced));
  const phase = modeOne(['expr', 'stmt'], 'a = 1\nb = ctx["missing"]');
  check('the FALLBACK phase is attributed with the fallback\'s own offset', atLine(phase) === '2', show(phase));
  check('… which the first phase\'s offset would have got wrong', 2 - EXPR_MODE.lineOffset < 1 && 2 === 3 - STMT_MODE.lineOffset + 0);
  const tail = modeOne(['expr'], 'ctx[');
  check('a compile error pointing into the wrapper\'s TAIL is capped at the author\'s last line', atLine(tail) === '1', show(tail));
  // The underflow the owner asked about: an import on the wrapper's FIRST line
  // fails on its own account. The runner reports no author line rather than the
  // wrong one — the text is the engine's, unchanged.
  const prologue = modeOne(['prologue'], 'x = 1\n');
  const rawPrologue = modeOne(['prologue'], 'x = 1\n', {}, { ...MODE_PROFILE, python: { ...MODE_PROFILE.python, modes: [{ ...PROLOGUE_MODE, lineOffset: 0 }] } });
  check('a failure ABOVE the author\'s code is not mapped at all, byte for byte',
    prologue.ok === false && prologue.error === rawPrologue.error && /blk\.py:1\b/.test(prologue.error), show(prologue));
  check('… and no line 0 or negative appears anywhere in it', !/line (0|-\d)|:(0|-\d)\b/.test(prologue.error), show(prologue));

  // The regression guard: nothing about a job that carries its own files
  // changed, with or without a profile that carries modes.
  const FILES_JOB = { id: 'p', language: 'python', mode: 'python-project', files: { 'main.py': 'def run(ctx):\n    return ctx["n"] * 2\n' }, entry: 'main', call: 'run', args: [{ n: 21 }] };
  const plain = send({ v: 1, jobs: [FILES_JOB] }).results[0];
  const beside = send({ v: 1, profile: MODE_PROFILE, jobs: [FILES_JOB] }).results[0];
  check('a files job is untouched by a profile that carries modes', same(plain, beside) && plain.value === 42, show([plain, beside]));
  const legacyFallback = send({ v: 1, profile: MODE_PROFILE, jobs: [{ ...FILES_JOB, files: { 'main.py': 'def run(ctx:\n' }, fallbackOnSourceError: { files: { 'main.py': 'def run(ctx):\n    return "old shape"\n' } } }] }).results[0];
  check('fallbackOnSourceError still behaves exactly as it did', legacyFallback.ok === true && legacyFallback.value === 'old shape', show(legacyFallback));
  const unmapped = send({ v: 1, profile: MODE_PROFILE, jobs: [{ ...FILES_JOB, files: { 'main.py': 'def run(ctx):\n    return ctx["missing"]\n' } }] }).results[0];
  check('a files job\'s lines are NOT rewritten: there is no wrapper to subtract', /main\.py:2/.test(unmapped.error), show(unmapped));

  // Called directly, with no profile to unfold from: a result, not a throw.
  const before = constructions;
  const orphan = runScriptJob(CountingEngine, { id: 'm', language: 'python', mode: 'python-project', modes: ['expr'], source: '1' }, { languages: identity.languages });
  check('a mode job handed straight to the runner with no profile fails closed as `host`',
    orphan.ok === false && orphan.errorKind === 'host' && /no profile that defines any/.test(orphan.error) && constructions === before, show(orphan));
  const missing = runScriptJob(CountingEngine, { id: 'm', language: 'python', mode: 'python-project', modes: ['nope'], source: '1' }, { languages: identity.languages, profile: MODE_PROFILE });
  check('… and one naming a mode the profile lacks says which mode',
    missing.ok === false && missing.errorKind === 'host' && /defines no script mode named "nope"/.test(missing.error) && constructions === before, show(missing));
} else {
  const r = modeOne(['expr'], '1');
  check('a mode job on a JavaScript-only install is `unsupported`', r.ok === false && r.errorKind === 'unsupported', show(r));
  check('… and was never attempted (no Engine built)', r.constructions === 0, String(r.constructions));
}

// ---------------------------------------------------------------------------
// 7. A real consumer's contract, vendored: the profile is ALL a host gets.
// ---------------------------------------------------------------------------
console.log('\npython modes: a vendored contract');
{
  const FIXTURE = JSON.parse(fs.readFileSync(path.join(HERE, 'fixtures', 'formlogic-python-contract.json'), 'utf8'));
  const vendored = { v: 1, preamble: '', preambleSha256: sha256(''), python: FIXTURE.python };
  const checked = validateScriptRequest({ v: 1, profile: vendored, jobs: [] }, { sha256 });
  check('the vendored contract is a valid profile', checked.ok === true, checked.ok ? '' : checked.error.message);
  check('it carries five wrappings, three file names and one entry point',
    FIXTURE.python.modes.length === 5
    && new Set(FIXTURE.python.modes.map(m => m.block)).size === 1
    && new Set(FIXTURE.python.modes.flatMap(m => Object.keys(m.files))).size === 1
    && new Set(FIXTURE.python.modes.map(m => m.lineOffset)).size === 2,
    show(FIXTURE.python.modes.map(m => [m.name, m.lineOffset])));
  check('its provenance says where it came from and at which commit',
    /^[0-9a-f]{40}$/.test(FIXTURE.vendoredFrom.commit) && Object.keys(FIXTURE.vendoredFrom.blobs).length === 6
    && Object.values(FIXTURE.vendoredFrom.blobs).every(b => /^[0-9a-f]{40}$/.test(b)));
  // Provenance that is CHECKED, not asserted: every case either names a case of
  // the vendored corpus (whose own id list is recorded here, at that blob) or
  // says it is this repo's own and why. A case cannot quietly become an
  // invention with somebody else's id on it.
  {
    const corpusIds = new Set(FIXTURE.vendoredFrom.corpusCaseIds);
    const orphans = FIXTURE.cases.filter(c => (c.formlogicId ? !corpusIds.has(c.formlogicId) : typeof c.addedByOaiy !== 'string'));
    check(`every case is a corpus case by id or says it is ours (${FIXTURE.cases.length} cases, ${corpusIds.size} corpus ids)`,
      orphans.length === 0 && corpusIds.size === 58, orphans.map(c => c.id).join(', '));
    const bothOrNeither = FIXTURE.cases.filter(c => (c.formlogicId === undefined) === (c.addedByOaiy === undefined));
    check('… and never both at once', bothOrNeither.length === 0, bothOrNeither.map(c => c.id).join(', '));
  }
  // The round trip: what the validator hands back is the contract that went in.
  {
    const roundTripped = checked.ok ? checked.request.profile.python : null;
    const fields = ['name', 'files', 'block', 'before', 'after', 'lineOffset', 'call'];
    const drift = (roundTripped?.modes ?? []).flatMap((m, i) =>
      fields.filter(f => !same(m[f], FIXTURE.python.modes[i][f])).map(f => `${m.name}.${f}`));
    check('the validator round-trips every mode field unchanged',
      roundTripped !== null && roundTripped.modes.length === FIXTURE.python.modes.length && drift.length === 0
      && roundTripped.contract === FIXTURE.python.contract && roundTripped.entry === FIXTURE.python.entry
      && roundTripped.call === FIXTURE.python.call && same(roundTripped.files, FIXTURE.python.files),
      drift.join(', '));
  }
  if (HAS_PYTHON) {
    for (const c of FIXTURE.cases) {
      const r = send({ v: 1, profile: vendored, jobs: [{ id: c.id, language: 'python', mode: 'python-project', modes: c.modes, source: c.source, args: c.args ?? [c.context ?? FIXTURE.context] }] });
      const got = 'error' in r ? { refused: r.error } : r.results[0];
      if ('value' in c.expect) {
        check(`${c.id}: ${c.proves}`, got.ok === true && same(got.value, c.expect.value), show(got));
      } else {
        const lines = (got.error ?? '').match(/logic_block\.py[":,. ]*(?:line )?(\d+)/g)?.map(s => Number(s.match(/(\d+)/)[1])) ?? [];
        check(`${c.id}: ${c.proves}`,
          got.ok === false && got.errorKind === c.expect.errorKind && new RegExp(c.expect.error).test(got.error)
          && (c.expect.authorLines === undefined || same(lines, c.expect.authorLines)),
          show(got));
      }
    }
    // The point of the whole widening, said once: the profile is the only
    // thing the host has, and nothing of the requester's is anywhere else.
    const source = fs.readFileSync(path.join(HERE, '..', 'vendor', 'oaiy-core', 'src', 'zipp-script.ts'), 'utf8');
    const schemas = ['script-profile.schema.json', 'script-request.schema.json'].map(n => fs.readFileSync(path.join(PROTOCOL, n), 'utf8')).join('\n');
    const theirs = [
      FIXTURE.python.contract, FIXTURE.python.call, FIXTURE.python.entry,
      ...Object.keys(FIXTURE.python.files),
      ...FIXTURE.python.modes.map(m => m.block),
      ...FIXTURE.python.modes.flatMap(m => Object.keys(m.files)),
      ...FIXTURE.python.modes.map(m => m.call).filter(Boolean),
    ].filter(name => name.length > 4);
    const leaked = [...new Set(theirs)].filter(name => source.includes(name) || schemas.includes(name));
    check('not one name of theirs is in the envelope or the schemas', leaked.length === 0, leaked.join(', '));
  }
}

// ---------------------------------------------------------------------------
// 8. The schemas agree with the code on every fixture this file sent.
// ---------------------------------------------------------------------------
console.log('\nschemas');
{
  const load = (name) => JSON.parse(fs.readFileSync(path.join(PROTOCOL, name), 'utf8'));
  const requestSchema = load('script-request.schema.json');
  const resultSchema = load('script-result.schema.json');
  const profileSchema = load('script-profile.schema.json');
  const validate = makeValidator([requestSchema, resultSchema, profileSchema]);

  check('the validator rejects a known-bad document', !validate(requestSchema, { v: 1, jobs: [{ id: 'x', mode: 'program' }] }));
  check('the validator accepts a known-good document', validate(requestSchema, { v: 1, jobs: [job('x', 'program', '1')] }));
  const visible = fixtures.filter(f => !f.schemaBlind);
  const disagreements = visible.filter(f => validate(requestSchema, f.request) !== f.accepted);
  check(`every request fixture validates exactly when the runner accepted it (${visible.length} fixtures, ${visible.filter(f => f.accepted).length} accepted)`,
    disagreements.length === 0, disagreements.map(d => `${d.accepted ? 'accepted but invalid' : 'refused but valid'}: ${show(d.request).slice(0, 200)}`).join('; '));
  const blind = fixtures.filter(f => f.schemaBlind);
  check('the sha/collision refusals are schema-valid (the host, not the schema, refuses them)', blind.length > 0 && blind.every(f => validate(requestSchema, f.request) && !f.accepted));
  const badResponses = responses.map(({ constructions: _c, ...r }) => r).filter(r => !validate(resultSchema, r));
  check(`every response validates against script-result.schema.json (${responses.length})`, badResponses.length === 0, badResponses.map(r => show(r).slice(0, 200)).join('; '));
  check('the profile schema accepts the profile fixture', validate(profileSchema, { v: 1, preamble: 'var a = 1;', preambleSha256: sha256('var a = 1;'), hooks: { lane: { prepare: 'prep' } }, python: { contract: 'x/1', files: { 'a.py': '' }, entry: 'a', call: 'run' } }));
  check('the profile schema rejects a hook with a reserved word', !validate(profileSchema, { v: 1, preamble: '', preambleSha256: sha256(''), hooks: { lane: { prepare: 'class' } } }));
  const steps = requestSchema.$defs.instructionSteps;
  check('the schema budget ceiling is the engine\'s (PROFILE.json) and the executor\'s',
    steps.maximum === profile.limits.maxInstructionBudgetSteps && steps.maximum === ZIPP_MAX_INSTRUCTION_BUDGET_STEPS && profileSchema.properties.instructionSteps.maximum === steps.maximum);
  check('the schema errorKind enum is SCRIPT_ERROR_KINDS', same(resultSchema.$defs.failed.properties.errorKind.enum, SCRIPT_ERROR_KINDS));
  check('the schema mode enum is SCRIPT_JS_MODES', same(requestSchema.$defs.javascriptJob.properties.mode.enum, SCRIPT_JS_MODES));
  // Refusing a shadowing key is a PROTOCOL change, so the schema says it too
  // and the two lists are one list. A host that validates before it sends gets
  // the same answer as the host that runs it.
  check('the schema refuses exactly SCRIPT_REFUSED_GLOBALS as globals keys',
    same(requestSchema.$defs.javascriptJob.properties.globals.propertyNames.not.enum, SCRIPT_REFUSED_GLOBALS),
    show(requestSchema.$defs.javascriptJob.properties.globals.propertyNames?.not?.enum));
  check('the schema names the languages the code accepts',
    requestSchema.$defs.javascriptJob.properties.language.const === 'javascript' && requestSchema.$defs.pythonJob.properties.language.const === 'python');
  // Every bound a mode is held to is stated once, in the code, and repeated in
  // the schema so a host that validates before it sends gets the same answer.
  check('the schema mode bounds are the code\'s',
    profileSchema.properties.python.properties.modes.maxItems === SCRIPT_MAX_PROFILE_MODES
    && requestSchema.$defs.pythonJob.properties.modes.maxItems === SCRIPT_MAX_JOB_MODES
    && profileSchema.$defs.pythonMode.properties.lineOffset.maximum === SCRIPT_MAX_LINE_OFFSET
    && profileSchema.$defs.pythonMode.properties.lineOffset.minimum === 0,
    `${profileSchema.properties.python.properties.modes.maxItems}/${requestSchema.$defs.pythonJob.properties.modes.maxItems}/${profileSchema.$defs.pythonMode.properties.lineOffset.maximum}`);
  check('a job names a mode by the same grammar a profile defines one with',
    requestSchema.$defs.pythonJob.properties.modes.items.$ref === `${profileSchema.$id}#/$defs/pythonModeName`
    && profileSchema.$defs.pythonMode.properties.name.$ref === '#/$defs/pythonModeName');
}

/**
 * The subset of JSON Schema 2020-12 the three files use, enough to hold the
 * code to them without a dependency: type, const, enum, properties, required,
 * additionalProperties, propertyNames, min/max on strings, numbers, arrays
 * and objects, pattern, items, allOf/anyOf/oneOf/not, if/then/else, $ref by
 * `$id` and `#/` pointer.
 */
function makeValidator(schemas) {
  const byId = new Map(schemas.map(s => [s.$id, s]));
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
    if (schema.enum && !schema.enum.some(e => same(e, v))) return false;
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
      if (schema.maxItems !== undefined && v.length > schema.maxItems) return false;
      if (schema.items !== undefined && !v.every(x => valid(schema.items, x, root))) return false;
    }
    if (typeOf(v) === 'object') {
      const keys = Object.keys(v);
      if (schema.required && !schema.required.every(k => keys.includes(k))) return false;
      if (schema.minProperties !== undefined && keys.length < schema.minProperties) return false;
      if (schema.propertyNames && !keys.every(k => valid(schema.propertyNames, k, root))) return false;
      for (const k of keys) {
        if (schema.properties && has(schema.properties, k)) {
          if (!valid(schema.properties[k], v[k], root)) return false;
        } else if (schema.additionalProperties !== undefined && !valid(schema.additionalProperties, v[k], root)) {
          return false;
        }
      }
    }
    if (schema.allOf && !schema.allOf.every(s => valid(s, v, root))) return false;
    if (schema.anyOf && !schema.anyOf.some(s => valid(s, v, root))) return false;
    if (schema.oneOf && schema.oneOf.filter(s => valid(s, v, root)).length !== 1) return false;
    if (schema.not && valid(schema.not, v, root)) return false;
    if (schema.if) {
      const branch = valid(schema.if, v, root) ? schema.then : schema.else;
      if (branch !== undefined && !valid(branch, v, root)) return false;
    }
    return true;
  }
  return (schema, v) => valid(schema, v, schema);
}

// ---------------------------------------------------------------------------
console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
assert.equal(failures.length, 0);
