// Guard A: the built CLI carries no way to compile a string into host code
// except the two sites this file names.
//
//     node test/no-host-eval.mjs [dist/oaiy.mjs [dist/oaiy-zipp-worker.mjs dist/oaiy-script-worker.mjs …]]
//
// Every `Function(...)` / `new Function(...)` (directly, via `globalThis` or
// `window`, or through the `.constructor.constructor` idiom that recovers a
// function constructor from any function), every `eval(...)`, every use of
// the `AsyncFunction` / `GeneratorFunction` names, and every import or
// require of `node:vm` in dist/oaiy.mjs and dist/oaiy-zipp-worker.mjs is
// found in the syntax tree — so the string literal in which oaiy-core keeps
// its browser Worker's bootstrap source is not a finding, and a call that is
// one IS, however it is spelled or where it lands after bundling.
//
// "However it is spelled" was a claim this file did not keep. A review walked
// twelve spellings past it, so it now also flags, by shape:
//
//   * ANY bare reference to `Function` or `eval` that is not the callee the
//     rules above already judge — which is what catches `const F = Function`,
//     `Reflect.construct(Function, …)`, ``Function`return 1` `` and the
//     indirect `(0, eval)('1')`. All three bundles reference those two names
//     in exactly two places, both allowlisted callees, so the rule costs
//     nothing and a third reference is a finding;
//   * ANY computed access on `globalThis`/`window`/`self`/`global`, because
//     `globalThis['eval']` and `globalThis['Fun'+'ction']` are the same door
//     with the handle moved. Zero in all three bundles today;
//   * `.constructor.constructor` with either half written as a computed
//     literal, and `.constructor` read off a CALL's result —
//     `Object.getPrototypeOf(async function(){}).constructor` has no name to
//     match on;
//   * a dynamic `import()` whose specifier is not a string literal (so
//     `import('node:' + 'vm')` cannot hide) or IS a `data:` URL (a module made
//     of a string, which is `eval` with an import statement);
//   * `new Worker(src, { eval: true })` — worker_threads' own way to run a
//     string — flagged whenever an `eval` key is present at all. The CLI's own
//     ZIPP worker spawns by path with no such key;
//   * `process.binding` / `process._linkedBinding` / `internalBinding`, which
//     reach Node's `contextify` bindings under the public API.
//
// The three the tripwire (Guard B) cannot see are the last three: a `data:`
// import, a worker_threads `eval: true`, and `process.binding`. The rest are
// belt as well as braces — B catches them at run time, A catches them in the
// bytes — and none of the twelve appears anywhere in the sources.
//
// The allowlist is by the SHAPE of the call's arguments, never by a line in
// the bundle (those move with every build):
//
//   * `new Function('try { return import.meta } catch(e) { return undefined }')`
//     — oaiy-core/src/logger.ts asking whether it runs under Vite. Permanent.
//   * `new Function('host', 'console', <identifier>)`
//     — oaiy-core/src/runtime.ts, the in-thread script path the browser
//     still uses for trusted flows. Unreachable from the CLI (`createCliEngine`
//     sets `requireScriptExecutor`, and test/zipp-guard.mjs's tripwire shows it
//     never fires) but present in the bundle until the runtime drops it
//     (Phase 6 of the ZIPP plan). When it goes, this entry goes with it — the
//     count below is exact, so its disappearance fails this guard on purpose.
//
// oaiy-core's five package loaders (`new Function('exports', 'require',
// 'module', code)` in custom-node-registry.ts, module-discovery.ts and
// node-extension-registry.ts) do not survive tree-shaking into either bundle
// today and are deliberately NOT allowlisted: should a change pull one in,
// this guard fails and the reason gets written down here, not waved through.
//
// The guard tests itself before it reports: a copy of the CLI bundle with
// `new Function('x')` appended must produce exactly one finding.
import * as acorn from 'acorn';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const bundles = process.argv.length > 2
  ? process.argv.slice(2)
  : [path.join(dist, 'oaiy.mjs'), path.join(dist, 'oaiy-zipp-worker.mjs'), path.join(dist, 'oaiy-script-worker.mjs')];

const LOGGER_PROBE = 'try { return import.meta } catch(e) { return undefined }';
const VM_SPECIFIERS = new Set(['vm', 'node:vm']);
const RECOVERY_NAMES = new Set(['AsyncFunction', 'GeneratorFunction', 'AsyncGeneratorFunction']);

/** Allowlisted shapes, by name. Each returns true when `args` is that site. */
const ALLOWED = {
  'logger import.meta probe (oaiy-core/src/logger.ts; permanent)': (args) =>
    args.length === 1 && isString(args[0], LOGGER_PROBE),
  "runtime in-thread script path new Function('host','console',script) (oaiy-core/src/runtime.ts; expires Phase 6)": (args) =>
    args.length === 3 && isString(args[0], 'host') && isString(args[1], 'console') && args[2].type === 'Identifier',
};
/** How many times each allowlisted shape is expected per bundle. */
const EXPECTED = {
  'oaiy.mjs': { [Object.keys(ALLOWED)[0]]: 1, [Object.keys(ALLOWED)[1]]: 1 },
  'oaiy-zipp-worker.mjs': { [Object.keys(ALLOWED)[0]]: 0, [Object.keys(ALLOWED)[1]]: 0 },
  // The leaf-script shell (PR5). zipp-script.ts's `new Function` / `eval` are
  // GUEST program text inside string literals, never calls in this realm.
  'oaiy-script-worker.mjs': { [Object.keys(ALLOWED)[0]]: 0, [Object.keys(ALLOWED)[1]]: 0 },
};

function isString(node, value) {
  return node && node.type === 'Literal' && typeof node.value === 'string' && (value === undefined || node.value === value);
}

/** `Function`, `globalThis.Function`, `window.Function`, `self.Function`. */
function namesFunction(callee) {
  if (callee.type === 'Identifier') return callee.name === 'Function';
  if (callee.type === 'MemberExpression' && !callee.computed && callee.property.type === 'Identifier') {
    return callee.property.name === 'Function' && callee.object.type === 'Identifier' &&
      ['globalThis', 'window', 'self', 'global'].includes(callee.object.name);
  }
  return false;
}

/** The name a member expression reads, written either way: `.x` or `["x"]`. */
function memberName(node) {
  if (!node || node.type !== 'MemberExpression') return null;
  if (node.computed) return node.property.type === 'Literal' ? String(node.property.value) : null;
  return node.property.type === 'Identifier' ? node.property.name : null;
}

/** `Worker`, `threads.Worker`, `(await import('node:worker_threads')).Worker`. */
function namesWorker(callee) {
  if (callee.type === 'Identifier') return callee.name === 'Worker';
  return memberName(callee) === 'Worker';
}

/** `x.constructor.constructor` — a function constructor recovered from any function. */
function isConstructorRecovery(node) {
  return memberName(node) === 'constructor' && memberName(node.object) === 'constructor';
}

/** `f(...).constructor` — the same recovery off a value with no name to match on. */
function isCallResultConstructor(node) {
  return node.type === 'MemberExpression' && memberName(node) === 'constructor' &&
    node.object.type === 'CallExpression';
}

/** The global objects a computed lookup could pull any intrinsic out of. */
const GLOBAL_OBJECTS = ['globalThis', 'window', 'self', 'global'];

/**
 * Whether this `Function`/`eval` identifier is a REFERENCE to the intrinsic,
 * rather than a name that merely reads the same: a property (`x.Function`), an
 * object key, a binding being declared, a parameter, an import/export
 * specifier — or the callee the `CallExpression`/`NewExpression` rules already
 * judge against the allowlist, which must not be reported twice.
 */
function isIntrinsicReference(node, parent) {
  if (!parent) return false;
  switch (parent.type) {
    case 'MemberExpression':
      return !(parent.property === node && !parent.computed);
    case 'Property':
      return !(parent.key === node && !parent.computed);
    case 'CallExpression':
    case 'NewExpression':
      return parent.callee !== node;
    case 'VariableDeclarator':
      return parent.id !== node;
    case 'FunctionDeclaration':
    case 'FunctionExpression':
    case 'ClassDeclaration':
    case 'ClassExpression':
      return parent.id !== node && !(parent.params ?? []).includes(node);
    case 'ArrowFunctionExpression':
      return !(parent.params ?? []).includes(node);
    case 'ImportSpecifier':
    case 'ImportDefaultSpecifier':
    case 'ImportNamespaceSpecifier':
    case 'ExportSpecifier':
    case 'MethodDefinition':
    case 'PropertyDefinition':
    case 'LabeledStatement':
    case 'BreakStatement':
    case 'ContinueStatement':
      return false;
    default:
      return true;
  }
}

/** Walk every node of an acorn tree, with its parent (no dependency on acorn-walk, which is not installed). */
function walk(node, visit, parent = null) {
  if (!node || typeof node.type !== 'string') return;
  visit(node, parent);
  for (const key of Object.keys(node)) {
    if (key === 'loc' || key === 'type') continue;
    const child = node[key];
    if (Array.isArray(child)) {
      for (const c of child) if (c && typeof c.type === 'string') walk(c, visit, node);
    } else if (child && typeof child.type === 'string') {
      walk(child, visit, node);
    }
  }
}

/**
 * Scan one bundle. Returns `{ findings, allowed }`: findings are violations
 * (`{ line, column, what, snippet }`), `allowed` counts each allowlisted shape.
 */
export function scan(source) {
  const tree = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module', locations: true, allowHashBang: true });
  const findings = [];
  const allowed = Object.fromEntries(Object.keys(ALLOWED).map((k) => [k, 0]));
  const lines = source.split('\n');
  const flag = (node, what) => {
    const { line, column } = node.loc.start;
    findings.push({ line, column, what, snippet: (lines[line - 1] ?? '').trim().slice(0, 120) });
  };

  walk(tree, (node, parent) => {
    if (node.type === 'ImportDeclaration' && isString(node.source) && VM_SPECIFIERS.has(node.source.value)) {
      flag(node, 'import of node:vm');
    }
    if (node.type === 'ImportExpression') {
      if (!isString(node.source)) flag(node, 'dynamic import of a specifier this file cannot read');
      else if (VM_SPECIFIERS.has(node.source.value)) flag(node, 'dynamic import of node:vm');
      else if (/^\s*data:/i.test(node.source.value)) flag(node, 'dynamic import of a data: URL');
    }
    if (node.type === 'CallExpression' && node.callee.type === 'Identifier' && node.callee.name === 'require' &&
        node.arguments.length === 1 && isString(node.arguments[0]) && VM_SPECIFIERS.has(node.arguments[0].value)) {
      flag(node, 'require of node:vm');
    }
    if (node.type === 'Identifier' && RECOVERY_NAMES.has(node.name)) {
      flag(node, `reference to ${node.name}`);
    }
    // A bare `Function`/`eval` that is not a callee the rules below judge:
    // an alias, an argument to `Reflect.construct`, a template tag, the
    // `eval` of `(0, eval)(...)`.
    if (node.type === 'Identifier' && (node.name === 'Function' || node.name === 'eval') &&
        isIntrinsicReference(node, parent)) {
      flag(node, `reference to ${node.name} (${parent.type})`);
    }
    // `globalThis['eval']`, `globalThis['Fun' + 'ction']`, and every other
    // spelling of the same lookup.
    if (node.type === 'MemberExpression' && node.computed && node.object.type === 'Identifier' &&
        GLOBAL_OBJECTS.includes(node.object.name)) {
      flag(node, `computed lookup on ${node.object.name}`);
    }
    if (node.type === 'MemberExpression' && isCallResultConstructor(node)) {
      flag(node, '.constructor read off a call result');
    }
    // `new Worker(source, { eval: true })`: worker_threads' own way to run a
    // string. Flagged on the PRESENCE of an `eval` key, whatever its value,
    // so a variable cannot hide it.
    if (node.type === 'NewExpression' && namesWorker(node.callee)) {
      const opts = node.arguments[1];
      if (opts && opts.type === 'ObjectExpression' &&
          opts.properties.some((p) => p.type === 'Property' && !p.computed &&
            ((p.key.type === 'Identifier' && p.key.name === 'eval') || isString(p.key, 'eval')))) {
        flag(node, 'new Worker(..., { eval: ... })');
      }
    }
    if (node.type === 'CallExpression') {
      const { callee } = node;
      if (callee.type === 'Identifier' && callee.name === 'eval') {
        flag(node, 'eval()');
      } else if (callee.type === 'Identifier' && callee.name === 'internalBinding') {
        flag(node, 'internalBinding()');
      } else if (['binding', '_linkedBinding'].includes(memberName(callee) ?? '')) {
        flag(node, `process.${memberName(callee)}()`);
      }
    }
    if (node.type === 'CallExpression' || node.type === 'NewExpression') {
      const { callee } = node;
      if (callee.type === 'Identifier' && callee.name === 'eval') {
        // already flagged above for a call; a `new eval` is not a thing
      } else if (isConstructorRecovery(callee)) {
        flag(node, 'constructor.constructor(...) recovery');
      } else if (namesFunction(callee)) {
        const shape = Object.keys(ALLOWED).find((name) => ALLOWED[name](node.arguments));
        if (shape) allowed[shape]++;
        else flag(node, `${node.type === 'NewExpression' ? 'new ' : ''}Function(...) outside the allowlist`);
      }
    }
  });
  return { findings, allowed };
}

function report(label, result, expected) {
  let ok = result.findings.length === 0;
  for (const f of result.findings) {
    console.error(`  FAIL ${label}:${f.line}:${f.column} ${f.what}\n       ${f.snippet}`);
  }
  if (expected) {
    for (const [shape, n] of Object.entries(expected)) {
      if (result.allowed[shape] !== n) {
        ok = false;
        console.error(`  FAIL ${label}: expected ${n} allowlisted site(s) of "${shape}", found ${result.allowed[shape]}`);
      }
    }
  }
  return ok;
}

// --- self-test: the scanner must see an injected site -----------------------
{
  const cli = bundles[0];
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-no-host-eval-'));
  try {
    const clean = fs.readFileSync(cli, 'utf8');
    const spiked = path.join(tmp, 'spiked.mjs');
    fs.writeFileSync(spiked, clean + "\nnew Function('x');\n");
    const before = scan(clean).findings.length;
    const after = scan(fs.readFileSync(spiked, 'utf8')).findings;
    assert.equal(after.length, before + 1, 'the scanner did not see an injected new Function(\'x\')');
    assert.match(after[after.length - 1].what, /outside the allowlist/);
    const recovered = scan(clean + "\n(async function () {}).constructor.constructor('y')();\n").findings;
    assert.equal(recovered.length, before + 1, 'the scanner did not see a constructor.constructor recovery');
    const evald = scan(clean + "\neval('1');\n").findings;
    assert.equal(evald.length, before + 1, 'the scanner did not see eval()');
    const vmd = scan(clean + "\nimport vm from 'node:vm';\n").findings;
    assert.equal(vmd.length, before + 1, 'the scanner did not see a node:vm import');
    console.log('PASS: the scanner sees an injected new Function, a constructor recovery, eval and node:vm');

    // Every spelling a review walked past this guard, each spiked into the
    // real bundle so the assertion is about THIS scanner on THIS input and not
    // about a copy of it. `>= 1` rather than `=== 1`: some shapes are two
    // findings at once (a computed lookup on `globalThis` that is ALSO a
    // constructor chain), and "at least one" is the property that matters.
    const SPELLINGS = [
      ["globalThis['Fun'+'ction']('return 1')()", 'a computed lookup on globalThis'],
      ["Reflect.construct(Function, ['return 1'])()", 'Function as an argument'],
      ['const F = Function; new F("return 1")()', 'Function aliased to a const'],
      ['Function`return 1`()', 'Function as a template tag'],
      ["(0, eval)('1')", 'indirect eval'],
      ["globalThis['eval']('1')", "globalThis['eval']"],
      ["globalThis['constructor']['constructor']('return 1')()", 'a computed constructor chain'],
      ['Object.getPrototypeOf(async function(){}).constructor("return 1")', '.constructor off a call result'],
      ["import('data:text/javascript,globalThis.x=1')", "import('data:…')"],
      ["new (await import('node:worker_threads')).Worker('1', { eval: true })", 'worker_threads with eval: true'],
      ["process.binding('contextify')", 'process.binding'],
      ["const vm2 = await import('node:' + 'vm'); vm2.runInThisContext('1')", "import('node:' + 'vm')"],
    ];
    for (const [spelling, label] of SPELLINGS) {
      // `await` needs a module top level, which the bundle is; the statement
      // is appended to it whole.
      const found = scan(`${clean}\n${spelling};\n`).findings;
      assert.ok(
        found.length >= before + 1,
        `the scanner did not see ${label}: ${spelling}`,
      );
    }
    console.log(`PASS: the scanner sees all ${SPELLINGS.length} spellings a review walked past it`);

    // And the counterweight: ordinary code that merely mentions these names
    // is NOT a finding, or the guard would be noise and get switched off.
    for (const innocent of [
      'const o = { Function: 1, eval: 2 }; export const k = o.Function + o.eval;',
      'export function h(x) { return x instanceof Object ? x.constructor : null; }',
      "const w = new Worker(new URL('./w.mjs', import.meta.url), { workerData: 1 });",
      "const m = await import('node:fs');",
      'export class Function2 {}',
    ]) {
      assert.equal(
        scan(`${clean}\n${innocent}\n`).findings.length,
        before,
        `the scanner flagged innocent code: ${innocent}`,
      );
    }
    console.log('PASS: naming these things without reaching for them is not a finding');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

// --- the bundles ------------------------------------------------------------
let ok = true;
for (const file of bundles) {
  const label = path.basename(file);
  const result = scan(fs.readFileSync(file, 'utf8'));
  const good = report(label, result, EXPECTED[label]);
  ok &&= good;
  if (good) {
    const kept = Object.entries(result.allowed).filter(([, n]) => n > 0).map(([s, n]) => `${n}× ${s.split(' (')[0]}`);
    console.log(`PASS: ${label} compiles no host code${kept.length ? ` beyond ${kept.join(', ')}` : ''}`);
  }
}
process.exit(ok ? 0 : 1);
