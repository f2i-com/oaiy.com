// Guard A: the built CLI carries no way to compile a string into host code
// except the two sites this file names.
//
//     node test/no-host-eval.mjs [dist/oaiy.mjs [dist/oaiy-zipp-worker.mjs …]]
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
  : [path.join(dist, 'oaiy.mjs'), path.join(dist, 'oaiy-zipp-worker.mjs')];

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

/** `x.constructor.constructor` — a function constructor recovered from any function. */
function isConstructorRecovery(node) {
  return node.type === 'MemberExpression' && !node.computed && node.property.type === 'Identifier' &&
    node.property.name === 'constructor' && node.object.type === 'MemberExpression' && !node.object.computed &&
    node.object.property.type === 'Identifier' && node.object.property.name === 'constructor';
}

/** Walk every node of an acorn tree (no dependency on acorn-walk, which is not installed). */
function walk(node, visit) {
  if (!node || typeof node.type !== 'string') return;
  visit(node);
  for (const key of Object.keys(node)) {
    if (key === 'loc' || key === 'type') continue;
    const child = node[key];
    if (Array.isArray(child)) {
      for (const c of child) if (c && typeof c.type === 'string') walk(c, visit);
    } else if (child && typeof child.type === 'string') {
      walk(child, visit);
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

  walk(tree, (node) => {
    if (node.type === 'ImportDeclaration' && isString(node.source) && VM_SPECIFIERS.has(node.source.value)) {
      flag(node, 'import of node:vm');
    }
    if (node.type === 'ImportExpression' && isString(node.source) && VM_SPECIFIERS.has(node.source.value)) {
      flag(node, 'dynamic import of node:vm');
    }
    if (node.type === 'CallExpression' && node.callee.type === 'Identifier' && node.callee.name === 'require' &&
        node.arguments.length === 1 && isString(node.arguments[0]) && VM_SPECIFIERS.has(node.arguments[0].value)) {
      flag(node, 'require of node:vm');
    }
    if (node.type === 'Identifier' && RECOVERY_NAMES.has(node.name)) {
      flag(node, `reference to ${node.name}`);
    }
    if (node.type === 'CallExpression' || node.type === 'NewExpression') {
      const { callee } = node;
      if (callee.type === 'Identifier' && callee.name === 'eval') {
        flag(node, 'eval()');
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
