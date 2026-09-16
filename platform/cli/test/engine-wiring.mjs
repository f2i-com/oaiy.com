// Every runtime the CLI builds goes through `createCliEngine`.
//
//     node test/engine-wiring.mjs
//
// `createCliEngine` (src/engine.ts) is where the ZIPP executor is attached and
// `requireScriptExecutor` set last, so a `createEngine(` anywhere else in
// cli/src or cli/test is a runtime that might run flows on the host engine.
// This reads the sources and allows exactly two things:
//
//   * a call inside the body of `createCliEngine` in src/engine.ts;
//   * the one raw call in test/zipp-engine.ts, recognised by its SHAPE — an
//     options object that hands the runtime `createZippThreadExecutor(…)` with
//     its own `workerEntry` and sets `requireScriptExecutor: true`. It exists
//     because the worker-crash case needs a worker entry that dies, which
//     `createCliEngine` rightly gives no caller a way to supply; it is still
//     a ZIPP executor and still required.
//
// An `import` of `createEngine` in any other file is a finding too, before it
// is ever called. Comments are skipped (this file's own prose would otherwise
// count). The checker runs on a synthetic source set first, to show it sees a
// stray call; then on the tree.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const cli = fileURLToPath(new URL('../', import.meta.url));

function* sources(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name === 'generated' || entry.name === 'fixtures') continue;
      yield* sources(p);
    } else if (/\.(ts|mts|mjs|js|cjs)$/.test(entry.name) && !entry.name.endsWith('.d.ts')) {
      const file = path.relative(cli, p).replace(/\\/g, '/');
      // This checker's own patterns and self-test strings are not wiring.
      if (file === 'test/engine-wiring.mjs') continue;
      yield { file, text: fs.readFileSync(p, 'utf8') };
    }
  }
}

/** Index of the matching `}` for the `{` at `open`, ignoring strings and comments coarsely. */
function closeOf(text, open) {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    const c = text[i];
    if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return i;
  }
  return -1;
}

/** Is `index` on a line that is a `//` or ` * ` comment? */
function inComment(text, index) {
  const lineStart = text.lastIndexOf('\n', index) + 1;
  const lead = text.slice(lineStart, index).trimStart();
  return lead.startsWith('//') || lead.startsWith('*') || lead.startsWith('/*');
}

/** The span of `export async function createCliEngine(…) { … }` in engine.ts. */
function cliEngineSpan(text) {
  const head = text.search(/export\s+async\s+function\s+createCliEngine\s*\(/);
  if (head === -1) return null;
  const open = text.indexOf('{', text.indexOf(')', head));
  const close = closeOf(text, open);
  return close === -1 ? null : [open, close];
}

/** The allowlisted zipp-engine.ts shape: the call's options object, by content. */
function isCrashInjectionSite(text, callIndex) {
  const open = text.indexOf('(', callIndex);
  const brace = text.indexOf('{', open);
  const close = closeOf(text, brace);
  if (brace === -1 || close === -1 || brace > open + 8) return false;
  const options = text.slice(brace, close + 1);
  return /createZippThreadExecutor\s*\(\s*\{[^}]*workerEntry\s*:/.test(options) &&
    /requireScriptExecutor\s*:\s*true/.test(options);
}

export function check(files) {
  const findings = [];
  let inside = 0;
  for (const { file, text } of files) {
    const isEngine = file === 'src/engine.ts';
    const span = isEngine ? cliEngineSpan(text) : null;
    if (isEngine && !span) findings.push(`${file}: createCliEngine not found`);

    for (const m of text.matchAll(/\bcreateEngine\s*\(/g)) {
      if (inComment(text, m.index)) continue;
      if (span && m.index > span[0] && m.index < span[1]) {
        inside++;
        continue;
      }
      if (file === 'test/zipp-engine.ts' && isCrashInjectionSite(text, m.index)) continue;
      const line = text.slice(0, m.index).split('\n').length;
      findings.push(`${file}:${line}: createEngine( outside createCliEngine`);
    }

    if (!isEngine && file !== 'test/zipp-engine.ts') {
      for (const m of text.matchAll(/import\s*\{[^}]*\bcreateEngine\b[^}]*\}\s*from/g)) {
        if (inComment(text, m.index)) continue;
        const line = text.slice(0, m.index).split('\n').length;
        findings.push(`${file}:${line}: imports createEngine`);
      }
    }
  }
  if (inside === 0) findings.push('src/engine.ts: createCliEngine never calls createEngine');
  return findings;
}

// --- self-test --------------------------------------------------------------
const tree = [...sources(path.join(cli, 'src')), ...sources(path.join(cli, 'test'))];
{
  const stray = [...tree, { file: 'src/stray.ts', text: "import { createEngine } from '../../ui/src/engine/createEngine';\nexport const e = createEngine({});\n" }];
  const f = check(stray);
  assert.ok(f.some((x) => x.startsWith('src/stray.ts:2:')), `a stray createEngine( was not seen: ${JSON.stringify(f)}`);
  assert.ok(f.some((x) => x.startsWith('src/stray.ts:1:')), `a stray import was not seen: ${JSON.stringify(f)}`);
  const widened = tree.map((s) => s.file === 'test/zipp-engine.ts'
    ? { ...s, text: s.text + "\nconst loose = createEngine({ requireScriptExecutor: false });\n" }
    : s);
  assert.ok(check(widened).some((x) => x.startsWith('test/zipp-engine.ts:')), 'a second raw call in zipp-engine.ts was not seen');
  console.log('PASS: the checker sees a stray createEngine( and a stray import');
}

// --- the tree ----------------------------------------------------------------
const findings = check(tree);
for (const f of findings) console.error(`  FAIL ${f}`);
if (findings.length === 0) {
  console.log(`PASS: every createEngine( in cli/src and cli/test is inside createCliEngine (${tree.length} files), bar the shaped crash-injection site in test/zipp-engine.ts`);
}
process.exit(findings.length ? 1 : 0);
