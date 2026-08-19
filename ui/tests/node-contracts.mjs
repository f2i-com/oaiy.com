/**
 * Structural check: every input handle a node DECLARES must actually be read by
 * the module compiler that handles it.
 *
 *     npm run test:contracts
 *
 * Why this exists. The core compiler keys a node's inputs map strictly on
 * `edge.targetHandle` (vendor/oaiy-core/src/compiler.ts). So if a node's JSON and
 * its React component declare a handle called `text` but the module compiler
 * reads `inputs.get('input')`, the lookup silently misses, the generated code
 * falls back to the literal `null`, and the node processes nothing — while the
 * flow compiles, runs and reports success. That is exactly what had happened to
 * `text_chunker`, and to `input_folder`'s documented "optional dynamic path".
 *
 * Nothing else catches this: it type-checks, it builds, and it produces a
 * successful run with wrong output.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MODULES = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'src', 'bundled-modules');

/**
 * Nodes whose inputs are deliberately consumed somewhere other than their own
 * module compiler. Each entry must say WHERE, so an entry can be re-checked
 * rather than trusted forever.
 */
const HANDLED_ELSEWHERE = {
  loop_start: 'the core compiler owns loop structure — it reads the `array`/`count` handles directly in generateLoopCode (vendor/oaiy-core/src/compiler.ts, `targetHandle === \'array\'`)',
  loop_end: 'same as loop_start — structure is emitted by the core compiler, not the module',
  macro_input: 'a macro boundary marker; its value is injected by the macro runner, not read from an edge',
  macro_output: 'a macro boundary marker; the core compiler wires it',
  subflow: 'inputs are mapped by name into the subflow call, not read as handles',
  macro: 'inputs are mapped by name into the macro call, not read as handles',
};

let pass = 0;
const problems = [];

const modules = fs.readdirSync(MODULES, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => d.name)
  .sort();

for (const mod of modules) {
  const compilerFile = path.join(MODULES, mod, 'compiler.ts');
  const nodesDir = path.join(MODULES, mod, 'nodes');
  if (!fs.existsSync(compilerFile) || !fs.existsSync(nodesDir)) continue;

  const src = fs.readFileSync(compilerFile, 'utf8');
  const read = new Set([...src.matchAll(/inputs\.get\(\s*'([^']+)'\s*\)/g)].map((m) => m[1]));

  for (const file of fs.readdirSync(nodesDir).filter((f) => f.endsWith('.json')).sort()) {
    const def = JSON.parse(fs.readFileSync(path.join(nodesDir, file), 'utf8'));
    const declared = (def.inputs ?? []).map((i) => i.id ?? i.name).filter(Boolean);
    if (declared.length === 0) continue;

    const id = def.id ?? file.replace(/\.json$/, '');
    if (id in HANDLED_ELSEWHERE) {
      pass++;
      console.log(`  – ${id.padEnd(24)} skipped: ${HANDLED_ELSEWHERE[id]}`);
      continue;
    }

    // The failure that matters is a node whose handles are ALL unread — that is
    // the "silently does nothing" shape. A node reading some but not all of its
    // handles is usually an optional input, so report it as informational.
    const unread = declared.filter((h) => !read.has(h));
    if (unread.length === declared.length) {
      problems.push(
        `${mod}/${file}: node "${id}" declares ${JSON.stringify(declared)} but its compiler reads ` +
        `none of them (it reads ${JSON.stringify([...read].sort())}). Edges to this node are dropped.`,
      );
      console.log(`  ✗ ${id.padEnd(24)} declares ${JSON.stringify(declared)}, compiler reads none`);
    } else if (unread.length > 0) {
      pass++;
      console.log(`  ✓ ${id.padEnd(24)} reads ${declared.length - unread.length}/${declared.length} handles (unread: ${JSON.stringify(unread)})`);
    } else {
      pass++;
      console.log(`  ✓ ${id.padEnd(24)} all ${declared.length} declared handle(s) read`);
    }
  }
}

console.log(`\n${'-'.repeat(60)}`);
if (problems.length) {
  console.log(`node contracts: ${problems.length} node(s) would silently drop their inputs\n`);
  for (const p of problems) console.log(`  - ${p}`);
  process.exit(1);
}
console.log(`node contracts: clean (${pass} nodes checked)`);
