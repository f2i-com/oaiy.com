// The flow editor's node types, compact, for the agent's flow tools: `npm run flow-nodes` writes
// src/agent/flowNodes.json from platform/ui/src/bundled-modules/*/nodes/*.json (run it when a node changes).
import { readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const modules = join(here, '../../platform/ui/src/bundled-modules');
const out = join(here, '../src/agent/flowNodes.json');

const oneLine = (s) => String(s ?? '').replace(/\s+/g, ' ').trim();
const nodes = [];
for (const mod of readdirSync(modules).sort()) {
  const dir = join(modules, mod, 'nodes');
  if (!existsSync(dir)) continue;
  for (const file of readdirSync(dir).filter((f) => f.endsWith('.json')).sort()) {
    const d = JSON.parse(readFileSync(join(dir, file), 'utf8'));
    nodes.push({
      type: d.id,
      name: d.name,
      module: mod,
      description: oneLine(d.description).slice(0, 220),
      inputs: (d.inputs ?? []).map((i) => ({ id: i.id, type: i.type ?? 'any' })),
      outputs: (d.outputs ?? []).map((o) => ({ id: o.id, type: o.type ?? 'any' })),
      properties: (d.properties ?? [])
        .filter((p) => p.id)
        .map((p) => ({
          id: p.id,
          type: p.type,
          ...(p.default !== undefined && p.default !== '' ? { default: p.default } : {}),
          ...(p.options ? { options: p.options.map((o) => (typeof o === 'object' ? o.value : o)).slice(0, 12) } : {}),
          ...(p.advanced ? { advanced: true } : {}),
        })),
    });
  }
}
writeFileSync(out, `${JSON.stringify(nodes, null, 1)}\n`);
console.log(`${nodes.length} node types → ${out}`);
