#!/usr/bin/env node
/**
 * Regenerate src/softn/knowledge/: the SoftN reference the agent searches and
 * reads, taken from softn.com (Apache-2.0):
 *
 *   index.generated.ts, components.generated.ts, docs.generated.ts
 *     — SoftN Studio's own agent knowledge (apps/softn-studio/src/lib/agent/
 *       knowledge/), itself generated from the component manifest and the
 *       published docs: every component's props, events and an example, and
 *       every guide as text;
 *   examples.generated.ts
 *     — a few complete apps from softn.com's catalogue (apps/softn-api/data/
 *       apps/<slug>/v1.softn), kept as the .softn bytes, for the agent to
 *       read or copy into the project.
 *
 * The output is committed (bot.computer works offline); run this after a
 * SoftN update:   SOFTN_REPO=../softn.com node scripts/generate-softn-knowledge.mjs
 */
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { unzipSync } from 'fflate';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(process.env.SOFTN_REPO || join(here, '..', '..', 'softn.com'));
const out = join(here, '..', 'src', 'softn', 'knowledge');
mkdirSync(out, { recursive: true });

let commit = 'unknown';
try {
  const head = readFileSync(join(repo, '.git', 'HEAD'), 'utf8').trim();
  commit = head.startsWith('ref: ') ? readFileSync(join(repo, '.git', head.slice(5)), 'utf8').trim() : head;
} catch {
  /* not a git checkout */
}
const header = (from) => `// Copied by scripts/generate-softn-knowledge.mjs from softn.com (${from} at ${commit.slice(0, 12)}),\n// Copyright f2i-com, Apache License 2.0. Do not edit by hand.\n`;

const studio = join(repo, 'apps/softn-studio/src/lib/agent/knowledge');
for (const name of ['index.generated.ts', 'components.generated.ts', 'docs.generated.ts']) {
  const source = readFileSync(join(studio, name), 'utf8');
  if (/^\s*import\s/m.test(source)) throw new Error(`${name} imports something; this copy expects a self-contained module`);
  writeFileSync(join(out, name), `${header(`apps/softn-studio/src/lib/agent/knowledge/${name}`)}${source}`);
  console.log(`copied ${name} (${(source.length / 1024).toFixed(1)} KiB)`);
}

// Small, varied, complete apps: CRUD with storage, a game with state and
// sound, components and charts across pages, 3D, WebGPU, device permissions.
const EXAMPLES = ['notes', 'twenty48', 'showcase', 'three-demo', 'gpu-demo', 'device-kit'];
const examples = [];
for (const slug of EXAMPLES) {
  const dir = join(repo, 'apps/softn-api/data/apps', slug);
  const bytes = readFileSync(join(dir, 'v1.softn'));
  const info = JSON.parse(readFileSync(join(dir, 'app.json'), 'utf8')).app ?? {};
  const files = Object.keys(unzipSync(new Uint8Array(bytes))).filter((n) => !n.endsWith('/')).sort();
  if (!files.includes('manifest.json')) throw new Error(`${slug}: no manifest.json at the root`);
  const manifest = JSON.parse(new TextDecoder().decode(unzipSync(new Uint8Array(bytes))['manifest.json']));
  examples.push({
    slug,
    name: String(manifest.name ?? info.name ?? slug),
    description: String(info.description ?? manifest.description ?? ''),
    tags: [info.category, ...(info.tags ?? [])].filter(Boolean),
    files,
    softn: bytes.toString('base64'),
  });
  console.log(`packed example ${slug} (${(bytes.length / 1024).toFixed(1)} KiB, ${files.length} files)`);
}
writeFileSync(
  join(out, 'examples.generated.ts'),
  `${header('apps/softn-api/data/apps/*/v1.softn')}
export interface SoftnExample { slug: string; name: string; description: string; tags: string[]; files: string[]; /** The .softn, base64. */ softn: string }
export const SOFTN_EXAMPLES: SoftnExample[] = ${JSON.stringify(examples, null, 1)};
`,
);
console.log(`wrote ${out}`);
