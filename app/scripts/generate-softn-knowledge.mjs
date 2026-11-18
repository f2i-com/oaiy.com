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
 *   web-starters.generated.ts
 *     — Studio's no-build Vanilla JS, React and Vue projects, with the local
 *       framework libraries and licenses shipped in the website examples;
 *   examples.generated.ts
 *     — a few complete apps from softn.com's catalogue (apps/softn-api/data/
 *       apps/<slug>/v1.softn), kept as the .softn bytes, for the agent to
 *       read or copy into the project.
 *
 * The output is committed (OAIY works offline); run this after a
 * SoftN update:   SOFTN_REPO=../softn.com node scripts/generate-softn-knowledge.mjs
 */
import { existsSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { registerHooks } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(process.env.SOFTN_REPO || join(here, '..', '..', 'softn.com'));
const out = join(here, '..', 'src', 'softn', 'knowledge');
mkdirSync(out, { recursive: true });

let commit = 'unknown';
try {
  commit = execFileSync('git', ['-C', repo, 'rev-parse', 'HEAD'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
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

const editor = join(repo, 'packages/@softn/editor-shared/src');
const builderUrl = pathToFileURL(join(editor, 'starters/buildStarter.ts')).href;
const { WEB_STARTERS } = await import(pathToFileURL(join(editor, 'webStarters.ts')).href);
const archiveNames = { vanilla: 'TrailPack', react: 'RatioBar', vue: 'Swatchbook' };
const archives = new Map(WEB_STARTERS.map(({ id }) => {
  const file = join(repo, 'apps/softn-site/public/examples', archiveNames[id] + '.softn');
  return [id, unzipSync(readFileSync(file))];
}));

// Studio's builder imports raw templates and virtual framework modules through
// Vite. Supply the same inputs when running that builder directly under Node.
const starterModules = registerHooks({
  resolve(id, context, nextResolve) {
    if (context.parentURL !== builderUrl) return nextResolve(id, context);
    let text;
    if (id.startsWith('./') && id.endsWith('?raw')) {
      text = readFileSync(new URL(id.slice(0, -4), builderUrl), 'utf8');
    } else if (id.startsWith('virtual:softn-starter/')) {
      const framework = id.slice('virtual:softn-starter/'.length);
      const library = archives.get(framework)?.[`__softn_generated__/vendor/${framework}.js`];
      if (!library) throw new Error(`The ${framework} example is missing its framework library`);
      text = strFromU8(library);
    } else return nextResolve(id, context);
    const module = Buffer.from('export default ' + JSON.stringify(text)).toString('base64');
    return { url: 'data:text/javascript;base64,' + module, shortCircuit: true };
  },
});
const starters = [];
try {
  const { buildStarter } = await import(builderUrl);
  for (const { id, label, name, description } of WEB_STARTERS) {
    const slug = id === 'vanilla' ? 'vanilla-js' : id;
    const files = await buildStarter(id, name, description, 'system');
    const entries = Object.fromEntries(files.map(({ path, content }) => [path, strToU8(content)]));
    for (const notice of ['LICENSE.txt', 'NOTICE.txt']) {
      const bytes = archives.get(id)[notice];
      if (!bytes) throw new Error(`The ${id} example is missing ${notice}`);
      entries[notice] = bytes;
    }
    const archive = zipSync(entries, { level: 9, mtime: new Date(2000, 0, 1) });
    starters.push({ slug, name, description, tags: [label, 'web app', 'starter'], files: Object.keys(entries).sort(), softn: Buffer.from(archive).toString('base64') });
    console.log(`packed web starter ${slug} (${archive.byteLength} bytes)`);
  }
} finally {
  starterModules.deregister();
}
writeFileSync(join(out, 'web-starters.generated.ts'), `${header('packages/@softn/editor-shared/src/starters and apps/softn-site/public/examples')}
import type { SoftnExample } from './examples.generated';
export const WEB_STARTER_EXAMPLES: SoftnExample[] = ${JSON.stringify(starters, null, 1)};
`);

// Small, varied, complete apps: CRUD with storage, a game with state and
// sound, components and charts across pages, 3D, WebGPU, device permissions.
const EXAMPLES = ['notes', 'twenty48', 'showcase', 'three-demo', 'gpu-demo', 'device-kit'];
// The catalogue is the site's own data and is not in every checkout (softn.com's tags since v0.0.19 have none of
// its apps): the examples committed here are then kept as they are.
if (!EXAMPLES.every((slug) => existsSync(join(repo, 'apps/softn-api/data/apps', slug, 'v1.softn')))) {
  console.log(`the catalogue's apps are not in ${repo}: examples.generated.ts is kept as it is`);
  console.log(`wrote ${out}`);
  process.exit(0);
}
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
