/**
 * Stage the OAIY CLI into src-tauri/resources/cli so Tauri ships it.
 *
 * The desktop bundles the headless CLI because that is what actually runs a
 * flow — without it a packaged app resolves nothing and every run fails
 * `runtime_unavailable`. The CLI's own `dist/` is a build output (gitignored),
 * so this copies it at build time rather than committing a duplicate that would
 * silently go stale the moment the CLI changed.
 *
 * Ships the esbuild bundle plus `undici`, which the bundle marks external and
 * imports eagerly. `playwright` and `sharp` are also external but are DYNAMIC
 * imports, so they stay optional — browser work has its own managed service.
 */
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const desktop = path.resolve(here, '..');
const cli = path.resolve(desktop, '..', 'cli');
const dest = path.join(desktop, 'src-tauri', 'resources', 'cli');

const bundle = path.join(cli, 'dist', 'oaiy.mjs');
const undici = path.join(cli, 'node_modules', 'undici');

function die(msg) {
  console.error(`sync-cli: ${msg}`);
  process.exit(1);
}

if (!fs.existsSync(bundle)) {
  // Try to produce it, so a fresh checkout builds without a manual step.
  if (!fs.existsSync(path.join(cli, 'node_modules'))) {
    die(`no CLI bundle at ${bundle} and cli/node_modules is missing — run "npm install" in cli/ first.`);
  }
  console.log('sync-cli: building the CLI bundle…');
  try {
    execFileSync('npm', ['run', 'build'], { cwd: cli, stdio: 'inherit', shell: true });
  } catch {
    die('the CLI build failed — fix it before building the desktop app.');
  }
}
if (!fs.existsSync(bundle)) die(`the CLI build produced no ${bundle}`);
if (!fs.existsSync(undici)) die(`undici is missing from cli/node_modules — run "npm install" in cli/.`);

fs.rmSync(dest, { recursive: true, force: true });
fs.mkdirSync(path.join(dest, 'node_modules'), { recursive: true });
fs.copyFileSync(bundle, path.join(dest, 'oaiy.mjs'));
fs.cpSync(undici, path.join(dest, 'node_modules', 'undici'), { recursive: true });
// Trim what only matters to undici's own development.
for (const junk of ['docs', 'test', 'types']) {
  fs.rmSync(path.join(dest, 'node_modules', 'undici', junk), { recursive: true, force: true });
}

const size = (p) =>
  fs.statSync(p).isDirectory()
    ? fs.readdirSync(p).reduce((n, f) => n + size(path.join(p, f)), 0)
    : fs.statSync(p).size;
console.log(`sync-cli: staged ${(size(dest) / 1024 / 1024).toFixed(1)} MB into src-tauri/resources/cli`);
