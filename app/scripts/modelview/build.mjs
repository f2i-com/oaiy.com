/**
 * Builds the live preview's 3D model viewer: scripts/modelview/viewer.js with
 * three.js, bundled into one classic script, public/modelview/model-viewer.js.
 *
 * One classic script, because the viewer runs in an opaque-origin frame: a
 * module script (or an import) from bot.computer's own server would be a
 * cross-origin request needing CORS headers, which the static build's host and
 * the desktop app do not send for /modelview/; a classic <script src> needs none.
 * vite.config.ts runs this on every dev server start and build; it rebuilds only
 * when the viewer or three.js changed.
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const SOURCE = join(here, 'viewer.js');
const THREE = join(root, 'node_modules', 'three', 'package.json');

/** Writes `out` (public/modelview/model-viewer.js) unless it is newer than what it is built from. */
export async function buildViewer(out = join(root, 'public', 'modelview', 'model-viewer.js')) {
  const stamp = `/* three ${JSON.parse(readFileSync(THREE, 'utf8')).version} */\n`;
  if (existsSync(out) && statSync(out).mtimeMs >= statSync(SOURCE).mtimeMs && readFileSync(out, 'utf8').startsWith(stamp)) return false;
  const { build } = await import('rolldown');
  const result = await build({
    input: SOURCE,
    cwd: root,
    logLevel: 'warn',
    write: false,
    output: { format: 'iife', minify: true, codeSplitting: false },
  });
  const chunk = result.output.find((o) => o.type === 'chunk');
  if (!chunk) throw new Error('the 3D model viewer did not build');
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, stamp + chunk.code);
  return true;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  buildViewer().then((built) => console.log(built ? 'built public/modelview/model-viewer.js' : 'public/modelview/model-viewer.js is up to date'));
}
