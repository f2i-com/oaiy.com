/**
 * Load a TypeScript module of this repository into a Node test.
 *
 * The suites bundle what they test with esbuild (as platform/ui/tests/support/loadTs.mjs does): no compile step of their
 * own, and the module under test is the source file, not a copy. `@oaiy/shared/...` is the repository's `shared/` folder.
 * The bundle is imported from a data: URL, so nothing is written to disk; it has no imports of its own except Node's.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

/** `web/` */
export const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
/** The repository root. */
export const ROOT = path.resolve(WEB, '..');
export const SHARED = path.join(ROOT, 'shared');

const loaded = new Map();

/** Bundle `entry` (a path from the repository root, such as `shared/providers/endpoints.ts`) and import it. Cached per entry. */
export function loadTs(entry) {
  if (!loaded.has(entry)) loaded.set(entry, build(entry));
  return loaded.get(entry);
}

async function build(entry) {
  const result = await esbuild.build({
    entryPoints: [path.join(ROOT, entry)],
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node24',
    write: false,
    alias: { '@oaiy/shared': SHARED },
    logLevel: 'silent',
  });
  const text = result.outputFiles[0].text;
  return import(`data:text/javascript;base64,${Buffer.from(text).toString('base64')}`);
}
