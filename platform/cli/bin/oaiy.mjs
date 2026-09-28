#!/usr/bin/env node
// Thin launcher → the esbuild-bundled CLI. Keeping this tiny means `npm link`
// / a global install points at a stable path while the real code lives in
// dist/oaiy.mjs (rebuilt by `npm run build`).
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, '..', 'dist', 'oaiy.mjs');

if (!fs.existsSync(dist)) {
  process.stderr.write(
    'oaiy: build artifact not found at dist/oaiy.mjs — run `npm run build` in the cli/ package first.\n',
  );
  process.exit(2);
}

// Silence node:sqlite's experimental warning here, before the bundle (whose
// hoisted `import "node:sqlite"` would otherwise emit it) loads.
const __origEmitWarning = process.emitWarning.bind(process);
process.emitWarning = (warning, ...rest) => {
  const msg = typeof warning === 'string' ? warning : warning?.message;
  if (typeof msg === 'string' && /SQLite is an experimental feature/.test(msg)) return;
  return __origEmitWarning(warning, ...rest);
};

await import(pathToFileURL(dist).href);
