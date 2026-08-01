/**
 * Bundle + run the engine-level tests (those that import the real createEngine +
 * node-host), reusing the CLI's esbuild config so the @tauri-apps→node-host and
 * oaiy-core aliases resolve exactly as in the shipped CLI. The lightweight unit tests
 * (module-ctx / cross-module isolation) don't need this — plain esbuild bundles them.
 */
import esbuild from 'esbuild';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { genBundledModules, commonBuildOptions } from '../esbuild.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// The engine tests import src/generated/bundled-modules (codegen). Ensure it's fresh.
genBundledModules();

const entries = ['parallel-jobs.ts', 'connector-flow.ts', 'logic-block-scope.ts', 'output-value-refs.ts'];
let failed = false;
for (const entry of entries) {
  const out = path.join(__dirname, '..', 'dist', `_engine_${entry.replace(/\.ts$/, '')}.mjs`);
  await esbuild.build({
    entryPoints: [path.join(__dirname, entry)],
    outfile: out,
    ...commonBuildOptions,
  });
  const r = spawnSync(process.execPath, [out], { stdio: 'inherit' });
  if ((r.status ?? 1) !== 0) failed = true;
}

process.exit(failed ? 1 : 0);
