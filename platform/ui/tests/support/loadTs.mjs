/**
 * Load a TypeScript module of the editor into a Node test.
 *
 * The suites bundle what they test with esbuild (as in-oaiy.mjs does): no compile step of
 * its own, and the module under test is the source file, not a copy. Packages stay external
 * (esbuild and vite are real dependencies of the plugin), so only the editor's own files are bundled.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';

export const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

let counter = 0;

/** Bundle `entry` (a path from platform/ui) and import it. */
export async function loadTs(entry) {
  // Beside the tests (not in the system's temp folder): the bundle imports esbuild and vite by name, and
  // has to be somewhere they resolve from. It is removed as soon as it is imported.
  const outfile = path.join(UI, 'tests', 'support', `.bundle-${process.pid}-${counter++}.mjs`);
  await esbuild.build({
    entryPoints: [path.join(UI, entry)],
    bundle: true,
    format: 'esm',
    platform: 'node',
    packages: 'external',
    outfile,
    logLevel: 'silent',
  });
  try {
    return await import(pathToFileURL(outfile).href);
  } finally {
    fs.rmSync(outfile, { force: true });
  }
}

/** A tiny check runner with the same output as the other suites. */
export function suite(title) {
  let pass = 0;
  const failures = [];
  return {
    async check(name, fn) {
      try {
        await fn();
        pass++;
        console.log(`  ok  ${name}`);
      } catch (e) {
        failures.push(`${name} — ${e.message}`);
        console.log(`  FAIL ${name} — ${e.stack || e.message}`);
      }
    },
    finish() {
      console.log(`\n${title}: ${pass} passed, ${failures.length} failed`);
      if (failures.length) process.exit(1);
    },
  };
}
