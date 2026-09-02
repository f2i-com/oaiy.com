/**
 * Bundle one test file with the CLI's own build options and run it.
 *
 *     node test/run-test.mjs test/foo.test.ts dist/_foo.mjs [--packages-external]
 *
 * These tests used to be bundled with bare `esbuild` CLI invocations in
 * package.json. That worked only when ui/node_modules happened to exist next
 * door: the engine sources live under ../ui, esbuild walks up from the
 * importing file to find their third-party imports (acorn, uuid), and on a
 * machine that installs the CLI alone — CI, for one — that walk finds nothing.
 * `commonBuildOptions` in esbuild.mjs carries the fallback to the CLI's own
 * node_modules, so the tests now share it instead of restating half of it.
 *
 * `--packages-external` keeps every third-party package as a runtime import,
 * which the SSRF-guard test needs so it exercises the real undici.
 */
import esbuild from 'esbuild';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { commonBuildOptions } from '../esbuild.mjs';

const cli = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [entry, outfile, ...flags] = process.argv.slice(2);
if (!entry || !outfile) {
  console.error('usage: node test/run-test.mjs <entry.ts> <outfile.mjs> [--packages-external]');
  process.exit(2);
}

const options = {
  ...commonBuildOptions,
  entryPoints: [path.resolve(cli, entry)],
  outfile: path.resolve(cli, outfile),
  logLevel: 'warning',
};
if (flags.includes('--packages-external')) {
  options.packages = 'external';
  delete options.external; // implied by packages: 'external'
}

await esbuild.build(options);
const run = spawnSync(process.execPath, [path.resolve(cli, outfile)], { stdio: 'inherit', cwd: cli });
process.exit(run.status ?? 1);
