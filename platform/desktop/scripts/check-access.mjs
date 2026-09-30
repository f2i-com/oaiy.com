// Runs the access model's checks locally (CI is paused, so nothing else runs them).
//
//   node scripts/check-access.mjs              from platform/desktop
//   node scripts/check-access.mjs --skip-boot  the unit tests only (no server is started)
//   node scripts/check-access.mjs --keep       leave routes.json where the tests wrote it and say where
//
// What it does:
//   1. `cargo test` for the access model's unit tests: the route table, the source walk that fails
//      when a route has no row, the exported file, and everything else under `auth::`.
//   2. The boot test, which starts `oaiy-server` on a port the system picks (never 17972, 17872 or
//      any other fixed port) and sends an anonymous request to every row of the table.
//   3. Reads the `routes.json` the tests wrote and checks it against itself: no duplicate row, every
//      row well formed, every scope named by a row spelled like a scope.
//
// The cargo command can be replaced with OAIY_CARGO (a program and its leading arguments). On a
// machine whose C++ tools are not on the PATH, run it through the wrapper that loads them.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const crate = path.resolve(here, '..', 'src-tauri');
const skipBoot = process.argv.includes('--skip-boot');
const cargo = (process.env.OAIY_CARGO ?? 'cargo').split(/\s+/).filter(Boolean);

// The tests write routes.json here, so this script knows where to read it.
const out = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-access-check-'));

function run(label, args) {
  console.log(`\n== ${label}\n$ ${[...cargo, ...args].join(' ')}`);
  const result = spawnSync(cargo[0], [...cargo.slice(1), ...args], {
    cwd: crate,
    stdio: 'inherit',
    env: { ...process.env, OAIY_ACCESS_OUT: out },
    shell: process.platform === 'win32',
  });
  if (result.status !== 0) {
    console.error(`\ncheck-access: ${label} failed (${result.status ?? result.error})`);
    process.exit(result.status || 1);
  }
}

run('access model: unit tests', ['test', '--locked', '--no-default-features', '--lib', '--', 'auth::', '--skip', 'oauth::']);
if (!skipBoot) run('access model: every route behind the guard (boot test)', ['test', '--locked', '--no-default-features', '--test', 'access_boot']);

const file = path.join(out, 'routes.json');
const problems = [];
let doc;
try {
  doc = JSON.parse(fs.readFileSync(file, 'utf8'));
} catch (e) {
  console.error(`check-access: ${file} was not written or is not JSON: ${e.message}`);
  process.exit(1);
}
if (doc.schema !== 'oaiy-access-routes/1') problems.push(`unexpected schema ${doc.schema}`);
const seen = new Set();
const scopeRows = new Map();
const scopeName = /^[a-z]+\.[a-z]+$/;
for (const r of doc.routes ?? []) {
  const key = `${r.method} ${r.pattern}`;
  if (seen.has(key)) problems.push(`duplicate row ${key}`);
  seen.add(key);
  if (!['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'ANY'].includes(r.method)) problems.push(`${key}: unknown method`);
  if (typeof r.pattern !== 'string' || !r.pattern.startsWith('/')) problems.push(`${key}: pattern`);
  if (![1, 2].includes(r.since)) problems.push(`${key}: since ${r.since}`);
  if (r.class === 'scope') {
    if (!scopeName.test(r.scope ?? '')) problems.push(`${key}: scope ${r.scope}`);
    scopeRows.set(r.scope, (scopeRows.get(r.scope) ?? 0) + 1);
  } else if (!['public', 'any-credential', 'console', 'session', 'desk'].includes(r.class)) {
    problems.push(`${key}: class ${r.class}`);
  }
}
if (problems.length) {
  console.error('\ncheck-access: routes.json has problems:\n  ' + problems.join('\n  '));
  process.exit(1);
}
const existing = doc.routes.filter((r) => r.since === 1).length;
console.log(`\ncheck-access: ok. ${doc.routes.length} rows (${existing} for routes that existed before the access model, ${doc.routes.length - existing} added or reserved), ${scopeRows.size} scopes named by a row.`);
if (process.argv.includes('--keep')) console.log(`routes.json: ${file}`);
else fs.rmSync(out, { recursive: true, force: true });
