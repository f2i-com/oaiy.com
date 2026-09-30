// Runs the access model's checks locally (CI is paused, so nothing else runs them).
//
//   node scripts/check-access.mjs              from platform/desktop
//   node scripts/check-access.mjs --skip-boot  the unit tests only (no server is started)
//   node scripts/check-access.mjs --keep       leave routes.json where the tests wrote it and say where
//
// What it does:
//   1. `cargo test` for the access model's unit tests: the route table, the source walk that fails
//      when a route has no row, the exported file, and everything else under `auth::`; and again with the
//      `web` feature (the login of `oaiy-server`: its parts, the in-process tests of the whole login).
//   2. The boot test (also with the `web` feature, and `login_e2e`: the real server and its console), which starts `oaiy-server` on a port the system picks (never 17972, 17872 or
//      any other fixed port) and sends an anonymous request to every row of the table.
//   3. Reads the `routes.json` the tests wrote and checks it against itself, in JavaScript, so that a
//      mistake in the Rust tables that its own tests share cannot hide: no duplicate row, every row
//      well formed, every scope a row or a preset names is one of the 56, the counts of the design
//      (48 core scopes: 18 read, 18 act, 12 dangerous; 6 reserved; and `calls.settings` and `calls.manage`, the two act scopes this code adds: 50 core, 20 act), only `owner` and `cli-admin`
//      hold a dangerous scope, no relay tier does, and `control.project` never travels without
//      `control.read`.
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
// The web login of oaiy-server is behind the `web` feature: its unit tests, the in-process tests of the login over a real
// store, and (with the boot tests) the real program driven over HTTP with its console.
run('access model: the web login (unit and in-process tests)', ['test', '--locked', '--no-default-features', '--features', 'web', '--lib', '--', 'auth::', '--skip', 'oauth::']);
if (!skipBoot) {
  run('access model: every route behind the guard (boot test)', ['test', '--locked', '--no-default-features', '--test', 'access_boot']);
  run('access model: every route behind the guard, with the web login (boot test)', ['test', '--locked', '--no-default-features', '--features', 'web', '--test', 'access_boot']);
  run('web login: the real server and its console', ['test', '--locked', '--no-default-features', '--features', 'web', '--test', 'login_e2e']);
}

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
// The scopes, presets and relay tiers.
const scopeInfo = new Map((doc.scopes ?? []).map((s) => [s.name, s]));
const count = (group) => (doc.scopes ?? []).filter((s) => s.group === group).length;
if (scopeInfo.size !== 56) problems.push(`${scopeInfo.size} scopes, expected 56`);
if ([count('read'), count('act'), count('dangerous'), count('reserved')].join() !== '18,20,12,6') {
  problems.push(`scope groups ${[count('read'), count('act'), count('dangerous'), count('reserved')].join('/')}, expected 18/20/12/6`);
}
for (const name of scopeRows.keys()) if (!scopeInfo.has(name)) problems.push(`a row names ${name}, which is not a scope`);
for (const r of doc.routes ?? []) {
  if (r.class === 'scope' && r.dangerous !== Boolean(scopeInfo.get(r.scope)?.dangerous)) problems.push(`${r.method} ${r.pattern}: dangerous flag`);
}
const mayHoldDangerous = new Set(['owner', 'cli-admin']);
for (const [name, list] of Object.entries(doc.presets ?? {})) {
  for (const s of list) {
    if (!scopeInfo.has(s)) problems.push(`preset ${name} names ${s}, which is not a scope`);
    if (scopeInfo.get(s)?.dangerous && !mayHoldDangerous.has(name)) problems.push(`preset ${name} holds the dangerous scope ${s}`);
    if (scopeInfo.get(s)?.group === 'reserved' && !['owner', 'ceremony'].includes(name)) problems.push(`preset ${name} holds the reserved scope ${s}`);
  }
  if (list.includes('control.project') && !list.includes('control.read')) problems.push(`preset ${name} has control.project without control.read`);
}
for (const [name, list] of Object.entries(doc.relayTiers ?? {})) {
  for (const s of list) {
    if (!scopeInfo.has(s)) problems.push(`relay tier ${name} names ${s}, which is not a scope`);
    if (scopeInfo.get(s)?.dangerous) problems.push(`relay tier ${name} holds the dangerous scope ${s}`);
  }
  if (list.includes('control.project') && !list.includes('control.read')) problems.push(`relay tier ${name} has control.project without control.read`);
}
if ((doc.presets?.owner ?? []).length !== 56) problems.push('owner does not hold all 56 scopes');
// Every scope has a route, but `control.project` (the change level of POST /api/mcp) and `relay.manage`
// (the relay design's routes).
for (const name of scopeInfo.keys()) {
  if (!scopeRows.has(name) && !['control.project', 'relay.manage'].includes(name)) problems.push(`scope ${name} has no route`);
}
if (problems.length) {
  console.error('\ncheck-access: routes.json has problems:\n  ' + problems.join('\n  '));
  process.exit(1);
}
const existing = doc.routes.filter((r) => r.since === 1).length;
console.log(`\ncheck-access: ok. ${doc.routes.length} rows (${existing} for routes that existed before the access model, ${doc.routes.length - existing} added or reserved), ${scopeRows.size} scopes named by a row, ${Object.keys(doc.presets).length} presets, ${Object.keys(doc.relayTiers).length} relay tiers.`);
if (process.argv.includes('--keep')) console.log(`routes.json: ${file}`);
else fs.rmSync(out, { recursive: true, force: true });
