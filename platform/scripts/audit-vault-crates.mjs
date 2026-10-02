#!/usr/bin/env node
// Audits the dependencies of the vault crates, oaiy-crypto and oaiy-keystore, and of the relay client core that builds on them (oaiy-relay-core), and of nothing else in the workspace.
//
//   node platform/scripts/audit-vault-crates.mjs
//
// `cargo audit` reads a Cargo.lock whole, and this repository's lock holds the desktop app, the media worker and everything else, so an advisory in any of them
// would fail a lane that is about the two crates that hold the vault's keys, and an advisory in one of those two would drown in the rest. So the two crates are
// copied into a workspace of their own with the repository's lock, `cargo metadata` rewrites that lock to what the two crates need (and keeps the versions the
// repository locked: nothing is resolved afresh), and `cargo audit` runs over it with warnings denied: a yanked or unmaintained dependency of these two crates is
// a failure too. The number of packages audited is printed, so that a scope that silently shrank to nothing is visible.
//
// Needs cargo and cargo-audit on the path (the CI lane installs a pinned cargo-audit); CARGO_AUDIT_ARGS adds arguments to `cargo audit` (for example `--no-fetch`
// on a machine that is offline).
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const CRATES = ['oaiy-crypto', 'oaiy-keystore', 'oaiy-relay-core'];
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

function run(args, options) {
  const result = spawnSync('cargo', args, { stdio: 'inherit', ...options });
  if (result.error) throw new Error(`cargo ${args.join(' ')}: ${result.error.message}`);
  if (result.status !== 0) throw new Error(`cargo ${args.join(' ')} exited with ${result.status}`);
}

export function packagesIn(lockText) {
  return [...lockText.matchAll(/^name = "([^"]+)"\s*\nversion = "([^"]+)"/gm)].map((m) => `${m[1]} ${m[2]}`);
}

function main() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-vault-audit-'));
  try {
    for (const crate of CRATES) {
      fs.cpSync(path.join(root, 'crates', crate), path.join(tmp, 'crates', crate), {
        recursive: true,
        filter: (source) => path.basename(source) !== 'target',
      });
    }
    fs.copyFileSync(path.join(root, 'Cargo.lock'), path.join(tmp, 'Cargo.lock'));
    fs.writeFileSync(path.join(tmp, 'Cargo.toml'), `[workspace]\nresolver = "2"\nmembers = [${CRATES.map((c) => JSON.stringify(`crates/${c}`)).join(', ')}]\n`);
    // rewrites Cargo.lock to the packages these two crates need, at the versions the repository locked
    run(['metadata', '--format-version', '1', '--quiet'], { cwd: tmp, stdio: ['ignore', 'ignore', 'inherit'] });
    const packages = packagesIn(fs.readFileSync(path.join(tmp, 'Cargo.lock'), 'utf8'));
    for (const crate of CRATES) {
      if (!packages.some((p) => p.startsWith(`${crate} `))) throw new Error(`${crate} is not in the lock that is being audited`);
    }
    if (packages.length < 20) throw new Error(`only ${packages.length} packages would be audited: the scope is wrong`);
    console.log(`auditing ${packages.length} packages: the dependencies of ${CRATES.join(' and ')}`);
    const extra = (process.env.CARGO_AUDIT_ARGS ?? '').split(/\s+/).filter(Boolean);
    run(['audit', '--deny', 'warnings', '--file', 'Cargo.lock', ...extra], { cwd: tmp });
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(`::error::${error.message}`);
    process.exit(1);
  }
}
