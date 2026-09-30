#!/usr/bin/env node
/**
 * The transfer contract folder (`docs/contracts/transfer/`, `transfer_v1`) is the Aokie repository's own, copied byte for byte:
 * `transfer-v1.md`, the `transfer-v1.*.fixture.json` files and `SHA256SUMS`. This checks that it still is.
 *
 *   1. Every fixture (`*.json`) in the folder parses and is listed in `SHA256SUMS` with a current digest, and `SHA256SUMS`
 *      lists nothing that is not there. The digest is SHA-256 of the bytes with CRLF folded to LF (as Aokie's own check does),
 *      so a checkout that adds carriage returns and the committed blob agree.
 *   2. The same for `oaiy-only/`: what only OAIY has (its own `SHA256SUMS`, which `--write-oaiy-sums` rewrites).
 *   3. `oaiy-only/SYNCED_FROM.json` is the lock: the Aokie commit the folder was copied from and the digest of every file in the
 *      folder as copied. This is the one place the commit is written (prose does not repeat it), and it is read here. Every file
 *      of the folder must match it, so an edit made here without a sync, or a sync without rewriting the lock, is a problem.
 *   4. When `AOKIE_TRANSFER_CONTRACTS` names the Aokie repository's copy of the folder, every file in it (the fixtures, the prose
 *      contract and `SHA256SUMS`) must be here and the same, and this folder must hold no file that is not there; and when that
 *      folder is in a git checkout, the lock's commit must be in it and its `docs/contracts/transfer` at that commit must be the
 *      files the lock records. `oaiy-only/` is never compared.
 *
 *      Without the variable none of that runs, and it says so loudly (SKIPPED, NOT VERIFIED AGAINST AOKIE, and a `::warning::`
 *      annotation under GitHub Actions): the Aokie checkout is not needed to build or test this repository, but a run without it
 *      has not shown that the folder is Aokie's, only that nothing here changed since the lock was written. `--require-aokie`
 *      makes that a failure, for the check that comes before a merge.
 *
 * Usage:  node scripts/check-transfer-contract.mjs [--require-aokie] [--dir <a copy of the folder>]
 *         node scripts/check-transfer-contract.mjs --write-lock <aokie commit sha>    (after copying Aokie's files over the folder:
 *                                                  rewrite the lock from what is there, and oaiy-only/SHA256SUMS)
 *         node scripts/check-transfer-contract.mjs --write-oaiy-sums    (rewrite oaiy-only/SHA256SUMS; never the shared one)
 * Env:    AOKIE_TRANSFER_CONTRACTS  path to the Aokie repository's docs/contracts/transfer folder
 * Exit:   0 when everything checked agrees, 1 otherwise.
 *
 * To take a new version of the contract, copy the files of Aokie's committed folder over this one (`git show <commit>:<path>` for
 * each), run `--write-lock <commit>`, run this with the variable set, and run the Rust tests (`ring::contract`, `ring::phrases`,
 * the call tests).
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_DIR = path.resolve(here, '..', 'docs', 'contracts', 'transfer');
const SUMS = 'SHA256SUMS';
const OAIY_ONLY = 'oaiy-only';

const folded = (buffer) => Buffer.from(buffer.toString('binary').replaceAll('\r\n', '\n'), 'binary');

/** SHA-256 of `buffer` with CRLF folded to LF, as lower-case hex. */
export const digest = (buffer) => createHash('sha256').update(folded(buffer)).digest('hex');

const filesIn = (dir) => readdirSync(dir).filter((name) => statSync(path.join(dir, name)).isFile()).sort();

/** The digests `dir/SHA256SUMS` lists (`{ listed: Map(name -> digest), problems }`). */
function readSums(dir, label) {
  const problems = [];
  const listed = new Map();
  const sums = path.join(dir, SUMS);
  if (!existsSync(sums)) {
    return { listed, problems: [`${label}${SUMS} is missing`] };
  }
  for (const line of readFileSync(sums, 'utf8').split(/\r?\n/).filter((l) => l.trim())) {
    const match = /^([0-9a-f]{64})  (.+)$/.exec(line);
    if (!match) problems.push(`${label}${SUMS}: malformed line: ${line}`);
    else if (listed.has(match[2])) problems.push(`${label}${SUMS}: ${match[2]} is listed twice`);
    else listed.set(match[2], match[1]);
  }
  return { listed, problems };
}

/** The fixtures of `dir` (its `.json` files) against its `SHA256SUMS`: parse, listed, current, and nothing listed that is gone. */
export function checkSums(dir, label = '') {
  if (!existsSync(dir)) return { problems: [`${label || 'the'} folder is missing: ${dir}`], files: 0 };
  const problems = [];
  const fixtures = filesIn(dir).filter((name) => name.endsWith('.json'));
  for (const name of fixtures) {
    try {
      JSON.parse(readFileSync(path.join(dir, name), 'utf8'));
    } catch (err) {
      problems.push(`${label}${name} is not valid JSON: ${err.message}`);
    }
  }
  const { listed, problems: sumProblems } = readSums(dir, label);
  problems.push(...sumProblems);
  for (const name of fixtures) {
    const actual = digest(readFileSync(path.join(dir, name)));
    if (!listed.has(name)) problems.push(`${label}${name} is not listed in ${SUMS}`);
    else if (listed.get(name) !== actual) problems.push(`${label}${name} changed and ${SUMS} was not updated\n    listed ${listed.get(name)}\n    actual ${actual}`);
  }
  for (const name of listed.keys()) {
    if (!fixtures.includes(name)) problems.push(`${label}${SUMS} lists ${name}, which is not in the folder`);
  }
  return { problems, files: fixtures.length };
}

/** Every file of Aokie's copy (`aokieDir`) is in `dir` and the same, and `dir` has no file of its own at the top level. */
export function compareWithAokie(dir, aokieDir) {
  if (!existsSync(aokieDir)) return { problems: [`AOKIE_TRANSFER_CONTRACTS does not exist: ${aokieDir}`], compared: 0 };
  const problems = [];
  const theirs = filesIn(aokieDir);
  const ours = new Set(filesIn(dir));
  let compared = 0;
  for (const name of theirs) {
    if (!ours.has(name)) {
      problems.push(`${name} is in Aokie's folder and missing here`);
      continue;
    }
    compared++;
    const a = digest(readFileSync(path.join(aokieDir, name)));
    const b = digest(readFileSync(path.join(dir, name)));
    if (a !== b) problems.push(`${name} differs from Aokie's\n    aokie ${a}\n    oaiy  ${b}`);
  }
  for (const name of ours) {
    if (!theirs.includes(name)) problems.push(`${name} is here and not in Aokie's folder (what only OAIY has belongs in ${OAIY_ONLY}/)`);
  }
  return { problems, compared };
}

const LOCK = 'SYNCED_FROM.json';
const LOCK_LABEL = `${OAIY_ONLY}/${LOCK}`;
const SHA_1 = /^[0-9a-f]{40}$/;
const SHA_256 = /^[0-9a-f]{64}$/;
/** Where the shared folder is in Aokie's repository. */
export const AOKIE_FOLDER = 'docs/contracts/transfer';

/** The digest of every file at the top of `dir`, by name: what the lock records of a copy. */
function digestsOf(dir) {
  return Object.fromEntries(filesIn(dir).map((name) => [name, digest(readFileSync(path.join(dir, name)))]));
}

/** The lock (`{ lock, problems }`): the commit, the folder in Aokie's repository, and the digest of each file as copied. */
export function readLock(dir = DEFAULT_DIR) {
  const file = path.join(dir, OAIY_ONLY, LOCK);
  if (!existsSync(file)) return { lock: null, problems: [`${LOCK_LABEL} is missing: it says which Aokie commit the folder was copied from`] };
  let lock;
  try {
    lock = JSON.parse(readFileSync(file, 'utf8'));
  } catch (err) {
    return { lock: null, problems: [`${LOCK_LABEL} is not valid JSON: ${err.message}`] };
  }
  const problems = [];
  if (typeof lock?.commit !== 'string' || !SHA_1.test(lock.commit)) problems.push(`${LOCK_LABEL}: "commit" is not a full 40 digit commit hash`);
  if (lock?.folder !== AOKIE_FOLDER) problems.push(`${LOCK_LABEL}: "folder" is not ${AOKIE_FOLDER}`);
  const files = lock?.files;
  if (!files || typeof files !== 'object' || Array.isArray(files) || Object.keys(files).length === 0) {
    problems.push(`${LOCK_LABEL}: "files" lists no file`);
  } else {
    for (const [name, sum] of Object.entries(files)) {
      if (typeof sum !== 'string' || !SHA_256.test(sum)) problems.push(`${LOCK_LABEL}: the digest of ${name} is not a SHA-256`);
    }
  }
  return { lock: problems.length === 0 ? lock : null, problems };
}

/** Every file of the folder is what the lock recorded, and the lock records no file that is gone. */
export function checkLock(dir = DEFAULT_DIR) {
  const { lock, problems } = readLock(dir);
  if (!lock) return { problems, lock: null };
  const actual = digestsOf(dir);
  const short = lock.commit.slice(0, 8);
  for (const [name, sum] of Object.entries(actual)) {
    if (!(name in lock.files)) problems.push(`${name} is in the folder and not in ${LOCK_LABEL}: a file added here, or a sync whose lock was not rewritten`);
    else if (lock.files[name] !== sum) problems.push(`${name} is not the file copied from Aokie ${short} (${LOCK_LABEL}): it was edited here, or a sync did not rewrite the lock\n    locked ${lock.files[name]}\n    actual ${sum}`);
  }
  for (const name of Object.keys(lock.files)) {
    if (!(name in actual)) problems.push(`${LOCK_LABEL} records ${name}, which is not in the folder`);
  }
  return { problems, lock };
}

const git = (cwd, args) => spawnSync('git', args, { cwd, encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 });

/**
 * The lock's commit against an Aokie checkout: `aokieDir` (a folder in it) is inside a git repository that has the commit, and the
 * files directly in `docs/contracts/transfer` at that commit are the files the lock records. `status` is `verified`, `failed`, or
 * `skipped` (with the reason in `note`) when `aokieDir` is not in a git checkout, so the commit cannot be looked up there.
 */
export function verifyCommit(lock, aokieDir) {
  const top = git(aokieDir, ['rev-parse', '--show-toplevel']);
  if (top.error || top.status !== 0) {
    return { status: 'skipped', note: `${aokieDir} is not inside a git checkout, so the commit ${lock.commit} was not looked up`, problems: [] };
  }
  const root = top.stdout.toString().trim();
  const exists = git(root, ['cat-file', '-e', `${lock.commit}^{commit}`]);
  if (exists.error || exists.status !== 0) {
    return { status: 'failed', note: '', problems: [`the commit ${lock.commit} named in ${LOCK_LABEL} is not in the Aokie checkout at ${root}`] };
  }
  const tree = git(root, ['ls-tree', '-z', lock.commit, `${lock.folder}/`]);
  if (tree.error || tree.status !== 0) {
    return { status: 'failed', note: '', problems: [`${lock.folder} could not be listed at ${lock.commit} in ${root}`] };
  }
  const problems = [];
  const atCommit = {};
  for (const entry of tree.stdout.toString('latin1').split('\0').filter(Boolean)) {
    const match = /^(\d+) (\w+) ([0-9a-f]{40})\t(.+)$/.exec(entry);
    if (!match || match[2] !== 'blob') continue;
    const blob = git(root, ['cat-file', 'blob', match[3]]);
    if (blob.error || blob.status !== 0) {
      problems.push(`${match[4]} could not be read at ${lock.commit}`);
      continue;
    }
    atCommit[path.posix.basename(match[4])] = digest(blob.stdout);
  }
  const short = lock.commit.slice(0, 8);
  for (const [name, sum] of Object.entries(atCommit)) {
    if (!(name in lock.files)) problems.push(`${name} is in Aokie's ${lock.folder} at ${short} and not in ${LOCK_LABEL}`);
    else if (lock.files[name] !== sum) problems.push(`${name} at Aokie ${short} is not what ${LOCK_LABEL} records\n    aokie  ${sum}\n    locked ${lock.files[name]}`);
  }
  for (const name of Object.keys(lock.files)) {
    if (!(name in atCommit)) problems.push(`${LOCK_LABEL} records ${name}, which is not in Aokie's ${lock.folder} at ${short}`);
  }
  return { status: problems.length === 0 ? 'verified' : 'failed', note: '', problems };
}

/**
 * Everything: the folder's sums, `oaiy-only/`'s, the lock, and, when `aokieDir` is given, the comparison with Aokie's copy and the
 * commit. `verified` says what was and was not checked against Aokie: `commit` is `verified`, `failed`, `skipped` or `null` (no
 * `aokieDir`), and `compared` the number of files compared (null: none).
 */
export function checkTransfer({ dir = DEFAULT_DIR, aokieDir = process.env.AOKIE_TRANSFER_CONTRACTS } = {}) {
  const shared = checkSums(dir);
  const own = checkSums(path.join(dir, OAIY_ONLY), `${OAIY_ONLY}/`);
  const locked = checkLock(dir);
  const problems = [...shared.problems, ...own.problems, ...locked.problems];
  let compared = null;
  let commit = null;
  let note = '';
  if (aokieDir) {
    const result = compareWithAokie(dir, aokieDir);
    compared = result.compared;
    problems.push(...result.problems);
    if (locked.lock && existsSync(aokieDir)) {
      const looked = verifyCommit(locked.lock, aokieDir);
      commit = looked.status;
      note = looked.note;
      problems.push(...looked.problems);
    }
  }
  return { problems, files: shared.files, ownFiles: own.files, compared, commit, note, lockedCommit: locked.lock?.commit ?? null };
}

/**
 * Write the lock for the folder as it is now, as copied from Aokie's `commit`, and rewrite `oaiy-only/SHA256SUMS` (which lists the
 * lock). Run after the files of a new Aokie commit have been copied over the folder.
 */
export function writeLock(commit, dir = DEFAULT_DIR) {
  if (!SHA_1.test(commit)) throw new Error(`not a full 40 digit commit hash: ${commit}`);
  const lock = {
    what: `The Aokie commit this folder was copied from, and the digest of each file as copied (SHA-256, carriage returns folded away). Read by scripts/check-transfer-contract.mjs; rewrite it with --write-lock <commit> after a sync, never by hand.`,
    repository: 'aokie',
    folder: AOKIE_FOLDER,
    commit,
    files: digestsOf(dir),
  };
  writeFileSync(path.join(dir, OAIY_ONLY, LOCK), `${JSON.stringify(lock, null, 2)}\n`);
  writeOaiySums(dir);
  return lock;
}

/** Rewrite `oaiy-only/SHA256SUMS` from what is there. The shared `SHA256SUMS` is Aokie's and is never written here. */
export function writeOaiySums(dir = DEFAULT_DIR) {
  const own = path.join(dir, OAIY_ONLY);
  const names = filesIn(own).filter((name) => name.endsWith('.json'));
  const lines = names.map((name) => `${digest(readFileSync(path.join(own, name)))}  ${name}`);
  writeFileSync(path.join(own, SUMS), `${lines.join('\n')}\n`);
  return names.length;
}

/** A line that stands out in a log: on GitHub Actions also an annotation on the run. */
function loud(message) {
  console.log(`check-transfer-contract: SKIPPED, ${message}`);
  if (process.env.GITHUB_ACTIONS) console.log(`::warning title=transfer contract not verified against Aokie::${message}`);
}

function main() {
  const args = process.argv.slice(2);
  // `--dir <folder>`: a copy of the contract folder instead of this repository's (the tests run the program on scratch copies).
  const named = args.indexOf('--dir');
  const dir = named >= 0 && args[named + 1] ? path.resolve(args[named + 1]) : DEFAULT_DIR;
  if (args.includes('--write-oaiy-sums')) {
    console.log(`check-transfer-contract: wrote ${OAIY_ONLY}/${SUMS} (${writeOaiySums(dir)} files)`);
    return 0;
  }
  const at = args.indexOf('--write-lock');
  if (at >= 0) {
    try {
      const lock = writeLock(args[at + 1] ?? '', dir);
      console.log(`check-transfer-contract: wrote ${LOCK_LABEL} (Aokie ${lock.commit}, ${Object.keys(lock.files).length} files) and ${OAIY_ONLY}/${SUMS}`);
      return 0;
    } catch (err) {
      console.error(`check-transfer-contract: ${err.message}`);
      return 1;
    }
  }
  const result = checkTransfer({ dir });
  if (result.problems.length > 0) {
    console.error(`check-transfer-contract: FAIL, ${result.problems.length} problem(s):`);
    for (const problem of result.problems) console.error(`  ${problem}`);
    return 1;
  }
  console.log(`check-transfer-contract: OK, ${result.files} fixtures match ${SUMS}, ${result.ownFiles} in ${OAIY_ONLY}/, every file is the one locked from Aokie ${result.lockedCommit}`);
  if (result.compared === null) {
    loud(`NOT VERIFIED AGAINST AOKIE: AOKIE_TRANSFER_CONTRACTS is not set, so neither Aokie's copy of the folder nor the commit ${result.lockedCommit} named in ${LOCK_LABEL} was looked at. Nothing here has changed since the lock was written, and that is all this run shows. Set it to Aokie's docs/contracts/transfer folder to check.`);
    return args.includes('--require-aokie') ? 1 : 0;
  }
  console.log(`check-transfer-contract: OK, ${result.compared} files identical with Aokie's copy`);
  if (result.commit === 'verified') {
    console.log(`check-transfer-contract: OK, the folder at Aokie commit ${result.lockedCommit} is the one locked`);
  } else {
    loud(`THE COMMIT WAS NOT VERIFIED: ${result.note}. The files were compared with that folder, but nothing shows it is the folder at ${result.lockedCommit}.`);
    if (args.includes('--require-aokie')) return 1;
  }
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exit(main());
}
