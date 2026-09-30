#!/usr/bin/env node
/**
 * The transfer contract folder (`docs/contracts/transfer/`, `transfer_v1`) is the Aokie repository's own, copied byte for byte:
 * `transfer-v1.md`, the `transfer-v1.*.fixture.json` files and `SHA256SUMS`. This checks that it still is.
 *
 *   1. Every fixture (`*.json`) in the folder parses and is listed in `SHA256SUMS` with a current digest, and `SHA256SUMS`
 *      lists nothing that is not there. The digest is SHA-256 of the bytes with CRLF folded to LF (as Aokie's own check does),
 *      so a checkout that adds carriage returns and the committed blob agree.
 *   2. The same for `oaiy-only/`: what only OAIY has (its own `SHA256SUMS`, which `--write-oaiy-sums` rewrites).
 *   3. When `AOKIE_TRANSFER_CONTRACTS` names the Aokie repository's copy of the folder, every file in it (the fixtures, the prose
 *      contract and `SHA256SUMS`) must be here and the same, and this folder must hold no file that is not there. Unset, the
 *      comparison does not run and says so: the Aokie checkout is not needed to build or test this repository. `oaiy-only/` is never
 *      compared.
 *
 * Usage:  node scripts/check-transfer-contract.mjs
 *         node scripts/check-transfer-contract.mjs --write-oaiy-sums    (rewrite oaiy-only/SHA256SUMS; never the shared one)
 * Env:    AOKIE_TRANSFER_CONTRACTS  path to the Aokie repository's docs/contracts/transfer folder
 * Exit:   0 when everything checked agrees, 1 otherwise.
 *
 * To take a new version of the contract, copy the files of Aokie's committed folder over this one (`git show <commit>:<path>` for
 * each), run this with the variable set, and run the Rust tests (`ring::contract`, `ring::phrases`, the call tests).
 */

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

/** Everything: the folder's sums, `oaiy-only/`'s, and, when `aokieDir` is given, the comparison. */
export function checkTransfer({ dir = DEFAULT_DIR, aokieDir = process.env.AOKIE_TRANSFER_CONTRACTS } = {}) {
  const shared = checkSums(dir);
  const own = checkSums(path.join(dir, OAIY_ONLY), `${OAIY_ONLY}/`);
  const problems = [...shared.problems, ...own.problems];
  let compared = null;
  if (aokieDir) {
    const result = compareWithAokie(dir, aokieDir);
    compared = result.compared;
    problems.push(...result.problems);
  }
  return { problems, files: shared.files, ownFiles: own.files, compared };
}

/** Rewrite `oaiy-only/SHA256SUMS` from what is there. The shared `SHA256SUMS` is Aokie's and is never written here. */
export function writeOaiySums(dir = DEFAULT_DIR) {
  const own = path.join(dir, OAIY_ONLY);
  const names = filesIn(own).filter((name) => name.endsWith('.json'));
  const lines = names.map((name) => `${digest(readFileSync(path.join(own, name)))}  ${name}`);
  writeFileSync(path.join(own, SUMS), `${lines.join('\n')}\n`);
  return names.length;
}

function main() {
  if (process.argv.includes('--write-oaiy-sums')) {
    console.log(`check-transfer-contract: wrote ${OAIY_ONLY}/${SUMS} (${writeOaiySums()} files)`);
    return 0;
  }
  const result = checkTransfer();
  if (result.problems.length > 0) {
    console.error(`check-transfer-contract: FAIL, ${result.problems.length} problem(s):`);
    for (const problem of result.problems) console.error(`  ${problem}`);
    return 1;
  }
  console.log(`check-transfer-contract: OK, ${result.files} fixtures match ${SUMS}, ${result.ownFiles} in ${OAIY_ONLY}/`);
  if (result.compared === null) {
    console.log("check-transfer-contract: NOTE, Aokie's copy was not compared; set AOKIE_TRANSFER_CONTRACTS to its docs/contracts/transfer folder.");
  } else {
    console.log(`check-transfer-contract: OK, ${result.compared} files identical with Aokie's copy`);
  }
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exit(main());
}
