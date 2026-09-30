import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { checkSums, checkTransfer, compareWithAokie, DEFAULT_DIR, digest, writeOaiySums } from './check-transfer-contract.mjs';

const script = fileURLToPath(new URL('./check-transfer-contract.mjs', import.meta.url));

/** A scratch copy of the contract folder, removed when the test ends. */
function copyOfFolder(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'oaiy-transfer-contract-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const dir = path.join(root, 'transfer');
  cpSync(DEFAULT_DIR, dir, { recursive: true });
  return dir;
}

/** What Aokie's copy is: the top-level files of the folder (no `oaiy-only/`). */
function copyAsAokieHasIt(t) {
  const dir = copyOfFolder(t);
  const aokie = path.join(path.dirname(dir), 'aokie');
  cpSync(dir, aokie, { recursive: true, filter: (from) => !from.includes(`${path.sep}oaiy-only`) });
  return { dir, aokie };
}

const edit = (file, change) => writeFileSync(file, change(readFileSync(file, 'utf8')));
const fixtures = (dir) => readdirSync(dir).filter((n) => n.endsWith('.json'));

test('the folder in this repository agrees with its own sums', () => {
  const result = checkTransfer({ dir: DEFAULT_DIR, aokieDir: undefined });
  assert.deepEqual(result.problems, []);
  assert.equal(result.files, 8, 'the eight shared fixtures');
  assert.equal(result.ownFiles, 3, 'and the three that only OAIY has');
  assert.equal(result.compared, null, 'no comparison without Aokie\'s copy');
});

test('the shared folder holds the prose contract, the fixtures and the sums, and nothing else at the top', () => {
  const files = readdirSync(DEFAULT_DIR).filter((n) => statSync(path.join(DEFAULT_DIR, n)).isFile()).sort();
  assert.deepEqual(files, ['SHA256SUMS', 'transfer-v1.md', ...fixtures(DEFAULT_DIR)].sort());
  assert.ok(fixtures(DEFAULT_DIR).every((n) => /^transfer-v1\.[a-z-]+\.fixture\.json$/.test(n)));
});

test('a changed fixture, an unlisted one, a stale line and a malformed line are each a problem', (t) => {
  const dir = copyOfFolder(t);
  const first = fixtures(dir)[0];
  edit(path.join(dir, first), (text) => text.replace('transfer_v1', 'transfer_v2'));
  assert.match(checkSums(dir).problems.join('\n'), new RegExp(`${first} changed and SHA256SUMS was not updated`));

  const other = copyOfFolder(t);
  writeFileSync(path.join(other, 'transfer-v1.extra.fixture.json'), '{}\n');
  assert.match(checkSums(other).problems.join('\n'), /transfer-v1\.extra\.fixture\.json is not listed in SHA256SUMS/);

  const stale = copyOfFolder(t);
  edit(path.join(stale, 'SHA256SUMS'), (text) => `${text}${'0'.repeat(64)}  transfer-v1.gone.fixture.json\n`);
  assert.match(checkSums(stale).problems.join('\n'), /lists transfer-v1\.gone\.fixture\.json, which is not in the folder/);

  const malformed = copyOfFolder(t);
  edit(path.join(malformed, 'SHA256SUMS'), (text) => `${text}not a digest line\n`);
  assert.match(checkSums(malformed).problems.join('\n'), /malformed line: not a digest line/);

  const twice = copyOfFolder(t);
  edit(path.join(twice, 'SHA256SUMS'), (text) => `${text}${text.split('\n')[0]}\n`);
  assert.match(checkSums(twice).problems.join('\n'), /is listed twice/);

  const broken = copyOfFolder(t);
  writeFileSync(path.join(broken, 'transfer-v1.broken.fixture.json'), '{ not json');
  assert.match(checkSums(broken).problems.join('\n'), /transfer-v1\.broken\.fixture\.json is not valid JSON/);
});

test('a missing SHA256SUMS is a problem, and so is a missing folder', (t) => {
  const dir = copyOfFolder(t);
  rmSync(path.join(dir, 'SHA256SUMS'));
  assert.match(checkSums(dir).problems.join('\n'), /SHA256SUMS is missing/);
  assert.match(checkSums(path.join(dir, 'nowhere'), 'oaiy-only/').problems.join('\n'), /folder is missing/);
});

test('the oaiy-only sums are checked the same way', (t) => {
  const dir = copyOfFolder(t);
  edit(path.join(dir, 'oaiy-only', 'phrases-oaiy.json'), (text) => text.replace('"about"', '"About"'));
  const problems = checkTransfer({ dir, aokieDir: undefined }).problems;
  assert.ok(problems.some((p) => p.startsWith('oaiy-only/phrases-oaiy.json changed and SHA256SUMS was not updated')), problems.join('\n'));
  // Rewriting them makes them right, and never touches the shared ones.
  const shared = readFileSync(path.join(dir, 'SHA256SUMS'));
  assert.equal(writeOaiySums(dir), 3);
  assert.deepEqual(checkTransfer({ dir, aokieDir: undefined }).problems, []);
  assert.deepEqual(readFileSync(path.join(dir, 'SHA256SUMS')), shared);
});

test('carriage returns are folded away: a checkout that adds them still agrees', (t) => {
  const dir = copyOfFolder(t);
  for (const name of [...fixtures(dir), 'transfer-v1.md', 'SHA256SUMS']) {
    edit(path.join(dir, name), (text) => text.replaceAll('\n', '\r\n'));
  }
  assert.deepEqual(checkTransfer({ dir, aokieDir: undefined }).problems, []);
  assert.equal(digest(Buffer.from('a\r\nb\n')), digest(Buffer.from('a\nb\n')));
  assert.notEqual(digest(Buffer.from('a\nb')), digest(Buffer.from('a\nb\n')));
});

test('an identical copy of Aokie\'s folder agrees, oaiy-only is left out of the comparison', (t) => {
  const { dir, aokie } = copyAsAokieHasIt(t);
  const result = checkTransfer({ dir, aokieDir: aokie });
  assert.deepEqual(result.problems, []);
  assert.equal(result.compared, 10, 'eight fixtures, the prose contract and the sums');
});

test('any difference from Aokie\'s copy is a problem: a fixture, the prose, the sums, a missing file, an extra file', (t) => {
  for (const name of ['transfer-v1.md', 'SHA256SUMS', 'transfer-v1.cancel.fixture.json']) {
    const { dir, aokie } = copyAsAokieHasIt(t);
    edit(path.join(aokie, name), (text) => `${text} `);
    const problems = compareWithAokie(dir, aokie).problems;
    assert.equal(problems.length, 1, problems.join('\n'));
    assert.match(problems[0], new RegExp(`^${name.replace('.', '\\.')} differs from Aokie's`));
  }
  const { dir, aokie } = copyAsAokieHasIt(t);
  writeFileSync(path.join(aokie, 'transfer-v1.new.fixture.json'), '{}\n');
  assert.match(compareWithAokie(dir, aokie).problems.join('\n'), /transfer-v1\.new\.fixture\.json is in Aokie's folder and missing here/);
  const mine = copyAsAokieHasIt(t);
  writeFileSync(path.join(mine.dir, 'README.md'), 'mine\n');
  assert.match(compareWithAokie(mine.dir, mine.aokie).problems.join('\n'), /README\.md is here and not in Aokie's folder/);
  assert.match(compareWithAokie(dir, path.join(aokie, 'nowhere')).problems.join('\n'), /does not exist/);
});

test('run as a program: OK without the variable (with a note), FAIL with a copy that differs', (t) => {
  const plain = spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: '' } });
  assert.equal(plain.status, 0, plain.stderr);
  assert.match(plain.stdout, /OK, 8 fixtures match SHA256SUMS, 3 in oaiy-only/);
  assert.match(plain.stdout, /Aokie's copy was not compared/);

  const { aokie } = copyAsAokieHasIt(t);
  const same = spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: aokie } });
  assert.equal(same.status, 0, same.stderr);
  assert.match(same.stdout, /10 files identical with Aokie's copy/);

  edit(path.join(aokie, 'transfer-v1.md'), (text) => `${text}\nAn extra line.\n`);
  const drift = spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: aokie } });
  assert.equal(drift.status, 1);
  assert.match(drift.stderr, /FAIL, 1 problem/);
  assert.match(drift.stderr, /transfer-v1\.md differs from Aokie's/);
});
