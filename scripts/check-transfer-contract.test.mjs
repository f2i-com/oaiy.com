import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { AOKIE_FOLDER, checkLock, checkSums, checkTransfer, compareWithAokie, DEFAULT_DIR, digest, readLock, verifyCommit, writeLock, writeOaiySums } from './check-transfer-contract.mjs';

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
  assert.equal(result.ownFiles, 4, 'and the four that only OAIY has: the phrases, the two starts and the lock');
  assert.equal(result.compared, null, 'no comparison without Aokie\'s copy');
  assert.equal(result.commit, null, 'and no commit looked up');
  assert.match(result.lockedCommit, /^[0-9a-f]{40}$/);
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
  assert.equal(writeOaiySums(dir), 4);
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
  assert.match(plain.stdout, /OK, 8 fixtures match SHA256SUMS, 4 in oaiy-only/);
  assert.match(plain.stdout, /SKIPPED, NOT VERIFIED AGAINST AOKIE/, 'a run that did not check Aokie says so');

  const { aokie } = copyAsAokieHasIt(t);
  const same = spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: aokie } });
  assert.equal(same.status, 0, same.stderr);
  assert.match(same.stdout, /10 files identical with Aokie's copy/);
  assert.match(same.stdout, /THE COMMIT WAS NOT VERIFIED/, 'a folder that is not in a git checkout cannot show the commit');

  edit(path.join(aokie, 'transfer-v1.md'), (text) => `${text}\nAn extra line.\n`);
  const drift = spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: aokie } });
  assert.equal(drift.status, 1);
  assert.match(drift.stderr, /FAIL, 1 problem/);
  assert.match(drift.stderr, /transfer-v1\.md differs from Aokie's/);
});

test('the lock names the Aokie commit and records every file, and this folder is what it records', () => {
  const { lock, problems } = readLock(DEFAULT_DIR);
  assert.deepEqual(problems, []);
  assert.match(lock.commit, /^[0-9a-f]{40}$/);
  assert.equal(lock.folder, AOKIE_FOLDER);
  const top = readdirSync(DEFAULT_DIR).filter((n) => statSync(path.join(DEFAULT_DIR, n)).isFile()).sort();
  assert.deepEqual(Object.keys(lock.files).sort(), top, 'every file at the top of the folder, and no other');
  assert.deepEqual(checkLock(DEFAULT_DIR).problems, []);
  // The commit is written in that file and nowhere in the prose: a second place would be the one that goes out of date.
  const prose = readFileSync(path.join(DEFAULT_DIR, 'oaiy-only', 'README.md'), 'utf8');
  assert.ok(!prose.includes(lock.commit), 'oaiy-only/README.md does not repeat the commit');
});

test('a file edited here, a file added, a lock missing a file, a lock naming a gone one and a lock that is not well formed are each a problem', (t) => {
  const edited = copyOfFolder(t);
  edit(path.join(edited, 'transfer-v1.md'), (text) => `${text}\nAn edit made here.\n`);
  // The prose contract is in no SHA256SUMS: the lock is what notices it.
  assert.deepEqual(checkSums(edited).problems, []);
  assert.match(checkLock(edited).problems.join('\n'), /transfer-v1\.md is not the file copied from Aokie [0-9a-f]{8} \(oaiy-only\/SYNCED_FROM\.json\)/);
  assert.match(checkTransfer({ dir: edited, aokieDir: undefined }).problems.join('\n'), /transfer-v1\.md is not the file copied from Aokie/);

  const added = copyOfFolder(t);
  writeFileSync(path.join(added, 'NOTES.md'), 'mine\n');
  assert.match(checkLock(added).problems.join('\n'), /NOTES\.md is in the folder and not in oaiy-only\/SYNCED_FROM\.json/);

  const gone = copyOfFolder(t);
  rmSync(path.join(gone, 'transfer-v1.md'));
  assert.match(checkLock(gone).problems.join('\n'), /records transfer-v1\.md, which is not in the folder/);

  const unlocked = copyOfFolder(t);
  rmSync(path.join(unlocked, 'oaiy-only', 'SYNCED_FROM.json'));
  assert.match(checkLock(unlocked).problems.join('\n'), /SYNCED_FROM\.json is missing/);

  const badCommit = copyOfFolder(t);
  edit(path.join(badCommit, 'oaiy-only', 'SYNCED_FROM.json'), (text) => text.replace(/"commit": "[0-9a-f]{40}"/, '"commit": "49c46d8"'));
  assert.match(checkLock(badCommit).problems.join('\n'), /"commit" is not a full 40 digit commit hash/);

  const badDigest = copyOfFolder(t);
  edit(path.join(badDigest, 'oaiy-only', 'SYNCED_FROM.json'), (text) => text.replace(/"transfer-v1\.md": "[0-9a-f]{64}"/, '"transfer-v1.md": "nope"'));
  assert.match(checkLock(badDigest).problems.join('\n'), /the digest of transfer-v1\.md is not a SHA-256/);

  const notJson = copyOfFolder(t);
  writeFileSync(path.join(notJson, 'oaiy-only', 'SYNCED_FROM.json'), '{ nope');
  assert.match(checkLock(notJson).problems.join('\n'), /SYNCED_FROM\.json is not valid JSON/);

  const empty = copyOfFolder(t);
  edit(path.join(empty, 'oaiy-only', 'SYNCED_FROM.json'), (text) => JSON.stringify({ ...JSON.parse(text), files: {} }));
  assert.match(checkLock(empty).problems.join('\n'), /"files" lists no file/);
});

test('writing the lock after a sync records the commit and the files, and the sums of oaiy-only follow', (t) => {
  const dir = copyOfFolder(t);
  edit(path.join(dir, 'transfer-v1.md'), (text) => `${text}\nSynced.\n`);
  assert.ok(checkLock(dir).problems.length > 0);
  const commit = 'abcdef0123456789abcdef0123456789abcdef01';
  const lock = writeLock(commit, dir);
  assert.equal(lock.commit, commit);
  assert.deepEqual(checkTransfer({ dir, aokieDir: undefined }).problems, []);
  assert.equal(checkTransfer({ dir, aokieDir: undefined }).lockedCommit, commit);
  assert.throws(() => writeLock('49c46d8', dir), /not a full 40 digit commit hash/);
  assert.throws(() => writeLock('', dir), /not a full 40 digit commit hash/);
});

/** A git repository standing for Aokie's: the top-level files of `dir` committed at `docs/contracts/transfer`. Answers its folder and commit. */
function aokieRepo(t, dir) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'oaiy-transfer-aokie-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const folder = path.join(root, ...AOKIE_FOLDER.split('/'));
  mkdirSync(folder, { recursive: true });
  for (const name of readdirSync(dir).filter((n) => statSync(path.join(dir, n)).isFile())) {
    writeFileSync(path.join(folder, name), readFileSync(path.join(dir, name)));
  }
  const run = (...args) => {
    const r = spawnSync('git', ['-c', 'user.name=izuc', '-c', 'user.email=the@lance.name', '-c', 'core.autocrlf=false', '-c', 'commit.gpgsign=false', ...args], { cwd: root, encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    return r.stdout.trim();
  };
  run('init', '-q');
  run('add', '-A');
  run('commit', '-q', '-m', 'the contract');
  return { root, folder, commit: run('rev-parse', 'HEAD'), run };
}

const hasGit = spawnSync('git', ['--version']).status === 0;

test('the commit in the lock is looked up in an Aokie checkout: found and the same is verified', { skip: !hasGit && 'git is not installed' }, (t) => {
  const dir = copyOfFolder(t);
  const aokie = aokieRepo(t, dir);
  const lock = writeLock(aokie.commit, dir);
  const result = checkTransfer({ dir, aokieDir: aokie.folder });
  assert.deepEqual(result.problems, []);
  assert.equal(result.commit, 'verified');
  assert.equal(result.compared, 10);
  assert.equal(verifyCommit(lock, aokie.folder).status, 'verified');
});

test('a commit that is not in the checkout, or whose folder is not what the lock records, is a problem', { skip: !hasGit && 'git is not installed' }, (t) => {
  const dir = copyOfFolder(t);
  const aokie = aokieRepo(t, dir);
  // The copy was synced from a commit that is not there (the lock says a commit nobody has).
  writeLock('1234567890abcdef1234567890abcdef12345678', dir);
  const missing = checkTransfer({ dir, aokieDir: aokie.folder });
  assert.equal(missing.commit, 'failed');
  assert.match(missing.problems.join('\n'), /the commit 1234567890abcdef1234567890abcdef12345678 named in oaiy-only\/SYNCED_FROM\.json is not in the Aokie checkout/);

  // The commit is there, but Aokie's folder at that commit is not what was copied: the lock names a later commit than the one the copy is of.
  const first = aokie.commit;
  writeFileSync(path.join(aokie.folder, 'transfer-v1.md'), `${readFileSync(path.join(aokie.folder, 'transfer-v1.md'), 'utf8')}\nA later line.\n`);
  aokie.run('commit', '-q', '-am', 'a later contract');
  const second = aokie.run('rev-parse', 'HEAD');
  const wrong = verifyCommit(writeLock(second, dir), aokie.folder);
  assert.equal(wrong.status, 'failed');
  assert.match(wrong.problems.join('\n'), /transfer-v1\.md at Aokie [0-9a-f]{8} is not what oaiy-only\/SYNCED_FROM\.json records/);
  // Named as the commit it was copied from, it is verified.
  assert.equal(verifyCommit(writeLock(first, dir), aokie.folder).status, 'verified');

  // A file the commit has and the lock does not, and one the lock has and the commit does not.
  writeFileSync(path.join(aokie.folder, 'transfer-v1.extra.fixture.json'), '{}\n');
  aokie.run('add', '-A');
  aokie.run('commit', '-q', '-m', 'a new fixture');
  const third = aokie.run('rev-parse', 'HEAD');
  assert.match(verifyCommit(writeLock(third, dir), aokie.folder).problems.join('\n'), /transfer-v1\.extra\.fixture\.json is in Aokie's docs\/contracts\/transfer at [0-9a-f]{8} and not in oaiy-only\/SYNCED_FROM\.json/);
  aokie.run('rm', '-q', 'docs/contracts/transfer/transfer-v1.cancel.fixture.json');
  aokie.run('commit', '-q', '-m', 'a fixture gone');
  const fourth = aokie.run('rev-parse', 'HEAD');
  assert.match(verifyCommit(writeLock(fourth, dir), aokie.folder).problems.join('\n'), /records transfer-v1\.cancel\.fixture\.json, which is not in Aokie's docs\/contracts\/transfer at [0-9a-f]{8}/);
});

test('a folder that is not in a git checkout cannot show the commit: skipped, and said so', (t) => {
  const { dir, aokie } = copyAsAokieHasIt(t);
  const looked = verifyCommit(readLock(dir).lock, aokie);
  assert.equal(looked.status, 'skipped');
  assert.match(looked.note, /is not inside a git checkout, so the commit [0-9a-f]{40} was not looked up/);
  assert.deepEqual(looked.problems, []);
});

test('run as a program: without Aokie it says SKIPPED and passes, --require-aokie fails, and GitHub Actions gets an annotation', (t) => {
  const dir = copyOfFolder(t);
  const run = (args, env) => spawnSync(process.execPath, [script, '--dir', dir, ...args], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: '', GITHUB_ACTIONS: '', ...env } });
  const plain = run([]);
  assert.equal(plain.status, 0, plain.stderr);
  assert.match(plain.stdout, /SKIPPED, NOT VERIFIED AGAINST AOKIE: AOKIE_TRANSFER_CONTRACTS is not set/);
  assert.doesNotMatch(plain.stdout, /::warning/, 'no annotation outside GitHub Actions');
  assert.equal(run(['--require-aokie']).status, 1, 'the check before a merge does not pass on trust');
  const ci = run([], { GITHUB_ACTIONS: 'true' });
  assert.equal(ci.status, 0, ci.stderr);
  assert.match(ci.stdout, /^::warning title=transfer contract not verified against Aokie::/m);
});

test('run as a program: a checkout that does not have the locked commit fails', { skip: !hasGit && 'git is not installed' }, (t) => {
  // The folder committed in a checkout of its own: the lock's commit (an earlier Aokie one) is not in it.
  const dir = copyOfFolder(t);
  const aokie = aokieRepo(t, dir);
  const r = spawnSync(process.execPath, [script, '--dir', dir], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: aokie.folder } });
  assert.equal(r.status, 1, r.stdout);
  assert.match(r.stderr, /named in oaiy-only\/SYNCED_FROM\.json is not in the Aokie checkout/);
});

test('run as a program: a checkout that has the locked commit is verified, and --require-aokie passes only then', { skip: !hasGit && 'git is not installed' }, (t) => {
  const dir = copyOfFolder(t);
  const aokie = aokieRepo(t, dir);
  const written = spawnSync(process.execPath, [script, '--dir', dir, '--write-lock', aokie.commit], { encoding: 'utf8' });
  assert.equal(written.status, 0, written.stderr);
  assert.match(written.stdout, new RegExp(`Aokie ${aokie.commit}, 10 files`));
  const env = { ...process.env, AOKIE_TRANSFER_CONTRACTS: aokie.folder };
  const checked = spawnSync(process.execPath, [script, '--dir', dir, '--require-aokie'], { encoding: 'utf8', env });
  assert.equal(checked.status, 0, checked.stderr + checked.stdout);
  assert.match(checked.stdout, /10 files identical with Aokie's copy/);
  assert.match(checked.stdout, new RegExp(`the folder at Aokie commit ${aokie.commit} is the one locked`));
  // The same, from a folder that is not in a git checkout: the files agree, the commit is not verified, and a required check fails.
  const { aokie: exported } = copyAsAokieHasIt(t);
  const loose = { ...process.env, AOKIE_TRANSFER_CONTRACTS: exported };
  const notLooked = spawnSync(process.execPath, [script, '--dir', dir], { encoding: 'utf8', env: loose });
  assert.equal(notLooked.status, 0, notLooked.stderr);
  assert.match(notLooked.stdout, /SKIPPED, THE COMMIT WAS NOT VERIFIED/);
  assert.equal(spawnSync(process.execPath, [script, '--dir', dir, '--require-aokie'], { encoding: 'utf8', env: loose }).status, 1);
});

test('run as a program: --write-lock refuses a short or a missing commit hash and writes nothing', (t) => {
  const dir = copyOfFolder(t);
  const before = readFileSync(path.join(dir, 'oaiy-only', 'SYNCED_FROM.json'), 'utf8');
  const bad = spawnSync(process.execPath, [script, '--dir', dir, '--write-lock', '49c46d8'], { encoding: 'utf8' });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /not a full 40 digit commit hash/);
  const none = spawnSync(process.execPath, [script, '--dir', dir, '--write-lock'], { encoding: 'utf8' });
  assert.equal(none.status, 1);
  assert.equal(readFileSync(path.join(dir, 'oaiy-only', 'SYNCED_FROM.json'), 'utf8'), before, 'the lock is as it was');
});

test('run as a program: a folder without its lock fails, whatever else is right', (t) => {
  // The workflow that releases relies on this: with no SYNCED_FROM.json nothing says which Aokie commit the folder is.
  const dir = copyOfFolder(t);
  rmSync(path.join(dir, 'oaiy-only', 'SYNCED_FROM.json'));
  const r = spawnSync(process.execPath, [script, '--dir', dir], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: '' } });
  assert.equal(r.status, 1, r.stdout);
  assert.match(r.stderr, /SYNCED_FROM\.json is missing: it says which Aokie commit the folder was copied from/);
  // ...and so does one whose lock is empty or is not the folder's.
  const empty = copyOfFolder(t);
  writeFileSync(path.join(empty, 'oaiy-only', 'SYNCED_FROM.json'), '{}\n');
  assert.equal(spawnSync(process.execPath, [script, '--dir', empty], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: '' } }).status, 1);
});

test('the release workflow refuses a release without the lock, and the release checklist runs the comparison with Aokie\'s checkout', () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const workflow = readFileSync(path.join(root, '.github', 'workflows', 'release.yml'), 'utf8');
  // The first job of the release, before anything is built, runs the check (which fails on a missing lock).
  const meta = workflow.slice(workflow.indexOf('\n  meta:'), workflow.indexOf('\n  verify:'));
  assert.match(meta, /- name: Transfer contract lock \(transfer_v1\)\s+run: node scripts\/check-transfer-contract\.mjs\s*\n/, 'the release workflow runs the contract check in its first job');
  assert.doesNotMatch(meta, /continue-on-error/, 'and it cannot be let through');
  // CI runs it too (the desktop leg), and the checklist says how to compare with Aokie's checkout, which the workflows cannot.
  assert.match(readFileSync(path.join(root, '.github', 'workflows', 'ci.yml'), 'utf8'), /npm run check:transfer-contract/);
  const releasing = readFileSync(path.join(root, 'docs', 'RELEASING.md'), 'utf8');
  assert.match(releasing, /AOKIE_TRANSFER_CONTRACTS=<Aokie checkout>\/docs\/contracts\/transfer node scripts\/check-transfer-contract\.mjs --require-aokie/);
  assert.match(releasing, /Do not tag on a NOT VERIFIED run/);
  assert.doesNotMatch(releasing, /OAIY_TRANSFER_CONTRACTS/, 'the variable the script reads is AOKIE_TRANSFER_CONTRACTS');
});

test('run as a program: the real folder of this repository passes, on its own sums and its lock', () => {
  const r = spawnSync(process.execPath, [script], { encoding: 'utf8', env: { ...process.env, AOKIE_TRANSFER_CONTRACTS: '' } });
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /OK, 8 fixtures match SHA256SUMS, 4 in oaiy-only\/, every file is the one locked from Aokie [0-9a-f]{40}/);
});
