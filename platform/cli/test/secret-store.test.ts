// The CLI's secret store: owner-only and whole, as the desktop's stores are (`secret_file.rs`). A world-readable file
// hands every key to every account on the machine, and one written in place leaves a broken file after a crash, which
// reads as no secrets at all, so the next store wrote over every other key.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-secret-store-'));
const data = path.join(root, 'new', 'data');
process.env.OAIY_DATA_DIR = data;
delete process.env.OAIY_TEST_SECRET;
// (after the environment is set: where the store lives is read as the module loads)
const { getSecret, storeSecret } = await import('../src/node-host/context');

const file = path.join(data, 'secrets.json');
const mode = (p: string) => fs.statSync(p).mode & 0o777;
const posix = process.platform !== 'win32';
const names = () => fs.readdirSync(data).sort();

try {
  const before = process.umask(0o022);
  storeSecret('OAIY_TEST_SECRET', 'first');
  assert.equal(getSecret('OAIY_TEST_SECRET'), 'first');
  assert.deepEqual(names(), ['secrets.json'], 'no staging file is left beside it');
  if (posix) {
    assert.equal(mode(file), 0o600, 'the file is the owner\'s alone');
    assert.equal(mode(data), 0o700, 'and so is a folder made for it');
  }

  // A file an older CLI left open to everyone is replaced by a private one, not narrowed after the write.
  if (posix) fs.chmodSync(file, 0o644);
  storeSecret('OAIY_OTHER', 'second');
  assert.equal(getSecret('OAIY_TEST_SECRET'), 'first', 'the other keys stay');
  assert.equal(getSecret('OAIY_OTHER'), 'second');
  if (posix) assert.equal(mode(file), 0o600);

  // A crash before the new file is in place leaves the old one whole, and no copy of the secrets.
  const whole = fs.readFileSync(file, 'utf-8');
  const rename = fs.renameSync;
  fs.renameSync = () => {
    throw new Error('the disk went away');
  };
  try {
    assert.throws(() => storeSecret('OAIY_THIRD', 'third'), /the disk went away/);
  } finally {
    fs.renameSync = rename;
  }
  assert.equal(fs.readFileSync(file, 'utf-8'), whole, 'the file is as it was');
  assert.deepEqual(names(), ['secrets.json'], 'and the staging file is gone');

  // Removing a key is a write too.
  storeSecret('OAIY_OTHER', '');
  assert.equal(getSecret('OAIY_OTHER'), undefined);
  if (posix) assert.equal(mode(file), 0o600);
  process.umask(before);
  console.log('secret store: ok');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
