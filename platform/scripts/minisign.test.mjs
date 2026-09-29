// The minisign check the release job makes before it publishes (minisign.mjs).
//
//   node --test platform/scripts/minisign.test.mjs
//
// The main proof is a REAL signature: testdata/ of the desktop's update module holds two stand-in installers (an MZ
// file for Windows, an ELF one for the AppImage) signed by the Tauri CLI under the names the bundler gives an
// installer, with a throwaway key whose private half is not kept anywhere: the same fixtures the desktop's Rust
// tests verify with minisign-verify. What this file accepts is what an installed OAIY accepts.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { MinisignError, parsePublicKey, parseSignature, verifyMinisign } from './minisign.mjs';
import { makeKeys, sign } from './minisign.testing.mjs';

const testdata = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'desktop', 'src-tauri', 'src', 'update', 'testdata');
const payload = fs.readFileSync(path.join(testdata, 'windows-setup.bin'));
const signature = fs.readFileSync(path.join(testdata, 'windows-setup.bin.sig'), 'utf8').trim();
const appImage = fs.readFileSync(path.join(testdata, 'linux-appimage.bin'));
const appImageSignature = fs.readFileSync(path.join(testdata, 'linux-appimage.bin.sig'), 'utf8').trim();
const pubkey = fs.readFileSync(path.join(testdata, 'throwaway.key.pub'), 'utf8').trim();

describe('a signature made by the Tauri CLI', () => {
  it('verifies with the public key of the key that signed it', () => {
    const result = verifyMinisign(payload, signature, pubkey);
    assert.equal(result.ok, true, result.reason);
    assert.match(result.trustedComment, /^timestamp:\d+\tfile:OAIY_0\.1\.0_x64-setup\.exe$/);
    assert.equal(parseSignature(signature).prehashed, true, 'the CLI signs the hash of the file');
  });

  it('is made for the file the bundler names, and says so in the comment the key covers: the AppImage too', () => {
    const result = verifyMinisign(appImage, appImageSignature, pubkey);
    assert.equal(result.ok, true, result.reason);
    assert.match(result.trustedComment, /^timestamp:\d+\tfile:OAIY_0\.1\.0_amd64\.AppImage$/);
    // Neither verifies for the other's bytes.
    assert.equal(verifyMinisign(appImage, signature, pubkey).ok, false);
    assert.equal(verifyMinisign(payload, appImageSignature, pubkey).ok, false);
  });

  it('does not verify with one byte of the file changed, or the file cut short', () => {
    for (const changed of [Buffer.concat([payload.subarray(0, 10), Buffer.from([payload[10] ^ 1]), payload.subarray(11)]), payload.subarray(0, payload.length - 1), Buffer.alloc(0)]) {
      const result = verifyMinisign(changed, signature, pubkey);
      assert.deepEqual(result, { ok: false, reason: 'the file does not match its signature' });
    }
  });

  it('does not verify with the comment changed: it is covered by the signature', () => {
    const lines = Buffer.from(signature, 'base64').toString('utf8').split('\n');
    // What a downgrade would need: the old signature pointed at a newer version's name.
    lines[2] = lines[2].replace('OAIY_0.1.0_x64-setup.exe', 'OAIY_9.9.9_x64-setup.exe');
    assert.ok(lines[2].includes('9.9.9'));
    const edited = Buffer.from(lines.join('\n')).toString('base64');
    assert.deepEqual(verifyMinisign(payload, edited, pubkey), { ok: false, reason: "the signature's comment was changed" });
  });

  it('does not verify with the public key of another key', () => {
    const other = makeKeys(9);
    assert.match(verifyMinisign(payload, signature, other.pubkey).reason, /another key/);
    // A key that takes the fixture key's NAME but is another key: the signature itself does not check.
    const named = makeKeys(1);
    const id = parsePublicKey(pubkey).keyId;
    const forged = Buffer.from(`untrusted comment: minisign public key\n${Buffer.concat([Buffer.from('Ed'), id, named.raw]).toString('base64')}\n`).toString('base64');
    assert.deepEqual(verifyMinisign(payload, signature, forged), { ok: false, reason: 'the file does not match its signature' });
  });
});

describe('signatures made here, in both forms', () => {
  it('verify: the prehashed one the CLI writes, and the older one', () => {
    const keys = makeKeys(3);
    const data = crypto.randomBytes(5000);
    for (const prehashed of [true, false]) {
      assert.equal(verifyMinisign(data, sign(keys, data, { prehashed }), keys.pubkey).ok, true, `prehashed ${prehashed}`);
      assert.equal(verifyMinisign(Buffer.concat([data, Buffer.from('x')]), sign(keys, data, { prehashed }), keys.pubkey).ok, false, `changed, prehashed ${prehashed}`);
    }
  });

  it('are refused by another key', () => {
    const data = Buffer.from('an installer');
    assert.equal(verifyMinisign(data, sign(makeKeys(3), data), makeKeys(4).pubkey).ok, false);
    // The same name, another key.
    assert.equal(verifyMinisign(data, sign(makeKeys(3), data), makeKeys(3).pubkey).ok, false);
  });
});

describe('what is not a key or a signature', () => {
  it('is refused in words, never taken for a pass', () => {
    const keys = makeKeys(3);
    const good = sign(keys, Buffer.from('x'));
    for (const bad of ['', 'not base64!', Buffer.from('hello').toString('base64'), keys.pubkey]) {
      assert.throws(() => parseSignature(bad), MinisignError, bad);
    }
    for (const bad of ['', 'not base64!', Buffer.from('hello').toString('base64'), good]) {
      assert.throws(() => parsePublicKey(bad), MinisignError, bad);
    }
    assert.throws(() => verifyMinisign(Buffer.from('x'), good, good), MinisignError);
    assert.throws(() => verifyMinisign(Buffer.from('x'), keys.pubkey, keys.pubkey), MinisignError);
  });

  it('refuses a key with another algorithm marker', () => {
    const keys = makeKeys(3);
    const text = Buffer.from(keys.pubkey, 'base64').toString('utf8').split('\n');
    const bin = Buffer.from(text[1], 'base64');
    bin[0] = 0x58;
    assert.throws(() => parsePublicKey(Buffer.from(`${text[0]}\n${bin.toString('base64')}\n`).toString('base64')), /minisign public key/);
  });
});

describe('the key this repository ships', () => {
  it('is a readable minisign public key', () => {
    const conf = JSON.parse(fs.readFileSync(path.resolve(testdata, '..', '..', '..', 'tauri.conf.json'), 'utf8'));
    const parsed = parsePublicKey(conf.plugins.updater.pubkey);
    assert.equal(parsed.keyId.length, 8);
    // And it is not the throwaway key of the fixture.
    assert.notEqual(conf.plugins.updater.pubkey, pubkey);
    assert.equal(verifyMinisign(payload, signature, conf.plugins.updater.pubkey).ok, false);
  });
});
