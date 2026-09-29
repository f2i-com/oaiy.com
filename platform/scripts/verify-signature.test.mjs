// The check a maintainer makes before the first release (verify-signature.mjs).
//
//   node --test platform/scripts/verify-signature.test.mjs
//
// Against the fixtures of the desktop's update module: stand-in installers signed by the Tauri CLI with a throwaway key.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { CannotCheck, DEFAULT_CONF, pubkeyFromConf, verifyFile } from './verify-signature.mjs';
import { makeKeys, sign } from './minisign.testing.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'verify-signature.mjs');
const testdata = path.resolve(here, '..', 'desktop', 'src-tauri', 'src', 'update', 'testdata');
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-verify-signature-test-'));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

const payload = fs.readFileSync(path.join(testdata, 'windows-setup.bin'));
const signature = fs.readFileSync(path.join(testdata, 'windows-setup.bin.sig'), 'utf8').trim();
const pubkey = fs.readFileSync(path.join(testdata, 'throwaway.key.pub'), 'utf8').trim();
const production = pubkeyFromConf(DEFAULT_CONF);

let counter = 0;
/** A probe file with a signature beside it, in a folder of its own; returns the file's path. */
function probe({ bytes = payload, sig = signature } = {}) {
  const dir = path.join(scratch, `probe-${++counter}`);
  fs.mkdirSync(dir);
  const file = path.join(dir, 'probe.bin');
  fs.writeFileSync(file, bytes);
  fs.writeFileSync(`${file}.sig`, `${sig}\n`);
  return file;
}
const run = (...args) => spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });
const conf = (key) => {
  const file = path.join(scratch, `conf-${++counter}.json`);
  fs.writeFileSync(file, JSON.stringify({ plugins: { updater: { pubkey: key } } }));
  return file;
};
const keyFile = (key) => {
  const file = path.join(scratch, `key-${++counter}.pub`);
  fs.writeFileSync(file, `${key}\n`);
  return file;
};

describe('the check', () => {
  it('passes a file with its own signature and the key that made it, and shows only what is public', () => {
    const file = probe();
    const result = run(file, '--conf', conf(pubkey));
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /^OK: probe\.bin verifies against the public key in conf-\d+\.json \(plugins\.updater\.pubkey\); key id [0-9A-F]{16}\n/);
    assert.match(result.stdout, /signed as: timestamp:\d+\tfile:OAIY_0\.1\.0_x64-setup\.exe/);
    // Not the signature, not a key.
    const shown = result.stdout + result.stderr;
    assert.ok(!shown.includes(signature) && !shown.includes(pubkey) && !shown.includes(signature.slice(0, 40)) && !shown.includes(pubkey.slice(0, 40)));
    // The same with the key as a file.
    assert.equal(run(file, '--pubkey-file', keyFile(pubkey)).status, 0);
  });

  it('does not pass a file that was changed, a signature of another file, or another key', () => {
    const changed = probe({ bytes: Buffer.concat([payload.subarray(0, 10), Buffer.from([payload[10] ^ 1]), payload.subarray(11)]) });
    const bad = run(changed, '--conf', conf(pubkey));
    assert.equal(bad.status, 1);
    assert.match(bad.stderr, /^NOT VERIFIED: probe\.bin does not verify against .*: the file does not match its signature\n/);
    // The right file and signature against the production key: the throwaway signature is not the production key's.
    const other = run(probe(), '--conf', conf(production));
    assert.equal(other.status, 1);
    assert.match(other.stderr, /another key than the public key names \(the key ids differ\)/);
    // And the key of this repository, as the default, refuses a throwaway signature the same way.
    assert.equal(run(probe()).status, 1);
  });

  it('says it cannot check, and does not pass, when something is missing or is not what it should be', () => {
    const file = probe();
    const noSignature = path.join(scratch, 'lonely.bin');
    fs.writeFileSync(noSignature, 'x');
    for (const [args, pattern] of [
      [[noSignature, '--conf', conf(pubkey)], /CANNOT CHECK: the signature .*lonely\.bin\.sig cannot be read/],
      [[path.join(scratch, 'nowhere.bin'), '--conf', conf(pubkey)], /CANNOT CHECK: the file .*nowhere\.bin cannot be read/],
      [[file, '--conf', path.join(scratch, 'nowhere.json')], /CANNOT CHECK: the configuration .*nowhere\.json cannot be read/],
      [[file, '--conf', conf('')], /CANNOT CHECK: .* has no plugins\.updater\.pubkey/],
      [[file, '--conf', conf('not a key')], /CANNOT CHECK: the public key is not a minisign public key/],
      [[file, '--conf', conf(signature)], /CANNOT CHECK: the public key is not a minisign public key/],
      [[probe({ sig: 'not a signature' }), '--conf', conf(pubkey)], /CANNOT CHECK: the signature is not a minisign signature/],
      [[probe({ sig: pubkey }), '--conf', conf(pubkey)], /CANNOT CHECK: the signature is not a minisign signature/],
      [[], /CANNOT CHECK: usage/],
      [[file, '--wat'], /CANNOT CHECK: unexpected argument "--wat"/],
      [[file, '--conf', conf(pubkey), '--pubkey-file', keyFile(pubkey)], /give --conf or --pubkey-file, not both/],
    ]) {
      const result = run(...args);
      assert.equal(result.status, 2, `${args.join(' ')}: ${result.stdout}${result.stderr}`);
      assert.match(result.stderr, pattern, args.join(' '));
    }
  });

  it('is what the release job checks with: the same answer as minisign.mjs for a key of its own making', () => {
    const keys = makeKeys(5);
    const data = Buffer.from('a probe');
    const good = verifyFile({ data, signature: sign(keys, data), pubkey: keys.pubkey });
    assert.equal(good.ok, true);
    assert.equal(good.keyId, keys.keyId.toString('hex').toUpperCase());
    assert.equal(verifyFile({ data: Buffer.from('another'), signature: sign(keys, data), pubkey: keys.pubkey }).ok, false);
    assert.throws(() => verifyFile({ data, signature: 'x', pubkey: keys.pubkey }), CannotCheck);
  });

  it('reads the key of this repository by default: a public minisign key, never a private one', () => {
    const text = Buffer.from(production, 'base64').toString('utf8');
    assert.match(text, /^untrusted comment: minisign public key: [0-9A-F]{16}\n/);
    assert.ok(!/secret key/i.test(text));
  });
});
