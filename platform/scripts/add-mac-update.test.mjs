// The Mac added to a published release's update feed (add-mac-update.mjs).
//
//   node --test platform/scripts/add-mac-update.test.mjs
//
// Each test starts from a feed and checksums as the release job publishes them (Windows and Linux), and the Mac's files as
// the Mac update workflow has them after its `sign` job: the packed app and its .sig.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { addMacEntry, imageName, updateSums } from './add-mac-update.mjs';
import { FeedError, macAsset } from './make-latest-json.mjs';
import { makeKeys, sign } from './minisign.testing.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'add-mac-update.mjs');
const conf = path.join(here, '..', 'desktop', 'src-tauri', 'tauri.conf.json');
const testdata = path.join(here, '..', 'desktop', 'src-tauri', 'src', 'update', 'testdata');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-mac-update-test-'));
after(() => fs.rmSync(root, { recursive: true, force: true }));

const VERSION = '0.1.9';
const ASSET = `oaiy-desktop-${VERSION}-macos-arm64.app.tar.gz`;
const SIGNED_AS = `OAIY_${VERSION}_aarch64.app.tar.gz`;
const IMAGE_SHA = 'a'.repeat(64);
const keys = makeKeys(4);
const PACKED = Buffer.concat([Buffer.from([0x1f, 0x8b, 0x08, 0x00]), Buffer.from('pretend this is the packed app')]);
const hex = (data) => crypto.createHash('sha256').update(data).digest('hex');

/** The feed the release job wrote for VERSION (make-latest-json.mjs's form). */
function published(version = VERSION) {
  const base = `https://github.com/f2i-com/oaiy.com/releases/download/v${version}`;
  return {
    version,
    notes: `OAIY ${version}\n\nWhat changed: https://github.com/f2i-com/oaiy.com/releases/tag/v${version}`,
    pub_date: '2026-10-19T01:02:03Z',
    platforms: {
      'windows-x86_64': { signature: sign(keys, Buffer.from('nsis'), { comment: `timestamp:1790000000\tfile:OAIY_${version}_x64-setup.exe` }), url: `${base}/oaiy-desktop-${version}-windows-x64-setup.exe` },
      'linux-x86_64': { signature: sign(keys, Buffer.from('appimage'), { comment: `timestamp:1790000000\tfile:OAIY_${version}_amd64.AppImage` }), url: `${base}/oaiy-desktop-${version}-linux-x86_64.AppImage` },
    },
  };
}

const feedText = (feed) => `${JSON.stringify(feed, null, 2)}\n`;

/** The release job's checksums for the feed. */
function sums(feed = published()) {
  return [
    `${hex(feedText(feed))}  latest.json`,
    `${'1'.repeat(64)}  oaiy-cli-${VERSION}.tar.gz`,
    `${'2'.repeat(64)}  oaiy-desktop-${VERSION}-linux-x86_64.AppImage`,
    `${'3'.repeat(64)}  oaiy-desktop-${VERSION}-windows-x64-setup.exe`,
    `${'4'.repeat(64)}  oaiy-web-${VERSION}.zip`,
    `${'5'.repeat(64)}  release-evidence-linux.json`,
    '',
  ].join('\n');
}

let counter = 0;
/** The Mac's folder: the packed app and its .sig (`sig` overrides it, null leaves it out). */
function macDir({ bytes = PACKED, signedAs = SIGNED_AS, sig, withKeys = keys } = {}) {
  const dir = path.join(root, `mac-${++counter}`);
  fs.mkdirSync(dir);
  if (bytes !== null) fs.writeFileSync(path.join(dir, ASSET), bytes);
  const signature = sig === undefined ? sign(withKeys, bytes ?? Buffer.alloc(0), { comment: `timestamp:1790000000\tfile:${signedAs}` }) : sig;
  if (signature !== null) fs.writeFileSync(path.join(dir, `${ASSET}.sig`), `${signature}\n`);
  return dir;
}

const add = (options = {}) => addMacEntry({ feed: published(), dir: macDir(), version: VERSION, pubkey: keys.pubkey, ...options });

describe('the feed with the Mac in it', () => {
  it('is the published feed with one entry added: the Mac, at its asset on this release, with the content of its .sig', () => {
    const dir = macDir();
    const next = addMacEntry({ feed: published(), dir, version: VERSION, tag: `v${VERSION}`, pubkey: keys.pubkey });
    assert.deepEqual(Object.keys(next), ['version', 'notes', 'pub_date', 'platforms']);
    assert.deepEqual(Object.keys(next.platforms), ['windows-x86_64', 'linux-x86_64', 'darwin-aarch64']);
    assert.equal(next.platforms['darwin-aarch64'].url, `https://github.com/f2i-com/oaiy.com/releases/download/v${VERSION}/${ASSET}`);
    assert.equal(next.platforms['darwin-aarch64'].signature, fs.readFileSync(path.join(dir, `${ASSET}.sig`), 'utf8').trim());
  });

  it('keeps everything else exactly: the version, the notes, the date and the other platforms, which read the same feed after it', () => {
    const before = published();
    const next = addMacEntry({ feed: structuredClone(before), dir: macDir(), version: VERSION, pubkey: keys.pubkey });
    const { 'darwin-aarch64': mac, ...others } = next.platforms;
    assert.ok(mac);
    assert.deepEqual(others, before.platforms);
    assert.deepEqual({ ...next, platforms: others }, before);
    // Written the way the release job writes a feed, so the other entries' text is the same too.
    assert.ok(feedText(next).startsWith(feedText(before).split('\n  }\n}')[0]), 'the published feed is the start of the new one');
  });

  it('is under the tag the release was published as, with or without its v', () => {
    assert.ok(add({ tag: VERSION }).platforms['darwin-aarch64'].url.includes(`/download/${VERSION}/`));
    assert.throws(() => add({ tag: 'v0.1.8' }), /is not the version 0\.1\.9/);
  });

  it('names the Mac by the key the updater plugin looks up on Apple silicon, and the asset the desktop takes', () => {
    assert.deepEqual(macAsset(VERSION), { key: 'darwin-aarch64', asset: ASSET, suffix: '.app.tar.gz', signedName: SIGNED_AS, what: 'the Mac app (an .app.tar.gz)' });
    assert.equal(imageName(VERSION), `oaiy-desktop-${VERSION}-macos-arm64.dmg`);
  });
});

describe('what is refused, with nothing written', () => {
  it('a feed for another version, or with no platforms', () => {
    assert.throws(() => add({ feed: published('0.1.8') }), /version "0\.1\.8", not 0\.1\.9/);
    assert.throws(() => add({ feed: { ...published(), platforms: {} } }), /lists no platforms/);
    assert.throws(() => add({ feed: [] }), /not a JSON object/);
  });

  it('a feed that has the Mac already: nothing is replaced', () => {
    const feed = published();
    feed.platforms['darwin-aarch64'] = { signature: 'x', url: 'y' };
    assert.throws(() => add({ feed }), /darwin-aarch64 entry already/);
  });

  it('a missing packed app, or a missing, empty or unreadable signature', () => {
    assert.throws(() => add({ dir: macDir({ bytes: null }) }), FeedError);
    assert.throws(() => add({ dir: macDir({ sig: null }) }), /no signature/);
    assert.throws(() => add({ dir: macDir({ sig: '' }) }), /empty/);
    assert.throws(() => add({ dir: macDir({ sig: 'not base64!' }) }), /not base64/);
  });

  it('a signature made by another key than the one every installed OAIY carries', () => {
    assert.throws(() => add({ dir: macDir({ withKeys: makeKeys(9) }) }), /does not verify against the public key/);
  });

  it('a packed app changed after it was signed', () => {
    const dir = macDir();
    fs.writeFileSync(path.join(dir, ASSET), Buffer.concat([PACKED, Buffer.from('!')]));
    assert.throws(() => add({ dir }), /does not verify/);
  });

  it('a signature made for an older release, for the bundler’s own name (which has no version), or for another kind of file', () => {
    assert.throws(() => add({ dir: macDir({ signedAs: 'OAIY_0.1.8_aarch64.app.tar.gz' }) }), /not version 0\.1\.9/);
    assert.throws(() => add({ dir: macDir({ signedAs: 'OAIY.app.tar.gz' }) }), /not version 0\.1\.9/);
    for (const other of [`OAIY_${VERSION}_x64-setup.exe`, `OAIY_${VERSION}_amd64.AppImage`, `oaiy-cli-${VERSION}.tar.gz`, `OAIY_${VERSION}_aarch64.dmg`]) {
      assert.throws(() => add({ dir: macDir({ signedAs: other }) }), /not a file whose name ends ".app.tar.gz"/, other);
    }
  });

  it('a file that is not a gzip file, though signed under the right name', () => {
    assert.throws(() => add({ dir: macDir({ bytes: Buffer.from('PK\x03\x04 a zip') }) }), /does not start as a gzip file does/);
  });

  it('a repository other than the one installed copies are pinned to, and a version that is not one', () => {
    assert.throws(() => add({ repo: 'someone/oaiy.com', pinnedRepo: 'f2i-com/oaiy.com' }), /not the one every installed OAIY reads/);
    assert.throws(() => add({ repo: 'not a repo' }), /owner\/name/);
    assert.throws(() => add({ version: '0.1' }), /not a version/);
  });
});

describe('the checksums', () => {
  it('give latest.json its new digest and add the packed app, its signature and the image, in sha256sum’s order', () => {
    const next = updateSums(sums(), { 'latest.json': 'f'.repeat(64), [ASSET]: 'b'.repeat(64), [`${ASSET}.sig`]: 'c'.repeat(64), [imageName(VERSION)]: IMAGE_SHA }, { mustHave: ['latest.json'] });
    const names = next.trim().split('\n').map((line) => line.split('  ')[1]);
    assert.deepEqual(names, [
      'latest.json',
      `oaiy-cli-${VERSION}.tar.gz`,
      `oaiy-desktop-${VERSION}-linux-x86_64.AppImage`,
      `oaiy-desktop-${VERSION}-macos-arm64.app.tar.gz`,
      `oaiy-desktop-${VERSION}-macos-arm64.app.tar.gz.sig`,
      `oaiy-desktop-${VERSION}-macos-arm64.dmg`,
      `oaiy-desktop-${VERSION}-windows-x64-setup.exe`,
      `oaiy-web-${VERSION}.zip`,
      'release-evidence-linux.json',
    ]);
    assert.ok(next.startsWith(`${'f'.repeat(64)}  latest.json\n`));
    // The release job's lines for the other files are as they were.
    for (const line of sums().trim().split('\n').slice(1)) assert.ok(next.includes(`${line}\n`), line);
    assert.ok(next.endsWith('\n') && !next.endsWith('\n\n'));
  });

  it('refuse checksums that are not the release job’s, and a digest that is not one', () => {
    assert.throws(() => updateSums('not a checksum line\n', {}), /not "<sha256>  <file>"/);
    assert.throws(() => updateSums(`${'1'.repeat(64)}  a\n${'2'.repeat(64)}  a\n`, {}), /twice/);
    assert.throws(() => updateSums(`${'1'.repeat(64)}  oaiy-cli.tar.gz\n`, {}, { mustHave: ['latest.json'] }), /does not list latest\.json/);
    assert.throws(() => updateSums(sums(), { 'latest.json': 'nope' }), /not a sha256 digest/);
    assert.throws(() => updateSums(sums(), { 'a/b': 'f'.repeat(64) }), /not a file name/);
  });
});

describe('as the Mac update workflow runs it', () => {
  function inputs(feed = published()) {
    const dir = path.join(root, `run-${++counter}`);
    fs.mkdirSync(dir);
    fs.writeFileSync(path.join(dir, 'latest.json'), feedText(feed));
    fs.writeFileSync(path.join(dir, 'SHA256SUMS.txt'), sums(feed));
    return dir;
  }
  const run = (args) => spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });

  it('writes the feed and the checksums into --out, and the checksums name the feed it wrote', () => {
    const release = inputs();
    const mac = macDir();
    const out = path.join(release, 'new');
    const result = run(['--feed', path.join(release, 'latest.json'), '--sums', path.join(release, 'SHA256SUMS.txt'), '--dir', mac, '--version', VERSION, '--tag', `v${VERSION}`, '--image-sha256', IMAGE_SHA, '--out', out]);
    assert.equal(result.status, 0, result.stderr);
    const feed = fs.readFileSync(path.join(out, 'latest.json'), 'utf8');
    assert.ok(JSON.parse(feed).platforms['darwin-aarch64']);
    const lines = fs.readFileSync(path.join(out, 'SHA256SUMS.txt'), 'utf8');
    assert.ok(lines.includes(`${hex(feed)}  latest.json\n`));
    assert.ok(lines.includes(`${hex(PACKED)}  ${ASSET}\n`));
    assert.ok(lines.includes(`${hex(fs.readFileSync(path.join(mac, `${ASSET}.sig`)))}  ${ASSET}.sig\n`));
    assert.ok(lines.includes(`${IMAGE_SHA}  ${imageName(VERSION)}\n`));
  });

  it('with --conf holds the signature to the key this repository ships: a test key is refused, and nothing is written', () => {
    const release = inputs();
    const out = path.join(release, 'new');
    const result = run(['--feed', path.join(release, 'latest.json'), '--sums', path.join(release, 'SHA256SUMS.txt'), '--dir', macDir(), '--version', VERSION, '--image-sha256', IMAGE_SHA, '--out', out, '--conf', conf]);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /::error::the Mac's update was NOT added to the feed: .*does not verify against the public key in tauri\.conf\.json/);
    assert.equal(fs.existsSync(out), false);
  });

  it('refuses a missing argument, one it does not know, and an image digest that is not one', () => {
    const release = inputs();
    const base = ['--feed', path.join(release, 'latest.json'), '--sums', path.join(release, 'SHA256SUMS.txt'), '--dir', macDir(), '--version', VERSION, '--out', path.join(release, 'new')];
    assert.match(run(base).stderr, /usage/);
    assert.match(run([...base, '--image-sha256', IMAGE_SHA, '--clobber', 'yes']).stderr, /unexpected argument "--clobber"/);
    assert.match(run([...base, '--image-sha256', 'abc']).stderr, /not the disk image's sha256 digest/);
  });

  it('accepts the packed app the Tauri CLI really signed, against the key that signed it (testdata/mac-app.bin)', () => {
    // A real signature, made by `tauri signer sign` for OAIY_0.1.0_aarch64.app.tar.gz with a throwaway key (testdata/README.txt).
    const dir = path.join(root, `real-${++counter}`);
    fs.mkdirSync(dir);
    const asset = macAsset('0.1.0').asset;
    fs.copyFileSync(path.join(testdata, 'mac-app.bin'), path.join(dir, asset));
    fs.copyFileSync(path.join(testdata, 'mac-app.bin.sig'), path.join(dir, `${asset}.sig`));
    const pubkey = fs.readFileSync(path.join(testdata, 'throwaway.key.pub'), 'utf8').trim();
    const next = addMacEntry({ feed: published('0.1.0'), dir, version: '0.1.0', pubkey });
    assert.ok(next.platforms['darwin-aarch64'].url.endsWith(`/v0.1.0/${asset}`));
    // ...and not as another version: the downgrade a feed could dress it as, the old file under the new name.
    const newer = macAsset('0.2.0').asset;
    fs.copyFileSync(path.join(dir, asset), path.join(dir, newer));
    fs.copyFileSync(path.join(dir, `${asset}.sig`), path.join(dir, `${newer}.sig`));
    assert.throws(() => addMacEntry({ feed: published('0.2.0'), dir, version: '0.2.0', pubkey }), /OAIY_0\.1\.0_aarch64\.app\.tar\.gz.*not version 0\.2\.0/);
  });
});
