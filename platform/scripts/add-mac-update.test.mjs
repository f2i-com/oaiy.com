// The Mac added to a published release's update feed (add-mac-update.mjs).
//
//   node --test platform/scripts/add-mac-update.test.mjs
//
// Each test starts from a feed and checksums as the release job publishes them (Windows and Linux), and the Mac's files as
// the Mac update workflow has them after its `sign` job: the packed app and its .sig. The last part is the Mac build's
// (mac-build.yml) version rule, the workflow's own text cut out and run as release-version.test.mjs runs release.yml's.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { addMacEntry, checkMacFiles, imageName, PENDING, planMacUpdate, updateSums } from './add-mac-update.mjs';
import { FeedError, macAsset } from './make-latest-json.mjs';
import { makeKeys, sign } from './minisign.testing.mjs';
import { parseYaml, workflowSteps } from './workflow-yaml.mjs';

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

describe('what a run does to the release as it is (--plan)', () => {
  const IMAGE = imageName(VERSION);
  /** The files of a release as the release job and the Mac build leave it, before any Mac update ran. */
  const published_files = ['latest.json', 'SHA256SUMS.txt', IMAGE, `${IMAGE}.sha256`, `oaiy-desktop-${VERSION}-windows-x64-setup.exe`, `oaiy-desktop-${VERSION}-linux-x86_64.AppImage`];
  const withMac = () => {
    const feed = published();
    feed.platforms['darwin-aarch64'] = { signature: 'c2ln', url: `https://github.com/f2i-com/oaiy.com/releases/download/v${VERSION}/${ASSET}` };
    return feed;
  };
  /** The plan for a release with `assets`, whose latest.json is `given.feed` (the published one unless said) and latest.pending.json `given.pendingFeed`. */
  const plan = (assets, given = {}) => planMacUpdate({ version: VERSION, assets, feed: 'feed' in given ? given.feed : published(), pendingFeed: given.pendingFeed });

  it('makes the update of a release that has none, swapping in the new feed and checksums', () => {
    assert.deepEqual(plan(published_files), { mode: 'make', reuseApp: false, reuseSig: false, swap: ['latest.json', 'SHA256SUMS.txt'] });
  });

  it('reuses the packed app and its .sig an earlier run uploaded before it failed, so this one can finish', () => {
    assert.deepEqual(plan([...published_files, ASSET, `${ASSET}.sig`, `${ASSET}.sha256`]), { mode: 'make', reuseApp: true, reuseSig: true, swap: ['latest.json', 'SHA256SUMS.txt'] });
    // Cut short between the two uploads: the archive is reused, and signed in this run.
    assert.deepEqual(plan([...published_files, ASSET]), { mode: 'make', reuseApp: true, reuseSig: false, swap: ['latest.json', 'SHA256SUMS.txt'] });
    // Pending files an earlier run uploaded and never swapped in are made again (the workflow removes them first).
    assert.equal(plan([...published_files, ASSET, PENDING['latest.json'], PENDING['SHA256SUMS.txt']]).mode, 'make');
  });

  it('finishes a swap an earlier run was cut short in, from the feed in place or the new one waiting', () => {
    // The feed is in place, the checksums are not.
    assert.deepEqual(plan([...published_files, ASSET, `${ASSET}.sig`, PENDING['SHA256SUMS.txt']], { feed: withMac() }), { mode: 'finish', reuseApp: false, reuseSig: false, swap: ['SHA256SUMS.txt'] });
    // Cut between removing the old feed and renaming the new one: the release has no latest.json, and the new one waits.
    const between = [...published_files.filter((name) => name !== 'latest.json'), ASSET, `${ASSET}.sig`, PENDING['latest.json'], PENDING['SHA256SUMS.txt']];
    assert.deepEqual(plan(between, { feed: undefined, pendingFeed: withMac() }), { mode: 'finish', reuseApp: false, reuseSig: false, swap: ['latest.json', 'SHA256SUMS.txt'] });
  });

  it('leaves alone a release whose feed lists the Mac already, whatever files it has', () => {
    assert.throws(() => plan([...published_files, ASSET, `${ASSET}.sig`, `${ASSET}.sha256`], { feed: withMac() }), /lists darwin-aarch64 already: 0\.1\.9 has its Mac update, and nothing is replaced/);
    // (the guard is the feed, not the files: a release whose Mac files were deleted still has its Mac entry)
    assert.throws(() => plan(published_files, { feed: withMac() }), /has its Mac update/);
  });

  it('refuses a release in a state no run leaves, and one that is not ready for its Mac update yet', () => {
    assert.throws(() => plan([...published_files, `${ASSET}.sig`]), /has oaiy-desktop-0\.1\.9-macos-arm64\.app\.tar\.gz\.sig but not .*which no run leaves: delete it/);
    assert.throws(() => plan([...published_files, `${ASSET}.sha256`]), /\.sha256 but not/);
    assert.throws(() => plan(published_files.filter((name) => name !== `${IMAGE}.sha256`)), /run mac-build\.yml with release=v0\.1\.9 first/);
    assert.throws(() => plan(published_files.filter((name) => name !== 'latest.json'), { feed: undefined }), /has no latest\.json/);
    assert.throws(() => plan([...published_files.filter((name) => name !== 'latest.json'), PENDING['latest.json']], { feed: undefined, pendingFeed: published() }), /put the release job's latest\.json back/);
    assert.throws(() => plan(published_files.filter((name) => name !== 'SHA256SUMS.txt')), /no SHA256SUMS\.txt/);
    assert.throws(() => plan(published_files, { feed: published('0.1.8') }), /for version "0\.1\.8", not 0\.1\.9/);
    assert.throws(() => plan(published_files, { feed: undefined }), /latest\.json is not a JSON object/);
    assert.throws(() => planMacUpdate({ version: '0.1', assets: published_files, feed: published() }), /not a version/);
  });

  it('puts the new files up under names that keep their extension, so each keeps its content type when renamed', () => {
    for (const [name, pending] of Object.entries(PENDING)) {
      assert.equal(path.extname(pending), path.extname(name));
      assert.notEqual(pending, name);
    }
    assert.deepEqual(Object.keys(PENDING), ['latest.json', 'SHA256SUMS.txt']);
  });
});

describe('the Mac files an earlier run left on the release (--check)', () => {
  it('pass when they verify against the key and were made for this version', () => {
    const dir = macDir();
    assert.equal(checkMacFiles({ dir, version: VERSION, pubkey: keys.pubkey }), fs.readFileSync(path.join(dir, `${ASSET}.sig`), 'utf8').trim());
  });

  it('are refused when another key signed them, they were made for another version, or the archive changed', () => {
    assert.throws(() => checkMacFiles({ dir: macDir({ withKeys: makeKeys(9) }), version: VERSION, pubkey: keys.pubkey }), /does not verify against the public key/);
    assert.throws(() => checkMacFiles({ dir: macDir({ signedAs: 'OAIY_0.1.8_aarch64.app.tar.gz' }), version: VERSION, pubkey: keys.pubkey }), /not version 0\.1\.9/);
    const dir = macDir();
    fs.appendFileSync(path.join(dir, ASSET), '!');
    assert.throws(() => checkMacFiles({ dir, version: VERSION, pubkey: keys.pubkey }), /does not verify/);
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

  it('--plan writes what it decided as lines for $GITHUB_OUTPUT, and says why when there is nothing to do', () => {
    const release = inputs();
    const assets = path.join(release, 'assets.txt');
    fs.writeFileSync(assets, `${['latest.json', 'SHA256SUMS.txt', imageName(VERSION), `${imageName(VERSION)}.sha256`, ASSET, `${ASSET}.sig`].join('\n')}\n`);
    const result = run(['--plan', '--version', VERSION, '--assets', assets, '--feed', path.join(release, 'latest.json')]);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, 'mode=make\nreuse_app=true\nreuse_sig=true\nswap=latest.json SHA256SUMS.txt\n');
    const feed = published();
    feed.platforms['darwin-aarch64'] = { signature: 'c2ln', url: 'https://example.invalid/' };
    fs.writeFileSync(path.join(release, 'latest.json'), feedText(feed));
    const done = run(['--plan', '--version', VERSION, '--assets', assets, '--feed', path.join(release, 'latest.json')]);
    assert.equal(done.status, 1);
    assert.match(done.stderr, /::error::nothing was done to the release: the release's feed lists darwin-aarch64 already/);
    assert.match(run(['--plan', '--check', '--version', VERSION]).stderr, /two runs/);
  });

  it('--check passes the files an earlier run uploaded, and refuses ones the key in tauri.conf.json did not sign', () => {
    // A tauri.conf.json that carries the test key, standing in for the repository's.
    const testConf = path.join(root, `conf-${++counter}.json`);
    fs.writeFileSync(testConf, JSON.stringify({ plugins: { updater: { pubkey: keys.pubkey } } }));
    const good = run(['--check', '--dir', macDir(), '--version', VERSION, '--conf', testConf]);
    assert.equal(good.status, 0, good.stderr);
    assert.match(good.stdout, /^OK: oaiy-desktop-0\.1\.9-macos-arm64\.app\.tar\.gz and its \.sig .*, signed with the key in /);
    const refused = run(['--check', '--dir', macDir(), '--version', VERSION, '--conf', conf]);
    assert.equal(refused.status, 1);
    assert.match(refused.stderr, /::error::the Mac files on the release are NOT reused: .*does not verify against the public key in tauri\.conf\.json/);
  });

  it('--check without --conf checks nothing and says so, rather than passing files whose signature it did not verify', () => {
    const unchecked = run(['--check', '--dir', macDir({ withKeys: makeKeys(9) }), '--version', VERSION]);
    assert.equal(unchecked.status, 1);
    assert.equal(unchecked.stdout, '');
    assert.match(unchecked.stderr, /::error::the Mac files on the release are NOT reused: --check needs --conf <tauri\.conf\.json>: without the key the signature is not checked/);
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

// A Mac build's version with a suffix is a test build, and an installed OAIY with such a version is offered only a release
// newer than the one the suffix is on (update::version): one named after an older release than the newest would be
// replaced by the newest, older code than its own, and one named after a release not out yet would never be offered it.
describe("mac-build.yml: a test build's version, held to the newest release", () => {
  const repo = path.resolve(here, '..', '..');
  const document = parseYaml(fs.readFileSync(path.join(repo, '.github', 'workflows', 'mac-build.yml'), 'utf8'));
  const steps = workflowSteps(document).filter((s) => s.job === 'dmg');
  const STEP = 'The version, held to the newest release';
  const step = steps.find((s) => s.name === STEP);
  const bash = spawnSync('bash', ['-c', 'true']).status === 0;

  /**
   * The step's script, run as GitHub runs it, with `version` as the input and a `gh` that answers `newest` as the tag of
   * releases/latest (null: it fails, as with no release or no network). Returns the run and the calls made to `gh`.
   */
  function stamp(version, newest) {
    const dir = path.join(root, `version-${++counter}`);
    fs.mkdirSync(path.join(dir, 'bin'), { recursive: true });
    const calls = path.join(dir, 'gh-calls.txt');
    fs.writeFileSync(
      path.join(dir, 'bin', 'gh'),
      ['#!/usr/bin/env bash', `printf '%s\\n' "$*" >> '${calls}'`, newest === null ? 'echo "gh: Not Found (HTTP 404)" >&2; exit 1' : `printf '%s\\n' '${newest}'`, ''].join('\n'),
      { mode: 0o755 },
    );
    const env = { ...process.env, PATH: `${path.join(dir, 'bin')}${path.delimiter}${process.env.PATH}`, GH_TOKEN: 'read-only', GITHUB_REPOSITORY: 'f2i-com/oaiy.com', VERSION: version };
    const run = spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', '-c', step.run], { encoding: 'utf8', env, cwd: dir });
    return { ...run, calls: fs.existsSync(calls) ? fs.readFileSync(calls, 'utf8').trim().split('\n') : [] };
  }

  it('has no default version, so a run that names none stamps none (and not one named after a release that is no longer the newest)', () => {
    const input = document.on.workflow_dispatch.inputs.version;
    assert.equal(input.default, '');
    assert.notEqual(input.required, 'true', 'a run for the engine or WebKit alone names no version');
  });

  it('is the first step after the checkout, before any of the project’s code runs, with a token that may only read', () => {
    assert.ok(step, `the dmg job of mac-build.yml has no step "${STEP}"`);
    assert.equal(steps.indexOf(step), 1);
    assert.equal(steps[0].uses?.startsWith('actions/checkout@'), true);
    assert.deepEqual({ ...step.env }, { GH_TOKEN: '${{ github.token }}', VERSION: '${{ inputs.version }}' }, 'the input reaches the script as data');
    assert.ok(!step.run.includes('${{'), 'nothing is pasted into the script');
    assert.deepEqual({ ...document.permissions }, { contents: 'read' });
    assert.equal(document.jobs.dmg.permissions, undefined, 'the job takes the workflow’s permissions');
  });

  it('passes a test build named after the newest release, asking GitHub only which release that is', { skip: !bash && 'no bash on this machine' }, () => {
    for (const version of ['0.1.9-mac.1', '0.1.9-mac.44', '0.1.9-rc.1']) {
      const r = stamp(version, 'v0.1.9');
      assert.equal(r.status, 0, r.stdout + r.stderr);
      assert.match(r.stdout, /is a test build of v0\.1\.9, the newest release/);
      assert.deepEqual(r.calls, ['api repos/f2i-com/oaiy.com/releases/latest --jq .tag_name']);
    }
  });

  it('refuses one named after an older release, which the newest would replace, and says what to use', { skip: !bash && 'no bash on this machine' }, () => {
    const r = stamp('0.1.8-mac.1', 'v0.1.9');
    assert.equal(r.status, 1);
    assert.match(r.stdout, /::error::0\.1\.8-mac\.1 is named after 0\.1\.8, and the newest release is v0\.1\.9: it would be offered 0\.1\.9, which is older code than its own, and be replaced by it\. Name a test build after the newest release: 0\.1\.9-mac\.1/);
    // (numbers, not text: 0.1.9 is older than 0.1.10)
    assert.match(stamp('0.1.9-mac.2', 'v0.1.10').stdout, /older code than its own.*0\.1\.10-mac\.1/);
  });

  it('refuses one named after a release that is not out, which would never be offered it', { skip: !bash && 'no bash on this machine' }, () => {
    for (const [version, newest] of [['0.1.9-rc.1', 'v0.1.8'], ['0.1.10-mac.1', 'v0.1.9'], ['1.0.0-beta.1', 'v0.1.9']]) {
      const r = stamp(version, newest);
      assert.equal(r.status, 1, version);
      assert.match(r.stdout, new RegExp(`::error::${version.replace(/\./g, '\\.')} is named after [0-9.]+, which is not released \\(the newest release is ${newest.replace(/\./g, '\\.')}\\): it would never be offered`));
      assert.match(r.stdout, new RegExp(`Name a test build after the newest release: ${newest.slice(1).replace(/\./g, '\\.')}-mac\\.1`));
    }
  });

  it('leaves a release’s own version to the job that adds the image to its release, and asks GitHub nothing for it', { skip: !bash && 'no bash on this machine' }, () => {
    for (const newest of ['v0.1.8', null]) {
      const r = stamp('0.1.9', newest);
      assert.equal(r.status, 0, r.stdout + r.stderr);
      assert.deepEqual(r.calls, []);
    }
  });

  it('refuses no version at all, naming the newest release to name a test build after', { skip: !bash && 'no bash on this machine' }, () => {
    const r = stamp('', 'v0.1.9');
    assert.equal(r.status, 1);
    assert.match(r.stdout, /::error::no version to stamp: .*0\.1\.9-mac\.1, as v0\.1\.9 is the newest/);
    assert.match(stamp('', null).stdout, /::error::no version to stamp/);
  });

  it('refuses a test build when the newest release cannot be read, and what is not a version', { skip: !bash && 'no bash on this machine' }, () => {
    const unread = stamp('0.1.9-mac.1', null);
    assert.equal(unread.status, 1);
    assert.match(unread.stdout, /::error::the newest release could not be read, so the test build 0\.1\.9-mac\.1 cannot be held to it/);
    const marker = path.join(root, 'version-marker');
    for (const version of ['v0.1.9-mac.1', '0.1.9-', '0.01.9-mac.1', '0.1.9-mac 1', '0.1.9-mac.1+abc', `0.1.9-mac.1$(touch ${marker})`, '0.1.9-mac.1"; exit 0; "']) {
      const r = stamp(version, 'v0.1.9');
      assert.equal(r.status, 1, version);
      assert.match(r.stdout, /is not a version: a release's \(0\.1\.9\), or a test build's \(0\.1\.9-mac\.1\)/, version);
      assert.deepEqual(r.calls, [], version);
    }
    assert.equal(fs.existsSync(marker), false, 'nothing the input said was run');
  });
});
