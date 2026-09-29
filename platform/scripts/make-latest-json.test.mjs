// The update feed of a release (make-latest-json.mjs).
//
//   node --test platform/scripts/make-latest-json.test.mjs
//
// Each test makes a release folder as the release job finds it after the desktop
// legs' uploads are merged: the renamed installers and the `.sig` files beside them.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildFeed, FeedError, platformAssets, readSignature, writeFeed } from './make-latest-json.mjs';

const script = path.join(path.dirname(fileURLToPath(import.meta.url)), 'make-latest-json.mjs');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-latest-json-test-'));
after(() => fs.rmSync(root, { recursive: true, force: true }));

const VERSION = '0.1.0';
const SETUP = `oaiy-desktop-${VERSION}-windows-x64-setup.exe`;
const MSI = `oaiy-desktop-${VERSION}-windows-x64.msi`;
const APPIMAGE = `oaiy-desktop-${VERSION}-linux-x86_64.AppImage`;

/** What the Tauri CLI writes into a .sig: the base64 of a minisign signature file. Distinct per label, so a mix-up shows. */
function signature(label) {
  const text = `untrusted comment: signature from tauri secret key\nRUR${label.padEnd(40, 'A')}==\ntrusted comment: timestamp:1 file:${label}\n${label.padEnd(60, 'B')}==\n`;
  return Buffer.from(text).toString('base64');
}

let counter = 0;
/** A release folder; `skip` names files left out, `only` replaces the default set. */
function release({ skip = [], extra = {}, sigs = {} } = {}) {
  const dir = path.join(root, `release-${++counter}`);
  fs.mkdirSync(dir);
  const files = {
    [SETUP]: 'nsis installer bytes',
    [`${SETUP}.sig`]: signature('windows-nsis') + '\n',
    [MSI]: 'msi installer bytes',
    [`${MSI}.sig`]: signature('windows-msi'),
    [APPIMAGE]: 'appimage bytes',
    [`${APPIMAGE}.sig`]: signature('linux-appimage'),
    'SHA256SUMS.txt': 'not yet',
    'oaiy-cli-0.1.0.tar.gz': 'cli',
    ...extra,
    ...sigs,
  };
  for (const [name, content] of Object.entries(files)) {
    if (!skip.includes(name)) fs.writeFileSync(path.join(dir, name), content);
  }
  return dir;
}

const feedOf = (dir, options = {}) => buildFeed({ dir, version: VERSION, pubDate: '2026-10-01T02:03:04Z', ...options });

describe('the feed of a complete release', () => {
  it('is the Tauri updater format: version, notes, pub_date and a platform entry with signature and url', () => {
    const feed = feedOf(release());
    assert.deepEqual(Object.keys(feed), ['version', 'notes', 'pub_date', 'platforms']);
    assert.equal(feed.version, '0.1.0');
    assert.equal(typeof feed.notes, 'string');
    assert.ok(feed.notes.includes('https://github.com/f2i-com/oaiy.com/releases/tag/v0.1.0'));
    assert.equal(feed.pub_date, '2026-10-01T02:03:04Z');
    assert.deepEqual(Object.keys(feed.platforms).sort(), ['linux-x86_64', 'windows-x86_64']);
    for (const entry of Object.values(feed.platforms)) assert.deepEqual(Object.keys(entry).sort(), ['signature', 'url']);
  });

  it('carries the CONTENT of each .sig file as its signature, and never a signature meant for another file', () => {
    const feed = feedOf(release());
    assert.equal(feed.platforms['windows-x86_64'].signature, signature('windows-nsis'));
    assert.equal(feed.platforms['linux-x86_64'].signature, signature('linux-appimage'));
  });

  it('points at the release asset by its download URL, on the tag the release was published under', () => {
    const feed = feedOf(release());
    assert.equal(feed.platforms['windows-x86_64'].url, `https://github.com/f2i-com/oaiy.com/releases/download/v0.1.0/${SETUP}`);
    assert.equal(feed.platforms['linux-x86_64'].url, `https://github.com/f2i-com/oaiy.com/releases/download/v0.1.0/${APPIMAGE}`);
    // A release may be tagged without the "v" (the workflow takes both), and a fork publishes under its own name.
    const bare = feedOf(release(), { tag: '0.1.0', repo: 'someone/oaiy-fork' });
    assert.equal(bare.platforms['linux-x86_64'].url, `https://github.com/someone/oaiy-fork/releases/download/0.1.0/${APPIMAGE}`);
  });

  it('updates Windows from the NSIS installer and never from the MSI, whatever else is in the folder', () => {
    const feed = feedOf(release());
    assert.ok(feed.platforms['windows-x86_64'].url.endsWith('-setup.exe'));
    assert.ok(!JSON.stringify(feed).includes('.msi'));
    assert.ok(!JSON.stringify(feed).includes(signature('windows-msi')));
    // The MSI alone is no Windows update at all.
    assert.throws(() => feedOf(release({ skip: [SETUP, `${SETUP}.sig`] })), /Windows NSIS installer is missing/);
  });

  it('dates the feed in RFC 3339 (now, when no date is given) and rejects a date that is not', () => {
    const now = buildFeed({ dir: release(), version: VERSION });
    assert.match(now.pub_date, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    assert.ok(Math.abs(Date.parse(now.pub_date) - Date.now()) < 60_000);
    assert.throws(() => feedOf(release(), { pubDate: '2026-10-01' }), /RFC 3339/);
    assert.throws(() => feedOf(release(), { pubDate: 'yesterday' }), /RFC 3339/);
    assert.equal(feedOf(release(), { pubDate: '2026-10-01T12:00:00+10:00' }).pub_date, '2026-10-01T12:00:00+10:00');
  });

  it('takes the notes it is given', () => {
    assert.equal(feedOf(release(), { notes: 'Fixes calls.' }).notes, 'Fixes calls.');
  });
});

describe('a release that cannot update a platform is an error, not a shorter feed', () => {
  it('refuses a missing Windows installer, and names it', () => {
    assert.throws(() => feedOf(release({ skip: [SETUP] })), (e) => e instanceof FeedError && e.message.includes(SETUP) && e.message.includes('windows-x86_64'));
  });

  it('refuses a missing Linux AppImage', () => {
    assert.throws(() => feedOf(release({ skip: [APPIMAGE] })), (e) => e instanceof FeedError && e.message.includes(APPIMAGE));
  });

  it('refuses a missing signature for either platform', () => {
    assert.throws(() => feedOf(release({ skip: [`${SETUP}.sig`] })), (e) => e instanceof FeedError && e.message.includes(`${SETUP}.sig`) && /no signature/.test(e.message));
    assert.throws(() => feedOf(release({ skip: [`${APPIMAGE}.sig`] })), (e) => e instanceof FeedError && e.message.includes(`${APPIMAGE}.sig`));
  });

  it('refuses an empty installer, an empty signature and one that is not a signature', () => {
    assert.throws(() => feedOf(release({ extra: { [APPIMAGE]: '' } })), /AppImage is missing/);
    assert.throws(() => feedOf(release({ sigs: { [`${SETUP}.sig`]: '  \n' } })), /is empty/);
    assert.throws(() => feedOf(release({ sigs: { [`${SETUP}.sig`]: 'not base64 at all!' } })), /not base64/);
    assert.throws(() => feedOf(release({ sigs: { [`${APPIMAGE}.sig`]: Buffer.from('hello there').toString('base64') } })), /not a minisign signature/);
    assert.throws(() => feedOf(release({ sigs: { [`${APPIMAGE}.sig`]: Buffer.from('untrusted comment: only one line').toString('base64') } })), /not a minisign signature/);
  });

  it('refuses a folder that does not exist', () => {
    assert.throws(() => feedOf(path.join(root, 'nowhere')), /does not exist/);
  });
});

describe('what it takes as input', () => {
  it('needs a version of the form N.N.N, and a tag that is that version', () => {
    for (const bad of ['', '0.1', '0.01.0', '1.2.3-beta.1', 'v0.1.0', undefined]) {
      assert.throws(() => buildFeed({ dir: release(), version: bad }), /is not a version/, String(bad));
    }
    assert.throws(() => feedOf(release(), { tag: 'v0.2.0' }), /is not the version/);
    assert.throws(() => feedOf(release(), { tag: '../evil' }), /is not the version/);
  });

  it('needs a repository of the form owner/name, so the URL cannot be pointed elsewhere', () => {
    for (const bad of ['', 'nope', 'a/b/c', 'a b/c', 'evil.example/x?y']) {
      assert.throws(() => feedOf(release(), { repo: bad }), /is not a repository/, bad);
    }
  });

  it('lists the platforms the desktop can update, by the feed keys the desktop looks up', () => {
    assert.deepEqual(platformAssets('1.2.3').map((p) => p.key), ['windows-x86_64', 'linux-x86_64']);
    assert.ok(platformAssets('1.2.3').every((p) => p.asset.includes('1.2.3')));
  });

  it('reads a signature file with a trailing newline as its content', () => {
    const file = path.join(root, 'one.sig');
    fs.writeFileSync(file, `${signature('x')}\r\n`);
    assert.equal(readSignature(file, 'x'), signature('x'));
  });
});

describe('writing latest.json', () => {
  it('writes the feed beside the release files by default, as JSON a Tauri updater parses', () => {
    const dir = release();
    const out = writeFeed({ dir, version: VERSION, pubDate: '2026-10-01T02:03:04Z' });
    assert.equal(out, path.join(dir, 'latest.json'));
    const text = fs.readFileSync(out, 'utf8');
    assert.ok(text.endsWith('\n'));
    assert.deepEqual(JSON.parse(text), feedOf(dir));
  });

  it('writes nothing when the release is incomplete', () => {
    const dir = release({ skip: [`${APPIMAGE}.sig`] });
    assert.throws(() => writeFeed({ dir, version: VERSION }), FeedError);
    assert.equal(fs.existsSync(path.join(dir, 'latest.json')), false);
  });
});

describe('as the release job runs it', () => {
  const run = (...args) => spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });

  it('writes latest.json and says so', () => {
    const dir = release();
    const result = run('--dir', dir, '--version', VERSION, '--tag', 'v0.1.0', '--repo', 'f2i-com/oaiy.com', '--pub-date', '2026-10-01T02:03:04Z');
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /wrote .*latest\.json/);
    const feed = JSON.parse(fs.readFileSync(path.join(dir, 'latest.json'), 'utf8'));
    assert.equal(feed.platforms['windows-x86_64'].signature, signature('windows-nsis'));
  });

  it('fails the step with a message that names what is missing, and writes no feed', () => {
    const dir = release({ skip: [`${SETUP}.sig`] });
    const result = run('--dir', dir, '--version', VERSION);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /::error::update feed \(latest\.json\) NOT written: there is no signature/);
    assert.equal(fs.existsSync(path.join(dir, 'latest.json')), false);
  });

  it('reads the notes from a file and the output path from --out', () => {
    const dir = release();
    const notes = path.join(root, 'notes.txt');
    fs.writeFileSync(notes, 'Calls keep their audio.\n');
    const out = path.join(root, 'elsewhere.json');
    const result = run('--dir', dir, '--version', VERSION, '--notes-file', notes, '--out', out);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(fs.readFileSync(out, 'utf8')).notes, 'Calls keep their audio.');
    assert.equal(fs.existsSync(path.join(dir, 'latest.json')), false);
  });

  it('refuses arguments it does not know, and a missing folder or version', () => {
    assert.equal(run('--dir', release(), '--version', VERSION, '--wat', 'x').status, 1);
    assert.equal(run('--version', VERSION).status, 1);
    assert.equal(run('--dir', release()).status, 1);
  });
});
