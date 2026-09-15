/**
 * The ZIPP installs take a release only when every layer checks, for both
 * bundles: each zip against ZIPP's SHA256SUMS, each file against its bundle's
 * own, BUILD-INFO.txt and the module itself against that variant of that
 * release, and both bundles against one commit.
 *
 *   node --test scripts/fetch-zipp-release.test.mjs
 *
 * Fixture releases are built in temp folders with a stand-in glue that reports
 * whatever profile a case needs, and SHA256SUMS rebuilt per case, so a case
 * trips only the refusal it is about. One case repacks the engines ui/vendor
 * holds, so the real glue and modules pass the same checks (install them
 * first; outside CI that case is skipped, saying why, when there are none).
 * Releases come from a folder (ZIPP_RELEASE_DIR) through the command line, or
 * from a stubbed fetch, in process or preloaded; nothing here reaches GitHub.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  BUNDLE_FILES, CURATED_NOTICES, REPOSITORY, VARIANTS, VENDOR_DIR, ZippReleaseError,
  bundleName, checkInstalls, checkInstallsOnline, ensureInstalls, installRelease, installedFiles, readCuratedNotices, removeStaleLock, resolveRelease, sha256, unzip, verifyRelease, withInstallLock, writeInstalls,
} from './fetch-zipp-release.mjs';
import { NOTICE_SOURCES, noticesRecord, regenerate } from './regen-zipp-notices.mjs';

const SCRIPT = fileURLToPath(new URL('./fetch-zipp-release.mjs', import.meta.url));
const VERSION = '1.2.3';
const TAG = `v${VERSION}`;
const COMMIT = 'c'.repeat(40);
const [WEB, PYTHON] = VARIANTS;
const quiet = () => {};

function tempDir(t, prefix) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}
const outDir = (t) => path.join(tempDir(t, 'zipp-install-'), 'vendor');
const escaped = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const readSource = (dir) => JSON.parse(fs.readFileSync(path.join(dir, 'SOURCE.json'), 'utf8'));
const CURATED_RECORD = path.join(path.dirname(CURATED_NOTICES), 'SOURCE.json');

/**
 * A zip as ZIPP's release job writes one: a top folder entry, then deflated
 * files (stored when `deflate` is false). `entries` are [name, bytes] pairs;
 * `flags` is every entry's general-purpose bit flag.
 */
function writeZip(entries, { deflate = true, flags = 0 } = {}) {
  const list = [...entries];
  const parts = [];
  const central = [];
  let offset = 0;
  for (const [name, data] of list) {
    const nameBytes = Buffer.from(name);
    const method = deflate && data.length ? 8 : 0;
    const body = method ? zlib.deflateRawSync(data) : data;
    const crc = zlib.crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(flags, 6);
    local.writeUInt16LE(method, 8);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBytes.length, 26);
    parts.push(local, nameBytes, body);
    const entry = Buffer.alloc(46);
    entry.writeUInt32LE(0x02014b50, 0);
    entry.writeUInt16LE(20, 4);
    entry.writeUInt16LE(20, 6);
    entry.writeUInt16LE(flags, 8);
    entry.writeUInt16LE(method, 10);
    entry.writeUInt32LE(crc, 16);
    entry.writeUInt32LE(body.length, 20);
    entry.writeUInt32LE(data.length, 24);
    entry.writeUInt16LE(nameBytes.length, 28);
    entry.writeUInt32LE(offset, 42);
    central.push(entry, nameBytes);
    offset += local.length + nameBytes.length + body.length;
  }
  const directory = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(list.length, 8);
  end.writeUInt16LE(list.length, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...parts, directory, end]);
}

/** A module that describes itself as `profile`, standing in for the engine where what it reports is the point. */
const standInGlue = (profile) => Buffer.from(`export function initSync() {}\nexport function zippProfile() { return ${JSON.stringify(JSON.stringify(profile))}; }\n`);
const buildInfo = (variant, { version, commit }) => Buffer.from(`version=${version}\ncommit=${commit}\nrustc=rustc 1.92.0 (ded5c06cf 2025-12-08)\nwasm-bindgen=wasm-bindgen 0.2.126\nvariant=${variant.variant}\nlanguages=${JSON.stringify(variant.languages)}\nstack-bytes=${variant.stackBytes}\ntarget=wasm32-unknown-unknown\n`);
const setBuildInfo = (files, key, value) => files.set('BUILD-INFO.txt', Buffer.from(files.get('BUILD-INFO.txt').toString('utf8').replace(new RegExp(`^${key}=.*$`, 'm'), `${key}=${value}`)));
const setGlue = (files, change) => files.set('zipp_wasm.js', standInGlue(change(JSON.parse(JSON.parse(/return (".*");/.exec(files.get('zipp_wasm.js').toString('utf8'))[1])))));

function fixtureFiles(variant, { version, commit }) {
  return new Map([
    ['zipp_wasm.js', standInGlue({ version, features: ['safe-sandbox', 'meter-only'], languages: variant.languages, source: { sha: commit } })],
    ['zipp_wasm.d.ts', Buffer.from(`export class Engine {} // ${variant.suffix}\n`)],
    ['zipp_wasm_bg.wasm', Buffer.from(`\0asm stand-in ${variant.suffix} module`)],
    ['zipp_wasm_bg.wasm.d.ts', Buffer.from(`export const memory: WebAssembly.Memory; // ${variant.suffix}\n`)],
    ['LICENSE-APACHE', Buffer.from('Apache License, Version 2.0\n')],
    ['BUILD-INFO.txt', buildInfo(variant, { version, commit })],
    ['PROFILE.json', Buffer.from(`${JSON.stringify({ engine: 'zipp-wasm', version, languages: variant.languages }, null, 2)}\n`)],
    // Listed by the bundle's SHA256SUMS, never installed.
    ['README.md', Buffer.from('# zipp-wasm\n')],
    ['host-sdk/zipp-host.mjs', Buffer.from('export {};\n')],
    ...(variant.needsNotices ? [['gpu-lab/LICENSE', Buffer.from('gpu-lab licence\n')], ['docs/TORCH_COMPATIBILITY.md', Buffer.from('torch\n')]] : []),
  ]);
}

/** A bundle's files as the install in ui/vendor holds them (checked by the caller). */
function vendorBundleFiles(variant) {
  const dir = path.join(VENDOR_DIR, variant.folder);
  return new Map(BUNDLE_FILES.filter((n) => n !== 'SHA256SUMS').map((n) => [n, fs.readFileSync(path.join(dir, n))]));
}

/**
 * A release folder holding SHA256SUMS and both bundle zips. Per variant suffix,
 * `edit` changes the bundle before its SHA256SUMS is written, `innerSums` that
 * text, `tamper` the bundle after it; `repack` packs the same files again
 * after the top-level sums are written, a valid zip whose bytes are not the
 * ones listed.
 */
function releaseFolder(t, { version = VERSION, commit = COMMIT, edit = {}, innerSums = {}, tamper = {}, repack, real = false } = {}) {
  const zips = {};
  const bundles = {};
  for (const variant of VARIANTS) {
    const files = real ? vendorBundleFiles(variant) : fixtureFiles(variant, { version, commit });
    edit[variant.suffix]?.(files);
    let sums = `${[...files].map(([name, bytes]) => `${sha256(bytes)}  ${name}`).join('\n')}\n`;
    if (innerSums[variant.suffix]) sums = innerSums[variant.suffix](sums);
    files.set('SHA256SUMS', Buffer.from(sums));
    tamper[variant.suffix]?.(files);
    const prefix = bundleName(version, variant);
    const entries = new Map([[`${prefix}/`, Buffer.alloc(0)], ...[...files].map(([name, bytes]) => [`${prefix}/${name}`, bytes])]);
    zips[variant.suffix] = writeZip(entries);
    bundles[variant.suffix] = { files, entries };
  }
  const top = Buffer.from(`${'a'.repeat(64)}  zipp-${version}-x86_64-unknown-linux-gnu.tar.gz\n${VARIANTS.map((v) => `${sha256(zips[v.suffix])}  ${bundleName(version, v)}.zip`).join('\n')}\n`);
  if (repack) zips[repack] = writeZip(bundles[repack].entries, { deflate: false });
  const dir = tempDir(t, 'zipp-release-fixture-');
  fs.writeFileSync(path.join(dir, 'SHA256SUMS'), top);
  for (const variant of VARIANTS) fs.writeFileSync(path.join(dir, `${bundleName(version, variant)}.zip`), zips[variant.suffix]);
  return { dir, top, zips, bundles, release: `v${version}` };
}

const cleanEnv = () => Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(ZIPP_|CI$|GITHUB_ACTIONS$)/i.test(key)));
const cli = (args, env = {}) => spawnSync(process.execPath, [SCRIPT, ...args], { env: { ...cleanEnv(), ...env }, encoding: 'utf8' });

function install(t, folder, { env = {}, args = [], out = outDir(t) } = {}) {
  return { out, run: cli(args, { ZIPP_RELEASE_DIR: folder.dir, ZIPP_OUT: out, ...env }) };
}

function refused({ run, out }, pattern) {
  assert.equal(run.status, 1, `${run.stdout}\n${run.stderr}`);
  assert.match(run.stderr, pattern);
  assert.doesNotMatch(run.stderr, /^\s+at /m, 'a refusal, not a stack trace');
  if (out) for (const variant of VARIANTS) assert.ok(!fs.existsSync(path.join(out, variant.folder)), `nothing is installed into ${variant.folder}`);
}

const editText = (file, change) => fs.writeFileSync(file, change(fs.readFileSync(file, 'utf8')));
const editSource = (dir, change) => editText(path.join(dir, 'SOURCE.json'), (text) => {
  const record = JSON.parse(text);
  change(record);
  return `${JSON.stringify(record, null, 2)}\n`;
});
/** Brings an install's own SHA256SUMS line for `name` in step with the file, so only the check a case is about can trip. */
const resum = (dir, name) => editText(path.join(dir, 'SHA256SUMS'), (text) => text.replace(new RegExp(`^[0-9a-f]{64}(  ${escaped(name)})$`, 'm'), `${sha256(fs.readFileSync(path.join(dir, name)))}$1`));
const serve = (responses) => {
  const calls = [];
  const fetch = async (url) => {
    calls.push(url);
    return responses.has(url) ? new Response(responses.get(url)) : new Response('', { status: 404 });
  };
  return { calls, fetch };
};

function spawnCli(args, env, preload = []) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [...preload, SCRIPT, ...args], { env: { ...cleanEnv(), ...env } });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
}

/** The pid of a process that has exited: what a killed install leaves in its lock. */
const deadPid = () => spawnSync(process.execPath, ['-e', '']).pid;

/** The pid of a process that runs until the test ends: an install still holding its lock. */
function livePid(t) {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });
  t.after(() => child.kill());
  return child.pid;
}

/** Every file and folder under `dir`, files by SHA-256. */
function snapshot(dir) {
  const tree = {};
  for (const name of fs.readdirSync(dir, { recursive: true })) {
    const file = path.join(dir, name);
    tree[name.replaceAll('\\', '/')] = fs.statSync(file).isDirectory() ? 'folder' : sha256(fs.readFileSync(file));
  }
  return tree;
}

/** A module for `node --import` that stands in for GitHub: `served` maps URLs to bytes, anything else is a 404. */
function fetchStub(dir, name, served) {
  const file = path.join(dir, name);
  const table = Object.fromEntries([...served].map(([url, bytes]) => [url, Buffer.from(bytes).toString('base64')]));
  fs.writeFileSync(file, `const served = ${JSON.stringify(table)};\nglobalThis.fetch = async (url) => (url in served ? new Response(Buffer.from(served[url], 'base64')) : new Response('', { status: 404 }));\n`);
  return ['--import', pathToFileURL(file).href];
}
const releaseAsset = (name) => `${REPOSITORY}/releases/download/${TAG}/${name}`;
/** What GitHub serves for a fixture release: its SHA256SUMS and both bundles. */
const published = (folder) => new Map([[releaseAsset('SHA256SUMS'), folder.top], ...VARIANTS.map((v) => [releaseAsset(`${bundleName(VERSION, v)}.zip`), folder.zips[v.suffix]])]);

test('a release that checks all the way down installs both bundles as exactly their install sets, and --check passes', (t) => {
  const folder = releaseFolder(t);
  // No tag: the release the folder holds.
  const { out, run } = install(t, folder);
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], 'no stage, previous install or lock is left behind');
  for (const variant of VARIANTS) {
    const dir = path.join(out, variant.folder);
    const record = readSource(dir);
    const { files } = folder.bundles[variant.suffix];
    assert.deepEqual(fs.readdirSync(dir).sort(), installedFiles(record.notices).sort(), `${variant.folder}: unshipped SHA256SUMS entries (README.md, docs/, gpu-lab/, host-sdk/) stay behind`);
    assert.deepEqual(
      { release: record.release, version: record.version, revision: record.revision, build: record.build, bundle: record.bundle, bundleSha256: record.bundleSha256, sumsSha256: record.sumsSha256, variant: record.variant, languages: record.languages, stackBytes: record.stackBytes, rustc: record.rustc, wasmBindgen: record.wasmBindgen, license: record.license, artifact: record.artifact, sha256: record.sha256, glueSha256: record.glueSha256, repository: record.repository },
      { release: TAG, version: VERSION, revision: COMMIT, build: 'release', bundle: `${bundleName(VERSION, variant)}.zip`, bundleSha256: sha256(folder.zips[variant.suffix]), sumsSha256: sha256(folder.top), variant: variant.variant, languages: variant.languages, stackBytes: variant.stackBytes, rustc: 'rustc 1.92.0 (ded5c06cf 2025-12-08)', wasmBindgen: '0.2.126', license: 'Apache-2.0', artifact: 'zipp_wasm_bg.wasm', sha256: sha256(files.get('zipp_wasm_bg.wasm')), glueSha256: sha256(files.get('zipp_wasm.js')), repository: REPOSITORY },
    );
    assert.ok(fs.readFileSync(path.join(dir, 'RELEASE-SHA256SUMS')).equals(folder.top), 'ZIPP\'s top-level SHA256SUMS, verbatim');
    for (const name of BUNDLE_FILES) assert.ok(fs.readFileSync(path.join(dir, name)).equals(files.get(name)), `${variant.folder}/${name} byte for byte`);
  }
  // Only the Python engine compiles in RustPython and Unicode data; the bundle ships no notices, so the curated copy is installed and says so.
  assert.equal(readSource(path.join(out, WEB.folder)).notices, null);
  const notices = readSource(path.join(out, PYTHON.folder)).notices;
  assert.deepEqual(notices, { file: 'THIRD_PARTY_LICENSES.txt', source: 'oaiy-curated', sha256: sha256(fs.readFileSync(CURATED_NOTICES)) });
  assert.ok(fs.readFileSync(path.join(out, PYTHON.folder, 'THIRD_PARTY_LICENSES.txt')).equals(fs.readFileSync(CURATED_NOTICES)));
  const check = cli(['--check'], { ZIPP_OUT: out });
  assert.equal(check.status, 0, `${check.stdout}\n${check.stderr}`);
  assert.match(check.stdout, new RegExp(`ZIPP ${escaped(TAG)} \\(cccccccc\\) in .* checks\\.`));
});

test('the engines ui/vendor holds, real glue and modules, pass the same checks when repacked', async (t) => {
  let installed;
  try {
    installed = checkInstalls(VENDOR_DIR);
  } catch (error) {
    const why = `ui/vendor holds no ZIPP install that checks; run node scripts/fetch-zipp-release.mjs (or npm test in ui) first`;
    // CI installs before it tests; a fresh clone has nothing to repack yet.
    if (!process.env.CI) return t.skip(why);
    assert.fail(`${why}\n${error.message}`);
  }
  const folder = releaseFolder(t, { version: installed.release.slice(1), real: true });
  const installs = await verifyRelease({ release: installed.release, sums: folder.top, zips: folder.zips });
  for (const { variant, source } of installs) {
    const was = installed.sources[variant.suffix];
    assert.deepEqual({ revision: source.revision, sha256: source.sha256, glueSha256: source.glueSha256, languages: source.languages }, { revision: was.revision, sha256: was.sha256, glueSha256: was.glueSha256, languages: was.languages }, variant.folder);
  }
});

test('notices a bundle ships itself are taken from it and recorded as the release\'s, for either bundle', (t) => {
  const withNotices = (files) => files.set('THIRD_PARTY_LICENSES.txt', Buffer.from('ZIPP\'s own notices\n'));
  const { out, run } = install(t, releaseFolder(t, { edit: { web: withNotices, 'web-python': withNotices } }));
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  for (const variant of VARIANTS) assert.deepEqual(readSource(path.join(out, variant.folder)).notices, { file: 'THIRD_PARTY_LICENSES.txt', source: 'zipp-release', sha256: sha256(Buffer.from('ZIPP\'s own notices\n')) });
  assert.equal(cli(['--check'], { ZIPP_OUT: out }).status, 0);
});

test('a zip that disagrees with the release\'s SHA256SUMS is refused, even one holding the very same files', async (t) => {
  for (const variant of VARIANTS) {
    const folder = releaseFolder(t, { repack: variant.suffix });
    // Every file inside checks, so only ZIPP's top-level SHA256SUMS can tell.
    const inside = unzip(folder.zips[variant.suffix]);
    for (const [name, bytes] of folder.bundles[variant.suffix].entries) if (!name.endsWith('/')) assert.ok(inside.get(name).equals(bytes), `${name} is in the repacked zip unchanged`);
    refused(install(t, folder), new RegExp(`${escaped(bundleName(VERSION, variant))}\\.zip has sha256 [0-9a-f]{64}; the ${escaped(TAG)} SHA256SUMS says [0-9a-f]{64}`));
    // verifyRelease holds each zip to the SHA256SUMS itself, whoever loaded it.
    await assert.rejects(verifyRelease({ release: TAG, sums: folder.top, zips: folder.zips }), (error) => error instanceof ZippReleaseError && error.message === `${bundleName(VERSION, variant)}.zip does not match the ${TAG} SHA256SUMS`);
  }
  const flipped = releaseFolder(t);
  const file = path.join(flipped.dir, `${bundleName(VERSION, PYTHON)}.zip`);
  const bytes = fs.readFileSync(file);
  bytes[bytes.length >> 1] ^= 0x01;
  fs.writeFileSync(file, bytes);
  refused(install(t, flipped), /web-python\.zip has sha256 [0-9a-f]{64}; the v1\.2\.3 SHA256SUMS says/);
});

test('a file that disagrees with its bundle\'s SHA256SUMS is refused', (t) => {
  const tamper = (files) => files.set('zipp_wasm.d.ts', Buffer.concat([files.get('zipp_wasm.d.ts'), Buffer.from(' ')]));
  refused(install(t, releaseFolder(t, { tamper: { web: tamper } })), /zipp_wasm\.d\.ts in zipp-wasm-1\.2\.3-web\.zip does not match the bundle's SHA256SUMS/);
  refused(install(t, releaseFolder(t, { tamper: { 'web-python': tamper } })), /zipp_wasm\.d\.ts in zipp-wasm-1\.2\.3-web-python\.zip does not match the bundle's SHA256SUMS/);
});

test('a file the install takes that its bundle\'s SHA256SUMS does not list, or one it lists that the bundle lacks, is refused', (t) => {
  const innerSums = (text) => text.split('\n').filter((line) => !line.endsWith('  LICENSE-APACHE')).join('\n');
  refused(install(t, releaseFolder(t, { innerSums: { web: innerSums } })), /LICENSE-APACHE in zipp-wasm-1\.2\.3-web\.zip is not listed in the bundle's SHA256SUMS/);
  const ghost = (text) => `${text}${'f'.repeat(64)}  docs/GHOST.md\n`;
  refused(install(t, releaseFolder(t, { innerSums: { 'web-python': ghost } })), /zipp-wasm-1\.2\.3-web-python\.zip lists docs\/GHOST\.md in its SHA256SUMS but does not carry it/);
});

test('each bundle has to be its own variant by what BUILD-INFO.txt says, whatever the zip is called', (t) => {
  const asWeb = (files) => {
    setBuildInfo(files, 'variant', 'javascript');
    setBuildInfo(files, 'languages', '["javascript"]');
    setBuildInfo(files, 'stack-bytes', '1048576');
  };
  refused(install(t, releaseFolder(t, { edit: { 'web-python': asWeb } })), /variant is javascript, not javascript-python/);
  const asPython = (files) => {
    setBuildInfo(files, 'variant', 'javascript-python');
    setBuildInfo(files, 'languages', '["javascript","python"]');
    setBuildInfo(files, 'stack-bytes', '16777216');
  };
  refused(install(t, releaseFolder(t, { edit: { web: asPython } })), /variant is javascript-python, not javascript \(the web bundle\)/);
  refused(install(t, releaseFolder(t, { edit: { 'web-python': (files) => setBuildInfo(files, 'stack-bytes', '1048576') } })), /BUILD-INFO\.txt stack-bytes is 1048576, not 16777216/);
  refused(install(t, releaseFolder(t, { edit: { web: (files) => setBuildInfo(files, 'languages', '["javascript","python"]') } })), /BUILD-INFO\.txt languages are \["javascript","python"\], not \["javascript"\]/);
  // The two zips under each other's names, with a SHA256SUMS to match: named for what it holds, not what it is called.
  const swapped = releaseFolder(t);
  fs.writeFileSync(path.join(swapped.dir, `${bundleName(VERSION, WEB)}.zip`), swapped.zips['web-python']);
  fs.writeFileSync(path.join(swapped.dir, `${bundleName(VERSION, PYTHON)}.zip`), swapped.zips.web);
  fs.writeFileSync(path.join(swapped.dir, 'SHA256SUMS'), `${sha256(swapped.zips['web-python'])}  ${bundleName(VERSION, WEB)}.zip\n${sha256(swapped.zips.web)}  ${bundleName(VERSION, PYTHON)}.zip\n`);
  refused(install(t, swapped), /zipp-wasm-1\.2\.3-web\.zip has no zipp-wasm-1\.2\.3-web\/ folder; its files are under zipp-wasm-1\.2\.3-web-python \(is it another bundle\?\)/);
});

test('bundles that name different commits are not one release, even when each checks on its own', (t) => {
  const other = 'd'.repeat(40);
  const edit = { web: (files) => {
    setBuildInfo(files, 'commit', other);
    setGlue(files, (profile) => ({ ...profile, source: { sha: other } }));
  } };
  refused(install(t, releaseFolder(t, { edit })), new RegExp(`the ZIPP v1\\.2\\.3 bundles name different commits: web d{40}, web-python c{40}`));
});

test('each module has to describe itself as its variant of the release: version, languages, the safe sandbox, the commit', (t) => {
  const saying = (suffix, change) => releaseFolder(t, { edit: { [suffix]: (files) => setGlue(files, (profile) => ({ ...profile, ...change })) } });
  refused(install(t, saying('web-python', { version: '9.9.9' })), /zippProfile\(\) version is 9\.9\.9, not 1\.2\.3/);
  refused(install(t, saying('web-python', { languages: ['javascript'] })), /zippProfile\(\) languages are \["javascript"\], not \["javascript","python"\]/);
  refused(install(t, saying('web', { languages: ['javascript', 'python'] })), /zippProfile\(\) languages are \["javascript","python"\], not \["javascript"\]/);
  refused(install(t, saying('web', { features: ['meter-only'] })), /zippProfile\(\) lacks the safe-sandbox feature/);
  refused(install(t, saying('web', { source: { sha: 'b'.repeat(40) } })), /zippProfile\(\) source\.sha is b{40}, not the BUILD-INFO\.txt commit c{40}/);
  const broken = releaseFolder(t, { edit: { web: (files) => files.set('zipp_wasm.js', Buffer.from('export function initSync() { throw new Error("trap"); }\n')) } });
  refused(install(t, broken), /the zipp-wasm-1\.2\.3-web\.zip engine does not load: trap/);
});

test('a PROFILE.json of another version, or a BUILD-INFO.txt commit that is not a full commit, is refused', (t) => {
  const edit = (files) => files.set('PROFILE.json', Buffer.from(files.get('PROFILE.json').toString('utf8').replace(`"version": "${VERSION}"`, '"version": "9.9.9"')));
  refused(install(t, releaseFolder(t, { edit: { web: edit } })), /PROFILE\.json says version 9\.9\.9/);
  refused(install(t, releaseFolder(t, { edit: { 'web-python': (files) => setBuildInfo(files, 'commit', 'cccccccc') } })), /not a full 40-hex commit/);
});

test('ZIPP_SUMS_SHA256 pins the release\'s SHA256SUMS', (t) => {
  refused(install(t, releaseFolder(t), { env: { ZIPP_SUMS_SHA256: '0'.repeat(64) } }), /ZIPP_SUMS_SHA256 says 0{64}/);
  refused({ run: cli(['--ensure'], { ZIPP_SUMS_SHA256: 'not-a-digest' }) }, /ZIPP_SUMS_SHA256 must be a sha256/);
  const folder = releaseFolder(t);
  assert.equal(install(t, folder, { env: { ZIPP_SUMS_SHA256: sha256(folder.top).toUpperCase() } }).run.status, 0);
});

test('a SHA256SUMS other than ZIPP_SUMS_SHA256 is refused before either bundle is downloaded', async (t) => {
  const folder = releaseFolder(t);
  const { calls, fetch } = serve(new Map([[`${REPOSITORY}/releases/download/${TAG}/SHA256SUMS`, folder.top]]));
  const out = outDir(t);
  await assert.rejects(
    installRelease({ vendorDir: out, tag: TAG, expectSumsSha256: '0'.repeat(64), cacheDir: tempDir(t, 'zipp-cache-'), fetch, log: quiet }),
    (error) => error instanceof ZippReleaseError && /ZIPP_SUMS_SHA256 says 0{64}/.test(error.message),
  );
  assert.deepEqual(calls, [`${REPOSITORY}/releases/download/${TAG}/SHA256SUMS`], 'the release\'s SHA256SUMS and nothing else');
  for (const variant of VARIANTS) assert.ok(!fs.existsSync(path.join(out, variant.folder)), `nothing is installed into ${variant.folder}`);
});

test('a folder or SHA256SUMS without both bundles is refused', (t) => {
  for (const variant of VARIANTS) {
    const folder = releaseFolder(t);
    fs.rmSync(path.join(folder.dir, `${bundleName(VERSION, variant)}.zip`));
    refused(install(t, folder), new RegExp(`holds no ${escaped(bundleName(VERSION, variant))}\\.zip`));
  }
  const onlyPython = releaseFolder(t);
  editText(path.join(onlyPython.dir, 'SHA256SUMS'), (text) => text.split('\n').filter((line) => !line.endsWith('-web.zip')).join('\n'));
  refused(install(t, onlyPython), /does not list zipp-wasm-1\.2\.3-web\.zip/);
  refused(install(t, releaseFolder(t), { args: ['v1.2.4'] }), /does not list zipp-wasm-1\.2\.4-web\.zip or zipp-wasm-1\.2\.4-web-python\.zip/);
});

test('an install replaces both whole folders, so a stale file does not survive', (t) => {
  const out = outDir(t);
  for (const variant of VARIANTS) {
    fs.mkdirSync(path.join(out, variant.folder), { recursive: true });
    fs.writeFileSync(path.join(out, variant.folder, 'README.md'), 'a committed engine\'s readme');
    fs.writeFileSync(path.join(out, variant.folder, 'UPSTREAM-SHA256SUMS'), 'stale');
  }
  const { run } = install(t, releaseFolder(t), { out });
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  for (const variant of VARIANTS) assert.deepEqual(fs.readdirSync(path.join(out, variant.folder)).sort(), installedFiles(readSource(path.join(out, variant.folder)).notices).sort());
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], 'neither the stages nor the previous installs are left behind');
});

test('--check fails after a one-byte edit, on a file its bundle\'s SHA256SUMS does not list, and on installs of two releases', (t) => {
  const { out, run } = install(t, releaseFolder(t));
  assert.equal(run.status, 0, run.stderr);
  const dir = path.join(out, WEB.folder);
  const file = path.join(dir, 'zipp_wasm_bg.wasm.d.ts');
  const original = fs.readFileSync(file);
  const edited = Buffer.from(original);
  edited[0] ^= 0x01;
  fs.writeFileSync(file, edited);
  refused({ run: cli(['--check'], { ZIPP_OUT: out }) }, /zipp-wasm does not check[\s\S]*zipp_wasm_bg\.wasm\.d\.ts does not match the bundle's SHA256SUMS/);
  fs.writeFileSync(file, original);
  editText(path.join(dir, 'SHA256SUMS'), (text) => text.split('\n').filter((line) => !line.endsWith('  zipp_wasm_bg.wasm.d.ts')).join('\n'));
  refused({ run: cli(['--check'], { ZIPP_OUT: out }) }, /zipp_wasm_bg\.wasm\.d\.ts is not listed in the bundle's SHA256SUMS/);

  // Each folder checks on its own; together they are not one ZIPP.
  const older = install(t, releaseFolder(t, { version: '1.2.2', commit: 'e'.repeat(40) }));
  const newer = install(t, releaseFolder(t));
  assert.equal(older.run.status + newer.run.status, 0);
  fs.rmSync(path.join(newer.out, PYTHON.folder), { recursive: true });
  fs.cpSync(path.join(older.out, PYTHON.folder), path.join(newer.out, PYTHON.folder), { recursive: true });
  refused({ run: cli(['--check'], { ZIPP_OUT: newer.out }) }, /are not one release[\s\S]*release differs: zipp-wasm v1\.2\.3, zipp-wasm-python v1\.2\.2[\s\S]*revision differs/);
  // The python engine copied over the web one is not the web install.
  const swapped = install(t, releaseFolder(t));
  fs.copyFileSync(path.join(swapped.out, PYTHON.folder, 'zipp_wasm_bg.wasm'), path.join(swapped.out, WEB.folder, 'zipp_wasm_bg.wasm'));
  refused({ run: cli(['--check'], { ZIPP_OUT: swapped.out }) }, /zipp_wasm_bg\.wasm does not match the bundle's SHA256SUMS[\s\S]*zipp_wasm_bg\.wasm is not the recorded/);
});

test('--check holds each install to everything its SOURCE.json records, one invariant at a time', (t) => {
  const { out: pristine, run } = install(t, releaseFolder(t));
  assert.equal(run.status, 0, run.stderr);
  assert.equal(checkInstalls(pristine).release, TAG);
  const E64 = 'e'.repeat(64);
  const E40 = 'e'.repeat(40);
  const setInfo = (key, value) => (dir) => {
    editText(path.join(dir, 'BUILD-INFO.txt'), (text) => text.replace(new RegExp(`^${key}=.*$`, 'm'), `${key}=${value}`));
    resum(dir, 'BUILD-INFO.txt');
  };
  const cases = [
    [PYTHON, 'ZIPP\'s SHA256SUMS edited', (dir) => fs.appendFileSync(path.join(dir, 'RELEASE-SHA256SUMS'), '\n'), /RELEASE-SHA256SUMS has sha256 [0-9a-f]{64}, not the recorded [0-9a-f]{64}/],
    [WEB, 'a recorded bundle digest', (dir) => editSource(dir, (s) => void (s.bundleSha256 = E64)), /RELEASE-SHA256SUMS does not list zipp-wasm-1\.2\.3-web\.zip with the recorded e{64}/],
    [WEB, 'a recorded revision', (dir) => editSource(dir, (s) => void (s.revision = E40)), /BUILD-INFO\.txt commit c{40} is not the recorded revision e{40}/],
    [PYTHON, 'a recorded stack', (dir) => editSource(dir, (s) => void (s.stackBytes = 1048576)), /SOURCE\.json records javascript-python \["javascript","python"\] with a 1048576-byte stack, not the web-python build/],
    [WEB, 'the web install recorded as Python', (dir) => editSource(dir, (s) => void (s.languages = ['javascript', 'python'])), /not the web build/],
    [PYTHON, 'a recorded engine digest', (dir) => editSource(dir, (s) => void (s.sha256 = E64)), /zipp_wasm_bg\.wasm is not the recorded e{64}/],
    [WEB, 'a recorded glue digest', (dir) => editSource(dir, (s) => void (s.glueSha256 = E64)), /zipp_wasm\.js is not the recorded e{64}/],
    [PYTHON, 'a recorded notices digest', (dir) => editSource(dir, (s) => void (s.notices.sha256 = E64)), /THIRD_PARTY_LICENSES\.txt is not the recorded e{64}/],
    [PYTHON, 'notices of unknown origin', (dir) => editSource(dir, (s) => void (s.notices.source = 'elsewhere')), /SOURCE\.json notices .* are not recorded for the web-python bundle/],
    [PYTHON, 'no notices recorded', (dir) => {
      editSource(dir, (s) => void (s.notices = null));
      fs.rmSync(path.join(dir, 'THIRD_PARTY_LICENSES.txt'));
    }, /SOURCE\.json notices null are not recorded for the web-python bundle/],
    [WEB, 'curated notices in the web install', (dir) => editSource(dir, (s) => void (s.notices = { file: 'THIRD_PARTY_LICENSES.txt', source: 'oaiy-curated', sha256: E64 })), /THIRD_PARTY_LICENSES\.txt is missing[\s\S]*notices .* are not recorded for the web bundle/],
    [PYTHON, 'curated notices recorded as the release\'s', (dir) => editSource(dir, (s) => void (s.notices.source = 'zipp-release')), /THIRD_PARTY_LICENSES\.txt is not listed in the bundle's SHA256SUMS/],
    [WEB, 'another repository', (dir) => editSource(dir, (s) => void (s.repository = 'https://example.com/zipp')), /SOURCE\.json repository is https:\/\/example\.com\/zipp/],
    [WEB, 'a local build', (dir) => editSource(dir, (s) => void (s.build = 'local')), /SOURCE\.json build is local, not a verified release install/],
    [PYTHON, 'a release that is not the version', (dir) => editSource(dir, (s) => void (s.release = 'v9.9.9')), /SOURCE\.json release v9\.9\.9 and version 1\.2\.3 do not agree/],
    [PYTHON, 'the web bundle recorded', (dir) => editSource(dir, (s) => void (s.bundle = 'zipp-wasm-1.2.3-web.zip')), /SOURCE\.json bundle zipp-wasm-1\.2\.3-web\.zip is not the web-python bundle/],
    [WEB, 'BUILD-INFO.txt of another commit', setInfo('commit', E40), /BUILD-INFO\.txt commit e{40} is not the recorded revision c{40}/],
    [PYTHON, 'BUILD-INFO.txt of another variant', setInfo('variant', 'javascript'), /BUILD-INFO\.txt variant is javascript, not javascript-python/],
    [WEB, 'PROFILE.json of another version', (dir) => {
      editText(path.join(dir, 'PROFILE.json'), (text) => text.replace(`"version": "${VERSION}"`, '"version": "9.9.9"'));
      resum(dir, 'PROFILE.json');
    }, /PROFILE\.json says version 9\.9\.9, not 1\.2\.3/],
    [PYTHON, 'another engine, listed', (dir) => {
      fs.appendFileSync(path.join(dir, 'zipp_wasm_bg.wasm'), 'x');
      resum(dir, 'zipp_wasm_bg.wasm');
    }, /zipp_wasm_bg\.wasm is not the recorded [0-9a-f]{64}/],
    [PYTHON, 'edited notices, recorded', (dir) => {
      fs.appendFileSync(path.join(dir, 'THIRD_PARTY_LICENSES.txt'), 'and one more\n');
      editSource(dir, (s) => void (s.notices.sha256 = sha256(fs.readFileSync(path.join(dir, 'THIRD_PARTY_LICENSES.txt')))));
    }, /THIRD_PARTY_LICENSES\.txt is not the curated copy in /],
    [WEB, 'a stray file', (dir) => fs.writeFileSync(path.join(dir, 'README.md'), ''), /README\.md is not part of the install/],
    [PYTHON, 'a folder', (dir) => fs.mkdirSync(path.join(dir, 'docs')), /docs is not a plain file/],
    [WEB, 'a missing file', (dir) => fs.rmSync(path.join(dir, 'LICENSE-APACHE')), /LICENSE-APACHE is missing/],
    [PYTHON, 'no SOURCE.json', (dir) => fs.rmSync(path.join(dir, 'SOURCE.json')), /holds no ZIPP install \(no SOURCE\.json\)/],
  ];
  for (const [variant, what, tamper, pattern] of cases) {
    const copy = path.join(tempDir(t, 'zipp-check-'), 'vendor');
    fs.cpSync(pristine, copy, { recursive: true });
    tamper(path.join(copy, variant.folder));
    assert.throws(() => checkInstalls(copy), (error) => error instanceof ZippReleaseError && pattern.test(error.message), `${variant.folder}: ${what}`);
  }
  // The curated copy an install is held to is the committed one, not whatever the install says.
  const otherCurated = path.join(tempDir(t, 'zipp-curated-'), 'THIRD_PARTY_LICENSES.txt');
  fs.writeFileSync(otherCurated, 'a different curated copy\n');
  assert.throws(() => checkInstalls(pristine, { curatedNotices: otherCurated }), /THIRD_PARTY_LICENSES\.txt is not the curated copy in /);
});

test('--resolve-only names the release and its SHA256SUMS digest without installing, and runs from a copy with no node_modules', (t) => {
  const folder = releaseFolder(t);
  const tree = tempDir(t, 'zipp-script-copy-');
  const script = path.join(tree, 'scripts', 'fetch-zipp-release.mjs');
  fs.mkdirSync(path.dirname(script), { recursive: true });
  fs.mkdirSync(path.join(tree, 'ui', 'vendor', 'zipp-notices'), { recursive: true });
  fs.copyFileSync(SCRIPT, script);
  for (const file of [CURATED_NOTICES, CURATED_RECORD]) fs.copyFileSync(file, path.join(tree, 'ui', 'vendor', 'zipp-notices', path.basename(file)));
  const copy = (args, env = {}, preload = []) => spawnSync(process.execPath, [...preload, script, ...args], { cwd: tree, encoding: 'utf8', env: { PATH: process.env.PATH, SYSTEMROOT: process.env.SYSTEMROOT, ...env } });

  const resolved = copy(['--resolve-only'], { ZIPP_RELEASE_DIR: folder.dir });
  assert.equal(resolved.status, 0, resolved.stderr);
  assert.deepEqual(JSON.parse(resolved.stdout), { release: TAG, sumsSha256: sha256(folder.top) });
  assert.deepEqual(fs.readdirSync(tree).sort(), ['scripts', 'ui'], 'nothing was installed or cached');
  refused({ run: copy(['--resolve-only'], { ZIPP_RELEASE_DIR: folder.dir, ZIPP_RELEASE: 'v1.2.4' }) }, /SHA256SUMS in .* does not list zipp-wasm-1\.2\.4-web\.zip/);
  refused({ run: copy(['--resolve-only', '--latest'], { ZIPP_RELEASE_DIR: folder.dir }) }, /ZIPP_RELEASE_DIR holds one release; leave out --latest/);
  refused({ run: copy(['--resolve-only', '--latest'], { ZIPP_RELEASE: TAG }) }, /name a release or --latest, not both/);
  refused({ run: copy(['--resolve-only', 'latest']) }, /"latest" is not a ZIPP release tag/);
  for (const args of [['--check', '--latest'], ['--online'], ['--ensure', '--check'], ['--frozen'], ['v1.2.3', 'v1.2.4']]) refused({ run: copy(args) }, /^fetch-zipp-release: Usage: /);

  // The copy installs into its own ui/vendor and checks it: the paths come from the script, not the working directory.
  const installed = copy([], { ZIPP_RELEASE_DIR: folder.dir });
  assert.equal(installed.status, 0, installed.stderr);
  const offline = copy(['--check']);
  assert.equal(offline.status, 0, offline.stderr);
  assert.equal(offline.stdout, `ZIPP ${TAG} (cccccccc) in ui/vendor checks.\n`);
  assert.equal(checkInstalls(path.join(tree, 'ui', 'vendor')).release, TAG);

  // --check --online as release.yml runs it before packaging: against what GitHub publishes, stubbed before the script loads.
  const online = copy(['--check', '--online'], {}, fetchStub(tree, 'published.mjs', published(folder)));
  assert.equal(online.status, 0, online.stderr);
  assert.equal(online.stdout, `ZIPP ${TAG} (cccccccc) in ui/vendor checks against the published release.\n`);
  const republished = new Map([...published(folder), [releaseAsset('SHA256SUMS'), Buffer.concat([folder.top, Buffer.from('\n')])]]);
  refused({ run: copy(['--check', '--online'], {}, fetchStub(tree, 'republished.mjs', republished)) }, /zipp-wasm\/RELEASE-SHA256SUMS is not the SHA256SUMS published under v1\.2\.3/);
  refused({ run: copy(['--check', '--online'], {}, fetchStub(tree, 'unpublished.mjs', new Map())) }, /releases\/download\/v1\.2\.3\/SHA256SUMS answered 404$/m);

  // With no tag and no folder, the latest release.
  const latestSums = Buffer.from(`${'3'.repeat(64)}  zipp-wasm-0.0.19-web.zip\n${'4'.repeat(64)}  zipp-wasm-0.0.19-web-python.zip\n`);
  const stub = fetchStub(tree, 'latest.mjs', new Map([[`${REPOSITORY}/releases/latest/download/SHA256SUMS`, latestSums], [`${REPOSITORY}/releases/download/v0.0.19/SHA256SUMS`, latestSums]]));
  for (const args of [['--resolve-only'], ['--resolve-only', '--latest']]) {
    const latest = copy(args, {}, stub);
    assert.equal(latest.status, 0, latest.stderr);
    assert.deepEqual(JSON.parse(latest.stdout), { release: 'v0.0.19', sumsSha256: sha256(latestSums) });
  }
});

test('the latest release is taken only when its SHA256SUMS is the one under its tag', async () => {
  const latestUrl = `${REPOSITORY}/releases/latest/download/SHA256SUMS`;
  const tagUrl = `${REPOSITORY}/releases/download/v0.0.19/SHA256SUMS`;
  const sums = (digit) => Buffer.from(`${digit.repeat(64)}  zipp-wasm-0.0.19-web.zip\n${digit.repeat(64)}  zipp-wasm-0.0.19-web-python.zip\n`);
  const same = serve(new Map([[latestUrl, sums('1')], [tagUrl, sums('1')]]));
  const resolved = await resolveRelease({ fetch: same.fetch });
  assert.equal(resolved.release, 'v0.0.19');
  assert.ok(resolved.sums.equals(sums('1')));
  assert.deepEqual(same.calls, [latestUrl, tagUrl], 'no API call: the latest redirect, then the tag\'s own asset');
  const moved = serve(new Map([[latestUrl, sums('1')], [tagUrl, sums('2')]]));
  await assert.rejects(resolveRelease({ latest: true, fetch: moved.fetch }), (error) => error instanceof ZippReleaseError && /latest release changed while it was read, so try again/.test(error.message));
  const webOnly = serve(new Map([[latestUrl, Buffer.from(`${'1'.repeat(64)}  zipp-wasm-0.0.19-web.zip\n`)]]));
  await assert.rejects(resolveRelease({ fetch: webOnly.fetch }), /SHA256SUMS lists 0 web-python bundles; expected exactly one/);
  const twoReleases = serve(new Map([[latestUrl, Buffer.from(`${'1'.repeat(64)}  zipp-wasm-0.0.19-web-python.zip\n${'2'.repeat(64)}  zipp-wasm-0.0.20-web-python.zip\n`)], [tagUrl, sums('1')]]));
  await assert.rejects(resolveRelease({ fetch: twoReleases.fetch }), /SHA256SUMS lists 2 web-python bundles; expected exactly one/);
  await assert.rejects(resolveRelease({ latest: true, tag: 'v0.0.19', fetch: same.fetch }), /name a release or --latest, not both/);
});

test('--ensure leaves intact installs alone without a single request, and reinstalls ones that fail --check', async (t) => {
  const folder = releaseFolder(t);
  const { out, run } = install(t, folder);
  assert.equal(run.status, 0, run.stderr);
  const calls = [];
  const fetch = async (url) => {
    calls.push(url);
    throw new Error('no request expected');
  };
  for (const wanted of [{}, { tag: TAG }, { expectSumsSha256: sha256(folder.top) }, { tag: TAG, expectSumsSha256: sha256(folder.top) }]) {
    assert.equal((await ensureInstalls({ vendorDir: out, fetch, log: quiet, ...wanted })).action, 'none', JSON.stringify(wanted));
  }
  const none = cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE: TAG, ZIPP_SUMS_SHA256: sha256(folder.top) });
  assert.equal(none.status, 0, none.stderr);
  assert.equal(none.stdout, '', 'nothing to say when nothing is done');
  assert.deepEqual(calls, []);
  fs.appendFileSync(path.join(out, PYTHON.folder, 'PROFILE.json'), ' ');
  const logs = [];
  const repaired = await ensureInstalls({ vendorDir: out, fetch, releaseDir: folder.dir, log: (line) => logs.push(line), warn: quiet });
  assert.equal(repaired.action, 'installed');
  assert.match(logs.join('\n'), /^Installing ZIPP from .*: the ZIPP install in .*zipp-wasm-python does not check/m);
  assert.equal(checkInstalls(out).release, TAG);
  assert.deepEqual(calls, [], 'the folder was the source');
  fs.rmSync(path.join(out, WEB.folder), { recursive: true });
  const missing = cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: folder.dir });
  assert.equal(missing.status, 0, missing.stderr);
  assert.match(missing.stdout, /zipp-wasm holds no ZIPP install/);
  assert.equal(checkInstalls(out).release, TAG);
});

test('--ensure replaces installs of a release other than ZIPP_RELEASE, or from another SHA256SUMS than ZIPP_SUMS_SHA256', (t) => {
  const first = releaseFolder(t);
  const { out, run } = install(t, first);
  assert.equal(run.status, 0, run.stderr);
  const next = releaseFolder(t, { version: '1.2.4' });
  const moved = cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: next.dir, ZIPP_RELEASE: 'v1.2.4' });
  assert.equal(moved.status, 0, moved.stderr);
  assert.match(moved.stdout, /holds ZIPP v1\.2\.3, not v1\.2\.4/);
  assert.equal(checkInstalls(out).release, 'v1.2.4');

  // The same release published again with other bytes beside the engines, so another SHA256SUMS.
  const again = releaseFolder(t, { version: '1.2.4', edit: { web: (files) => files.set('README.md', Buffer.from('# zipp-wasm, published again\n')) } });
  assert.notEqual(sha256(again.top), sha256(next.top));
  const reinstalled = cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: again.dir, ZIPP_SUMS_SHA256: sha256(again.top) });
  assert.equal(reinstalled.status, 0, reinstalled.stderr);
  assert.match(reinstalled.stdout, /installed from a SHA256SUMS other than ZIPP_SUMS_SHA256/);
  assert.equal(checkInstalls(out).sumsSha256, sha256(again.top));
  refused({ run: cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: again.dir, ZIPP_SUMS_SHA256: '0'.repeat(64) }) }, /ZIPP_SUMS_SHA256 says 0{64}/);
  assert.equal(checkInstalls(out).sumsSha256, sha256(again.top), 'the refusal left the installs alone');
});

test('--check --online compares both installs with what the release publishes now, byte for byte', async (t) => {
  const folder = releaseFolder(t);
  const { out, run } = install(t, folder);
  assert.equal(run.status, 0, run.stderr);
  const sumsUrl = `${REPOSITORY}/releases/download/${TAG}/SHA256SUMS`;
  const zipUrl = (variant) => `${REPOSITORY}/releases/download/${TAG}/${bundleName(VERSION, variant)}.zip`;
  const published = new Map([[sumsUrl, folder.top], ...VARIANTS.map((v) => [zipUrl(v), folder.zips[v.suffix]])]);
  const { fetch, calls } = serve(published);
  assert.equal((await checkInstallsOnline(out, { fetch })).release, TAG);
  assert.deepEqual(calls.sort(), [sumsUrl, ...VARIANTS.map(zipUrl)].sort());

  // An edited file whose line in the install's own SHA256SUMS was rewritten to match passes offline ...
  const edited = install(t, folder);
  fs.appendFileSync(path.join(edited.out, PYTHON.folder, 'zipp_wasm.d.ts'), '\n');
  resum(path.join(edited.out, PYTHON.folder), 'zipp_wasm.d.ts');
  assert.equal(checkInstalls(edited.out).release, TAG);
  // ... and not against what the release publishes.
  await assert.rejects(checkInstallsOnline(edited.out, { fetch }), (error) => /\n {2}- zipp-wasm-python\/SHA256SUMS is not the file in the published /.test(error.message) && /\n {2}- zipp-wasm-python\/zipp_wasm\.d\.ts is not the file in the published /.test(error.message));

  published.set(sumsUrl, Buffer.concat([folder.top, Buffer.from('\n')]));
  await assert.rejects(checkInstallsOnline(out, { fetch }), /zipp-wasm\/RELEASE-SHA256SUMS is not the SHA256SUMS published under v1\.2\.3/);
  published.set(sumsUrl, folder.top);
  published.set(zipUrl(WEB), releaseFolder(t, { repack: 'web' }).zips.web);
  await assert.rejects(checkInstallsOnline(out, { fetch }), /the published zipp-wasm-1\.2\.3-web\.zip is not the recorded [0-9a-f]{64}/);
});

test('installs into one folder take turns: hooks started together all succeed, and one of them installs', async (t) => {
  const folder = releaseFolder(t);
  const out = outDir(t);
  const runs = await Promise.all([0, 1, 2].map(() => spawnCli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: folder.dir })));
  for (const run of runs) assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  assert.equal(runs.filter((run) => /Installed ZIPP/.test(run.stdout)).length, 1, 'the others found the installs done');
  assert.equal(checkInstalls(out).release, TAG);
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], 'no stage, previous install or lock is left');
});

// Ctrl+C during a first install leaves its lock; the hooks started after it must not each take it over.
test('hooks started together on a lock a dead install left: one of them installs and every one succeeds', { timeout: 120_000 }, async (t) => {
  const folder = releaseFolder(t);
  // Holds every hook until the same instant, so all of them find the dead lock at once.
  const gate = path.join(tempDir(t, 'zipp-gate-'), 'gate.mjs');
  fs.writeFileSync(gate, 'const at = Number(process.env.LOCK_RACE_AT);\nconst wait = at - Date.now() - 20;\nif (wait > 0) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, wait);\nwhile (Date.now() < at);\n');
  for (let round = 0; round < 5; round++) {
    const out = outDir(t);
    fs.mkdirSync(out, { recursive: true });
    fs.writeFileSync(path.join(out, '.zipp-wasm.lock'), String(deadPid()));
    const at = String(Date.now() + 1500);
    const runs = await Promise.all(Array.from({ length: 8 }, () => spawnCli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: folder.dir, LOCK_RACE_AT: at }, ['--import', pathToFileURL(gate).href])));
    for (const run of runs) assert.equal(run.status, 0, `round ${round}: ${run.stdout}\n${run.stderr}`);
    assert.equal(runs.filter((run) => /Installed ZIPP/.test(run.stdout)).length, 1, `round ${round}: the others found the installs done`);
    assert.equal(checkInstalls(out).release, TAG);
    assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], `round ${round}: no lock, stage or previous install is left`);
  }
});

test('a lock whose process is gone is taken over with what it left; a live one is waited for', { timeout: 60_000 }, async (t) => {
  const folder = releaseFolder(t);
  const out = outDir(t);
  fs.mkdirSync(out, { recursive: true });
  const lock = path.join(out, '.zipp-wasm.lock');
  const gone = spawnSync(process.execPath, ['-e', '']).pid;
  fs.writeFileSync(lock, String(gone));
  fs.mkdirSync(path.join(out, `.zipp-wasm-python.stage-${gone}-0badf00d`));
  const taken = cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: folder.dir });
  assert.equal(taken.status, 0, taken.stderr);
  assert.doesNotMatch(taken.stderr, /Waiting for/, 'a dead holder is not waited on');
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python']);

  // Held by a live process: nothing happens until it lets go.
  fs.rmSync(path.join(out, WEB.folder), { recursive: true });
  const holder = livePid(t);
  fs.writeFileSync(lock, `${holder} 0123456789abcdef`);
  let released = false;
  const timer = setTimeout(() => {
    released = true;
    fs.rmSync(lock);
  }, 700);
  const warnings = [];
  const result = await ensureInstalls({ vendorDir: out, releaseDir: folder.dir, log: quiet, warn: (line) => warnings.push(line) });
  clearTimeout(timer);
  assert.ok(released, 'the install did not go ahead while the lock was held');
  assert.equal(result.action, 'installed');
  assert.match(warnings.join('\n'), new RegExp(`Waiting for the ZIPP install another process \\(${holder}\\) is making`));
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python']);
});

// Without this a hook would sit out the whole wait behind a lock left by a killed install whose pid Windows gave to another process.
test('a lock older than any install is taken over at once, even when its pid is alive again', { timeout: 30_000 }, async (t) => {
  const folder = releaseFolder(t);
  const out = outDir(t);
  fs.mkdirSync(out, { recursive: true });
  const lock = path.join(out, '.zipp-wasm.lock');
  // A live process stands in for the unrelated one that now has the dead holder's pid (a lock written by an earlier version: the pid alone).
  fs.writeFileSync(lock, String(livePid(t)));
  const eleven = new Date(Date.now() - 11 * 60_000);
  fs.utimesSync(lock, eleven, eleven);
  const warnings = [];
  const result = await ensureInstalls({ vendorDir: out, releaseDir: folder.dir, log: quiet, warn: (line) => warnings.push(line) });
  assert.equal(result.action, 'installed');
  assert.deepEqual(warnings.filter((line) => /Waiting for/.test(line)), [], 'an expired lock is not waited on');
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], 'the expired lock is gone with the install');
});

test('a lock naming this process is an earlier install\'s that had its pid, unless this process holds it', { timeout: 60_000 }, async (t) => {
  const folder = releaseFolder(t);
  const out = outDir(t);
  fs.mkdirSync(out, { recursive: true });
  fs.writeFileSync(path.join(out, '.zipp-wasm.lock'), String(process.pid));
  const warnings = [];
  const result = await ensureInstalls({ vendorDir: out, releaseDir: folder.dir, log: quiet, warn: (line) => warnings.push(line) });
  assert.equal(result.action, 'installed');
  assert.deepEqual(warnings.filter((line) => /Waiting for/.test(line)), [], 'this process does not wait for itself');
  // Two installs in this process take turns like two processes do.
  fs.rmSync(path.join(out, PYTHON.folder), { recursive: true });
  const both = await Promise.all([0, 1].map(() => ensureInstalls({ vendorDir: out, releaseDir: folder.dir, log: quiet, warn: quiet })));
  assert.deepEqual(both.map((r) => r.action).sort(), ['installed', 'none']);
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python']);
});

test('an empty lock is one being written this instant, unless it has been empty a while', { timeout: 60_000 }, async (t) => {
  const out = outDir(t);
  fs.mkdirSync(out, { recursive: true });
  const lock = path.join(out, '.zipp-wasm.lock');
  const warnings = [];
  const held = () => withInstallLock(out, async () => 'held', { warn: (line) => warnings.push(line) });
  fs.writeFileSync(lock, '');
  const old = new Date(Date.now() - 31_000);
  fs.utimesSync(lock, old, old);
  assert.equal(await held(), 'held');
  assert.deepEqual(warnings, [], 'empty for 31 s: its writer is gone');
  fs.writeFileSync(lock, '');
  const timer = setTimeout(() => fs.rmSync(lock), 600);
  assert.equal(await held(), 'held');
  clearTimeout(timer);
  assert.match(warnings.join('\n'), /Waiting for the ZIPP install another process \(starting\) is making/);
});

// A lock another hook has just removed can refuse its successor for a moment on Windows (delete pending).
test('a lock that cannot be created for a moment is tried again on Windows, and refused elsewhere', async (t) => {
  const out = outDir(t);
  const lock = path.join(out, '.zipp-wasm.lock');
  const { writeFileSync } = fs;
  let refusals = 3;
  t.mock.method(fs, 'writeFileSync', function (file, ...rest) {
    if (refusals > 0 && path.resolve(String(file)) === lock) {
      refusals--;
      throw Object.assign(new Error(`EPERM: operation not permitted, open '${file}'`), { code: 'EPERM' });
    }
    return writeFileSync.call(this, file, ...rest);
  });
  const locked = withInstallLock(out, async () => 'held', { warn: quiet });
  if (process.platform === 'win32') assert.equal(await locked, 'held');
  else await assert.rejects(locked, (error) => error instanceof ZippReleaseError && /cannot lock .* for an install \(.*: EPERM\)/.test(error.message));
});

test('a dead lock is removed only while it is still that lock, and by one waiter at a time', { timeout: 60_000 }, async (t) => {
  const dir = tempDir(t, 'zipp-lock-');
  const lock = path.join(dir, '.zipp-wasm.lock');
  const turn = `${lock}.break`;
  const dead = `${deadPid()} 00000000deadbeef`;
  const live = `${livePid(t)} 0000000011111111`;
  const seen = { text: dead, pid: Number(dead.split(' ')[0]), age: 0 };
  // Another waiter took it over and wrote its own lock since this one read the dead one: left alone.
  fs.writeFileSync(lock, live);
  assert.equal(await removeStaleLock(lock, seen), true);
  assert.equal(fs.readFileSync(lock, 'utf8'), live);
  // Still the dead lock: removed, and the turn given back.
  fs.writeFileSync(lock, dead);
  assert.equal(await removeStaleLock(lock, seen), true);
  assert.deepEqual(fs.readdirSync(dir), []);
  // Another waiter's turn: nothing is removed.
  fs.writeFileSync(lock, dead);
  fs.writeFileSync(turn, live);
  assert.equal(await removeStaleLock(lock, seen), true);
  assert.deepEqual(fs.readdirSync(dir).sort(), ['.zipp-wasm.lock', '.zipp-wasm.lock.break']);
  assert.equal(fs.readFileSync(turn, 'utf8'), live);
  // A turn whose waiter died is cleared, and the next one removes the dead lock.
  fs.writeFileSync(turn, dead);
  assert.equal(await removeStaleLock(lock, seen), true);
  assert.deepEqual(fs.readdirSync(dir), ['.zipp-wasm.lock']);
  assert.equal(await removeStaleLock(lock, seen), true);
  assert.deepEqual(fs.readdirSync(dir), []);
  // One that cannot be removed is looked at again after a pause, not in a spin, and the turn is given back.
  fs.writeFileSync(lock, dead);
  const { rmSync } = fs;
  const rm = t.mock.method(fs, 'rmSync', function (file, ...rest) {
    if (path.resolve(String(file)) === lock) throw Object.assign(new Error(`EPERM: operation not permitted, unlink '${file}'`), { code: 'EPERM' });
    return rmSync.call(this, file, ...rest);
  });
  const began = Date.now();
  assert.equal(await removeStaleLock(lock, seen), false);
  assert.ok(Date.now() - began >= 15, 'a pause before the next look');
  assert.deepEqual(fs.readdirSync(dir), ['.zipp-wasm.lock']);
  // Never removable: refused in the end, not retried forever.
  await assert.rejects(withInstallLock(dir, async () => assert.fail('the lock was never free'), { warn: quiet }), (error) => error instanceof ZippReleaseError && new RegExp(`cannot remove the install lock .*, which process ${seen.pid} left; delete it`).test(error.message));
  rm.mock.restore();
  assert.deepEqual(fs.readdirSync(dir), ['.zipp-wasm.lock']);
});

test('an install whose lock is no longer its own writes nothing and leaves that lock alone', async (t) => {
  const out = outDir(t);
  const lock = path.join(out, '.zipp-wasm.lock');
  const other = `${livePid(t)} 00000000cafef00d`;
  const takenOver = (error) => error instanceof ZippReleaseError && new RegExp(`install lock .* is no longer this install's \\(process ${other.split(' ')[0]} holds it\\); nothing was written, so run it again`).test(error.message);
  await assert.rejects(withInstallLock(out, async (assertHeld) => {
    assertHeld();
    fs.writeFileSync(lock, other);
    assertHeld();
  }, { warn: quiet }), takenOver);
  assert.equal(fs.readFileSync(lock, 'utf8'), other, 'the lock another install holds is not released');
  // Taken over while the bundles download: refused before either folder is written.
  fs.rmSync(lock);
  const folder = releaseFolder(t);
  const served = published(folder);
  const fetch = async (url) => {
    if (url.endsWith('.zip')) fs.writeFileSync(lock, other);
    return served.has(url) ? new Response(served.get(url)) : new Response('', { status: 404 });
  };
  await assert.rejects(installRelease({ vendorDir: out, tag: TAG, cacheDir: tempDir(t, 'zipp-cache-'), fetch, log: quiet, warn: quiet }), takenOver);
  assert.deepEqual(fs.readdirSync(out), ['.zipp-wasm.lock']);
  assert.equal(fs.readFileSync(lock, 'utf8'), other);
});

test('a folder that changes while it is checked is a refusal, and --ensure installs it again', async (t) => {
  const folder = releaseFolder(t);
  const { out, run } = install(t, folder);
  assert.equal(run.status, 0, run.stderr);
  const { readFileSync, readdirSync } = fs;
  const gone = (file) => Object.assign(new Error(`ENOENT: no such file or directory, open '${file}'`), { code: 'ENOENT' });
  // Listed, then swapped away by another install before it is read.
  const glue = path.join(out, WEB.folder, 'zipp_wasm.js');
  let vanish = 2;
  t.mock.method(fs, 'readFileSync', function (file, ...rest) {
    if (vanish > 0 && path.resolve(String(file)) === glue) {
      vanish--;
      throw gone(file);
    }
    return readFileSync.call(this, file, ...rest);
  });
  assert.throws(() => checkInstalls(out), (error) => error instanceof ZippReleaseError && /zipp-wasm does not check[\s\S]*zipp_wasm\.js cannot be read \(ENOENT\)/.test(error.message) && !/zipp_wasm\.js is missing/.test(error.message));
  const logs = [];
  const repaired = await ensureInstalls({ vendorDir: out, releaseDir: folder.dir, log: (line) => logs.push(line), warn: quiet });
  assert.equal(repaired.action, 'installed');
  assert.match(logs.join('\n'), /zipp_wasm\.js cannot be read \(ENOENT\)/);
  assert.equal(checkInstalls(out).release, TAG);
  // The whole folder gone between its SOURCE.json and its listing.
  t.mock.method(fs, 'readdirSync', function (dir, ...rest) {
    if (path.resolve(String(dir)) === path.join(out, PYTHON.folder)) throw gone(dir);
    return readdirSync.call(this, dir, ...rest);
  });
  assert.throws(() => checkInstalls(out), (error) => error instanceof ZippReleaseError && /zipp-wasm-python cannot be listed \(ENOENT\)/.test(error.message));
});

test('a swap that fails part way puts back what it had swapped: the folders never hold two releases', async (t) => {
  const older = releaseFolder(t, { version: '1.2.2', commit: 'e'.repeat(40) });
  const folder = releaseFolder(t);
  const installs = await verifyRelease({ release: TAG, sums: folder.top, zips: folder.zips });
  const { renameSync } = fs;
  for (const had of ['both installs', 'the web install only']) {
    const { out, run } = install(t, older);
    assert.equal(run.status, 0, run.stderr);
    if (had === 'the web install only') fs.rmSync(path.join(out, PYTHON.folder), { recursive: true });
    const before = snapshot(out);
    // The Python stage cannot take its place after the web one has.
    const rename = t.mock.method(fs, 'renameSync', function (from, to) {
      if (path.basename(String(to)) === PYTHON.folder && path.basename(String(from)).startsWith(`.${PYTHON.folder}.stage-`)) throw Object.assign(new Error(`EXDEV: cross-device link not permitted, rename '${from}' -> '${to}'`), { code: 'EXDEV' });
      return renameSync.call(this, from, to);
    });
    assert.throws(() => writeInstalls(out, installs), (error) => error instanceof ZippReleaseError && /could not write the installs into .*: EXDEV/.test(error.message), had);
    rename.mock.restore();
    assert.deepEqual(snapshot(out), before, `${had}: v1.2.2 as it was, and no stage or previous folder left`);
  }
});

test('what landed on disk is checked, not what was meant to land', async (t) => {
  const folder = releaseFolder(t);
  const { writeFileSync } = fs;
  // A disk, or a scanner, that changes one file on its way into a stage.
  t.mock.method(fs, 'writeFileSync', function (file, data, ...rest) {
    const changed = /\.stage-/.test(String(file)) && path.basename(String(file)) === 'zipp_wasm.d.ts';
    return writeFileSync.call(this, file, changed ? Buffer.from('not what the bundle holds\n') : data, ...rest);
  });
  await assert.rejects(installRelease({ vendorDir: outDir(t), releaseDir: folder.dir, log: quiet, warn: quiet }), (error) => error instanceof ZippReleaseError && /zipp-wasm does not check[\s\S]*zipp_wasm\.d\.ts does not match the bundle's SHA256SUMS/.test(error.message));
});

test('installs that cannot be written are a refusal, not a stack trace', (t) => {
  const file = path.join(tempDir(t, 'zipp-install-'), 'not-a-folder');
  fs.writeFileSync(file, '');
  assert.throws(() => writeInstalls(path.join(file, 'vendor'), [{ variant: WEB, files: new Map([['SOURCE.json', Buffer.from('{}')]]) }]), (error) => error instanceof ZippReleaseError && /could not write the installs into /.test(error.message));
  refused({ run: cli(['--ensure'], { ZIPP_OUT: path.join(file, 'vendor'), ZIPP_RELEASE_DIR: releaseFolder(t).dir }) }, /^fetch-zipp-release: cannot create .* for a ZIPP install/);
});

test('offline, a release downloaded before is named for ZIPP_RELEASE_DIR, never used unasked', async (t) => {
  const folder = releaseFolder(t);
  const cacheDir = tempDir(t, 'zipp-cache-');
  const offline = async () => {
    throw new TypeError('fetch failed');
  };
  await assert.rejects(resolveRelease({ tag: TAG, cacheDir, fetch: offline }), (error) => /could not fetch .*: fetch failed$/.test(error.message));
  await assert.rejects(resolveRelease({ cacheDir, fetch: offline }), (error) => /could not fetch .*latest.*: fetch failed$/.test(error.message));
  for (const version of ['1.2.2', VERSION]) {
    const release = version === VERSION ? folder : releaseFolder(t, { version });
    fs.cpSync(release.dir, path.join(cacheDir, `v${version}`), { recursive: true });
  }
  const cached = path.join(cacheDir, TAG);
  await assert.rejects(resolveRelease({ tag: 'v1.2.2', cacheDir, fetch: offline }), new RegExp(`: fetch failed; to install the v1\\.2\\.2 downloaded earlier without the network, set ZIPP_RELEASE_DIR=${escaped(path.join(cacheDir, 'v1.2.2'))}$`));
  // With no tag, the newest one downloaded.
  await assert.rejects(resolveRelease({ cacheDir, fetch: offline }), new RegExp(`set ZIPP_RELEASE_DIR=${escaped(cached)}$`));
  // GitHub answering that there is no such release is not a network failure: no copy stands in for it.
  const missing = async () => new Response('', { status: 404 });
  await assert.rejects(resolveRelease({ tag: TAG, cacheDir, fetch: missing }), (error) => /answered 404$/.test(error.message));
  // The folder named is a release folder as it is.
  const out = outDir(t);
  const run = cli(['--ensure'], { ZIPP_OUT: out, ZIPP_RELEASE_DIR: cached });
  assert.equal(run.status, 0, run.stderr);
});

test('the zip reader takes stored and deflated entries and refuses a damaged one', () => {
  const entries = new Map([['top/', Buffer.alloc(0)], ['top/a.txt', Buffer.from('alpha '.repeat(100))], ['top/empty', Buffer.alloc(0)]]);
  for (const deflate of [true, false]) {
    const files = unzip(writeZip(entries, { deflate }));
    assert.deepEqual([...files.keys()], ['top/a.txt', 'top/empty'], 'folder entries are not files');
    assert.ok(files.get('top/a.txt').equals(entries.get('top/a.txt')));
  }
  const stored = writeZip(entries, { deflate: false });
  const at = stored.indexOf('alpha');
  stored[at] ^= 0x20;
  assert.throws(() => unzip(stored), /not a readable zip: top\/a\.txt fails its size or CRC-32/);
  assert.throws(() => unzip(Buffer.from('not a zip at all, just text')), /not a readable zip: no end of central directory/);
  assert.throws(() => unzip(writeZip(entries, { flags: 0x1 })), /not a readable zip: top\/a\.txt is encrypted/);
  // Which of two copies an extractor keeps is its own choice; neither is taken.
  assert.throws(() => unzip(writeZip([['top/a.txt', Buffer.from('one')], ['top/a.txt', Buffer.from('two')]])), /not a readable zip: top\/a\.txt is in it twice/);
  // A Node without zlib.crc32 is told which ones have it.
  const { crc32 } = zlib;
  zlib.crc32 = undefined;
  try {
    assert.throws(() => unzip(stored), new RegExp(`needs zlib\\.crc32, in Node 20\\.15, 22\\.2 and newer \\(this is ${escaped(process.version)}\\)`));
  } finally {
    zlib.crc32 = crc32;
  }
});

test('the curated notices must be the file their SOURCE.json records, and a release newer than they were generated from installs with a prompt to regenerate them', async (t) => {
  // In process the prompt goes to `warn` wherever this runs; the GitHub Actions form is checked through the command line below.
  const actions = process.env.GITHUB_ACTIONS;
  delete process.env.GITHUB_ACTIONS;
  t.after(() => {
    if (actions !== undefined) process.env.GITHUB_ACTIONS = actions;
  });
  const record = JSON.parse(fs.readFileSync(CURATED_RECORD, 'utf8'));
  const dir = tempDir(t, 'zipp-curated-');
  const curatedNotices = path.join(dir, 'THIRD_PARTY_LICENSES.txt');
  fs.copyFileSync(CURATED_NOTICES, curatedNotices);
  const recordAs = (change) => fs.writeFileSync(path.join(dir, 'SOURCE.json'), JSON.stringify({ ...record, ...change }));
  const folder = releaseFolder(t);
  const warned = async (change) => {
    recordAs(change);
    const warnings = [];
    await installRelease({ vendorDir: outDir(t), releaseDir: folder.dir, curatedNotices, log: quiet, warn: (line) => warnings.push(line) });
    return warnings;
  };
  const older = await warned({ release: 'v1.2.2' });
  assert.equal(older.length, 1, older.join('\n'));
  assert.match(older[0], /^ZIPP v1\.2\.3 is newer than v1\.2\.2, the release .*THIRD_PARTY_LICENSES\.txt was generated from; if it compiles in more third-party code, regenerate the notices: node scripts\/regen-zipp-notices\.mjs <zipp\.org checkout> v1\.2\.3$/);
  for (const release of [TAG, 'v1.2.4']) assert.deepEqual(await warned({ release }), [], `generated from ${release}`);

  const refusedWith = (pattern) => assert.rejects(installRelease({ vendorDir: outDir(t), releaseDir: folder.dir, curatedNotices, log: quiet, warn: quiet }), (error) => error instanceof ZippReleaseError && pattern.test(error.message));
  recordAs({ sha256: 'e'.repeat(64) });
  await refusedWith(/THIRD_PARTY_LICENSES\.txt is not the file .*SOURCE\.json records \(v\d+\.\d+\.\d+, sha256 e{64}\); regenerate both with node scripts\/regen-zipp-notices\.mjs/);
  recordAs({ release: 'latest' });
  await refusedWith(/is not the file .*SOURCE\.json records \(latest, /);
  fs.rmSync(path.join(dir, 'SOURCE.json'));
  await refusedWith(/SOURCE\.json does not record the curated notices \(ENOENT\)/);

  // Through the command line the prompt goes to stderr, and in GitHub Actions it is an annotation on the run.
  const local = install(t, folder).run;
  assert.equal(local.status, 0, local.stderr);
  assert.match(local.stderr, new RegExp(`^ZIPP v1\\.2\\.3 is newer than ${escaped(record.release)}, `, 'm'));
  const annotated = install(t, folder, { env: { GITHUB_ACTIONS: 'true' } }).run;
  assert.equal(annotated.status, 0, annotated.stderr);
  assert.match(annotated.stdout, new RegExp(`^::warning::ZIPP v1\\.2\\.3 is newer than ${escaped(record.release)}, `, 'm'));
  assert.doesNotMatch(annotated.stderr, /is newer than/);
});

test('the curated notices regenerate from zipp.org\'s sources by one recipe, and the committed pair is what it gives', (t) => {
  assert.deepEqual(JSON.parse(fs.readFileSync(CURATED_RECORD, 'utf8')), noticesRecord(JSON.parse(fs.readFileSync(CURATED_RECORD, 'utf8')).release, fs.readFileSync(CURATED_NOTICES, 'utf8')), 'SOURCE.json records the committed notices');
  // Each source as it is, final newline included, and the whole trimmed once: Softn's recipe, byte for byte.
  const sources = { [NOTICE_SOURCES.rustpython]: 'MIT License\n\nCopyright (c) 2020 RustPython Team\n', [NOTICE_SOURCES.unicode]: 'UNICODE LICENSE V3\n\nSPDX-License-Identifier: Unicode-3.0\n' };
  const read = (file) => sources[file];
  const text = 'ZIPP engine: Apache-2.0. See the source repository for its complete notices.\n\nRustPython parser (MIT):\nMIT License\n\nCopyright (c) 2020 RustPython Team\n\n\nUnicode data:\nUNICODE LICENSE V3\n\nSPDX-License-Identifier: Unicode-3.0\n';
  const dir = tempDir(t, 'zipp-notices-');
  const notices = path.join(dir, 'THIRD_PARTY_LICENSES.txt');
  assert.deepEqual(regenerate({ read, release: TAG, dir }), []);
  assert.equal(fs.readFileSync(notices, 'utf8'), text);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dir, 'SOURCE.json'), 'utf8')), { repository: REPOSITORY, release: TAG, sources: ['crates/rustpython-parser-fork/LICENSE', 'LICENSE-UNICODE'], file: 'THIRD_PARTY_LICENSES.txt', sha256: sha256(Buffer.from(text)) });
  assert.deepEqual(readCuratedNotices(notices), { bytes: Buffer.from(text), release: TAG }, 'what the installer takes');
  assert.deepEqual(regenerate({ read, release: TAG, dir, check: true }), []);
  assert.deepEqual(regenerate({ read, release: 'v1.2.4', dir, check: true }), ['SOURCE.json does not record v1.2.4 and that file\'s SHA-256']);
  fs.appendFileSync(notices, 'edited\n');
  assert.deepEqual(regenerate({ read, release: TAG, dir, check: true }), ['THIRD_PARTY_LICENSES.txt is not what the v1.2.3 sources give']);
  assert.throws(() => readCuratedNotices(notices), /is not the file .* records/);
  const usage = spawnSync(process.execPath, [fileURLToPath(new URL('./regen-zipp-notices.mjs', import.meta.url)), dir, 'latest'], { encoding: 'utf8' });
  assert.equal(usage.status, 1);
  assert.match(usage.stderr, /^Usage: node scripts\/regen-zipp-notices\.mjs <zipp\.org checkout> vX\.Y\.Z \[--check\]/);
});
