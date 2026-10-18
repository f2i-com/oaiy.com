/**
 * The ZIPP installs take a release only when every layer checks, for both
 * engines: each module against ZIPP's SHA256SUMS, its zipp.meta record, its
 * zipp.glue section and its stack, and what it reports loaded under that glue,
 * against that variant of that release; the embed kit against SHA256SUMS and
 * each file in it against the kit's own; and all of it against one commit.
 *
 *   node --test scripts/fetch-zipp-release.test.mjs
 *
 * Fixture releases are built in temp folders. Each module is a minimal valid
 * WebAssembly binary (Rust's stack pointer as its first global, then the
 * zipp.meta and zipp.glue custom sections) whose glue is a stand-in that
 * reports whatever profile a case needs; SHA256SUMS, the kit's own and its
 * modules.json are rebuilt per case, so a case trips only the refusal it is
 * about. One case repacks the engines ui/vendor holds into a release, so the
 * real glue and modules pass the same checks and give back the installed files
 * (install them first; outside CI that case is skipped, saying why, when there
 * are none). Releases come from a folder (ZIPP_RELEASE_DIR) through the command
 * line, or from a stubbed fetch, in process or preloaded; nothing here reaches
 * GitHub.
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
  CURATED_NOTICES, KIT_FILES, NOTICES, REPOSITORY, VARIANTS, VENDOR_DIR, ZippReleaseError,
  checkInstalls, checkInstallsOnline, ensureInstalls, installRelease, installedFiles, kitName, moduleName, readCuratedNotices, removeStaleLock, resolveRelease, sha256, unzip, verifyRelease, wasmCustomSections, wasmStackBytes, withInstallLock, writeInstalls,
} from './fetch-zipp-release.mjs';
import { noticeSources, noticesRecord, regenerate } from './regen-zipp-notices.mjs';

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
/** A SHA256SUMS text without the line for `name`. */
const dropLine = (text, name) => text.split('\n').filter((line) => !line.endsWith(`  ${name}`)).join('\n');

/**
 * A zip as ZIPP's release job writes one: deflated files (stored when
 * `deflate` is false). `entries` are [name, bytes] pairs; `flags` is every
 * entry's general-purpose bit flag.
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

// ---------------------------------------------------------------------------
// WebAssembly fixtures
// ---------------------------------------------------------------------------
const uleb = (n) => {
  const out = [];
  do {
    let byte = n & 0x7f;
    n = Math.floor(n / 128);
    if (n) byte |= 0x80;
    out.push(byte);
  } while (n);
  return Buffer.from(out);
};
/** A signed LEB128, as i32.const takes its operand: 1 MiB is 80 80 c0 00, where an unsigned one would be 80 80 40. */
const sleb = (n) => {
  const out = [];
  for (;;) {
    const byte = n & 0x7f;
    n >>= 7;
    const done = (n === 0 && !(byte & 0x40)) || (n === -1 && byte & 0x40);
    out.push(done ? byte : byte | 0x80);
    if (done) return Buffer.from(out);
  }
};
const wasmSection = (id, body) => Buffer.concat([Buffer.from([id]), uleb(body.length), body]);
const customSection = (name, payload) => wasmSection(0, Buffer.concat([uleb(Buffer.byteLength(name)), Buffer.from(name), payload]));
const WASM_HEADER = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
/** A module with one more custom section at its end: still valid, and still carrying what it carried. */
const appendSection = (bytes, name, payload) => Buffer.concat([bytes, customSection(name, payload)]);

/**
 * A minimal valid WebAssembly module, laid out as rustc lays out what matters
 * here: its first global is the mutable i32 stack pointer, set by `i32.const
 * <stack>` (an i64 one when `stackType` says so), and `custom` sections follow
 * in order.
 */
function wasmModule({ stack, stackType = 'i32', custom = [] }) {
  const type = stackType === 'i32' ? [0x7f, 0x01, 0x41] : [0x7e, 0x01, 0x42];
  const globals = Buffer.concat([uleb(1), Buffer.from(type), sleb(stack), Buffer.from([0x0b])]);
  return Buffer.concat([WASM_HEADER, wasmSection(6, globals), ...custom.map(([name, payload]) => customSection(name, payload))]);
}

/** Glue that describes the module as `profile`, standing in for the engine where what it reports is the point. */
const standInGlue = (profile) => Buffer.from(`export function initSync() {}\nexport function zippProfile() { return ${JSON.stringify(JSON.stringify(profile))}; }\n`);

/**
 * What a fixture module is made of, for a case to change before buildModule
 * assembles it: its stack, its zipp.meta record (glueSha256 is filled in from
 * the glue unless a case sets one), the profile its stand-in glue reports,
 * `glue` (null for none), `record` (the record's bytes, null for none),
 * `custom` sections after the two, and `bytes` to replace the lot.
 */
function moduleParts(variant, { version, commit }) {
  return {
    stack: variant.stackBytes,
    meta: { format: 1, variant: variant.variant, version, commit, languages: variant.languages },
    profile: { engine: 'zipp-wasm', version, features: ['safe-sandbox', 'meter-only'], languages: variant.languages, source: { sha: commit }, limits: { lifetimeSteps: 50000000, maxInstructionBudgetSteps: 2000000000 } },
    custom: [],
  };
}
function buildModule(parts) {
  if (parts.bytes) return parts.bytes;
  const glue = parts.glue === undefined ? standInGlue(parts.profile) : parts.glue;
  const meta = { ...parts.meta, ...(glue && !('glueSha256' in parts.meta) ? { glueSha256: sha256(glue) } : {}) };
  const record = parts.record === undefined ? Buffer.from(JSON.stringify(meta)) : parts.record;
  return wasmModule({ stack: parts.stack, stackType: parts.stackType, custom: [...(record ? [['zipp.meta', record]] : []), ...(glue ? [['zipp.glue', glue]] : []), ...parts.custom] });
}
/** The fixture module of `variant` of the default release, changed by `edit`. */
const rebuilt = (variant, edit = () => {}) => {
  const parts = moduleParts(variant, { version: VERSION, commit: COMMIT });
  edit(parts);
  return buildModule(parts);
};

// ---------------------------------------------------------------------------
// Release fixtures
// ---------------------------------------------------------------------------
const kitFolder = (version) => kitName(version).replace(/\.zip$/, '');
/** The embed kit's files, by path inside its folder; modules.json is added once the modules exist. */
function kitFixture({ version, commit }) {
  return new Map([
    ['BUILD-INFO.txt', Buffer.from(`version=${version}\ncommit=${commit}\nkit-source=${'9'.repeat(40)}\nloader=1.2.0\nhost-sdk=1.3.0\n`)],
    ['LICENSE-APACHE', Buffer.from('Apache License, Version 2.0\n')],
    ['zipp-engine.d.ts', Buffer.from('export class Engine {}\nexport function initSync(module: unknown): unknown;\n')],
    // Listed by the kit's SHA256SUMS, never installed.
    ['README.md', Buffer.from('# ZIPP WebAssembly embed kit\n')],
    ['zipp.mjs', Buffer.from('export {};\n')],
    ['zipp.d.ts', Buffer.from('export {};\n')],
    ['docs/zipp-wasm.md', Buffer.from('# zipp-wasm\n')],
    ['host-sdk/zipp-host.mjs', Buffer.from('export {};\n')],
    ['gpu-lab/LICENSE', Buffer.from('gpu-lab licence\n')],
  ]);
}
const DECOY_LITE = '5'.repeat(64);
const modulesJson = ({ version, commit }, modules) => Buffer.from(`${JSON.stringify({
  version,
  commit,
  modules: [
    ...VARIANTS.map((v) => ({ file: moduleName(version, v), variant: v.variant, languages: v.languages, bytes: modules[v.suffix].length, sha256: sha256(modules[v.suffix]) })),
    { file: `zipp-${version}-javascript-lite.wasm`, variant: 'lite', languages: ['javascript'], bytes: 1, sha256: DECOY_LITE },
  ],
}, null, 2)}\n`);
const setBuildInfo = (files, key, value) => files.set('BUILD-INFO.txt', Buffer.from(files.get('BUILD-INFO.txt').toString('utf8').replace(new RegExp(`^${key}=.*$`, 'm'), `${key}=${value}`)));
const setJson = (files, name, change) => {
  const value = JSON.parse(files.get(name).toString('utf8'));
  change(value);
  files.set(name, Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
};

/**
 * A release folder holding SHA256SUMS, both modules and the embed kit, built
 * from parts a case can change at each layer: `edit[suffix](parts)` a module
 * before it is assembled; `kit(files)` the kit's files (modules.json written)
 * before its SHA256SUMS, `kitSums(text)` that text and `tamperKit(files)` the
 * kit after it; `kitTop` the kit's folder; `sums(text)` ZIPP's SHA256SUMS, and
 * `tamper[suffix](bytes)` a module after that lists it; `repack` packs the kit
 * again after it is listed, a valid zip of the same files whose bytes are not
 * the ones listed. `real` takes the modules, declarations and licence
 * ui/vendor holds.
 */
function releaseFolder(t, { version = VERSION, commit = COMMIT, edit = {}, kit: editKit, kitSums, tamperKit, kitTop = kitFolder(version), sums: editSums, tamper = {}, repack = false, real = false } = {}) {
  const parts = {};
  const listed = {};
  for (const variant of VARIANTS) {
    if (real) listed[variant.suffix] = fs.readFileSync(path.join(VENDOR_DIR, variant.folder, 'zipp_wasm_bg.wasm'));
    else {
      parts[variant.suffix] = moduleParts(variant, { version, commit });
      edit[variant.suffix]?.(parts[variant.suffix]);
      listed[variant.suffix] = buildModule(parts[variant.suffix]);
    }
  }
  const kitFiles = kitFixture({ version, commit });
  if (real) for (const [installed, from] of Object.entries(KIT_FILES)) kitFiles.set(from, fs.readFileSync(path.join(VENDOR_DIR, PYTHON.folder, installed)));
  kitFiles.set('modules.json', modulesJson({ version, commit }, listed));
  editKit?.(kitFiles);
  let inner = `${[...kitFiles].map(([name, bytes]) => `${sha256(bytes)}  ${name}`).join('\n')}\n`;
  if (kitSums) inner = kitSums(inner);
  kitFiles.set('SHA256SUMS', Buffer.from(inner));
  tamperKit?.(kitFiles);
  const kitEntries = new Map([...kitFiles].map(([name, bytes]) => [`${kitTop}/${name}`, bytes]));
  let kit = writeZip(kitEntries);
  let text = `${[
    `${'a'.repeat(64)}  zipp-${version}-linux-x86_64`,
    `${'b'.repeat(64)}  zipp-${version}-windows-x86_64-installer.msi`,
    ...VARIANTS.map((v) => `${sha256(listed[v.suffix])}  ${moduleName(version, v)}`),
    // Published beside them and never taken: the anchored names must not match these.
    `${DECOY_LITE}  zipp-${version}-javascript-lite.wasm`,
    `${'6'.repeat(64)}  zipp-${version}-javascript-python-base.wasm`,
    `${'7'.repeat(64)}  zipp-${version}-torch-package.wasm`,
    `${'8'.repeat(64)}  zipp-${version}-javascript-python.wasm.br`,
    `${sha256(kit)}  ${kitName(version)}`,
  ].join('\n')}\n`;
  if (editSums) text = editSums(text);
  const top = Buffer.from(text);
  if (repack) kit = writeZip(kitEntries, { deflate: false });
  const modules = Object.fromEntries(VARIANTS.map((v) => [v.suffix, tamper[v.suffix] ? tamper[v.suffix](listed[v.suffix]) : listed[v.suffix]]));
  const dir = tempDir(t, 'zipp-release-fixture-');
  fs.writeFileSync(path.join(dir, 'SHA256SUMS'), top);
  for (const variant of VARIANTS) fs.writeFileSync(path.join(dir, moduleName(version, variant)), modules[variant.suffix]);
  fs.writeFileSync(path.join(dir, kitName(version)), kit);
  return { dir, top, modules, listed, kit, kitFiles, kitEntries, parts, release: `v${version}` };
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
/**
 * Put `bytes` in place of an install's module and record it as published:
 * its SOURCE.json digests, and both installs' RELEASE-SHA256SUMS and their
 * digest, so only what the module itself carries can be wrong.
 */
function replaceModule(vendor, variant, bytes) {
  const dir = path.join(vendor, variant.folder);
  const { bundle } = readSource(dir);
  fs.writeFileSync(path.join(dir, 'zipp_wasm_bg.wasm'), bytes);
  const sums = fs.readFileSync(path.join(dir, 'RELEASE-SHA256SUMS'), 'utf8').replace(new RegExp(`^[0-9a-f]{64}(  ${escaped(bundle)})$`, 'm'), `${sha256(bytes)}$1`);
  for (const each of VARIANTS) {
    fs.writeFileSync(path.join(vendor, each.folder, 'RELEASE-SHA256SUMS'), sums);
    editSource(path.join(vendor, each.folder), (s) => {
      s.sumsSha256 = sha256(Buffer.from(sums));
      if (each === variant) s.sha256 = s.bundleSha256 = sha256(bytes);
    });
  }
}
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
/** What GitHub serves for a fixture release: its SHA256SUMS, both modules and the kit. */
const published = (folder) => new Map([[releaseAsset('SHA256SUMS'), folder.top], ...VARIANTS.map((v) => [releaseAsset(moduleName(VERSION, v)), folder.modules[v.suffix]]), [releaseAsset(kitName(VERSION)), folder.kit]]);

test('a release that checks all the way down installs both engines as exactly their install sets, and --check passes', (t) => {
  const folder = releaseFolder(t);
  // No tag: the release the folder holds.
  const { out, run } = install(t, folder);
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], 'no stage, previous install or lock is left behind');
  const kitFiles = Object.fromEntries(Object.entries(KIT_FILES).map(([name, from]) => [name, { from, sha256: sha256(folder.kitFiles.get(from)) }]));
  for (const variant of VARIANTS) {
    const dir = path.join(out, variant.folder);
    const record = readSource(dir);
    const module = folder.modules[variant.suffix];
    const { profile } = folder.parts[variant.suffix];
    const glue = standInGlue(profile);
    // PROFILE.json is no longer published: it is what the engine reports, as the bundles wrote it.
    const profileJson = Buffer.from(`${JSON.stringify(profile, null, 2)}\n`);
    assert.deepEqual(fs.readdirSync(dir).sort(), installedFiles(record.notices).sort(), `${variant.folder}: nothing else the kit holds (README.md, docs/, gpu-lab/, host-sdk/, BUILD-INFO.txt, modules.json) is installed`);
    assert.deepEqual(record, {
      repository: REPOSITORY,
      release: TAG,
      version: VERSION,
      revision: COMMIT,
      build: 'release',
      bundle: moduleName(VERSION, variant),
      bundleSha256: sha256(module),
      sumsSha256: sha256(folder.top),
      variant: variant.variant,
      languages: variant.languages,
      stackBytes: variant.stackBytes,
      license: 'Apache-2.0',
      artifact: 'zipp_wasm_bg.wasm',
      sha256: sha256(module),
      glueSha256: sha256(glue),
      profileSha256: sha256(profileJson),
      kit: { file: kitName(VERSION), sha256: sha256(folder.kit), files: kitFiles },
      notices: variant.needsNotices ? { file: NOTICES, source: 'oaiy-curated', sha256: sha256(fs.readFileSync(CURATED_NOTICES)) } : null,
    }, variant.folder);
    const expected = {
      'zipp_wasm_bg.wasm': module,
      'zipp_wasm.js': glue,
      'zipp_wasm.d.ts': folder.kitFiles.get('zipp-engine.d.ts'),
      'LICENSE-APACHE': folder.kitFiles.get('LICENSE-APACHE'),
      'PROFILE.json': profileJson,
      'RELEASE-SHA256SUMS': folder.top,
    };
    for (const [name, bytes] of Object.entries(expected)) assert.ok(fs.readFileSync(path.join(dir, name)).equals(bytes), `${variant.folder}/${name} byte for byte`);
  }
  // Only the Python engine compiles in Unicode data; the release ships no notices, so the curated copy is installed and says so.
  assert.ok(fs.readFileSync(path.join(out, PYTHON.folder, NOTICES)).equals(fs.readFileSync(CURATED_NOTICES)));
  const check = cli(['--check'], { ZIPP_OUT: out });
  assert.equal(check.status, 0, `${check.stdout}\n${check.stderr}`);
  assert.match(check.stdout, new RegExp(`ZIPP ${escaped(TAG)} \\(cccccccc\\) in .* checks\\.`));
});

test('the engines ui/vendor holds, real glue and modules, pass the same checks repacked into a release, and give back the installed files', async (t) => {
  let installed;
  try {
    installed = checkInstalls(VENDOR_DIR);
  } catch (error) {
    const why = `ui/vendor holds no ZIPP install that checks; run node scripts/fetch-zipp-release.mjs (or npm test in ui) first`;
    // CI installs before it tests; a fresh clone has nothing to repack yet.
    if (!process.env.CI) return t.skip(why);
    assert.fail(`${why}\n${error.message}`);
  }
  // The installed modules as they are, a kit of the installed declarations and licence, and a SHA256SUMS of both.
  const folder = releaseFolder(t, { version: installed.release.slice(1), commit: installed.revision, real: true });
  const installs = await verifyRelease({ release: installed.release, sums: folder.top, modules: folder.modules, kit: folder.kit });
  const fields = ['revision', 'version', 'bundle', 'bundleSha256', 'sha256', 'glueSha256', 'profileSha256', 'variant', 'languages', 'stackBytes', 'notices'];
  for (const { variant, files, source } of installs) {
    const dir = path.join(VENDOR_DIR, variant.folder);
    // Above all PROFILE.json: what the real engine reports through its own glue is the file installed.
    for (const name of installedFiles(source.notices).filter((n) => !['RELEASE-SHA256SUMS', 'SOURCE.json'].includes(n))) {
      assert.ok(files.get(name).equals(fs.readFileSync(path.join(dir, name))), `${variant.folder}/${name} is what the real release gives`);
    }
    const was = installed.sources[variant.suffix];
    assert.deepEqual(Object.fromEntries(fields.map((f) => [f, source[f]])), Object.fromEntries(fields.map((f) => [f, was[f]])), variant.folder);
    assert.deepEqual(source.kit.files, was.kit.files, `${variant.folder}: the kit's files`);
  }
});

test('notices the kit ships itself are taken from it into both installs and recorded as the release\'s', (t) => {
  const notices = Buffer.from('ZIPP\'s own notices\n');
  const { out, run } = install(t, releaseFolder(t, { kit: (files) => files.set(NOTICES, notices) }));
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  for (const variant of VARIANTS) {
    const dir = path.join(out, variant.folder);
    const record = readSource(dir);
    assert.deepEqual(record.notices, { file: NOTICES, source: 'zipp-release', sha256: sha256(notices) }, variant.folder);
    assert.deepEqual(record.kit.files[NOTICES], { from: NOTICES, sha256: sha256(notices) }, `${variant.folder}: a file taken from the kit`);
    assert.ok(fs.readFileSync(path.join(dir, NOTICES)).equals(notices));
  }
  assert.equal(cli(['--check'], { ZIPP_OUT: out }).status, 0);
  // Listed by the kit's SHA256SUMS, like everything else taken from it.
  const unlisted = releaseFolder(t, { kit: (files) => files.set(NOTICES, notices), kitSums: (text) => dropLine(text, NOTICES) });
  refused(install(t, unlisted), /THIRD_PARTY_LICENSES\.txt in zipp-1\.2\.3-wasm-embed-kit\.zip is not listed in the kit's SHA256SUMS/);
});

test('a module or kit that disagrees with the release\'s SHA256SUMS is refused, even one carrying the very same record, glue and files', async (t) => {
  for (const variant of VARIANTS) {
    // Still a valid module with the same record, glue and stack: only ZIPP's SHA256SUMS can tell.
    const folder = releaseFolder(t, { tamper: { [variant.suffix]: (bytes) => appendSection(bytes, 'name', Buffer.from('built again')) } });
    const [module, listed] = [folder.modules[variant.suffix], folder.listed[variant.suffix]];
    for (const name of ['zipp.meta', 'zipp.glue']) assert.deepEqual(wasmCustomSections(module, name), wasmCustomSections(listed, name), `${name} is carried unchanged`);
    assert.equal(wasmStackBytes(module), variant.stackBytes);
    refused(install(t, folder), new RegExp(`${escaped(moduleName(VERSION, variant))} has sha256 [0-9a-f]{64}; the ${escaped(TAG)} SHA256SUMS says [0-9a-f]{64}`));
    // verifyRelease holds each module to the SHA256SUMS itself, whoever loaded it.
    await assert.rejects(verifyRelease({ release: TAG, sums: folder.top, modules: folder.modules, kit: folder.kit }), (error) => error instanceof ZippReleaseError && error.message === `${moduleName(VERSION, variant)} does not match the ${TAG} SHA256SUMS`);
  }
  const repacked = releaseFolder(t, { repack: true });
  // Every file inside checks, so only ZIPP's top-level SHA256SUMS can tell.
  const inside = unzip(repacked.kit);
  for (const [name, bytes] of repacked.kitEntries) assert.ok(inside.get(name).equals(bytes), `${name} is in the repacked kit unchanged`);
  refused(install(t, repacked), /zipp-1\.2\.3-wasm-embed-kit\.zip has sha256 [0-9a-f]{64}; the v1\.2\.3 SHA256SUMS says [0-9a-f]{64}/);
  await assert.rejects(verifyRelease({ release: TAG, sums: repacked.top, modules: repacked.modules, kit: repacked.kit }), (error) => error instanceof ZippReleaseError && error.message === `${kitName(VERSION)} does not match the ${TAG} SHA256SUMS`);
  const flipped = releaseFolder(t);
  const file = path.join(flipped.dir, moduleName(VERSION, PYTHON));
  const bytes = fs.readFileSync(file);
  bytes[bytes.length >> 1] ^= 0x01;
  fs.writeFileSync(file, bytes);
  refused(install(t, flipped), /zipp-1\.2\.3-javascript-python\.wasm has sha256 [0-9a-f]{64}; the v1\.2\.3 SHA256SUMS says/);
});

test('a kit file that disagrees with the kit\'s own SHA256SUMS is refused, whether the install takes it or not', (t) => {
  const append = (name) => (files) => files.set(name, Buffer.concat([files.get(name), Buffer.from(' ')]));
  refused(install(t, releaseFolder(t, { tamperKit: append('zipp-engine.d.ts') })), /zipp-engine\.d\.ts in zipp-1\.2\.3-wasm-embed-kit\.zip does not match the kit's SHA256SUMS/);
  refused(install(t, releaseFolder(t, { tamperKit: append('LICENSE-APACHE') })), /LICENSE-APACHE in zipp-1\.2\.3-wasm-embed-kit\.zip does not match the kit's SHA256SUMS/);
  refused(install(t, releaseFolder(t, { tamperKit: append('host-sdk/zipp-host.mjs') })), /host-sdk\/zipp-host\.mjs in zipp-1\.2\.3-wasm-embed-kit\.zip does not match the kit's SHA256SUMS/);
  refused(install(t, releaseFolder(t, { tamperKit: (files) => files.delete('SHA256SUMS') })), /zipp-1\.2\.3-wasm-embed-kit\.zip carries no SHA256SUMS/);
});

test('a file the install reads from the kit that its SHA256SUMS does not list, one it lists that the kit lacks, or one the kit lacks, is refused', (t) => {
  for (const name of ['zipp-engine.d.ts', 'LICENSE-APACHE', 'BUILD-INFO.txt', 'modules.json']) {
    refused(install(t, releaseFolder(t, { kitSums: (text) => dropLine(text, name) })), new RegExp(`${escaped(name)} in zipp-1\\.2\\.3-wasm-embed-kit\\.zip is not listed in the kit's SHA256SUMS`));
  }
  refused(install(t, releaseFolder(t, { kitSums: (text) => `${text}${'f'.repeat(64)}  docs/GHOST.md\n` })), /zipp-1\.2\.3-wasm-embed-kit\.zip lists docs\/GHOST\.md in its SHA256SUMS but does not carry it/);
  refused(install(t, releaseFolder(t, { kit: (files) => files.delete('zipp-engine.d.ts') })), /zipp-1\.2\.3-wasm-embed-kit\.zip has no zipp-engine\.d\.ts/);
  // Another release's kit under this one's name, with a SHA256SUMS to match.
  refused(install(t, releaseFolder(t, { kitTop: 'zipp-1.2.2-wasm-embed-kit' })), /zipp-1\.2\.3-wasm-embed-kit\.zip has no zipp-1\.2\.3-wasm-embed-kit\/ folder; its files are under zipp-1\.2\.2-wasm-embed-kit \(is it another release's kit\?\)/);
});

test('the kit has to describe the release: BUILD-INFO.txt and modules.json name its version, the modules\' commit and both modules by their SHA256SUMS digests', (t) => {
  const kitRefused = (kit, pattern) => refused(install(t, releaseFolder(t, { kit })), pattern);
  const E40 = 'e'.repeat(40);
  kitRefused((files) => setBuildInfo(files, 'version', '9.9.9'), /zipp-1\.2\.3-wasm-embed-kit\.zip is not the ZIPP v1\.2\.3 embed kit of commit c{40}:\n {2}- BUILD-INFO\.txt says version 9\.9\.9, not 1\.2\.3/);
  kitRefused((files) => setBuildInfo(files, 'commit', E40), /BUILD-INFO\.txt commit e{40} is not the modules' commit c{40}/);
  kitRefused((files) => setJson(files, 'modules.json', (m) => void (m.version = '9.9.9')), /modules\.json says version 9\.9\.9, not 1\.2\.3/);
  kitRefused((files) => setJson(files, 'modules.json', (m) => void (m.commit = E40)), /modules\.json commit e{40} is not the modules' commit c{40}/);
  kitRefused((files) => setJson(files, 'modules.json', (m) => void (m.modules.find((e) => e.file === moduleName(VERSION, PYTHON)).sha256 = 'e'.repeat(64))), /modules\.json gives zipp-1\.2\.3-javascript-python\.wasm sha256 e{64}; the v1\.2\.3 SHA256SUMS says [0-9a-f]{64}/);
  kitRefused((files) => setJson(files, 'modules.json', (m) => void (m.modules = m.modules.filter((e) => e.file !== moduleName(VERSION, WEB)))), /modules\.json does not list zipp-1\.2\.3-javascript\.wasm$/m);
  kitRefused((files) => setJson(files, 'modules.json', (m) => void m.modules.push({ ...m.modules[0] })), /modules\.json lists zipp-1\.2\.3-javascript\.wasm 2 times/);
  kitRefused((files) => files.set('modules.json', Buffer.from('{ not json')), /zipp-1\.2\.3-wasm-embed-kit\.zip modules\.json is not JSON/);
});

test('each module has to be its own variant by what its zipp.meta and its stack say, whatever the file is called', (t) => {
  const asWeb = (parts) => Object.assign(parts, { stack: WEB.stackBytes, meta: { ...parts.meta, variant: WEB.variant, languages: WEB.languages } });
  refused(install(t, releaseFolder(t, { edit: { 'web-python': asWeb } })), /zipp-1\.2\.3-javascript-python\.wasm is not the ZIPP v1\.2\.3 javascript-python engine:[\s\S]*zipp\.meta variant is javascript, not javascript-python[\s\S]*its stack is 1048576 bytes, not 16777216/);
  const asPython = (parts) => Object.assign(parts, { stack: PYTHON.stackBytes, meta: { ...parts.meta, variant: PYTHON.variant, languages: PYTHON.languages } });
  refused(install(t, releaseFolder(t, { edit: { web: asPython } })), /zipp-1\.2\.3-javascript\.wasm is not the ZIPP v1\.2\.3 javascript engine:[\s\S]*zipp\.meta variant is javascript-python, not javascript$/m);
  const each = (suffix, edit, pattern) => refused(install(t, releaseFolder(t, { edit: { [suffix]: edit } })), pattern);
  each('web-python', (parts) => void (parts.stack = 1048576), /its stack is 1048576 bytes, not 16777216/);
  each('web-python', (parts) => void (parts.stackType = 'i64'), /its stack is not readable \(its first global is not an i32 stack pointer\), not 16777216/);
  each('web', (parts) => void (parts.meta.languages = ['javascript', 'python']), /zipp\.meta languages are \["javascript","python"\], not \["javascript"\]/);
  each('web', (parts) => void (parts.meta.version = '9.9.9'), /zipp\.meta says version 9\.9\.9, not 1\.2\.3/);
  each('web', (parts) => void (parts.meta.format = 2), /zipp\.meta format is 2, not 1/);
  // The two modules under each other's names, with a SHA256SUMS to match: named for what it holds, not what it is called.
  const swapped = releaseFolder(t);
  fs.writeFileSync(path.join(swapped.dir, moduleName(VERSION, WEB)), swapped.modules['web-python']);
  fs.writeFileSync(path.join(swapped.dir, moduleName(VERSION, PYTHON)), swapped.modules.web);
  fs.writeFileSync(path.join(swapped.dir, 'SHA256SUMS'), `${sha256(swapped.modules['web-python'])}  ${moduleName(VERSION, WEB)}\n${sha256(swapped.modules.web)}  ${moduleName(VERSION, PYTHON)}\n${sha256(swapped.kit)}  ${kitName(VERSION)}\n`);
  refused(install(t, swapped), /zipp-1\.2\.3-javascript\.wasm is not the ZIPP v1\.2\.3 javascript engine:[\s\S]*zipp\.meta variant is javascript-python, not javascript/);
});

test('a module has to carry one zipp.meta record and one zipp.glue section, name a full commit, and carry the glue its record names', (t) => {
  const module = (edit, suffix = 'web') => install(t, releaseFolder(t, { edit: { [suffix]: edit } }));
  refused(module((parts) => parts.custom.push(['zipp.meta', Buffer.from(JSON.stringify(parts.meta))])), /zipp-1\.2\.3-javascript\.wasm carries 2 zipp\.meta records, not one/);
  refused(module((parts) => void (parts.record = null)), /zipp-1\.2\.3-javascript\.wasm carries 0 zipp\.meta records, not one/);
  refused(module((parts) => void (parts.record = Buffer.from('{ not json'))), /zipp-1\.2\.3-javascript\.wasm's zipp\.meta record is not a JSON object/);
  refused(module((parts) => void (parts.glue = null)), /zipp-1\.2\.3-javascript\.wasm carries 0 zipp\.glue sections, not one/);
  refused(module((parts) => parts.custom.push(['zipp.glue', standInGlue(parts.profile)])), /zipp-1\.2\.3-javascript\.wasm carries 2 zipp\.glue sections, not one/);
  refused(module((parts) => void (parts.meta.glueSha256 = 'e'.repeat(64)), 'web-python'), /zipp-1\.2\.3-javascript-python\.wasm is not the ZIPP v1\.2\.3 javascript-python engine:\n {2}- its zipp\.glue section has sha256 [0-9a-f]{64}; zipp\.meta says glueSha256 e{64}/);
  refused(module((parts) => void (parts.meta.commit = 'cccccccc'), 'web-python'), /zipp\.meta commit "cccccccc" is not a full 40-hex commit/);
  refused(module((parts) => void (parts.bytes = Buffer.from('not a WebAssembly module'))), /zipp-1\.2\.3-javascript\.wasm is not a WebAssembly module/);
});

test('modules that name different commits are not one release, even when each checks on its own', (t) => {
  const other = 'd'.repeat(40);
  const edit = { web: (parts) => {
    parts.meta.commit = other;
    parts.profile.source = { sha: other };
  } };
  refused(install(t, releaseFolder(t, { edit })), /the ZIPP v1\.2\.3 modules name different commits: web d{40}, web-python c{40}/);
});

test('each module, loaded under its own glue, has to describe itself as its variant of the release: version, languages, the safe sandbox, the commit', (t) => {
  const saying = (suffix, change) => releaseFolder(t, { edit: { [suffix]: (parts) => Object.assign(parts.profile, change) } });
  refused(install(t, saying('web-python', { version: '9.9.9' })), /zippProfile\(\) version is 9\.9\.9, not 1\.2\.3/);
  refused(install(t, saying('web-python', { languages: ['javascript'] })), /zippProfile\(\) languages are \["javascript"\], not \["javascript","python"\]/);
  refused(install(t, saying('web', { languages: ['javascript', 'python'] })), /zippProfile\(\) languages are \["javascript","python"\], not \["javascript"\]/);
  refused(install(t, saying('web', { features: ['meter-only'] })), /zippProfile\(\) lacks the safe-sandbox feature/);
  refused(install(t, saying('web', { source: { sha: 'b'.repeat(40) } })), /the zipp-1\.2\.3-javascript\.wasm engine does not describe itself as ZIPP v1\.2\.3 javascript:\n {2}- zippProfile\(\) source\.sha is b{40}, not the zipp\.meta commit c{40}/);
  const broken = releaseFolder(t, { edit: { web: (parts) => void (parts.glue = Buffer.from('export function initSync() { throw new Error("trap"); }\n')) } });
  refused(install(t, broken), /the zipp-1\.2\.3-javascript\.wasm engine does not load under its own glue: trap/);
});

test('ZIPP_SUMS_SHA256 pins the release\'s SHA256SUMS', (t) => {
  refused(install(t, releaseFolder(t), { env: { ZIPP_SUMS_SHA256: '0'.repeat(64) } }), /ZIPP_SUMS_SHA256 says 0{64}/);
  refused({ run: cli(['--ensure'], { ZIPP_SUMS_SHA256: 'not-a-digest' }) }, /ZIPP_SUMS_SHA256 must be a sha256/);
  const folder = releaseFolder(t);
  assert.equal(install(t, folder, { env: { ZIPP_SUMS_SHA256: sha256(folder.top).toUpperCase() } }).run.status, 0);
});

test('a SHA256SUMS other than ZIPP_SUMS_SHA256 is refused before either module or the kit is downloaded', async (t) => {
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

test('a folder or SHA256SUMS without both modules and the kit is refused', (t) => {
  for (const name of [...VARIANTS.map((v) => moduleName(VERSION, v)), kitName(VERSION)]) {
    const folder = releaseFolder(t);
    fs.rmSync(path.join(folder.dir, name));
    refused(install(t, folder), new RegExp(`holds no ${escaped(name)}$`, 'm'));
  }
  refused(install(t, releaseFolder(t, { sums: (text) => dropLine(text, moduleName(VERSION, WEB)) })), /does not list zipp-1\.2\.3-javascript\.wasm$/m);
  refused(install(t, releaseFolder(t, { sums: (text) => dropLine(text, kitName(VERSION)) })), /does not list zipp-1\.2\.3-wasm-embed-kit\.zip$/m);
  refused(install(t, releaseFolder(t), { args: ['v1.2.4'] }), /does not list zipp-1\.2\.4-javascript\.wasm, zipp-1\.2\.4-javascript-python\.wasm or zipp-1\.2\.4-wasm-embed-kit\.zip/);
});

test('a release published as zip bundles, as every one before v0.0.26 was, is refused by name wherever it comes from', async (t) => {
  const zipSums = Buffer.from(`${'1'.repeat(64)}  zipp-0.0.24-linux-x86_64\n${'2'.repeat(64)}  zipp-wasm-0.0.24-web.zip\n${'3'.repeat(64)}  zipp-wasm-0.0.24-web-python.zip\n`);
  const named = /lists zip bundles \(zipp-wasm-0\.0\.24-web\.zip, zipp-wasm-0\.0\.24-web-python\.zip\) and no WebAssembly modules: releases before v0\.0\.26 publish zip bundles, which this installer no longer reads; install v0\.0\.26 or later/;
  const dir = tempDir(t, 'zipp-zip-release-');
  fs.writeFileSync(path.join(dir, 'SHA256SUMS'), zipSums);
  for (const name of ['zipp-wasm-0.0.24-web.zip', 'zipp-wasm-0.0.24-web-python.zip']) fs.writeFileSync(path.join(dir, name), 'a zip bundle');
  // The release a folder holds, and one named.
  refused(install(t, { dir }), named);
  refused(install(t, { dir }, { env: { ZIPP_RELEASE: 'v0.0.24' } }), named);
  // The latest, and a tag, from GitHub.
  const latestUrl = `${REPOSITORY}/releases/latest/download/SHA256SUMS`;
  const tagUrl = `${REPOSITORY}/releases/download/v0.0.24/SHA256SUMS`;
  const { fetch, calls } = serve(new Map([[latestUrl, zipSums], [tagUrl, zipSums]]));
  await assert.rejects(resolveRelease({ fetch }), (error) => error instanceof ZippReleaseError && named.test(error.message));
  await assert.rejects(resolveRelease({ tag: 'v0.0.24', fetch }), (error) => error instanceof ZippReleaseError && named.test(error.message));
  await assert.rejects(installRelease({ vendorDir: outDir(t), latest: true, cacheDir: tempDir(t, 'zipp-cache-'), fetch, log: quiet }), named);
  assert.ok(calls.every((url) => url.endsWith('/SHA256SUMS')), 'no bundle is downloaded');
});

test('an install replaces both whole folders, so a stale file does not survive', (t) => {
  const out = outDir(t);
  for (const variant of VARIANTS) {
    fs.mkdirSync(path.join(out, variant.folder), { recursive: true });
    fs.writeFileSync(path.join(out, variant.folder, 'README.md'), 'a committed engine\'s readme');
    // What an install of a zip bundle held and this one does not.
    for (const name of ['BUILD-INFO.txt', 'SHA256SUMS', 'zipp_wasm_bg.wasm.d.ts']) fs.writeFileSync(path.join(out, variant.folder, name), 'from a bundle');
  }
  const { run } = install(t, releaseFolder(t), { out });
  assert.equal(run.status, 0, `${run.stdout}\n${run.stderr}`);
  for (const variant of VARIANTS) assert.deepEqual(fs.readdirSync(path.join(out, variant.folder)).sort(), installedFiles(readSource(path.join(out, variant.folder)).notices).sort());
  assert.deepEqual(fs.readdirSync(out).sort(), ['zipp-wasm', 'zipp-wasm-python'], 'neither the stages nor the previous installs are left behind');
});

test('--check fails after a one-byte edit, on an engine copied over the other, and on installs of two releases', (t) => {
  const { out, run } = install(t, releaseFolder(t));
  assert.equal(run.status, 0, run.stderr);
  const dir = path.join(out, WEB.folder);
  const file = path.join(dir, 'zipp_wasm.d.ts');
  const original = fs.readFileSync(file);
  const edited = Buffer.from(original);
  edited[0] ^= 0x01;
  fs.writeFileSync(file, edited);
  refused({ run: cli(['--check'], { ZIPP_OUT: out }) }, /zipp-wasm does not check[\s\S]*zipp_wasm\.d\.ts is not the kit's zipp-engine\.d\.ts recorded/);
  fs.writeFileSync(file, original);
  // One byte of the glue the module carries: neither the module nor its glue is the one installed now.
  const wasmFile = path.join(dir, 'zipp_wasm_bg.wasm');
  const wasm = fs.readFileSync(wasmFile);
  const changed = Buffer.from(wasm);
  changed[changed.lastIndexOf('zippProfile')] ^= 0x20;
  fs.writeFileSync(wasmFile, changed);
  refused({ run: cli(['--check'], { ZIPP_OUT: out }) }, /zipp_wasm_bg\.wasm is not the recorded [0-9a-f]{64}[\s\S]*its zipp\.glue section has sha256 [0-9a-f]{64}; zipp\.meta says glueSha256 [0-9a-f]{64}[\s\S]*zipp_wasm\.js is not the glue zipp_wasm_bg\.wasm carries/);
  fs.writeFileSync(wasmFile, wasm);
  assert.equal(checkInstalls(out).release, TAG);

  // Each folder checks on its own; together they are not one ZIPP.
  const older = install(t, releaseFolder(t, { version: '1.2.2', commit: 'e'.repeat(40) }));
  const newer = install(t, releaseFolder(t));
  assert.equal(older.run.status + newer.run.status, 0);
  fs.rmSync(path.join(newer.out, PYTHON.folder), { recursive: true });
  fs.cpSync(path.join(older.out, PYTHON.folder), path.join(newer.out, PYTHON.folder), { recursive: true });
  refused({ run: cli(['--check'], { ZIPP_OUT: newer.out }) }, /are not one release[\s\S]*release differs: zipp-wasm v1\.2\.3, zipp-wasm-python v1\.2\.2[\s\S]*revision differs/);
  // The Python engine copied over the JavaScript one is not the JavaScript install.
  const swapped = install(t, releaseFolder(t));
  fs.copyFileSync(path.join(swapped.out, PYTHON.folder, 'zipp_wasm_bg.wasm'), path.join(swapped.out, WEB.folder, 'zipp_wasm_bg.wasm'));
  refused({ run: cli(['--check'], { ZIPP_OUT: swapped.out }) }, /zipp_wasm_bg\.wasm is not the recorded [0-9a-f]{64}[\s\S]*zipp_wasm_bg\.wasm: zipp\.meta variant is javascript-python, not javascript[\s\S]*zipp_wasm\.js is not the glue zipp_wasm_bg\.wasm carries[\s\S]*zipp_wasm_bg\.wasm: its stack is 16777216 bytes, not 1048576/);
});

test('--check holds each install to everything its SOURCE.json records, one invariant at a time', (t) => {
  const { out: pristine, run } = install(t, releaseFolder(t));
  assert.equal(run.status, 0, run.stderr);
  assert.equal(checkInstalls(pristine).release, TAG);
  const E64 = 'e'.repeat(64);
  const E40 = 'e'.repeat(40);
  const asPublished = (variant, edit) => (dir) => replaceModule(path.dirname(dir), variant, rebuilt(variant, edit));
  const cases = [
    [PYTHON, 'ZIPP\'s SHA256SUMS edited', (dir) => fs.appendFileSync(path.join(dir, 'RELEASE-SHA256SUMS'), '\n'), /RELEASE-SHA256SUMS has sha256 [0-9a-f]{64}, not the recorded [0-9a-f]{64}/],
    [WEB, 'a recorded module digest', (dir) => editSource(dir, (s) => void (s.bundleSha256 = E64)), /SOURCE\.json sha256 [0-9a-f]{64} is not the bundleSha256 e{64}; the module is installed as published[\s\S]*RELEASE-SHA256SUMS does not list zipp-1\.2\.3-javascript\.wasm with the recorded e{64}/],
    [WEB, 'a recorded revision', (dir) => editSource(dir, (s) => void (s.revision = E40)), /zipp_wasm_bg\.wasm: zipp\.meta commit c{40} is not the recorded revision e{40}/],
    [PYTHON, 'a recorded stack', (dir) => editSource(dir, (s) => void (s.stackBytes = 1048576)), /SOURCE\.json records javascript-python \["javascript","python"\] with a 1048576-byte stack, not the javascript-python engine/],
    [WEB, 'the JavaScript install recorded as Python', (dir) => editSource(dir, (s) => void (s.languages = ['javascript', 'python'])), /not the javascript engine/],
    [PYTHON, 'a recorded engine digest', (dir) => editSource(dir, (s) => void (s.sha256 = E64)), /zipp_wasm_bg\.wasm is not the recorded e{64}/],
    [WEB, 'a recorded glue digest', (dir) => editSource(dir, (s) => void (s.glueSha256 = E64)), /zipp_wasm_bg\.wasm: zipp\.meta glueSha256 [0-9a-f]{64} is not the recorded e{64}[\s\S]*zipp_wasm\.js is not the recorded e{64}/],
    [PYTHON, 'a recorded profile digest', (dir) => editSource(dir, (s) => void (s.profileSha256 = E64)), /PROFILE\.json is not the recorded e{64}/],
    [WEB, 'a recorded kit digest', (dir) => editSource(dir, (s) => void (s.kit.sha256 = E64)), /RELEASE-SHA256SUMS does not list zipp-1\.2\.3-wasm-embed-kit\.zip with the recorded e{64}/],
    [PYTHON, 'another release\'s kit', (dir) => editSource(dir, (s) => void (s.kit.file = 'zipp-1.2.2-wasm-embed-kit.zip')), /SOURCE\.json kit "zipp-1\.2\.2-wasm-embed-kit\.zip" is not the release's embed kit \(zipp-1\.2\.3-wasm-embed-kit\.zip\) with its sha256/],
    [WEB, 'a recorded kit file digest', (dir) => editSource(dir, (s) => void (s.kit.files['zipp_wasm.d.ts'].sha256 = E64)), /zipp_wasm\.d\.ts is not the kit's zipp-engine\.d\.ts recorded \(e{64}\)/],
    [PYTHON, 'a kit file recorded as another', (dir) => editSource(dir, (s) => void (s.kit.files['LICENSE-APACHE'].from = 'README.md')), /SOURCE\.json kit\.files records LICENSE-APACHE as \{"from":"README\.md",[^}]*\}, not the kit's LICENSE-APACHE with its sha256/],
    [PYTHON, 'a recorded notices digest', (dir) => editSource(dir, (s) => void (s.notices.sha256 = E64)), /THIRD_PARTY_LICENSES\.txt is not the recorded e{64}/],
    [PYTHON, 'notices of unknown origin', (dir) => editSource(dir, (s) => void (s.notices.source = 'elsewhere')), /SOURCE\.json notices .* are not recorded for the javascript-python engine/],
    [PYTHON, 'no notices recorded', (dir) => {
      editSource(dir, (s) => void (s.notices = null));
      fs.rmSync(path.join(dir, NOTICES));
    }, /SOURCE\.json notices null are not recorded for the javascript-python engine/],
    [WEB, 'curated notices in the JavaScript install', (dir) => editSource(dir, (s) => void (s.notices = { file: NOTICES, source: 'oaiy-curated', sha256: E64 })), /THIRD_PARTY_LICENSES\.txt is missing[\s\S]*notices .* are not recorded for the javascript engine/],
    [PYTHON, 'curated notices recorded as the release\'s', (dir) => editSource(dir, (s) => void (s.notices.source = 'zipp-release')), /SOURCE\.json kit\.files records THIRD_PARTY_LICENSES\.txt as undefined, not the kit's THIRD_PARTY_LICENSES\.txt with its sha256/],
    [WEB, 'another repository', (dir) => editSource(dir, (s) => void (s.repository = 'https://example.com/zipp')), /SOURCE\.json repository is https:\/\/example\.com\/zipp/],
    [WEB, 'a local build', (dir) => editSource(dir, (s) => void (s.build = 'local')), /SOURCE\.json build is local, not a verified release install/],
    [PYTHON, 'a release that is not the version', (dir) => editSource(dir, (s) => void (s.release = 'v9.9.9')), /SOURCE\.json release v9\.9\.9 and version 1\.2\.3 do not agree/],
    [PYTHON, 'the JavaScript module recorded', (dir) => editSource(dir, (s) => void (s.bundle = 'zipp-1.2.3-javascript.wasm')), /SOURCE\.json bundle zipp-1\.2\.3-javascript\.wasm is not the javascript-python module \(zipp-1\.2\.3-javascript-python\.wasm\)/],
    [WEB, 'a module of another commit, recorded as published', asPublished(WEB, (parts) => void (parts.meta.commit = E40)), /zipp_wasm_bg\.wasm: zipp\.meta commit e{40} is not the recorded revision c{40}/],
    [PYTHON, 'a module with another stack, recorded as published', asPublished(PYTHON, (parts) => void (parts.stack = 1048576)), /zipp_wasm_bg\.wasm: its stack is 1048576 bytes, not 16777216/],
    [WEB, 'a module with two glue sections, recorded as published', asPublished(WEB, (parts) => parts.custom.push(['zipp.glue', standInGlue(parts.profile)])), /zipp_wasm_bg\.wasm carries 2 zipp\.glue sections, not one/],
    [WEB, 'a zipp_wasm.js that is not the module\'s glue, recorded', (dir) => {
      fs.appendFileSync(path.join(dir, 'zipp_wasm.js'), '// and one more line\n');
      editSource(dir, (s) => void (s.glueSha256 = sha256(fs.readFileSync(path.join(dir, 'zipp_wasm.js')))));
    }, /zipp_wasm\.js is not the glue zipp_wasm_bg\.wasm carries \(its zipp\.glue section\)/],
    [PYTHON, 'PROFILE.json edited', (dir) => fs.appendFileSync(path.join(dir, 'PROFILE.json'), ' '), /PROFILE\.json is not the recorded [0-9a-f]{64}, the profile the engine reported at install/],
    [WEB, 'PROFILE.json of another version, recorded', (dir) => {
      editText(path.join(dir, 'PROFILE.json'), (text) => text.replace(`"version": "${VERSION}"`, '"version": "9.9.9"'));
      editSource(dir, (s) => void (s.profileSha256 = sha256(fs.readFileSync(path.join(dir, 'PROFILE.json')))));
    }, /PROFILE\.json says version 9\.9\.9, not 1\.2\.3/],
    [PYTHON, 'another engine', (dir) => fs.appendFileSync(path.join(dir, 'zipp_wasm_bg.wasm'), 'x'), /zipp_wasm_bg\.wasm is not the recorded [0-9a-f]{64}/],
    [PYTHON, 'edited notices, recorded', (dir) => {
      fs.appendFileSync(path.join(dir, NOTICES), 'and one more\n');
      editSource(dir, (s) => void (s.notices.sha256 = sha256(fs.readFileSync(path.join(dir, NOTICES)))));
    }, /THIRD_PARTY_LICENSES\.txt is not the curated copy in /],
    [WEB, 'a stray file', (dir) => fs.writeFileSync(path.join(dir, 'README.md'), ''), /README\.md is not part of the install/],
    [WEB, 'a file the zip bundles shipped', (dir) => fs.writeFileSync(path.join(dir, 'BUILD-INFO.txt'), 'version=1.2.3\n'), /BUILD-INFO\.txt is not part of the install/],
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
  const otherCurated = path.join(tempDir(t, 'zipp-curated-'), NOTICES);
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
  refused({ run: copy(['--resolve-only'], { ZIPP_RELEASE_DIR: folder.dir, ZIPP_RELEASE: 'v1.2.4' }) }, /SHA256SUMS in .* does not list zipp-1\.2\.4-javascript\.wasm/);
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
  const latestSums = Buffer.from(`${'3'.repeat(64)}  zipp-0.0.30-javascript.wasm\n${'4'.repeat(64)}  zipp-0.0.30-javascript-python.wasm\n${'5'.repeat(64)}  zipp-0.0.30-javascript-python-base.wasm\n${'6'.repeat(64)}  zipp-0.0.30-wasm-embed-kit.zip\n`);
  const stub = fetchStub(tree, 'latest.mjs', new Map([[`${REPOSITORY}/releases/latest/download/SHA256SUMS`, latestSums], [`${REPOSITORY}/releases/download/v0.0.30/SHA256SUMS`, latestSums]]));
  for (const args of [['--resolve-only'], ['--resolve-only', '--latest']]) {
    const latest = copy(args, {}, stub);
    assert.equal(latest.status, 0, latest.stderr);
    assert.deepEqual(JSON.parse(latest.stdout), { release: 'v0.0.30', sumsSha256: sha256(latestSums) });
  }
});

test('the latest release is taken only when its SHA256SUMS is the one under its tag and lists both modules and the kit', async () => {
  const latestUrl = `${REPOSITORY}/releases/latest/download/SHA256SUMS`;
  const tagUrl = `${REPOSITORY}/releases/download/v0.0.30/SHA256SUMS`;
  const line = (digit, name) => `${digit.repeat(64)}  ${name}\n`;
  const sums = (digit) => Buffer.from(['javascript.wasm', 'javascript-python.wasm', 'javascript-python-base.wasm', 'wasm-embed-kit.zip'].map((n) => line(digit, `zipp-0.0.30-${n}`)).join(''));
  const same = serve(new Map([[latestUrl, sums('1')], [tagUrl, sums('1')]]));
  const resolved = await resolveRelease({ fetch: same.fetch });
  assert.equal(resolved.release, 'v0.0.30');
  assert.ok(resolved.sums.equals(sums('1')));
  assert.deepEqual(same.calls, [latestUrl, tagUrl], 'no API call: the latest redirect, then the tag\'s own asset');
  const moved = serve(new Map([[latestUrl, sums('1')], [tagUrl, sums('2')]]));
  await assert.rejects(resolveRelease({ latest: true, fetch: moved.fetch }), (error) => error instanceof ZippReleaseError && /latest release changed while it was read, so try again/.test(error.message));
  // The base module is not the javascript-python one, however alike their names.
  const noPython = serve(new Map([[latestUrl, Buffer.from(line('1', 'zipp-0.0.30-javascript.wasm') + line('1', 'zipp-0.0.30-javascript-python-base.wasm'))]]));
  await assert.rejects(resolveRelease({ fetch: noPython.fetch }), /SHA256SUMS lists 0 javascript-python modules; expected exactly one/);
  const twoReleases = serve(new Map([[latestUrl, Buffer.from(line('1', 'zipp-0.0.30-javascript-python.wasm') + line('2', 'zipp-0.0.31-javascript-python.wasm'))], [tagUrl, sums('1')]]));
  await assert.rejects(resolveRelease({ fetch: twoReleases.fetch }), /SHA256SUMS lists 2 javascript-python modules; expected exactly one/);
  const noKit = Buffer.from(line('1', 'zipp-0.0.30-javascript.wasm') + line('1', 'zipp-0.0.30-javascript-python.wasm'));
  await assert.rejects(resolveRelease({ fetch: serve(new Map([[latestUrl, noKit], [tagUrl, noKit]])).fetch }), /the SHA256SUMS published under v0\.0\.30 does not list zipp-0\.0\.30-wasm-embed-kit\.zip$/);
  await assert.rejects(resolveRelease({ latest: true, tag: 'v0.0.30', fetch: same.fetch }), /name a release or --latest, not both/);
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

  // The same release published again with other bytes beside the engines (a CLI binary), so another SHA256SUMS.
  const again = releaseFolder(t, { version: '1.2.4', sums: (text) => text.replace('a'.repeat(64), 'f'.repeat(64)) });
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
  const served = published(folder);
  const { fetch, calls } = serve(served);
  assert.equal((await checkInstallsOnline(out, { fetch })).release, TAG);
  assert.deepEqual(calls.sort(), [releaseAsset('SHA256SUMS'), ...VARIANTS.map((v) => releaseAsset(moduleName(VERSION, v))), releaseAsset(kitName(VERSION))].sort(), 'the kit once, for both installs');

  // An edited file whose recorded digest was rewritten to match passes offline ...
  const edited = install(t, folder);
  const dir = path.join(edited.out, PYTHON.folder);
  fs.appendFileSync(path.join(dir, 'zipp_wasm.d.ts'), '\n');
  editSource(dir, (s) => void (s.kit.files['zipp_wasm.d.ts'].sha256 = sha256(fs.readFileSync(path.join(dir, 'zipp_wasm.d.ts')))));
  assert.equal(checkInstalls(edited.out).release, TAG);
  // ... and not against what the release publishes.
  await assert.rejects(checkInstallsOnline(edited.out, { fetch }), (error) => /\n {2}- zipp-wasm-python\/zipp_wasm\.d\.ts is not zipp-engine\.d\.ts in the published zipp-1\.2\.3-wasm-embed-kit\.zip/.test(error.message) && !/zipp-wasm\/zipp_wasm\.d\.ts/.test(error.message));

  served.set(releaseAsset('SHA256SUMS'), Buffer.concat([folder.top, Buffer.from('\n')]));
  await assert.rejects(checkInstallsOnline(out, { fetch }), /zipp-wasm\/RELEASE-SHA256SUMS is not the SHA256SUMS published under v1\.2\.3/);
  served.set(releaseAsset('SHA256SUMS'), folder.top);
  served.set(releaseAsset(moduleName(VERSION, WEB)), appendSection(folder.modules.web, 'name', Buffer.from('built again')));
  await assert.rejects(checkInstallsOnline(out, { fetch }), /zipp-wasm\/zipp_wasm_bg\.wasm is not the published zipp-1\.2\.3-javascript\.wasm/);
  served.set(releaseAsset(moduleName(VERSION, WEB)), folder.modules.web);
  served.set(releaseAsset(kitName(VERSION)), releaseFolder(t, { repack: true }).kit);
  await assert.rejects(checkInstallsOnline(out, { fetch }), /the published zipp-1\.2\.3-wasm-embed-kit\.zip is not the recorded [0-9a-f]{64}/);
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
  // Taken over while the modules download: refused before either folder is written.
  fs.rmSync(lock);
  const folder = releaseFolder(t);
  const served = published(folder);
  const fetch = async (url) => {
    if (url.endsWith('.wasm')) fs.writeFileSync(lock, other);
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
  const installs = await verifyRelease({ release: TAG, sums: folder.top, modules: folder.modules, kit: folder.kit });
  const { renameSync } = fs;
  for (const had of ['both installs', 'the JavaScript install only']) {
    const { out, run } = install(t, older);
    assert.equal(run.status, 0, run.stderr);
    if (had === 'the JavaScript install only') fs.rmSync(path.join(out, PYTHON.folder), { recursive: true });
    const before = snapshot(out);
    // The Python stage cannot take its place after the JavaScript one has.
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
    return writeFileSync.call(this, file, changed ? Buffer.from('not what the kit holds\n') : data, ...rest);
  });
  await assert.rejects(installRelease({ vendorDir: outDir(t), releaseDir: folder.dir, log: quiet, warn: quiet }), (error) => error instanceof ZippReleaseError && /zipp-wasm does not check[\s\S]*zipp_wasm\.d\.ts is not the kit's zipp-engine\.d\.ts recorded/.test(error.message));
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
  // A newer release cached as zip bundles, by a version of this script before v0.0.26: not something it can install now.
  const zipped = path.join(cacheDir, 'v1.2.9');
  fs.mkdirSync(zipped);
  fs.writeFileSync(path.join(zipped, 'SHA256SUMS'), `${'2'.repeat(64)}  zipp-wasm-1.2.9-web.zip\n${'3'.repeat(64)}  zipp-wasm-1.2.9-web-python.zip\n`);
  for (const name of ['zipp-wasm-1.2.9-web.zip', 'zipp-wasm-1.2.9-web-python.zip']) fs.writeFileSync(path.join(zipped, name), 'a zip bundle');
  const cached = path.join(cacheDir, TAG);
  await assert.rejects(resolveRelease({ tag: 'v1.2.2', cacheDir, fetch: offline }), new RegExp(`: fetch failed; to install the v1\\.2\\.2 downloaded earlier without the network, set ZIPP_RELEASE_DIR=${escaped(path.join(cacheDir, 'v1.2.2'))}$`));
  // With no tag, the newest one downloaded that can be installed.
  await assert.rejects(resolveRelease({ cacheDir, fetch: offline }), new RegExp(`set ZIPP_RELEASE_DIR=${escaped(cached)}$`));
  await assert.rejects(resolveRelease({ tag: 'v1.2.9', cacheDir, fetch: offline }), (error) => /: fetch failed$/.test(error.message));
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

test('the WebAssembly readers find a module\'s custom sections and its stack from its bytes, and nothing in anything else', () => {
  for (const variant of VARIANTS) {
    const parts = moduleParts(variant, { version: VERSION, commit: COMMIT });
    const bytes = buildModule(parts);
    assert.ok(WebAssembly.validate(bytes), `the ${variant.variant} fixture is a valid module`);
    assert.equal(wasmStackBytes(bytes), variant.stackBytes);
    assert.deepEqual(wasmCustomSections(bytes, 'zipp.glue'), [standInGlue(parts.profile)]);
    assert.deepEqual(JSON.parse(wasmCustomSections(bytes, 'zipp.meta')[0]), { ...parts.meta, glueSha256: sha256(standInGlue(parts.profile)) });
    assert.deepEqual(wasmCustomSections(bytes, 'zipp'), [], 'a section is found by its whole name');
  }
  assert.deepEqual([...sleb(1048576)], [0x80, 0x80, 0xc0, 0x00], 'i32.const takes a signed LEB128');
  const module = rebuilt(WEB);
  assert.equal(wasmCustomSections(module.subarray(0, module.length - 3), 'zipp.glue'), null, 'a module cut short');
  assert.equal(wasmCustomSections(Buffer.from('not a module'), 'zipp.meta'), null);
  assert.equal(wasmStackBytes(Buffer.from('not a module')), null);
  assert.equal(wasmStackBytes(wasmModule({ stack: -16 })), null, 'a negative stack pointer is no stack');
  assert.equal(wasmStackBytes(wasmModule({ stack: 1048576, stackType: 'i64' })), null, 'nor is an i64 global');
  assert.equal(wasmStackBytes(Buffer.concat([WASM_HEADER, customSection('zipp.meta', Buffer.from('{}'))])), null, 'no globals, no stack');
});

test('the curated notices must be the file their SOURCE.json records, and a release newer or older than the one they were generated from installs with a prompt to regenerate them', async (t) => {
  // In process the prompt goes to `warn` wherever this runs; the GitHub Actions form is checked through the command line below.
  const actions = process.env.GITHUB_ACTIONS;
  delete process.env.GITHUB_ACTIONS;
  t.after(() => {
    if (actions !== undefined) process.env.GITHUB_ACTIONS = actions;
  });
  const record = JSON.parse(fs.readFileSync(CURATED_RECORD, 'utf8'));
  const dir = tempDir(t, 'zipp-curated-');
  const curatedNotices = path.join(dir, NOTICES);
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
  // Older than the notices can need more than they name as well (releases before v0.0.21 compile in RustPython).
  const newer = await warned({ release: 'v1.2.4' });
  assert.equal(newer.length, 1, newer.join('\n'));
  assert.match(newer[0], /^ZIPP v1\.2\.3 is older than v1\.2\.4, the release .*THIRD_PARTY_LICENSES\.txt was generated from; if it compiles in third-party code that release no longer does, regenerate the notices: node scripts\/regen-zipp-notices\.mjs <zipp\.org checkout> v1\.2\.3$/);
  assert.deepEqual(await warned({ release: TAG }), [], `generated from ${TAG}`);

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
  const sources = { 'crates/rustpython-parser-fork/LICENSE': 'MIT License\n\nCopyright (c) 2020 RustPython Team\n', 'LICENSE-UNICODE': 'UNICODE LICENSE V3\n\nSPDX-License-Identifier: Unicode-3.0\n' };
  const read = (file) => sources[file];
  // Releases before v0.0.21 compiled in the RustPython parser fork, and their notices carry its licence.
  assert.deepEqual(noticeSources(TAG), { unicode: 'LICENSE-UNICODE' });
  assert.deepEqual(noticeSources('v0.0.20'), { rustpython: 'crates/rustpython-parser-fork/LICENSE', unicode: 'LICENSE-UNICODE' });
  const older = tempDir(t, 'zipp-notices-');
  assert.deepEqual(regenerate({ read, release: 'v0.0.20', dir: older }), []);
  const olderText = 'ZIPP engine: Apache-2.0. See the source repository for its complete notices.\n\nRustPython parser (MIT):\nMIT License\n\nCopyright (c) 2020 RustPython Team\n\n\nUnicode data:\nUNICODE LICENSE V3\n\nSPDX-License-Identifier: Unicode-3.0\n';
  assert.equal(fs.readFileSync(path.join(older, NOTICES), 'utf8'), olderText);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(older, 'SOURCE.json'), 'utf8')).sources, ['crates/rustpython-parser-fork/LICENSE', 'LICENSE-UNICODE']);
  const text = 'ZIPP engine: Apache-2.0. See the source repository for its complete notices.\n\nUnicode data:\nUNICODE LICENSE V3\n\nSPDX-License-Identifier: Unicode-3.0\n';
  const dir = tempDir(t, 'zipp-notices-');
  const notices = path.join(dir, NOTICES);
  assert.deepEqual(regenerate({ read, release: TAG, dir }), []);
  assert.equal(fs.readFileSync(notices, 'utf8'), text);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dir, 'SOURCE.json'), 'utf8')), { repository: REPOSITORY, release: TAG, sources: ['LICENSE-UNICODE'], file: NOTICES, sha256: sha256(Buffer.from(text)) });
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
