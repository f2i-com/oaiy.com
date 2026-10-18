#!/usr/bin/env node
/**
 * Install OAIY's two ZIPP engines from one published zipp.org release, and
 * check an install. Neither is committed: ui's test and build hooks run --ensure.
 *
 *   node scripts/fetch-zipp-release.mjs [vX.Y.Z]            install a release (no tag: the latest)
 *   node scripts/fetch-zipp-release.mjs --latest            install the latest release
 *   node scripts/fetch-zipp-release.mjs --check             verify both installs, offline
 *   node scripts/fetch-zipp-release.mjs --check --online    and against the published release
 *   node scripts/fetch-zipp-release.mjs --resolve-only [vX.Y.Z | --latest]   print {release, sumsSha256}
 *   node scripts/fetch-zipp-release.mjs --ensure            install unless both installs check
 *
 * Both engines come from the same release and the same SHA256SUMS:
 *   ui/vendor/zipp-wasm/         zipp-<v>-javascript.wasm, JavaScript only: the browser flow sandbox
 *   ui/vendor/zipp-wasm-python/  zipp-<v>-javascript-python.wasm: for the CLI, Desktop and headless server
 *
 * Environment: ZIPP_RELEASE=vX.Y.Z names the release; ZIPP_RELEASE_DIR=<dir>
 * reads SHA256SUMS, both modules and the embed kit from a folder instead of
 * GitHub (offline; with no tag, the release that folder holds);
 * ZIPP_SUMS_SHA256 is the digest the release's SHA256SUMS must have; ZIPP_OUT
 * installs into <dir>/zipp-wasm and <dir>/zipp-wasm-python instead of ui/vendor.
 *
 * OAIY builds no ZIPP crate, so nothing in the repository names a release: with
 * no tag the latest is taken. CI resolves it once per run (--resolve-only) and
 * hands every job ZIPP_RELEASE and ZIPP_SUMS_SHA256, so one run sees one ZIPP.
 * --ensure never looks for a newer release: an install that checks, and is the
 * release and SHA256SUMS the environment names (if it names one), is kept
 * without a request.
 *
 * Since v0.0.26 a ZIPP release publishes each WebAssembly engine as one plain
 * `.wasm` file that carries its own wasm-bindgen bindings in a `zipp.glue`
 * custom section, beside a `zipp.meta` record of its format, variant, version,
 * commit, languages and the glue's sha256. Releases before v0.0.26 published
 * zip bundles instead, which this script no longer reads.
 *
 * Each install takes its module as published, byte for byte, as
 * zipp_wasm_bg.wasm (its sha256 is its line in ZIPP's SHA256SUMS), and the
 * module's `zipp.glue` section, byte for byte, as zipp_wasm.js: the very
 * bindings the bundles shipped under that name, so nothing that loads an
 * install changes. The record must name that variant of that release, its
 * languages, a full commit and the glue's digest; the stack must be the
 * variant's (the initial value of the module's first global, Rust's stack
 * pointer, which starts at the top of the stack); both modules must name one
 * commit; and loaded under its own glue each module must report the release,
 * the safe sandbox, its languages and that commit. A release no longer
 * publishes PROFILE.json: the install writes what the engine's zippProfile()
 * reports, as the bundles wrote it, and records its digest.
 *
 * The release's embed kit, zipp-<v>-wasm-embed-kit.zip, gives both installs
 * the engine's TypeScript declarations (its zipp-engine.d.ts, installed as
 * zipp_wasm.d.ts) and LICENSE-APACHE. The kit must match SHA256SUMS, every file
 * in it its own SHA256SUMS, and its BUILD-INFO.txt and modules.json must name
 * the release, the modules' commit and each module's digest in SHA256SUMS.
 * Nothing is post-processed: a changed byte breaks that chain. The release
 * ships no third-party notices, so the Unicode notices come from
 * ui/vendor/zipp-notices/ ('oaiy-curated') until it does (a kit that ships a
 * THIRD_PARTY_LICENSES.txt is taken as 'zipp-release'). That folder's
 * SOURCE.json names the release they were generated from; any other release is
 * installed with a warning to regenerate them (scripts/regen-zipp-notices.mjs).
 *
 * Node built-ins only (the zip and WebAssembly section readers are below): the
 * CLI and Desktop CI lanes run this without ui's node_modules.
 */
import { createHash, randomBytes } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { isDeepStrictEqual } from 'node:util';
import { fileURLToPath } from 'node:url';

export const REPOSITORY = 'https://github.com/f2i-com/zipp.org';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const VENDOR_DIR = path.join(ROOT, 'ui', 'vendor');
export const CURATED_NOTICES = path.join(VENDOR_DIR, 'zipp-notices', 'THIRD_PARTY_LICENSES.txt');
export const CACHE_DIR = path.join(ROOT, '.zipp-release');

export const NOTICES = 'THIRD_PARTY_LICENSES.txt';
export const CURATED = 'oaiy-curated';
/** The first release published as plain modules; every release before it published zip bundles. */
export const FIRST_MODULE_RELEASE = 'v0.0.26';
export const META_SECTION = 'zipp.meta';
export const GLUE_SECTION = 'zipp.glue';
export const META_FORMAT = 1;
/**
 * The two engines of a release OAIY installs. `module` names its release file,
 * zipp-<v>-<module>.wasm (a file's name and its record's variant need not be
 * one word: ZIPP's javascript-lite module records variant "lite"); `suffix` is
 * the zip bundle each replaced ('web', 'web-python'), by which installs are
 * still keyed. Only the Python engine redistributes third-party (Unicode) data.
 */
export const VARIANTS = [
  { folder: 'zipp-wasm', suffix: 'web', module: 'javascript', variant: 'javascript', languages: ['javascript'], stackBytes: 1048576, needsNotices: false },
  { folder: 'zipp-wasm-python', suffix: 'web-python', module: 'javascript-python', variant: 'javascript-python', languages: ['javascript', 'python'], stackBytes: 16777216, needsNotices: true },
];
/** Taken from the embed kit byte for byte, by the name each is installed as. */
export const KIT_FILES = { 'zipp_wasm.d.ts': 'zipp-engine.d.ts', 'LICENSE-APACHE': 'LICENSE-APACHE' };
/** An install: the module and its glue, the kit's files, the engine's profile, ZIPP's top-level SHA256SUMS, SOURCE.json and, when recorded, the notices. */
export const installedFiles = (notices) => ['zipp_wasm_bg.wasm', 'zipp_wasm.js', ...Object.keys(KIT_FILES), 'PROFILE.json', 'RELEASE-SHA256SUMS', 'SOURCE.json', ...(notices ? [NOTICES] : [])];

/** A refusal: the release, a module, the kit or an install is not what it has to be. */
export class ZippReleaseError extends Error {}
const refuse = (message) => {
  throw new ZippReleaseError(message);
};
const fail = (what, problems) => {
  if (problems.length) refuse(`${what}:\n  - ${problems.join('\n  - ')}`);
};

export const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
export const isReleaseTag = (tag) => /^v\d+\.\d+\.\d+$/.test(tag ?? '');
/** A release's file for one of the engines OAIY installs. */
export const moduleName = (version, variant) => `zipp-${version}-${variant.module}.wasm`;
/** A release's embed kit, whose files are under the folder of the same name. */
export const kitName = (version) => `zipp-${version}-wasm-embed-kit.zip`;
const kitFolder = (version) => kitName(version).replace(/\.zip$/, '');
/** What a release must publish for an install: both modules and the kit. */
const releaseFiles = (version) => [...VARIANTS.map((v) => moduleName(version, v)), kitName(version)];
/** Negative, zero or positive as release tag `a` is older than, the same as or newer than `b`. */
export const compareReleases = (a, b) => {
  const [x, y] = [a, b].map((t) => t.slice(1).split('.').map(Number));
  return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
};
const HEX64 = /^[0-9a-f]{64}$/;
const COMMIT = /^[0-9a-f]{40}$/;
const rel = (p) => {
  const relative = path.relative(ROOT, p);
  return relative.startsWith('..') || path.isAbsolute(relative) ? p : relative.replaceAll('\\', '/') || '.';
};
/** `a`, `a or b`, `a, b or c`. */
const either = (names) => (names.length > 1 ? `${names.slice(0, -1).join(', ')} or ${names.at(-1)}` : names.join(''));

/** SHA256SUMS lines are `<hex>  <name>` (or `<hex> *<name>`). */
export function parseSums(text) {
  const sums = new Map();
  for (const line of String(text).split('\n')) {
    const m = /^([0-9a-f]{64})\s+\*?(.+?)\s*$/i.exec(line);
    if (m) sums.set(m[2], m[1].toLowerCase());
  }
  return sums;
}

/** BUILD-INFO.txt lines are `key=value`; a value may itself hold `=`. */
export function parseBuildInfo(text) {
  const info = {};
  for (const line of String(text).split('\n')) {
    const at = line.indexOf('=');
    if (at > 0) info[line.slice(0, at).trim()] = line.slice(at + 1).trim();
  }
  return info;
}

// ---------------------------------------------------------------------------
// What a module says about itself
// ---------------------------------------------------------------------------

/**
 * A reader over `b` from `at`, never past `end`: `byte()` and `uleb()` (an
 * unsigned LEB128) advance it, and throw past the end or on a malformed number.
 */
function reader(b, at, end) {
  const r = {
    at,
    byte() {
      if (r.at >= end) throw new RangeError('read past the end');
      return b[r.at++];
    },
    uleb() {
      let value = 0;
      for (let shift = 0; ; shift += 7) {
        if (shift > 28) throw new RangeError('malformed LEB128');
        const byte = r.byte();
        value += (byte & 0x7f) * 2 ** shift;
        if (!(byte & 0x80)) return value;
      }
    },
  };
  return r;
}

/**
 * A WebAssembly binary's sections as {id, start, end} (each payload's bounds),
 * or null for anything that is not a version 1 module, or runs past its end.
 * Read from the bytes; nothing is compiled, so --check can afford it.
 */
function wasmSections(b) {
  if (b.length < 8 || b.readUInt32LE(0) !== 0x6d736100 || b.readUInt32LE(4) !== 1) return null;
  const sections = [];
  const r = reader(b, 8, b.length);
  try {
    while (r.at < b.length) {
      const id = r.byte();
      const size = r.uleb();
      if (r.at + size > b.length) return null;
      sections.push({ id, start: r.at, end: r.at + size });
      r.at += size;
    }
  } catch {
    return null;
  }
  return sections;
}

/**
 * The payloads of every custom section called `name` in a WebAssembly binary,
 * in order, or null for anything that is not one. A custom section (id 0) may
 * appear any number of times, anywhere; ZIPP's release modules carry their
 * `zipp.meta` record and `zipp.glue` bindings this way.
 */
export function wasmCustomSections(bytes, name) {
  const b = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  const sections = wasmSections(b);
  if (!sections) return null;
  const found = [];
  try {
    for (const { id, start, end } of sections) {
      if (id !== 0) continue;
      const r = reader(b, start, end);
      const length = r.uleb();
      if (r.at + length > end) return null;
      if (b.toString('utf8', r.at, r.at + length) === name) found.push(b.subarray(r.at + length, end));
    }
  } catch {
    return null;
  }
  return found;
}

/**
 * The stack a WebAssembly module built by rustc was linked with, in bytes: the
 * initial value of its first defined global, which is Rust's stack pointer.
 * The wasm32 targets lay the stack out first and the pointer starts at its
 * top, so that value is the stack's size (`-z stack-size`). Null when the
 * first global is not a mutable i32 set by one constant, or for anything that
 * is not a module.
 */
export function wasmStackBytes(bytes) {
  const b = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  const globals = wasmSections(b)?.find((s) => s.id === 6);
  if (!globals) return null;
  try {
    const r = reader(b, globals.start, globals.end);
    if (r.uleb() < 1) return null;
    // i32, mutable, i32.const
    if (r.byte() !== 0x7f || r.byte() !== 0x01 || r.byte() !== 0x41) return null;
    // i32.const's operand is a signed LEB128: a last byte with bit 6 set is negative, never a stack.
    let value = 0;
    for (let shift = 0; ; shift += 7) {
      const byte = r.byte();
      value += (byte & 0x7f) * 2 ** shift;
      if (!(byte & 0x80)) {
        if (byte & 0x40 || shift > 28) return null;
        break;
      }
    }
    return r.byte() === 0x0b ? value : null;
  } catch {
    return null;
  }
}

/** A module's `zipp.meta` record, parsed; `what` names the module in a refusal. */
export function moduleMeta(bytes, what) {
  const records = wasmCustomSections(bytes, META_SECTION);
  if (records === null) refuse(`${what} is not a WebAssembly module`);
  if (records.length !== 1) refuse(`${what} carries ${records.length} ${META_SECTION} records, not one`);
  let meta;
  try {
    meta = JSON.parse(records[0].toString('utf8'));
  } catch {}
  if (!meta || typeof meta !== 'object' || Array.isArray(meta)) refuse(`${what}'s ${META_SECTION} record is not a JSON object`);
  return meta;
}

/** A module's `zipp.glue` section, the wasm-bindgen bindings it carries, which it must carry exactly once. */
export function moduleGlue(bytes, what) {
  const sections = wasmCustomSections(bytes, GLUE_SECTION);
  if (sections === null) refuse(`${what} is not a WebAssembly module`);
  if (sections.length !== 1) refuse(`${what} carries ${sections.length} ${GLUE_SECTION} sections, not one`);
  return Buffer.from(sections[0]);
}

/**
 * What a module's record must say to be `variant` of `version`, as problems:
 * format 1, that variant and its languages, a full commit (the recorded
 * `revision`, when there is one), and the digest of the glue it carries (and
 * the recorded `glueSha256`, when there is one).
 */
function metaProblems(meta, glue, variant, { version, revision, glueSha256 }) {
  const problems = [];
  if (meta.format !== META_FORMAT) problems.push(`${META_SECTION} format is ${JSON.stringify(meta.format)}, not ${META_FORMAT}`);
  if (meta.version !== version) problems.push(`${META_SECTION} says version ${meta.version}, not ${version}`);
  if (meta.variant !== variant.variant) problems.push(`${META_SECTION} variant is ${meta.variant}, not ${variant.variant}`);
  if (!isDeepStrictEqual(meta.languages, variant.languages)) problems.push(`${META_SECTION} languages are ${JSON.stringify(meta.languages)}, not ${JSON.stringify(variant.languages)}`);
  if (!COMMIT.test(meta.commit ?? '')) problems.push(`${META_SECTION} commit ${JSON.stringify(meta.commit)} is not a full 40-hex commit`);
  else if (revision !== undefined && meta.commit !== revision) problems.push(`${META_SECTION} commit ${meta.commit} is not the recorded revision ${revision}`);
  if (sha256(glue) !== meta.glueSha256) problems.push(`its ${GLUE_SECTION} section has sha256 ${sha256(glue)}; ${META_SECTION} says glueSha256 ${meta.glueSha256}`);
  else if (glueSha256 !== undefined && meta.glueSha256 !== glueSha256) problems.push(`${META_SECTION} glueSha256 ${meta.glueSha256} is not the recorded ${glueSha256}`);
  return problems;
}

/** The stack a module was linked with, as problems: the variant's, read from its first global. */
function stackProblems(wasm, variant) {
  const stack = wasmStackBytes(wasm);
  return stack === variant.stackBytes ? [] : [`its stack is ${stack === null ? 'not readable (its first global is not an i32 stack pointer)' : `${stack} bytes`}, not ${variant.stackBytes}`];
}

// ---------------------------------------------------------------------------
// Zip reading
// ---------------------------------------------------------------------------

/**
 * The files in a zip, by name. Stored and deflated entries only, as ZIPP's
 * release job writes them; every entry's size and CRC-32 are checked. The
 * sizes come from the central directory, so data descriptors need no parsing.
 */
export function unzip(zip) {
  if (typeof zlib.crc32 !== 'function') refuse(`reading ZIPP's embed kit needs zlib.crc32, in Node 20.15, 22.2 and newer (this is ${process.version})`);
  const buf = Buffer.isBuffer(zip) ? zip : Buffer.from(zip);
  const bad = (why) => refuse(`not a readable zip: ${why}`);
  let eocd = -1;
  for (let at = buf.length - 22; at >= Math.max(0, buf.length - 22 - 0xffff); at--) {
    if (buf.readUInt32LE(at) === 0x06054b50) {
      eocd = at;
      break;
    }
  }
  if (eocd < 0) bad('no end of central directory');
  const count = buf.readUInt16LE(eocd + 10);
  let at = buf.readUInt32LE(eocd + 16);
  if (count === 0xffff || at === 0xffffffff) bad('zip64 archives are not supported');
  const files = new Map();
  for (let i = 0; i < count; i++) {
    if (at + 46 > eocd || buf.readUInt32LE(at) !== 0x02014b50) bad(`central directory entry ${i} is malformed`);
    const flags = buf.readUInt16LE(at + 8);
    const method = buf.readUInt16LE(at + 10);
    const crc = buf.readUInt32LE(at + 16);
    const compressed = buf.readUInt32LE(at + 20);
    const size = buf.readUInt32LE(at + 24);
    const nameLength = buf.readUInt16LE(at + 28);
    const local = buf.readUInt32LE(at + 42);
    const name = buf.toString('utf8', at + 46, at + 46 + nameLength);
    at += 46 + nameLength + buf.readUInt16LE(at + 30) + buf.readUInt16LE(at + 32);
    if (name.endsWith('/')) continue;
    if (flags & 0x1) bad(`${name} is encrypted`);
    if (files.has(name)) bad(`${name} is in it twice`);
    if (local + 30 > buf.length || buf.readUInt32LE(local) !== 0x04034b50) bad(`${name} has no local header`);
    const start = local + 30 + buf.readUInt16LE(local + 26) + buf.readUInt16LE(local + 28);
    if (start + compressed > buf.length) bad(`${name} runs past the end of the zip`);
    const raw = buf.subarray(start, start + compressed);
    let bytes;
    if (method === 0) bytes = Buffer.from(raw);
    else if (method === 8) {
      try {
        bytes = zlib.inflateRawSync(raw, { maxOutputLength: Math.max(size, 1) });
      } catch (error) {
        bad(`${name} does not inflate to its recorded ${size} bytes (${error.message})`);
      }
    } else bad(`${name} uses compression method ${method}`);
    if (bytes.length !== size || zlib.crc32(bytes) !== crc) bad(`${name} fails its size or CRC-32`);
    files.set(name, bytes);
  }
  return files;
}

/** The kit's files, named inside its top folder. */
function kitEntries(zip, version) {
  const top = kitFolder(version);
  const all = unzip(zip);
  const files = new Map();
  for (const [name, bytes] of all) if (name.startsWith(`${top}/`)) files.set(name.slice(top.length + 1), bytes);
  if (!files.size) {
    const tops = [...new Set([...all.keys()].map((name) => name.split('/')[0]))];
    refuse(`${kitName(version)} has no ${top}/ folder; its files are under ${tops.join(', ') || 'nothing'} (is it another release's kit?)`);
  }
  return files;
}

// ---------------------------------------------------------------------------
// Where the release comes from
// ---------------------------------------------------------------------------
const releaseUrl = (tag, name) => `${REPOSITORY}/releases/download/${tag}/${name}`;
// GitHub's redirect to the newest release's asset: no API call, no token, no rate limit.
const LATEST_SUMS_URL = `${REPOSITORY}/releases/latest/download/SHA256SUMS`;

async function download(url, fetchImpl) {
  let response;
  try {
    response = await fetchImpl(url, { signal: AbortSignal.timeout(120_000) });
    if (response.ok) return Buffer.from(await response.arrayBuffer());
  } catch (error) {
    const unreachable = new ZippReleaseError(`could not fetch ${url}: ${error.message}`);
    unreachable.unreachable = true;
    throw unreachable;
  }
  refuse(`${url} answered ${response.status}`);
}

function readReleaseFile(releaseDir, name) {
  const file = path.join(releaseDir, name);
  if (!fs.existsSync(file)) refuse(`ZIPP_RELEASE_DIR ${releaseDir} holds no ${name}`);
  return fs.readFileSync(file);
}

/**
 * A SHA256SUMS of a release before v0.0.26, which lists zip bundles and no
 * modules, is refused by name: this script cannot read what it publishes.
 */
function refuseZipRelease(listed, what) {
  const bundles = [...listed.keys()].filter((name) => /^zipp-wasm-\d+\.\d+\.\d+-web(?:-python)?\.zip$/.test(name));
  if (bundles.length) refuse(`${what} lists zip bundles (${bundles.join(', ')}) and no WebAssembly modules: releases before ${FIRST_MODULE_RELEASE} publish zip bundles, which this installer no longer reads; install ${FIRST_MODULE_RELEASE} or later`);
}

/** The one release a SHA256SUMS lists a javascript-python module of. */
export function versionFromSums(sums) {
  const listed = parseSums(sums);
  const versions = [...listed.keys()].map((n) => /^zipp-(\d+\.\d+\.\d+)-javascript-python\.wasm$/.exec(n)?.[1]).filter(Boolean);
  if (!versions.length) refuseZipRelease(listed, 'SHA256SUMS');
  if (versions.length !== 1) refuse(`SHA256SUMS lists ${versions.length} javascript-python modules; expected exactly one`);
  return versions[0];
}

function requireFilesListed(sums, release, where) {
  const listed = parseSums(sums);
  const version = release.slice(1);
  const missing = releaseFiles(version).filter((name) => !listed.has(name));
  if (!missing.length) return;
  if (VARIANTS.every((v) => !listed.has(moduleName(version, v)))) refuseZipRelease(listed, `the SHA256SUMS ${where}`);
  refuse(`the SHA256SUMS ${where} does not list ${either(missing)}`);
}

/** A release downloaded earlier, named for ZIPP_RELEASE_DIR when GitHub cannot be reached; never used unasked. */
function cachedHint(error, cacheDir, tag) {
  if (!error.unreachable) return error;
  // Only a folder this script can install from: a release's modules and kit, never the zip bundles an older one cached.
  const complete = (t) => ['SHA256SUMS', ...releaseFiles(t.slice(1))].every((n) => fs.existsSync(path.join(cacheDir, t, n)));
  let tags = [];
  try {
    tags = fs.readdirSync(cacheDir).filter((t) => isReleaseTag(t) && (!tag || t === tag) && complete(t));
  } catch {}
  const newest = tags.sort(compareReleases).at(-1);
  if (!newest) return error;
  return new ZippReleaseError(`${error.message}; to install the ${newest} downloaded earlier without the network, set ZIPP_RELEASE_DIR=${path.join(cacheDir, newest)}`);
}

/**
 * The release to take and its top-level SHA256SUMS bytes, which must list both
 * modules and the embed kit. A tag is taken as named; ZIPP_RELEASE_DIR with no
 * tag is the release that folder holds; otherwise the latest, whose SHA256SUMS
 * must be byte for byte the one under its own tag, so the tag recorded is the
 * release really read.
 */
export async function resolveRelease({ tag, latest = false, releaseDir, cacheDir = CACHE_DIR, fetch: fetchImpl = globalThis.fetch } = {}) {
  if (latest && tag) refuse('name a release or --latest, not both');
  if (tag !== undefined && !isReleaseTag(tag)) refuse(`"${tag}" is not a ZIPP release tag (vMAJOR.MINOR.PATCH)`);
  if (releaseDir) {
    if (latest) refuse('ZIPP_RELEASE_DIR holds one release; leave out --latest');
    const sums = readReleaseFile(releaseDir, 'SHA256SUMS');
    const release = tag ?? `v${versionFromSums(sums)}`;
    requireFilesListed(sums, release, `in ${releaseDir}`);
    return { release, sums };
  }
  if (tag) {
    let sums;
    try {
      sums = await download(releaseUrl(tag, 'SHA256SUMS'), fetchImpl);
    } catch (error) {
      throw cachedHint(error, cacheDir, tag);
    }
    requireFilesListed(sums, tag, `published under ${tag}`);
    return { release: tag, sums };
  }
  let latestSums;
  try {
    latestSums = await download(LATEST_SUMS_URL, fetchImpl);
  } catch (error) {
    throw cachedHint(error, cacheDir);
  }
  const release = `v${versionFromSums(latestSums)}`;
  const tagged = await download(releaseUrl(release, 'SHA256SUMS'), fetchImpl);
  if (!latestSums.equals(tagged)) refuse(`the latest release's SHA256SUMS is not the one published under ${release}; the latest release changed while it was read, so try again`);
  requireFilesListed(tagged, release, `published under ${release}`);
  return { release, sums: tagged };
}

/** A release file (a module or the kit), checked against the top-level SHA256SUMS; downloads are cached per tag. */
async function loadReleaseFile({ release, sums, name, releaseDir, cacheDir = CACHE_DIR, fetch: fetchImpl = globalThis.fetch }) {
  const expected = parseSums(sums).get(name);
  if (!expected) refuse(`the ${release} SHA256SUMS does not list ${name}`);
  let bytes;
  if (releaseDir) bytes = readReleaseFile(releaseDir, name);
  else {
    const cached = path.join(cacheDir, release, name);
    bytes = fs.existsSync(cached) ? fs.readFileSync(cached) : null;
    if (!bytes || sha256(bytes) !== expected) bytes = await download(releaseUrl(release, name), fetchImpl);
  }
  const actual = sha256(bytes);
  if (actual !== expected) refuse(`${name} has sha256 ${actual}; the ${release} SHA256SUMS says ${expected}`);
  if (!releaseDir) {
    // Best effort, and whole files only: another install may be reading the cache.
    try {
      fs.mkdirSync(path.join(cacheDir, release), { recursive: true });
      for (const [file, content] of [['SHA256SUMS', sums], [name, bytes]]) {
        const target = path.join(cacheDir, release, file);
        const partial = `${target}.${process.pid}-${randomBytes(4).toString('hex')}.partial`;
        fs.writeFileSync(partial, content);
        try {
          renameWithRetry(partial, target);
        } finally {
          fs.rmSync(partial, { force: true });
        }
      }
    } catch {}
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// Verify a release and build the installs
// ---------------------------------------------------------------------------

/**
 * The curated notices and the release whose sources they were generated from,
 * as SOURCE.json beside them records it with their SHA-256.
 */
export function readCuratedNotices(file = CURATED_NOTICES) {
  const recordFile = path.join(path.dirname(file), 'SOURCE.json');
  const regenerate = 'regenerate both with node scripts/regen-zipp-notices.mjs';
  let record;
  try {
    record = JSON.parse(fs.readFileSync(recordFile, 'utf8'));
  } catch (error) {
    refuse(`${rel(recordFile)} does not record the curated notices (${error.code ?? error.message}); ${regenerate}`);
  }
  const bytes = fs.readFileSync(file);
  if (!isReleaseTag(record?.release) || record.sha256 !== sha256(bytes)) refuse(`${rel(file)} is not the file ${rel(recordFile)} records (${record?.release}, sha256 ${record?.sha256}); ${regenerate}`);
  return { bytes, release: record.release };
}

/** What a module reports about itself, loaded under its own glue; nothing is written to disk for this. */
async function engineProfile(glue, wasm) {
  // A data: URL per call: each import is a fresh module with its own instance,
  // so the second engine is not answered by the first.
  const source = Buffer.concat([glue, Buffer.from(`\n// ${randomBytes(8).toString('hex')}\n`)]);
  const module = await import(`data:text/javascript;base64,${source.toString('base64')}`);
  module.initSync({ module: wasm });
  return JSON.parse(module.zippProfile());
}

/** A release file must be the bytes its SHA256SUMS line names; returns that digest. */
function listedDigest(sums, release, name, bytes) {
  const listed = parseSums(sums).get(name);
  if (!listed) refuse(`the ${release} SHA256SUMS does not list ${name}`);
  if (sha256(bytes) !== listed) refuse(`${name} does not match the ${release} SHA256SUMS`);
  return listed;
}

/**
 * One module, verified all the way down: its line in SHA256SUMS, its record,
 * glue and stack, and what it reports loaded under that glue. Returns what an
 * install takes from it, PROFILE.json among them.
 */
async function verifyModule({ release, sums, wasm, variant }) {
  const version = release.slice(1);
  const name = moduleName(version, variant);
  const bundleSha256 = listedDigest(sums, release, name, wasm);
  const meta = moduleMeta(wasm, name);
  const glue = moduleGlue(wasm, name);
  fail(`${name} is not the ZIPP ${release} ${variant.variant} engine`, [...metaProblems(meta, glue, variant, { version }), ...stackProblems(wasm, variant)]);
  let profile;
  try {
    profile = await engineProfile(glue, wasm);
  } catch (error) {
    refuse(`the ${name} engine does not load under its own glue: ${error.message}`);
  }
  fail(`the ${name} engine does not describe itself as ZIPP ${release} ${variant.variant}`, [
    ...(profile?.version !== version ? [`zippProfile() version is ${profile?.version}, not ${version}`] : []),
    ...(!profile?.features?.includes('safe-sandbox') ? ['zippProfile() lacks the safe-sandbox feature'] : []),
    ...(!isDeepStrictEqual(profile?.languages, variant.languages) ? [`zippProfile() languages are ${JSON.stringify(profile?.languages)}, not ${JSON.stringify(variant.languages)}`] : []),
    ...(profile?.source?.sha !== meta.commit ? [`zippProfile() source.sha is ${profile?.source?.sha}, not the ${META_SECTION} commit ${meta.commit}`] : []),
  ]);
  // As the bundles published it: what the engine reports, two-space JSON and a newline.
  const profileJson = Buffer.from(`${JSON.stringify(profile, null, 2)}\n`);
  return { name, bundleSha256, meta, glue, profileJson };
}

/**
 * The embed kit, held to the release: its zip to SHA256SUMS, every file in it
 * to its own SHA256SUMS, its BUILD-INFO.txt and modules.json to the release and
 * `commit`, and both modules as modules.json names them to the digests
 * SHA256SUMS gives them. Returns the files both installs take from it, the
 * record SOURCE.json keeps of it, and its notices, if it ever ships them.
 */
function verifyKit({ release, sums, kit, commit }) {
  const version = release.slice(1);
  const name = kitName(version);
  const kitSha256 = listedDigest(sums, release, name, kit);
  const entries = kitEntries(kit, version);
  if (!entries.has('SHA256SUMS')) refuse(`${name} carries no SHA256SUMS`);
  const inner = parseSums(entries.get('SHA256SUMS'));
  // Everything the kit lists is what it lists, and everything taken from it is listed.
  for (const [file, digest] of inner) {
    if (!entries.has(file)) refuse(`${name} lists ${file} in its SHA256SUMS but does not carry it`);
    if (sha256(entries.get(file)) !== digest) refuse(`${file} in ${name} does not match the kit's SHA256SUMS`);
  }
  const taken = [...Object.entries(KIT_FILES), ...(entries.has(NOTICES) ? [[NOTICES, NOTICES]] : [])];
  for (const file of [...taken.map(([, from]) => from), 'BUILD-INFO.txt', 'modules.json']) {
    if (!entries.has(file)) refuse(`${name} has no ${file}`);
    if (!inner.has(file)) refuse(`${file} in ${name} is not listed in the kit's SHA256SUMS`);
  }

  const problems = [];
  const info = parseBuildInfo(entries.get('BUILD-INFO.txt'));
  if (info.version !== version) problems.push(`BUILD-INFO.txt says version ${info.version}, not ${version}`);
  if (info.commit !== commit) problems.push(`BUILD-INFO.txt commit ${info.commit} is not the modules' commit ${commit}`);
  let modules;
  try {
    modules = JSON.parse(entries.get('modules.json').toString('utf8'));
  } catch {
    refuse(`${name} modules.json is not JSON`);
  }
  if (modules?.version !== version) problems.push(`modules.json says version ${modules?.version}, not ${version}`);
  if (modules?.commit !== commit) problems.push(`modules.json commit ${modules?.commit} is not the modules' commit ${commit}`);
  const listed = parseSums(sums);
  const named = Array.isArray(modules?.modules) ? modules.modules : [];
  for (const variant of VARIANTS) {
    const file = moduleName(version, variant);
    const matches = named.filter((m) => m?.file === file);
    if (matches.length !== 1) problems.push(matches.length ? `modules.json lists ${file} ${matches.length} times` : `modules.json does not list ${file}`);
    else if (matches[0].sha256 !== listed.get(file)) problems.push(`modules.json gives ${file} sha256 ${matches[0].sha256}; the ${release} SHA256SUMS says ${listed.get(file)}`);
  }
  fail(`${name} is not the ZIPP ${release} embed kit of commit ${commit}`, problems);

  const files = new Map(Object.entries(KIT_FILES).map(([installed, from]) => [installed, entries.get(from)]));
  const record = { file: name, sha256: kitSha256, files: Object.fromEntries(taken.map(([installed, from]) => [installed, { from, sha256: sha256(entries.get(from)) }])) };
  return { files, record, notices: entries.get(NOTICES) ?? null };
}

/**
 * Verify a release's SHA256SUMS, both modules (`modules` by variant suffix)
 * and its embed kit, and return what to install, per variant. Refuses with
 * ZippReleaseError.
 */
export async function verifyRelease({ release, sums, modules, kit, expectSumsSha256, curatedNotices = CURATED_NOTICES }) {
  if (!isReleaseTag(release)) refuse(`"${release}" is not a ZIPP release tag`);
  if (expectSumsSha256 && sha256(sums) !== expectSumsSha256.toLowerCase()) refuse(`the ${release} SHA256SUMS has sha256 ${sha256(sums)}; ZIPP_SUMS_SHA256 says ${expectSumsSha256}`);
  const version = release.slice(1);
  const verified = [];
  for (const variant of VARIANTS) {
    if (!modules?.[variant.suffix]) refuse(`no ${moduleName(version, variant)} was given for ${release}`);
    verified.push({ variant, wasm: modules[variant.suffix], ...(await verifyModule({ release, sums, wasm: modules[variant.suffix], variant })) });
  }
  // One release job builds both; modules from two commits are not one ZIPP.
  if (new Set(verified.map((m) => m.meta.commit)).size !== 1) refuse(`the ZIPP ${release} modules name different commits: ${verified.map((m) => `${m.variant.suffix} ${m.meta.commit}`).join(', ')}`);
  const commit = verified[0].meta.commit;
  if (!kit) refuse(`no ${kitName(version)} was given for ${release}`);
  const { files: kitFiles, record: kitRecord, notices: kitNotices } = verifyKit({ release, sums, kit, commit });

  const installs = [];
  for (const { variant, wasm, name, bundleSha256, meta, glue, profileJson } of verified) {
    const files = new Map([['zipp_wasm_bg.wasm', wasm], ['zipp_wasm.js', glue], ...kitFiles, ['PROFILE.json', profileJson]]);
    let notices = null;
    let curatedFrom;
    if (kitNotices) {
      files.set(NOTICES, kitNotices);
      notices = { file: NOTICES, source: 'zipp-release', sha256: sha256(kitNotices) };
    } else if (variant.needsNotices) {
      if (!fs.existsSync(curatedNotices)) refuse(`ZIPP ${release} ships no ${NOTICES} and the curated copy ${rel(curatedNotices)} is missing`);
      const curated = readCuratedNotices(curatedNotices);
      files.set(NOTICES, curated.bytes);
      notices = { file: NOTICES, source: CURATED, sha256: sha256(curated.bytes) };
      curatedFrom = curated.release;
    }
    files.set('RELEASE-SHA256SUMS', sums);

    const source = {
      repository: REPOSITORY,
      release,
      version,
      revision: meta.commit,
      build: 'release',
      bundle: name,
      bundleSha256,
      sumsSha256: sha256(sums),
      variant: meta.variant,
      languages: meta.languages,
      stackBytes: wasmStackBytes(wasm),
      license: 'Apache-2.0',
      artifact: 'zipp_wasm_bg.wasm',
      sha256: sha256(wasm),
      glueSha256: sha256(glue),
      profileSha256: sha256(profileJson),
      kit: kitRecord,
      notices,
    };
    files.set('SOURCE.json', Buffer.from(`${JSON.stringify(source, null, 2)}\n`));
    installs.push({ variant, files, source, curatedFrom });
  }
  return installs;
}

// Windows refuses a rename now and then while a scanner holds a handle.
function renameWithRetry(from, to) {
  for (let attempt = 0; ; attempt++) {
    try {
      return fs.renameSync(from, to);
    } catch (error) {
      if (attempt >= 5 || !['EPERM', 'EBUSY', 'EACCES'].includes(error.code)) throw error;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100 * (attempt + 1));
    }
  }
}

const pidAlive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === 'EPERM';
  }
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// An install takes seconds, and minutes at most (every request times out at
// 120 s): a lock this old outlived its holder even if its pid was reused.
const LOCK_STALE_MS = 10 * 60_000;
const LOCK_FILE = '.zipp-wasm.lock';
// Windows refuses to create or open a file another process is deleting that
// instant; under contention that passes in milliseconds. Elsewhere these are real.
const lockBusy = (error) => process.platform === 'win32' && ['EPERM', 'EBUSY', 'EACCES'].includes(error.code);
/** The locks this process holds: a lock naming this pid is live only if it is one of them. */
const heldLocks = new Set();
const lockToken = () => `${process.pid} ${randomBytes(8).toString('hex')}`;

/** A lock file as {text, pid, age}, or null once it is gone. Its text is `<pid> <nonce>`. */
function readLock(file) {
  try {
    const text = fs.readFileSync(file, 'utf8');
    return { text, pid: Number(text.split(' ')[0]), age: Date.now() - fs.statSync(file).mtimeMs };
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

/** Whether the install a lock names may still be running. */
function lockLive({ text, pid, age }) {
  if (age > LOCK_STALE_MS) return false;
  // An empty lock is one being written this instant, unless it has been empty a while.
  if (!(Number.isInteger(pid) && pid > 0)) return age <= 30_000;
  // This pid but not a lock this process holds: an install that had the pid before.
  if (pid === process.pid) return heldLocks.has(text);
  return pidAlive(pid);
}

/** Remove a lock this process wrote, only while it still holds `token`. */
function releaseLock(file, token) {
  try {
    if (fs.readFileSync(file, 'utf8') === token) fs.rmSync(file, { force: true });
  } catch {}
}

/**
 * Remove the dead lock `stale` unless it has changed since it was read.
 * Waiters take turns at this under `<lock>.break`: otherwise two could find
 * the same dead lock, and the second remove the live one the first had just
 * written, so both would install. A turn left by a waiter that died is cleared.
 * False when something could not be removed or written (busy, or read-only).
 */
export async function removeStaleLock(lock, stale) {
  const breaker = `${lock}.break`;
  const token = lockToken();
  let done = true;
  try {
    fs.writeFileSync(breaker, token, { flag: 'wx' });
  } catch (error) {
    // Another waiter's turn, or one whose waiter died, which is cleared for the next.
    if (error.code !== 'EEXIST') done = false;
    else {
      try {
        const other = readLock(breaker);
        if (other && !lockLive(other) && fs.readFileSync(breaker, 'utf8') === other.text) fs.rmSync(breaker, { force: true });
      } catch {
        done = false;
      }
    }
    await sleep(20);
    return done;
  }
  try {
    const now = readLock(lock);
    if (now && now.text === stale.text && !lockLive(now)) fs.rmSync(lock, { force: true });
  } catch {
    done = false;
  } finally {
    releaseLock(breaker, token);
  }
  // The next look decides; a pause, not a spin.
  if (!done) await sleep(20);
  return done;
}

/**
 * Run `fn(assertHeld)` holding `.zipp-wasm.lock` in the vendor folder. Hooks
 * started together (npm test and npm run build in two terminals) would
 * otherwise swap folders under each other; the one that waits finds the
 * install done. A lock whose process is gone, or that is older than any
 * install, is taken over. `assertHeld()` refuses once the lock is not this
 * install's, so nothing is written under another install's lock.
 */
export async function withInstallLock(vendorDir, fn, { warn = console.warn } = {}) {
  const lock = path.join(vendorDir, LOCK_FILE);
  try {
    fs.mkdirSync(vendorDir, { recursive: true });
  } catch (error) {
    refuse(`cannot create ${rel(vendorDir)} for a ZIPP install (${error.code ?? error.message})`);
  }
  const token = lockToken();
  const started = Date.now();
  let told = false;
  let busy = 0;
  let stuck = 0;
  for (;;) {
    try {
      fs.writeFileSync(lock, token, { flag: 'wx' });
      break;
    } catch (error) {
      if (error.code !== 'EEXIST') {
        if (lockBusy(error) && ++busy < 200) {
          await sleep(20);
          continue;
        }
        refuse(`cannot lock ${rel(vendorDir)} for an install (${rel(lock)}: ${error.code ?? error.message})`);
      }
    }
    let holder;
    try {
      holder = readLock(lock);
    } catch (error) {
      if (lockBusy(error) && ++busy < 200) {
        await sleep(20);
        continue;
      }
      refuse(`cannot read the install lock ${rel(lock)} (${error.code ?? error.message}); if no install is running, delete it`);
    }
    if (!holder) continue; // released while it was read
    busy = 0;
    if (!lockLive(holder)) {
      // A dead lock that can never be removed (read-only, say) is refused in the end, not retried forever.
      if (!(await removeStaleLock(lock, holder)) && ++stuck >= 100) refuse(`cannot remove the install lock ${rel(lock)}, which process ${holder.pid || '(unknown)'} left; delete it`);
      continue;
    }
    stuck = 0;
    if (Date.now() - started > LOCK_STALE_MS) refuse(`${rel(lock)} has been held by process ${holder.pid || '(unknown)'} for ${Math.round(holder.age / 1000)} s; if no install is running, delete it`);
    if (!told) warn(`Waiting for the ZIPP install another process (${holder.pid || 'starting'}) is making into ${rel(vendorDir)} ...`);
    told = true;
    await sleep(250);
  }
  heldLocks.add(token);
  const assertHeld = () => {
    let now = null;
    try {
      now = readLock(lock);
    } catch {}
    if (now?.text !== token) refuse(`the install lock ${rel(lock)} is no longer this install's (${now ? `process ${now.pid || '(unknown)'} holds it` : 'it is gone'}); nothing was written, so run it again`);
  };
  try {
    return await fn(assertHeld);
  } finally {
    heldLocks.delete(token);
    releaseLock(lock, token);
  }
}

/**
 * Write every install to a sibling stage and swap them in, so a stale file
 * from an earlier install cannot survive, a failed write leaves the old
 * installs, and a failed swap puts back the ones already swapped: the folders
 * hold one release, never a mix. Callers hold the install lock.
 */
export function writeInstalls(vendorDir, installs) {
  const suffix = `${process.pid}-${randomBytes(4).toString('hex')}`;
  const staged = [];
  try {
    fs.mkdirSync(vendorDir, { recursive: true });
    // Leftovers of an install that died; a live process's stage is its own.
    for (const leftover of fs.readdirSync(vendorDir)) {
      const owner = /^\.zipp-wasm(?:-python)?\.(?:stage|previous)-(\d+)-/.exec(leftover);
      if (owner && (Number(owner[1]) === process.pid || !pidAlive(Number(owner[1])))) fs.rmSync(path.join(vendorDir, leftover), { recursive: true, force: true });
    }
    for (const { variant, files } of installs) {
      const entry = {
        dir: path.join(vendorDir, variant.folder),
        stage: path.join(vendorDir, `.${variant.folder}.stage-${suffix}`),
        previous: path.join(vendorDir, `.${variant.folder}.previous-${suffix}`),
      };
      staged.push(entry);
      fs.mkdirSync(entry.stage);
      for (const [name, bytes] of files) fs.writeFileSync(path.join(entry.stage, name), bytes);
    }
    const swapped = [];
    try {
      for (const entry of staged) {
        entry.had = fs.existsSync(entry.dir);
        if (entry.had) renameWithRetry(entry.dir, entry.previous);
        try {
          renameWithRetry(entry.stage, entry.dir);
        } catch (error) {
          if (entry.had) renameWithRetry(entry.previous, entry.dir);
          throw error;
        }
        swapped.push(entry);
      }
    } catch (error) {
      for (const entry of swapped.reverse()) {
        fs.rmSync(entry.dir, { recursive: true, force: true });
        if (entry.had) renameWithRetry(entry.previous, entry.dir);
      }
      throw error;
    }
    for (const entry of staged) if (entry.had) fs.rmSync(entry.previous, { recursive: true, force: true });
  } catch (error) {
    if (error instanceof ZippReleaseError) throw error;
    refuse(`could not write the installs into ${rel(vendorDir)}: ${error.message}`);
  } finally {
    for (const entry of staged) {
      try {
        fs.rmSync(entry.stage, { recursive: true, force: true });
      } catch {}
    }
  }
}

async function installReleaseLocked({ vendorDir, tag, latest, releaseDir, cacheDir, expectSumsSha256, fetch: fetchImpl = globalThis.fetch, curatedNotices = CURATED_NOTICES, log = console.log, warn = console.warn, assertHeld = () => {} }) {
  const { release, sums } = await resolveRelease({ tag, latest, releaseDir, cacheDir, fetch: fetchImpl });
  if (expectSumsSha256 && sha256(sums) !== expectSumsSha256.toLowerCase()) refuse(`the ${release} SHA256SUMS has sha256 ${sha256(sums)}; ZIPP_SUMS_SHA256 says ${expectSumsSha256}`);
  log(`Taking the ${VARIANTS.map((v) => v.module).join(' and ')} modules and the embed kit from ZIPP ${release}${releaseDir ? ` in ${releaseDir}` : ''} ...`);
  const version = release.slice(1);
  const load = (name) => loadReleaseFile({ release, sums, name, releaseDir, cacheDir, fetch: fetchImpl });
  const modules = {};
  for (const variant of VARIANTS) modules[variant.suffix] = await load(moduleName(version, variant));
  const kit = await load(kitName(version));
  const installs = await verifyRelease({ release, sums, modules, kit, expectSumsSha256, curatedNotices });
  // Nothing redistributes the Python engine before O1, so drift is a prompt, not a refusal.
  // An older release can need more than the notices name, too: releases before v0.0.21 compile in RustPython.
  const curatedFrom = installs.find((i) => i.curatedFrom)?.curatedFrom;
  const drift = curatedFrom ? compareReleases(release, curatedFrom) : 0;
  if (drift) {
    const note = drift > 0
      ? `ZIPP ${release} is newer than ${curatedFrom}, the release ${rel(curatedNotices)} was generated from; if it compiles in more third-party code, regenerate the notices: node scripts/regen-zipp-notices.mjs <zipp.org checkout> ${release}`
      : `ZIPP ${release} is older than ${curatedFrom}, the release ${rel(curatedNotices)} was generated from; if it compiles in third-party code that release no longer does, regenerate the notices: node scripts/regen-zipp-notices.mjs <zipp.org checkout> ${release}`;
    if (process.env.GITHUB_ACTIONS === 'true') log(`::warning::${note}`);
    else warn(note);
  }
  assertHeld();
  writeInstalls(vendorDir, installs);
  // What landed on disk, not what was meant to.
  const installed = checkInstalls(vendorDir, { curatedNotices });
  const curated = installs.some((i) => i.source.notices?.source === CURATED);
  log(`Installed ZIPP ${release} (${installed.revision.slice(0, 8)}) into ${installs.map((i) => `${rel(path.join(vendorDir, i.variant.folder))} (${i.source.sha256.slice(0, 12)})`).join(' and ')}${curated ? `; Python notices: the curated copy (${release} ships none)` : ''}`);
  return installed;
}

/** Resolve, fetch (or read), verify and install a release into both folders. Returns what --check reports. */
export async function installRelease({ vendorDir = VENDOR_DIR, warn, ...options } = {}) {
  return withInstallLock(vendorDir, (assertHeld) => installReleaseLocked({ vendorDir, warn, assertHeld, ...options }), { warn });
}

// ---------------------------------------------------------------------------
// Check an install
// ---------------------------------------------------------------------------

/** The files an install takes from the kit, as [installed name, the kit's name]: its notices too when they are the release's. */
const kitTaken = (notices) => [...Object.entries(KIT_FILES), ...(notices?.source === 'zipp-release' ? [[NOTICES, NOTICES]] : [])];

/**
 * Check one install offline and return its SOURCE.json; nothing is compiled.
 * The module must be the file RELEASE-SHA256SUMS lists, byte for byte, its
 * record (read again from the installed bytes) and stack those SOURCE.json
 * records; zipp_wasm.js must be that module's own glue; the kit's files the
 * digests recorded, the kit itself listed in RELEASE-SHA256SUMS;
 * RELEASE-SHA256SUMS the digest recorded; and PROFILE.json the profile the
 * engine reported at install.
 */
export function checkInstall(dir, variant, { curatedNotices = CURATED_NOTICES } = {}) {
  const where = rel(dir);
  // Read without the install lock: a folder another install swaps mid-read is a refusal, which --ensure repairs.
  let sourceText;
  try {
    sourceText = fs.readFileSync(path.join(dir, 'SOURCE.json'), 'utf8');
  } catch (error) {
    if (['ENOENT', 'ENOTDIR'].includes(error.code)) refuse(`${where} holds no ZIPP install (no SOURCE.json); run node scripts/fetch-zipp-release.mjs`);
    refuse(`${where}/SOURCE.json cannot be read (${error.code ?? error.message}); run node scripts/fetch-zipp-release.mjs`);
  }
  let source;
  try {
    source = JSON.parse(sourceText);
  } catch (error) {
    refuse(`${where}/SOURCE.json is not JSON: ${error.message}`);
  }

  const problems = [];
  const notices = source.notices ?? null;
  const expected = installedFiles(notices);
  const present = new Map();
  const unreadable = new Set();
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (error) {
    refuse(`${where} cannot be listed (${error.code ?? error.message}); run node scripts/fetch-zipp-release.mjs`);
  }
  for (const entry of entries) {
    if (!entry.isFile()) problems.push(`${entry.name} is not a plain file`);
    else if (!expected.includes(entry.name)) problems.push(`${entry.name} is not part of the install`);
    else {
      try {
        present.set(entry.name, fs.readFileSync(path.join(dir, entry.name)));
      } catch (error) {
        unreadable.add(entry.name);
        problems.push(`${entry.name} cannot be read (${error.code ?? error.message})`);
      }
    }
  }
  for (const name of expected) if (!present.has(name) && !unreadable.has(name)) problems.push(`${name} is missing`);

  const version = source.version;
  if (source.build !== 'release') problems.push(`SOURCE.json build is ${source.build}, not a verified release install`);
  if (source.repository !== REPOSITORY) problems.push(`SOURCE.json repository is ${source.repository}, not ${REPOSITORY}`);
  if (!isReleaseTag(source.release) || source.release !== `v${version}`) problems.push(`SOURCE.json release ${source.release} and version ${version} do not agree`);
  if (!COMMIT.test(source.revision ?? '')) problems.push('SOURCE.json revision is not a full commit');
  if (source.bundle !== moduleName(version, variant)) problems.push(`SOURCE.json bundle ${source.bundle} is not the ${variant.module} module (${moduleName(version, variant)})`);
  for (const field of ['bundleSha256', 'sumsSha256', 'sha256', 'glueSha256', 'profileSha256']) if (!HEX64.test(source[field] ?? '')) problems.push(`SOURCE.json ${field} is not a sha256`);
  if (source.bundleSha256 !== source.sha256) problems.push(`SOURCE.json sha256 ${source.sha256} is not the bundleSha256 ${source.bundleSha256}; the module is installed as published`);
  if (source.variant !== variant.variant || !isDeepStrictEqual(source.languages, variant.languages) || source.stackBytes !== variant.stackBytes) {
    problems.push(`SOURCE.json records ${source.variant} ${JSON.stringify(source.languages)} with a ${source.stackBytes}-byte stack, not the ${variant.variant} engine`);
  }
  if (source.artifact !== 'zipp_wasm_bg.wasm') problems.push(`SOURCE.json artifact is ${source.artifact}`);
  const kit = source.kit ?? {};
  if (kit.file !== kitName(version) || !HEX64.test(kit.sha256 ?? '')) problems.push(`SOURCE.json kit ${JSON.stringify(kit.file)} is not the release's embed kit (${kitName(version)}) with its sha256`);
  for (const [name, from] of kitTaken(notices)) {
    const file = kit.files?.[name];
    if (file?.from !== from || !HEX64.test(file?.sha256 ?? '')) problems.push(`SOURCE.json kit.files records ${name} as ${JSON.stringify(file)}, not the kit's ${from} with its sha256`);
    else if (present.has(name) && sha256(present.get(name)) !== file.sha256) problems.push(`${name} is not the kit's ${from} recorded (${file.sha256})`);
  }
  const allowedNotices = variant.needsNotices ? ['zipp-release', CURATED] : ['zipp-release'];
  if (notices === null ? variant.needsNotices : notices.file !== NOTICES || !allowedNotices.includes(notices.source) || !HEX64.test(notices.sha256 ?? '')) {
    problems.push(`SOURCE.json notices ${JSON.stringify(source.notices)} are not recorded for the ${variant.variant} engine`);
  }

  if (present.has('RELEASE-SHA256SUMS')) {
    const sums = present.get('RELEASE-SHA256SUMS');
    const listed = parseSums(sums);
    if (sha256(sums) !== source.sumsSha256) problems.push(`RELEASE-SHA256SUMS has sha256 ${sha256(sums)}, not the recorded ${source.sumsSha256}`);
    if (listed.get(source.bundle) !== source.bundleSha256) problems.push(`RELEASE-SHA256SUMS does not list ${source.bundle} with the recorded ${source.bundleSha256}`);
    if (listed.get(kit.file) !== kit.sha256) problems.push(`RELEASE-SHA256SUMS does not list ${kit.file} with the recorded ${kit.sha256}`);
  }
  if (present.has('zipp_wasm_bg.wasm')) {
    const wasm = present.get('zipp_wasm_bg.wasm');
    if (sha256(wasm) !== source.sha256) problems.push(`zipp_wasm_bg.wasm is not the recorded ${source.sha256}`);
    let meta;
    let glue;
    try {
      meta = moduleMeta(wasm, 'zipp_wasm_bg.wasm');
      glue = moduleGlue(wasm, 'zipp_wasm_bg.wasm');
    } catch (error) {
      if (!(error instanceof ZippReleaseError)) throw error;
      problems.push(error.message);
    }
    if (glue) {
      problems.push(...metaProblems(meta, glue, variant, { version, revision: source.revision, glueSha256: source.glueSha256 }).map((p) => `zipp_wasm_bg.wasm: ${p}`));
      if (present.has('zipp_wasm.js') && !glue.equals(present.get('zipp_wasm.js'))) problems.push(`zipp_wasm.js is not the glue zipp_wasm_bg.wasm carries (its ${GLUE_SECTION} section)`);
    }
    problems.push(...stackProblems(wasm, variant).map((p) => `zipp_wasm_bg.wasm: ${p}`));
  }
  if (present.has('zipp_wasm.js') && sha256(present.get('zipp_wasm.js')) !== source.glueSha256) problems.push(`zipp_wasm.js is not the recorded ${source.glueSha256}`);
  if (present.has('PROFILE.json')) {
    const profile = present.get('PROFILE.json');
    if (sha256(profile) !== source.profileSha256) problems.push(`PROFILE.json is not the recorded ${source.profileSha256}, the profile the engine reported at install`);
    let profileVersion;
    try {
      profileVersion = JSON.parse(profile.toString('utf8')).version;
    } catch {}
    if (profileVersion !== version) problems.push(`PROFILE.json says version ${profileVersion}, not ${version}`);
  }
  if (notices && present.has(NOTICES)) {
    if (sha256(present.get(NOTICES)) !== notices.sha256) problems.push(`${NOTICES} is not the recorded ${notices.sha256}`);
    else if (notices.source === CURATED && !(fs.existsSync(curatedNotices) && fs.readFileSync(curatedNotices).equals(present.get(NOTICES)))) problems.push(`${NOTICES} is not the curated copy in ${rel(curatedNotices)}`);
  }
  fail(`the ZIPP install in ${where} does not check (run node scripts/fetch-zipp-release.mjs)`, problems);
  return source;
}

/**
 * Check both installs offline, and that they are one release: the same tag,
 * the same SHA256SUMS and the same commit. Returns {release, revision,
 * sumsSha256, sources} with SOURCE.json by variant suffix.
 */
export function checkInstalls(vendorDir = VENDOR_DIR, { curatedNotices } = {}) {
  const sources = {};
  for (const variant of VARIANTS) sources[variant.suffix] = checkInstall(path.join(vendorDir, variant.folder), variant, { curatedNotices });
  const all = Object.values(sources);
  const differ = ['release', 'sumsSha256', 'revision'].filter((field) => new Set(all.map((s) => s[field])).size !== 1);
  fail(`the ZIPP installs in ${rel(vendorDir)} are not one release (run node scripts/fetch-zipp-release.mjs)`, differ.map((field) => `${field} differs: ${VARIANTS.map((v) => `${v.folder} ${sources[v.suffix][field]}`).join(', ')}`));
  const [{ release, revision, sumsSha256 }] = all;
  return { release, revision, sumsSha256, sources };
}

/**
 * --check, then both installs against the release as published now: ZIPP's
 * SHA256SUMS, each module, and the kit (downloaded once) and every file taken
 * from it.
 */
export async function checkInstallsOnline(vendorDir = VENDOR_DIR, { fetch: fetchImpl = globalThis.fetch, curatedNotices } = {}) {
  const installed = checkInstalls(vendorDir, { curatedNotices });
  const { release } = installed;
  const problems = [];
  const read = (dir, name) => {
    try {
      return fs.readFileSync(path.join(dir, name));
    } catch {
      return Buffer.alloc(0);
    }
  };
  const sums = await download(releaseUrl(release, 'SHA256SUMS'), fetchImpl);
  for (const variant of VARIANTS) {
    const dir = path.join(vendorDir, variant.folder);
    const source = installed.sources[variant.suffix];
    if (!sums.equals(read(dir, 'RELEASE-SHA256SUMS'))) problems.push(`${variant.folder}/RELEASE-SHA256SUMS is not the SHA256SUMS published under ${release}`);
    const module = await download(releaseUrl(release, source.bundle), fetchImpl);
    if (!module.equals(read(dir, 'zipp_wasm_bg.wasm'))) problems.push(`${variant.folder}/zipp_wasm_bg.wasm is not the published ${source.bundle}`);
  }
  // Both installs name the one kit (--check holds each to the release's kit name and its line in one SHA256SUMS).
  const { file: kitFile } = installed.sources[VARIANTS[0].suffix].kit;
  const kit = await download(releaseUrl(release, kitFile), fetchImpl);
  const recorded = [...new Set(VARIANTS.map((v) => installed.sources[v.suffix].kit.sha256))].filter((digest) => digest !== sha256(kit));
  for (const digest of recorded) problems.push(`the published ${kitFile} is not the recorded ${digest}`);
  if (!recorded.length) {
    const entries = kitEntries(kit, release.slice(1));
    for (const variant of VARIANTS) {
      const dir = path.join(vendorDir, variant.folder);
      for (const [name, { from }] of Object.entries(installed.sources[variant.suffix].kit.files)) {
        if (!entries.get(from)?.equals(read(dir, name))) problems.push(`${variant.folder}/${name} is not ${from} in the published ${kitFile}`);
      }
    }
  }
  fail(`the ZIPP installs in ${rel(vendorDir)} are not what ${release} publishes`, problems);
  return installed;
}

/**
 * Install unless both installs check and are the release (ZIPP_RELEASE) and
 * SHA256SUMS (ZIPP_SUMS_SHA256) asked for, when either is. Asked for neither,
 * an intact install is kept without a request; a missing or broken one is
 * replaced by the release in ZIPP_RELEASE_DIR, or else the latest.
 */
export async function ensureInstalls({ vendorDir = VENDOR_DIR, tag, expectSumsSha256, log = console.log, warn = console.warn, ...install } = {}) {
  // The lock covers the check as well: whoever waited finds the install done.
  return withInstallLock(vendorDir, async (assertHeld) => {
    let reason;
    try {
      const installed = checkInstalls(vendorDir, { curatedNotices: install.curatedNotices });
      if (tag && installed.release !== tag) reason = `${rel(vendorDir)} holds ZIPP ${installed.release}, not ${tag}`;
      else if (expectSumsSha256 && installed.sumsSha256 !== expectSumsSha256.toLowerCase()) reason = `${rel(vendorDir)} was installed from a SHA256SUMS other than ZIPP_SUMS_SHA256 ${expectSumsSha256.slice(0, 12)}`;
      else return { action: 'none', ...installed };
    } catch (error) {
      if (!(error instanceof ZippReleaseError)) throw error;
      reason = error.message;
    }
    log(`Installing ZIPP ${tag ?? (install.releaseDir ? `from ${install.releaseDir}` : 'latest')}: ${reason}`);
    return { action: 'installed', ...(await installReleaseLocked({ vendorDir, tag, expectSumsSha256, log, warn, assertHeld, ...install })) };
  }, { warn });
}

/** What the environment asks of an install (ZIPP_RELEASE, ZIPP_RELEASE_DIR, ZIPP_SUMS_SHA256). */
export function releaseOptions(env = process.env) {
  const expectSumsSha256 = env.ZIPP_SUMS_SHA256 || undefined;
  if (expectSumsSha256 && !HEX64.test(expectSumsSha256.toLowerCase())) refuse('ZIPP_SUMS_SHA256 must be a sha256 (64 hex)');
  return { tag: env.ZIPP_RELEASE || undefined, releaseDir: env.ZIPP_RELEASE_DIR ? path.resolve(env.ZIPP_RELEASE_DIR) : undefined, expectSumsSha256 };
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
const USAGE = 'Usage: node scripts/fetch-zipp-release.mjs [vX.Y.Z | --latest] | --check [--online] | --resolve-only [vX.Y.Z | --latest] | --ensure';

async function main(args) {
  const flags = new Set(['--latest', '--check', '--online', '--resolve-only', '--ensure']);
  const positional = args.filter((a) => !a.startsWith('--'));
  const unknown = args.filter((a) => a.startsWith('--') && !flags.has(a));
  const has = (flag) => args.includes(flag);
  const modes = ['--check', '--resolve-only', '--ensure'].filter(has);
  if (unknown.length || positional.length > 1 || modes.length > 1 || (has('--online') && !has('--check')) || (has('--latest') && (has('--check') || has('--ensure'))) || (has('--check') && positional.length)) refuse(USAGE);

  const env = process.env;
  const vendorDir = env.ZIPP_OUT ? path.resolve(env.ZIPP_OUT) : VENDOR_DIR;
  const { releaseDir, expectSumsSha256, tag: envTag } = releaseOptions(env);
  const latest = has('--latest');
  const tag = positional[0] ?? envTag;
  if (latest && tag) refuse('name a release or --latest, not both');

  if (has('--check')) {
    const installed = has('--online') ? await checkInstallsOnline(vendorDir) : checkInstalls(vendorDir);
    console.log(`ZIPP ${installed.release} (${installed.revision.slice(0, 8)}) in ${rel(vendorDir)} checks${has('--online') ? ' against the published release' : ''}.`);
  } else if (has('--resolve-only')) {
    const { release, sums } = await resolveRelease({ tag, latest, releaseDir });
    console.log(JSON.stringify({ release, sumsSha256: sha256(sums) }));
  } else if (has('--ensure')) {
    await ensureInstalls({ vendorDir, tag, expectSumsSha256, releaseDir });
  } else {
    await installRelease({ vendorDir, tag, latest, releaseDir, expectSumsSha256 });
  }
}

/**
 * Whether this file is the program `node` was started on. Node names a module by where its file really is, so the
 * path it was started with is followed through its links too: a Mac's temporary folder is one (/var to /private/var),
 * and a script started from a copy there would otherwise say and do nothing.
 */
function startedAsTheProgram() {
  if (!process.argv[1]) return false;
  const self = fileURLToPath(import.meta.url);
  if (path.resolve(process.argv[1]) === self) return true;
  try {
    return fs.realpathSync(process.argv[1]) === self;
  } catch {
    return false;
  }
}

if (startedAsTheProgram()) {
  main(process.argv.slice(2)).catch((error) => {
    if (!(error instanceof ZippReleaseError)) throw error;
    console.error(`fetch-zipp-release: ${error.message}`);
    process.exitCode = 1;
  });
}
