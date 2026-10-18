#!/usr/bin/env node
/**
 * Install the Zipp engine (JavaScript + Python, no torch) from a published
 * zipp.org release into public/zipp/. The engine is generated, not committed:
 * the release's SHA256SUMS digest is pinned here, and everything taken from
 * the release must match it.
 *
 * From ZIPP v0.0.26 a release publishes each WebAssembly module as one .wasm
 * file that carries its own wasm-bindgen bindings in a `zipp.glue` custom
 * section, beside a `zipp.meta` record of its format, variant, version, commit,
 * languages and the bindings' sha256; the type declarations and the licence
 * are in an embed kit. The install keeps the names the Agent loads:
 *
 *   zipp_wasm_bg.wasm   the javascript-python-base module, byte for byte
 *   zipp_wasm.js        its zipp.glue section, byte for byte
 *   zipp_wasm.d.ts      the kit's zipp-engine.d.ts
 *   LICENSE-APACHE      the kit's
 *
 *   node scripts/fetch-zipp.mjs            install the pinned release
 *   node scripts/fetch-zipp.mjs --ensure   install unless public/zipp/ already holds it
 *
 * ZIPP_RELEASE_DIR=<dir> reads SHA256SUMS, the module and the kit from a
 * folder instead of GitHub (offline builds).
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { unzipSync } from 'fflate';

const RELEASE = 'v0.0.26';
const VERSION = RELEASE.slice(1);
const VARIANT = 'javascript-python-base';
const LANGUAGES = ['javascript', 'python'];
const MODULE = `zipp-${VERSION}-${VARIANT}.wasm`;
const KIT = `zipp-${VERSION}-wasm-embed-kit`;
// The digest of the release's top-level SHA256SUMS, so a changed release file
// cannot slip in under the same tag.
const SUMS_SHA256 = process.env.ZIPP_SUMS_SHA256 || 'dfc1d9067175f35cfc662334220e480fae32f2f2da735b4fdec5c9a52c9f385c';
/** The installed files. */
const FILES = ['zipp_wasm.js', 'zipp_wasm_bg.wasm', 'zipp_wasm.d.ts', 'LICENSE-APACHE'];
/** The kit's files the install takes, by the name they are installed under. */
const FROM_KIT = { 'zipp_wasm.d.ts': 'zipp-engine.d.ts', 'LICENSE-APACHE': 'LICENSE-APACHE' };

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'public', 'zipp');
const stamp = join(out, 'SOURCE.json');

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const text = (bytes) => new TextDecoder().decode(bytes);

function parseSums(source) {
  const sums = new Map();
  for (const line of source.split(/\r?\n/)) {
    const m = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line.trim());
    if (m) sums.set(m[2].trim(), m[1]);
  }
  return sums;
}

/** BUILD-INFO.txt lines are `key=value`. */
function parseBuildInfo(source) {
  const info = {};
  for (const line of source.split(/\r?\n/)) {
    const at = line.indexOf('=');
    if (at > 0) info[line.slice(0, at).trim()] = line.slice(at + 1).trim();
  }
  return info;
}

/** The payloads of every custom section called `name` in a WebAssembly binary, in order; null for anything else. */
function customSections(bytes, name) {
  if (bytes.length < 8 || bytes[0] !== 0x00 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d) return null;
  let at = 8;
  const leb = () => {
    let value = 0;
    for (let shift = 0; ; shift += 7) {
      if (at >= bytes.length || shift > 28) throw new RangeError('malformed LEB128');
      const byte = bytes[at++];
      value += (byte & 0x7f) * 2 ** shift;
      if (!(byte & 0x80)) return value;
    }
  };
  const found = [];
  try {
    while (at < bytes.length) {
      const id = bytes[at++];
      const end = leb() + at;
      if (end > bytes.length) return null;
      if (id === 0) {
        const length = leb();
        if (at + length > end) return null;
        if (text(bytes.subarray(at, at + length)) === name) found.push(bytes.subarray(at + length, end));
      }
      at = end;
    }
    return found;
  } catch {
    return null;
  }
}

async function fetchBytes(name) {
  if (process.env.ZIPP_RELEASE_DIR) return new Uint8Array(readFileSync(join(process.env.ZIPP_RELEASE_DIR, name)));
  const url = `https://github.com/f2i-com/zipp.org/releases/download/${RELEASE}/${name}`;
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`GET ${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

function installed() {
  if (!existsSync(stamp)) return false;
  try {
    const source = JSON.parse(readFileSync(stamp, 'utf8'));
    if (source.release !== RELEASE || source.sums_sha256 !== SUMS_SHA256) return false;
    return FILES.every((f) => existsSync(join(out, f)) && sha256(readFileSync(join(out, f))) === source.files[f]);
  } catch {
    return false;
  }
}

/** The module, checked: its digest is its line in SHA256SUMS, its record names this release's engine, its glue is the one the record names. */
function checkModule(wasm, sums) {
  if (sums.get(MODULE) !== sha256(wasm)) throw new Error(`${MODULE} does not match the release SHA256SUMS`);
  const metas = customSections(wasm, 'zipp.meta');
  const glues = customSections(wasm, 'zipp.glue');
  if (!metas || !glues) throw new Error(`${MODULE} is not a WebAssembly module`);
  if (metas.length !== 1 || glues.length !== 1) throw new Error(`${MODULE} carries ${metas.length} zipp.meta records and ${glues.length} zipp.glue sections, not one of each`);
  let meta;
  try {
    meta = JSON.parse(text(metas[0]));
  } catch {
    throw new Error(`${MODULE}'s zipp.meta record is not JSON`);
  }
  const glue = glues[0];
  const problems = [];
  if (meta.format !== 1) problems.push(`format ${JSON.stringify(meta.format)}, not 1`);
  if (meta.version !== VERSION) problems.push(`version ${meta.version}, not ${VERSION}`);
  if (meta.variant !== VARIANT) problems.push(`variant ${meta.variant}, not ${VARIANT}`);
  if (!isDeepStrictEqual(meta.languages, LANGUAGES)) problems.push(`languages ${JSON.stringify(meta.languages)}, not ${JSON.stringify(LANGUAGES)}`);
  if (!/^[0-9a-f]{40}$/.test(meta.commit ?? '')) problems.push(`commit ${JSON.stringify(meta.commit)} is not a full commit`);
  if (sha256(glue) !== meta.glueSha256) problems.push(`its zipp.glue section is ${sha256(glue)}, the record says ${meta.glueSha256}`);
  if (problems.length) throw new Error(`${MODULE} is not ZIPP ${RELEASE}'s ${VARIANT} engine: ${problems.join('; ')}`);
  return { meta, glue };
}

/** The kit's files the install takes, checked against the release SHA256SUMS, the kit's own and its BUILD-INFO.txt. */
function kitFiles(zip, sums, commit) {
  const name = `${KIT}.zip`;
  if (sums.get(name) !== sha256(zip)) throw new Error(`${name} does not match the release SHA256SUMS`);
  const entries = unzipSync(zip);
  const inKit = (file) => entries[`${KIT}/${file}`];
  const inner = parseSums(text(inKit('SHA256SUMS') ?? new Uint8Array()));
  const info = parseBuildInfo(text(inKit('BUILD-INFO.txt') ?? new Uint8Array()));
  if (info.version !== VERSION || info.commit !== commit) throw new Error(`${name}'s BUILD-INFO.txt names ${info.version} ${info.commit}, not ${VERSION} ${commit}`);
  const files = {};
  for (const [as, file] of Object.entries(FROM_KIT)) {
    const bytes = inKit(file);
    if (!bytes) throw new Error(`${name} has no ${file}`);
    if (inner.get(file) !== sha256(bytes)) throw new Error(`${file} does not match the kit's SHA256SUMS`);
    files[as] = bytes;
  }
  return files;
}

async function main() {
  if (process.argv.includes('--ensure') && installed()) {
    console.log(`zipp ${RELEASE} is installed in public/zipp/`);
    return;
  }
  console.log(`fetching zipp ${RELEASE} (${MODULE}) ...`);
  const sumsBytes = await fetchBytes('SHA256SUMS');
  if (sha256(sumsBytes) !== SUMS_SHA256) throw new Error(`the ${RELEASE} SHA256SUMS does not match the pinned digest (or ZIPP_SUMS_SHA256)`);
  const sums = parseSums(text(sumsBytes));
  const wasm = await fetchBytes(MODULE);
  const { meta, glue } = checkModule(wasm, sums);
  const kit = await fetchBytes(`${KIT}.zip`);
  const files = { 'zipp_wasm.js': glue, 'zipp_wasm_bg.wasm': wasm, ...kitFiles(kit, sums, meta.commit) };
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  const digests = {};
  for (const f of FILES) {
    writeFileSync(join(out, f), files[f]);
    digests[f] = sha256(files[f]);
  }
  const source = {
    release: RELEASE,
    commit: meta.commit,
    variant: VARIANT,
    languages: LANGUAGES,
    module: MODULE,
    module_sha256: sha256(wasm),
    glue_sha256: meta.glueSha256,
    kit: `${KIT}.zip`,
    kit_sha256: sha256(kit),
    sums_sha256: SUMS_SHA256,
    files: digests,
  };
  writeFileSync(stamp, JSON.stringify(source, null, 2));
  console.log(`installed zipp ${RELEASE} (${meta.commit.slice(0, 8)}) into public/zipp/`);
}

main().catch((error) => {
  console.error(`fetch-zipp: ${error.message}`);
  process.exit(1);
});
