#!/usr/bin/env node
/**
 * Install the Zipp engine (zipp-wasm, JavaScript + Python, no torch) from a
 * published zipp.org release into public/zipp/. The engine is generated, not
 * committed: the release's SHA256SUMS digest is pinned here, the bundle must
 * match it, and every file taken from the bundle must match the bundle's own
 * SHA256SUMS.
 *
 *   node scripts/fetch-zipp.mjs            install the pinned release
 *   node scripts/fetch-zipp.mjs --ensure   install unless public/zipp/ already holds it
 *
 * ZIPP_RELEASE_DIR=<dir> reads SHA256SUMS and the bundle from a folder instead
 * of GitHub (offline builds).
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { unzipSync } from 'fflate';

const RELEASE = 'v0.0.21';
const VERSION = RELEASE.slice(1);
const BUNDLE = `zipp-wasm-${VERSION}-web-python-base`;
// The digest of the release's top-level SHA256SUMS, so a changed release file
// cannot slip in under the same tag.
const SUMS_SHA256 = process.env.ZIPP_SUMS_SHA256 || '47b8fc05f2e0750cd98084894937049d7967a298804a4e03baf9a5b556ea0793';
const FILES = ['zipp_wasm.js', 'zipp_wasm_bg.wasm', 'zipp_wasm.d.ts', 'LICENSE-APACHE', 'BUILD-INFO.txt'];

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'public', 'zipp');
const stamp = join(out, 'SOURCE.json');

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function parseSums(text) {
  const sums = new Map();
  for (const line of text.split(/\r?\n/)) {
    const m = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line.trim());
    if (m) sums.set(m[2].trim(), m[1]);
  }
  return sums;
}

async function fetchBytes(name) {
  if (process.env.ZIPP_RELEASE_DIR) return readFileSync(join(process.env.ZIPP_RELEASE_DIR, name));
  const url = `https://github.com/f2i-com/zipp.org/releases/download/${RELEASE}/${name}`;
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`GET ${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

function installed() {
  if (!existsSync(stamp)) return false;
  try {
    const source = JSON.parse(readFileSync(stamp, 'utf8'));
    if (source.release !== RELEASE) return false;
    return FILES.every((f) => existsSync(join(out, f)) && sha256(readFileSync(join(out, f))) === source.files[f]);
  } catch {
    return false;
  }
}

async function main() {
  if (process.argv.includes('--ensure') && installed()) {
    console.log(`zipp ${RELEASE} is installed in public/zipp/`);
    return;
  }
  console.log(`fetching zipp ${RELEASE} (${BUNDLE}) ...`);
  const sumsBytes = await fetchBytes('SHA256SUMS');
  if (SUMS_SHA256 && sha256(sumsBytes) !== SUMS_SHA256) throw new Error('the release SHA256SUMS does not match ZIPP_SUMS_SHA256');
  const sums = parseSums(new TextDecoder().decode(sumsBytes));
  const zipName = `${BUNDLE}.zip`;
  const zip = await fetchBytes(zipName);
  if (sums.get(zipName) !== sha256(zip)) throw new Error(`${zipName} does not match the release SHA256SUMS`);
  const entries = unzipSync(zip);
  const inner = parseSums(new TextDecoder().decode(entries[`${BUNDLE}/SHA256SUMS`] ?? new Uint8Array()));
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  const files = {};
  for (const f of FILES) {
    const bytes = entries[`${BUNDLE}/${f}`];
    if (!bytes) throw new Error(`${zipName} has no ${f}`);
    const expected = inner.get(f);
    if (expected && expected !== sha256(bytes)) throw new Error(`${f} does not match the bundle's SHA256SUMS`);
    writeFileSync(join(out, f), bytes);
    files[f] = sha256(bytes);
  }
  writeFileSync(stamp, JSON.stringify({ release: RELEASE, bundle: BUNDLE, zip_sha256: sha256(zip), files }, null, 2));
  console.log(`installed zipp ${RELEASE} into public/zipp/`);
}

main().catch((error) => {
  console.error(`fetch-zipp: ${error.message}`);
  process.exit(1);
});
