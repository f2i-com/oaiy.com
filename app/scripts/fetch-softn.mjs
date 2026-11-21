#!/usr/bin/env node
/**
 * Install SoftN's hosted app runtime (the sandboxed shell that renders a SoftN
 * app in an opaque-origin iframe) from a published softn.com release into
 * public/softn/. Optional: without it, OAIY still writes, checks and
 * exports SoftN apps, and only the live preview is unavailable.
 *
 *   node scripts/fetch-softn.mjs            install the pinned release
 *   node scripts/fetch-softn.mjs --ensure   install unless it is already there
 *
 * The release zip must match its published SHA-256, pinned here. Its
 * hosted-runtime/ folder is installed whole: from v0.0.25 it carries no ONNX
 * or speech payloads (a SoftN app's local AI runs GGUF models on ZIPP).
 * SOFTN_RELEASE_DIR=<dir> reads the zip from a folder instead of GitHub. From
 * GitHub, the zip is asked of its API's asset address first and the plain
 * download address second (release-download.mjs); a GITHUB_TOKEN or GH_TOKEN
 * goes to the API alone.
 */
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { bridgeCurrent, installBridge } from './softn-bridge/install.mjs';
import { downloadReleaseFile } from './release-download.mjs';
import { unzipSync } from 'fflate';

const RELEASE = 'v0.0.50';
const ZIP = `softn-formlogic-runtime-${RELEASE}.zip`;
const ZIP_SHA256 = process.env.SOFTN_ZIP_SHA256 || 'a2f77dcf3f1cbf7f4c1d0e8cf8ad40f86485fe622d35d81e37b8bb526c73944d';
const PREFIX = 'hosted-runtime/';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'public', 'softn');
const stamp = join(out, 'SOURCE.json');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function installed() {
  try {
    const source = JSON.parse(readFileSync(stamp, 'utf8'));
    return source.release === RELEASE && source.zip_sha256 === ZIP_SHA256 && existsSync(join(out, 'index.html'));
  } catch {
    return false;
  }
}

function extractRuntime(zip, stage) {
  const entries = unzipSync(zip, { filter: ({ name }) => name.replaceAll('\\', '/').startsWith(PREFIX) });
  const digests = new Map();
  let bytes = 0;
  for (const [name, data] of Object.entries(entries)) {
    const rel = name.replaceAll('\\', '/').slice(PREFIX.length);
    if (!rel || rel.endsWith('/')) continue;
    if (rel.split('/').some(part => !part || /[. ]$|[<>:"|?*\x00-\x1f]/.test(part) || /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part))) {
      throw new Error(`invalid hosted runtime path ${name}`);
    }
    const target = join(stage, rel);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, data, { flag: 'wx' });
    digests.set(rel, sha256(data));
    bytes += data.byteLength;
  }
  const manifest = JSON.parse(readFileSync(join(stage, 'runtime-manifest.json'), 'utf8'));
  if (!manifest.files?.['index.html']) throw new Error('the hosted runtime manifest names no index.html');
  for (const [name, digest] of Object.entries(manifest.files)) {
    if (digests.get(name) !== digest) throw new Error(`the hosted runtime is missing or does not match its manifest: ${name}`);
  }
  return { files: digests.size, bytes };
}

// Keep the last working preview until all files, the bridge and the stamp are ready.
function replaceRuntime(prepare) {
  const stage = mkdtempSync(join(root, '.softn-install-'));
  const previous = `${stage}.previous`;
  try {
    const result = prepare(stage);
    if (!installBridge(stage, RELEASE) || !bridgeCurrent(stage)) throw new Error('the hosted runtime index could not load the preview bridge');
    mkdirSync(dirname(out), { recursive: true });
    const hadPrevious = existsSync(out);
    if (hadPrevious) renameSync(out, previous);
    try {
      renameSync(stage, out);
    } catch (error) {
      if (hadPrevious) renameSync(previous, out);
      throw error;
    }
    if (hadPrevious) rmSync(previous, { recursive: true, force: true });
    return result;
  } finally {
    rmSync(stage, { recursive: true, force: true });
  }
}

async function main() {
  if (process.argv.includes('--ensure') && installed()) {
    if (!bridgeCurrent(out)) replaceRuntime(stage => cpSync(out, stage, { recursive: true }));
    console.log(`SoftN runtime ${RELEASE} is installed in public/softn/`);
    return;
  }
  let zip;
  if (process.env.SOFTN_RELEASE_DIR) {
    zip = readFileSync(join(process.env.SOFTN_RELEASE_DIR, ZIP));
  } else {
    console.log(`fetching ${ZIP} from the softn.com ${RELEASE} release ...`);
    // GitHub's API asset address first, the plain download address second; the digest below holds either way.
    zip = await downloadReleaseFile({ repository: 'f2i-com/softn.com', tag: RELEASE, name: ZIP });
  }
  if (sha256(zip) !== ZIP_SHA256) throw new Error(`${ZIP} does not match its pinned SHA-256`);
  const inventory = replaceRuntime(stage => {
    const inventory = extractRuntime(zip, stage);
    writeFileSync(join(stage, 'SOURCE.json'), JSON.stringify({ release: RELEASE, zip: ZIP, zip_sha256: ZIP_SHA256, ...inventory }, null, 2));
    return inventory;
  });
  console.log(`installed the SoftN runtime ${RELEASE} into public/softn/ (${inventory.files} files, ${(inventory.bytes / 1e6).toFixed(1)} MB)`);
}

main().catch((error) => {
  console.error(`fetch-softn: ${error.message}`);
  process.exit(1);
});
