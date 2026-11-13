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
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { installBridge } from './softn-bridge/install.mjs';
import { downloadReleaseFile } from './release-download.mjs';
import { unzipSync } from 'fflate';

const RELEASE = 'v0.0.40';
const ZIP = `softn-formlogic-runtime-${RELEASE}.zip`;
const ZIP_SHA256 = process.env.SOFTN_ZIP_SHA256 || '1555b931b4dc0b2a23338e6372b0c5b135b0b7bb42831baaf5692f8c6d6ddca0';
const PREFIX = 'hosted-runtime/';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'public', 'softn');
const stamp = join(out, 'SOURCE.json');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function installed() {
  try {
    const source = JSON.parse(readFileSync(stamp, 'utf8'));
    return source.release === RELEASE && existsSync(join(out, 'index.html'));
  } catch {
    return false;
  }
}

async function main() {
  if (process.argv.includes('--ensure') && installed()) {
    installBridge(out);
    console.log(`SoftN runtime ${RELEASE} is installed in public/softn/`);
    return;
  }
  let zip;
  if (process.env.SOFTN_RELEASE_DIR) {
    zip = readFileSync(join(process.env.SOFTN_RELEASE_DIR, ZIP));
  } else {
    console.log(`fetching ${ZIP} from the softn.com ${RELEASE} release (about 32 MB) ...`);
    // GitHub's API asset address first, the plain download address second; the digest below holds either way.
    zip = await downloadReleaseFile({ repository: 'f2i-com/softn.com', tag: RELEASE, name: ZIP });
  }
  if (sha256(zip) !== ZIP_SHA256) throw new Error(`${ZIP} does not match its pinned SHA-256`);
  const entries = unzipSync(zip, { filter: (f) => f.name.startsWith(PREFIX) });
  rmSync(out, { recursive: true, force: true });
  let count = 0;
  let bytes = 0;
  for (const [name, data] of Object.entries(entries)) {
    if (name.endsWith('/')) continue;
    const rel = name.slice(PREFIX.length);
    if (!rel || rel.split('/').some((seg) => seg === '..')) continue;
    const target = join(out, rel);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, data);
    count++;
    bytes += data.byteLength;
  }
  installBridge(out);
  writeFileSync(stamp, JSON.stringify({ release: RELEASE, zip: ZIP, zip_sha256: ZIP_SHA256, files: count, bytes }, null, 2));
  console.log(`installed the SoftN runtime ${RELEASE} into public/softn/ (${count} files, ${(bytes / 1e6).toFixed(1)} MB)`);
}

main().catch((error) => {
  console.error(`fetch-softn: ${error.message}`);
  process.exit(1);
});
