// The release asset `oaiy-cli-<v>.tar.gz` (scripts/pack-cli-asset.mjs), built
// here exactly as release.yml builds it: from a COPY of a staged CLI folder
// (desktop/scripts/sync-cli-lib.mjs's `stageEngine` over this build's dist/ +
// node_modules/undici), with an inner `SHA256SUMS` and an `oaiy-cli.json` that
// IS the copy's `capabilities --json` output.
//
//     node test/cli-asset.mjs [path/to/oaiy.mjs]
//
//   * the tarball extracts with the platform's `tar`; every file's sha256 is
//     the one SHA256SUMS records and SHA256SUMS lists every file but itself
//     (and `sha256sum -c SHA256SUMS` agrees where the tool exists);
//   * `oaiy-cli.json` is byte-for-byte what the EXTRACTED CLI prints for
//     `capabilities --json`, and reports protocols {run, script, profile}, a
//     ready engine and the staged SOURCE.json digest;
//   * both worker shells, the engine folder and undici are in the asset;
//   * a staged CLI whose engine does not check is refused and no asset is
//     written — even when the builder's OWN environment points OAIY_ZIPP_ASSET_DIR
//     at a good engine, because the copy runs with every OAIY_* scrubbed.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { stageEngine, STAGED_FILES } from '../../desktop/scripts/sync-cli-lib.mjs';
import { packCliAsset, parseSums, CAPABILITIES_FILE, SUMS_FILE } from '../../scripts/pack-cli-asset.mjs';

const cli = process.argv[2] ?? fileURLToPath(new URL('../dist/oaiy.mjs', import.meta.url));
const dist = path.dirname(cli);
const cliRoot = fileURLToPath(new URL('../', import.meta.url));
const undici = path.join(cliRoot, 'node_modules', 'undici');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-cli-asset-test-'));

const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^OAIY_/i.test(k)));

try {
  // --- stage, as sync-cli.mjs does, into a folder of our own ------------------
  const staged = path.join(dir, 'resources-cli');
  fs.mkdirSync(path.join(staged, 'node_modules'), { recursive: true });
  stageEngine(dist, staged);
  assert.ok(fs.existsSync(undici), 'cli/node_modules/undici is needed to run the bundle');
  fs.cpSync(undici, path.join(staged, 'node_modules', 'undici'), { recursive: true });
  for (const junk of ['docs', 'test', 'types']) fs.rmSync(path.join(staged, 'node_modules', 'undici', junk), { recursive: true, force: true });
  const stagedSource = JSON.parse(fs.readFileSync(path.join(staged, 'zipp', 'SOURCE.json'), 'utf8'));
  const before = fs.readdirSync(staged).sort();

  // --- build the asset ---------------------------------------------------------
  const out = path.join(dir, 'oaiy-cli-0.0.0-test.tar.gz');
  const summaryFile = path.join(dir, 'summary.json');
  const t0 = Date.now();
  const result = packCliAsset(staged, out, { summary: summaryFile });
  assert.ok(fs.existsSync(out), 'the asset was written');
  assert.equal(result.sha256, sha256(fs.readFileSync(out)), 'the summary records the asset digest');
  assert.deepEqual(fs.readdirSync(staged).sort(), before, 'the staged folder was not touched (a COPY was packed)');
  const summary = JSON.parse(fs.readFileSync(summaryFile, 'utf8'));
  assert.equal(summary.asset, path.basename(out));
  assert.deepEqual(summary.capabilities.protocols, { run: 1, script: 1, profile: 1 });
  console.log(`PASS: pack-cli-asset builds the asset from a copy in ${Date.now() - t0} ms (${result.files} files, ${(result.bytes / 1024 / 1024).toFixed(1)} MB) and leaves the staged folder untouched`);

  // --- extract with the platform's tar and check the inner sums -----------------
  const extracted = path.join(dir, 'extracted');
  fs.mkdirSync(extracted);
  // Relative paths from `dir`: GNU tar reads `C:\…` as a remote host.
  const tar = spawnSync('tar', ['-xzf', path.basename(out), '-C', path.basename(extracted)], { cwd: dir, encoding: 'utf8', windowsHide: true });
  assert.equal(tar.status, 0, `tar could not extract the asset: ${tar.stderr || tar.error?.message}`);
  const walk = (d, rel = '') => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(d, e.name), `${rel}${e.name}/`) : [`${rel}${e.name}`]);
  const files = walk(extracted).sort();
  const sums = parseSums(fs.readFileSync(path.join(extracted, SUMS_FILE), 'utf8'));
  for (const [file, digest] of sums) {
    assert.equal(sha256(fs.readFileSync(path.join(extracted, file))), digest, `${file} differs from its SHA256SUMS entry`);
  }
  assert.deepEqual([...sums.keys()].sort(), files.filter((f) => f !== SUMS_FILE), 'SHA256SUMS lists every file but itself');
  assert.ok(sums.has(CAPABILITIES_FILE), 'SHA256SUMS covers oaiy-cli.json');
  for (const rel of [...STAGED_FILES.map((f) => f.replace(/\\/g, '/')), 'node_modules/undici/package.json']) {
    assert.ok(files.includes(rel), `the asset is missing ${rel}`);
  }
  const sumsText = fs.readFileSync(path.join(extracted, SUMS_FILE), 'utf8');
  assert.match(sumsText, /^[0-9a-f]{64}  \S/m, 'the two-space sha256sum form');
  assert.deepEqual(sumsText.split('\n').filter(Boolean), [...sumsText.split('\n').filter(Boolean)].sort((a, b) => (a.slice(66) < b.slice(66) ? -1 : 1)), 'SHA256SUMS is sorted by path');
  const sha256sum = spawnSync('sha256sum', ['-c', '--quiet', SUMS_FILE], { cwd: extracted, encoding: 'utf8', windowsHide: true });
  if (sha256sum.error) {
    console.log('  (sha256sum is not on PATH here; the inner sums were checked in-process only)');
  } else {
    assert.equal(sha256sum.status, 0, `sha256sum -c disagrees: ${sha256sum.stdout}${sha256sum.stderr}`);
  }
  console.log(`PASS: the asset extracts with tar; ${sums.size} inner sums match${sha256sum.error ? '' : ' (sha256sum -c agrees)'}, both worker shells, zipp/ and undici are inside`);

  // --- oaiy-cli.json IS the extracted CLI's capabilities --json --------------
  const capsText = fs.readFileSync(path.join(extracted, CAPABILITIES_FILE), 'utf8');
  const caps = JSON.parse(capsText);
  const rerun = spawnSync(process.execPath, [path.join(extracted, 'oaiy.mjs'), 'capabilities', '--json'], { cwd: extracted, env, encoding: 'utf8', windowsHide: true });
  assert.equal(rerun.status, 0, `the extracted CLI does not report a ready engine: ${rerun.stderr}`);
  assert.equal(rerun.stdout, capsText, 'oaiy-cli.json is byte-for-byte the extracted CLI\'s `capabilities --json`');
  assert.deepEqual(caps.protocols, { run: 1, script: 1, profile: 1 });
  assert.equal(caps.engine.status, 'ready');
  assert.equal(caps.engine.wasmSha256, stagedSource.sha256, 'the engine digest is the staged SOURCE.json digest');
  assert.equal(caps.engine.release, stagedSource.release);
  assert.deepEqual(caps.run.languages, ['javascript']);
  assert.deepEqual(caps.script.languages, stagedSource.languages);
  assert.deepEqual(summary.capabilities, caps, 'the summary carries the same report');
  console.log(`PASS: oaiy-cli.json equals the extracted CLI's capabilities --json verbatim (protocols ${JSON.stringify(caps.protocols)}, engine ${caps.engine.release} ready)`);

  // --- a CLI whose engine does not check is refused; the environment is scrubbed ---
  const broken = path.join(dir, 'broken');
  fs.cpSync(staged, broken, { recursive: true });
  const wasm = path.join(broken, 'zipp', 'zipp_wasm_bg.wasm');
  const bytes = fs.readFileSync(wasm);
  bytes[Math.floor(bytes.length / 2)] ^= 0x01;
  fs.writeFileSync(wasm, bytes);
  const refusedOut = path.join(dir, 'never.tar.gz');
  // OAIY_ZIPP_ASSET_DIR in OUR environment names a GOOD engine: the copy must not see it.
  process.env.OAIY_ZIPP_ASSET_DIR = path.join(staged, 'zipp');
  try {
    assert.throws(() => packCliAsset(broken, refusedOut), /does not report a ready engine[\s\S]*sha256/, 'a corrupt staged engine must be refused');
  } finally {
    delete process.env.OAIY_ZIPP_ASSET_DIR;
  }
  assert.ok(!fs.existsSync(refusedOut), 'no asset is written for a refused CLI');
  // And the same folder, unbroken, packs — so the refusal above was the engine's, not the builder's.
  fs.writeFileSync(wasm, fs.readFileSync(path.join(staged, 'zipp', 'zipp_wasm_bg.wasm')));
  packCliAsset(broken, refusedOut);
  assert.ok(fs.existsSync(refusedOut));
  console.log('PASS: a staged CLI whose engine does not check is refused with no asset written, even with a good OAIY_ZIPP_ASSET_DIR in the builder\'s own environment');
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
