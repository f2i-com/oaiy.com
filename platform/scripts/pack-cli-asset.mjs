/**
 * Pack the staged OAIY CLI into the standalone release asset `oaiy-cli-<v>.tar.gz`.
 *
 *     node scripts/pack-cli-asset.mjs <staged-cli-dir> <out.tar.gz> [--summary <file>] [--node <exe>]
 *
 * `<staged-cli-dir>` is what `desktop/scripts/sync-cli.mjs` stages into
 * `desktop/src-tauri/resources/cli`: `oaiy.mjs`, both worker shells, `zipp/`
 * and `node_modules/undici`. The asset is OS-neutral (JavaScript + wasm) and
 * is what a consumer that embeds the CLI — FormLogic's fetcher, resolving the
 * latest OAIY release per run — downloads and verifies.
 *
 * What goes in, and why in this order:
 *
 *   1. The folder is COPIED first. `package-server.mjs` has already inventoried
 *      `resources/cli` into `headless-dist/distribution.json` and
 *      `smoke-server.mjs` re-hashes that inventory; writing two files into the
 *      source folder would make the two disagree. Nothing here touches the
 *      staged folder.
 *   2. `oaiy-cli.json` — the copy's own `oaiy capabilities --json` output,
 *      VERBATIM (decision 5-3): the protocols record (`run`, `script`,
 *      `profile`), the engine identity and its `ready` status, `run.languages`
 *      and `script.languages`. It is produced by running the copy with every
 *      `OAIY_*` variable scrubbed from the environment, so it describes the
 *      tarball and not this machine; a CLI whose engine does not check
 *      (exit 1) is refused, and no asset is written.
 *   3. `SHA256SUMS` — `<sha256>  <path>` for every other file, sorted by path,
 *      the two-space form `sha256sum -c SHA256SUMS` reads.
 *   4. The tarball: ustar entries in the same sorted order, gzip. Written here
 *      rather than by the platform's `tar` so the format is the same on every
 *      runner and needs no tool.
 *   5. Read back and checked: every file in the archive against `SHA256SUMS`,
 *      no file missing or extra, `oaiy-cli.json` valid. A packing error is a
 *      failed build, not a bad asset.
 *
 * `--summary <file>` writes `{ asset, sha256, bytes, files, capabilities }` for
 * the release evidence. `--node <exe>` picks the Node that runs the copy
 * (default: this process's).
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import zlib from 'node:zlib';

export const CAPABILITIES_FILE = 'oaiy-cli.json';
export const SUMS_FILE = 'SHA256SUMS';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

class PackError extends Error {}

/** Every regular file under `dir`, as sorted POSIX-relative paths. Symlinks are refused: nothing here should be one. */
function inventory(dir) {
  const out = [];
  const walk = (rel) => {
    const full = rel ? path.join(dir, rel) : dir;
    for (const entry of fs.readdirSync(full, { withFileTypes: true })) {
      const child = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isSymbolicLink()) throw new PackError(`${child} is a symbolic link; the asset carries files only`);
      if (entry.isDirectory()) walk(child);
      else if (entry.isFile()) out.push(child);
      else throw new PackError(`${child} is neither a file nor a directory`);
    }
  };
  walk('');
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

/** The environment the copy runs in: this one minus every OAIY_* override (asset dir, tokens, debug). */
function scrubbedEnv() {
  const env = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (!/^OAIY_/i.test(key)) env[key] = value;
  }
  return env;
}

// --- ustar writer -----------------------------------------------------------

function octal(n, width) {
  const s = n.toString(8);
  if (s.length > width - 1) throw new PackError(`value ${n} does not fit a ${width}-byte tar field`);
  return s.padStart(width - 1, '0') + '\0';
}

/** Split a long path into ustar `prefix` (≤155) and `name` (≤100) at a `/`. */
function splitName(name) {
  if (Buffer.byteLength(name) <= 100) return { prefix: '', name };
  for (let i = name.length - 1; i > 0; i--) {
    if (name[i] !== '/') continue;
    const prefix = name.slice(0, i);
    const rest = name.slice(i + 1);
    if (Buffer.byteLength(prefix) <= 155 && Buffer.byteLength(rest) <= 100 && rest.length > 0) return { prefix, name: rest };
  }
  throw new PackError(`path too long for a ustar header: ${name}`);
}

function header({ name, size, mode, mtime, type }) {
  const block = Buffer.alloc(512, 0);
  const split = splitName(name);
  block.write(split.name, 0, 100, 'utf8');
  block.write(octal(mode, 8), 100);
  block.write(octal(0, 8), 108); // uid
  block.write(octal(0, 8), 116); // gid
  block.write(octal(size, 12), 124);
  block.write(octal(mtime, 12), 136);
  block.write('        ', 148); // checksum placeholder: eight spaces
  block.write(type, 156, 1);
  block.write('ustar\0', 257);
  block.write('00', 263);
  block.write('root', 265, 32); // uname
  block.write('root', 297, 32); // gname
  block.write(octal(0, 8), 329);
  block.write(octal(0, 8), 337);
  block.write(split.prefix, 345, 155, 'utf8');
  let sum = 0;
  for (const byte of block) sum += byte;
  block.write(sum.toString(8).padStart(6, '0') + '\0 ', 148);
  return block;
}

/** Build a ustar archive of `files` (sorted relative paths) under `dir`, with a directory entry for every folder. */
export function tarOf(dir, files, { mtime = Math.floor(Date.now() / 1000) } = {}) {
  const parts = [];
  const dirs = new Set();
  for (const file of files) {
    const segments = file.split('/');
    for (let i = 1; i < segments.length; i++) dirs.add(segments.slice(0, i).join('/'));
  }
  for (const d of [...dirs].sort()) {
    parts.push(header({ name: d + '/', size: 0, mode: 0o755, mtime, type: '5' }));
  }
  for (const file of files) {
    const bytes = fs.readFileSync(path.join(dir, file));
    parts.push(header({ name: file, size: bytes.length, mode: 0o644, mtime, type: '0' }));
    parts.push(bytes);
    const pad = (512 - (bytes.length % 512)) % 512;
    if (pad) parts.push(Buffer.alloc(pad, 0));
  }
  parts.push(Buffer.alloc(1024, 0));
  return Buffer.concat(parts);
}

/** Read a ustar archive back: `Map<path, Buffer>` of its regular files. */
export function untar(archive) {
  const files = new Map();
  let at = 0;
  const text = (start, len) => archive.subarray(start, start + len).toString('utf8').replace(/\0.*$/s, '');
  while (at + 512 <= archive.length) {
    const block = archive.subarray(at, at + 512);
    if (block.every((b) => b === 0)) break;
    const name = text(at, 100);
    const size = parseInt(text(at + 124, 12).trim() || '0', 8);
    const type = String.fromCharCode(block[156]) || '0';
    const prefix = text(at + 345, 155);
    const full = prefix ? `${prefix}/${name}` : name;
    at += 512;
    if (type === '0' || type === '\0') files.set(full, Buffer.from(archive.subarray(at, at + size)));
    at += Math.ceil(size / 512) * 512;
  }
  return files;
}

// --- the asset --------------------------------------------------------------

/**
 * Run the copy's `capabilities --json`. The report is the file, byte for byte;
 * the exit code says whether the staged engine checks.
 */
function capabilitiesOf(copy, nodeExe) {
  const run = spawnSync(nodeExe, [path.join(copy, 'oaiy.mjs'), 'capabilities', '--json'], {
    cwd: copy,
    env: scrubbedEnv(),
    encoding: 'utf8',
    windowsHide: true,
    timeout: 60_000,
  });
  if (run.error) throw new PackError(`could not run the staged CLI: ${run.error.message}`);
  if (run.status !== 0) {
    throw new PackError(`the staged CLI does not report a ready engine (exit ${run.status}):\n${run.stdout}${run.stderr}`);
  }
  let report;
  try {
    report = JSON.parse(run.stdout);
  } catch (e) {
    throw new PackError(`capabilities --json did not print JSON: ${e.message}\n${run.stdout}`);
  }
  if (typeof report?.protocols !== 'object' || report?.engine?.status !== 'ready') {
    throw new PackError(`capabilities --json is not a ready report: ${run.stdout}`);
  }
  return { text: run.stdout, report };
}

export function sumsText(dir, files) {
  return files.map((file) => `${sha256(fs.readFileSync(path.join(dir, file)))}  ${file}\n`).join('');
}

/** Parse `SHA256SUMS` text back into `Map<path, sha256>`. */
export function parseSums(text) {
  const out = new Map();
  for (const line of text.split('\n')) {
    if (!line) continue;
    const m = /^([0-9a-f]{64})  (.+)$/.exec(line);
    if (!m) throw new PackError(`malformed SHA256SUMS line: ${JSON.stringify(line)}`);
    out.set(m[2], m[1]);
  }
  return out;
}

/** Check an archive's bytes against its own SHA256SUMS and oaiy-cli.json. Throws naming the first fault. */
export function verifyAsset(archiveBytes) {
  const files = untar(zlib.gunzipSync(archiveBytes));
  const sums = files.get(SUMS_FILE);
  if (!sums) throw new PackError(`the asset carries no ${SUMS_FILE}`);
  const expected = parseSums(sums.toString('utf8'));
  for (const [file, digest] of expected) {
    const bytes = files.get(file);
    if (!bytes) throw new PackError(`${SUMS_FILE} lists ${file}, which is not in the asset`);
    const actual = sha256(bytes);
    if (actual !== digest) throw new PackError(`${file} has sha256 ${actual} in the asset, not the ${digest} ${SUMS_FILE} records`);
  }
  for (const file of files.keys()) {
    if (file !== SUMS_FILE && !expected.has(file)) throw new PackError(`${file} is in the asset but not in ${SUMS_FILE}`);
  }
  const caps = files.get(CAPABILITIES_FILE);
  if (!caps) throw new PackError(`the asset carries no ${CAPABILITIES_FILE}`);
  let report;
  try {
    report = JSON.parse(caps.toString('utf8'));
  } catch (e) {
    throw new PackError(`${CAPABILITIES_FILE} is not JSON: ${e.message}`);
  }
  if (typeof report?.protocols !== 'object' || report?.engine?.status !== 'ready') {
    throw new PackError(`${CAPABILITIES_FILE} is not a ready capabilities report`);
  }
  for (const required of ['oaiy.mjs', 'oaiy-zipp-worker.mjs', 'oaiy-script-worker.mjs', 'zipp/zipp_wasm_bg.wasm', 'zipp/SOURCE.json']) {
    if (!files.has(required)) throw new PackError(`the asset is missing ${required}`);
  }
  return { files: files.size, report };
}

/**
 * Pack `stagedDir` into `out`. Returns the summary. Throws `PackError` (or a
 * filesystem error) and writes no asset when anything does not check.
 */
export function packCliAsset(stagedDir, out, { nodeExe = process.execPath, summary = null } = {}) {
  const staged = path.resolve(stagedDir);
  if (!fs.existsSync(path.join(staged, 'oaiy.mjs'))) throw new PackError(`${staged} holds no staged CLI (no oaiy.mjs)`);
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-cli-asset-'));
  try {
    const copy = path.join(work, 'cli');
    fs.cpSync(staged, copy, { recursive: true, dereference: false });
    for (const stale of [CAPABILITIES_FILE, SUMS_FILE]) {
      if (fs.existsSync(path.join(copy, stale))) throw new PackError(`${staged} already contains ${stale}; the staged folder must be the CLI alone`);
    }
    const caps = capabilitiesOf(copy, nodeExe);
    fs.writeFileSync(path.join(copy, CAPABILITIES_FILE), caps.text);
    const files = inventory(copy);
    fs.writeFileSync(path.join(copy, SUMS_FILE), sumsText(copy, files));
    const archive = zlib.gzipSync(tarOf(copy, [...files, SUMS_FILE]), { level: 9 });
    const checked = verifyAsset(archive);
    fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
    fs.writeFileSync(out, archive);
    const result = {
      asset: path.basename(out),
      sha256: sha256(archive),
      bytes: archive.length,
      files: checked.files,
      capabilities: caps.report,
    };
    if (summary) fs.writeFileSync(summary, JSON.stringify(result, null, 2) + '\n');
    return result;
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}

const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;
if (isMain) {
  const args = process.argv.slice(2);
  const take = (flag) => {
    const at = args.indexOf(flag);
    if (at === -1) return null;
    const [, value] = args.splice(at, 2);
    if (!value) throw new PackError(`${flag} needs a value`);
    return value;
  };
  try {
    const summary = take('--summary');
    const nodeExe = take('--node') ?? process.execPath;
    const [stagedDir, out] = args;
    if (!stagedDir || !out || args.length !== 2) {
      console.error(`usage: node ${path.relative(process.cwd(), fileURLToPath(import.meta.url))} <staged-cli-dir> <out.tar.gz> [--summary <file>] [--node <exe>]`);
      process.exit(2);
    }
    const result = packCliAsset(stagedDir, out, { nodeExe, summary });
    const { protocols, engine } = result.capabilities;
    console.log(
      `pack-cli-asset: wrote ${out} (${result.files} files, ${(result.bytes / 1024 / 1024).toFixed(1)} MB, sha256 ${result.sha256.slice(0, 12)}) — ` +
        `protocols ${JSON.stringify(protocols)}, engine ${engine.name} ${engine.release} ${String(engine.wasmSha256).slice(0, 12)}`,
    );
  } catch (e) {
    console.error(`pack-cli-asset: ${e instanceof PackError ? e.message : e.stack ?? e}`);
    process.exit(1);
  }
}
