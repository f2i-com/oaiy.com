/**
 * What `sync-cli.mjs` stages and how it is checked — as functions, so a test can
 * drive them against a fixture without wiping the real `resources/cli`.
 *
 * The CLI runs every flow on the ZIPP VM, so the desktop ships the engine with
 * it: the two worker shells (workflow and leaf-script) and the five files
 * `cli/esbuild.mjs` stages into `dist/zipp/` beside the bundle. A bundle
 * staged without them resolves, probes
 * `engine_unavailable`, and every run fails — so staging is verified, not
 * assumed, and the identity comes from the staged `SOURCE.json`, never from a
 * literal in this file.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

/** The bundle, the worker shells it spawns, and the engine folder. */
export const BUNDLE = 'oaiy.mjs';
/** The workflow shell (`oaiy run`). */
export const WORKER = 'oaiy-zipp-worker.mjs';
/** The leaf-script shell (`oaiy script`, the Desktop's script host). */
export const SCRIPT_WORKER = 'oaiy-script-worker.mjs';
/** Exactly the shells `cli/esbuild.mjs` (`ZIPP_WORKERS`) builds beside the bundle; the CLI refuses to run without both. */
export const WORKERS = [WORKER, SCRIPT_WORKER];
export const ZIPP_DIR = 'zipp';
/** Exactly what `cli/esbuild.mjs` (`ZIPP_STAGED`) puts in `dist/zipp/` — keep the two lists equal. */
export const ZIPP_STAGED = ['zipp_wasm_bg.wasm', 'PROFILE.json', 'SOURCE.json', 'LICENSE-APACHE', 'THIRD_PARTY_LICENSES.txt'];

/** Every staged path, relative to the destination, that a run needs. */
export const STAGED_FILES = [BUNDLE, ...WORKERS, ...ZIPP_STAGED.map((name) => path.join(ZIPP_DIR, name))];

export class StagingError extends Error {}

const sha256 = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');

/** The dist files whose absence means the CLI has to be (re)built. */
export function missingBuildOutputs(dist) {
  return [BUNDLE, ...WORKERS, path.join(ZIPP_DIR, 'zipp_wasm_bg.wasm'), path.join(ZIPP_DIR, 'SOURCE.json')]
    .filter((rel) => !fs.existsSync(path.join(dist, rel)));
}

/**
 * Copy the bundle, the worker and the engine folder from `dist` into `dest`.
 * `dest` must already exist; the caller decides what else lives there.
 */
export function stageEngine(dist, dest) {
  for (const rel of STAGED_FILES) {
    const from = path.join(dist, rel);
    if (!fs.existsSync(from)) throw new StagingError(`the CLI build produced no ${from}`);
    const to = path.join(dest, rel);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    fs.copyFileSync(from, to);
  }
}

/**
 * Check a staged folder against its own record and its source.
 *
 *   * every file in STAGED_FILES is present;
 *   * the staged wasm's sha256 is the staged `SOURCE.json.sha256`, and the
 *     notices file's is `SOURCE.json.notices.sha256`;
 *   * the bundle contains that wasm digest — the `__ZIPP_WASM_SHA256__` define
 *     `cli/esbuild.mjs` bakes in — so bundle and bytes were built together;
 *   * when `dist` is given, every staged file is byte-equal to its source.
 *
 * Throws a `StagingError` naming the first file that fails.
 */
export function verifyStaged(dest, dist = null) {
  for (const rel of STAGED_FILES) {
    if (!fs.existsSync(path.join(dest, rel))) throw new StagingError(`staged CLI is missing ${rel}`);
  }
  let source;
  try {
    source = JSON.parse(fs.readFileSync(path.join(dest, ZIPP_DIR, 'SOURCE.json'), 'utf8'));
  } catch (error) {
    throw new StagingError(`staged ${path.join(ZIPP_DIR, 'SOURCE.json')} is not readable JSON: ${error.message}`);
  }
  const hex64 = /^[0-9a-f]{64}$/;
  if (!hex64.test(source.sha256 ?? '')) throw new StagingError(`staged ${path.join(ZIPP_DIR, 'SOURCE.json')} records no wasm sha256`);
  if (!hex64.test(source.notices?.sha256 ?? '')) throw new StagingError(`staged ${path.join(ZIPP_DIR, 'SOURCE.json')} records no notices sha256`);

  const wasm = path.join(ZIPP_DIR, 'zipp_wasm_bg.wasm');
  const wasmSha = sha256(path.join(dest, wasm));
  if (wasmSha !== source.sha256) {
    throw new StagingError(`staged ${wasm} has sha256 ${wasmSha}, not the ${source.sha256} its SOURCE.json records`);
  }
  const notices = path.join(ZIPP_DIR, source.notices.file || 'THIRD_PARTY_LICENSES.txt');
  const noticesSha = fs.existsSync(path.join(dest, notices)) ? sha256(path.join(dest, notices)) : null;
  if (noticesSha !== source.notices.sha256) {
    throw new StagingError(`staged ${notices} has sha256 ${noticesSha}, not the ${source.notices.sha256} its SOURCE.json records`);
  }
  if (!fs.readFileSync(path.join(dest, BUNDLE), 'utf8').includes(source.sha256)) {
    throw new StagingError(`staged ${BUNDLE} was not built for the staged engine (it does not carry wasm sha256 ${source.sha256})`);
  }
  if (dist) {
    for (const rel of STAGED_FILES) {
      if (!fs.readFileSync(path.join(dest, rel)).equals(fs.readFileSync(path.join(dist, rel)))) {
        throw new StagingError(`staged ${rel} differs from its source ${path.join(dist, rel)}`);
      }
    }
  }
  return { release: source.release, wasmSha256: source.sha256 };
}
