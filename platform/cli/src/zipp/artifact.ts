/**
 * The ZIPP engine the CLI ships — finding it, proving it is the one this build
 * was made from, compiling it once.
 *
 * The CLI embeds the web-python bundle of one ZIPP release. Its identity is
 * not written anywhere in this source tree: `cli/esbuild.mjs` verifies the
 * installed release (`checkInstall`, the same check ui's tests run), reads its
 * `SOURCE.json`, and bakes the result into the bundle as two `define`s —
 * `__ZIPP_WASM_SHA256__` and `__ZIPP_ENGINE__`. The `.wasm` itself is staged
 * to `dist/zipp/` beside the CLI. A build and its artifact therefore travel as
 * a pair, and this module is where the pair is checked: the staged bytes must
 * hash to the digest the build recorded, or there is no engine.
 *
 * There is no fallback of any kind. A missing or altered artifact is
 * `EngineUnavailableError` (code `engine_unavailable`), raised before a
 * `JobManager` exists and so before any job can be submitted; `runFlow` turns
 * it into a failed result that never ran. The in-thread `new Function` path
 * the runtime still carries for the browser is unreachable from here —
 * `createCliEngine` forces `requireScriptExecutor: true`.
 *
 * `OAIY_ZIPP_ASSET_DIR` relocates the staged folder (tests point it at an
 * empty or corrupted copy). It cannot substitute an engine: whatever is there
 * still has to hash to the build's digest.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/** The engine cannot be had; nothing was run. `code` is what the CLI reports. */
export class EngineUnavailableError extends Error {
  readonly code = 'engine_unavailable' as const;
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'EngineUnavailableError';
  }
}

/** What the build recorded from the installed release's SOURCE.json. */
export interface ZippEngineIdentity {
  name: 'zipp';
  release: string;
  version: string;
  revision: string;
  variant: string;
  bundle: string;
  wasmSha256: string;
  glueSha256: string;
  languages: string[];
}

export interface ZippArtifact {
  /** Compiled once per process; structured-cloned to each worker via `workerData`. */
  module: WebAssembly.Module;
  identity: ZippEngineIdentity;
  /** The staged `zipp_wasm_bg.wasm`, whose sha256 is `identity.wasmSha256`. */
  wasmPath: string;
  /** The bundled worker shell (`dist/oaiy-zipp-worker.mjs`), confirmed present. */
  workerEntry: URL;
}

export interface ZippArtifactOptions {
  /**
   * Where `zipp_wasm_bg.wasm` is staged. Default: `OAIY_ZIPP_ASSET_DIR`, else
   * `zipp/` beside this bundle (`dist/zipp/`; the engine-level test bundles
   * live in the same `dist/`, so the relative path serves them too).
   */
  assetDir?: string;
  /** The worker shell to spawn. Default: `oaiy-zipp-worker.mjs` beside this bundle. */
  workerEntry?: URL;
}

const WASM_FILE = 'zipp_wasm_bg.wasm';

/**
 * The identity baked in at build time. A bundle built without
 * `cli/esbuild.mjs`'s defines has no engine, and says so.
 */
export function zippEngineIdentity(): ZippEngineIdentity {
  if (typeof __ZIPP_WASM_SHA256__ !== 'string' || typeof __ZIPP_ENGINE__ !== 'string') {
    throw new EngineUnavailableError(
      'ZIPP engine unavailable: this build carries no engine identity (not built by cli/esbuild.mjs)',
    );
  }
  const identity = JSON.parse(__ZIPP_ENGINE__) as ZippEngineIdentity;
  if (identity.name !== 'zipp' || identity.wasmSha256 !== __ZIPP_WASM_SHA256__) {
    throw new EngineUnavailableError('ZIPP engine unavailable: the build recorded an inconsistent engine identity');
  }
  return identity;
}

/** Successful loads, by location. A refusal is never remembered. */
const loaded = new Map<string, Promise<ZippArtifact>>();

/**
 * Locate, verify and compile the engine. Resolved once per location for the
 * life of the process; rejects with `EngineUnavailableError` and nothing else.
 */
export function loadZippArtifact(opts: ZippArtifactOptions = {}): Promise<ZippArtifact> {
  const assetDir = path.resolve(
    opts.assetDir ?? (process.env.OAIY_ZIPP_ASSET_DIR || fileURLToPath(new URL('./zipp/', import.meta.url))),
  );
  const workerEntry = opts.workerEntry ?? new URL('./oaiy-zipp-worker.mjs', import.meta.url);
  const key = `${assetDir}\n${workerEntry.href}`;

  let pending = loaded.get(key);
  if (!pending) {
    pending = load(assetDir, workerEntry);
    loaded.set(key, pending);
    pending.catch(() => loaded.delete(key));
  }
  return pending;
}

async function load(assetDir: string, workerEntry: URL): Promise<ZippArtifact> {
  const identity = zippEngineIdentity();
  const wasmPath = path.join(assetDir, WASM_FILE);
  const unavailable = (what: string, cause?: unknown) =>
    new EngineUnavailableError(`ZIPP engine unavailable: ${what}`, cause === undefined ? undefined : { cause });

  try {
    if (!(await fs.stat(workerEntry)).isFile()) throw new Error('not a file');
  } catch (e) {
    throw unavailable(`the worker shell ${fileURLToPath(workerEntry)} is missing`, e);
  }

  let bytes: Buffer;
  try {
    bytes = await fs.readFile(wasmPath);
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code;
    throw unavailable(`${wasmPath} cannot be read${code ? ` (${code})` : ''}`, e);
  }

  const actual = createHash('sha256').update(bytes).digest('hex');
  if (actual !== identity.wasmSha256) {
    throw unavailable(`${wasmPath} has sha256 ${actual}, not the ${identity.wasmSha256} this build was made from`);
  }

  let module: WebAssembly.Module;
  try {
    // A fresh Uint8Array: `compile` takes a BufferSource, and a Buffer may sit
    // on a pooled or shared ArrayBuffer.
    module = await WebAssembly.compile(new Uint8Array(bytes));
  } catch (e) {
    throw unavailable(`${wasmPath} does not compile: ${e instanceof Error ? e.message : String(e)}`, e);
  }

  return { module, identity, wasmPath, workerEntry };
}
