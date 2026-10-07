/**
 * Stage the portable language-model engine into src-tauri/resources/engines, so the installer carries it.
 *
 * OAIY's engines run a language model with `oaiy-llm-server`, whose programs used to be a separate, hand-built
 * channel: an installed OAIY could download a model and had nothing to run it with. The server, staged under the name
 * `oaiy-llm-server-webgpu` (every model on any graphics card through WebGPU, else the CPU), is about 15 MB and
 * needs nothing a system lacks, so every installer carries it. The desktop finds it in `<install>/resources/engines`
 * (src-tauri/src/engines.rs), after a build of one's own put beside the program.
 *
 * This does not build the engine: `cargo build` of it takes minutes, and the release workflow runs it before
 * `tauri build`. It checks that the build is there, copies it, and verifies the copy against the build; a missing or
 * empty build is a failed build, because an installer without it would offer models it cannot run.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export class EngineStagingError extends Error {}

export const ENGINE = 'oaiy-llm-server-webgpu';

/** How to build it, from the repository's root: what an error tells the person to run. */
export const BUILD_COMMAND = 'cargo build --release -p oaiy-llm-server --no-default-features --features webgpu --bin oaiy-llm-server-webgpu';

/** The program's file name on `platform`. */
export function engineFile(platform = process.platform) {
  return platform === 'win32' ? `${ENGINE}.exe` : ENGINE;
}

/** Where it is staged: listed in tauri.conf.json's `bundle.resources`, and where engines.rs looks once installed. */
export function stagedEnginesDir(desktop) {
  return path.join(desktop, 'src-tauri', 'resources', 'engines');
}

const posix = (p) => p.split(path.sep).join('/');
const sha256 = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

/**
 * Copy the release build of the engine (`<repo>/target/release`) into the staged folder, alone, and verify it.
 * `repo` is the repository's root, `desktop` is `platform/desktop`.
 */
export function stageEngine({ repo, desktop, platform = process.platform, log = () => {} }) {
  const name = engineFile(platform);
  const source = path.join(repo, 'target', 'release', name);
  if (!fs.existsSync(source) || !fs.statSync(source).isFile()) {
    throw new EngineStagingError(`the portable language-model engine is not built: ${posix(path.relative(repo, source))} is missing. Build it from the repository's root with: ${BUILD_COMMAND}`);
  }
  const bytes = fs.statSync(source).size;
  if (bytes === 0) throw new EngineStagingError(`${posix(path.relative(repo, source))} is empty. Build it again with: ${BUILD_COMMAND}`);
  const dir = stagedEnginesDir(desktop);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const dest = path.join(dir, name);
  fs.copyFileSync(source, dest);
  if (platform !== 'win32') fs.chmodSync(dest, 0o755);
  if (sha256(dest) !== sha256(source)) {
    fs.rmSync(dir, { recursive: true, force: true });
    throw new EngineStagingError(`the staged copy of ${name} is not its build`);
  }
  log(`staged ${name} (${(bytes / 1024 / 1024).toFixed(1)} MB) into ${posix(path.relative(repo, dest))}, verified`);
  return { file: dest, bytes };
}
