/**
 * Stage OAIY's engines into src-tauri/resources/engines, so the installer carries them.
 *
 * OAIY's engines run a language model with `oaiy-llm-server`, whose programs used to be a separate, hand-built
 * channel: an installed OAIY could download a model and had nothing to run it with. The server, staged under the name
 * `oaiy-llm-server-webgpu` (every model on any graphics card through WebGPU, else the CPU), is about 15 MB and
 * needs nothing a system lacks, so every installer carries it. So does the media worker, `oaiy-media` (pictures,
 * video, music, sound effects, speech and 3D models, on any graphics card through WebGPU, some 25 MB): without it an
 * installed OAIY offered image models it had nothing to make a picture with. And OAIY's voice, `oaiy-voice` (speech
 * to text and text to speech for calls, some 15 MB), which the OAIY Voice service runs: no installer carried it, so
 * an installed OAIY's voice said "program not found". The desktop finds the engines in `<install>/resources/engines`
 * (src-tauri/src/engines.rs) and the voice there too (services/registry.rs, `oaiy_program`), after a build of one's
 * own put beside the program.
 *
 * This does not build them: `cargo build` of each takes minutes, and the release workflows run it before
 * `tauri build`. It checks that each build is there, copies it, and verifies the copy against the build; a missing or
 * empty build is a failed build, because an installer without it would offer models it cannot run.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export class EngineStagingError extends Error {}

/**
 * The engines, each with what it is and how to build it from the repository's root (what an error tells the person to
 * run; the release workflow runs each with `--locked`).
 */
export const ENGINES = [
  {
    name: 'oaiy-llm-server-webgpu',
    what: 'the portable language-model engine',
    build: 'cargo build --release -p oaiy-llm-server --no-default-features --features webgpu --bin oaiy-llm-server-webgpu',
  },
  {
    name: 'oaiy-media',
    what: 'the media engine (pictures, video, music, sound and speech)',
    build: 'cargo build --release -p oaiy-media',
  },
  {
    name: 'oaiy-voice',
    what: "OAIY's voice for calls (speech to text and text to speech)",
    build: 'cargo build --release -p oaiy-voice',
  },
];

/** The language-model engine's name and build command (its own, as before the media engine joined it). */
export const ENGINE = ENGINES[0].name;
export const BUILD_COMMAND = ENGINES[0].build;

/** Engine `name`'s file name on `platform`. */
export function engineFile(name = ENGINE, platform = process.platform) {
  return platform === 'win32' ? `${name}.exe` : name;
}

/** Where they are staged: listed in tauri.conf.json's `bundle.resources`, and where engines.rs looks once installed. */
export function stagedEnginesDir(desktop) {
  return path.join(desktop, 'src-tauri', 'resources', 'engines');
}

const posix = (p) => p.split(path.sep).join('/');
const sha256 = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

/**
 * Copy the release build of every engine (`<repo>/target/release`) into the staged folder, alone, and verify each.
 * `repo` is the repository's root, `desktop` is `platform/desktop`. Nothing is staged where one of them is not built.
 */
export function stageEngines({ repo, desktop, platform = process.platform, log = () => {} }) {
  const builds = ENGINES.map((engine) => {
    const name = engineFile(engine.name, platform);
    const source = path.join(repo, 'target', 'release', name);
    if (!fs.existsSync(source) || !fs.statSync(source).isFile()) {
      throw new EngineStagingError(`${engine.what} is not built: ${posix(path.relative(repo, source))} is missing. Build it from the repository's root with: ${engine.build}`);
    }
    const bytes = fs.statSync(source).size;
    if (bytes === 0) throw new EngineStagingError(`${posix(path.relative(repo, source))} is empty. Build it again with: ${engine.build}`);
    return { name, source, bytes };
  });
  const dir = stagedEnginesDir(desktop);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  return builds.map(({ name, source, bytes }) => {
    const dest = path.join(dir, name);
    fs.copyFileSync(source, dest);
    if (platform !== 'win32') fs.chmodSync(dest, 0o755);
    if (sha256(dest) !== sha256(source)) {
      fs.rmSync(dir, { recursive: true, force: true });
      throw new EngineStagingError(`the staged copy of ${name} is not its build`);
    }
    log(`staged ${name} (${(bytes / 1024 / 1024).toFixed(1)} MB) into ${posix(path.relative(repo, dest))}, verified`);
    return { file: dest, bytes };
  });
}
