// OAIY's engines the installer carries (stage-engines-lib.mjs), the language-model engine, the media engine and the
// voice: staged where engines.rs (and, for the voice, the services' registry) looks, refused when one was not built,
// and wired so that `tauri build` cannot make an installer without them.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { BUILD_COMMAND, ENGINE, ENGINES, EngineStagingError, engineFile, stageEngines, stagedEnginesDir } from './stage-engines-lib.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const desktop = path.resolve(here, '..');
const repository = path.resolve(desktop, '..', '..');
const read = (...parts) => fs.readFileSync(path.join(desktop, ...parts), 'utf8');

let base;
let repo;
let fakeDesktop;
beforeEach(() => {
  base = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-stage-engines-'));
  repo = path.join(base, 'repo');
  fakeDesktop = path.join(repo, 'platform', 'desktop');
  fs.mkdirSync(path.join(fakeDesktop, 'src-tauri', 'resources'), { recursive: true });
});
afterEach(() => fs.rmSync(base, { recursive: true, force: true }));

const build = (platform, name, bytes = Buffer.from(`MZ ${name}`)) => {
  const dir = path.join(repo, 'target', 'release');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, engineFile(name, platform)), bytes);
};
const buildAll = (platform) => ENGINES.forEach(({ name }) => build(platform, name));

describe("OAIY's engines", () => {
  it('are the language-model engine, the media engine and the voice', () => {
    expect(ENGINES.map((e) => e.name)).toEqual(['oaiy-llm-server-webgpu', 'oaiy-media', 'oaiy-voice']);
    expect(ENGINE).toBe('oaiy-llm-server-webgpu');
    expect(BUILD_COMMAND).toBe(ENGINES[0].build);
  });

  it('are staged alone where engines.rs looks, as the builds made them', () => {
    for (const platform of ['win32', 'linux']) {
      buildAll(platform);
      const leftover = path.join(stagedEnginesDir(fakeDesktop), 'stale.exe');
      fs.mkdirSync(path.dirname(leftover), { recursive: true });
      fs.writeFileSync(leftover, 'old');
      const lines = [];
      const staged = stageEngines({ repo, desktop: fakeDesktop, platform, log: (l) => lines.push(l) });
      const names = ENGINES.map(({ name }) => (platform === 'win32' ? `${name}.exe` : name));
      expect(staged.map((s) => s.file)).toEqual(names.map((n) => path.join(fakeDesktop, 'src-tauri', 'resources', 'engines', n)));
      ENGINES.forEach(({ name }, i) => {
        expect(fs.readFileSync(staged[i].file, 'utf8')).toBe(`MZ ${name}`);
        expect(staged[i].bytes).toBe(`MZ ${name}`.length);
      });
      expect(fs.readdirSync(stagedEnginesDir(fakeDesktop)).sort()).toEqual([...names].sort());
      expect(lines.join('\n').match(/verified/g)).toHaveLength(ENGINES.length);
    }
  });

  it('are refused when one was not built, nothing staged, and the error says how to build that one', () => {
    for (const missing of ENGINES) {
      fs.rmSync(path.join(repo, 'target'), { recursive: true, force: true });
      ENGINES.filter((e) => e !== missing).forEach(({ name }) => build('win32', name));
      expect(() => stageEngines({ repo, desktop: fakeDesktop, platform: 'win32' })).toThrow(EngineStagingError);
      expect(() => stageEngines({ repo, desktop: fakeDesktop, platform: 'win32' })).toThrow(missing.build);
      expect(fs.existsSync(path.join(stagedEnginesDir(fakeDesktop), engineFile(missing.name, 'win32')))).toBe(false);
    }
  });

  it('are refused when a build is empty', () => {
    buildAll('linux');
    build('linux', 'oaiy-media', Buffer.alloc(0));
    expect(() => stageEngines({ repo, desktop: fakeDesktop, platform: 'linux' })).toThrow(/oaiy-media is empty/);
  });

  it('are listed in bundle.resources, made for a build that stages nothing, and staged by Tauri before it builds', () => {
    const config = JSON.parse(read('src-tauri', 'tauri.conf.json'));
    const manifest = JSON.parse(read('package.json'));
    expect(config.bundle.resources).toContain('resources/engines');
    // Tauri fails on a listed folder that is missing: build.rs makes it for the tests and tauri dev.
    expect(read('src-tauri', 'build.rs')).toContain('"resources/engines"');
    expect(manifest.scripts['stage-engines']).toBe('node scripts/stage-engines.mjs');
    expect(manifest.scripts['build:bundle']).toContain('npm run stage-engines');
    // The build commands the errors name are the ones the release workflow runs, and the Mac's builds each engine.
    const release = fs.readFileSync(path.join(repository, '.github', 'workflows', 'release.yml'), 'utf8');
    const mac = fs.readFileSync(path.join(repository, '.github', 'workflows', 'mac-build.yml'), 'utf8');
    for (const { build: command } of ENGINES) {
      expect(release).toContain(command.replace('cargo build --release', 'cargo build --release --locked'));
    }
    // (the Mac's language-model engine is built with `tinygpu`, so its command is its own)
    for (const { name, build: command } of ENGINES.filter((e) => e.name !== 'oaiy-llm-server-webgpu')) {
      expect(mac).toContain(command.replace('cargo build --release', 'cargo build --release --locked'));
      expect(release, `release.yml does not check ${name} starts`).toContain(`target/release/${name} --help > /dev/null`);
      expect(mac, `mac-build.yml does not check ${name} starts`).toContain(`target/release/${name} --help > /dev/null`);
    }
    expect(mac).toContain('--features tinygpu --bin oaiy-llm-server-webgpu');
  });

  const git = spawnSync('git', ['--version']).status === 0;
  it.skipIf(!git)('are git-ignored where they are staged', () => {
    for (const { name } of ENGINES) {
      const staged = path.relative(repository, path.join(stagedEnginesDir(desktop), engineFile(name)));
      expect(spawnSync('git', ['check-ignore', '-q', staged], { cwd: repository }).status, `${staged} is not ignored`).toBe(0);
    }
  });
});
