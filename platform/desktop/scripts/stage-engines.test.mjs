// The portable language-model engine the installer carries (stage-engines-lib.mjs): staged where engines.rs looks,
// refused when it was not built, and wired so that `tauri build` cannot make an installer without it.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { BUILD_COMMAND, ENGINE, EngineStagingError, engineFile, stageEngine, stagedEnginesDir } from './stage-engines-lib.mjs';

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

const build = (platform, bytes = Buffer.from('MZ portable engine')) => {
  const dir = path.join(repo, 'target', 'release');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, engineFile(platform)), bytes);
};

describe('the portable language-model engine', () => {
  it('is staged alone where engines.rs looks, as the build made it', () => {
    for (const platform of ['win32', 'linux']) {
      build(platform);
      const leftover = path.join(stagedEnginesDir(fakeDesktop), 'stale.exe');
      fs.mkdirSync(path.dirname(leftover), { recursive: true });
      fs.writeFileSync(leftover, 'old');
      const lines = [];
      const { file, bytes } = stageEngine({ repo, desktop: fakeDesktop, platform, log: (l) => lines.push(l) });
      expect(file).toBe(path.join(fakeDesktop, 'src-tauri', 'resources', 'engines', platform === 'win32' ? `${ENGINE}.exe` : ENGINE));
      expect(fs.readFileSync(file, 'utf8')).toBe('MZ portable engine');
      expect(bytes).toBe(18);
      expect(fs.readdirSync(stagedEnginesDir(fakeDesktop))).toEqual([path.basename(file)]);
      expect(lines.join('\n')).toContain('verified');
    }
  });

  it('is refused when it was not built, and the error says how to build it', () => {
    expect(() => stageEngine({ repo, desktop: fakeDesktop, platform: 'win32' })).toThrow(EngineStagingError);
    expect(() => stageEngine({ repo, desktop: fakeDesktop, platform: 'win32' })).toThrow(BUILD_COMMAND);
  });

  it('is refused when its build is empty', () => {
    build('linux', Buffer.alloc(0));
    expect(() => stageEngine({ repo, desktop: fakeDesktop, platform: 'linux' })).toThrow(/is empty/);
  });

  it('is listed in bundle.resources, made for a build that stages nothing, and staged by Tauri before it builds', () => {
    const config = JSON.parse(read('src-tauri', 'tauri.conf.json'));
    const manifest = JSON.parse(read('package.json'));
    expect(config.bundle.resources).toContain('resources/engines');
    // Tauri fails on a listed folder that is missing: build.rs makes it for the tests and tauri dev.
    expect(read('src-tauri', 'build.rs')).toContain('"resources/engines"');
    expect(manifest.scripts['stage-engines']).toBe('node scripts/stage-engines.mjs');
    expect(manifest.scripts['build:bundle']).toContain('npm run stage-engines');
    // The build command the error names is the one the release workflow runs.
    const release = fs.readFileSync(path.join(repository, '.github', 'workflows', 'release.yml'), 'utf8');
    expect(release).toContain(BUILD_COMMAND.replace('cargo build --release', 'cargo build --release --locked'));
  });

  const git = spawnSync('git', ['--version']).status === 0;
  it.skipIf(!git)('is git-ignored where it is staged', () => {
    const staged = path.relative(repository, path.join(stagedEnginesDir(desktop), engineFile()));
    expect(spawnSync('git', ['check-ignore', '-q', staged], { cwd: repository }).status, `${staged} is not ignored`).toBe(0);
  });
});
