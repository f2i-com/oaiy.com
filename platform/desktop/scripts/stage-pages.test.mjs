// @vitest-environment node
//
// The staging `stage-pages.mjs` does: the builds of the Agent and the flow editor
// copied into src-tauri/resources/{app,flows}, which the installer carries.
// Driven against fixtures here, never the real folders. What this pins is that a
// page that is missing, empty, incomplete or different from its source is a
// FAILED build naming the file and what to run — not an installer that ships and
// opens on "the page is not built" on the user's machine — and that the layout
// it stages is the one embed.rs looks in.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  PAGES,
  STAGE,
  StagingError,
  checkPage,
  localReferences,
  readTree,
  stagePage,
  stagePages,
  stagedPath,
  verifyStaged,
} from './stage-pages-lib.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const desktop = path.resolve(here, '..');
const repository = path.resolve(desktop, '..', '..');
const read = (...parts) => fs.readFileSync(path.join(desktop, ...parts), 'utf8').replace(/\r\n/g, '\n');

const [AGENT, FLOWS] = PAGES;

let root;
let dest;

/** A build shaped like Vite's output for a page: its start document loading two assets, and what the page requires. */
function writeBuild(dir, page, extra = {}) {
  const files = {
    [page.start]:
      '<!doctype html><html><head><script type="module" crossorigin src="/assets/main-1.js"></script>' +
      '<link rel="stylesheet" crossorigin href="/assets/main-1.css"></head><body><div id="app"></div></body></html>',
    'assets/main-1.js': 'export const main = 1;\n',
    'assets/main-1.css': 'body { margin: 0 }\n',
    ...Object.fromEntries(page.required.map((rel) => [rel, `${rel} of ${page.folder}`])),
    ...extra,
  };
  for (const [rel, text] of Object.entries(files)) {
    const file = path.join(dir, rel);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, text);
  }
  return files;
}

/** A repository shaped like this one, with both pages built. */
function writeRepository(base) {
  writeBuild(path.join(base, ...AGENT.source.split('/')), AGENT);
  writeBuild(path.join(base, ...FLOWS.source.split('/')), FLOWS);
  fs.mkdirSync(path.join(base, 'platform', 'desktop', 'src-tauri', 'resources'), { recursive: true });
  return { repo: base, desktop: path.join(base, 'platform', 'desktop') };
}

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-stage-pages-test-'));
  dest = path.join(root, 'staged');
});

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

describe('the pages, as the desktop names them', () => {
  it("stage each page where embed.rs looks for it and under the start document it opens on", () => {
    const embed = read('src-tauri', 'src', 'embed.rs');
    // The arms of `Page::folder()` and `Page::start()`, by page.
    const arms = (name) => {
      const from = embed.indexOf(`fn ${name}(self)`);
      const body = embed.slice(from, embed.indexOf('\n    }\n', from));
      return Object.fromEntries([...body.matchAll(/Page::(\w+) => "([^"]*)"/g)].map((m) => [m[1], m[2]]));
    };
    const folders = arms('folder');
    const starts = arms('start');
    expect(folders).toMatchObject({ Agent: AGENT.folder, Flows: FLOWS.folder });
    expect(starts.Agent).toBe(`/${AGENT.start}`);
    expect(starts.Flows).toBe(`/${FLOWS.start}`);
  });

  it('are built where the repository builds them, and staged among the resources', () => {
    expect(AGENT.source).toBe('app/dist');
    expect(FLOWS.source).toBe('platform/ui/dist');
    for (const page of PAGES) {
      expect(fs.existsSync(path.join(repository, ...page.source.split('/').slice(0, -1), 'package.json')), page.source).toBe(true);
      expect(stagedPath(desktop, page)).toBe(path.join(desktop, 'src-tauri', 'resources', page.folder));
    }
  });

  it('are listed in bundle.resources, and staged by Tauri before it builds, so an installer cannot go without them', () => {
    const config = JSON.parse(read('src-tauri', 'tauri.conf.json'));
    const manifest = JSON.parse(read('package.json'));
    for (const page of PAGES) expect(config.bundle.resources, page.folder).toContain(`resources/${page.folder}`);
    // Tauri runs beforeBuildCommand whichever way it is started, so this is what stops `npx tauri build` too.
    expect(config.build.beforeBuildCommand).toBe('npm run build:bundle');
    expect(manifest.scripts['build:bundle']).toMatch(/^npm run stage-pages && npm run stage-engines && npm run build$/);
    expect(manifest.scripts['stage-pages']).toBe('node scripts/stage-pages.mjs');
    expect(manifest.scripts['tauri:build']).toMatch(/^tauri build\b/);
    // The plain build stays what the checks run: the pages are the installer's, not the tests'.
    expect(manifest.scripts.build).not.toContain('stage-pages');
  });

  it('are made for a build that stages nothing, so the tests and tauri dev still build', () => {
    // Tauri's build fails on a listed folder that is missing and skips one that is empty.
    const build = read('src-tauri', 'build.rs');
    for (const page of PAGES) expect(build).toContain(`"resources/${page.folder}"`);
    expect(build).toContain('create_dir_all');
  });

  const git = spawnSync('git', ['--version']).status === 0;
  it.skipIf(!git)('are git-ignored where they are staged', () => {
    for (const page of PAGES) {
      const staged = path.relative(repository, path.join(stagedPath(desktop, page), page.start));
      const result = spawnSync('git', ['check-ignore', '-q', staged], { cwd: repository });
      expect(result.status, `${staged} is not ignored`).toBe(0);
    }
  });
});

describe('what a page has to be', () => {
  it('reads the local files a document loads, and not addresses, data, anchors, comments or inline code', () => {
    const html = `
      <link rel="icon" href="/favicon.svg" />
      <link rel="apple-touch-icon" href="apple.png?v=2#x">
      <script type="module" crossorigin src="/assets/app-A1.js"></script>
      <script src='./local.js'></script>
      <script src="https://cdn.example/x.js"></script>
      <script src="//cdn.example/y.js"></script>
      <img src="data:image/png;base64,AAAA">
      <a href="#top">top</a> <a href="mailto:a@example.org">mail</a>
      <a href="/">home</a>
      <script>const s = document.createElement('script'); s.src = "/inline.js";</script>
      <style>@import url("/style-import.css"); a { background: url(x.png) }</style>
      <!-- <script src="/commented.js"></script> -->
      <link href="/with%20space.css">`;
    expect(localReferences(html).sort()).toEqual(['./local.js', '/', '/assets/app-A1.js', '/favicon.svg', '/with space.css', 'apple.png'].sort());
  });

  it('is a start document with everything it loads, and what the page requires', () => {
    writeBuild(dest, AGENT);
    const tree = checkPage(dest, AGENT);
    expect([...tree.keys()].sort()).toEqual(['assets/main-1.css', 'assets/main-1.js', 'index.html', ...AGENT.required].sort());
    writeBuild(path.join(root, 'flows'), FLOWS);
    expect(checkPage(path.join(root, 'flows'), FLOWS).has('app.html')).toBe(true);
  });

  it('is not a build that is not there, and says what to build', () => {
    expect(() => checkPage(dest, AGENT)).toThrow(StagingError);
    expect(() => checkPage(dest, AGENT)).toThrow(/the build of the Agent is not at .*staged: run "npm run build:desktop \(in app\/\)" first/);
    expect(() => checkPage(dest, FLOWS)).toThrow(/the build of the flow editor is not at .*: run "npm run build \(in platform\/ui\/\)" first/);
  });

  it('is not an empty build', () => {
    fs.mkdirSync(path.join(dest, 'assets'), { recursive: true });
    expect(() => checkPage(dest, FLOWS)).toThrow(/the build of the flow editor at .* is empty: run "npm run build \(in platform\/ui\/\)"/);
  });

  it('is not a build without its start document', () => {
    writeBuild(dest, AGENT);
    fs.rmSync(path.join(dest, 'index.html'));
    expect(() => checkPage(dest, AGENT)).toThrow(/the build of the Agent at .* has no index\.html: run "npm run build:desktop/);
  });

  it("is not a build of the Agent without its ZIPP engine or SoftN's runtime, which only build:desktop fetches", () => {
    for (const rel of AGENT.required) {
      const dir = path.join(root, rel.replace(/\W/g, '-'));
      writeBuild(dir, AGENT);
      fs.rmSync(path.join(dir, rel));
      expect(() => checkPage(dir, AGENT), rel).toThrow(new RegExp(`has no ${rel.replace(/[./]/g, '\\$&')}: run "npm run build:desktop`));
    }
  });

  it('is not a start document that loads a file the build does not have', () => {
    writeBuild(dest, FLOWS);
    fs.rmSync(path.join(dest, 'assets', 'main-1.js'));
    expect(() => checkPage(dest, FLOWS)).toThrow(/app\.html in .* loads \/assets\/main-1\.js, which is not there: run "npm run build/);
  });

  it('is not a start document that loads a file outside the page', () => {
    writeBuild(dest, FLOWS, { 'app.html': '<script src="/../secret.js"></script>' });
    expect(() => checkPage(dest, FLOWS)).toThrow(/loads \/\.\.\/secret\.js, which is outside the page/);
  });

  it('may load a folder, as the SoftN runtime is', () => {
    writeBuild(dest, FLOWS, { 'app.html': '<iframe src="/assets/"></iframe><link href="assets/main-1.css">' });
    expect(checkPage(dest, FLOWS).size).toBeGreaterThan(0);
  });

  it('has no symbolic links', () => {
    writeBuild(dest, FLOWS);
    try {
      fs.symlinkSync(path.join(dest, 'app.html'), path.join(dest, 'link.html'));
    } catch {
      return; // no permission to make one here (Windows without developer mode)
    }
    expect(() => readTree(dest)).toThrow(/link\.html is a symbolic link/);
  });
});

describe('staging a page', () => {
  it('copies the build, and the copy is verified against it', () => {
    const source = path.join(root, 'app-dist');
    writeBuild(source, AGENT, { 'assets/deep/nested/font.woff2': 'font' });
    const result = stagePage(source, dest, AGENT);
    expect(result.files).toBe(readTree(source).size);
    expect(readTree(dest)).toEqual(readTree(source));
    expect(verifyStaged(dest, AGENT, source)).toEqual(result);
  });

  it('replaces what was staged before, so nothing of an earlier build is left', () => {
    const source = path.join(root, 'ui-dist');
    writeBuild(source, FLOWS);
    writeBuild(dest, FLOWS, { 'assets/old-hash.js': 'from the last build' });
    stagePage(source, dest, FLOWS);
    expect(fs.existsSync(path.join(dest, 'assets', 'old-hash.js'))).toBe(false);
    expect(readTree(dest)).toEqual(readTree(source));
  });

  it('leaves what was staged before alone when the build is not a page', () => {
    const source = path.join(root, 'ui-dist');
    writeBuild(dest, FLOWS);
    expect(() => stagePage(source, dest, FLOWS)).toThrow(/the build of the flow editor is not at/);
    expect(fs.existsSync(path.join(dest, 'app.html'))).toBe(true);
  });

  it('is verified as a page by itself, and against its source', () => {
    const source = path.join(root, 'ui-dist');
    writeBuild(source, FLOWS);
    stagePage(source, dest, FLOWS);

    fs.appendFileSync(path.join(dest, 'assets', 'main-1.js'), '// tampered\n');
    expect(() => verifyStaged(dest, FLOWS, source)).toThrow(/staged flows\/assets\/main-1\.js differs from its source .*main-1\.js \(sha256 [0-9a-f]{64}, not [0-9a-f]{64}\)/);
    expect(verifyStaged(dest, FLOWS)).toBeTruthy(); // still a page by itself; only its source says it is not the build

    fs.copyFileSync(path.join(source, 'assets', 'main-1.js'), path.join(dest, 'assets', 'main-1.js'));
    fs.writeFileSync(path.join(dest, 'assets', 'extra.js'), 'not from the build');
    expect(() => verifyStaged(dest, FLOWS, source)).toThrow(/staged flows\/assets\/extra\.js is not in its source/);
    fs.rmSync(path.join(dest, 'assets', 'extra.js'));

    fs.rmSync(path.join(dest, 'assets', 'main-1.css'));
    expect(() => verifyStaged(dest, FLOWS, source)).toThrow(/app\.html in .* loads \/assets\/main-1\.css, which is not there: run "npm run stage-pages/);

    fs.rmSync(path.join(dest, 'app.html'));
    expect(() => verifyStaged(dest, FLOWS, source)).toThrow(/the staged copy of the flow editor at .* has no app\.html: run "npm run stage-pages/);
    expect(STAGE).toBe('npm run stage-pages (in platform/desktop/)');
  });

  it('names a file missing from the copy though its source has it', () => {
    const source = path.join(root, 'ui-dist');
    writeBuild(source, FLOWS, { 'assets/only-in-source.js': 'x' });
    stagePage(source, dest, FLOWS);
    fs.rmSync(path.join(dest, 'assets', 'only-in-source.js'));
    expect(() => verifyStaged(dest, FLOWS, source)).toThrow(/staged flows\/assets\/only-in-source\.js is missing, though .* exists/);
  });
});

describe('staging both pages', () => {
  it('stages both from the repository and says what it did', () => {
    const { repo, desktop: dir } = writeRepository(path.join(root, 'repo'));
    const lines = [];
    const staged = stagePages({ repo, desktop: dir, log: (line) => lines.push(line) });
    expect(staged.map((s) => s.page.folder)).toEqual(['app', 'flows']);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatch(/^staged the Agent \(app\/dist\) into platform\/desktop\/src-tauri\/resources\/app: \d+ files, [\d.]+ MB, verified$/);
    expect(lines[1]).toMatch(/^staged the flow editor \(platform\/ui\/dist\) into platform\/desktop\/src-tauri\/resources\/flows: \d+ files, [\d.]+ MB, verified$/);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'app', 'index.html'))).toBe(true);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'flows', 'app.html'))).toBe(true);
  });

  it('stages all of them or none: a page that cannot be staged takes the other with it', () => {
    const { repo, desktop: dir } = writeRepository(path.join(root, 'repo'));
    stagePages({ repo, desktop: dir });
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'app', 'index.html'))).toBe(true);
    // The flow editor's build goes; so does the Agent's staged copy, which would otherwise be bundled without its partner.
    fs.rmSync(path.join(repo, 'platform', 'ui', 'dist'), { recursive: true });
    expect(() => stagePages({ repo, desktop: dir })).toThrow(/the build of the flow editor is not at/);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'app'))).toBe(false);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'flows'))).toBe(false);
  });

  it('stages nothing of a page whose start document loads a file its build does not have', () => {
    const { repo, desktop: dir } = writeRepository(path.join(root, 'repo'));
    fs.writeFileSync(path.join(repo, 'app', 'dist', 'index.html'), '<script src="/missing.js"></script>');
    expect(() => stagePages({ repo, desktop: dir })).toThrow(/index\.html in .* loads \/missing\.js, which is not there/);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'app'))).toBe(false);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'flows'))).toBe(false);
  });
});

describe('the script', () => {
  /** The real scripts, in a repository shaped like this one, so they find their own folders as they do here. */
  function scriptIn(base) {
    const { repo, desktop: dir } = writeRepository(base);
    const scripts = path.join(dir, 'scripts');
    fs.mkdirSync(scripts, { recursive: true });
    for (const name of ['stage-pages.mjs', 'stage-pages-lib.mjs']) fs.copyFileSync(path.join(here, name), path.join(scripts, name));
    return { repo, dir, script: path.join(scripts, 'stage-pages.mjs') };
  }
  const run = (script) => spawnSync(process.execPath, [script], { encoding: 'utf8', cwd: os.tmpdir() });

  it('stages both pages, from wherever it is run, and prints what it did', () => {
    const { dir, script } = scriptIn(path.join(root, 'repo'));
    const result = run(script);
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).toMatch(/stage-pages: staged the Agent \(app\/dist\) into platform\/desktop\/src-tauri\/resources\/app: \d+ files/);
    expect(result.stdout).toMatch(/stage-pages: staged the flow editor \(platform\/ui\/dist\) into platform\/desktop\/src-tauri\/resources\/flows: \d+ files/);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'app', 'index.html'))).toBe(true);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'flows', 'app.html'))).toBe(true);
  });

  it('fails the build, naming what to run, when a page was not built', () => {
    const { repo, dir, script } = scriptIn(path.join(root, 'repo'));
    fs.rmSync(path.join(repo, 'app', 'dist'), { recursive: true });
    const result = run(script);
    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/^stage-pages: the build of the Agent is not at .*: run "npm run build:desktop \(in app\/\)" first/);
    expect(fs.existsSync(path.join(dir, 'src-tauri', 'resources', 'app'))).toBe(false);
  });

  it('fails the build when a page was built empty', () => {
    const { repo, script } = scriptIn(path.join(root, 'repo'));
    fs.rmSync(path.join(repo, 'platform', 'ui', 'dist'), { recursive: true });
    fs.mkdirSync(path.join(repo, 'platform', 'ui', 'dist'));
    const result = run(script);
    expect(result.status).toBe(1);
    expect(result.stderr).toMatch(/stage-pages: the build of the flow editor at .* is empty: run "npm run build \(in platform\/ui\/\)"/);
  });
});
