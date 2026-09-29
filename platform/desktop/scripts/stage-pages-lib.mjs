/**
 * What `stage-pages.mjs` stages and how it is checked — as functions, so a test
 * can drive them against a fixture without wiping the real `resources/app` and
 * `resources/flows`.
 *
 * OAIY Desktop shows two web pages in its window and serves them itself
 * (`src-tauri/src/embed.rs`): the Agent (`app/`) and the flow editor
 * (`platform/ui`). Their builds are not in the repository, and embed.rs only
 * falls back to the repository's own build folders, so an installer that does not
 * carry them opens on "the page is not built" everywhere but the machine that
 * built it. The builds are copied into `src-tauri/resources/{app,flows}`, which
 * `tauri.conf.json` lists in `bundle.resources`, and the copy is verified: a page
 * that is missing, empty, incomplete or different from its source is a failed
 * build, not an installer that fails on the user's machine.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

export class StagingError extends Error {}

/**
 * The pages, as embed.rs names them.
 *
 *   folder    the page's folder among the bundle's resources: `Page::folder()`.
 *             tauri.conf.json lists `resources/<folder>`, and Tauri keeps that
 *             path, so the installed page is `<resource dir>/resources/<folder>`.
 *   start     the document the page opens on: `Page::start()`. A folder without
 *             it is not a page (embed.rs skips it, then falls back).
 *   source    the page's build folder, from the repository's root.
 *   build     how to make it.
 *   required  what the page cannot work without besides its start document and
 *             what that loads: the Agent's ZIPP engine (its code sandbox) and the
 *             SoftN runtime its app preview needs, which only `build:desktop`
 *             fetches.
 */
export const PAGES = [
  {
    title: 'the Agent',
    folder: 'app',
    start: 'index.html',
    source: 'app/dist',
    build: 'npm run build:desktop (in app/)',
    required: ['zipp/zipp_wasm.js', 'zipp/zipp_wasm_bg.wasm', 'softn/index.html'],
  },
  {
    title: 'the flow editor',
    folder: 'flows',
    start: 'app.html',
    source: 'platform/ui/dist',
    build: 'npm run build (in platform/ui/)',
    required: [],
  },
];

/** Where a page is staged, from the desktop folder (`platform/desktop`). */
export const stagedPath = (desktop, page) => path.join(desktop, 'src-tauri', 'resources', page.folder);

const posix = (p) => p.split(path.sep).join('/');

/**
 * Every file under `dir` as `{ 'assets/x.js': { size, sha256 } }`, keyed by its
 * path from `dir` with forward slashes. A symbolic link is refused: a page's
 * files are copied as they are, and nothing here should be one.
 */
export function readTree(dir) {
  const tree = new Map();
  const walk = (relative) => {
    const here = relative ? path.join(dir, relative) : dir;
    for (const entry of fs.readdirSync(here, { withFileTypes: true })) {
      const child = relative ? `${relative}/${entry.name}` : entry.name;
      const full = path.join(dir, child);
      if (entry.isSymbolicLink()) throw new StagingError(`${full} is a symbolic link; a page's files are copied as they are, not followed`);
      if (entry.isDirectory()) walk(child);
      else if (entry.isFile()) {
        const bytes = fs.readFileSync(full);
        tree.set(child, { size: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
      }
    }
  };
  walk('');
  return tree;
}

/**
 * The local files an HTML document loads: each `src` and `href` that is not an
 * address of another origin, a data URI or an anchor, without its query or
 * fragment. Comments and inline script and style bodies are left out, so what
 * they say about `src=` is not read as a reference.
 */
export function localReferences(html) {
  const text = html
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, (element) => element.slice(0, element.indexOf('>') + 1));
  const references = new Set();
  for (const match of text.matchAll(/\b(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')/gi)) {
    const reference = (match[1] ?? match[2]).trim();
    if (!reference || /^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(reference)) continue;
    const bare = reference.split(/[?#]/)[0];
    if (!bare) continue;
    try {
      references.add(decodeURIComponent(bare));
    } catch {
      references.add(bare);
    }
  }
  return [...references];
}

/** What to run when a staged copy is not a page. */
export const STAGE = 'npm run stage-pages (in platform/desktop/)';

/**
 * A build folder, or a staged one, is a page: it exists, holds files, holds the
 * start document and what the page cannot work without, and the start document
 * loads nothing that is not there. Returns its files (see `readTree`) or throws
 * a `StagingError` that says what to do. `staged` says which of the two `dir` is.
 */
export function checkPage(dir, page, staged = false) {
  const what = `${staged ? 'staged copy' : 'build'} of ${page.title}`;
  const fix = `run "${staged ? STAGE : page.build}"`;
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    throw new StagingError(`the ${what} is not at ${dir}: ${fix} first`);
  }
  const tree = readTree(dir);
  if (tree.size === 0) throw new StagingError(`the ${what} at ${dir} is empty: ${fix}`);
  for (const rel of [page.start, ...page.required]) {
    if (!tree.has(rel)) throw new StagingError(`the ${what} at ${dir} has no ${rel}: ${fix}`);
  }
  const folders = new Set();
  for (const rel of tree.keys()) {
    for (let at = rel.lastIndexOf('/'); at > 0; at = rel.lastIndexOf('/', at - 1)) folders.add(rel.slice(0, at));
  }
  const html = fs.readFileSync(path.join(dir, page.start), 'utf8');
  for (const reference of localReferences(html)) {
    const rel = path.posix.normalize(reference.startsWith('/') ? reference.slice(1) : path.posix.join(path.posix.dirname(page.start), reference)).replace(/\/$/, '');
    if (rel === '' || rel === '.') continue;
    if (rel.startsWith('..')) throw new StagingError(`${page.start} in ${dir} loads ${reference}, which is outside the page`);
    if (!tree.has(rel) && !folders.has(rel)) {
      throw new StagingError(`${page.start} in ${dir} loads ${reference}, which is not there: ${fix}`);
    }
  }
  return tree;
}

const total = (tree) => [...tree.values()].reduce((sum, file) => sum + file.size, 0);

/**
 * Check a staged folder: it is a page by itself, and, given its source, exactly
 * its source — every file present and byte-equal, and nothing else. Returns
 * `{ files, bytes }`.
 */
export function verifyStaged(dest, page, source = null) {
  const staged = checkPage(dest, page, true);
  if (source) {
    const built = readTree(source);
    for (const [rel, file] of built) {
      const copy = staged.get(rel);
      if (!copy) throw new StagingError(`staged ${page.folder}/${rel} is missing, though ${path.join(source, rel)} exists`);
      if (copy.sha256 !== file.sha256) {
        throw new StagingError(`staged ${page.folder}/${rel} differs from its source ${path.join(source, rel)} (sha256 ${copy.sha256}, not ${file.sha256})`);
      }
    }
    for (const rel of staged.keys()) {
      if (!built.has(rel)) throw new StagingError(`staged ${page.folder}/${rel} is not in its source ${source}`);
    }
  }
  return { files: staged.size, bytes: total(staged) };
}

/**
 * Copy one page's build to where the bundle takes it from: the build is checked
 * first, what was staged before is removed, and the copy is verified against the
 * build. Returns `{ files, bytes }`.
 */
export function stagePage(source, dest, page) {
  checkPage(source, page);
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(source, dest, { recursive: true });
  return verifyStaged(dest, page, source);
}

/**
 * Stage every page from the repository's build folders. All of them or none:
 * when one cannot be staged, nothing stays staged, so a later step never
 * bundles a copy that no longer matches its source or a page missing its
 * partner. `repo` is the repository's root, `desktop` is `platform/desktop`.
 */
export function stagePages({ repo, desktop, log = () => {} }) {
  const staged = [];
  try {
    for (const page of PAGES) {
      const source = path.join(repo, ...page.source.split('/'));
      const dest = stagedPath(desktop, page);
      const { files, bytes } = stagePage(source, dest, page);
      log(`staged ${page.title} (${posix(path.relative(repo, source))}) into ${posix(path.relative(repo, dest))}: ${files} files, ${(bytes / 1024 / 1024).toFixed(1)} MB, verified`);
      staged.push({ page, files, bytes });
    }
  } catch (error) {
    for (const page of PAGES) fs.rmSync(stagedPath(desktop, page), { recursive: true, force: true });
    throw error;
  }
  return staged;
}
