/**
 * The site's links into this repository, and the desktop page's service library.
 *
 *     npm run test:landing
 *
 * - Links. The site is `platform/ui` of a repository that also holds the Agent,
 *   the crates and the docs, so a folder a page links to is a path from the
 *   repository root. The desktop page linked to `desktop` and
 *   `api/service-library` (their old places, before the repositories were
 *   merged) and both went to nothing. Each folder in landing/repoLinks.ts must
 *   be in the repository (tracked by git, or on disk where there is no git), the
 *   URLs must be GitHub tree links on the repository's branch, and no page may
 *   write such a link by hand where this cannot see it.
 * - Library. The desktop page's library comes from `/api/service-library`,
 *   which the standalone release build does not serve. A 404 (or a page in
 *   place of JSON) is `unavailable`, with no retry to offer; no answer, a
 *   server error or a reply that is not a library is `error`, which a retry can
 *   fix; a real library is `ok`.
 *
 * The modules are TypeScript, so they are bundled for Node with esbuild (as
 * in-oaiy.mjs does). The browser side of these states is tests/e2e.mjs.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = path.resolve(UI, '..', '..'); // the repository root: ui -> platform -> repo

let pass = 0;
const failures = [];
async function check(name, fn) {
  try {
    await fn();
    pass++;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failures.push(`${name} — ${e.message}`);
    console.log(`  FAIL ${name} — ${e.stack || e.message}`);
  }
}

const bundlePath = path.join(os.tmpdir(), `oaiy-landing-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export * from './src/landing/repoLinks.ts';
export * from './src/landing/serviceLibrary.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
});
const L = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

// ---------------------------------------------------------------------------
// Links into the repository
// ---------------------------------------------------------------------------

/** Whether `rel` (from the repository root) is a folder the repository has. */
function inRepository(rel) {
  const listed = spawnSync('git', ['ls-files', '--', rel], { cwd: ROOT, encoding: 'utf8' });
  if (!listed.error && listed.status === 0) return listed.stdout.trim().length > 0;
  // No git here (a source archive): the folder on disk is what there is to check.
  return fs.statSync(path.join(ROOT, rel), { throwIfNoEntry: false })?.isDirectory() === true;
}

await check('every folder the pages link to is in the repository, from its root', () => {
  const folders = Object.entries(L.REPO_FOLDERS);
  assert.ok(folders.length >= 2, 'the desktop page links to two');
  for (const [name, rel] of folders) {
    assert.ok(!rel.startsWith('/') && !rel.includes('\\') && !rel.endsWith('/'), `${name}: ${rel} is a path from the root`);
    assert.ok(inRepository(rel), `${name}: "${rel}" is not in the repository`);
  }
  // What the buttons promise is what GitHub shows under the folder.
  assert.ok(inRepository(`${L.REPO_FOLDERS.desktop}/README.md`), "the desktop's installation documentation is its README");
  assert.ok(inRepository(`${L.REPO_FOLDERS.serviceLibrary}/README.md`));
});

await check("the folders that moved are not linked at their old places", () => {
  for (const name of Object.keys(L.REPO_FOLDERS)) {
    const url = L.repoFolderUrl(name);
    assert.doesNotMatch(url, /\/tree\/main\/(desktop|api)\b/, `${name}: ${url}`);
  }
  assert.equal(L.repoFolderUrl('desktop'), 'https://github.com/f2i-com/oaiy.com/tree/main/platform/desktop');
  assert.equal(L.repoFolderUrl('serviceLibrary'), 'https://github.com/f2i-com/oaiy.com/tree/main/platform/api/service-library');
  assert.equal(L.RELEASES_URL, 'https://github.com/f2i-com/oaiy.com/releases/latest');
});

await check('no page writes a link into the repository by hand: they all come from repoLinks.ts', () => {
  const files = fs.readdirSync(path.join(UI, 'src/landing'))
    .filter((f) => /\.(tsx?|css)$/.test(f) && f !== 'repoLinks.ts')
    .map((f) => path.join(UI, 'src/landing', f));
  files.push(...['index.html', 'desktop.html', 'app.html'].map((f) => path.join(UI, f)));
  assert.ok(files.length >= 6);
  for (const file of files) {
    const text = fs.readFileSync(file, 'utf8');
    const hand = text.match(/f2i-com\/oaiy\.com|\/(?:tree|blob|raw)\/(?:main|master)\//g);
    assert.equal(hand, null, `${path.relative(UI, file)} links into the repository by hand (${hand}); add the folder to REPO_FOLDERS in landing/repoLinks.ts instead`);
  }
});

await check('the desktop page takes its links from repoLinks.ts', () => {
  const page = fs.readFileSync(path.join(UI, 'src/landing/DesktopPage.tsx'), 'utf8');
  assert.match(page, /repoFolderUrl\('desktop'\)/);
  assert.match(page, /repoFolderUrl\('serviceLibrary'\)/);
  assert.match(page, /RELEASES_URL/);
});

// ---------------------------------------------------------------------------
// The service library
// ---------------------------------------------------------------------------

const TEMPLATE = { file: 'ollama.json', name: 'Ollama', description: 'Local language models.', category: 'LLM', icon: 'x', count: 1, size: 812, downloadUrl: '/api/service-library/ollama.json' };
const json = (body, init = {}) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' }, ...init });
const answering = (reply) => async () => (typeof reply === 'function' ? reply() : reply);

await check('a library the API serves is listed', async () => {
  const seen = [];
  const controller = new AbortController();
  const result = await L.loadServiceLibrary('http://127.0.0.1:8099', controller.signal, async (url, init) => {
    seen.push({ url, init });
    return json({ services: [TEMPLATE, { file: 'a.json', name: 'A', description: '', downloadUrl: '/api/service-library/a.json' }] });
  });
  assert.equal(result.state, 'ok');
  assert.deepEqual(result.items.map((i) => i.file), ['ollama.json', 'a.json']);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].url, 'http://127.0.0.1:8099/api/service-library');
  assert.equal(seen[0].init.headers.Accept, 'application/json');
  assert.equal(seen[0].init.signal, controller.signal, 'the page can give up on it');
  // On the site's own origin the base is empty.
  const own = [];
  const empty = await L.loadServiceLibrary('', undefined, async (url) => { own.push(url); return json({ services: [] }); });
  assert.deepEqual(own, ['/api/service-library']);
  assert.deepEqual(empty, { state: 'ok', items: [] });
});

await check('no /api here (a 404, or a page in place of JSON) is unavailable, not something to retry', async () => {
  for (const status of [404, 405, 410, 501]) {
    assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(new Response('not here', { status }))), { state: 'unavailable' }, String(status));
  }
  // A host that answers every path with its front page.
  const front = new Response('<!doctype html><title>OAIY</title>', { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } });
  assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(front)), { state: 'unavailable' });
  const plain = new Response('Not Found', { status: 200, headers: { 'content-type': 'text/plain' } });
  assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(plain)), { state: 'unavailable' });
});

await check('no answer, a server error, or a reply that is not a library is an error a retry can fix', async () => {
  const network = () => { throw new TypeError('Failed to fetch'); };
  assert.deepEqual(await L.loadServiceLibrary('', undefined, network), { state: 'error' });
  const timedOut = () => { throw new DOMException('This operation was aborted', 'AbortError'); };
  assert.deepEqual(await L.loadServiceLibrary('', undefined, timedOut), { state: 'error' });
  for (const status of [500, 502, 503, 401, 403, 429]) {
    assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(new Response('x', { status }))), { state: 'error' }, String(status));
  }
  const corrupt = new Response('{"services": [', { status: 200, headers: { 'content-type': 'application/json' } });
  assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(corrupt)), { state: 'error' }, 'JSON that stops short');
  const cutOff = { status: 200, ok: true, headers: new Headers({ 'content-type': 'application/json' }), json: () => Promise.reject(new DOMException('aborted', 'AbortError')) };
  assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(cutOff)), { state: 'error' }, 'the body cut off by the timeout');
});

await check('JSON that is not a library of the API\'s own downloads is an error', async () => {
  const wrong = [
    null,
    [],
    {},
    { services: 'ollama' },
    { services: [null] },
    { services: [{ ...TEMPLATE, name: 3 }] },
    { services: [{ ...TEMPLATE, category: 4 }] },
    { services: [{ ...TEMPLATE, downloadUrl: 'https://example.invalid/ollama.json' }] },
    { services: [{ ...TEMPLATE, downloadUrl: '/other/ollama.json' }] },
    { services: [{ ...TEMPLATE, downloadUrl: '/api/service-library\\..\\x.json' }] },
    { services: [TEMPLATE, { file: 'b.json' }] },
  ];
  for (const body of wrong) {
    assert.deepEqual(await L.loadServiceLibrary('', undefined, answering(json(body))), { state: 'error' }, JSON.stringify(body));
  }
  // A template with no category is fine.
  const { category, ...noCategory } = TEMPLATE;
  assert.equal((await L.loadServiceLibrary('', undefined, answering(json({ services: [noCategory] })))).state, 'ok');
});

await check('the desktop page shows each state, and offers a retry only where one can help', () => {
  const page = fs.readFileSync(path.join(UI, 'src/landing/DesktopPage.tsx'), 'utf8');
  assert.match(page, /loadServiceLibrary\(API_BASE, controller\.signal\)/);
  const unavailable = page.slice(page.indexOf("state === 'unavailable'"), page.indexOf("state === 'error'"));
  assert.match(unavailable, /does not serve the service library/);
  assert.match(unavailable, /repoFolderUrl\('serviceLibrary'\)/);
  assert.doesNotMatch(unavailable, /Try again|setAttempt/, 'no retry where there is no library');
  const error = page.slice(page.indexOf("state === 'error'"), page.indexOf("state === 'ok' && items.length === 0"));
  assert.match(error, /Try again/);
  assert.match(error, /setAttempt/);
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
