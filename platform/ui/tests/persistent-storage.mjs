/**
 * The editor asks the browser to keep its storage.
 *
 *     npm run test:persistent-storage
 *
 * `requestPersistentStorage` (lib/persistentStorage.ts) is handed a stand-in for
 * `navigator`, so no global is replaced:
 *   - a browser without the API, a page it is not offered to, a refusal, a
 *     rejection and a throw all answer false, and none of them fails;
 *   - storage that is already persistent is not asked again;
 *   - the browser is asked once per navigator, however often the page asks;
 *   - the call does not wait for the answer;
 *   - main.tsx makes the call when the editor starts, without waiting for it.
 *
 * The module is TypeScript, so it is bundled for Node with esbuild (as
 * in-oaiy.mjs does).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as esbuild from 'esbuild';

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

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

const bundlePath = path.join(os.tmpdir(), `oaiy-persistent-storage-${process.pid}.mjs`);
await esbuild.build({
  entryPoints: [path.join(UI, 'src/lib/persistentStorage.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
});
const { requestPersistentStorage } = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });

// A rejection nobody handled would be a failure the page never sees coming.
const unhandled = [];
process.on('unhandledRejection', (reason) => unhandled.push(reason));
const tick = () => new Promise((resolve) => setTimeout(resolve, 5));

/** A navigator whose storage records its calls and answers as told. */
function navigatorWith({ persist, persisted } = {}) {
  const calls = { persist: 0, persisted: 0 };
  const storage = {};
  if (persist) storage.persist = () => { calls.persist++; return persist(); };
  if (persisted) storage.persisted = () => { calls.persisted++; return persisted(); };
  return { nav: { storage }, calls };
}

await check('a browser with no navigator, no storage or no persist() answers false', async () => {
  assert.equal(await requestPersistentStorage(null), false);
  assert.equal(await requestPersistentStorage({}), false);
  assert.equal(await requestPersistentStorage({ storage: {} }), false);
  assert.equal(await requestPersistentStorage({ storage: { persisted: async () => true } }), false, 'persisted() alone is not the API');
});

await check("Node's own navigator (no storage manager) answers false through the default argument", async () => {
  assert.equal(await requestPersistentStorage(), false);
});

await check('persist() answering true or false is passed on, and it is asked once', async () => {
  const yes = navigatorWith({ persist: async () => true });
  assert.equal(await requestPersistentStorage(yes.nav), true);
  assert.equal(yes.calls.persist, 1);
  const no = navigatorWith({ persist: async () => false });
  assert.equal(await requestPersistentStorage(no.nav), false);
  assert.equal(no.calls.persist, 1);
  // Only a real `true` counts.
  assert.equal(await requestPersistentStorage(navigatorWith({ persist: async () => 'yes' }).nav), false);
});

await check('a rejection, a throw and a storage getter that throws all answer false without failing', async () => {
  const rejects = navigatorWith({ persist: () => Promise.reject(new Error('denied')) });
  assert.equal(await requestPersistentStorage(rejects.nav), false);
  const throws = navigatorWith({ persist: () => { throw new Error('not allowed here'); } });
  assert.equal(await requestPersistentStorage(throws.nav), false);
  const hostile = { get storage() { throw new DOMException('blocked', 'SecurityError'); } };
  assert.equal(await requestPersistentStorage(hostile), false);
  await tick();
  assert.deepEqual(unhandled, [], 'nothing was left unhandled');
});

await check('storage that is already persistent is not asked again', async () => {
  const already = navigatorWith({ persisted: async () => true, persist: async () => true });
  assert.equal(await requestPersistentStorage(already.nav), true);
  assert.equal(already.calls.persisted, 1);
  assert.equal(already.calls.persist, 0);
  // Not yet persistent: it is asked.
  const not = navigatorWith({ persisted: async () => false, persist: async () => true });
  assert.equal(await requestPersistentStorage(not.nav), true);
  assert.equal(not.calls.persist, 1);
  // persisted() failing does not stop it asking.
  const broken = navigatorWith({ persisted: () => Promise.reject(new Error('no')), persist: async () => true });
  assert.equal(await requestPersistentStorage(broken.nav), true);
  assert.equal(broken.calls.persist, 1);
});

await check('asked twice, the browser is asked once and both get the same answer', async () => {
  const one = navigatorWith({ persist: async () => true });
  const first = requestPersistentStorage(one.nav);
  const second = requestPersistentStorage(one.nav);
  assert.equal(first, second, 'the same promise');
  assert.equal(await first, true);
  assert.equal(await requestPersistentStorage(one.nav), true, 'and later');
  assert.equal(one.calls.persist, 1);
  // Another navigator is its own question.
  const other = navigatorWith({ persist: async () => false });
  assert.equal(await requestPersistentStorage(other.nav), false);
  assert.equal(other.calls.persist, 1);
});

await check('it does not wait for the answer', async () => {
  const slow = navigatorWith({ persist: () => new Promise(() => {}) });
  const started = Date.now();
  const answer = requestPersistentStorage(slow.nav);
  assert.equal(typeof answer.then, 'function');
  assert.ok(Date.now() - started < 50, 'came back at once');
  assert.equal(slow.calls.persist, 1, 'and had already asked');
  let settled = false;
  answer.then(() => { settled = true; });
  await tick();
  assert.equal(settled, false, 'still waiting on the browser');
});

await check('main.tsx asks when the editor starts, and does not wait for it', () => {
  const main = fs.readFileSync(path.join(UI, 'src/main.tsx'), 'utf8');
  assert.match(main, /^import \{ requestPersistentStorage \} from '\.\/lib\/persistentStorage';$/m);
  assert.match(main, /^void requestPersistentStorage\(\);$/m, 'called once, at the top level, not awaited');
  assert.equal((main.match(/requestPersistentStorage\(/g) ?? []).length, 1, 'once');
  // The landing pages have their own entries and never ask.
  for (const entry of ['main.tsx', 'desktop-main.tsx']) {
    assert.doesNotMatch(fs.readFileSync(path.join(UI, 'src/landing', entry), 'utf8'), /persistentStorage/, `landing/${entry}`);
  }
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
