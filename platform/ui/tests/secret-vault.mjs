/**
 * The flow editor's API keys are sealed in IndexedDB, not left in localStorage.
 *
 *     npm run test:secret-vault
 *
 * The web build kept a flow's API keys in plain text in localStorage
 * (`oaiy_web_secrets`). tauri-shim/secretVault.ts seals them the way the Agent
 * seals its provider keys: AES-GCM under a non-extractable key kept in
 * IndexedDB. That keeps them out of localStorage and out of a script's reach to
 * export; it does not protect a copy of the browser profile, whose files hold the
 * key beside the ciphertext (the header of secretVault.ts says what it does and
 * does not do, and nothing here claims more).
 *
 * Node has WebCrypto but no IndexedDB, so the vault is handed a small IndexedDB
 * of this file's own (serial transactions, structured clones, and the failures a
 * browser gives: a refused write, a database that will not open or never
 * answers, a store wiped under a live page). What is checked:
 *
 *   - a seal and unseal round trip, across a page load; the stored bytes are
 *     ciphertext, one record per secret, under a key that cannot be exported;
 *   - the move from plaintext: every value sealed, read back and opened before
 *     the plaintext is removed, and the plaintext wins where both name a secret;
 *   - sealing that cannot be done (no IndexedDB, no WebCrypto, the database
 *     refusing to open, a refused write, a read back that does not match, a store
 *     that never answers, a plaintext that cannot be removed) loses no key: the
 *     plaintext stays, reads and writes go on through it, and it is said once;
 *   - a wiped key store, a wiped database, a damaged record: nothing throws;
 *   - overlapping saves keep every key; two tabs starting together share one key;
 *   - the shim's `get_secrets` / `store_secret` (core.ts) are the vault's, and
 *     the editor starts it.
 *
 * The real IndexedDB is a browser's: see the browser check in the commit that
 * added this (Chromium against `vite preview`), which is not part of `npm test`.
 * The modules are TypeScript with aliases only the bundler resolves, so they are
 * bundled for Node with esbuild (as flow-services.mjs does).
 */
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
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
const tick = (ms = 5) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// A stand-in IndexedDB. Transactions run one at a time, in order (stricter than
// a browser's, which is all the vault needs); values are structured clones;
// events arrive in later tasks, as a browser's do.
// ---------------------------------------------------------------------------
function createFakeIDB() {
  const dbs = new Map();
  const knobs = { hangOpen: false, openError: null, hung: [], failures: [], corruptReads: false, delay: {}, holds: [], held: [] };
  const puts = {};
  const domError = (name, message) => new DOMException(message, name);

  /** A failure made ahead of time: `op` on `store`, after `skip` matches, `times` times. */
  function injected(op, store) {
    for (const f of knobs.failures) {
      if (f.op !== op || (f.store && f.store !== store)) continue;
      if (f.skip > 0) { f.skip--; continue; }
      if (f.times <= 0) continue;
      f.times--;
      return f.error;
    }
    return null;
  }

  /** A request made to hang ahead of time, the same way: it is not carried out, and its transaction does not finish, until `release()`. */
  function holding(op, store) {
    for (const h of knobs.holds) {
      if (h.op !== op || (h.store && h.store !== store)) continue;
      if (h.skip > 0) { h.skip--; continue; }
      if (h.times <= 0) continue;
      h.times--;
      return true;
    }
    return false;
  }

  function makeDb() {
    const db = { stores: new Map(), conns: new Set(), queue: [], active: null };
    db.pump = () => {
      if (db.active || db.queue.length === 0) return;
      db.active = db.queue.shift();
      db.active.begin();
    };
    return db;
  }

  class Tx {
    constructor(conn, names, mode) {
      this.conn = conn; this.db = conn.db; this.names = names; this.mode = mode;
      this.requests = []; this.state = 'waiting'; this.waiting = false; this.error = null;
      this.oncomplete = null; this.onerror = null; this.onabort = null;
      this.db.queue.push(this);
      queueMicrotask(() => this.db.pump());
    }
    objectStore(name) {
      if (!this.names.includes(name)) throw domError('NotFoundError', `store ${name} is not in this transaction`);
      const tx = this;
      const add = (op, args, exec) => {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null, op, args, exec };
        if (tx.state === 'finished') throw domError('TransactionInactiveError', 'The transaction has finished.');
        tx.requests.push(req);
        if (tx.state === 'active' && !tx.busy) tx.run();
        return req;
      };
      const data = () => this.db.stores.get(name);
      const seen = (v) => {
        const c = structuredClone(v);
        if (knobs.corruptReads && name === 'secrets' && c && c.data) new Uint8Array(c.data).fill(7);
        return c;
      };
      return {
        get: (key) => add('get', [key], () => seen(data().get(key))),
        getAll: () => add('getAll', [], () => [...data().keys()].sort().map((k) => seen(data().get(k)))),
        getAllKeys: () => add('getAllKeys', [], () => [...data().keys()].sort()),
        put: (value, key) => add('put', [key], () => { data().set(key, structuredClone(value)); puts[name] = (puts[name] ?? 0) + 1; return key; }),
        delete: (key) => add('delete', [key], () => { data().delete(key); }),
        _store: name,
      };
    }
    begin() {
      this.state = 'active';
      setImmediate(() => this.run());
    }
    run() {
      if (this.state !== 'active' || this.busy) return;
      const req = this.requests.shift();
      if (!req) {
        if (this.waiting) return;
        this.waiting = true;
        setImmediate(() => {
          this.waiting = false;
          if (this.requests.length) return this.run();
          this.finish();
        });
        return;
      }
      this.busy = true;
      // A hung transaction keeps every later one on its database waiting, as a browser's does.
      if (holding(req.op, this.currentStore(req))) knobs.held.push(() => this.perform(req));
      else this.perform(req);
    }
    perform(req) {
      let result; let error = null;
      try {
        error = injected(req.op, this.currentStore(req));
        if (!error) result = req.exec();
      } catch (e) { error = e; }
      const settle = () => {
        this.busy = false;
        if (this.state !== 'active') return;
        if (error) {
          req.error = error;
          req.onerror?.({ target: req });
          return this.abort(error);
        }
        req.result = result;
        try { req.onsuccess?.({ target: req }); } catch (e) { return this.abort(e); }
        this.run();
      };
      // A store answers in a later task; only a slow one (knobs.delay) takes real time.
      if (knobs.delay[req.op]) setTimeout(settle, knobs.delay[req.op]);
      else setImmediate(settle);
    }
    currentStore(req) { return this.names.length === 1 ? this.names[0] : undefined; }
    finish() {
      this.state = 'finished';
      this.db.active = null;
      this.oncomplete?.({ target: this });
      this.db.pump();
    }
    abort(error) {
      this.error = error; this.state = 'finished';
      this.db.active = null;
      this.onabort?.({ target: this });
      this.db.pump();
    }
  }

  class Conn {
    constructor(db) {
      this.db = db; this.closed = false; this.onversionchange = null; this.onclose = null;
      this.objectStoreNames = { contains: (n) => db.stores.has(n) };
    }
    createObjectStore(name) { this.db.stores.set(name, new Map()); }
    transaction(names, mode) {
      if (this.closed) throw domError('InvalidStateError', 'The database connection is closing.');
      const list = Array.isArray(names) ? names : [names];
      for (const n of list) if (!this.db.stores.has(n)) throw domError('NotFoundError', `no store ${n}`);
      return new Tx(this, list, mode);
    }
    close() { this.closed = true; this.db.conns.delete(this); }
  }

  const factory = {
    open(name) {
      const req = { result: undefined, error: null, onsuccess: null, onerror: null, onupgradeneeded: null, onblocked: null };
      const go = () => {
        if (knobs.openError) { req.error = knobs.openError; return req.onerror?.({ target: req }); }
        let db = dbs.get(name); let fresh = false;
        if (!db) { db = makeDb(); dbs.set(name, db); fresh = true; }
        const conn = new Conn(db);
        db.conns.add(conn);
        req.result = conn;
        if (fresh) req.onupgradeneeded?.({ target: req });
        req.onsuccess?.({ target: req });
      };
      setImmediate(() => (knobs.hangOpen ? knobs.hung.push(go) : go()));
      return req;
    }
  };

  return {
    factory,
    knobs,
    /** How many records each store has been given, by name (`keys`, `secrets`). */
    puts,
    /** The map behind one store of one database (a stored key or record, by name). */
    raw: (store, dbName = 'oaiy-web-secrets') => dbs.get(dbName)?.stores.get(store),
    has: () => dbs.has('oaiy-web-secrets'),
    /** "Clear site data": every database gone, every connection closed and told so. */
    wipe() {
      for (const db of dbs.values()) for (const conn of [...db.conns]) { conn.closed = true; conn.onclose?.(); }
      dbs.clear();
    },
    /** Connections closed without a word (the vault learns only when it next uses one). */
    closeQuietly() { for (const db of dbs.values()) for (const conn of db.conns) conn.closed = true; },
    fail(op, error, { store, skip = 0, times = 1 } = {}) { knobs.failures.push({ op, store, skip, times, error }); },
    /** Make `op` on `store` hang (after `skip` of them) until `release()`. */
    hold(op, { store, skip = 0, times = 1 } = {}) { knobs.holds.push({ op, store, skip, times }); },
    /** The browser answers at last: whatever was hung goes through, and nothing more is held. */
    release() {
      knobs.hangOpen = false;
      knobs.holds.length = 0;
      for (const go of knobs.hung.splice(0)) go();
      for (const go of knobs.held.splice(0)) go();
    },
  };
}

function createStorage(initial = {}) {
  const m = new Map(Object.entries(initial));
  const s = {
    raw: m, failSet: false, failRemove: false, failGet: false,
    getItem: (k) => { if (s.failGet) throw new DOMException('blocked', 'SecurityError'); return m.has(k) ? m.get(k) : null; },
    setItem: (k, v) => { if (s.failSet) throw new DOMException('full', 'QuotaExceededError'); m.set(k, String(v)); },
    removeItem: (k) => { if (s.failRemove) throw new Error('cannot remove'); m.delete(k); },
  };
  return s;
}

// ---------------------------------------------------------------------------
// Bundle the vault, and the shim that uses it. The shim is loaded once, after the
// page's globals are set, because the vault it holds reads them when it is made.
// ---------------------------------------------------------------------------
const PLAIN = 'oaiy_web_secrets';
const shimWorld = { idb: createFakeIDB(), storage: createStorage() };
const preseeded = { OPENAI_API_KEY: 'sk-shim-old-1', ANTHROPIC_API_KEY: 'sk-shim-old-2' };
shimWorld.storage.setItem(PLAIN, JSON.stringify(preseeded));
globalThis.indexedDB = shimWorld.idb.factory;
globalThis.localStorage = shimWorld.storage;
globalThis.window = { setTimeout, clearTimeout, setInterval, clearInterval, addEventListener() {}, location: { protocol: 'https:', origin: 'https://oaiy.com' } };

const stub = path.join(os.tmpdir(), `oaiy-ui-stub-${process.pid}.mjs`);
fs.writeFileSync(stub, 'export function invalidateDynamicOptions() {}\nexport function subscribeToDynamicOptionsInvalidation() { return () => {}; }\n');
const bundlePath = path.join(os.tmpdir(), `oaiy-secret-vault-${process.pid}.mjs`);
await esbuild.build({
  stdin: {
    contents: `export { createSecretVault, secretVault, PLAINTEXT_SECRETS_KEY } from './src/tauri-shim/secretVault.ts';
export { invoke } from './src/tauri-shim/core.ts';`,
    resolveDir: UI,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundlePath,
  logLevel: 'silent',
  plugins: [{
    name: 'oaiy-aliases',
    setup(build) {
      build.onResolve({ filter: /^oaiy-ui-components$/ }, () => ({ path: stub }));
      // Loaded only when a flow runs ffmpeg, which none of this does; its imports are Vite's.
      build.onResolve({ filter: /^\.\/ffmpeg$/ }, () => ({ path: './ffmpeg', external: true }));
    },
  }],
});
const V = await import(pathToFileURL(bundlePath).href);
fs.rmSync(bundlePath, { force: true });
fs.rmSync(stub, { force: true });

/** A browser's storage, and a way to load the page again (a new vault on the same storage). */
function world({ plain, env: over = {} } = {}) {
  const idb = createFakeIDB();
  const storage = createStorage(plain === undefined ? {} : { [PLAIN]: typeof plain === 'string' ? plain : JSON.stringify(plain) });
  const warns = [];
  const details = [];
  const env = { indexedDB: idb.factory, crypto: webcrypto, storage, warn: (message, detail) => { warns.push(message); details.push(detail); }, timeoutMs: 1500, ...over };
  return { idb, storage, warns, details, env, load: () => V.createSecretVault(env) };
}
const KEY_A = 'sk-ant-api03-AAAA-secret-value-1';
const KEY_B = 'sk-proj-BBBB-secret-value-2';
const bytesOf = (v) => Buffer.from(v.data).toString('latin1') + Buffer.from(v.iv).toString('latin1');
const plainStored = (w) => w.storage.raw.get(PLAIN) ?? null;
/** What `promise` settles with, or 'HUNG' if it has not settled in `ms`: a hung call fails a test rather than hanging it. */
const orHung = (promise, ms = 3000) => {
  let timer;
  const hung = new Promise((resolve) => { timer = setTimeout(() => resolve('HUNG'), ms); });
  return Promise.race([promise, hung]).finally(() => clearTimeout(timer));
};

// ---------------------------------------------------------------------------
// Seal and unseal
// ---------------------------------------------------------------------------
await check('a secret sealed on one page load opens on the next, and none is left in the clear', async () => {
  const w = world();
  const one = w.load();
  assert.equal(await one.ready(), 'sealed');
  await one.set('OPENAI_API_KEY', KEY_A);
  await one.set('UNICODE', 'πάντα ῥεῖ 🔑 "quoted" \\ back');
  assert.deepEqual(await one.get(['OPENAI_API_KEY', 'UNICODE', 'NOT_SET']), { OPENAI_API_KEY: KEY_A, UNICODE: 'πάντα ῥεῖ 🔑 "quoted" \\ back' });
  // A reload: a new vault on the same storage.
  const two = w.load();
  assert.equal(await two.ready(), 'sealed');
  assert.deepEqual(await two.get(['OPENAI_API_KEY', 'UNICODE']), { OPENAI_API_KEY: KEY_A, UNICODE: 'πάντα ῥεῖ 🔑 "quoted" \\ back' });
  assert.equal(plainStored(w), null, 'nothing was ever written as plaintext');
  assert.deepEqual(w.warns, []);
});

await check('an empty value deletes a secret; unknown, empty and repeated names are harmless', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  await vault.set('B', KEY_B);
  await vault.set('A', '');
  assert.deepEqual(await vault.get(['A', 'B', 'B', '', 'C']), { B: KEY_B });
  assert.deepEqual(await vault.get([]), {});
  assert.deepEqual(await w.load().get(['A', 'B']), { B: KEY_B }, 'and so after a reload');
  await vault.set('', 'nameless'); // nothing to name: ignored
  assert.deepEqual([...w.idb.raw('secrets').keys()], ['B']);
  // A name that is also a property of every object is only a name.
  await vault.set('__proto__', 'proto-value');
  await vault.set('constructor', 'ctor-value');
  assert.deepEqual(await w.load().get(['__proto__', 'constructor', 'toString']), { ['__proto__']: 'proto-value', constructor: 'ctor-value' });
});

await check('the store holds ciphertext, one record per secret, under a key that cannot be exported', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('OPENAI_API_KEY', KEY_A);
  await vault.set('ANTHROPIC_API_KEY', KEY_B);
  const records = w.idb.raw('secrets');
  assert.deepEqual([...records.keys()].sort(), ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY']);
  for (const record of records.values()) {
    assert.equal(record.iv.byteLength, 12, 'an AES-GCM nonce');
    assert.ok(record.data.byteLength > KEY_A.length, 'ciphertext with its tag');
    assert.ok(!bytesOf(record).includes('secret-value'), 'not the value');
  }
  const [a, b] = [...records.values()];
  assert.notDeepEqual(Buffer.from(a.iv), Buffer.from(b.iv), 'a nonce each');
  const key = w.idb.raw('keys').get('secret-key');
  assert.equal(w.idb.raw('keys').size, 1);
  assert.equal(key.extractable, false);
  assert.equal(key.algorithm.name, 'AES-GCM');
  assert.equal(key.algorithm.length, 256);
  await assert.rejects(webcrypto.subtle.exportKey('raw', key), 'its bytes cannot be read');
  // Saving the same value twice does not seal it the same way twice.
  await vault.set('OPENAI_API_KEY', KEY_A);
  assert.notDeepEqual(Buffer.from(records.get('OPENAI_API_KEY').iv), Buffer.from(a.iv));
});

await check('overlapping saves keep every key, and a read sees the saves before it', async () => {
  const w = world();
  const vault = w.load();
  // Not awaited one by one, as the editor does.
  const saves = [vault.set('A', '1'), vault.set('A', '2'), vault.set('B', '3'), vault.set('C', '4'), vault.set('C', '')];
  const read = vault.get(['A', 'B', 'C']);
  await Promise.all(saves);
  assert.deepEqual(await read, { A: '2', B: '3' });
  assert.deepEqual(await w.load().get(['A', 'B', 'C']), { A: '2', B: '3' });
});

await check('two tabs starting together make one key and both can read what the other saved', async () => {
  // Both find no key and both make one before either has stored it. Only the first is kept, and both
  // use it: a tab that sealed with a key the store then lost would lose what it moved in from plaintext.
  const w = world({ plain: { FROM_PLAINTEXT: KEY_A } });
  let made = 0;
  let bothMade;
  const together = new Promise((resolve) => { bothMade = resolve; });
  const subtle = new Proxy(webcrypto.subtle, {
    get(target, prop) {
      if (prop === 'generateKey') {
        return async (...args) => {
          const key = await target.generateKey(...args);
          if (++made === 2) bothMade();
          await together;
          return key;
        };
      }
      const value = target[prop];
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  const crypto = { getRandomValues: webcrypto.getRandomValues.bind(webcrypto), subtle };
  const tabs = [V.createSecretVault({ ...w.env, crypto }), V.createSecretVault({ ...w.env, crypto })];
  assert.deepEqual(await Promise.all(tabs.map((t) => t.ready())), ['sealed', 'sealed']);
  assert.equal(made, 2, 'both really did make a key');
  assert.equal(w.idb.puts.keys, 1, 'and the key store was written once');
  await Promise.all([tabs[0].set('A', KEY_A), tabs[1].set('B', KEY_B)]);
  assert.equal(w.idb.raw('keys').size, 1, 'one key');
  assert.deepEqual(await tabs[0].get(['A', 'B', 'FROM_PLAINTEXT']), { A: KEY_A, B: KEY_B, FROM_PLAINTEXT: KEY_A }, 'each sees the other');
  assert.deepEqual(await tabs[1].get(['A', 'B', 'FROM_PLAINTEXT']), { A: KEY_A, B: KEY_B, FROM_PLAINTEXT: KEY_A });
  assert.deepEqual(await w.load().get(['A', 'B', 'FROM_PLAINTEXT']), { A: KEY_A, B: KEY_B, FROM_PLAINTEXT: KEY_A });
  assert.equal(plainStored(w), null);
});

// ---------------------------------------------------------------------------
// The move from plaintext
// ---------------------------------------------------------------------------
await check('plaintext keys are sealed on the first load, and the plaintext is gone', async () => {
  const w = world({ plain: { OPENAI_API_KEY: KEY_A, ANTHROPIC_API_KEY: KEY_B } });
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed');
  assert.equal(plainStored(w), null, 'removed');
  assert.deepEqual(await vault.get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']), { OPENAI_API_KEY: KEY_A, ANTHROPIC_API_KEY: KEY_B });
  for (const record of w.idb.raw('secrets').values()) assert.ok(!bytesOf(record).includes('secret-value'));
  // And on the next load they are still there, from the sealed store alone.
  assert.deepEqual(await w.load().get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']), { OPENAI_API_KEY: KEY_A, ANTHROPIC_API_KEY: KEY_B });
  assert.deepEqual(w.warns, []);
});

await check('where the plaintext and the sealed store both name a secret the plaintext is the newer', async () => {
  const w = world();
  const before = w.load();
  await before.set('A', 'sealed-old');
  await before.set('B', 'sealed-only');
  w.storage.setItem(PLAIN, JSON.stringify({ A: 'plain-new', C: 'plain-only' }));
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed');
  assert.deepEqual(await vault.get(['A', 'B', 'C']), { A: 'plain-new', B: 'sealed-only', C: 'plain-only' });
  assert.equal(plainStored(w), null);
});

await check('empty and non-string plaintext entries are not moved; an empty map is cleared', async () => {
  const w = world({ plain: { A: '', B: 5, C: null, D: 'kept', E: ['x'] } });
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed');
  assert.deepEqual(await vault.get(['A', 'B', 'C', 'D', 'E']), { D: 'kept' });
  assert.deepEqual([...w.idb.raw('secrets').keys()], ['D']);
  const empty = world({ plain: {} });
  assert.equal(await empty.load().ready(), 'sealed');
  assert.equal(plainStored(empty), null);
});

await check("plaintext that is not a map of secrets is left alone, said once, and does not stop sealing", async () => {
  for (const junk of ['not json {', '[1,2,3]', 'null', '"text"']) {
    const w = world({ plain: junk });
    const vault = w.load();
    assert.equal(await vault.ready(), 'sealed', junk);
    assert.equal(plainStored(w), junk, `${junk} stays`);
    assert.equal(w.warns.length, 1, junk);
    await vault.set('A', KEY_A);
    assert.deepEqual(await vault.get(['A']), { A: KEY_A });
  }
});

await check('a plaintext that cannot be removed is emptied instead; if it cannot be emptied either it stays, and is still the newer', async () => {
  const emptied = world({ plain: { A: 'a-1', B: 'b-1' } });
  emptied.storage.failRemove = true;
  const first = emptied.load();
  assert.equal(await first.ready(), 'sealed');
  assert.equal(plainStored(emptied), '{}', 'no keys left in it');
  assert.deepEqual(await first.get(['A', 'B']), { A: 'a-1', B: 'b-1' });

  const w = world({ plain: { A: 'a-1', B: 'b-1' } });
  w.storage.failRemove = true;
  w.storage.failSet = true; // neither removing nor emptying works
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed', 'sealed all the same: every value was read back');
  w.storage.failRemove = false;
  w.storage.failSet = false;
  assert.deepEqual(JSON.parse(plainStored(w)), { A: 'a-1', B: 'b-1' }, 'the plaintext is still there');
  assert.deepEqual(await vault.get(['A', 'B']), { A: 'a-1', B: 'b-1' });
  await vault.set('A', 'a-2');
  assert.deepEqual(JSON.parse(plainStored(w)), { B: 'b-1' }, 'A can no longer come back from it, or be hidden by it');
  assert.deepEqual(await vault.get(['A', 'B']), { A: 'a-2', B: 'b-1' });
  await vault.set('B', '');
  assert.equal(plainStored(w), null, 'and it goes when it is empty');
  assert.deepEqual(await vault.get(['A', 'B']), { A: 'a-2' });
  const later = w.load();
  assert.equal(await later.ready(), 'sealed');
  assert.deepEqual(await later.get(['A', 'B']), { A: 'a-2' }, 'nothing reverted at the next load');
});

await check('plaintext that is still there is the newer, at every read: it is laid over what is sealed', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', 'sealed-a');
  await vault.set('B', 'sealed-b');
  await vault.set('C', 'sealed-c');
  // Another tab that could not seal left this behind while this one was open.
  w.storage.setItem(PLAIN, JSON.stringify({ A: 'newer-a', B: '', D: 'plain-d' }));
  assert.deepEqual(await vault.get(['A', 'B', 'C', 'D']), { A: 'newer-a', C: 'sealed-c', D: 'plain-d' }, 'B is deleted by its empty value');
  assert.deepEqual(await vault.get(['B']), {});
});

// ---------------------------------------------------------------------------
// When sealing is not possible
// ---------------------------------------------------------------------------
const OLD = { OPENAI_API_KEY: KEY_A, ANTHROPIC_API_KEY: KEY_B };

/**
 * The plaintext still holds every key, and the vault reads and writes through it, saying so once.
 * A delete is kept as an empty value where the sealed store may be used at another load (that load
 * takes the delete to the store), and is just the name gone where it never can be (`sealable: false`).
 */
async function staysPlaintext(w, vault, why, { sealable = true } = {}) {
  assert.equal(await vault.ready(), 'plaintext', why);
  assert.deepEqual(JSON.parse(plainStored(w)), OLD, `${why}: the plaintext is where it was`);
  assert.deepEqual(await vault.get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'NOT_SET']), OLD, `${why}: and is read`);
  await vault.set('OPENAI_API_KEY', 'sk-changed');
  await vault.set('ANTHROPIC_API_KEY', '');
  await vault.set('NEW_KEY', 'sk-new');
  const written = sealable
    ? { OPENAI_API_KEY: 'sk-changed', ANTHROPIC_API_KEY: '', NEW_KEY: 'sk-new' }
    : { OPENAI_API_KEY: 'sk-changed', NEW_KEY: 'sk-new' };
  assert.deepEqual(JSON.parse(plainStored(w)), written, `${why}: and written`);
  assert.deepEqual(await vault.get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'NEW_KEY']), { OPENAI_API_KEY: 'sk-changed', NEW_KEY: 'sk-new' });
  assert.equal(w.warns.length, 1, `${why}: said once, not per operation (${w.warns.join(' | ')})`);
  assert.match(w.warns[0], /plain localStorage/);
}

await check('no IndexedDB: the keys stay in plaintext, and are used as before', async () => {
  const w = world({ plain: OLD, env: { indexedDB: undefined } });
  await staysPlaintext(w, w.load(), 'no IndexedDB', { sealable: false });
});

await check('no WebCrypto (an http:// page): the keys stay in plaintext, and are used as before', async () => {
  const w = world({ plain: OLD, env: { crypto: { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) } } });
  await staysPlaintext(w, w.load(), 'no subtle', { sealable: false });
  assert.equal(w.idb.has(), false, 'the database was not even opened');
});

await check('a database that will not open: the keys stay in plaintext', async () => {
  const w = world({ plain: OLD });
  w.idb.knobs.openError = new DOMException('denied', 'SecurityError');
  await staysPlaintext(w, w.load(), 'open refused');
});

await check('a write the store refuses part-way through the move: nothing is removed, the vault stays sealed, and every key is readable', async () => {
  const w = world({ plain: OLD });
  w.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'secrets', skip: 1 });
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed', 'the store was read: what it refused was one write');
  assert.deepEqual(JSON.parse(plainStored(w)), OLD, 'the plaintext is where it was');
  assert.deepEqual(await vault.get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'NOT_SET']), OLD, 'and every key is read: the plaintext is the newer');
  assert.equal(w.warns.length, 1, 'said once');
  assert.match(w.warns[0], /plain localStorage/);
  // The store takes writes again, so saves go to it, and each leaves the plaintext.
  await vault.set('OPENAI_API_KEY', 'sk-changed');
  assert.deepEqual(JSON.parse(plainStored(w)), { ANTHROPIC_API_KEY: KEY_B }, 'a name saved in the store is no longer in the plaintext');
  await vault.set('ANTHROPIC_API_KEY', '');
  await vault.set('NEW_KEY', 'sk-new');
  assert.equal(plainStored(w), null, 'and it is gone when it is empty');
  assert.deepEqual(await vault.get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'NEW_KEY']), { OPENAI_API_KEY: 'sk-changed', NEW_KEY: 'sk-new' });
  // The next page load has nothing left to move.
  const next = w.load();
  assert.equal(await next.ready(), 'sealed');
  assert.deepEqual(await next.get(['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'NEW_KEY']), { OPENAI_API_KEY: 'sk-changed', NEW_KEY: 'sk-new' });
  assert.equal(w.warns.length, 1, 'and says nothing more');
});

await check('a store that refuses every write hides no key that was sealed before, load after load', async () => {
  // The disk is full: the sealed records can be read, and nothing can be written to them.
  const w = world();
  const first = w.load();
  await first.set('K3', 'k3-sealed');
  w.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'secrets', times: 1000 });
  await first.set('K2', 'k2-plain');
  assert.deepEqual(JSON.parse(plainStored(w)), { K2: 'k2-plain' }, 'a refused save is kept in plaintext');
  for (const load of [1, 2]) {
    const vault = w.load();
    assert.equal(await vault.ready(), 'sealed', `load ${load}: the move fails again, and the store is still read`);
    assert.deepEqual(await vault.get(['K3', 'K2']), { K3: 'k3-sealed', K2: 'k2-plain' }, `load ${load}: the sealed key is not hidden`);
    await vault.set(`KEPT_${load}`, `kept-${load}`);
    assert.deepEqual(await vault.get(['K3', `KEPT_${load}`]), { K3: 'k3-sealed', [`KEPT_${load}`]: `kept-${load}` }, `load ${load}: a save is kept, and read`);
  }
  // There is room again: the next load moves every kept key in.
  w.idb.knobs.failures.length = 0;
  const later = w.load();
  assert.equal(await later.ready(), 'sealed');
  assert.equal(plainStored(w), null);
  const all = { K3: 'k3-sealed', K2: 'k2-plain', KEPT_1: 'kept-1', KEPT_2: 'kept-2' };
  assert.deepEqual(await later.get(Object.keys(all)), all);
  assert.deepEqual(await w.load().get(Object.keys(all)), all, 'from the sealed store alone');
});

await check('a delete the store refuses while the plaintext is moved in is still a delete, and hides no other key', async () => {
  const w = world();
  const before = w.load();
  await before.set('A', KEY_A);
  await before.set('B', KEY_B);
  w.idb.fail('delete', new DOMException('locked', 'UnknownError'), { store: 'secrets' });
  await before.set('A', ''); // refused: kept as an empty value in the plaintext
  assert.deepEqual(JSON.parse(plainStored(w)), { A: '' });
  w.idb.fail('delete', new DOMException('locked', 'UnknownError'), { store: 'secrets' }); // and refused again by the move
  const stuck = w.load();
  assert.equal(await stuck.ready(), 'sealed');
  assert.deepEqual(await stuck.get(['A', 'B']), { B: KEY_B }, 'A is gone as far as anyone can tell, and B is not hidden');
  assert.deepEqual(JSON.parse(plainStored(w)), { A: '' }, 'the delete is still to be made');
  const next = w.load();
  assert.equal(await next.ready(), 'sealed');
  assert.deepEqual(await next.get(['A', 'B']), { B: KEY_B });
  assert.deepEqual([...w.idb.raw('secrets').keys()], ['B'], 'made at last');
  assert.equal(plainStored(w), null);
});

await check('a value that does not read back the same is not trusted: the plaintext stays', async () => {
  const w = world({ plain: OLD });
  w.idb.knobs.corruptReads = true;
  await staysPlaintext(w, w.load(), 'read back differs');
});

await check('a key that cannot be made: the keys stay in plaintext', async () => {
  const w = world({ plain: OLD, env: { crypto: { getRandomValues: webcrypto.getRandomValues.bind(webcrypto), subtle: { generateKey: () => Promise.reject(new Error('no entropy')) } } } });
  await staysPlaintext(w, w.load(), 'no key');
});

await check('a store that never answers: the keys stay in plaintext after the wait, and a late answer removes nothing', async () => {
  const w = world({ plain: OLD, env: { timeoutMs: 40 } });
  w.idb.knobs.hangOpen = true;
  const started = Date.now();
  await staysPlaintext(w, w.load(), 'open never answers');
  assert.ok(Date.now() - started >= 30, 'it waited');
  // The browser wakes up after the page gave up on it. The move must not finish then.
  w.idb.release();
  await tick(150);
  assert.deepEqual(JSON.parse(plainStored(w)), { OPENAI_API_KEY: 'sk-changed', ANTHROPIC_API_KEY: '', NEW_KEY: 'sk-new' }, 'the plaintext this page relies on is untouched');
});

await check('a key deleted while the store could not be opened is deleted there too, at the next load', async () => {
  const w = world();
  const before = w.load();
  await before.set('A', KEY_A);
  await before.set('B', KEY_B);
  w.idb.knobs.openError = new DOMException('denied', 'SecurityError'); // this page load cannot open the store
  const degraded = w.load();
  assert.equal(await degraded.ready(), 'plaintext');
  await degraded.set('A', ''); // revoked here
  await degraded.set('B', 'b-rotated'); // rotated here
  assert.deepEqual(await degraded.get(['A', 'B']), { B: 'b-rotated' }, 'gone, as far as this page can tell');
  w.idb.knobs.openError = null; // the next load can open it
  const next = w.load();
  assert.equal(await next.ready(), 'sealed');
  assert.deepEqual(await next.get(['A', 'B']), { B: 'b-rotated' }, 'A stayed deleted, B is the rotated one');
  assert.deepEqual([...w.idb.raw('secrets').keys()], ['B'], 'the sealed record of A is gone');
  assert.equal(plainStored(w), null);
});

await check('a move the page gave up on stops at once: it seals nothing more, not even a key deleted since', async () => {
  const w = world({ plain: { K1: 'v1', K2: 'v2', K3: 'v3' }, env: { timeoutMs: 150 } });
  w.idb.hold('put', { store: 'secrets', skip: 1 }); // K1 goes in; the write of K2 never answers
  const vault = w.load();
  assert.equal(await vault.ready(), 'plaintext', 'the move ran out of time');
  await vault.set('K3', ''); // deleted here, before the move got to it
  w.idb.release(); // the browser answers at last, and the abandoned move carries on
  await tick(150);
  assert.deepEqual([...w.idb.raw('secrets').keys()].sort(), ['K1', 'K2'], 'K3 was not sealed after it was deleted');
  assert.deepEqual(JSON.parse(plainStored(w)), { K1: 'v1', K2: 'v2', K3: '' }, 'the plaintext this page relied on is untouched');
  // The next load has all the time it needs (the short patience was only to make the first give up).
  const next = V.createSecretVault({ ...w.env, timeoutMs: 5000 });
  assert.equal(await next.ready(), 'sealed');
  assert.deepEqual(await next.get(['K1', 'K2', 'K3']), { K1: 'v1', K2: 'v2' }, 'and K3 stays deleted');
  assert.equal(plainStored(w), null);
});

await check('a store too slow to finish the move in time: what the page kept meanwhile is not removed when the move ends', async () => {
  const w = world({ plain: { K1: 'v1', K2: 'v2', K3: 'v3', K4: 'v4' }, env: { timeoutMs: 250 } });
  w.idb.knobs.delay.put = 80; // five writes: well over the patience
  const vault = w.load();
  assert.equal(await vault.ready(), 'plaintext', 'gave up part-way through');
  await vault.set('LATE', 'kept-in-plaintext'); // written while the abandoned move is still going
  await tick(900); // the move runs to its end
  assert.deepEqual(JSON.parse(plainStored(w)), { K1: 'v1', K2: 'v2', K3: 'v3', K4: 'v4', LATE: 'kept-in-plaintext' }, 'the plaintext is untouched');
  assert.deepEqual(await vault.get(['K1', 'LATE']), { K1: 'v1', LATE: 'kept-in-plaintext' });
});

await check('a storage that cannot be read at all is an empty store, and sealing works', async () => {
  const w = world();
  w.storage.failGet = true;
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed');
  await vault.set('A', KEY_A);
  assert.deepEqual(await vault.get(['A']), { A: KEY_A });
});

await check('in plaintext mode a refused write is said once, as it was, and the rest still works', async () => {
  const w = world({ plain: OLD, env: { indexedDB: undefined } });
  const vault = w.load();
  assert.equal(await vault.ready(), 'plaintext');
  w.storage.failSet = true;
  await vault.set('X', 'x-1');
  await vault.set('Y', 'y-1');
  assert.ok(w.warns.some((m) => /could not persist secret/.test(m)), w.warns.join(' | '));
  assert.equal(w.warns.filter((m) => /could not persist secret/.test(m)).length, 1);
  assert.deepEqual(JSON.parse(plainStored(w)), OLD, 'what was there is intact');
});

await check('in plaintext mode a damaged plaintext is treated as empty and replaced by the next save, as before', async () => {
  for (const junk of ['not json {', '[1,2]', 'null']) {
    const w = world({ plain: junk, env: { indexedDB: undefined } });
    const vault = w.load();
    assert.equal(await vault.ready(), 'plaintext');
    assert.deepEqual(await vault.get(['A']), {});
    await vault.set('A', KEY_A);
    assert.deepEqual(JSON.parse(plainStored(w)), { A: KEY_A }, junk);
    assert.deepEqual(await vault.get(['A']), { A: KEY_A });
  }
});

// ---------------------------------------------------------------------------
// A store that was wiped
// ---------------------------------------------------------------------------
await check('a wiped key store: nothing throws, the keys that cannot be opened are said once, and new ones work', async () => {
  const w = world();
  const before = w.load();
  await before.set('A', KEY_A);
  await before.set('B', KEY_B);
  w.idb.raw('keys').clear(); // the key is gone, the sealed records are left
  const vault = w.load();
  assert.equal(await vault.ready(), 'sealed');
  assert.deepEqual(await vault.get(['A', 'B']), {}, 'not there, as far as anyone can tell');
  assert.equal(w.warns.length, 1);
  assert.match(w.warns[0], /cannot be opened/);
  assert.equal(w.idb.raw('secrets').size, 0, 'what could never open is not kept');
  await vault.set('A', 'entered-again');
  assert.deepEqual(await vault.get(['A', 'B']), { A: 'entered-again' });
  assert.deepEqual(await w.load().get(['A', 'B']), { A: 'entered-again' });
  assert.equal(w.idb.raw('keys').size, 1);
});

await check('a wiped database under a live page: reads and saves go on, on a fresh one', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  w.idb.wipe(); // "clear site data", the tab still open
  assert.deepEqual(await vault.get(['A']), {}, 'cleared means cleared');
  await vault.set('B', KEY_B);
  assert.deepEqual(await vault.get(['A', 'B']), { B: KEY_B });
  assert.deepEqual(await w.load().get(['B']), { B: KEY_B });
  assert.deepEqual(w.warns, [], 'not a failure');
});

await check('a connection closed without a word is replaced once, and the read and the save go through the store', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  w.idb.closeQuietly();
  assert.deepEqual(await vault.get(['A']), { A: KEY_A });
  assert.deepEqual(w.warns, [], 'a read on a closed connection is retried, not given up on');
  w.idb.closeQuietly();
  await vault.set('B', KEY_B);
  assert.equal(plainStored(w), null, 'and so is a save: it did not fall back to plaintext');
  assert.deepEqual([...w.idb.raw('secrets').keys()].sort(), ['A', 'B'], 'B is sealed in the store');
  assert.deepEqual(w.warns, [], 'and nothing was said');
  assert.deepEqual(await vault.get(['A', 'B']), { A: KEY_A, B: KEY_B });
});

await check('one damaged record costs that key only, and is not deleted', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  await vault.set('B', KEY_B);
  w.idb.raw('secrets').set('A', { iv: new Uint8Array(12), data: new ArrayBuffer(40) });
  w.idb.raw('secrets').set('C', 'not a record');
  const next = w.load();
  assert.equal(await next.ready(), 'sealed');
  assert.deepEqual(await next.get(['A', 'B', 'C']), { B: KEY_B });
  assert.deepEqual([...w.idb.raw('secrets').keys()].sort(), ['A', 'B', 'C'], 'a key that did exist is not thrown away');
  assert.equal(w.warns.filter((m) => /cannot be opened/.test(m)).length, 1, 'said once');
});

await check('a store that fails mid-session: reads fall back to what this tab holds, and it is said once', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  w.idb.fail('get', new DOMException('disk error', 'UnknownError'), { times: 100 });
  assert.deepEqual(await vault.get(['A']), { A: KEY_A }, 'from what this tab last saw');
  assert.deepEqual(await vault.get(['A', 'B']), { A: KEY_A });
  assert.equal(w.warns.filter((m) => /could not read/.test(m)).length, 1);
});

await check('a store that stops answering does not hold a read up for ever: the keys this tab holds are given', async () => {
  const w = world({ env: { timeoutMs: 150 } });
  const vault = w.load();
  await vault.set('A', KEY_A);
  w.idb.hold('get', { store: 'secrets' }); // the next read of the records never answers
  const answer = await orHung(vault.get(['A', 'B']));
  w.idb.release();
  assert.notEqual(answer, 'HUNG', 'a hung read gives up');
  assert.deepEqual(answer, { A: KEY_A }, 'and gives what this tab last saw');
  assert.equal(w.warns.filter((m) => /could not read/.test(m)).length, 1, 'said once');
});

await check('a store that stops answering does not hold a save up for ever: the key is kept in plaintext, and moved in at the next load', async () => {
  const w = world({ env: { timeoutMs: 150 } });
  const vault = w.load();
  await vault.set('A', KEY_A);
  w.idb.hold('put', { store: 'secrets' }); // the next write never answers
  const saved = await orHung(vault.set('B', KEY_B).then(() => 'saved'));
  w.idb.release(); // it lands late, after the page gave up on it
  assert.equal(saved, 'saved', 'a hung save gives up');
  assert.deepEqual(JSON.parse(plainStored(w)), { B: KEY_B }, 'and keeps the key where it can');
  assert.equal(w.warns.filter((m) => /could not save/.test(m)).length, 1, 'said once');
  assert.deepEqual(await vault.get(['A', 'B']), { A: KEY_A, B: KEY_B });
  await tick(50);
  const next = V.createSecretVault({ ...w.env, timeoutMs: 5000 });
  assert.equal(await next.ready(), 'sealed');
  assert.equal(plainStored(w), null);
  assert.deepEqual(await next.get(['A', 'B']), { A: KEY_A, B: KEY_B });
});

await check('a save the store refuses is kept in plaintext, said once, and moved in at the next load', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', 'a-1');
  w.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'secrets', times: 2 });
  await vault.set('A', 'a-2'); // refused
  await vault.set('B', KEY_B); // refused
  assert.deepEqual(JSON.parse(plainStored(w)), { A: 'a-2', B: KEY_B }, 'kept, not lost');
  assert.equal(w.warns.filter((m) => /could not save/.test(m)).length, 1, 'said once for two refusals');
  assert.deepEqual(await vault.get(['A', 'B']), { A: 'a-2', B: KEY_B }, 'and used');
  await vault.set('C', 'c-1'); // the store takes this one
  assert.deepEqual(JSON.parse(plainStored(w)), { A: 'a-2', B: KEY_B }, 'a save that works leaves the others where they are');
  assert.deepEqual(await vault.get(['A', 'B', 'C']), { A: 'a-2', B: KEY_B, C: 'c-1' });
  // A reload moves them in.
  const next = w.load();
  assert.equal(await next.ready(), 'sealed');
  assert.equal(plainStored(w), null);
  assert.deepEqual(await next.get(['A', 'B', 'C']), { A: 'a-2', B: KEY_B, C: 'c-1' });
  assert.deepEqual(await w.load().get(['A', 'B', 'C']), { A: 'a-2', B: KEY_B, C: 'c-1' }, 'from the sealed store alone');
});

await check('a delete the store refuses hides the old value at once, and is made at the next load', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  await vault.set('B', KEY_B);
  w.idb.fail('delete', new DOMException('locked', 'UnknownError'), { store: 'secrets' });
  await vault.set('A', ''); // refused
  assert.deepEqual(JSON.parse(plainStored(w)), { A: '' });
  assert.deepEqual(await vault.get(['A', 'B']), { B: KEY_B }, 'it is gone as far as anyone can tell');
  const next = w.load();
  assert.equal(await next.ready(), 'sealed');
  assert.equal(plainStored(w), null);
  assert.deepEqual(await next.get(['A', 'B']), { B: KEY_B });
  assert.deepEqual([...w.idb.raw('secrets').keys()], ['B'], 'deleted where it was sealed');
});

await check('a save that can be kept nowhere is an error the editor hears', async () => {
  const w = world();
  const vault = w.load();
  await vault.set('A', KEY_A);
  w.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'secrets' });
  w.storage.failSet = true;
  await assert.rejects(vault.set('B', KEY_B), /quota/);
  assert.deepEqual(await vault.get(['A']), { A: KEY_A }, 'and what was there is intact');
});

// ---------------------------------------------------------------------------
// What the vault says
// ---------------------------------------------------------------------------
await check('a warning that has no detail is one argument to the console, not a stray "undefined"', async () => {
  const calls = [];
  const original = console.warn;
  console.warn = (...args) => { calls.push(args); };
  try {
    const w = world({ env: { warn: undefined } }); // the vault's own way of saying it: console.warn
    const before = w.load();
    await before.set('A', KEY_A);
    w.idb.raw('keys').clear(); // the key is gone: what was sealed cannot be opened
    const vault = w.load();
    assert.equal(await vault.ready(), 'sealed');
    w.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'secrets' });
    await vault.set('B', KEY_B); // refused, with the error to say so
  } finally {
    console.warn = original;
  }
  assert.equal(calls.length, 2, calls.map((c) => c[0]).join(' | '));
  assert.equal(calls[0].length, 1, 'no detail: the message alone');
  assert.match(calls[0][0], /cannot be opened/);
  assert.equal(calls[1].length, 2, 'a detail: the message and the error');
  assert.match(calls[1][0], /could not save/);
  assert.equal(calls[1][1].name, 'QuotaExceededError');
});

await check('no warning carries the value of a key, in its message or in its detail', async () => {
  const SECRET = 'sk-never-in-a-log-4c9d2e7a51';
  const show = (d) => (d instanceof Error ? `${d.name}: ${d.message}\n${d.stack ?? ''}` : typeof d === 'object' && d !== null ? JSON.stringify(d) : String(d));
  const said = (w) => [...w.warns, ...w.details.map(show)].join('\n');
  const worlds = {};
  {
    const w = (worlds['a wiped key store'] = world());
    await w.load().set('A', SECRET);
    w.idb.raw('keys').clear();
    await w.load().ready();
  }
  {
    const w = (worlds['a move the store refuses'] = world({ plain: { A: SECRET } }));
    w.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'secrets', times: 5 });
    const vault = w.load();
    await vault.ready();
    await vault.set('B', SECRET);
  }
  {
    const w = (worlds['a store that will not open'] = world({ plain: { A: SECRET } }));
    w.idb.knobs.openError = new DOMException('denied', 'SecurityError');
    const vault = w.load();
    await vault.ready();
    await vault.set('B', SECRET);
  }
  {
    const w = (worlds['a value that does not read back'] = world({ plain: { A: SECRET } }));
    w.idb.knobs.corruptReads = true;
    await w.load().ready();
  }
  {
    const w = (worlds['plaintext that is damaged'] = world({ plain: `{"A": "${SECRET}` }));
    await w.load().ready();
  }
  {
    const w = (worlds['plaintext that cannot be written'] = world({ plain: { A: SECRET }, env: { indexedDB: undefined } }));
    const vault = w.load();
    await vault.ready();
    w.storage.failSet = true;
    await vault.set('B', SECRET);
  }
  {
    const w = (worlds['a store that fails while the page runs'] = world());
    const vault = w.load();
    await vault.set('A', SECRET);
    w.idb.fail('get', new DOMException('disk error', 'UnknownError'), { times: 10 });
    await vault.get(['A']);
    w.idb.fail('put', new DOMException('disk error', 'UnknownError'), { store: 'secrets' });
    await vault.set('B', SECRET);
  }
  for (const [what, w] of Object.entries(worlds)) {
    assert.ok(w.warns.length > 0, `${what}: it says something`);
    assert.ok(!said(w).includes(SECRET), `${what}: the value is in what it says: ${said(w)}`);
  }
});

// ---------------------------------------------------------------------------
// The shim and the editor
// ---------------------------------------------------------------------------
await check("the shim's get_secrets and store_secret are the vault's: a first call moves the old plaintext in", async () => {
  assert.deepEqual(JSON.parse(shimWorld.storage.raw.get(PLAIN)), preseeded, 'plaintext before the first call');
  const got = await V.invoke('get_secrets', { keys: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'MISSING'] });
  assert.deepEqual(got, preseeded);
  assert.equal(shimWorld.storage.raw.has(PLAIN), false, 'and it is not plaintext after');
  assert.equal(shimWorld.idb.raw('secrets').size, 2);
  for (const record of shimWorld.idb.raw('secrets').values()) assert.ok(!bytesOf(record).includes('sk-shim-old'));
});

await check('store_secret seals, an empty value deletes, and odd arguments are harmless', async () => {
  assert.equal(await V.invoke('store_secret', { key: 'GEMINI_API_KEY', value: 'AIza-shim-3' }), null);
  assert.deepEqual(await V.invoke('get_secrets', { keys: ['GEMINI_API_KEY'] }), { GEMINI_API_KEY: 'AIza-shim-3' });
  assert.equal(shimWorld.storage.raw.has(PLAIN), false, 'never plaintext again');
  assert.equal(await V.invoke('store_secret', { key: 'GEMINI_API_KEY', value: '' }), null);
  assert.deepEqual(await V.invoke('get_secrets', { keys: ['GEMINI_API_KEY'] }), {});
  assert.equal(await V.invoke('store_secret', { key: 'OPENAI_API_KEY', value: null }), null, 'no value deletes');
  assert.deepEqual(await V.invoke('get_secrets', { keys: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'] }), { ANTHROPIC_API_KEY: 'sk-shim-old-2' });
  const before = shimWorld.idb.raw('secrets').size;
  assert.equal(await V.invoke('store_secret', { value: 'no key name' }), null);
  assert.equal(await V.invoke('store_secret', undefined), null);
  assert.deepEqual(await V.invoke('get_secrets', {}), {});
  assert.deepEqual(await V.invoke('get_secrets', { keys: 'OPENAI_API_KEY' }), {});
  assert.deepEqual(await V.invoke('get_secrets'), {});
  assert.equal(shimWorld.idb.raw('secrets').size, before);
});

await check('the shim keeps no key of its own, and the editor starts the vault when it starts', () => {
  const core = fs.readFileSync(path.join(UI, 'src/tauri-shim/core.ts'), 'utf8');
  assert.doesNotMatch(core, /oaiy_web_secrets|SECRETS_KEY|readSecretMap|writeSecretMap/);
  assert.match(core, /import \{ secretVault \} from '\.\/secretVault';/);
  assert.match(core, /secretVault\.get\(keys\)/);
  assert.match(core, /secretVault\.set\(key, value\)/);
  const main = fs.readFileSync(path.join(UI, 'src/main.tsx'), 'utf8');
  assert.match(main, /^import \{ secretVault \} from '\.\/tauri-shim\/secretVault';$/m);
  assert.match(main, /^void secretVault\.ready\(\);$/m, 'started at the top level, not awaited');
  // The plaintext map's name is written in one place.
  const holders = [];
  const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).forEach((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : /\.(tsx?|json)$/.test(e.name) && holders.push(path.join(dir, e.name))));
  walk(path.join(UI, 'src'));
  const named = holders.filter((f) => fs.readFileSync(f, 'utf8').includes('oaiy_web_secrets')).map((f) => path.relative(UI, f).replace(/\\/g, '/'));
  assert.deepEqual(named, ['src/tauri-shim/secretVault.ts']);
});

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
