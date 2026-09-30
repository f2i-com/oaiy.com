/**
 * The secret store of the providers origin (web/providers/src/vault.ts, design 3.3), on a fake IndexedDB whose transactions end when
 * control leaves them, as a browser's do.
 */
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { describe, it } from 'node:test';
import { KEY, M, anotherDocument, makeHolder, slowCrypto } from '../support/holder.mjs';

const stored = (holder) => holder.idb.raw('vault').get('record');
const allBytes = (holder) => {
  const parts = [];
  for (const store of ['vault', 'keys', 'records', 'meta', 'budget']) {
    for (const value of holder.idb.raw(store)?.values() ?? []) {
      parts.push(JSON.stringify(value, (k, v) => (v instanceof Uint8Array ? Buffer.from(v).toString('latin1') : v instanceof ArrayBuffer ? Buffer.from(v).toString('latin1') : v)));
    }
  }
  return parts.join('\n');
};

describe('the device vault', () => {
  it('opens as an unlocked `device` vault, and says which ways there are to open it', async () => {
    const h = makeHolder();
    assert.deepEqual(await h.vault.ready(), { mode: 'device', locked: false });
    assert.deepEqual(await h.vault.wrappers(), [{ type: 'device', id: 'device' }]);
    assert.equal(await h.vault.health(), 'ok');
    assert.deepEqual(await h.vault.names(), []);
  });

  it('a secret set is a secret got, across a reload, with unicode and a long value; an empty value deletes', async () => {
    const h = makeHolder();
    const long = 'k'.repeat(20000);
    await h.vault.set('key:a', KEY);
    await h.vault.set('key:b', 'πάντα ῥεῖ 🔑 "quoted" \\ back');
    await h.vault.set('long', long);
    assert.deepEqual(await h.vault.get(['key:a', 'key:b', 'long', 'missing']), { 'key:a': KEY, 'key:b': 'πάντα ῥεῖ 🔑 "quoted" \\ back', long });
    const reloaded = anotherDocument(h);
    assert.deepEqual(await reloaded.vault.get(['key:a']), { 'key:a': KEY }, 'a second document opens it');
    assert.deepEqual((await h.vault.names()).sort(), ['key:a', 'key:b', 'long']);
    await h.vault.set('key:a', '');
    assert.deepEqual(await h.vault.get(['key:a', 'key:b']), { 'key:b': 'πάντα ῥεῖ 🔑 "quoted" \\ back' });
    assert.deepEqual(await reloaded.vault.get(['key:a']), {}, 'and the other document sees the delete');
  });

  it('names that are properties of every object are only names', async () => {
    const h = makeHolder();
    await h.vault.set('__proto__', 'proto-value');
    await h.vault.set('constructor', 'ctor-value');
    const got = await h.vault.get(['__proto__', 'constructor', 'toString', 'hasOwnProperty']);
    assert.equal(Object.getOwnPropertyNames(got).sort().join(), '__proto__,constructor');
    assert.equal(Object.getOwnPropertyDescriptor(got, '__proto__').value, 'proto-value');
    assert.equal(got.constructor, 'ctor-value');
    assert.deepEqual(await h.vault.get(['toString']), {});
  });

  it('what is stored is ciphertext: no key, and no trace of the value, in any store; the wrapping key cannot be exported', async () => {
    const h = makeHolder();
    await h.vault.set('key:a', KEY);
    await h.vault.set('key:b', 'a-second-secret-value-1234567890');
    const all = allBytes(h);
    for (const piece of [KEY, KEY.slice(0, 16), KEY.slice(-16), 'a-second-secret-value']) assert.ok(!all.includes(piece), `${piece} is stored in the clear`);
    const record = stored(h);
    assert.equal(record.v, 1);
    assert.equal(record.wrappers.length, 1);
    assert.equal(record.wrappers[0].type, 'device');
    for (const item of Object.values(record.items)) {
      assert.equal(item.iv.byteLength, 12, 'an AES-GCM nonce');
      assert.ok(item.ct.byteLength > 17, 'ciphertext with its tag');
    }
    const [a, b] = Object.values(record.items);
    assert.notDeepEqual(Buffer.from(a.iv), Buffer.from(b.iv), 'a nonce each');
    const kek = h.idb.raw('keys').get('device');
    assert.equal(kek.extractable, false);
    assert.equal(kek.algorithm.name, 'AES-GCM');
    assert.equal(kek.algorithm.length, 256);
    await assert.rejects(webcrypto.subtle.exportKey('raw', kek), 'its bytes cannot be read');
    const before = Buffer.from(record.items['key:a'].iv);
    await h.vault.set('key:a', KEY);
    assert.notDeepEqual(Buffer.from(stored(h).items['key:a'].iv), before, 'the same value is not sealed the same way twice');
  });

  it('a value moved from one name to another does not open (the name is part of what is sealed)', async () => {
    const h = makeHolder();
    await h.vault.set('key:a', KEY);
    await h.vault.set('key:b', 'other-value-abcdefgh');
    const record = stored(h);
    record.items['key:c'] = record.items['key:a']; // copied to another name by someone with the database
    h.idb.raw('vault').set('record', record);
    assert.deepEqual(await h.vault.get(['key:a', 'key:c']), { 'key:a': KEY }, 'the copy is refused');
  });

  it('the wrapper is bound to its record: a wrapper taken from another vault does not open this one', async () => {
    const a = makeHolder();
    const b = makeHolder();
    await a.vault.set('key:a', KEY);
    await b.vault.set('key:b', 'b-secret-value-123456');
    const rec = stored(a);
    rec.wrappers = stored(b).wrappers;
    a.idb.raw('vault').set('record', rec);
    a.idb.raw('keys').set('device', b.idb.raw('keys').get('device'));
    const fresh = anotherDocument(a);
    assert.equal(await fresh.vault.health(), 'damaged');
  });

  it('two documents that start together make one vault, and each reads what the other saved', async () => {
    const h = makeHolder();
    const other = anotherDocument(h);
    await Promise.all([h.vault.ready(), other.vault.ready()]);
    assert.equal(h.idb.raw('keys').size, 1);
    assert.equal(h.idb.counters.puts.vault >= 1, true);
    await Promise.all([h.vault.set('key:a', KEY), other.vault.set('key:b', 'second-doc-secret-value')]);
    assert.deepEqual(await h.vault.get(['key:a', 'key:b']), { 'key:a': KEY, 'key:b': 'second-doc-secret-value' });
    assert.deepEqual(await other.vault.get(['key:a', 'key:b']), { 'key:a': KEY, 'key:b': 'second-doc-secret-value' });
    assert.equal(h.idb.raw('vault').size, 1);
  });

  it('overlapping saves from two documents keep every secret (a read-modify-write is one transaction)', async () => {
    const h = makeHolder();
    const other = anotherDocument(h);
    await h.vault.ready();
    const names = Array.from({ length: 12 }, (_, i) => `key:${i}`);
    await Promise.all(names.map((name, i) => (i % 2 ? other : h).vault.set(name, `value-${i}-${'x'.repeat(20)}`)));
    const got = await h.vault.get(names);
    assert.equal(Object.keys(got).length, 12);
    for (const [i, name] of names.entries()) assert.equal(got[name], `value-${i}-${'x'.repeat(20)}`);
  });

  it('cryptography is never awaited inside a transaction: a slow WebCrypto changes nothing', async () => {
    const h = makeHolder({ crypto: slowCrypto(6) });
    await h.vault.set('key:a', KEY);
    await h.vault.set('key:b', 'b-value-1234567');
    assert.deepEqual(await h.vault.get(['key:a', 'key:b']), { 'key:a': KEY, 'key:b': 'b-value-1234567' });
    const other = anotherDocument(h, { crypto: slowCrypto(6) });
    await Promise.all([other.vault.set('key:c', 'c-value-1234567'), h.vault.set('key:d', 'd-value-1234567')]);
    assert.deepEqual(Object.keys(await h.vault.get(['key:a', 'key:b', 'key:c', 'key:d'])).sort(), ['key:a', 'key:b', 'key:c', 'key:d']);
  });

  it('a wrapping key that is gone (browser data cleared in part) is `damaged`: what was sealed is not read, and a new secret is kept', async () => {
    const h = makeHolder();
    await h.vault.set('key:a', KEY);
    h.idb.raw('keys').clear();
    const fresh = anotherDocument(h);
    assert.equal(await fresh.vault.health(), 'damaged');
    assert.deepEqual(await fresh.vault.get(['key:a']), {}, 'nothing throws, and nothing is guessed');
    await fresh.vault.set('key:b', 'entered-again-secret');
    assert.equal(await fresh.vault.health(), 'ok');
    assert.deepEqual(await fresh.vault.get(['key:a', 'key:b']), { 'key:b': 'entered-again-secret' });
    assert.deepEqual(await fresh.vault.names(), ['key:b'], 'what could never open is not kept');
  });

  it('a corrupted wrapper is damaged too, and a database wiped under a live page is made again', async () => {
    const h = makeHolder();
    await h.vault.set('key:a', KEY);
    const rec = stored(h);
    rec.wrappers[0].ct = new Uint8Array(rec.wrappers[0].ct.length);
    h.idb.raw('vault').set('record', rec);
    assert.equal(await anotherDocument(h).vault.health(), 'damaged');

    const w = makeHolder();
    await w.vault.set('key:a', KEY);
    w.idb.wipe();
    assert.deepEqual(await w.vault.get(['key:a']), {}, 'cleared means cleared');
    await w.vault.set('key:b', 'after-the-wipe-secret');
    assert.deepEqual(await w.vault.get(['key:b']), { 'key:b': 'after-the-wipe-secret' });
  });

  it('where secrets cannot be sealed nothing is kept: no WebCrypto, no IndexedDB, and no plaintext fallback', async () => {
    const noSubtle = makeHolder({ crypto: { getRandomValues: (a) => webcrypto.getRandomValues(a) } });
    await assert.rejects(noSubtle.vault.ready(), { name: 'VaultError', code: 'unavailable' });
    await assert.rejects(noSubtle.vault.set('key:a', KEY), { name: 'VaultError' });
    assert.ok(!allBytes(noSubtle).includes(KEY));

    const noIdb = M.vault.createDeviceVault(M.db.openDb({ indexedDB: undefined }), { crypto: webcrypto });
    await assert.rejects(noIdb.ready(), /no IndexedDB/);
    await assert.rejects(noIdb.set('key:a', KEY), /no IndexedDB/);
  });

  it('lock does nothing a device vault could undo, and unlock has no wrapper to use', async () => {
    const h = makeHolder();
    await h.vault.set('key:a', KEY);
    assert.deepEqual(await h.vault.lock(), { mode: 'device', locked: false });
    assert.deepEqual(await h.vault.get(['key:a']), { 'key:a': KEY });
    await assert.rejects(h.vault.unlock('passphrase', 'hunter2'), { name: 'VaultError', code: 'unsupported' });
  });

  it('bad names and non-text values are refused', async () => {
    const h = makeHolder();
    for (const name of ['', 5, null, undefined, 'x'.repeat(201)]) await assert.rejects(h.vault.set(name, 'v'), { name: 'VaultError' }, String(name));
    for (const value of [5, null, undefined, {}]) await assert.rejects(h.vault.set('key:a', value), { name: 'VaultError' }, String(value));
    assert.deepEqual(await h.vault.get([]), {});
  });
});
