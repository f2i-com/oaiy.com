/**
 * A key that is stored and cannot be opened (the review's F12): the row said "key stored", and the request went out with no key, so a
 * provider that needs one answered 401 and the person was told their key was wrong. Now the row says the key is unreadable and to enter
 * it again, and no request is made without it: nothing leaves, and nothing is counted.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { brokerWorld, jsonResponse } from '../support/broker-world.mjs';
import { KEY, M, anotherDocument, input } from '../support/holder.mjs';

const { keyText } = M.words;

const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

/** Damage the stored item of a provider's key: a bit of its ciphertext is flipped, as a partly cleared or damaged profile might leave it. */
async function damage(w, id = w.record.id) {
  const record = await w.db.get('vault', 'record');
  const name = `key:${id}`;
  const ct = new Uint8Array(record.items[name].ct);
  ct[0] ^= 0xff;
  record.items[name] = { ...record.items[name], ct };
  await w.db.put('vault', 'record', record);
}

const CHAT = { path: '/chat/completions', method: 'POST', body: '{"model":"m","messages":[]}', headers: [['content-type', 'application/json']] };

describe('a key that is stored and cannot be opened', () => {
  it('is said to be unreadable in the list, not "stored"; a healthy key and a provider with no key are not', async () => {
    const w = await world({ handler: () => jsonResponse({}) });
    const healthy = await w.store.save(input({ name: 'Healthy' }), 'sk-healthy-KEY-1234567890');
    const none = await w.store.save(input({ name: 'Keyless', kind: 'local-server', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' }));
    await damage(w);
    const summaries = await w.store.summaries();
    const of = (id) => summaries.find((s) => s.id === id);
    assert.deepEqual([of(w.record.id).hasKey, of(w.record.id).keyUnreadable], [true, true]);
    assert.equal(of(healthy.record.id).hasKey, true);
    assert.ok(!('keyUnreadable' in of(healthy.record.id)), 'a key that opens says nothing more');
    assert.equal(of(none.record.id).hasKey, false);
    assert.ok(!('keyUnreadable' in of(none.record.id)));
    // The same over the port: an app is told, and can tell its own person.
    const client = await w.connect();
    const listed = (await client.call({ op: 'list' })).result;
    assert.equal(listed.find((s) => s.id === w.record.id).keyUnreadable, true);
  });

  it('is refused when a request would carry it: nothing leaves, and nothing is counted against the app', async () => {
    const w = await world({ handler: () => jsonResponse({}) });
    await damage(w);
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, ...CHAT });
    assert.equal(result.error.code, 'key-unreadable');
    assert.match(result.error.message, /cannot be read.*enter it again/i);
    assert.equal(result.head, undefined);
    assert.equal(w.fetchStub.calls.length, 0, 'no request was made, with a key or without one');
    assert.equal((await w.budget.usage('agent')).used, 0, 'and it was not counted');
  });

  it('is refused by models, test and probe as well, each in its own shape', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'm' }] }), record: { model: 'm' } });
    await damage(w);
    const client = await w.connect();
    for (const op of ['models', 'test']) {
      const result = (await client.call({ op, provider: w.record.id })).result;
      assert.equal(result.ok, false, op);
      assert.equal(result.error.kind, 'key-unreadable', op);
      assert.match(result.error.message, /enter it again/i, op);
    }
    assert.deepEqual((await client.call({ op: 'probe', provider: w.record.id })).result, { contextTokens: null, how: null });
    assert.equal(w.fetchStub.calls.length, 0, 'none of the three asked anybody');
  });

  it('the store\'s own reads say so: key() throws KeyUnreadable, where a provider with no key gives an empty one', async () => {
    const w = await world({ handler: () => jsonResponse({}) });
    const keyless = await w.store.save(input({ name: 'Keyless', kind: 'local-server', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' }));
    await damage(w);
    await assert.rejects(() => w.store.key(w.record.id), (e) => e instanceof M.store.KeyUnreadable && /enter it again/i.test(e.message));
    assert.equal(await w.store.key(keyless.record.id), '');
  });

  it('a provider that was saved with no key is not one with an unreadable key: its requests go out as they always did', async () => {
    const w = await world({ key: null, record: { kind: 'local-server', serverKind: 'ollama', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' }, handler: () => jsonResponse({ ok: true }) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, ...CHAT });
    assert.equal(result.head.status, 200);
    assert.equal(w.fetchStub.calls.length, 1);
    assert.equal(w.fetchStub.calls[0].headers.authorization, undefined);
    const listed = (await client.call({ op: 'list' })).result[0];
    assert.equal(listed.hasKey, false);
    assert.ok(!('keyUnreadable' in listed));
  });

  it('is put right by entering the key again: the row is a key stored, and the request goes out with the new one', async () => {
    const w = await world({ handler: () => jsonResponse({ ok: true }) });
    await damage(w);
    const client = await w.connect();
    assert.equal((await client.fetch({ provider: w.record.id, ...CHAT })).error.code, 'key-unreadable');
    await w.store.setKey(w.record.id, 'sk-new-KEY-abcdefghijklmnop');
    const listed = (await client.call({ op: 'list' })).result[0];
    assert.equal(listed.hasKey, true);
    assert.ok(!('keyUnreadable' in listed));
    const result = await client.fetch({ provider: w.record.id, ...CHAT });
    assert.equal(result.head.status, 200);
    assert.equal(w.fetchStub.calls.at(-1).headers.authorization, 'Bearer sk-new-KEY-abcdefghijklmnop');
  });

  it('is what a vault whose key is gone leaves (site data cleared in part): every stored key is unreadable, a provider with none is not', async () => {
    const w = await world({ handler: () => jsonResponse({}) });
    const keyless = await w.store.save(input({ name: 'Keyless', kind: 'local-server', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' }));
    await w.db.delete('keys', 'device');
    // A document that opens after the damage has nothing cached.
    const fresh = anotherDocument(w);
    try {
      const summaries = await fresh.store.summaries();
      assert.equal(summaries.find((s) => s.id === w.record.id).keyUnreadable, true);
      assert.ok(!('keyUnreadable' in summaries.find((s) => s.id === keyless.record.id)));
      await assert.rejects(() => fresh.store.key(w.record.id), M.store.KeyUnreadable);
      assert.equal(await fresh.store.key(keyless.record.id), '');
    } finally {
      fresh.store.close();
    }
  });

  it('never puts the key, or a word of what damaged it, in what an app is told', async () => {
    const w = await world({ handler: () => jsonResponse({}) });
    await damage(w);
    const client = await w.connect();
    await client.fetch({ provider: w.record.id, ...CHAT });
    await client.call({ op: 'models', provider: w.record.id });
    await client.call({ op: 'list' });
    assert.ok(!client.everything().includes(KEY.slice(0, 12)));
  });
});

describe('what a row says about a key', () => {
  it('a key that opens is "key stored", none is "no key", and one that cannot be opened says to enter it again', () => {
    assert.equal(keyText({ hasKey: true }), 'key stored');
    assert.equal(keyText({ hasKey: false }), 'no key');
    assert.equal(keyText({ hasKey: true, keyUnreadable: true }), 'key unreadable: re-enter it');
  });
});
