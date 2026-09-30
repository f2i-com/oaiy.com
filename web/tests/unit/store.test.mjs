/**
 * The records and keys (web/providers/src/records.ts, store.ts): what a record may be, that no key is ever in one, and that a stored
 * key is not sent somewhere new without being typed again.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { KEY, M, anotherDocument, input, makeHolder, sleep } from '../support/holder.mjs';

const json = (value) => JSON.stringify(value);

describe('what a record may be (records.ts)', () => {
  const ok = (over = {}) => M.records.validateRecord(input(over), 'p1');
  const bad = (over) => {
    const v = ok(over);
    assert.equal(v.ok, false, JSON.stringify(over));
    return v.errors;
  };

  it('a sound input is a record: the address normalised, the auth from the dialect, via broker', () => {
    const v = ok({ baseUrl: 'https://api.openai.com', model: ' gpt-x ', extraHeaders: [{ name: 'openai-organization', value: ' org-1 ' }, { name: 'X-Title', value: '' }] });
    assert.equal(v.ok, true);
    assert.deepEqual(v.record, {
      v: 1,
      id: 'p1',
      name: 'Work OpenAI',
      dialect: 'openai',
      baseUrl: 'https://api.openai.com/v1',
      auth: 'bearer',
      caps: ['chat'],
      kind: 'external',
      via: 'broker',
      model: 'gpt-x',
      preset: 'openai',
      extraHeaders: [{ name: 'OpenAI-Organization', value: 'org-1' }],
    });
    const a = M.records.validateRecord(input({ dialect: 'anthropic', baseUrl: 'https://api.anthropic.com/v1/messages' }), 'p2');
    assert.equal(a.record.auth, 'x-api-key');
    assert.equal(a.record.baseUrl, 'https://api.anthropic.com/v1');
  });

  it('a name is one short line; an address is one a provider can be called at; a service on the internet is https', () => {
    assert.ok(bad({ name: '' }).name);
    assert.ok(bad({ name: 'x'.repeat(81) }).name);
    assert.ok(bad({ name: 'two\nlines' }).name);
    assert.ok(bad({ name: 5 }).name);
    for (const baseUrl of ['', 'not a url', 'ftp://x/v1', 'javascript:alert(1)', 'https://u:p@api.example/v1', 'https://api.example/v1?x=1', 'https://api.example/v1#f', undefined, 5, {}]) assert.ok(bad({ baseUrl }).baseUrl, String(baseUrl));
    assert.match(bad({ baseUrl: 'http://api.example.com/v1' }).baseUrl, /https/);
    assert.equal(ok({ baseUrl: 'http://localhost:11434/v1' }).ok, true, 'plain http to this computer is fine');
    assert.equal(ok({ baseUrl: 'http://192.168.1.5:8000/v1', kind: 'local-server' }).ok, true, 'and a server on this network is one the person runs');
  });

  it('a dialect, a kind, a server kind, capabilities and a preset are from their lists', () => {
    assert.ok(bad({ dialect: 'gemini' }).dialect);
    assert.ok(bad({ dialect: undefined }).dialect);
    assert.ok(bad({ kind: 'browser-engine' }).kind, 'the engine\'s record is made by the engine, not by a form');
    assert.ok(bad({ kind: 'gateway' }).kind);
    assert.ok(bad({ serverKind: 'ollama' }).serverKind, 'a service on the internet has no server kind');
    assert.ok(bad({ kind: 'local-server', serverKind: 'nginx', baseUrl: 'http://localhost:1/v1' }).serverKind);
    assert.ok(bad({ caps: ['chat', 'root'] }).caps);
    assert.ok(bad({ caps: 'chat' }).caps);
    assert.ok(bad({ preset: 'Has Space' }).preset);
    assert.deepEqual(ok({ caps: ['tools', 'chat'] }).record.caps, ['chat', 'tools']);
    assert.equal(ok({ kind: 'local-server', serverKind: 'ollama', baseUrl: 'http://localhost:11434' }).record.serverKind, 'ollama');
  });

  it('extra headers are the four allowed names, once each, short and on one line', () => {
    assert.ok(bad({ extraHeaders: [{ name: 'Authorization', value: 'x' }] }).extraHeaders);
    assert.ok(bad({ extraHeaders: [{ name: 'Host', value: 'evil' }] }).extraHeaders);
    assert.ok(bad({ extraHeaders: [{ name: 'X-Title', value: 'a' }, { name: 'x-title', value: 'b' }] }).extraHeaders);
    assert.ok(bad({ extraHeaders: [{ name: 'X-Title', value: 'a\r\nInjected: 1' }] }).extraHeaders);
    assert.ok(bad({ extraHeaders: [{ name: 'X-Title', value: 'a'.repeat(513) }] }).extraHeaders);
    assert.ok(bad({ extraHeaders: 'X-Title: a' }).extraHeaders);
    assert.deepEqual(ok({ extraHeaders: [{ name: 'http-referer', value: 'https://app.example' }] }).record.extraHeaders, [{ name: 'HTTP-Referer', value: 'https://app.example' }]);
    assert.equal(ok({ extraHeaders: [] }).record.extraHeaders, undefined);
  });

  it('a model, a context window and a number of agents are what they claim to be', () => {
    assert.ok(bad({ model: 'x'.repeat(201) }).model);
    assert.ok(bad({ model: 'a\nb' }).model);
    assert.equal(ok({ model: '' }).record.model, undefined);
    assert.equal(ok({ contextTokens: '32768', parallelAgents: 3 }).record.contextTokens, 32768);
    for (const contextTokens of [1023, 10_000_001, 'lots', 1.5, -5]) assert.ok(bad({ contextTokens }).contextTokens, String(contextTokens));
    for (const parallelAgents of [0, 17, 'x']) assert.ok(bad({ parallelAgents }).parallelAgents, String(parallelAgents));
  });

  it('an id is `p_` and twelve hex digits, and a record\'s host is where its key would go', () => {
    assert.match(M.records.newRecordId((n) => new Uint8Array(n).fill(171)), /^p_abababababab$/);
    assert.equal(M.records.hostOf('http://localhost:11434/v1'), 'localhost:11434');
    assert.equal(M.records.hostOf('junk'), '');
  });

  it('moving where a key goes is a different address or a different dialect, not a rename or a model', () => {
    const base = ok().record;
    assert.equal(M.records.movesKey(base, { ...base, name: 'Other', model: 'm', caps: ['chat', 'tools'] }), false);
    assert.equal(M.records.movesKey(base, { ...base, baseUrl: 'https://evil.example/v1' }), true);
    assert.equal(M.records.movesKey(base, { ...base, baseUrl: 'https://api.openai.com/v2' }), true);
    assert.equal(M.records.movesKey(base, { ...base, dialect: 'anthropic' }), true);
  });
});

describe('the store', () => {
  it('a saved provider is listed without its key, and the key is in the vault under key:<id> and nowhere else', async () => {
    const h = makeHolder();
    const saved = await h.store.save(input(), KEY);
    assert.equal(saved.ok, true);
    const id = saved.record.id;
    assert.match(id, /^p_[0-9a-f]{12}$/);
    assert.equal(await h.store.key(id), KEY);
    assert.deepEqual(await h.vault.names(), [`key:${id}`]);
    const summaries = await h.store.summaries();
    assert.deepEqual(summaries, [{ id, name: 'Work OpenAI', dialect: 'openai', host: 'api.openai.com', caps: ['chat'], model: null, hasKey: true, kind: 'external', locked: false }]);
    for (const view of [json(summaries), json(await h.store.list()), json([...h.idb.raw('records').values()])]) {
      assert.ok(!view.includes(KEY) && !view.includes(KEY.slice(0, 10)) && !view.includes(KEY.slice(-10)));
    }
    assert.equal(await h.store.hasKey(id), true);
  });

  it('an invalid input saves nothing and says what is wrong by field; an id that is not a provider is refused', async () => {
    const h = makeHolder();
    const refused = await h.store.save(input({ baseUrl: 'ftp://x' }), KEY);
    assert.equal(refused.ok, false);
    assert.equal(refused.code, 'invalid');
    assert.ok(refused.errors.baseUrl);
    assert.equal((await h.store.list()).length, 0);
    assert.deepEqual(await h.vault.names(), [], 'and the key was not kept either');
    const nobody = await h.store.save({ ...input(), id: 'p_nothere000000' });
    assert.equal(nobody.ok, false);
    assert.ok(nobody.errors.id);
  });

  it('an edit that does not move the key keeps it without asking; one that does needs the key typed again', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input(), KEY);
    const renamed = await h.store.save({ ...input({ name: 'Renamed', model: 'gpt-x' }), id: record.id });
    assert.equal(renamed.ok, true);
    assert.equal(await h.store.key(record.id), KEY, 'the stored key is kept');

    const moved = await h.store.save({ ...input({ baseUrl: 'https://evil.example/v1' }), id: record.id });
    assert.equal(moved.ok, false);
    assert.equal(moved.code, 'retype-key');
    assert.equal((await h.store.get(record.id)).baseUrl, 'https://api.openai.com/v1', 'nothing changed');

    const dialect = await h.store.save({ ...input({ dialect: 'anthropic' }), id: record.id });
    assert.equal(dialect.code, 'retype-key');

    const retyped = await h.store.save({ ...input({ baseUrl: 'https://proxy.example/v1' }), id: record.id }, 'sk-new-typed-key-1234567890');
    assert.equal(retyped.ok, true);
    assert.equal((await h.store.get(record.id)).baseUrl, 'https://proxy.example/v1');
    assert.equal(await h.store.key(record.id), 'sk-new-typed-key-1234567890');
  });

  it('a record with no key can be moved freely: there is no key to redirect', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input({ kind: 'local-server', baseUrl: 'http://localhost:11434/v1' }));
    const moved = await h.store.save({ ...input({ kind: 'local-server', baseUrl: 'http://localhost:1234/v1' }), id: record.id });
    assert.equal(moved.ok, true);
  });

  it('a key that is removed is gone, and removing a provider removes its key', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input(), KEY);
    await h.store.setKey(record.id, '');
    assert.equal(await h.store.hasKey(record.id), false);
    await h.store.setKey(record.id, KEY);
    await h.store.remove(record.id);
    assert.deepEqual(await h.store.list(), []);
    assert.deepEqual(await h.vault.names(), []);
    await assert.rejects(h.store.setKey('p_nothere000000', KEY), /no such provider/);
  });

  it('a record that cannot be written leaves no key behind for a provider that does not exist', async () => {
    const h = makeHolder();
    await h.vault.ready();
    h.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'records' });
    await assert.rejects(h.store.save(input(), KEY), /quota/);
    assert.deepEqual(await h.vault.names(), []);
    assert.deepEqual(await h.store.list(), []);
  });

  it('a record that cannot be written on an edit puts the old key back: a new key is not left pointing at the old address', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input(), KEY);
    h.idb.fail('put', new DOMException('quota', 'QuotaExceededError'), { store: 'records' });
    await assert.rejects(h.store.save({ ...input({ baseUrl: 'https://proxy.example/v1' }), id: record.id }, 'sk-new-typed-key-1234567890'), /quota/);
    assert.equal(await h.store.key(record.id), KEY, 'the key it had');
    assert.equal((await h.store.get(record.id)).baseUrl, 'https://api.openai.com/v1');
  });

  it('the model is the one field a page can change: setModel is one transaction, and an unknown provider is refused', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input({ name: 'Zed' }), KEY);
    let changes = 0;
    h.store.onChange(() => changes++);
    await h.store.rememberModels(record.id, ['gpt-x', 'gpt-y']);
    assert.equal(await h.store.setModel(record.id, 'gpt-x', 'flows'), 'ok');
    assert.equal(changes, 1);
    assert.equal(await h.store.setModel(record.id, 'gpt-x', 'flows'), 'ok', 'the same model again');
    assert.equal(changes, 1, 'is no change');
    assert.equal(await h.store.setModel('p_nothere000000', 'gpt-x', 'flows'), 'no-provider');
    // Made at the same moment as an edit in the other document: neither is lost.
    const other = anotherDocument(h);
    await Promise.all([other.store.setModel(record.id, 'gpt-y', 'agent'), h.store.save({ ...input({ name: 'Renamed' }), id: record.id })]);
    const after = await h.store.get(record.id);
    assert.equal(after.name, 'Renamed');
    assert.ok(['gpt-x', 'gpt-y'].includes(after.model ?? 'gpt-x'));
  });

  it('an app may choose only a model the provider listed (or the record already has); the owner may name any; the record says who chose', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input({ model: 'configured' }), KEY);
    assert.equal(await h.store.setModel(record.id, 'gpt-x', 'agent'), 'unknown-model', 'nothing listed yet');
    assert.equal(await h.store.setModel(record.id, 'configured', 'agent'), 'ok', 'the record\'s own model is always allowed');
    assert.equal((await h.store.get(record.id)).modelChosenBy, 'agent');
    await h.store.rememberModels(record.id, ['gpt-x', '', 5, 'a'.repeat(300), 'gpt-x', ...Array.from({ length: 600 }, (_, i) => `m${i}`)]);
    const kept = await h.db.get('meta', `models:${record.id}`);
    assert.equal(kept.length, 500, 'at most 500 are kept');
    assert.ok(!kept.includes('') && !kept.includes(5) && kept.every((m) => m.length <= 200) && kept.filter((m) => m === 'gpt-x').length === 1);
    assert.equal(await h.store.setModel(record.id, 'gpt-x', 'flows'), 'ok');
    assert.equal((await h.store.get(record.id)).modelChosenBy, 'flows');
    assert.equal(await h.store.setModel(record.id, 'Security notice: re-enter your key', 'flows'), 'unknown-model');
    assert.equal((await h.store.get(record.id)).model, 'gpt-x', 'nothing changed');
    // The owner names any model, and it is no longer marked as an app's.
    assert.equal(await h.store.setModel(record.id, 'my-own-model'), 'ok');
    const owner = await h.store.get(record.id);
    assert.deepEqual([owner.model, owner.modelChosenBy], ['my-own-model', undefined]);
  });

  it('an edit that leaves an app\'s model alone keeps the mark; one that changes it is the owner\'s; removing the provider removes what was listed', async () => {
    const h = makeHolder();
    const { record } = await h.store.save(input(), KEY);
    await h.store.rememberModels(record.id, ['gpt-x']);
    await h.store.setModel(record.id, 'gpt-x', 'agent');
    await h.store.save({ ...input({ name: 'Renamed', model: 'gpt-x' }), id: record.id });
    assert.equal((await h.store.get(record.id)).modelChosenBy, 'agent');
    await h.store.save({ ...input({ name: 'Renamed', model: 'typed-by-owner' }), id: record.id });
    assert.equal((await h.store.get(record.id)).modelChosenBy, undefined);
    await h.store.remove(record.id);
    assert.equal(await h.db.get('meta', `models:${record.id}`), undefined);
  });

  it('a gateway mirror is never one a page can pick a model for (the holder does not hold it)', async () => {
    const h = makeHolder();
    h.idb.raw('records'); // the store exists once opened
    await h.store.list();
    await h.db.put('records', 'gw', { v: 1, id: 'gw', name: 'Mirror', dialect: 'openai', baseUrl: 'https://x.example/v1', auth: 'bearer', caps: ['chat'], kind: 'external', via: 'gateway' });
    assert.equal(await h.store.setModel('gw', 'm'), 'no-provider');
  });

  it('other documents of the origin are told, over a BroadcastChannel, when the list changes', async () => {
    const one = new BroadcastChannel('oaiy-providers');
    const a = makeHolder({ channel: one });
    const other = anotherDocument(a);
    let heard = 0;
    const two = new BroadcastChannel('oaiy-providers');
    const listener = M.store.createStore(other.db, other.vault, { random: a.random, channel: two });
    listener.onChange(() => heard++);
    await a.store.save(input());
    await sleep(60);
    assert.equal(heard, 1);
    a.store.close();
    listener.close();
  });
});
