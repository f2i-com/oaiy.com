/**
 * A model id from a provider's list is text the Providers page (where a key is typed) shows, and an app may choose one with `setModel`
 * (the review's low 5): an id with a bidirectional override, a zero-width or a control character could make that page read as something
 * else (a "Security notice", an address). Such an id is not a model: it is left out of the list when the list is read, refused by
 * `setModel`, not kept for an app to choose, and not a name an owner may type. An id is also up to 200 characters, at list time.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { brokerWorld, jsonResponse } from '../support/broker-world.mjs';
import { M } from '../support/holder.mjs';

const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

const BAD = {
  'a right-to-left override': 'gpt-x‮elbuod',
  'a left-to-right override': 'gpt-‭x',
  'a bidi isolate': 'gpt-⁦x⁩',
  'a left-to-right mark': 'gpt-x‎',
  'an Arabic letter mark': 'gpt-x؜',
  'a zero-width space': 'gpt​-x',
  'a zero-width joiner': 'gpt‍-x',
  'a word joiner': 'gpt⁠-x',
  'a byte order mark': '﻿gpt-x',
  'a soft hyphen': 'gpt­-x',
  'a line separator': 'gpt -x',
  'a paragraph separator': 'gpt -x',
  'a control character': 'gpt\u0007-x',
  'a line break': 'gpt-x\nSecurity notice',
  'a C1 control': 'gpt\u0085-x',
  'a lone surrogate': 'gpt-\uD800x',
  'a tag character': 'gpt-x\u{E0041}',
  '201 characters': 'm'.repeat(201),
};
const GOOD = ['gpt-4o', 'claude-3-5-sonnet-20241022', 'llama3:latest', 'vendor/model:free', 'モデル-1', 'ünï-cödé', 'm'.repeat(200), 'a b'];

const listing = (ids, extra = {}) => () => jsonResponse({ data: ids.map((id) => ({ id, ...extra })) });

describe('an id that is not fit to show is not a model', () => {
  it('is left out of the list an app is given by models, and a good one is kept (200 characters is the longest)', async () => {
    const w = await world({ handler: listing([...Object.values(BAD), ...GOOD]) });
    const client = await w.connect();
    const result = (await client.call({ op: 'models', provider: w.record.id })).result;
    assert.equal(result.ok, true);
    assert.deepEqual(result.models.map((m) => m.id).sort(), [...GOOD].sort());
  });

  it('is left out of Ollama\'s own list too', async () => {
    const w = await world({
      record: { kind: 'local-server', serverKind: 'ollama', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' },
      handler: (call) => (call.url.endsWith('/api/tags') ? jsonResponse({ models: [...Object.values(BAD), ...GOOD].map((name) => ({ name })) }) : jsonResponse({}, 404)),
    });
    const client = await w.connect();
    const result = (await client.call({ op: 'models', provider: w.record.id })).result;
    assert.equal(result.ok, true, JSON.stringify(result));
    assert.deepEqual(result.models.map((m) => m.id).sort(), [...GOOD].sort());
  });

  it('a label with such characters is not passed on: the model shows as its id', async () => {
    const w = await world({
      handler: () => jsonResponse({ data: [{ id: 'gpt-4o', display_name: 'GPT‮-4o' }, { id: 'gpt-4', display_name: 'GPT four' }, { id: 'gpt-3', display_name: 'l'.repeat(201) }] }),
    });
    const client = await w.connect();
    const models = (await client.call({ op: 'models', provider: w.record.id })).result.models;
    const by = Object.fromEntries(models.map((m) => [m.id, m.label]));
    assert.equal(by['gpt-4o'], undefined);
    assert.equal(by['gpt-4'], 'GPT four');
    assert.equal(by['gpt-3'], undefined);
  });

  it('cannot be chosen by an app, whatever the provider listed: setModel refuses each, and nothing is stored', async () => {
    const w = await world({ handler: listing([...Object.values(BAD), 'gpt-4o']) });
    const client = await w.connect();
    await client.call({ op: 'models', provider: w.record.id });
    for (const [what, id] of Object.entries(BAD)) {
      const reply = await client.call({ op: 'setModel', provider: w.record.id, model: id });
      assert.equal(reply.ok, false, what);
      assert.ok(['unknown-model', 'bad-request'].includes(reply.error.code), `${what}: ${reply.error.code}`);
    }
    assert.equal((await w.store.get(w.record.id)).model, undefined);
    assert.equal((await client.call({ op: 'setModel', provider: w.record.id, model: 'gpt-4o' })).ok, true, 'and the good one still can be');
  });

  it('is refused where the request is read, before anything looks it up (a second layer behind the list and the store)', () => {
    for (const [what, id] of Object.entries(BAD)) {
      const parsed = M.sharedProtocol.parseRequest({ id: 1, op: 'setModel', provider: 'p_abc', model: id });
      assert.equal(parsed.ok, false, what);
      assert.equal(parsed.code, 'bad-request', what);
    }
    assert.equal(M.sharedProtocol.parseRequest({ id: 1, op: 'setModel', provider: 'p_abc', model: 'gpt-4o' }).ok, true);
  });

  it('is not kept for an app to choose from: the store keeps only ids fit to show, even if it is told others', async () => {
    const w = await world();
    await w.store.rememberModels(w.record.id, [...Object.values(BAD), 'gpt-4o']);
    for (const id of Object.values(BAD)) assert.equal(await w.store.setModel(w.record.id, id, 'agent'), 'unknown-model', JSON.stringify(id));
    assert.equal(await w.store.setModel(w.record.id, 'gpt-4o', 'agent'), 'ok');
  });

  it('is not a name the owner may type either, for a model or for a provider\'s name', () => {
    const { validateRecord } = M.records;
    const base = { name: 'Work', dialect: 'openai', kind: 'external', baseUrl: 'https://api.example.com/v1' };
    for (const [what, id] of Object.entries(BAD)) {
      // (A byte order mark at the start is trimmed away as white space, and what is kept is the name without it.)
      if (what === '201 characters' || what === 'a byte order mark') continue;
      assert.equal(validateRecord({ ...base, model: id }, 'p1').ok, false, `model: ${what}`);
      assert.equal(validateRecord({ ...base, name: `Work ${id}` }, 'p1').ok, false, `name: ${what}`);
    }
    assert.equal(validateRecord({ ...base, model: 'm'.repeat(201) }, 'p1').ok, false);
    for (const id of GOOD) assert.equal(validateRecord({ ...base, model: id }, 'p1').ok, true, id);
  });
});
