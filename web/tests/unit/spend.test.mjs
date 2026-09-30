/**
 * What the holder bounds beyond a request count (design 3.2, threat 6): the size of a request, the bytes an app sends in an hour, a cap
 * on a reply's length where a record sets one, and an hour for the modal's own buttons. These bound VOLUME. They do not bound COST (a
 * request can name any model the key may use), and the Providers page says so; the tests are of the bounds that exist.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { brokerWorld, jsonResponse } from '../support/broker-world.mjs';
import { M, makeHolder } from '../support/holder.mjs';

const MiB = 1024 * 1024;
const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

const chat = (client, w, body) => client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body, headers: [['content-type', 'application/json']] });
const jsonBody = (extra, size = 0) => JSON.stringify({ model: 'm', messages: [{ role: 'user', content: 'x'.repeat(size) }], ...extra });

describe('the size of a request', () => {
  it('a body over the default 1 MiB is refused before anything is sent or counted; one at the limit goes', async () => {
    const w = await world({ handler: () => jsonResponse({ ok: true }) });
    const client = await w.connect();
    const big = await chat(client, w, 'x'.repeat(MiB + 1));
    assert.equal(big.error.code, 'too-large');
    assert.match(big.error.message, new RegExp(`up to ${MiB} bytes`));
    assert.equal(w.fetchStub.calls.length, 0);
    assert.equal((await w.budget.usage('agent')).used, 0, 'and it was not counted');
    assert.equal((await chat(client, w, 'x'.repeat(MiB))).head.status, 200);
  });

  it('a record may raise its own limit (never past the ceiling of 32 MiB), and a record with a lower one is held to it', async () => {
    const roomy = await world({ record: { maxBodyBytes: 2 * MiB }, handler: () => jsonResponse({}) });
    const c1 = await roomy.connect();
    assert.equal((await chat(c1, roomy, 'x'.repeat(MiB + MiB / 2))).head.status, 200);
    assert.equal((await chat(c1, roomy, 'x'.repeat(2 * MiB + 1))).error.code, 'too-large');
    const tight = await world({ record: { maxBodyBytes: 4096 }, handler: () => jsonResponse({}) });
    const c2 = await tight.connect();
    assert.equal((await chat(c2, tight, 'x'.repeat(4097))).error.code, 'too-large');
    assert.equal((await chat(c2, tight, 'x'.repeat(4096))).head.status, 200);
    assert.equal(M.records.validateRecord({ name: 'x', dialect: 'openai', kind: 'external', baseUrl: 'https://api.openai.com/v1', maxBodyBytes: 33 * MiB }, 'p').ok, false, 'a record cannot ask for more than the ceiling');
    assert.equal(M.records.validateRecord({ name: 'x', dialect: 'openai', kind: 'external', baseUrl: 'https://api.openai.com/v1', maxBodyBytes: 100 }, 'p').ok, false);
  });

  it('bytes are counted the way they are sent: text as UTF-8, and what a cap rewrites is what is counted', async () => {
    const w = await world({ handler: () => jsonResponse({}) });
    const client = await w.connect();
    await chat(client, w, JSON.stringify({ s: 'é€😀' })); // 2 + 3 + 4 bytes of text
    const usage = await w.budget.usage('agent');
    assert.equal(usage.bytes, new TextEncoder().encode(JSON.stringify({ s: 'é€😀' })).byteLength);
  });
});

describe('a cap on the length of a reply, put in the request by the holder', () => {
  const sent = (w) => JSON.parse(w.fetchStub.calls.at(-1).body);

  it('OpenAI dialect: no ask gets the cap, a bigger ask is cut to it, a smaller one is kept, both spellings are held', async () => {
    const w = await world({ record: { maxOutputTokens: 100 }, handler: () => jsonResponse({}) });
    const client = await w.connect();
    await chat(client, w, jsonBody({}));
    assert.equal(sent(w).max_tokens, 100);
    await chat(client, w, jsonBody({ max_tokens: 5000 }));
    assert.equal(sent(w).max_tokens, 100);
    await chat(client, w, jsonBody({ max_tokens: 50 }));
    assert.equal(sent(w).max_tokens, 50);
    await chat(client, w, jsonBody({ max_completion_tokens: 999999 }));
    assert.deepEqual([sent(w).max_completion_tokens, sent(w).max_tokens], [100, undefined], 'the spelling the app used is the one that is cut');
    await chat(client, w, jsonBody({ max_tokens: 400, max_completion_tokens: 300 }));
    assert.deepEqual([sent(w).max_tokens, sent(w).max_completion_tokens], [100, 100]);
    for (const odd of ['lots', -5, 0, null, Number.NaN, [1]]) {
      await chat(client, w, jsonBody({ max_tokens: odd }));
      assert.equal(sent(w).max_tokens, 100, `max_tokens ${JSON.stringify(odd)}`);
    }
    assert.equal(sent(w).model, 'm', 'the rest of the request is as it was');
    assert.equal(sent(w).messages[0].role, 'user');
  });

  it('a request for more than one reply (n, best_of) is refused when a cap is set: max_tokens 100 with n 128 is 12,800 tokens, not 100 (the review\'s low 6)', async () => {
    const w = await world({ record: { maxOutputTokens: 100 }, handler: () => jsonResponse({}) });
    const client = await w.connect();
    for (const extra of [{ n: 128 }, { n: 2 }, { n: '128' }, { n: 1.5 }, { n: -1 }, { n: 0 }, { best_of: 20 }, { best_of: '3' }, { max_tokens: 100, n: 128 }, { n: 1, best_of: 8 }, { n: [1] }, { n: {} }]) {
      const result = await chat(client, w, jsonBody(extra));
      assert.equal(result.error?.code, 'bad-body', JSON.stringify(extra));
      assert.match(result.error.message, /more than one reply/, JSON.stringify(extra));
    }
    assert.equal(w.fetchStub.calls.length, 0, 'nothing was sent for any of them');
    for (const extra of [{ n: 1 }, { n: null }, { best_of: 1 }, {}]) {
      await chat(client, w, jsonBody(extra));
      assert.equal(sent(w).max_tokens, 100, JSON.stringify(extra));
    }
    assert.equal(w.fetchStub.calls.length, 4, 'one reply each is sent, capped');
  });

  it('with no cap set, n and best_of are the app\'s own to ask (untouched byte for byte)', async () => {
    const plain = await world({ handler: () => jsonResponse({}) });
    const c = await plain.connect();
    const body = '{"model":"m","n":4,"best_of":4,"max_tokens":50}';
    await chat(c, plain, body);
    assert.equal(plain.fetchStub.calls.at(-1).body, body);
  });

  it('Anthropic dialect: max_tokens is required and is cut to the cap', async () => {
    const w = await world({ record: { dialect: 'anthropic', baseUrl: 'https://api.anthropic.com/v1', preset: 'anthropic', maxOutputTokens: 256 }, handler: () => jsonResponse({}) });
    const client = await w.connect();
    const post = (extra) => client.fetch({ provider: w.record.id, path: '/messages', method: 'POST', body: JSON.stringify({ model: 'm', messages: [], ...extra }) });
    await post({});
    assert.equal(sent(w).max_tokens, 256);
    await post({ max_tokens: 100000 });
    assert.equal(sent(w).max_tokens, 256);
    await post({ max_tokens: 10 });
    assert.equal(sent(w).max_tokens, 10);
  });

  it('a chat body the holder cannot read is refused, not sent uncapped; other paths and records with no cap are untouched byte for byte', async () => {
    const w = await world({ record: { maxOutputTokens: 100 }, handler: () => jsonResponse({}) });
    const client = await w.connect();
    for (const body of ['not json', '[1,2,3]', '"a string"', 'null', '']) {
      const result = await chat(client, w, body);
      assert.equal(result.error?.code, 'bad-body', JSON.stringify(body));
    }
    assert.equal(w.fetchStub.calls.length, 0, 'nothing was sent');
    // An embeddings request has no reply length to cap.
    const raw = ' {"input":"x",   "model":"e"} ';
    await client.fetch({ provider: w.record.id, path: '/embeddings', method: 'POST', body: raw });
    assert.equal(w.fetchStub.calls.at(-1).body, raw);
    const plain = await world({ handler: () => jsonResponse({}) });
    const c = await plain.connect();
    const untouched = ' {"model":"m",  "max_tokens":999999} ';
    await chat(c, plain, untouched);
    assert.equal(plain.fetchStub.calls.at(-1).body, untouched, 'no cap set: not a byte changed');
  });

  it('a cap is a whole number of tokens from 1 to a million, in the record', () => {
    for (const bad of [0, -1, 1.5, 1_000_001, 'lots']) assert.equal(M.records.validateRecord({ name: 'x', dialect: 'openai', kind: 'external', baseUrl: 'https://api.openai.com/v1', maxOutputTokens: bad }, 'p').ok, false, String(bad));
    assert.equal(M.records.validateRecord({ name: 'x', dialect: 'openai', kind: 'external', baseUrl: 'https://api.openai.com/v1', maxOutputTokens: '4096', maxBodyBytes: 262144 }, 'p').record.limits.maxOutputTokens, 4096);
  });
});

describe('the bytes an app may send in an hour', () => {
  it('are counted with the requests: the request that would pass the limit is refused, and says which limit it was', async () => {
    let at = 1_000_000;
    const w = await world({ now: () => at, handler: () => jsonResponse({}) });
    await w.budget.setByteLimit('agent', 5000);
    const client = await w.connect();
    assert.equal((await chat(client, w, 'x'.repeat(2000))).head.status, 200);
    assert.equal((await chat(client, w, 'x'.repeat(2000))).head.status, 200);
    const refused = await chat(client, w, 'x'.repeat(2000));
    assert.equal(refused.error.code, 'budget');
    assert.match(refused.error.message, /sent its 5 KiB for this hour/);
    assert.ok(refused.error.retryAfterMs > 0);
    assert.equal(w.fetchStub.calls.length, 2, 'the third never left');
    assert.deepEqual(await w.budget.usage('agent'), { used: 2, limit: 600, bytes: 4000, byteLimit: 5000 });
    // A smaller one still fits, and a GET (no body) costs a request and no bytes.
    assert.equal((await chat(client, w, 'x'.repeat(900))).head.status, 200);
    assert.equal((await client.fetch({ provider: w.record.id, path: '/models', method: 'GET' })).head.status, 200);
    assert.equal((await w.budget.usage('agent')).bytes, 4900);
    // The hour slides: when the first two have left, the bytes are back.
    at += 60 * 60 * 1000 + 1;
    assert.equal((await chat(client, w, 'x'.repeat(2000))).head.status, 200);
  });

  it('are kept per app, in the database, and a limit of 0 shuts an app out', async () => {
    const h = makeHolder();
    await h.budget.setByteLimit('flows', 0);
    assert.equal((await h.budget.take('flows', 1)).reason, 'bytes');
    assert.equal((await h.budget.take('agent', 1)).ok, true, 'the other app has its own');
    const reloaded = (await import('../support/holder.mjs')).anotherDocument(h);
    assert.equal(await reloaded.budget.byteLimit('flows'), 0);
    assert.equal(await reloaded.budget.byteLimit('agent'), 64 * MiB);
    for (const bad of [-1, 1.5, 5 * 1024 * MiB, '5', NaN]) await assert.rejects(h.budget.setByteLimit('agent', bad), /whole number/, String(bad));
  });

  it('a count taken before bytes were counted (a bare moment) still counts as a request', async () => {
    const h = makeHolder();
    await h.db.put('budget', 'agent', [Date.now() - 1000, [Date.now() - 500, 300]]);
    const usage = await h.budget.usage('agent');
    assert.deepEqual([usage.used, usage.bytes], [2, 300]);
  });
});

describe('the embedded modal\'s buttons count too', () => {
  it('the modal is an app with an hour of its own (60 requests), which a tester wired to it spends: the third call over a limit of two is refused', async () => {
    const h = makeHolder();
    assert.equal(await h.budget.limit('modal'), 60);
    assert.equal(await h.budget.limit('agent'), 600);
    await h.budget.setLimit('modal', 2);
    const record = { v: 1, id: 'p1', name: 'X', dialect: 'openai', baseUrl: 'https://api.openai.com/v1', auth: 'bearer', caps: ['chat'], kind: 'external', via: 'broker' };
    let sent = 0;
    const fetchImpl = async () => {
      sent++;
      return jsonResponse({ data: [{ id: 'm' }] });
    };
    const tester = M.tester.createTester({ fetchImpl, page: h.page, key: async () => 'k', take: async (bytes) => { const t = await h.budget.take('modal', bytes); return t.ok ? { ok: true } : t; } });
    assert.equal((await tester.models(record)).ok, true);
    assert.equal((await tester.models(record)).ok, true);
    const third = await tester.models(record);
    assert.equal(third.ok, false);
    assert.equal(third.error.kind, 'budget');
    assert.match(third.error.message, /2 requests for this hour/);
    assert.equal(sent, 2);
  });
});
