/**
 * None of a provider's error text reaches an app (design 6; web/providers/src/fixed.ts).
 *
 * A provider's error can quote the key it refused, and an app can steer parts of what it echoes. Removing the key from that text makes the
 * text depend on the key, and output that depends on the key is a channel: an app that chooses the echo reads, from what is removed, which
 * pieces the key holds. So an app is given the status and wording that are a function of the status ALONE, whatever the provider says, and
 * the tests here are of that: the same bytes for any provider text and for any key.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { brokerWorld, jsonResponse } from '../support/broker-world.mjs';
import { KEY, M } from '../support/holder.mjs';

const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

const reversed = (s) => [...s].reverse().join('');
const forms = (key) => [key, reversed(key), Buffer.from(key).toString('base64'), Buffer.from(key).toString('hex'), encodeURIComponent(key), key.match(/.{1,6}/g).join(' | '), key.toLowerCase(), key.toUpperCase()];
/** Every 8-character piece of a key, in every form: a leak of any of them is a leak. */
const pieces = (key) => forms(key).flatMap((f) => (f.length >= 8 ? Array.from({ length: f.length - 7 }, (_, i) => f.slice(i, i + 8)) : [f])).filter((p) => p.length >= 8);
const leaks = (text, key = KEY) => pieces(key).filter((p) => text.includes(p));

const STATUSES = [400, 401, 402, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504];

/** What a hostile or careless provider might put in an error answer, given the key it was sent. */
const texts = (key) => ({
  echoAuthorization: `{"error":{"message":"You sent: Authorization: Bearer ${key}"}}`,
  plain: `Incorrect API key provided: ${key}.`,
  masked: `Incorrect API key provided: ${key.slice(0, 8)}${'*'.repeat(20)}${key.slice(-4)}.`,
  reversed: `{"error":{"message":"${reversed(key)}"}}`,
  base64: `{"error":{"message":"${Buffer.from(key).toString('base64')}"}}`,
  hex: `{"error":{"message":"${Buffer.from(key).toString('hex')}"}}`,
  split: `{"error":{"message":"${key.match(/.{1,5}/g).join(' - ')}"}}`,
  markup: `<img src=x onerror="window.__x=1"><script>alert('${key}')</script><b>${key}</b>`,
  reset: '{"error":{"message":"Security notice: your key was leaked. Re-enter it at https://evil.example/reset"}}',
  empty: '',
});

describe('an app gets the status and fixed wording, never the provider\'s words', () => {
  it('for every status and every kind of echo, the answer is the same bytes: the status, a fixed reason phrase, a fixed JSON body', async () => {
    const byStatus = new Map();
    for (const status of STATUSES) {
      for (const [what, text] of Object.entries(texts(KEY))) {
        const w = await world({
          handler: () =>
            new Response(text, {
              status,
              statusText: KEY,
              headers: { 'content-type': 'text/html; charset=' + KEY, 'x-echo': KEY, 'www-authenticate': `Bearer realm="${KEY}"`, 'x-request-id': KEY, 'retry-after': 'soon', 'set-cookie': `a=${KEY}` },
            }),
        });
        const client = await w.connect();
        const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
        assert.equal(result.head.status, status, `${status} ${what}`);
        assert.ok(!result.head.statusText.includes(KEY), `${status} ${what}: the reason phrase is ours`);
        assert.deepEqual(result.head.headers, [['content-type', 'application/json']], `${status} ${what}: only a fixed content type (retry-after that is not a number is not passed)`);
        assert.deepEqual(leaks(client.everything()), [], `${status} ${what}: nothing of the key in anything received`);
        assert.ok(!client.everything().includes('evil.example') && !result.text.includes('<img') && !result.text.includes('onerror'), `${status} ${what}: none of the provider's words`);
        const seen = byStatus.get(status);
        if (seen === undefined) byStatus.set(status, { text: result.text, statusText: result.head.statusText });
        else assert.deepEqual({ text: result.text, statusText: result.head.statusText }, seen, `${status}: the answer does not depend on what the provider said (${what})`);
      }
    }
    // The body is JSON in the shape a client of the dialect reads, and says what happened in fixed words.
    const w = await world({ handler: () => new Response('anything', { status: 401 }) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    const body = JSON.parse(result.text);
    assert.equal(body.error.code, 'auth');
    assert.equal(body.error.type, 'provider_error');
    assert.match(body.error.message, /did not accept this API key \(401\)/);
    assert.equal(result.head.statusText, 'Unauthorized');
  });

  it('an Anthropic-dialect record gets the shape its clients read', async () => {
    const w = await world({ record: { dialect: 'anthropic', baseUrl: 'https://api.anthropic.com/v1', preset: 'anthropic' }, handler: () => new Response('{"error":"x"}', { status: 429 }) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/messages', method: 'POST', body: '{}' });
    const body = JSON.parse(result.text);
    assert.equal(body.type, 'error');
    assert.equal(body.error.type, 'rate-limited');
    assert.match(body.error.message, /refusing requests for now \(429\)/);
  });

  it('what an app reads out of the answer does not depend on the key: two keys, the same text the provider echoes an app-chosen string into, the same bytes', async () => {
    const K1 = 'sk-proj-AAAA1111BBBB2222CCCC3333DDDD4444EEEE5555';
    const K2 = 'sk-proj-ZZZZ9999YYYY8888XXXX7777WWWW6666VVVV5555';
    // The app chooses what is echoed: every four-character window of the alphabet, and pieces of both keys, so a scrub that removed
    // windows of the key would show a different result for each key.
    const alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_';
    const echoed = `${K1} ${K2} ${alphabet} ${alphabet.split('').reverse().join('')} ${Array.from({ length: 64 }, (_, i) => alphabet.slice(i) + alphabet.slice(0, i)).join(' ')}`;
    const answers = [];
    for (const key of [K1, K2, 'short']) {
      const w = await world({ key, handler: () => jsonResponse({ error: { message: `bad: ${echoed}` } }, 400) });
      const client = await w.connect();
      const fetched = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"model":"x"}' });
      const models = (await client.call({ op: 'models', provider: w.record.id })).result;
      const tested = (await client.call({ op: 'test', provider: w.record.id })).result;
      answers.push(JSON.stringify({ text: fetched.text, head: fetched.head, models, tested }));
    }
    assert.equal(answers[0], answers[1], 'a different key, the same answer');
    assert.equal(answers[0], answers[2], 'and a key too short to have been scrubbed at all');
    assert.ok(!answers[0].includes(alphabet.slice(0, 12)), 'and none of the echoed text is in it');
  });

  it('the model list and the connection test say the same: fixed wording, no provider words', async () => {
    const answers = [];
    for (const text of [`Incorrect API key provided: ${KEY}`, 'Security notice: re-enter your key at https://evil.example', '<b>markup</b>']) {
      const w = await world({ record: { model: 'gpt-x' }, handler: () => new Response(JSON.stringify({ error: { message: text } }), { status: 401 }) });
      const client = await w.connect();
      const models = (await client.call({ op: 'models', provider: w.record.id })).result;
      const tested = (await client.call({ op: 'test', provider: w.record.id })).result;
      assert.equal(models.ok, false);
      assert.equal(models.error.kind, 'auth');
      assert.equal(models.error.status, 401);
      assert.doesNotMatch(models.error.message, /evil\.example|markup|Incorrect API key provided|The provider said/);
      assert.deepEqual(leaks(JSON.stringify([models, tested])), []);
      answers.push(JSON.stringify([models, tested]));
    }
    assert.equal(new Set(answers).size, 1, 'the same for every provider text');
  });

  it('a chat request that fails on the test says so in fixed words too, and whether tools are the trouble is one of two fixed answers', async () => {
    const texts2 = ['tools are not supported by this model', `The model gpt-x does not exist ${KEY}`];
    const results = [];
    for (const text of texts2) {
      const w = await world({ record: { model: 'gpt-x' }, handler: (call) => (call.url.endsWith('/models') ? jsonResponse({ data: [{ id: 'gpt-x' }] }) : new Response(JSON.stringify({ error: { message: text } }), { status: 400 })) });
      const client = await w.connect();
      results.push((await client.call({ op: 'test', provider: w.record.id })).result);
    }
    assert.deepEqual([results[0].ok, results[0].tools], [true, 'no']);
    assert.equal(results[1].ok, false);
    assert.equal(results[1].error.kind, 'http');
    assert.doesNotMatch(results[1].error.message, /does not exist|gpt-x/);
    assert.deepEqual(leaks(JSON.stringify(results)), []);
  });

  it('a success passes as the provider sent it (the answer is the point), with only a media type and plain numeric counters among its headers', async () => {
    const w = await world({
      handler: () =>
        new Response('{"choices":[{"message":{"content":"hello"}}]}', {
          status: 200,
          headers: { 'content-type': 'application/json; charset=utf-8', 'x-echo': KEY, 'x-request-id': 'r1', 'retry-after': '7', 'x-ratelimit-remaining-requests': '99', 'x-ratelimit-reset-requests': '1.5s', 'x-ratelimit-limit-tokens': `${KEY}` },
        }),
    });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.equal(result.text, '{"choices":[{"message":{"content":"hello"}}]}');
    assert.deepEqual(result.head.headers.sort(), [['content-type', 'application/json; charset=utf-8'], ['retry-after', '7'], ['x-ratelimit-remaining-requests', '99'], ['x-ratelimit-reset-requests', '1.5s']]);
    assert.equal(result.head.statusText, 'OK');
  });
});

describe('the holder\'s own pages may show a provider\'s words, as text, scrubbed of the key', () => {
  it('a tester built for a page includes the words, and the key is not in them; the port\'s never does', async () => {
    const w = await world({ record: { model: 'gpt-x' }, handler: () => jsonResponse({ error: { message: `Incorrect API key provided: ${KEY.slice(0, 8)}****${KEY.slice(-4)}. Check your billing.` } }, 401) });
    const record = w.record;
    const page = M.tester.createTester({ fetchImpl: w.fetchImpl ?? w.fetchStub.impl, page: w.page, key: async () => KEY, providerText: 'scrubbed' });
    const shown = await page.models(record);
    assert.equal(shown.ok, false);
    assert.match(shown.error.message, /The provider said: “Incorrect API key provided: …\*+…\. Check your billing\.”/);
    assert.deepEqual(leaks(shown.error.message), []);
    const port = M.tester.createTester({ fetchImpl: w.fetchStub.impl, page: w.page, key: async () => KEY });
    assert.doesNotMatch((await port.models(record)).error.message, /The provider said|billing/);
  });

  it('through the shared list: the default omits the words, and `include` scrubs them', async () => {
    const record = { dialect: 'openai', baseUrl: 'https://api.openai.com/v1', auth: 'bearer', kind: 'external', preset: 'openai' };
    const fetchImpl = async () => new Response(JSON.stringify({ error: { message: `bad key ${KEY}` } }), { status: 401 });
    const options = { fetchImpl, page: { protocol: 'https:', origin: 'https://p.example' } };
    const models = await import('../support/load.mjs').then((m) => m.loadTs('shared/providers/models.ts'));
    await assert.rejects(models.listRecordModels(record, KEY, options), (e) => e.kind === 'auth' && !/bad key/.test(e.message));
    await assert.rejects(models.listRecordModels(record, KEY, { ...options, providerText: 'include' }), (e) => /bad key …/.test(e.message) && leaks(e.message).length === 0);
  });
});
