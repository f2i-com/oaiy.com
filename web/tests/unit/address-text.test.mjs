/**
 * What an app is told of a provider's address is its host, as `list` says, and no more (the review's low 3): a 404 said "Nothing answered
 * at http://127.0.0.1:65477/v1", the whole base URL, and a base path can hold an account or a tenant id. The words an app reads name the
 * origin only. The holder's own pages, which no app can read, still show the whole address to the person who typed it.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { brokerWorld, jsonResponse } from '../support/broker-world.mjs';
import { M } from '../support/holder.mjs';

const TENANT = 'tenant-4471-acme-payroll';
const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

const notFound = () => jsonResponse({ error: { message: 'no such route' } }, 404);
const BASES = {
  'a service on the internet': { baseUrl: `https://api.example.com/v1/accounts/${TENANT}` },
  'a server on this computer': { kind: 'local-server', serverKind: 'other', baseUrl: `http://127.0.0.1:65477/${TENANT}/v1`, preset: 'local-server' },
};

describe('a 404 names the provider\'s origin, not its base path', () => {
  for (const [what, record] of Object.entries(BASES)) {
    it(`${what}: through fetch, models and test, nothing an app receives has the path`, async () => {
      const w = await world({ record: { ...record, model: 'm' }, handler: notFound });
      const client = await w.connect();
      const chat = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
      assert.equal(chat.head.status, 404);
      assert.match(chat.text, /Nothing answered at (https?:\/\/[^/ ]+) \(404\)/, 'it still says where, by origin');
      const models = (await client.call({ op: 'models', provider: w.record.id })).result;
      assert.equal(models.ok, false);
      assert.match(models.error.message, /Nothing answered at (https?:\/\/[^/ ]+) \(404\)/);
      const tested = (await client.call({ op: 'test', provider: w.record.id })).result;
      assert.equal(tested.ok, false);
      assert.ok(!client.everything().includes(TENANT), 'the base path is nowhere in what the app was sent');
      assert.ok(!client.everything().includes('/v1'), 'and no part of the path is');
      const listed = (await client.call({ op: 'list' })).result[0];
      assert.ok(!JSON.stringify(listed).includes(TENANT), 'the list gives the host alone');
    });
  }

  it('and no other failure carries it either: every status, a 200 that is not a model list, and a call that gets no answer, on fetch, models and test', async () => {
    const answers = {
      '400': () => jsonResponse({ error: 'bad' }, 400),
      '401': () => jsonResponse({ error: 'no' }, 401),
      '403': () => jsonResponse({ error: 'no' }, 403),
      '429': () => jsonResponse({ error: 'slow' }, 429),
      '500': () => jsonResponse({ error: 'oops' }, 500),
      '502': () => new Response('<html>Bad gateway</html>', { status: 502 }),
      'a 200 with a body that is not JSON': () => new Response('<html>hello</html>', { status: 200, headers: { 'content-type': 'text/html' } }),
      'a 200 that is JSON but not a model list': () => jsonResponse({ hello: 'world' }),
      'no answer at all': () => {
        throw new TypeError('Failed to fetch');
      },
    };
    for (const [record_what, record] of Object.entries(BASES)) {
      for (const [what, handler] of Object.entries(answers)) {
        const w = await world({ record: { ...record, model: 'm' }, handler });
        const client = await w.connect();
        await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
        await client.fetch({ provider: w.record.id, path: '/models', method: 'GET' });
        await client.call({ op: 'models', provider: w.record.id });
        await client.call({ op: 'test', provider: w.record.id });
        const seen = client.everything();
        assert.ok(!seen.includes(TENANT), `${record_what}, ${what}: the tenant name is in what the app received`);
        // (Words like "usually ending in /v1" are a hint about addresses in general; what must not be there is THIS address with a path.)
        assert.ok(!/api\.example\.com\/|:65477\//.test(seen), `${record_what}, ${what}: the address with a path is in what the app received`);
      }
    }
  });

  it('the holder\'s own pages show the whole address to the person who set it up (scrubbed text, for a page no app can read)', async () => {
    const w = await world({ record: { ...BASES['a service on the internet'], model: 'm' }, handler: notFound });
    const record = await w.store.get(w.record.id);
    const tester = M.tester.createTester({ fetchImpl: w.fetchStub.impl, page: w.page, key: () => w.store.key(record.id), providerText: 'scrubbed' });
    const result = await tester.models(record);
    assert.equal(result.ok, false);
    assert.ok(result.error.message.includes(`https://api.example.com/v1/accounts/${TENANT}`), result.error.message);
  });
});
