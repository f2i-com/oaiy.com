/**
 * The holder's port protocol, over a real MessageChannel (web/providers/src/protocol.ts, fetcher.ts, test.ts; design 3.2 and 6).
 *
 * Every normative rule of 3.2 has a test here, and the browser test E3 repeats the ones that matter across real origins:
 *   - who may speak: the origin (not `null`, not a look-alike), the source (only the parent), one port, a hello;
 *   - the operations there are (and no other), and that no reply to anything a page sends ever holds the key;
 *   - the redirect control: the address is the record's, the path is one of a fixed list, the headers are the holder's, the answer
 *     of a redirect is refused, an error is scrubbed of the key;
 *   - the stream: head, chunks as they come, end; abort; timeout; too many at once; the budget.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { findInText, needlesFor } from '../e2e/leakscan.mjs';
import { AGENT, FLOWS, APPS, brokerWorld, jsonResponse, streamResponse } from '../support/broker-world.mjs';
import { KEY, M, input, sleep } from '../support/holder.mjs';

const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

// The key, and pieces of it, in every shape a page might be handed one (leakscan.mjs: reversed, base64, hex, character codes, ...).
const NEEDLES = needlesFor(KEY);
const leaks = (text) => findInText(text, NEEDLES);

describe('who may speak', () => {
  it('an allowed page, from its parent, with one port and a hello, is answered with the version and the operations there are', async () => {
    const w = await world();
    const client = await w.connect();
    const hello = client.inbox[0];
    assert.deepEqual(hello, { t: 'hello', v: 1, ops: ['abort', 'fetch', 'list', 'models', 'probe', 'setModel', 'status', 'test', 'ui.open'], app: 'agent' });
    assert.equal(w.broker.connections(), 1);
    const flows = await w.connect({ origin: FLOWS });
    assert.equal(flows.inbox[0].app, 'flows', 'the app is what the ORIGIN says');
  });

  it('a page that is not in the list is not answered: another origin, a look-alike, `null` (a sandboxed page), nothing', async () => {
    const w = await world();
    for (const origin of ['https://evil.example', 'https://agent.example.evil.example', 'https://agent.example:8443', 'http://agent.example', 'null', '', undefined]) {
      assert.equal(await w.connect({ origin }), null, String(origin));
    }
    assert.equal(w.broker.connections(), 0);
  });

  it('a message from another frame of an allowed page (same origin, not the parent) is not answered', async () => {
    const w = await world();
    assert.equal(await w.connect({ source: { name: 'a sibling frame' } }), null);
    assert.equal(await w.connect({ source: null }), null);
    assert.equal(await w.connect({ source: undefined }), null);
  });

  it('a hello with no port, two ports, or a body that is not a hello is not answered; an `origin` in the data is not read', async () => {
    const w = await world();
    for (const over of [
      { ports: [] },
      { data: { op: 'list' } },
      { data: { op: 'hello' } },
      { data: 'hello' },
      { data: null },
      { data: { op: 'hello', v: 0 } },
    ]) assert.equal(await w.connect(over), null, JSON.stringify(over.data ?? over.ports));
    const two = [new MessageChannel(), new MessageChannel()];
    assert.equal(w.broker.onWindowMessage({ origin: AGENT, source: w.parent, data: { op: 'hello', v: 1 }, ports: [two[0].port2, two[1].port2] }), false);
    for (const c of two) {
      c.port1.close();
      c.port2.close();
    }
    // The page says it is the Agent; the browser says it is evil.
    assert.equal(await w.connect({ origin: 'https://evil.example', data: { op: 'hello', v: 1, origin: AGENT, app: 'agent' } }), null);
  });

  it('with no list of apps (a deployment that listed none) nobody is answered', async () => {
    const w = await world({ apps: new Map() });
    assert.equal(await w.connect(), null);
  });
});

describe('the operations', () => {
  it('list tells a page the providers, and never the key; status says the vault and the engine', async () => {
    const w = await world({ record: { name: 'Work' } });
    const client = await w.connect();
    const listed = await client.call({ op: 'list' });
    assert.equal(listed.ok, true);
    assert.deepEqual(listed.result, [{ id: w.record.id, name: 'Work', dialect: 'openai', host: 'api.openai.com', caps: ['chat'], model: null, hasKey: true, kind: 'external', locked: false }]);
    const status = await client.call({ op: 'status' });
    assert.deepEqual(status.result, { mode: 'device', locked: false, engine: { state: 'none', model: null, progress: null } });
    assert.deepEqual(leaks(client.everything()), []);
  });

  it('a name that is not an operation is refused, and nothing a page sends makes the holder say the key', async () => {
    const w = await world();
    const client = await w.connect();
    const hostile = [
      'getKey', 'get', 'getSecret', 'secret', 'key', 'reveal', 'export', 'dump', 'create', 'add', 'edit', 'update', 'put', 'save', 'set', 'setKey', 'setBaseUrl',
      'delete', 'remove', 'unlock', 'lock', 'setPassphrase', 'setMode', 'wrappers', 'vault', 'names', 'summaries', 'hello', 'constructor', '__proto__', 'toString', 'eval',
    ];
    for (const op of hostile) {
      const reply = await client.call({ op, provider: w.record.id, id2: 'x', key: 'K', name: `key:${w.record.id}`, names: [`key:${w.record.id}`] });
      assert.equal(reply.ok, false, op);
      assert.equal(reply.error.code, 'unknown-op', op);
    }
    // And the operations that exist, asked for what they do not do.
    for (const message of [
      { op: 'list', includeKey: true, withSecrets: true },
      { op: 'status', reveal: true },
      { op: 'models', provider: w.record.id, key: 'x', includeKey: true },
      { op: 'setModel', provider: w.record.id, model: 'fake-chat', key: KEY, baseUrl: 'https://evil.example/v1', name: 'Renamed' },
      { op: 'ui.open', target: 'manage', edit: w.record.id },
      { op: 'abort', target: 999 },
    ]) await client.call(message);
    assert.deepEqual(leaks(client.everything()), [], 'not one reply or push holds any piece of the key');
    const after = await w.store.get(w.record.id);
    assert.equal(after.baseUrl, 'https://api.openai.com/v1', 'a baseUrl in a message changed nothing');
    assert.equal(after.name, 'Work OpenAI', 'and neither did a name');
    assert.equal(after.model, 'fake-chat', 'the model is the one thing a page can set');
    assert.equal(after.modelChosenBy, 'agent', 'and the record says which app chose it');
  });

  it('a request that is malformed is refused with its id when it has one, and dropped when it has none', async () => {
    const w = await world();
    const client = await w.connect();
    client.raw({ id: 5, op: 'fetch' });
    assert.equal((await client.waitFor((m) => m.id === 5 && 'ok' in m)).error.code, 'bad-request');
    const before = client.inbox.length;
    for (const junk of [null, 'fetch', 5, [], { op: 'list' }, { id: 'x', op: 'list' }]) client.raw(junk);
    await sleep(60);
    assert.equal(client.inbox.length, before, 'nothing to answer to');
  });

  it('ui.open says which of the page\'s two ways to show the providers was asked for; the holder opens nothing', async () => {
    const w = await world();
    const client = await w.connect();
    assert.deepEqual((await client.call({ op: 'ui.open', target: 'pick' })).result, { action: 'pick' });
    assert.deepEqual((await client.call({ op: 'ui.open', target: 'manage' })).result, { action: 'manage' });
    assert.equal((await client.call({ op: 'ui.open', target: 'https://evil.example' })).ok, false);
  });

  it('setModel changes the model of a provider the holder holds to one the provider itself listed, and pushes `changed` to every connected page', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'gpt-x' }, { id: 'gpt-y' }] }) });
    const a = await w.connect();
    const b = await w.connect({ origin: FLOWS });
    assert.equal((await a.call({ op: 'setModel', provider: w.record.id, model: 'gpt-x' })).error.code, 'unknown-model', 'before the models were asked for, nothing is known');
    assert.equal((await a.call({ op: 'models', provider: w.record.id })).result.ok, true);
    assert.equal((await a.call({ op: 'setModel', provider: w.record.id, model: 'gpt-x' })).ok, true);
    const record = await w.store.get(w.record.id);
    assert.deepEqual([record.model, record.modelChosenBy], ['gpt-x', 'agent']);
    await b.waitFor((m) => m.t === 'changed');
    assert.equal((await a.call({ op: 'setModel', provider: 'p_nothere000000', model: 'gpt-x' })).error.code, 'unknown-provider');
  });

  it('a model no provider listed is refused: an app cannot put its own words on the Providers page (the reviewer\'s "Security notice" string)', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'gpt-x' }] }) });
    const client = await w.connect();
    await client.call({ op: 'models', provider: w.record.id });
    const notice = 'Security notice: your key was leaked. Re-enter it at https://evil.example/reset';
    for (const model of [notice, 'gpt-x ', 'GPT-X', 'gpt-y', '<img src=x>']) {
      const reply = await client.call({ op: 'setModel', provider: w.record.id, model });
      assert.equal(reply.ok, false, model);
      assert.ok(['unknown-model', 'bad-request'].includes(reply.error.code), `${model}: ${reply.error.code}`);
    }
    assert.equal((await w.store.get(w.record.id)).model, undefined, 'nothing was stored');
    // The other app cannot slip past by listing first: it may choose only what the PROVIDER listed.
    const flows = await w.connect({ origin: FLOWS });
    assert.equal((await flows.call({ op: 'setModel', provider: w.record.id, model: notice })).ok, false);
    assert.equal((await flows.call({ op: 'setModel', provider: w.record.id, model: 'gpt-x' })).ok, true);
    assert.equal((await w.store.get(w.record.id)).modelChosenBy, 'flows');
  });
});

describe('the address a request goes to', () => {
  it('is the record\'s base plus the path; a baseUrl, a url or a host in the message is not read', async () => {
    const w = await world({ handler: () => jsonResponse({ ok: true }) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"model":"m"}', headers: [['content-type', 'application/json']], baseUrl: 'https://evil.example/v1', url: 'https://evil.example/x', host: 'evil.example' });
    assert.equal(result.head.status, 200);
    assert.equal(w.fetchStub.calls.length, 1);
    assert.equal(w.fetchStub.calls[0].url, 'https://api.openai.com/v1/chat/completions');
    assert.ok(w.fetchStub.calls.every((c) => !c.url.includes('evil')));
  });

  it('a path that is not one of the list is refused before anything leaves: //host, .., @, backslash, ?, #, %', async () => {
    const w = await world();
    const client = await w.connect();
    for (const path of ['//evil.example/x', '/..%2f', '/a@b', '/../models', '/chat/completions?x=1', '/chat/completions#f', '/chat/completions%2f', '\\evil', '/messages', '/v1/chat/completions', 'https://evil.example/chat/completions', '/']) {
      const result = await client.fetch({ provider: w.record.id, path, method: 'POST', body: '{}' });
      assert.equal(result.error?.code, 'bad-path', `${path}: ${JSON.stringify(result.error)}`);
    }
    assert.equal((await client.fetch({ provider: w.record.id, path: '/models', method: 'POST' })).error.code, 'bad-method');
    assert.equal(w.fetchStub.calls.length, 0, 'nothing was sent for any of them');
  });

  it('a query is serialised by the holder; a name or a value that is not plain is refused', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [] }) });
    const client = await w.connect();
    await client.fetch({ provider: w.record.id, path: '/models', method: 'GET', query: [['limit', '10'], ['after_id', 'a b&c=d/../e']] });
    assert.equal(w.fetchStub.calls[0].url, 'https://api.openai.com/v1/models?limit=10&after_id=a+b%26c%3Dd%2F..%2Fe');
    assert.equal((await client.fetch({ provider: w.record.id, path: '/models', method: 'GET', query: [['a b', 'x']] })).error.code, 'bad-query');
    assert.equal(w.fetchStub.calls.length, 1);
  });

  it('a provider that is not there, or is a gateway\'s mirror, is refused', async () => {
    const w = await world();
    await w.db.put('records', 'gw', { v: 1, id: 'gw', name: 'Mirror', dialect: 'openai', baseUrl: 'https://x.example/v1', auth: 'bearer', caps: ['chat'], kind: 'external', via: 'gateway' });
    const client = await w.connect();
    assert.equal((await client.fetch({ provider: 'p_nothere000000', path: '/models', method: 'GET' })).error.code, 'unknown-provider');
    assert.equal((await client.fetch({ provider: 'gw', path: '/models', method: 'GET' })).error.code, 'unknown-provider');
    assert.equal((await client.call({ op: 'models', provider: 'gw' })).error.code, 'unknown-provider');
    assert.equal(w.fetchStub.calls.length, 0);
  });
});

describe('the headers a request carries', () => {
  it('the key, the dialect\'s and the record\'s own are the holder\'s; a page chooses only content-type and accept', async () => {
    const w = await world({ record: { extraHeaders: [{ name: 'OpenAI-Organization', value: 'org-1' }] }, handler: () => jsonResponse({}) });
    const client = await w.connect();
    await client.fetch({
      provider: w.record.id,
      path: '/chat/completions',
      method: 'POST',
      body: '{}',
      headers: [['Content-Type', 'application/json'], ['Accept', 'text/event-stream'], ['Authorization', 'Bearer evil'], ['x-api-key', 'evil'], ['OpenAI-Organization', 'org-evil'], ['Host', 'evil.example'], ['Cookie', 'a=b'], ['Origin', 'https://evil.example']],
    });
    assert.deepEqual(w.fetchStub.calls[0].headers, { 'content-type': 'application/json', accept: 'text/event-stream', authorization: `Bearer ${KEY}`, 'openai-organization': 'org-1' });
  });

  it('Anthropic gets its own key header and the direct-browser header; the request never carries credentials, a referrer or a redirect', async () => {
    const w = await world({ record: { dialect: 'anthropic', baseUrl: 'https://api.anthropic.com/v1', preset: 'anthropic' }, handler: () => jsonResponse({}) });
    const client = await w.connect();
    await client.fetch({ provider: w.record.id, path: '/messages', method: 'POST', body: '{}', headers: [['content-type', 'application/json']] });
    const call = w.fetchStub.calls[0];
    assert.deepEqual(call.headers, { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true', 'x-api-key': KEY });
    assert.equal(call.url, 'https://api.anthropic.com/v1/messages');
    assert.equal(call.redirect, 'manual');
    assert.equal(call.credentials, 'omit');
  });

  it('a header value with a line break is refused, and nothing is sent', async () => {
    const w = await world();
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}', headers: [['content-type', 'a\r\nX-Injected: 1']] });
    assert.equal(result.error.code, 'bad-headers');
    assert.equal(w.fetchStub.calls.length, 0);
  });
});

describe('what comes back', () => {
  it('a stream is a head, chunks as they arrive, then the end; the client can read it as it comes', async () => {
    const w = await world({ handler: () => streamResponse(['data: a\n\n', 'data: b\n\n', 'data: c\n\n'], { gapMs: 40 }) });
    const client = await w.connect();
    const started = Date.now();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"stream":true}' });
    assert.equal(result.head.status, 200);
    assert.equal(result.head.headers.find(([k]) => k === 'content-type')[1], 'text/event-stream');
    assert.equal(result.text, 'data: a\n\ndata: b\n\ndata: c\n\n');
    assert.ok(result.chunks >= 3, `one message per chunk: ${result.chunks}`);
    assert.deepEqual(result.events.map((e) => e.t).filter((t, i, all) => t !== all[i - 1]), ['head', 'chunk', 'end']);
    assert.ok(Date.now() - started >= 70);
  });

  it('a set-cookie header, the encoding of the body and any header that is not a media type or a plain counter are not passed on', async () => {
    const w = await world({ handler: () => new Response('{}', { status: 200, headers: { 'content-type': 'application/json', 'x-request-id': 'r1', 'retry-after': '3', 'set-cookie': 'a=b' } }) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/models', method: 'GET' });
    const names = result.head.headers.map(([k]) => k.toLowerCase());
    assert.ok(names.includes('content-type') && names.includes('retry-after'));
    for (const omitted of ['x-request-id', 'set-cookie', 'content-encoding', 'content-length', 'transfer-encoding']) assert.ok(!names.includes(omitted), omitted);
  });

  it('an error a provider sends is not passed on: its status is kept, its words are ours (the full set of cases is provider-text.test.mjs)', async () => {
    const masked = `${KEY.slice(0, 8)}${'*'.repeat(20)}${KEY.slice(-4)}`;
    const w = await world({ handler: () => jsonResponse({ error: { message: `Incorrect API key provided: ${masked}. Also here it is in full: ${KEY}.`, code: 'invalid_api_key' } }, 401) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.equal(result.head.status, 401);
    assert.doesNotMatch(result.text, /Incorrect API key provided|invalid_api_key/);
    assert.match(result.text, /did not accept this API key \(401\)/);
    assert.deepEqual(leaks(result.text), []);
    assert.deepEqual(leaks(client.everything()), []);
  });

  it('a redirect is not followed and not passed on: nothing else is asked, and the page is told', async () => {
    let asked = 0;
    const w = await world({ handler: () => { asked++; return new Response(null, { status: 302, headers: { location: 'https://evil.example/leak' } }); } });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.equal(result.error.code, 'redirect');
    assert.equal(result.head, undefined, 'no head: the redirect is not an answer');
    assert.equal(asked, 1);
    assert.equal(w.fetchStub.calls.length, 1);
    assert.ok(!client.everything().includes('evil.example'), 'and the place it pointed is not passed on');
  });

  it('a call that gets no answer says why: the server is up and refuses CORS, is down, or the address is blocked as mixed content', async () => {
    const down = await world({ handler: (call) => { throw new TypeError(`Failed to fetch ${call.url}`); }, record: { kind: 'local-server', serverKind: 'ollama', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' } });
    const dc = await down.connect();
    const gone = await dc.fetch({ provider: down.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.equal(gone.error.code, 'network');
    assert.match(gone.error.message, /Nothing answered at http:\/\/localhost:11434/);

    // Up: the plain call fails (CORS), the no-cors probe of the origin answers.
    const up = await world({
      handler: (call) => {
        if (call.mode === 'no-cors') return new Response(null, { status: 200 });
        throw new TypeError('Failed to fetch');
      },
      record: { kind: 'local-server', serverKind: 'ollama', baseUrl: 'http://localhost:11434/v1', preset: 'local-server' },
    });
    const uc = await up.connect();
    const refused = await uc.fetch({ provider: up.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.equal(refused.error.code, 'network');
    assert.match(refused.error.message, /does not allow requests from https:\/\/providers\.example/);
    assert.match(refused.error.message, /OLLAMA_ORIGINS="https:\/\/providers\.example"/);
    const probe = up.fetchStub.calls.find((c) => c.mode === 'no-cors');
    assert.equal(probe.url, 'http://localhost:11434/', 'the probe asks the origin, with nothing of ours');
    assert.equal(probe.credentials, 'omit');
    assert.equal(probe.headers.authorization, undefined);

    const mixed = await world({ handler: () => { throw new TypeError('Failed to fetch'); }, record: { kind: 'local-server', baseUrl: 'http://192.168.1.5:8000/v1', preset: 'local-server' } });
    const mc = await mixed.connect();
    const blocked = await mc.fetch({ provider: mixed.record.id, path: '/models', method: 'GET' });
    assert.match(blocked.error.message, /plain http:\/\/ address.*192\.168\.1\.5:8000/);
    assert.equal(mixed.fetchStub.calls.length, 1, 'no probe is made for mixed content');
  });

  it('an OpenAI service that is up and answers nothing readable is told to check the key', async () => {
    const w = await world({
      handler: (call) => {
        if (call.mode === 'no-cors') return new Response(null, { status: 200 });
        throw new TypeError('Failed to fetch');
      },
    });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.match(result.error.message, /could not read the answer.*check the key/);
  });
});

describe('abort, timeout and how many at once', () => {
  it('abort stops the request upstream and ends the stream with `aborted`', async () => {
    let cancelled = false;
    let signal;
    const w = await world({ handler: (call) => { signal = call.signal; return streamResponse(Array.from({ length: 50 }, (_, i) => `data: ${i}\n\n`), { gapMs: 30, onCancel: () => (cancelled = true) }); } });
    const client = await w.connect();
    const id = client.start({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"stream":true}' });
    await client.waitFor((m) => m.id === id && m.t === 'chunk');
    assert.equal((await client.call({ op: 'abort', target: id })).ok, true);
    const end = await client.waitFor((m) => m.id === id && (m.t === 'error' || m.t === 'end'));
    assert.equal(end.t, 'error');
    assert.equal(end.error.code, 'aborted');
    assert.equal(signal.aborted, true, 'the upstream request was aborted');
    await sleep(30);
    assert.equal(cancelled, true, 'and its body was cancelled');
  });

  it('one page cannot abort the requests of another: an abort names a request of its own connection', async () => {
    const w = await world({ handler: () => streamResponse(Array.from({ length: 40 }, (_, i) => `d${i}`), { gapMs: 25 }) });
    const a = await w.connect();
    const b = await w.connect({ origin: FLOWS });
    const idA = a.start({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    await a.waitFor((m) => m.id === idA && m.t === 'chunk');
    await b.call({ op: 'abort', target: idA });
    await sleep(120);
    assert.ok(!a.inbox.some((m) => m.id === idA && m.t === 'error'), 'it went on');
    await a.call({ op: 'abort', target: idA });
  });

  it('a request that takes longer than its timeout ends with `timeout`', async () => {
    const w = await world({ handler: (call) => new Promise((_, reject) => call.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))) });
    const client = await w.connect();
    const result = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}', timeoutMs: 1000 }, { timeoutMs: 4000 });
    assert.equal(result.error.code, 'timeout');
  });

  it('only eight requests are open at once on one connection', async () => {
    const w = await world({ handler: (call) => new Promise((_, reject) => call.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))) });
    const client = await w.connect();
    const ids = Array.from({ length: 8 }, () => client.start({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' }));
    await sleep(80);
    const ninth = await client.fetch({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{}' });
    assert.equal(ninth.error.code, 'too-many');
    for (const id of ids) await client.call({ op: 'abort', target: id });
  });
});

describe('the budget, where a request leaves', () => {
  it('an app has its requests for the hour, and is told when it has made them; another app is not affected', async () => {
    const w = await world({ handler: () => jsonResponse({ ok: true }) });
    await w.budget.setLimit('agent', 3);
    const agent = await w.connect();
    const flows = await w.connect({ origin: FLOWS });
    for (let i = 0; i < 3; i++) assert.equal((await agent.fetch({ provider: w.record.id, path: '/models', method: 'GET' })).head.status, 200);
    const over = await agent.fetch({ provider: w.record.id, path: '/models', method: 'GET' });
    assert.equal(over.error.code, 'budget');
    assert.ok(over.error.retryAfterMs > 0);
    assert.match(over.error.message, /3 requests/);
    assert.equal(w.fetchStub.calls.length, 3, 'the fourth never left');
    assert.equal((await flows.fetch({ provider: w.record.id, path: '/models', method: 'GET' })).head.status, 200);
  });

  it('models, test and probe are counted too: asking another way is not another allowance', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'm' }] }), record: { model: 'm' } });
    await w.budget.setLimit('agent', 2);
    const client = await w.connect();
    assert.equal((await client.call({ op: 'models', provider: w.record.id })).result.ok, true);
    assert.equal((await client.call({ op: 'models', provider: w.record.id })).result.ok, true);
    const third = (await client.call({ op: 'models', provider: w.record.id })).result;
    assert.equal(third.ok, false);
    assert.equal(third.error.kind, 'budget');
    assert.equal((await client.call({ op: 'test', provider: w.record.id })).result.error.kind, 'budget');
    assert.equal(w.fetchStub.calls.length, 2);
  });

  it('the budget is spent by the request that leaves, not by list or status', async () => {
    const w = await world();
    await w.budget.setLimit('agent', 1);
    const client = await w.connect();
    for (let i = 0; i < 5; i++) await client.call({ op: 'list' });
    await client.call({ op: 'status' });
    assert.deepEqual(await w.budget.usage('agent'), { used: 0, limit: 1, bytes: 0, byteLimit: 64 * 1024 * 1024 });
  });
});

describe('models, test and probe', () => {
  it('models lists the provider\'s models, and a wrong key is an answer, not a failure of the protocol', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'gpt-x', created: 5 }, { id: 'gpt-y', created: 9 }] }) });
    const client = await w.connect();
    const found = (await client.call({ op: 'models', provider: w.record.id })).result;
    assert.equal(found.ok, true);
    assert.deepEqual(found.models.map((m) => m.id), ['gpt-y', 'gpt-x']);
    assert.equal(w.fetchStub.calls[0].headers.authorization, `Bearer ${KEY}`);

    const bad = await world({ handler: () => jsonResponse({ error: { message: `Incorrect API key provided: ${KEY.slice(0, 8)}****${KEY.slice(-4)}.` } }, 401) });
    const bc = await bad.connect();
    const refused = (await bc.call({ op: 'models', provider: bad.record.id })).result;
    assert.equal(refused.ok, false);
    assert.equal(refused.error.kind, 'auth');
    assert.deepEqual(leaks(JSON.stringify(refused)), []);
    assert.deepEqual(leaks(bc.everything()), []);
  });

  it('test says whether the model takes tools (a 400 about tools is "no"), and a model this key may not use is an error', async () => {
    const answers = { tools: () => jsonResponse({ choices: [] }), notools: () => jsonResponse({ error: { message: 'tools are not supported by this model' } }, 400), forbidden: () => jsonResponse({ error: { message: 'no access' } }, 403), missing: () => jsonResponse({ error: { message: 'The model does not exist' } }, 404) };
    const cases = { tools: [true, 'yes'], notools: [true, 'no'], forbidden: [false, undefined], missing: [false, undefined] };
    for (const [name, [ok, tools]] of Object.entries(cases)) {
      const w = await world({ record: { model: 'gpt-x' }, handler: (call) => (call.url.endsWith('/models') ? jsonResponse({ data: [{ id: 'gpt-x' }] }) : answers[name]()) });
      const client = await w.connect();
      const result = (await client.call({ op: 'test', provider: w.record.id })).result;
      assert.equal(result.ok, ok, name);
      assert.equal(result.tools, tools, name);
      const chat = w.fetchStub.calls.find((c) => c.url.endsWith('/chat/completions'));
      assert.deepEqual(JSON.parse(chat.body).tools?.[0]?.function?.name, 'noop', 'one dummy tool');
      assert.equal(JSON.parse(chat.body).max_tokens, 1);
    }
  });

  it('test with no model chosen lists the models and says the tools are not known', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'gpt-x' }] }) });
    const client = await w.connect();
    const result = (await client.call({ op: 'test', provider: w.record.id })).result;
    assert.deepEqual({ ok: result.ok, tools: result.tools, models: result.models.map((m) => m.id) }, { ok: true, tools: 'unknown', models: ['gpt-x'] });
    assert.equal(w.fetchStub.calls.length, 1, 'no chat request without a model');
  });

  it('probe finds the context window a server says it has, and asks only its own origin', async () => {
    const w = await world({
      record: { kind: 'local-server', serverKind: 'ollama', baseUrl: 'http://localhost:11434/v1', model: 'llama3', preset: 'local-server' },
      handler: (call) => (call.url === 'http://localhost:11434/api/ps' ? jsonResponse({ models: [{ name: 'llama3:latest', context_length: 8192 }] }) : jsonResponse({}, 404)),
    });
    const client = await w.connect();
    assert.deepEqual((await client.call({ op: 'probe', provider: w.record.id })).result, { contextTokens: 8192, how: 'the size Ollama loaded the model with (/api/ps)' });
    assert.ok(w.fetchStub.calls.every((c) => c.url.startsWith('http://localhost:11434/')));
  });
});

describe('the whole surface, once more', () => {
  it('a battery of hostile messages over one connection never gets the key back, in any reply, push or stream', async () => {
    const masked = `${KEY.slice(0, 8)}${'*'.repeat(30)}${KEY.slice(-4)}`;
    const w = await world({
      record: { model: 'gpt-x' },
      handler: (call) => (call.url.endsWith('/models') ? jsonResponse({ error: { message: `bad key ${masked} / ${KEY}` } }, 401) : jsonResponse({ error: { message: `echo ${KEY}` } }, 400)),
    });
    const client = await w.connect();
    const id = w.record.id;
    const battery = [
      { op: 'list' }, { op: 'status' }, { op: 'models', provider: id }, { op: 'test', provider: id }, { op: 'probe', provider: id },
      { op: 'fetch', provider: id, path: '/models', method: 'GET' },
      { op: 'fetch', provider: id, path: '/chat/completions', method: 'POST', body: '{}', headers: [['authorization', 'x']] },
      { op: 'getKey', provider: id }, { op: 'get', names: [`key:${id}`] }, { op: 'export' }, { op: 'setKey', provider: id, key: 'x' },
    ];
    for (const message of battery) {
      if (message.op === 'fetch') await client.fetch(message);
      else await client.call(message);
    }
    assert.deepEqual(leaks(client.everything()), []);
  });
});
