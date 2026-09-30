/**
 * The port protocol (shared/broker/protocol.ts, design 3.2): the operations there are and are not, what a request may
 * carry, and which origins may speak. Every normative rule of 3.2 has a test here or in broker.test.mjs, and the browser
 * checks of E3 repeat the ones that matter across real origins.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const P = await loadTs('shared/broker/protocol.ts');

const ok = (data) => {
  const parsed = P.parseRequest(data);
  assert.equal(parsed.ok, true, JSON.stringify(parsed));
  return parsed;
};
const refused = (data) => {
  const parsed = P.parseRequest(data);
  assert.equal(parsed.ok, false, `accepted: ${JSON.stringify(data)}`);
  return parsed;
};

describe('the operations an app may call', () => {
  it('are exactly these nine, and no other exists', () => {
    assert.deepEqual([...P.OPS], ['abort', 'fetch', 'list', 'models', 'probe', 'setModel', 'status', 'test', 'ui.open']);
  });

  it('none of them returns, edits or redirects a key, or adds, edits or deletes a provider', () => {
    // The names a hostile page would try. None is an operation; each is refused as unknown.
    const hostile = [
      'getKey', 'get', 'getSecret', 'getSecrets', 'secret', 'key', 'reveal', 'export', 'import', 'dump', 'backup', 'read', 'readKey',
      'create', 'add', 'addProvider', 'edit', 'update', 'put', 'save', 'set', 'setKey', 'setBaseUrl', 'setUrl', 'setProvider', 'delete', 'remove', 'deleteProvider',
      'unlock', 'lock', 'setPassphrase', 'changePassphrase', 'setMode', 'wrappers', 'vault', 'store', 'db', 'eval', 'hello', 'constructor', '__proto__', 'toString', 'hasOwnProperty',
      'FETCH', 'Fetch', ' fetch', 'fetch ', 'ui', 'ui.close', 'ui.manage', 'UI.OPEN',
    ];
    for (const op of hostile) {
      const parsed = refused({ id: 1, op });
      assert.equal(parsed.code, 'unknown-op', op);
      assert.equal(parsed.id, 1);
    }
  });

  it('a request with no op, an op that is not a string, or no numeric id is refused', () => {
    assert.equal(refused({ id: 1 }).code, 'unknown-op');
    for (const op of [5, null, {}, ['fetch'], true]) assert.equal(refused({ id: 1, op }).code, 'unknown-op');
    for (const id of [undefined, '1', -1, 1.5, 2 ** 31, NaN, null, {}]) assert.equal(refused({ id, op: 'list' }).id, null, String(id));
    for (const junk of [null, undefined, 'fetch', 5, [], [{ id: 1, op: 'list' }]]) assert.equal(refused(junk).code, 'bad-request');
  });
});

describe('what each operation carries', () => {
  it('list and status carry nothing else: whatever else is sent is not read', () => {
    assert.deepEqual(ok({ id: 3, op: 'list', key: 'K', baseUrl: 'http://evil.example' }).request, { op: 'list' });
    assert.deepEqual(ok({ id: 3, op: 'status', anything: 1 }).request, { op: 'status' });
  });

  it('fetch is rebuilt from its known fields: a baseUrl, a url, a key or an authorization header in it is not carried', () => {
    const parsed = ok({
      id: 7,
      op: 'fetch',
      provider: 'p_1',
      path: '/chat/completions',
      method: 'POST',
      body: '{"model":"m"}',
      headers: [['content-type', 'application/json']],
      baseUrl: 'https://evil.example/v1',
      url: 'https://evil.example/v1/chat/completions',
      key: 'K',
      apiKey: 'K',
      origin: 'https://evil.example',
      redirect: 'follow',
    });
    assert.deepEqual(parsed.request, {
      op: 'fetch',
      provider: 'p_1',
      path: '/chat/completions',
      method: 'POST',
      body: '{"model":"m"}',
      headers: [['content-type', 'application/json']],
      timeoutMs: P.DEFAULT_TIMEOUT_MS,
    });
    const text = JSON.stringify(parsed.request);
    for (const forbidden of ['evil', 'baseUrl', 'apiKey', 'redirect']) assert.ok(!text.includes(forbidden), forbidden);
  });

  it('fetch needs a provider id that is a plain name, a path that is a short string, and GET or POST', () => {
    const base = { id: 1, op: 'fetch', provider: 'p1', path: '/models', method: 'GET' };
    ok(base);
    for (const provider of [undefined, '', 5, 'has space', 'a/b', 'a?b', 'x'.repeat(97), '../x', 'p\u0000', { a: 1 }]) assert.equal(refused({ ...base, provider }).code, 'bad-request', String(provider));
    for (const path of [undefined, 5, {}, 'x'.repeat(65)]) assert.equal(refused({ ...base, path }).code, 'bad-request', String(path));
    for (const method of [undefined, 'get', 'DELETE', 'PUT', 'PATCH', 'CONNECT', 5]) assert.equal(refused({ ...base, method }).code, 'bad-request', String(method));
  });

  it('the path itself is not judged here (it needs the record\'s dialect): a path of any short text passes on to the check that does', () => {
    assert.equal(ok({ id: 1, op: 'fetch', provider: 'p1', path: '//evil.example/x', method: 'GET' }).request.path, '//evil.example/x');
  });

  it('a body is text or bytes, and no larger than the cap', () => {
    const base = { id: 1, op: 'fetch', provider: 'p1', path: '/chat/completions', method: 'POST' };
    ok({ ...base, body: 'text' });
    ok({ ...base, body: new ArrayBuffer(8) });
    ok({ ...base, body: new Uint8Array(8) });
    ok({ ...base, body: null });
    for (const body of [5, {}, [1], true, new Blob(['x']), () => 1]) assert.equal(refused({ ...base, body }).code, 'bad-request', String(body));
    assert.equal(refused({ ...base, body: new ArrayBuffer(P.MAX_BODY_BYTES + 1) }).code, 'bad-request');
    assert.equal(refused({ ...base, body: 'x'.repeat(P.MAX_BODY_BYTES / 3 + 1) }).code, 'bad-request');
  });

  it('a timeout is clamped between one second and ten minutes; a query and headers are short lists of string pairs', () => {
    const base = { id: 1, op: 'fetch', provider: 'p1', path: '/models', method: 'GET' };
    assert.equal(ok({ ...base, timeoutMs: 1 }).request.timeoutMs, P.MIN_TIMEOUT_MS);
    assert.equal(ok({ ...base, timeoutMs: 10 ** 9 }).request.timeoutMs, P.MAX_TIMEOUT_MS);
    assert.equal(ok({ ...base, timeoutMs: 45_500.7 }).request.timeoutMs, 45_500);
    for (const timeoutMs of ['5', NaN, Infinity, {}]) assert.equal(refused({ ...base, timeoutMs }).code, 'bad-request', String(timeoutMs));
    assert.deepEqual(ok({ ...base, query: [['limit', '10']] }).request.query, [['limit', '10']]);
    for (const query of ['a=b', { a: 'b' }, [['a']], [['a', 1]], Array.from({ length: 17 }, (_, i) => [`k${i}`, 'v'])]) assert.equal(refused({ ...base, query }).code, 'bad-request', JSON.stringify(query));
    for (const headers of [{ a: 'b' }, 'a: b', [['a', 1]], [['', 'x']]]) assert.equal(refused({ ...base, headers }).code, 'bad-request', JSON.stringify(headers));
  });

  it('models, test and probe name a provider; setModel names a provider and a plain model; ui.open opens pick or manage; abort names a request', () => {
    for (const op of ['models', 'test', 'probe']) {
      assert.deepEqual(ok({ id: 2, op, provider: 'p1', extra: 1 }).request, { op, provider: 'p1' });
      assert.equal(refused({ id: 2, op }).code, 'bad-request');
      assert.equal(refused({ id: 2, op, provider: 'a b' }).code, 'bad-request');
    }
    assert.deepEqual(ok({ id: 2, op: 'setModel', provider: 'p1', model: 'gpt-x', baseUrl: 'x' }).request, { op: 'setModel', provider: 'p1', model: 'gpt-x' });
    for (const model of [undefined, '', ' x', 'x ', 'a\nb', 'x'.repeat(201), 5, {}]) assert.equal(refused({ id: 2, op: 'setModel', provider: 'p1', model }).code, 'bad-request', String(model));
    assert.deepEqual(ok({ id: 2, op: 'ui.open', target: 'pick' }).request, { op: 'ui.open', target: 'pick' });
    assert.deepEqual(ok({ id: 2, op: 'ui.open', target: 'manage' }).request, { op: 'ui.open', target: 'manage' });
    for (const target of [undefined, 'edit', 'add', 'https://evil.example', '_blank', 5]) assert.equal(refused({ id: 2, op: 'ui.open', target }).code, 'bad-request', String(target));
    assert.deepEqual(ok({ id: 2, op: 'abort', target: 9 }).request, { op: 'abort', target: 9 });
    for (const target of [undefined, '9', -1, 1.5]) assert.equal(refused({ id: 2, op: 'abort', target }).code, 'bad-request', String(target));
  });
});

describe('the hello, and the origins that may send it', () => {
  it('is {op:"hello", v} with a positive whole version; an app name or origin in it is not read', () => {
    assert.deepEqual(P.parseHello({ op: 'hello', v: 1, app: 'flows', origin: 'https://agent.example' }), { op: 'hello', v: 1 });
    for (const data of [null, 'hello', { op: 'hello' }, { op: 'hello', v: 0 }, { op: 'hello', v: 1.5 }, { op: 'hello', v: '1' }, { op: 'list', v: 1 }, [], { op: 'hello', v: 10 ** 9 }]) assert.equal(P.parseHello(data), null, JSON.stringify(data));
  });

  it('the list of origins is parsed from the deploy-time meta, and an exact origin is required', () => {
    const map = P.parseAppOrigins('agent=https://agent.example.org flows=https://flows.example.org:8443');
    assert.equal(map.get('https://agent.example.org'), 'agent');
    assert.equal(map.get('https://flows.example.org:8443'), 'flows');
    assert.equal(map.size, 2);
    assert.equal(P.parseAppOrigins('agent=http://agent.web.localhost:5000').get('http://agent.web.localhost:5000'), 'agent', 'plain http on this computer, for tests');
  });

  it('a malformed list allows nobody: it fails closed', () => {
    for (const content of [
      'agent=http://agent.example.org', // plain http off this computer
      'agent=https://agent.example.org/', // not an exact origin
      'agent=https://agent.example.org/path',
      'agent=https://user@agent.example.org',
      'agent=https://agent.example.org flows=oops',
      'agent https://agent.example.org',
      'AGENT=https://agent.example.org',
      '=https://agent.example.org',
      'agent=https://a.example agent2=https://a.example', // one origin, two apps
      'agent=null',
      'agent=*',
      'agent=https://*.example.org',
      'agent=javascript:alert(1)',
      'agent=data:text/html,x',
    ]) {
      assert.equal(P.parseAppOrigins(content).size, 0, content);
    }
    for (const notText of [null, undefined, 5, {}]) assert.equal(P.parseAppOrigins(notText).size, 0);
    assert.equal(P.parseAppOrigins('').size, 0);
  });

  it('a message is answered only from an origin in the list: not null, not empty, not a look-alike', () => {
    const map = P.parseAppOrigins('agent=https://agent.example.org flows=https://flows.example.org');
    assert.equal(P.appForOrigin(map, 'https://agent.example.org'), 'agent');
    assert.equal(P.appForOrigin(map, 'https://flows.example.org'), 'flows');
    for (const origin of ['null', '', 'https://agent.example.org.evil.example', 'https://evil.example', 'http://agent.example.org', 'https://agent.example.org:444', 'https://AGENT.example.org', 'https://sub.agent.example.org', undefined, null, 5]) {
      assert.equal(P.appForOrigin(map, origin), null, String(origin));
    }
    assert.equal(P.appForOrigin(new Map(), 'https://agent.example.org'), null, 'with no list nobody is let in');
  });
});
