/**
 * Where a provider's endpoints are, and what a page may ask of one (shared/providers/endpoints.ts).
 *
 *     npm run test:unit
 *
 * Two halves. The Agent's own (`providerEndpoints`, `providerHeaders`) moved to shared/ unchanged: these tests pin what it
 * answered before it moved. The record's half is the redirect control of design 3.2: a request is the record's base plus ONE
 * path of a fixed list, with the query and the headers built here, and whatever a page sends beyond that is refused or left
 * out.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const E = await loadTs('shared/providers/endpoints.ts');

const record = (over = {}) => ({
  v: 1,
  id: 'p1',
  name: 'Test',
  dialect: 'openai',
  baseUrl: 'https://api.openai.com/v1',
  auth: 'bearer',
  caps: ['chat'],
  kind: 'external',
  via: 'broker',
  ...over,
});

describe('the Agent endpoints, as they were before they moved', () => {
  it('a bare server address gets /v1, and all three saved shapes give the same endpoints', () => {
    const want = { chat: 'http://localhost:11434/v1/chat/completions', models: 'http://localhost:11434/v1/models', ollamaTags: 'http://localhost:11434/api/tags' };
    for (const baseUrl of ['http://localhost:11434', 'http://localhost:11434/v1', 'http://localhost:11434/v1/chat/completions', 'http://localhost:11434/']) {
      assert.deepEqual(E.providerEndpoints({ type: 'local', baseUrl }), want, baseUrl);
    }
  });

  it('an anthropic base ends in /messages, and there is no Ollama fallback for a cloud provider', () => {
    assert.deepEqual(E.providerEndpoints({ type: 'anthropic', baseUrl: 'https://api.anthropic.com' }), {
      chat: 'https://api.anthropic.com/v1/messages',
      models: 'https://api.anthropic.com/v1/models',
    });
  });

  it('an empty base is the type default, and an unparseable one is left recognisable', () => {
    assert.equal(E.providerEndpoints({ type: 'openai' }).chat, 'https://api.openai.com/v1/chat/completions');
    assert.equal(E.providerEndpoints({ type: 'local', serverKind: 'lmstudio' }).models, 'http://localhost:1234/v1/models');
    assert.deepEqual(E.providerEndpoints({ type: 'custom', baseUrl: 'not a url' }), { chat: 'not a url', models: 'not a url' });
  });

  it('the Agent headers: a key in the form the dialect wants, and the organisation only for openai', () => {
    assert.deepEqual(E.providerHeaders({ type: 'anthropic', apiKey: 'K' }, true), {
      'Content-Type': 'application/json',
      'x-api-key': 'K',
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    });
    assert.deepEqual(E.providerHeaders({ type: 'openai', apiKey: 'K', orgId: ' org-1 ' }), { Authorization: 'Bearer K', 'OpenAI-Organization': 'org-1' });
    assert.deepEqual(E.providerHeaders({ type: 'local', apiKey: '', orgId: 'x' }), {}, 'a keyless local server gets none');
  });
});

describe('the path a page may ask for (design 3.2: a fixed list, no pattern)', () => {
  it('each path of the list is accepted for its dialect and method, and only that', () => {
    assert.equal(E.checkApiPath('openai', '/chat/completions', 'POST'), '/chat/completions');
    assert.equal(E.checkApiPath('anthropic', '/messages', 'POST'), '/messages');
    assert.equal(E.checkApiPath('openai', '/models', 'GET'), '/models');
    assert.equal(E.checkApiPath('anthropic', '/models', 'GET'), '/models');
    for (const p of ['/images/generations', '/audio/speech', '/audio/transcriptions', '/embeddings']) assert.equal(E.checkApiPath('openai', p, 'POST'), p);
    assert.throws(() => E.checkApiPath('anthropic', '/chat/completions', 'POST'), { code: 'bad-path' });
    assert.throws(() => E.checkApiPath('openai', '/messages', 'POST'), { code: 'bad-path' });
    assert.throws(() => E.checkApiPath('openai', '/models', 'POST'), { code: 'bad-method' });
    assert.throws(() => E.checkApiPath('openai', '/chat/completions', 'GET'), { code: 'bad-method' });
  });

  it('every shape of an escape from the base is refused: //host, .., @, backslash, ?, #, any %, spaces, control characters', () => {
    const hostile = [
      '//evil.example/x',
      '/..%2f',
      '/a@b',
      '/../models',
      '/chat/completions/..',
      '/chat/completions/../../x',
      '/chat/completions?x=1',
      '/chat/completions#frag',
      '/chat/completions%2f',
      '/chat%2fcompletions',
      '/%2e%2e/models',
      '\\chat\\completions',
      '/chat/completions\\..\\x',
      '/chat/completions ',
      ' /chat/completions',
      '/chat/completions\n',
      '/chat/completions\u0000',
      'chat/completions',
      '',
      '/',
      '/CHAT/COMPLETIONS',
      '/chat/completions/',
      '/chat//completions',
      'https://evil.example/chat/completions',
      '@evil.example/chat/completions',
    ];
    for (const path of hostile) assert.throws(() => E.checkApiPath('openai', path, 'POST'), { name: 'RequestRefused', code: 'bad-path' }, JSON.stringify(path));
    for (const notAString of [undefined, null, 5, {}, ['/models'], true]) assert.throws(() => E.checkApiPath('openai', notAString, 'GET'), { code: 'bad-path' });
    // The property names of every object are not paths.
    for (const path of ['/__proto__', '/constructor', '/toString', '/hasOwnProperty']) assert.throws(() => E.checkApiPath('openai', path, 'GET'), { code: 'bad-path' });
  });
});

describe('the base of a record', () => {
  it('normalises what a person types: a default /v1, no trailing slash, a pasted endpoint cut off', () => {
    const cases = {
      'https://api.openai.com': 'https://api.openai.com/v1',
      'https://api.openai.com/': 'https://api.openai.com/v1',
      'https://api.openai.com/v1/': 'https://api.openai.com/v1',
      'https://api.openai.com/v1/chat/completions': 'https://api.openai.com/v1',
      'https://api.anthropic.com/v1/messages': 'https://api.anthropic.com/v1',
      'https://openrouter.ai/api/v1/models': 'https://openrouter.ai/api/v1',
      'https://generativelanguage.googleapis.com/v1beta/openai': 'https://generativelanguage.googleapis.com/v1beta/openai',
      'http://localhost:11434': 'http://localhost:11434/v1',
      '  http://127.0.0.1:8080/v1  ': 'http://127.0.0.1:8080/v1',
      'HTTPS://API.EXAMPLE.COM/v1': 'https://api.example.com/v1',
    };
    for (const [typed, want] of Object.entries(cases)) assert.equal(E.normalizeApiBase(typed), want, typed);
  });

  it('refuses what cannot be a base: another scheme, a user name, a query, a fragment, no host, not a string', () => {
    for (const typed of [
      'ftp://x.example/v1',
      'javascript:alert(1)',
      'file:///etc/passwd',
      'https://user:pass@api.example.com/v1',
      'https://user@api.example.com/v1',
      'https://api.example.com/v1?api-version=2024',
      'https://api.example.com/v1#x',
      'https://',
      'api.example.com/v1',
      '',
      '/v1',
      'https://api.example.com/v1/%2e%2e/x',
      'https://api.example.com/a\\b',
      'https://api.example.com/v1/../x',
      'https://api.example.com/v1/./x',
      'https://api.example.com/v1/..',
      'https://api.example.com/ v1',
    ]) {
      assert.equal(E.normalizeApiBase(typed), null, typed);
    }
    for (const notAString of [undefined, null, 5, {}, []]) assert.equal(E.normalizeApiBase(notAString), null);
  });
});

describe('the address a request goes to (buildRequestUrl)', () => {
  it('is the record base plus the path, then the query the holder writes itself', () => {
    assert.equal(E.buildRequestUrl(record(), '/chat/completions', 'POST'), 'https://api.openai.com/v1/chat/completions');
    assert.equal(E.buildRequestUrl(record({ baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai' }), '/models', 'GET'), 'https://generativelanguage.googleapis.com/v1beta/openai/models');
    assert.equal(E.buildRequestUrl(record({ dialect: 'anthropic', baseUrl: 'https://api.anthropic.com/v1' }), '/models', 'GET', [['limit', '1000'], ['after_id', 'model x&y=z']]), 'https://api.anthropic.com/v1/models?limit=1000&after_id=model+x%26y%3Dz');
    assert.equal(E.buildRequestUrl(record({ baseUrl: 'http://localhost:11434/v1' }), '/models', 'GET'), 'http://localhost:11434/v1/models');
  });

  it('a value in the query cannot become part of the path: it is written by URLSearchParams', () => {
    const url = E.buildRequestUrl(record(), '/models', 'GET', [['x', '/../..//evil.example#y']]);
    const parsed = new URL(url);
    assert.equal(parsed.origin, 'https://api.openai.com');
    assert.equal(parsed.pathname, '/v1/models');
    assert.equal(parsed.hash, '');
  });

  it('refuses a query that is not a short list of plain pairs', () => {
    for (const query of ['x=1', {}, [['a']], [['a', 'b', 'c']], [[1, 'x']], [['a', 5]], [['a b', 'x']], [['a', 'x\ny']], [['a=b', 'x']], Array.from({ length: 17 }, (_, i) => [`k${i}`, 'v']), [['a', 'x'.repeat(600)]]]) {
      assert.throws(() => E.buildRequestUrl(record(), '/models', 'GET', query), { code: 'bad-query' }, JSON.stringify(query));
    }
  });

  it('refuses a record whose own base is not a normalised one (a stored record cannot be edited into an escape)', () => {
    for (const baseUrl of ['https://evil.example@api.openai.com/v1', 'https://api.openai.com/v1/', 'https://api.openai.com', 'https://api.openai.com/v1?x=1', 'javascript:alert(1)', 'HTTPS://API.OPENAI.COM/v1']) {
      assert.throws(() => E.buildRequestUrl(record({ baseUrl }), '/models', 'GET'), { code: 'bad-base' }, baseUrl);
    }
  });

  it('property: no path of a set of hostile prefixes and suffixes, however built, gives an address off the record base', () => {
    const pieces = ['', '/', '//', '..', '/..', '%2f', '%2e%2e', '@', '\\', '?', '#', 'evil.example', 'https:', ':', '\u0000', ' ', '/models', '/chat/completions', 'x'];
    let checked = 0;
    for (const a of pieces) for (const b of pieces) for (const c of pieces) {
      const path = `${a}${b}${c}`;
      let url = null;
      try {
        url = E.buildRequestUrl(record(), path, 'POST');
      } catch (e) {
        assert.equal(e.name, 'RequestRefused', path);
        continue;
      }
      // If one got through, it is exactly an allowed path on the record's own origin.
      const parsed = new URL(url);
      assert.equal(parsed.origin, 'https://api.openai.com', path);
      assert.equal(parsed.pathname, '/v1/chat/completions', path);
      checked++;
    }
    assert.ok(checked >= 1, 'the allowed path itself passed');
  });
});

describe('the headers of a request (recordHeaders)', () => {
  it('a page may choose only content-type and accept; anything else it sends is left out', () => {
    const headers = E.recordHeaders(record(), 'KEY', {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      Authorization: 'Bearer evil',
      'x-api-key': 'evil',
      'OpenAI-Organization': 'org-evil',
      Host: 'evil.example',
      Cookie: 'a=b',
      Origin: 'https://evil.example',
    });
    assert.deepEqual(headers, { 'content-type': 'application/json', accept: 'text/event-stream', authorization: 'Bearer KEY' });
  });

  it('an evil header of any case cannot replace the key the holder attaches', () => {
    for (const name of ['authorization', 'AUTHORIZATION', 'Authorization', 'X-Api-Key']) {
      const headers = E.recordHeaders(record(), 'REAL', [[name, 'Bearer evil']]);
      assert.equal(headers.authorization, 'Bearer REAL');
      assert.equal(headers['x-api-key'], undefined);
    }
  });

  it('anthropic: the key goes in x-api-key with its fixed headers; none when there is no key; none for auth "none"', () => {
    const a = record({ dialect: 'anthropic', auth: 'x-api-key', baseUrl: 'https://api.anthropic.com/v1' });
    assert.deepEqual(E.recordHeaders(a, 'K'), { 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true', 'x-api-key': 'K' });
    assert.equal(E.recordHeaders(a, '')['x-api-key'], undefined);
    assert.deepEqual(E.recordHeaders(record({ auth: 'none' }), 'K'), {});
  });

  it('the record\'s own extra headers are attached, and only the allowed names', () => {
    const r = record({
      extraHeaders: [
        { name: 'OpenAI-Organization', value: 'org-1' },
        { name: 'HTTP-Referer', value: 'https://app.example' },
        { name: 'X-Title', value: 'OAIY' },
        { name: 'Authorization', value: 'Bearer stolen' },
        { name: 'Host', value: 'evil.example' },
        { name: 'x-title-2', value: 'no' },
        { name: 'anthropic-beta', value: 'a\r\nInjected: 1' },
      ],
    });
    assert.deepEqual(E.recordHeaders(r, ''), { 'openai-organization': 'org-1', 'http-referer': 'https://app.example', 'x-title': 'OAIY' });
  });

  it('a value with a line break or a control character is refused, not passed on', () => {
    assert.throws(() => E.recordHeaders(record(), 'K', { 'content-type': 'a\r\nInjected: 1' }), { code: 'bad-headers' });
    assert.throws(() => E.recordHeaders(record(), 'K', { accept: 'x'.repeat(600) }), { code: 'bad-headers' });
  });
});
