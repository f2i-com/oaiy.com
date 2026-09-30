/**
 * What the holder does around a call to a provider (web/providers/src/net.ts, probe.ts): the guard on its own calls, and finding out
 * why a call that never got an answer failed.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { M } from '../support/holder.mjs';
import { jsonResponse, stubFetch } from '../support/broker-world.mjs';

const PAGE = { protocol: 'https:', origin: 'https://providers.example' };

describe('the guard on the holder\'s own calls', () => {
  const record = { baseUrl: 'https://api.openai.com/v1' };

  it('goes only to the provider\'s own origin, and says so like a browser does (a TypeError)', async () => {
    const f = stubFetch(() => jsonResponse({}));
    const guarded = M.net.guardedFetch(record, f.impl);
    await guarded('https://api.openai.com/v1/models');
    await guarded('https://api.openai.com/api/other'); // the same origin, another path: the holder's own code chose it
    for (const url of ['https://evil.example/v1/models', 'http://api.openai.com/v1/models', 'https://api.openai.com.evil.example/v1', 'https://api.openai.com:8443/v1/models']) {
      await assert.rejects(guarded(url), TypeError, url);
    }
    assert.equal(f.calls.length, 2);
  });

  it('never follows a redirect, sends no credentials and no referrer, and counts against the budget when asked', async () => {
    const f = stubFetch(() => jsonResponse({}));
    let taken = 0;
    const guarded = M.net.guardedFetch(record, f.impl, async () => (taken++ < 2 ? { ok: true } : { ok: false, limit: 2, retryAfterMs: 5 }));
    await guarded('https://api.openai.com/v1/models');
    await guarded('https://api.openai.com/v1/models');
    assert.equal(f.calls[0].redirect, 'manual');
    assert.equal(f.calls[0].credentials, 'omit');
    await assert.rejects(guarded('https://api.openai.com/v1/models'), { name: 'BudgetExhausted', limit: 2 });
    assert.equal(f.calls.length, 2);

    const redirecting = M.net.guardedFetch(record, stubFetch(() => new Response(null, { status: 307, headers: { location: 'https://evil.example/' } })).impl);
    await assert.rejects(redirecting('https://api.openai.com/v1/models'), /redirect is not followed/);
  });
});

describe('why a call failed', () => {
  const local = { kind: 'local-server', serverKind: 'ollama' };
  const external = { kind: 'external' };

  it('a server that answers a no-cors call is up: a local one refused CORS, one on the internet gave nothing readable', async () => {
    const up = stubFetch(() => new Response(null, { status: 200 }));
    const a = await M.net.classifyFailure(local, 'http://localhost:11434/v1/models', PAGE, up.impl);
    assert.equal(a.kind, 'cors');
    assert.match(a.message, /allow https:\/\/providers\.example/i);
    const b = await M.net.classifyFailure(external, 'https://api.openai.com/v1/chat/completions', PAGE, up.impl);
    assert.equal(b.kind, 'unreadable');
    assert.equal(up.calls[0].mode, 'no-cors');
    assert.equal(up.calls[0].url, 'http://localhost:11434/');
    assert.equal(up.calls[1].url, 'https://api.openai.com/');
  });

  it('no answer at all is down (or blocked), and http to a machine that is not this one from an https page is mixed content, decided without a request', async () => {
    const down = stubFetch(() => { throw new TypeError('Failed to fetch'); });
    assert.equal((await M.net.classifyFailure(local, 'http://localhost:11434/v1/models', PAGE, down.impl)).kind, 'unreachable');
    const none = stubFetch(() => jsonResponse({}));
    const mixed = await M.net.classifyFailure(local, 'http://192.168.1.5:8000/v1/models', PAGE, none.impl);
    assert.equal(mixed.kind, 'mixed-content');
    assert.equal(none.calls.length, 0);
    assert.equal((await M.net.classifyFailure(local, 'http://192.168.1.5:8000/v1/models', { protocol: 'http:', origin: 'http://p.localhost' }, down.impl)).kind, 'unreachable', 'an http page has no mixed content to block');
  });

  it('a server on this computer or network is told about the permission a browser may want; one on the internet is not', async () => {
    const down = stubFetch(() => { throw new TypeError('Failed to fetch'); });
    assert.match((await M.net.classifyFailure(local, 'http://localhost:11434/v1', PAGE, down.impl)).message, /local network/);
    assert.match((await M.net.classifyFailure({ kind: 'local-server' }, 'http://192.168.1.5:8000/v1', { protocol: 'http:', origin: 'http://x.localhost' }, down.impl)).message, /local network/);
    assert.doesNotMatch((await M.net.classifyFailure(external, 'https://api.example.com/v1', PAGE, down.impl)).message, /local network/);
  });

  it('what counts as this computer or this network', () => {
    for (const url of ['http://localhost:1', 'http://127.0.0.1:1', 'http://[::1]:1', 'http://a.localhost:1', 'http://10.1.2.3', 'http://172.16.0.1', 'http://172.31.255.1', 'http://192.168.0.9', 'http://169.254.1.1', 'http://100.64.0.1', 'http://printer.local']) {
      assert.equal(M.net.isLocalAddress(url), true, url);
    }
    for (const url of ['https://api.openai.com', 'http://172.32.0.1', 'http://100.128.0.1', 'http://11.0.0.1', 'http://192.169.0.1', 'junk']) assert.equal(M.net.isLocalAddress(url), false, url);
  });
});

describe('context detection', () => {
  const rec = (over) => ({ dialect: 'openai', baseUrl: 'http://localhost:1234/v1', auth: 'bearer', kind: 'local-server', model: 'm', ...over });

  it('asks each server the way it answers, and stops at the first that says', async () => {
    const lm = stubFetch((c) => (c.url === 'http://localhost:1234/api/v0/models/m' ? jsonResponse({ loaded_context_length: 16384 }) : jsonResponse({}, 404)));
    assert.deepEqual(await M.probe.probeContext(rec({ serverKind: 'lmstudio' }), '', lm.impl), { contextTokens: 16384, how: 'LM Studio (/api/v0/models)' });
    const oaiy = stubFetch((c) => (c.url === 'http://127.0.0.1:8080/v1/discovery' ? jsonResponse({ llm: { context_tokens: 32768 } }) : jsonResponse({}, 404)));
    assert.equal((await M.probe.probeContext(rec({ serverKind: 'oaiy', baseUrl: 'http://127.0.0.1:8080/v1' }), '', oaiy.impl)).contextTokens, 32768);
    const props = stubFetch((c) => (c.url === 'http://localhost:8080/props' ? jsonResponse({ default_generation_settings: { n_ctx: 4096 } }) : jsonResponse({ data: [] })));
    assert.deepEqual(await M.probe.probeContext(rec({ baseUrl: 'http://localhost:8080/v1' }), '', props.impl), { contextTokens: 4096, how: 'llama.cpp (/props)' });
    const list = stubFetch(() => jsonResponse({ data: [{ id: 'm', max_model_len: 65536 }] }));
    assert.equal((await M.probe.probeContext(rec({ baseUrl: 'http://localhost:8000/v1' }), '', list.impl)).contextTokens, 65536);
  });

  it('says nothing when nothing says, and for Anthropic or a record with no model', async () => {
    const none = stubFetch(() => jsonResponse({}, 404));
    assert.deepEqual(await M.probe.probeContext(rec({}), '', none.impl), { contextTokens: null, how: null });
    assert.deepEqual(await M.probe.probeContext(rec({ dialect: 'anthropic' }), '', none.impl), { contextTokens: null, how: null });
    assert.deepEqual(await M.probe.probeContext(rec({ model: undefined }), '', none.impl), { contextTokens: null, how: null });
    assert.equal(none.calls.length > 0, true);
  });

  it('a number too small to be a window is not one', async () => {
    const f = stubFetch(() => jsonResponse({ data: [{ id: 'm', context_length: 100 }] }));
    assert.equal((await M.probe.probeContext(rec({}), '', f.impl)).contextTokens, null);
  });
});
