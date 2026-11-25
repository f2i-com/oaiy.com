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
    await assert.rejects(redirecting('https://api.openai.com/v1/models'), (e) => e instanceof M.net.RedirectRefused && /redirect is not followed/i.test(e.message));
  });

  it('tester classifies redirects as redirect and chat Timeouts as timeout without running a no-cors probe', async () => {
    const rec = { id: 'p_1', v: 1, name: 'Test', dialect: 'openai', baseUrl: 'https://api.openai.com/v1', auth: 'bearer', caps: ['chat'], kind: 'external', via: 'broker', model: 'm' };

    const redirModels = stubFetch(() => new Response(null, { status: 302, headers: { location: 'https://evil.example/' } }));
    const t1 = M.tester.createTester({ fetchImpl: redirModels.impl, page: PAGE, key: async () => 'sk-test' });
    const r1 = await t1.models(rec);
    assert.equal(r1.ok, false);
    assert.equal(r1.error.kind, 'redirect');
    assert.equal(redirModels.calls.length, 1, 'no no-cors probe on redirect');

    const redirChat = stubFetch((c) => (c.url.endsWith('/models') ? jsonResponse({ data: [{ id: 'm' }] }) : new Response(null, { status: 307, headers: { location: 'https://evil.example/' } })));
    const t2 = M.tester.createTester({ fetchImpl: redirChat.impl, page: PAGE, key: async () => 'sk-test' });
    const r2 = await t2.test(rec);
    assert.equal(r2.ok, false);
    assert.equal(r2.error.kind, 'redirect');
    assert.equal(redirChat.calls.length, 2, 'models + chat only, no no-cors probe');

    const timeoutChat = stubFetch((c) => {
      if (c.url.endsWith('/models')) return jsonResponse({ data: [{ id: 'm' }] });
      throw new DOMException('The operation timed out.', 'TimeoutError');
    });
    const t3 = M.tester.createTester({ fetchImpl: timeoutChat.impl, page: PAGE, key: async () => 'sk-test' });
    const r3 = await t3.test(rec);
    assert.equal(r3.ok, false);
    assert.equal(r3.error.kind, 'timeout');
    assert.equal(timeoutChat.calls.length, 2, 'models + chat only, no no-cors probe on timeout');
  });

  it('local-server tester never falls back to /api/tags on RedirectRefused or BudgetExhausted', async () => {
    const localRec = {
      id: 'p_local',
      v: 1,
      name: 'Local Ollama',
      dialect: 'openai',
      baseUrl: 'http://localhost:11434/v1',
      auth: 'bearer',
      caps: ['chat'],
      kind: 'local-server',
      preset: 'local-server',
      serverKind: 'ollama',
      via: 'broker',
    };

    // Even when /api/tags would succeed, a redirect on /v1/models must stop immediately and return redirect.
    const redirLocal = stubFetch((c) => {
      if (c.url.endsWith('/v1/models')) return new Response(null, { status: 302, headers: { location: 'https://evil.example/' } });
      return jsonResponse({ models: [{ name: 'llama3:8b' }] });
    });
    const tRedir = M.tester.createTester({ fetchImpl: redirLocal.impl, page: PAGE, key: async () => '' });
    const rRedir = await tRedir.models(localRec);
    assert.equal(rRedir.ok, false);
    assert.equal(rRedir.error.kind, 'redirect');
    assert.deepEqual(redirLocal.calls.map((c) => c.url), ['http://localhost:11434/v1/models'], 'must not fall back to /api/tags on redirect');

    // When the budget is already exhausted on /v1/models, take() must not be called again for /api/tags.
    let takeCalls = 0;
    const budgetLocal = stubFetch(() => jsonResponse({ models: [{ name: 'llama3:8b' }] }));
    const tBudget = M.tester.createTester({
      fetchImpl: budgetLocal.impl,
      page: PAGE,
      key: async () => '',
      take: async () => {
        takeCalls++;
        return { ok: false, reason: 'requests', limit: 5, byteLimit: 1024, retryAfterMs: 1000 };
      },
    });
    const rBudget = await tBudget.models(localRec);
    assert.equal(rBudget.ok, false);
    assert.equal(rBudget.error.kind, 'budget');
    assert.equal(takeCalls, 1, 'budget checked once, no /api/tags fallback');
    assert.equal(budgetLocal.calls.length, 0, 'no network call when budget is exhausted');
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

  it('only a server on this computer or network is probed: an external record (OpenAI, a custom service) is asked nothing, and gets no key sent to /props', async () => {
    for (const over of [{ kind: 'external', baseUrl: 'https://api.openai.com/v1' }, { kind: 'external', preset: 'custom', baseUrl: 'https://models.example.com/v1' }, { kind: 'external', serverKind: 'ollama', baseUrl: 'https://ollama.example.com/v1' }]) {
      const f = stubFetch(() => jsonResponse({ data: [{ id: 'm', context_length: 4096 }] }));
      assert.deepEqual(await M.probe.probeContext(rec(over), 'SECRET-KEY-VALUE', f.impl), { contextTokens: null, how: null }, JSON.stringify(over));
      assert.equal(f.calls.length, 0, `no request at all for ${over.baseUrl}`);
    }
  });

  it('the key goes only to an address under the record\'s base: the servers\' own information endpoints outside it are asked without it', async () => {
    const KEY2 = 'SECRET-KEY-VALUE-1234567890';
    const f = stubFetch((c) => {
      if (c.url.endsWith('/api/ps') || c.url.endsWith('/api/show') || c.url.endsWith('/props') || c.url.includes('/api/v0/models/')) return jsonResponse({}, 404);
      return jsonResponse({ data: [{ id: 'm' }] });
    });
    for (const serverKind of ['ollama', 'lmstudio', 'oaiy', 'other']) {
      f.calls.length = 0;
      await M.probe.probeContext(rec({ serverKind, baseUrl: serverKind === 'ollama' ? 'http://localhost:11434/v1' : serverKind === 'lmstudio' ? 'http://localhost:1234/v1' : 'http://localhost:8080/v1' }), KEY2, f.impl);
      assert.ok(f.calls.length >= 2, serverKind);
      for (const call of f.calls) {
        const url = new URL(call.url);
        const under = url.pathname === '/v1' || url.pathname.startsWith('/v1/');
        assert.equal(call.headers.authorization !== undefined, under, `${serverKind}: ${call.url} ${under ? 'is under the base and carries the key' : 'is outside it and carries none'}`);
        assert.ok(!JSON.stringify(call.headers).includes(KEY2) || under, `${call.url}: no key outside the base`);
      }
    }
  });

  it('Ollama\'s own model list (outside the API base) is asked only of a local server, and without the key', async () => {
    const KEY2 = 'SECRET-KEY-VALUE-1234567890';
    const local = stubFetch((c) => (c.url.endsWith('/api/tags') ? jsonResponse({ models: [{ name: 'llama3:8b' }] }) : jsonResponse({}, 404)));
    const found = await M.models.listRecordModels({ dialect: 'openai', baseUrl: 'http://localhost:11434/v1', auth: 'bearer', kind: 'local-server', preset: 'local-server', serverKind: 'ollama' }, KEY2, { fetchImpl: local.impl, page: PAGE });
    assert.deepEqual(found.map((m) => m.id), ['llama3:8b']);
    const tags = local.calls.find((c) => c.url.endsWith('/api/tags'));
    assert.equal(tags.headers.authorization, undefined, 'no key to /api/tags');
    assert.equal(local.calls.find((c) => c.url.endsWith('/v1/models')).headers.authorization, `Bearer ${KEY2}`, 'the key to the model list under the base');
    const external = stubFetch(() => jsonResponse({}, 404));
    await assert.rejects(M.models.listRecordModels({ dialect: 'openai', baseUrl: 'https://models.example.com/v1', auth: 'bearer', kind: 'external', preset: 'custom' }, KEY2, { fetchImpl: external.impl, page: PAGE }), { kind: 'not-found' });
    assert.ok(external.calls.every((c) => !c.url.includes('/api/tags')), 'an external service is not asked for Ollama\'s list');
  });

  it('a number too small to be a window is not one', async () => {
    const f = stubFetch(() => jsonResponse({ data: [{ id: 'm', context_length: 100 }] }));
    assert.equal((await M.probe.probeContext(rec({}), '', f.impl)).contextTokens, null);
  });
});
