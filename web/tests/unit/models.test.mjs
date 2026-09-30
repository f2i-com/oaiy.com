/**
 * The model lists (shared/providers/models.ts): what the Agent's `listModels` did before it moved, and that the same walk for
 * a record and its key (`listRecordModels`) asks the same questions and gives the same answers.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const M = await loadTs('shared/providers/models.ts');
const A = await loadTs('shared/providers/adapters.ts');

const PAGE = { protocol: 'https:', origin: 'https://p.example' };
const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

/** A fetch that answers from a table of `url -> body|Response|Error`, and records what it was asked. */
function fakeFetch(table) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, headers: init?.headers, method: init?.method });
    const answer = table[url];
    if (answer === undefined) throw new TypeError('Failed to fetch');
    if (answer instanceof Error) throw answer;
    return answer instanceof Response ? answer.clone() : json(answer);
  };
  return { impl, calls };
}

describe('listing models', () => {
  it('an OpenAI-style server: GET {base}/models with the key, newest first', async () => {
    const f = fakeFetch({
      'https://api.openai.com/v1/models': { data: [{ id: 'old', created: 100 }, { id: 'new', created: 200 }, { id: 'undated' }, { id: 'a-image', type: 'image' }] },
    });
    const models = await M.listModels({ type: 'openai', apiKey: 'K', orgId: 'org-1' }, { fetchImpl: f.impl, page: PAGE });
    assert.deepEqual(models.map((m) => m.id), ['new', 'old', 'undated']);
    assert.equal(f.calls.length, 1);
    assert.deepEqual(f.calls[0].headers, { Authorization: 'Bearer K', 'OpenAI-Organization': 'org-1' });
  });

  it('Anthropic pages are followed and joined, with the key in x-api-key', async () => {
    const f = fakeFetch({
      'https://api.anthropic.com/v1/models?limit=1000': { data: [{ id: 'a' }], has_more: true, last_id: 'a' },
      'https://api.anthropic.com/v1/models?limit=1000&after_id=a': { data: [{ id: 'b' }, { id: 'a' }], has_more: false },
    });
    const models = await M.listModels({ type: 'anthropic', apiKey: 'K' }, { fetchImpl: f.impl, page: PAGE });
    assert.deepEqual(models.map((m) => m.id), ['a', 'b']);
    assert.equal(f.calls[0].headers['x-api-key'], 'K');
    assert.equal(f.calls[0].headers['anthropic-version'], '2023-06-01');
  });

  it('a local server with no /models falls back to Ollama\'s tags, and reports its first error when that fails too', async () => {
    const tags = { models: [{ name: 'llama3:8b', modified_at: '2026-01-02T00:00:00Z' }, { model: 'qwen3:4b' }] };
    const f = fakeFetch({ 'http://localhost:11434/v1/models': json({ error: 'nope' }, 404), 'http://localhost:11434/api/tags': tags });
    const models = await M.listModels({ type: 'local', serverKind: 'ollama', apiKey: '', baseUrl: 'http://localhost:11434' }, { fetchImpl: f.impl, page: PAGE });
    assert.deepEqual(models.map((m) => m.id).sort(), ['llama3:8b', 'qwen3:4b']);
    const g = fakeFetch({ 'http://localhost:11434/v1/models': json({}, 404) });
    await assert.rejects(M.listModels({ type: 'local', apiKey: '', baseUrl: 'http://localhost:11434' }, { fetchImpl: g.impl, page: PAGE }), { kind: 'not-found' });
  });

  it('a wrong key is an auth error with the provider\'s words, a down server a network error, an https page to plain http is mixed content', async () => {
    const auth = fakeFetch({ 'https://api.openai.com/v1/models': json({ error: { message: 'bad key' } }, 401) });
    await assert.rejects(M.listModels({ type: 'openai', apiKey: 'K' }, { fetchImpl: auth.impl, page: PAGE }), (e) => e.kind === 'auth' && /bad key/.test(e.message) && e.status === 401);
    const down = fakeFetch({});
    await assert.rejects(M.listModels({ type: 'openai', apiKey: 'K' }, { fetchImpl: down.impl, page: PAGE }), { kind: 'network' });
    await assert.rejects(M.listModels({ type: 'custom', apiKey: '', baseUrl: 'http://192.168.1.5:8000' }, { fetchImpl: down.impl, page: PAGE }), { kind: 'mixed-content' });
    assert.equal(down.calls.length, 1, 'mixed content is decided before any request');
  });

  it('an answer that is not a model list is invalid-response; a slow one times out; an aborted one is cancelled', async () => {
    const odd = fakeFetch({ 'https://api.openai.com/v1/models': { hello: 'world' } });
    await assert.rejects(M.listModels({ type: 'openai', apiKey: 'K' }, { fetchImpl: odd.impl, page: PAGE }), { kind: 'invalid-response' });
    const slow = (url, init) => new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))));
    await assert.rejects(M.listModels({ type: 'openai', apiKey: 'K' }, { fetchImpl: slow, page: PAGE, timeoutMs: 20 }), { kind: 'timeout' });
    const controller = new AbortController();
    const cancelled = M.listModels({ type: 'openai', apiKey: 'K' }, { fetchImpl: slow, page: PAGE, signal: controller.signal });
    controller.abort();
    await assert.rejects(cancelled, { kind: 'cancelled' });
  });

  it('chat models are told from the rest by capability words, never by a list of models', () => {
    const { models, hidden } = M.chatModels([{ id: 'gpt-x' }, { id: 'text-embedding-3' }, { id: 'whisper-1' }, { id: 'claude-x' }, { id: 'dall-e-3' }]);
    assert.deepEqual(models.map((m) => m.id), ['gpt-x', 'claude-x']);
    assert.equal(hidden, 3);
  });
});

describe('the same walk for a record and its key', () => {
  it('asks the same addresses with the same headers and gives the same list as the Agent\'s listModels', async () => {
    const configs = [
      { id: 'o', type: 'openai', name: 'O', apiKey: 'K1', orgId: 'org-1' },
      { id: 'c', type: 'custom', name: 'C', apiKey: 'K2', baseUrl: 'https://openrouter.ai/api/v1' },
      { id: 'l', type: 'local', name: 'L', apiKey: '', serverKind: 'ollama', baseUrl: 'http://localhost:11434' },
      { id: 'a', type: 'anthropic', name: 'A', apiKey: 'K3' },
    ];
    for (const config of configs) {
      const table = {
        'https://api.openai.com/v1/models': { data: [{ id: 'x' }, { id: 'y', created: 5 }] },
        'https://openrouter.ai/api/v1/models': { data: [{ id: 'x' }, { id: 'y', created: 5 }] },
        'http://localhost:11434/v1/models': { data: [{ id: 'x' }, { id: 'y', created: 5 }] },
        'https://api.anthropic.com/v1/models?limit=1000': { data: [{ id: 'x' }, { id: 'y', created: 5 }] },
      };
      const agent = fakeFetch(table);
      const holder = fakeFetch(table);
      const want = await M.listModels(config, { fetchImpl: agent.impl, page: PAGE });
      const { record, apiKey } = A.recordFromAgentConfig(config);
      const got = await M.listRecordModels(record, apiKey, { fetchImpl: holder.impl, page: PAGE });
      assert.deepEqual(got, want, config.id);
      assert.deepEqual(holder.calls.map((c) => c.url), agent.calls.map((c) => c.url), config.id);
      const lower = (h) => Object.fromEntries(Object.entries(h).map(([k, v]) => [k.toLowerCase(), v]));
      assert.deepEqual(lower(holder.calls[0].headers), lower(agent.calls[0].headers), config.id);
    }
  });
});
