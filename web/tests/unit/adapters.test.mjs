/**
 * The provider record and the shapes it converts to and from (shared/providers/adapters.ts, design 4.1): every pair is a
 * round trip, and no record carries a key.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadTs } from '../support/load.mjs';

const A = await loadTs('shared/providers/adapters.ts');
const T = await loadTs('shared/providers/types.ts');
const E = await loadTs('shared/providers/endpoints.ts');

/** Agent providers as they are saved: each type, with and without an address, model, organisation and limits. */
const AGENT_CONFIGS = [
  { id: 'a1', type: 'anthropic', name: 'Claude', apiKey: 'sk-ant-KEY-1', modelId: 'claude-x' },
  { id: 'a2', type: 'anthropic', name: 'Claude EU', apiKey: 'sk-ant-KEY-2', baseUrl: 'https://api.anthropic.com/v1/messages', modelId: ' claude-y ', contextTokens: 200000 },
  { id: 'o1', type: 'openai', name: 'OpenAI', apiKey: 'sk-proj-KEY-3' },
  { id: 'o2', type: 'openai', name: 'OpenAI work', apiKey: 'sk-proj-KEY-4', baseUrl: 'https://api.openai.com', modelId: 'gpt-x', orgId: 'org-abc', parallelAgents: 2 },
  { id: 'o3', type: 'openai', name: 'OpenAI blank org', apiKey: 'k', baseUrl: 'https://api.openai.com/v1/', orgId: '   ' },
  { id: 'c1', type: 'custom', name: 'OpenRouter', apiKey: 'or-KEY-5', baseUrl: 'https://openrouter.ai/api/v1', modelId: 'vendor/model:free' },
  { id: 'c2', type: 'custom', name: 'Gemini', apiKey: 'AIza-KEY-6', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', modelId: 'gemini-x' },
  { id: 'c3', type: 'custom', name: 'Somewhere', apiKey: '', baseUrl: 'http://192.168.1.5:8000/v1/chat/completions' },
  { id: 'l1', type: 'local', name: 'Ollama', apiKey: '', serverKind: 'ollama', baseUrl: 'http://localhost:11434', modelId: 'llama3' },
  { id: 'l2', type: 'local', name: 'LM Studio', apiKey: '', serverKind: 'lmstudio' },
  { id: 'l3', type: 'local', name: 'OAIY', apiKey: 'k', serverKind: 'oaiy', baseUrl: 'http://127.0.0.1:8080/v1', followEngine: true, detectedContext: { model: 'm', tokens: 8192, how: 'x', at: 1 }, contextTokens: 16000 },
  { id: 'l4', type: 'local', name: 'Other', apiKey: '', baseUrl: 'http://localhost:8080' },
];

describe('the Agent provider and the record', () => {
  it('each Agent provider becomes a record and comes back exactly, with its key beside it', () => {
    for (const config of AGENT_CONFIGS) {
      const made = A.recordFromAgentConfig(config);
      assert.ok(made, config.id);
      assert.equal(made.apiKey, config.apiKey);
      const back = A.agentConfigFromRecord(made.record, made.apiKey, config);
      assert.deepEqual(back, config, config.id);
    }
  });

  it('a record made from nothing but its own fields gives an Agent provider that makes the same record again', () => {
    for (const config of AGENT_CONFIGS) {
      const first = A.recordFromAgentConfig(config);
      const alone = A.agentConfigFromRecord(first.record, first.apiKey);
      const second = A.recordFromAgentConfig(alone);
      assert.deepEqual(second.record, first.record, config.id);
      assert.equal(second.apiKey, first.apiKey);
    }
  });

  it('the record says what the dialect, the address and the kind are, defined once', () => {
    const rec = (id) => A.recordFromAgentConfig(AGENT_CONFIGS.find((c) => c.id === id)).record;
    assert.deepEqual(
      { d: rec('a1').dialect, a: rec('a1').auth, b: rec('a1').baseUrl, k: rec('a1').kind, v: rec('a1').via },
      { d: 'anthropic', a: 'x-api-key', b: 'https://api.anthropic.com/v1', k: 'external', v: 'broker' },
    );
    assert.deepEqual({ d: rec('o2').dialect, a: rec('o2').auth, b: rec('o2').baseUrl, e: rec('o2').extraHeaders }, { d: 'openai', a: 'bearer', b: 'https://api.openai.com/v1', e: [{ name: 'OpenAI-Organization', value: 'org-abc' }] });
    assert.equal(rec('c2').baseUrl, 'https://generativelanguage.googleapis.com/v1beta/openai');
    assert.equal(rec('c3').baseUrl, 'http://192.168.1.5:8000/v1');
    assert.deepEqual({ k: rec('l1').kind, s: rec('l1').serverKind, b: rec('l1').baseUrl }, { k: 'local-server', s: 'ollama', b: 'http://localhost:11434/v1' });
    assert.equal(rec('l2').baseUrl, 'http://localhost:1234/v1', 'no address is the kind\'s own');
    assert.equal(rec('a2').model, 'claude-y', 'a model is trimmed');
    assert.equal(rec('a2').baseUrl, 'https://api.anthropic.com/v1', 'a pasted endpoint is cut off');
  });

  it('a record is never given a key: no field of any record holds one, whatever the config carried', () => {
    for (const config of AGENT_CONFIGS) {
      const { record, apiKey } = A.recordFromAgentConfig(config);
      if (apiKey.length >= 6) assert.ok(!JSON.stringify(record).includes(apiKey), config.id);
      for (const value of Object.values(record)) assert.notEqual(value, apiKey || undefined, `${config.id}: a field holds the key`);
      assert.ok(!('apiKey' in record) && !('key' in record));
    }
    assert.equal(T.providerKeyName('p1'), 'key:p1');
  });

  it('an address a record cannot hold (a query string, another scheme) gives null, not a guess', () => {
    assert.equal(A.recordFromAgentConfig({ id: 'x', type: 'custom', name: 'Azure', apiKey: 'k', baseUrl: 'https://x.openai.azure.com/v1?api-version=2024' }), null);
    assert.equal(A.recordFromAgentConfig({ id: 'x', type: 'custom', name: 'X', apiKey: 'k', baseUrl: 'ftp://x/v1' }), null);
  });

  it('the type an Agent provider has is what its record says', () => {
    const cases = [
      [{ dialect: 'anthropic', kind: 'external' }, 'anthropic'],
      [{ dialect: 'openai', kind: 'local-server' }, 'local'],
      [{ dialect: 'openai', kind: 'external', preset: 'openai' }, 'openai'],
      [{ dialect: 'openai', kind: 'external', preset: 'gemini' }, 'custom'],
      [{ dialect: 'openai', kind: 'external' }, 'custom'],
    ];
    for (const [record, type] of cases) assert.equal(A.providerTypeOf(record), type, JSON.stringify(record));
  });
});

describe('the flow editor service and the record', () => {
  const rec = (over = {}) => ({ ...A.recordFromAgentConfig(AGENT_CONFIGS[3]).record, ...over });

  it('a record becomes a service that names it and holds no key', () => {
    const service = A.flowsServiceFromRecord(rec());
    assert.equal(service.endpoint, 'oaiy-provider://o2/chat/completions');
    assert.equal(service.apiFormat, 'openai');
    assert.deepEqual(service.nodeTypes, ['ai_llm', 'service_call']);
    assert.equal(service.model, 'gpt-x');
    assert.equal(service.responsePath, 'choices.0.message.content');
    assert.ok(!('apiKeyConstant' in service), 'the flow never holds the key');
    assert.ok(!/\{\{apiKey\}\}/.test(JSON.stringify(service)), 'and the service never asks a flow for one');
    assert.equal(A.flowsServiceId('o2'), 'provider:o2');
  });

  it('an anthropic record speaks the messages shape', () => {
    const service = A.flowsServiceFromRecord(rec({ dialect: 'anthropic', id: 'a1' }));
    assert.equal(service.endpoint, 'oaiy-provider://a1/messages');
    assert.equal(service.apiFormat, 'anthropic');
    assert.equal(service.responsePath, 'content.0.text');
  });

  it('the endpoint names the record and the path, and only such an endpoint does', () => {
    for (const id of ['o2', 'p_1a2b3c', 'weird id/with:chars']) {
      const service = A.flowsServiceFromRecord(rec({ id }));
      assert.deepEqual(A.providerRefOfEndpoint(service.endpoint), { id, path: '/chat/completions' }, id);
    }
    for (const notOurs of ['https://api.openai.com/v1/chat/completions', 'oaiy-provider://', 'oaiy-provider:///chat/completions', 'oaiy-provider://x', 'oaiy-provider://%E0%A4%A/x', undefined, 7]) {
      assert.equal(A.providerRefOfEndpoint(notOurs), null, String(notOurs));
    }
  });

  it('a model-less record gives a service with no model', () => {
    assert.ok(!('model' in A.flowsServiceFromRecord(rec({ model: undefined }))));
  });
});

describe('the gateway record and the record', () => {
  const PUBLIC = {
    id: 'gw-openai',
    name: 'Work OpenAI',
    category: 'cloud',
    protocol: 'openai',
    baseUrl: 'https://api.openai.com/',
    model: 'gpt-x',
    capabilities: ['chat', 'transcription'],
    enabled: true,
    allowLocal: false,
    hasKey: true,
  };

  it('a gateway provider is a read-only mirror: via gateway, with the gateway\'s base as an API base', () => {
    const record = A.recordFromGatewayProvider(PUBLIC);
    assert.equal(record.via, 'gateway');
    assert.equal(record.baseUrl, 'https://api.openai.com/v1');
    assert.deepEqual(record.caps, ['chat', 'transcription']);
    assert.equal(record.kind, 'external');
    assert.equal(record.model, 'gpt-x');
    assert.ok(!('hasKey' in record) && !JSON.stringify(record).includes('key'), 'not even the fact of a key is in the record');
  });

  it('a record comes back as the gateway\'s own input, and the pair round-trips', () => {
    for (const p of [PUBLIC, { ...PUBLIC, id: 'g2', protocol: 'anthropic', baseUrl: 'https://api.anthropic.com', capabilities: [], model: null, category: null }, { ...PUBLIC, id: 'g3', baseUrl: 'http://localhost:11434', allowLocal: true, capabilities: ['chat'] }]) {
      const record = A.recordFromGatewayProvider(p);
      const input = A.gatewayInputFromRecord(record);
      assert.equal(input.id, p.id);
      assert.equal(input.protocol, p.protocol);
      assert.equal(input.baseUrl, p.baseUrl.replace(/\/+$/, ''), 'the gateway appends /v1 itself, so the record\'s /v1 is taken off again');
      assert.equal(input.allowLocal, p.allowLocal);
      assert.equal(input.model, p.model ?? undefined);
      assert.ok(!('apiKey' in input) && !('hasKey' in input));
      assert.deepEqual(A.recordFromGatewayProvider({ ...input, hasKey: p.hasKey, model: input.model ?? null, category: input.category ?? null }), record, 'and makes the same record again');
    }
  });

  it('a base the gateway cannot use (Gemini\'s /v1beta/openai) has no gateway input, until the gateway stops appending /v1', () => {
    const gemini = A.recordFromAgentConfig(AGENT_CONFIGS.find((c) => c.id === 'c2')).record;
    assert.equal(A.gatewayInputFromRecord(gemini), null);
    // OpenRouter, Groq and OpenAI can be proxied: the gateway's base plus its own /v1 is their API base.
    const openrouter = A.recordFromAgentConfig(AGENT_CONFIGS.find((c) => c.id === 'c1')).record;
    assert.equal(A.gatewayInputFromRecord(openrouter).baseUrl, 'https://openrouter.ai/api');
  });

  it('a realtime-only capability is not one the record has', () => {
    assert.deepEqual(A.recordFromGatewayProvider({ ...PUBLIC, capabilities: ['realtime'] }).caps, []);
  });
});

describe('the extra headers a record may carry', () => {
  it('are the four names of the design and no other', () => {
    assert.deepEqual([...T.EXTRA_HEADER_NAMES].sort(), ['anthropic-beta', 'http-referer', 'openai-organization', 'x-title']);
    assert.equal(A.isExtraHeaderName('X-Title'), true);
    assert.equal(A.isExtraHeaderName('Authorization'), false);
    assert.equal(E.recordHeaders({ dialect: 'openai', auth: 'bearer', extraHeaders: [{ name: 'Authorization', value: 'x' }] }, '').authorization, undefined);
  });
});
