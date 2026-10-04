// What the Agent runs on: the desktop's choice (its engine, ChatGPT through
// OAIY's Codex connector, or one of its AI providers through its gateway), the
// provider each kind of conversation takes from it, and what the person is told
// when OAIY is not signed in to ChatGPT.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { sendTurn } from '../../src/agent/protocol';
import { AIProviderError } from '../../src/agent/providers/aiProvider';
import { CHATGPT_SIGN_IN } from '../../src/agent/providers/chatgpt';
import type { ProviderConfig } from '../../src/agent/providers/types';
import {
  CALL_ROUTE,
  CHATGPT_CONTEXT_TOKENS,
  CODEX_ROUTE,
  ENGINE,
  LUNA_CALL_ROUTE,
  callRoute,
  chatgptProvider,
  codexDefaultModel,
  desktopProvider,
  modelChipText,
  parseAgentModel,
  pickCodexModel,
  providerFor,
  readAgentModel,
  sameModel,
  type AgentKind,
} from '../../src/desktop/agentModel';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

const DESK = { origin: 'http://127.0.0.1:17972', token: 'desk-token' };
/** The engine's provider as the app sets it up (followed from OAIY's Engines). */
const ENGINE_PROVIDER: ProviderConfig = { id: 'oaiy', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:8080', modelId: 'Qwen3.8-Flash-Next', followEngine: true };
const KINDS: AgentKind[] = ['project', 'setup', 'runner', 'call', 'sms', 'task'];

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

describe("the desktop's choice", () => {
  it('reads engine, ChatGPT, and ChatGPT with a model; anything else is not a choice', () => {
    expect(parseAgentModel({ model: { source: 'engine' } })).toEqual(ENGINE);
    expect(parseAgentModel({ model: { source: 'chatgpt' } })).toEqual({ source: 'chatgpt' });
    expect(parseAgentModel({ model: { source: 'chatgpt', model: ' gpt-5.5 ' } })).toEqual({ source: 'chatgpt', model: 'gpt-5.5' });
    expect(parseAgentModel({ model: { source: 'gemini' } })).toBeNull();
    expect(parseAgentModel({})).toBeNull();
    expect(sameModel({ source: 'chatgpt', model: 'a' }, { source: 'chatgpt', model: 'a' })).toBe(true);
    expect(sameModel({ source: 'chatgpt', model: 'a' }, { source: 'chatgpt' })).toBe(false);
    expect(sameModel(ENGINE, { source: 'chatgpt' })).toBe(false);
  });

  it('reads a provider with its model and the name the desktop gives it; one without either is not a choice', () => {
    expect(parseAgentModel({ model: { source: 'provider', provider: 'lm-studio', model: ' qwen3.5-9b ' }, providerName: 'LM Studio' })).toEqual({
      source: 'provider',
      provider: 'lm-studio',
      model: 'qwen3.5-9b',
      name: 'LM Studio',
    });
    expect(parseAgentModel({ model: { source: 'provider', provider: 'ollama', model: 'gemma4' } })).toEqual({ source: 'provider', provider: 'ollama', model: 'gemma4' });
    expect(parseAgentModel({ model: { source: 'provider', provider: 'lm-studio' } })).toBeNull();
    expect(parseAgentModel({ model: { source: 'provider', model: 'm' } })).toBeNull();
    const lm = { source: 'provider', provider: 'lm-studio', model: 'a' } as const;
    // Renamed, the same; another model or provider, not.
    expect(sameModel(lm, { ...lm, name: 'LM Studio' })).toBe(true);
    expect(sameModel(lm, { ...lm, model: 'b' })).toBe(false);
    expect(sameModel(lm, { ...lm, provider: 'ollama' })).toBe(false);
    expect(sameModel(lm, { source: 'chatgpt', model: 'a' })).toBe(false);
  });

  it('asks the desktop with its token; an older desktop (404) means the engine; one that cannot be asked keeps what was known (null)', async () => {
    const asked: Array<{ url: string; auth: string | null }> = [];
    const answer = (resp: Response | Error) => async (url: string, init: RequestInit) => {
      asked.push({ url, auth: new Headers(init.headers).get('authorization') });
      if (resp instanceof Error) throw resp;
      return resp;
    };
    expect(await readAgentModel(DESK, undefined, answer(json({ model: { source: 'chatgpt', model: 'gpt-5.5' } })))).toEqual({ source: 'chatgpt', model: 'gpt-5.5' });
    expect(asked[0]).toEqual({ url: 'http://127.0.0.1:17972/api/agent/preferences', auth: 'Bearer desk-token' });
    expect(await readAgentModel(DESK, undefined, answer(json({ error: 'no' }, 404)))).toEqual(ENGINE);
    expect(await readAgentModel(DESK, undefined, answer(json({ error: 'origin not allowed' }, 403)))).toBeNull();
    expect(await readAgentModel(DESK, undefined, answer(new TypeError('Failed to fetch')))).toBeNull();
  });
});

describe('the provider each kind of conversation runs on', () => {
  it('on the engine: every conversation, calls included, takes the app\'s own provider as it is (nothing changes)', () => {
    for (const kind of KINDS) expect(providerFor(kind, ENGINE, ENGINE_PROVIDER, DESK, null)).toBe(ENGINE_PROVIDER);
    // A ChatGPT choice with no desktop to reach it by is the engine too.
    expect(providerFor('project', { source: 'chatgpt' }, ENGINE_PROVIDER, null, 'gpt-5.5')).toBe(ENGINE_PROVIDER);
  });

  it('on ChatGPT: the generic route for everything but a call, as a custom OpenAI-style server with the desktop token as its key', () => {
    for (const kind of KINDS.filter((k) => k !== 'call')) {
      const p = providerFor(kind, { source: 'chatgpt', model: 'gpt-5.5' }, ENGINE_PROVIDER, DESK, null)!;
      expect(p).toMatchObject({
        type: 'custom',
        serverKind: 'other',
        name: 'ChatGPT',
        apiKey: 'desk-token',
        baseUrl: `http://127.0.0.1:17972/api/ai/providers/${CODEX_ROUTE}/v1`,
        modelId: 'gpt-5.5',
        contextTokens: CHATGPT_CONTEXT_TOKENS,
        parallelAgents: 1,
      });
      expect(p.followEngine).toBeUndefined();
    }
  });

  it('on a desktop AI provider: every conversation, calls included, through its gateway route with the desktop token, on the model chosen', () => {
    const choice = { source: 'provider', provider: 'lm-studio', model: 'qwen3.5-9b', name: 'LM Studio' } as const;
    for (const kind of KINDS) {
      const p = providerFor(kind, choice, ENGINE_PROVIDER, DESK, 'gpt-5.5')!;
      expect(p).toEqual(desktopProvider(DESK, choice));
      expect(p).toMatchObject({
        id: 'oaiy-provider-lm-studio',
        type: 'custom',
        serverKind: 'other',
        name: 'LM Studio',
        apiKey: 'desk-token',
        baseUrl: 'http://127.0.0.1:17972/api/ai/providers/lm-studio/v1',
        modelId: 'qwen3.5-9b',
        parallelAgents: 1,
      });
      expect(p.followEngine).toBeUndefined();
    }
    expect(modelChipText(desktopProvider(DESK, choice))).toBe('LM Studio · qwen3.5-9b');
    // No name from the desktop: its id.
    expect(desktopProvider(DESK, { source: 'provider', provider: 'ollama', model: 'gemma4' }).name).toBe('ollama');
    // No desktop to reach it by: the engine.
    expect(providerFor('project', choice, ENGINE_PROVIDER, null, null)).toBe(ENGINE_PROVIDER);
  });

  it("on ChatGPT with no model named: Codex's default, once it is known (and until then, none: the run looks it up first)", () => {
    expect(providerFor('setup', { source: 'chatgpt' }, ENGINE_PROVIDER, DESK, 'gpt-5.5')!.modelId).toBe('gpt-5.5');
    expect(providerFor('setup', { source: 'chatgpt' }, ENGINE_PROVIDER, DESK, null)!.modelId).toBe('');
  });

  it("a call takes the live-call route: reasoning off by default, Luna's fast one when the choice names Luna; never an empty model", () => {
    const call = providerFor('call', { source: 'chatgpt' }, ENGINE_PROVIDER, DESK, null)!;
    expect(call.baseUrl).toBe(`http://127.0.0.1:17972/api/ai/providers/${CALL_ROUTE}/v1`);
    expect(call.modelId).toBeTruthy();
    expect(call).toMatchObject({ type: 'custom', serverKind: 'other', apiKey: 'desk-token', contextTokens: CHATGPT_CONTEXT_TOKENS });
    expect(providerFor('call', { source: 'chatgpt', model: 'gpt-5.5' }, ENGINE_PROVIDER, DESK, null)!.baseUrl).toContain('/openai-codex-agent-none/v1');
    const luna = providerFor('call', { source: 'chatgpt', model: 'gpt-5.6-luna' }, ENGINE_PROVIDER, DESK, null)!;
    expect(luna.baseUrl).toBe(`http://127.0.0.1:17972/api/ai/providers/${LUNA_CALL_ROUTE}/v1`);
    expect(luna.modelId).toBe('gpt-5.6-luna');
    expect(callRoute('GPT-5.6-Luna')).toBe('openai-codex-agent-luna-low-fast');
    expect(callRoute(undefined)).toBe('openai-codex-agent-none');
    // The call and the rest are told apart (queues and remembered windows go by the id).
    expect(chatgptProvider(DESK, 'call', 'x').id).not.toBe(chatgptProvider(DESK, 'sms', 'x').id);
  });

  it('a call on ChatGPT is sent its tools, streamed, with no reasoning effort and none of llama.cpp\'s template switches', async () => {
    const fake = fakeProvider('openai', [{ text: 'Hello, how can I help?' }]);
    const call = providerFor('call', { source: 'chatgpt' }, ENGINE_PROVIDER, DESK, null)!;
    const tools = [{ name: 'end_call', description: 'Hang up.', parameters: { type: 'object' } }];
    await sendTurn(call, 'You are on a live phone call.', [{ role: 'user', text: 'Hi' }], tools, { reasoning: 'none' });
    expect(fake.urls[0]).toBe('http://127.0.0.1:17972/api/ai/providers/openai-codex-agent-none/v1/chat/completions');
    expect(fake.headers[0].authorization).toBe('Bearer desk-token');
    const body = fake.bodies[0];
    expect(body.stream).toBe(true);
    expect((body.tools as unknown[]).length).toBe(1);
    expect(body.model).toBe(call.modelId);
    expect(body.reasoning_effort).toBeUndefined();
    expect(body.chat_template_kwargs).toBeUndefined();
  });

  it('the model chip says which it is', () => {
    expect(modelChipText(chatgptProvider(DESK, 'project', 'gpt-5.5'))).toBe('ChatGPT · gpt-5.5');
    expect(modelChipText(chatgptProvider(DESK, 'project', ''))).toBe('ChatGPT · default model');
    expect(modelChipText(chatgptProvider(DESK, 'project', 'gpt-5.5'), 'sign in needed')).toBe('ChatGPT · sign in needed');
    expect(modelChipText(ENGINE_PROVIDER)).toBe('OAIY · Qwen3.8-Flash-Next');
    expect(modelChipText(null)).toBe('Set up AI…');
  });
});

describe("Codex's default model", () => {
  it("is the catalogue's isDefault row, else its first", () => {
    expect(pickCodexModel({ data: [{ id: 'gpt-5.6-luna' }, { id: 'gpt-5.5', isDefault: true }] })).toBe('gpt-5.5');
    expect(pickCodexModel({ data: [{ id: 'gpt-5.6-luna' }, { id: 'gpt-5.5' }] })).toBe('gpt-5.6-luna');
    expect(pickCodexModel({ data: [] })).toBeNull();
    expect(pickCodexModel(null)).toBeNull();
  });

  it("is read from the connector's model list with the desktop token, and nothing else is tried", async () => {
    const urls: string[] = [];
    const model = await codexDefaultModel(DESK, undefined, async (url, init) => {
      urls.push(url);
      expect(new Headers(init.headers).get('authorization')).toBe('Bearer desk-token');
      return json({ object: 'list', data: [{ id: 'gpt-5.6-luna', object: 'model' }, { id: 'gpt-5.5', object: 'model', isDefault: true }] });
    });
    expect(model).toBe('gpt-5.5');
    expect(urls).toEqual(['http://127.0.0.1:17972/api/ai/providers/openai-codex-agent/v1/models']);
  });

  it('signed out (428), says where to sign in, asking nowhere else (no Ollama-style fallback)', async () => {
    const urls: string[] = [];
    const send = async (url: string) => {
      urls.push(url);
      return json({ error: { code: 'codex_not_authenticated', message: 'not signed in to ChatGPT — start a login from OAIY Desktop → Providers' } }, 428);
    };
    await expect(codexDefaultModel(DESK, undefined, send)).rejects.toMatchObject({ message: CHATGPT_SIGN_IN, status: 428 });
    expect(urls).toHaveLength(1);
    await expect(codexDefaultModel(DESK, undefined, async () => json({ error: { code: 'codex_unavailable', message: 'codex is not installed' } }, 503))).rejects.toThrow('HTTP 503): codex is not installed');
  });
});

describe('signed out of ChatGPT (428)', () => {
  const signedOut = { status: 428, body: JSON.stringify({ error: { code: 'codex_not_authenticated', message: 'not signed in to ChatGPT — start a login from OAIY Desktop → Providers' } }) };

  it('a request is refused with where to sign in', async () => {
    fakeProvider('openai', [{ error: signedOut }]);
    const p = chatgptProvider(DESK, 'project', 'gpt-5.5');
    const error = await sendTurn(p, 'system', [{ role: 'user', text: 'hi' }], []).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(AIProviderError);
    expect(error).toMatchObject({ kind: 'http', status: 428, message: CHATGPT_SIGN_IN });
    expect(CHATGPT_SIGN_IN).toMatch(/Settings → Agent/);
  });

  it('so is a stream that stops with the same code', async () => {
    const stream = `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: '' } }] })}\n\ndata: ${JSON.stringify({ error: { code: 'codex_not_authenticated', message: 'not signed in' } })}\n\ndata: [DONE]\n\n`;
    vi.stubGlobal('fetch', async () => new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } }));
    const error = await sendTurn(chatgptProvider(DESK, 'setup', 'gpt-5.5'), 'system', [{ role: 'user', text: 'hi' }], []).catch((e: unknown) => e);
    expect(error).toMatchObject({ message: CHATGPT_SIGN_IN });
  });

  it('the Agent says so and stops: no retry, and nothing else is asked instead', async () => {
    const fake = fakeProvider('openai', [{ error: signedOut }, { text: 'should not be asked' }]);
    const events: AgentEvent[] = [];
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => chatgptProvider(DESK, 'setup', 'gpt-5.5'), projectSummary: () => '' });
    await agent.run('Set up my business phone', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(1);
    expect(fake.urls[0]).toContain('/api/ai/providers/openai-codex-agent/v1/chat/completions');
    expect(events.filter((e) => e.type === 'error')).toEqual([{ type: 'error', message: CHATGPT_SIGN_IN }]);
  });
});

describe('what must be ready before a run', () => {
  it('a failure is the run\'s error, and nothing is sent', async () => {
    const fake = fakeProvider('openai', [{ text: 'should not be asked' }]);
    const events: AgentEvent[] = [];
    const agent = new Agent({
      vfs: new Vfs(),
      gate: new NetGate(),
      provider: () => chatgptProvider(DESK, 'setup', ''),
      projectSummary: () => '',
      prepare: async () => {
        throw new AIProviderError('http', CHATGPT_SIGN_IN, { status: 428 });
      },
    });
    await agent.run('hello', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(0);
    expect(events).toEqual([{ type: 'error', message: CHATGPT_SIGN_IN }]);
    // Never "has no model chosen": the model is looked up before, or the sign-in is asked for.
    expect(JSON.stringify(events)).not.toMatch(/no model chosen/);
  });

  it('what it readies is used by the run (the model looked up)', async () => {
    const fake = fakeProvider('openai', [{ text: 'Hi!' }]);
    let model = '';
    const agent = new Agent({
      vfs: new Vfs(),
      gate: new NetGate(),
      provider: () => chatgptProvider(DESK, 'project', model),
      projectSummary: () => '',
      prepare: async () => {
        model = 'gpt-5.5';
      },
    });
    await agent.run('hello', () => {});
    expect(fake.bodies[0].model).toBe('gpt-5.5');
  });
});
