// What a failed call to a provider says (shared/providers/errors.ts), as the Agent's calls give it their context
// (agent/providers/aiProvider.ts): OAIY's engines in OAIY's own window, a server by its name, a paired computer's.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { describeConnectionError } from '@oaiy/shared/providers/errors';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { AIProviderError, postProviderJSON } from '../../src/agent/providers/aiProvider';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { ANTHROPIC, OPENAI, fakeProvider } from './fakeProvider';

const ENGINE = 'http://127.0.0.1:17999/v1/chat/completions';

describe('a call that got no answer', () => {
  it("in OAIY's own window, OAIY's engines are not answering: no CORS advice, and OAIY, not Studio", () => {
    const text = describeConnectionError('network', { type: 'local', serverKind: 'oaiy', url: ENGINE, pageOrigin: 'oaiy://localhost', ownWindow: true });
    expect(text).toMatch(/^OAIY’s engines are not answering \(http:\/\/127\.0\.0\.1:17999\)\. Start them in OAIY’s Engines page/);
    expect(text).not.toMatch(/CORS|cors_origins|Studio/);
  });

  it('in a tab, an OAIY that does not answer may be closed to the page: where to allow it is said', () => {
    const text = describeConnectionError('network', { type: 'local', serverKind: 'oaiy', url: ENGINE, pageOrigin: 'https://agent.example' });
    expect(text).toContain('This page could not reach http://127.0.0.1:17999');
    expect(text).toContain('add https://agent.example to gateway.cors_origins');
    expect(text).not.toContain('Studio');
  });

  it("OAIY Desktop's own routes (ChatGPT, a paired computer's) are OAIY not answering, in its window and in a tab", () => {
    const route = 'http://127.0.0.1:17998/api/network/nodes/n1/v1/chat/completions';
    const own = describeConnectionError('network', { type: 'custom', serverKind: 'other', url: route, name: 'Studio Mac (OAIY Network)', ownWindow: true, viaDesktop: true });
    expect(own).toBe('OAIY is not answering for Studio Mac (OAIY Network) (http://127.0.0.1:17998). Check that OAIY is running, then try again.');
    const tab = describeConnectionError('network', { type: 'custom', serverKind: 'other', url: route, name: 'ChatGPT', pageOrigin: 'https://agent.example', viaDesktop: true });
    expect(tab).toMatch(/^OAIY Desktop is not answering for ChatGPT at http:\/\/127\.0\.0\.1:17998\./);
    expect(tab).not.toMatch(/CORS/);
  });

  it("another local server keeps its CORS advice in OAIY's window (Ollama does not let OAIY's window in by itself)", () => {
    const text = describeConnectionError('network', { type: 'local', serverKind: 'ollama', url: 'http://localhost:11434/v1/chat/completions', pageOrigin: 'oaiy://localhost', ownWindow: true, name: 'Ollama' });
    expect(text).toContain('OLLAMA_ORIGINS');
  });
});

describe('a server that answered with an error is named', () => {
  it("OAIY's engine is OAIY's engine, not the local server", () => {
    const text = describeConnectionError('server', { type: 'local', serverKind: 'oaiy', url: ENGINE, name: 'OAIY', detail: 'engine crashed' }, 500);
    expect(text).toBe('OAIY’s engine had an error of its own (500). Try again in a moment. The provider said: “engine crashed”');
  });

  it("a paired computer's, ChatGPT and a local server by their names; a cloud API by its kind", () => {
    expect(describeConnectionError('server', { type: 'custom', serverKind: 'other', url: ENGINE, name: 'Studio Mac (OAIY Network)' }, 500)).toMatch(/^Studio Mac \(OAIY Network\) had an error of its own \(500\)/);
    expect(describeConnectionError('http', { type: 'local', serverKind: 'lmstudio', url: ENGINE, name: 'LM Studio' }, 400)).toMatch(/^LM Studio answered with an error \(400\)/);
    // (no name: as it was)
    expect(describeConnectionError('server', { type: 'custom', url: ENGINE }, 502)).toMatch(/^The server had an error of its own \(502\)/);
    expect(describeConnectionError('server', { type: 'openai', url: 'https://api.openai.com/v1', name: 'my key' }, 500)).toMatch(/^OpenAI had an error/);
  });
});

describe("the Agent's own calls", () => {
  afterEach(() => vi.unstubAllGlobals());
  const engine: ProviderConfig = { id: 'p1', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:17999', modelId: 'qwen3.8-27b' };

  it("say which provider failed: OAIY's engine at a 500, a paired computer's by its name", async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: { message: 'out of memory' } }), { status: 500 })));
    const failed = await postProviderJSON(engine, ENGINE, {}, {}).catch((e: unknown) => e);
    expect(failed).toBeInstanceOf(AIProviderError);
    expect((failed as Error).message).toMatch(/^OAIY’s engine had an error of its own \(500\)/);
    const worker: ProviderConfig = { id: 'oaiy-worker-n1', type: 'custom', serverKind: 'other', name: 'Studio Mac (OAIY Network)', apiKey: 't', baseUrl: 'http://127.0.0.1:17998/api/network/nodes/n1/v1', modelId: 'm' };
    const theirs = await postProviderJSON(worker, `${worker.baseUrl}/chat/completions`, {}, {}).catch((e: unknown) => e);
    expect((theirs as Error).message).toMatch(/^Studio Mac \(OAIY Network\) had an error of its own \(500\)/);
  });

  it("an OAIY that is not running is a network failure (the page looks at the engines again on it), here with a tab's CORS words", async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    }));
    const failed = await postProviderJSON(engine, ENGINE, {}, {}).catch((e: unknown) => e);
    expect(failed instanceof AIProviderError && failed.kind).toBe('network');
    // (this file's page is not OAIY's window: no desktop was given before it was first read)
    expect((failed as Error).message).toContain('This page could not reach http://127.0.0.1:17999');
  });
});

describe('an overload said with a status, before any stream', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });
  const engine: ProviderConfig = { id: 'p1', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:17999', modelId: 'qwen3.8-27b' };
  // (Anthropic's when it is overloaded; OAIY's engine's when several conversations send it pictures at once, and when its
  // model worker has stopped, crates/oaiy-llm-server/src/api.rs)
  const overloaded = { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } };
  const busy = { error: { message: 'the server is busy decoding other pictures: try again shortly', type: 'invalid_request_error', code: 'busy' } };
  const stopped = { error: { message: 'the model worker has stopped', type: 'invalid_request_error', code: 'unavailable' } };

  it('is a rate limit, waited for (as long as it asks) and asked again; a worker that stopped is not', async () => {
    const failure = async (provider: ProviderConfig, status: number, body: unknown, headers: Record<string, string> = {}) => {
      vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } })));
      const error = (await postProviderJSON(provider, ENGINE, {}, {}).catch((e: unknown) => e)) as AIProviderError;
      return { kind: error.kind, status: error.status, wait: error.retryAfterMs };
    };
    expect(await failure(ANTHROPIC, 529, overloaded)).toEqual({ kind: 'rate-limited', status: 529, wait: undefined });
    expect(await failure(engine, 503, busy)).toEqual({ kind: 'rate-limited', status: 503, wait: undefined });
    expect(await failure(OPENAI, 503, { error: { message: 'The engine is currently overloaded, please try again later', type: 'server_error' } })).toEqual({ kind: 'rate-limited', status: 503, wait: undefined });
    expect(await failure(engine, 503, { error: { message: 'starting' } }, { 'retry-after': '2' })).toEqual({ kind: 'rate-limited', status: 503, wait: 2000 });
    // (no wait brings it back)
    expect(await failure(engine, 503, stopped)).toEqual({ kind: 'http', status: 503, wait: undefined });
    expect(await failure(engine, 500, { error: { message: 'engine crashed' } })).toEqual({ kind: 'http', status: 500, wait: undefined });
  });

  it("is waited for by the Agent, which asks again, as it does for the same overload inside a stream", async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    for (const [wire, provider, error] of [
      ['anthropic', ANTHROPIC, { status: 529, body: JSON.stringify(overloaded) }],
      ['openai', engine, { status: 503, body: JSON.stringify(busy) }],
    ] as const) {
      const fake = fakeProvider(wire, [{ error }, { text: 'ok' }]);
      const events: AgentEvent[] = [];
      const run = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => 'p' }).run('hi', (e) => events.push(e));
      // (the wait passes on the test's clock)
      for (let i = 0; i < 40 && fake.bodies.length < 2; i++) await vi.advanceTimersByTimeAsync(500);
      await run;
      expect(fake.bodies, wire).toHaveLength(2);
      expect(events.some((e) => e.type === 'status' && e.message === 'The provider is busy; retrying in 5 s'), wire).toBe(true);
      expect(events.at(-1), wire).toMatchObject({ type: 'done', text: 'ok' });
    }
    // A worker that stopped: said at once.
    const fake = fakeProvider('openai', [{ error: { status: 503, body: JSON.stringify(stopped) } }, { text: 'never asked' }]);
    const events: AgentEvent[] = [];
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => engine, projectSummary: () => 'p' }).run('hi', (e) => events.push(e));
    expect(fake.bodies).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'error', message: expect.stringContaining('the model worker has stopped') });
  });
});
