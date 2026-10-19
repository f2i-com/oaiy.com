// What a failed call to a provider says (shared/providers/errors.ts), as the Agent's calls give it their context
// (agent/providers/aiProvider.ts): OAIY's engines in OAIY's own window, a server by its name, a paired computer's.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { describeConnectionError } from '@oaiy/shared/providers/errors';
import { AIProviderError, postProviderJSON } from '../../src/agent/providers/aiProvider';
import type { ProviderConfig } from '../../src/agent/providers/types';

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
