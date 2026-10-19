// In OAIY's own window (its desktop given before the page first asks which window it is), the Agent's call to OAIY's
// engines that gets no answer says they are not answering: OAIY lets its window in, so it is never about CORS.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AIProviderError, postProviderJSON } from '../../src/agent/providers/aiProvider';
import { HELP } from '../../src/commands';
import type { ProviderConfig } from '../../src/agent/providers/types';

// (given before anything asks which window this is, as OAIY's window gives it before the page runs)
vi.hoisted(() => {
  (globalThis as { __OAIY_DESKTOP__?: unknown }).__OAIY_DESKTOP__ = { origin: 'http://127.0.0.1:17998', token: 'test-token' };
});

describe("OAIY's own window", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("a stopped engine: OAIY's engines are not answering, and nothing about CORS or Studio", async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    }));
    const engine: ProviderConfig = { id: 'p1', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:17999', modelId: 'qwen3.8-27b' };
    const failed = await postProviderJSON(engine, 'http://127.0.0.1:17999/v1/chat/completions', {}, {}).catch((e: unknown) => e);
    expect(failed instanceof AIProviderError && failed.kind).toBe('network');
    expect((failed as Error).message).toBe("OAIY’s engines are not answering (http://127.0.0.1:17999). Start them in OAIY’s Engines page, or wait while they start, then try again.");
    const chatgpt: ProviderConfig = { id: 'oaiy-chatgpt', type: 'custom', serverKind: 'other', name: 'ChatGPT', apiKey: 't', baseUrl: 'http://127.0.0.1:17998/api/ai/providers/codex/v1', modelId: 'gpt-5.5' };
    const theirs = await postProviderJSON(chatgpt, `${chatgpt.baseUrl}/chat/completions`, {}, {}).catch((e: unknown) => e);
    expect((theirs as Error).message).toBe('OAIY is not answering for ChatGPT (http://127.0.0.1:17998). Check that OAIY is running, then try again.');
  });

  it("Settings' model list, with the engines stopped: the same words, not CORS advice", async () => {
    const { listModels } = await import('../../src/agent/providers/providerConnection');
    const fetchImpl = vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    }) as unknown as typeof fetch;
    const engine = { type: 'local' as const, serverKind: 'oaiy' as const, apiKey: '', baseUrl: 'http://127.0.0.1:17999' };
    const failed = (await listModels(engine, { fetchImpl, ownWindow: true }).catch((e: unknown) => e)) as Error;
    expect(failed.message).toBe("OAIY’s engines are not answering (http://127.0.0.1:17999). Start them in OAIY’s Engines page, or wait while they start, then try again.");
    // (a browser tab is another matter: there it may be OAIY refusing the page)
    const tab = (await listModels(engine, { fetchImpl }).catch((e: unknown) => e)) as Error;
    expect(tab.message).toMatch(/CORS/);
  });

  it('/help says where everything runs here (this window, not a tab), and lists /softn', () => {
    expect(HELP).toContain('Everything runs in this window, on this computer');
    expect(HELP).not.toContain('this tab');
    expect(HELP).toContain('/softn …');
  });
});
