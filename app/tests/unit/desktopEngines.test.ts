import { afterEach, describe, expect, it, vi } from 'vitest';
import { Desktop } from '../../src/desktop/bridge';

afterEach(() => vi.unstubAllGlobals());

/** The desktop's engines relay, answering each request with the next of `replies` (a status and a body). */
function relay(replies: Array<[number, unknown]>) {
  const requests: Array<{ url: string; init: RequestInit }> = [];
  vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit) => {
    requests.push({ url, init });
    const [status, body] = replies.shift() ?? [500, { error: 'no reply' }];
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  }));
  return requests;
}

describe("the engines' models, through the desktop", () => {
  it('chooses the language model the engines run, as their Engines page does', async () => {
    const requests = relay([[200, { group: 'llm', model: 'qwen3.5-9b' }], [400, { error: 'llm.default_model nope is not one of the listed models' }]]);
    const desktop = new Desktop('http://127.0.0.1:17972', 'tok');
    await desktop.chooseEngineModel('qwen3.5-9b');
    expect(requests[0].url).toBe('http://127.0.0.1:17972/api/engines/defaults');
    expect(requests[0].init.method).toBe('PUT');
    expect(JSON.parse(String(requests[0].init.body))).toEqual({ group: 'llm', model: 'qwen3.5-9b' });
    expect((requests[0].init.headers as Record<string, string>).authorization).toBe('Bearer tok');
    // The engines' reason for refusing is the error.
    await expect(desktop.chooseEngineModel('nope')).rejects.toThrow('is not one of the listed models');
  });
});
