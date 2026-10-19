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

describe("the paired computers' language models (OAIY Network)", () => {
  it("each with its computer, whether it answers, and its window where the desktop says them", async () => {
    const llm = { kind: 'llm', locality: 'network', default: false };
    relay([[200, { running: true, services: [
      { ...llm, id: 'engine:llm:node.a1b2c3d4/Qwen3.8-Flash-Next', model: 'Qwen3.8-Flash-Next', nodeId: 'a1b2c3d4', nodeName: 'Studio Mac', ready: true, nodeOnline: true, contextTokens: 131_072 },
      // (the computer answers, its engines do not)
      { ...llm, id: 'engine:llm:node.e5f6/Qwen3.8-27B', model: 'Qwen3.8-27B', nodeId: 'e5f6', nodeName: 'Office PC', ready: false, nodeOnline: true },
      // (a desktop from before says neither: nothing is made up)
      { ...llm, id: 'engine:llm:node.c0ffee/Gemma-4', model: 'Gemma-4', nodeId: 'c0ffee', ready: false, contextTokens: 'big' },
      // (this computer's own, and a paired computer's picture model, are not offered)
      { kind: 'llm', id: 'engine:llm/qwen3.5-9b', model: 'qwen3.5-9b', locality: 'local', ready: true },
      { kind: 'image', id: 'engine:image:node.a1b2c3d4/flux', model: 'flux', nodeId: 'a1b2c3d4', nodeName: 'Studio Mac', ready: true },
    ] }]]);
    const models = await new Desktop('http://127.0.0.1:17972', 'tok').networkModels();
    expect(models).toEqual([
      { id: 'engine:llm:node.a1b2c3d4/Qwen3.8-Flash-Next', model: 'Qwen3.8-Flash-Next', node: 'a1b2c3d4', nodeName: 'Studio Mac', ready: true, nodeOnline: true, contextTokens: 131_072, chosen: false },
      { id: 'engine:llm:node.e5f6/Qwen3.8-27B', model: 'Qwen3.8-27B', node: 'e5f6', nodeName: 'Office PC', ready: false, nodeOnline: true, chosen: false },
      { id: 'engine:llm:node.c0ffee/Gemma-4', model: 'Gemma-4', node: 'c0ffee', nodeName: 'another computer', ready: false, chosen: false },
    ]);
  });
});
