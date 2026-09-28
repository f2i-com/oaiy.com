import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, wellFormed, type AgentEvent } from '../../src/agent/agent';
import { outputLimit } from '../../src/agent/context';
import type { Turn } from '../../src/agent/protocol';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

describe('a conversation every provider accepts', () => {
  it('answers calls left without results, drops empty replies, and joins user turns', () => {
    const turns: Turn[] = [
      { role: 'user', text: 'one' },
      { role: 'assistant', text: '', calls: [{ id: 'a', name: 'read_file', input: {} }, { id: 'b', name: 'read_file', input: {} }] },
      { role: 'tool', results: [{ id: 'a', name: 'read_file', content: 'ok', isError: false }] },
      { role: 'assistant', text: '', calls: [] },
      { role: 'user', text: 'two' },
      { role: 'assistant', text: '', calls: [{ id: 'c', name: 'grep', input: {} }] },
      { role: 'user', text: 'three' },
      { role: 'user', text: 'four' },
    ];
    const out = wellFormed(turns);
    expect(out.map((t) => t.role)).toEqual(['user', 'assistant', 'tool', 'user', 'assistant', 'tool', 'user']);
    const firstResults = (out[2] as Extract<Turn, { role: 'tool' }>).results;
    expect(firstResults.map((r) => [r.id, r.isError])).toEqual([['a', false], ['b', true]]);
    expect((out[5] as Extract<Turn, { role: 'tool' }>).results[0]).toMatchObject({ id: 'c', isError: true });
    expect((out[6] as Extract<Turn, { role: 'user' }>).text).toBe('three\n\nfour');
  });
});

describe('the output limit', () => {
  it('is read out of servers’ errors', () => {
    expect(outputLimit('max_tokens: 16384 > 8192, which is the maximum allowed number of output tokens for claude-3-5-sonnet')).toBe(8192);
    expect(outputLimit('Invalid max_tokens value, the valid range of max_tokens is [1, 8192]')).toBe(8192);
    expect(outputLimit('This model supports at most 4096 completion tokens, whereas you provided 16384.')).toBe(4096);
    expect(outputLimit('invalid tools field')).toBeNull();
  });

  it('is kept to after a server refuses more, and the request is sent again', async () => {
    const fake = fakeProvider('openai', [
      { error: { status: 400, body: JSON.stringify({ error: { message: 'This model supports at most 4096 completion tokens, whereas you provided 16384.' } }) } },
      { text: 'fine' },
    ]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...OPENAI, modelId: 'gpt-4o', baseUrl: 'https://example.test/v1' }), projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run('hi', (e) => events.push(e));
    expect(fake.bodies[1].max_tokens).toBe(4096);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'fine' });
  });
});
