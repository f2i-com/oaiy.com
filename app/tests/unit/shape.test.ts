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

describe("OAIY's Engines choose the model", () => {
  it('a provider that follows them names no model, so the engine answers with the one chosen there', async () => {
    const fake = fakeProvider('openai', [{ text: 'Hi.' }, { text: 'Hi again.' }]);
    const provider = { ...OPENAI, type: 'local' as const, serverKind: 'oaiy' as const, baseUrl: 'http://127.0.0.1:8080', modelId: 'Qwen3.8-Flash-Next', followEngine: true };
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => '', conversation: true });
    await agent.run('Hello', () => {});
    expect(fake.bodies[0]).not.toHaveProperty('model');
    provider.followEngine = false;
    await agent.run('Hello again', () => {});
    expect(fake.bodies[1].model).toBe('Qwen3.8-Flash-Next');
  });
});

describe('a reply the person stopped', () => {
  it('is kept as far as it got, and the model is told it was stopped, so the next request is not joined to it', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const fake = fakeProvider('openai', [{ text: 'Here is the first half of the plan, and', hold: { at: 20, until: held } }, { text: 'Sure.' }]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...OPENAI, baseUrl: 'https://example.test/v1' }), projectSummary: () => 'p', keepStopped: true });
    const stop = new AbortController();
    const events: AgentEvent[] = [];
    await agent.run('Write a long plan', (e) => {
      events.push(e);
      if (e.type === 'text' && !stop.signal.aborted) setTimeout(() => stop.abort(), 20);
    }, stop.signal);
    release();
    expect(events.some((e) => e.type === 'notice' && e.message === 'Stopped.')).toBe(true);
    expect(agent.turns.map((t) => t.role)).toEqual(['user', 'assistant', 'user']);
    expect((agent.turns[1] as Extract<Turn, { role: 'assistant' }>).text).toBe('Here is the first ha');
    await agent.run('Just say hi instead', () => {});
    const sent = (fake.bodies[1].messages as Array<{ role: string; content: unknown }>).filter((m) => m.role !== 'system');
    const text = (m: { content: unknown }) => (typeof m.content === 'string' ? m.content : JSON.stringify(m.content));
    expect(sent.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(text(sent[0])).toContain('Write a long plan');
    expect(text(sent[0])).not.toContain('Just say hi');
    expect(text(sent[1])).toBe('Here is the first ha');
    expect(text(sent[2])).toMatch(/^\[OAIY\] The person stopped this reply[\s\S]*Just say hi instead/);
  });

  it('stopped before it wrote anything, is still a request the model is told was stopped', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const fake = fakeProvider('openai', [{ text: 'Never shown', hold: { at: 0, until: held } }, { text: 'Sure.' }]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...OPENAI, baseUrl: 'https://example.test/v1' }), projectSummary: () => 'p', keepStopped: true });
    const stop = new AbortController();
    await agent.run('First request', (e) => {
      if (e.type === 'prompt') setTimeout(() => stop.abort(), 30);
    }, stop.signal);
    release();
    expect(agent.turns.map((t) => t.role)).toEqual(['user', 'user']);
    await agent.run('Second request', () => {});
    const sent = (fake.bodies[1].messages as Array<{ role: string; content: unknown }>).filter((m) => m.role !== 'system');
    const text = (m: { content: unknown }) => (typeof m.content === 'string' ? m.content : JSON.stringify(m.content));
    expect(text(sent[0])).toMatch(/First request[\s\S]*\[OAIY\] The person stopped this reply[\s\S]*Second request/);
  });

  it('on the phone (no keepStopped), nothing is added: its sessions keep what the caller heard', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    fakeProvider('openai', [{ text: 'We are open from nine.', hold: { at: 10, until: held } }]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...OPENAI, baseUrl: 'https://example.test/v1' }), projectSummary: () => 'p', conversation: true });
    const stop = new AbortController();
    await agent.run('When are you open?', (e) => {
      if (e.type === 'text' && !stop.signal.aborted) setTimeout(() => stop.abort(), 20);
    }, stop.signal);
    release();
    expect(agent.turns.map((t) => t.role)).toEqual(['user']);
  });
});
