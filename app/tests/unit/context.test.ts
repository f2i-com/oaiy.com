import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { budgetFor, contextWindow, overflowWindow } from '../../src/agent/context';
import type { Turn } from '../../src/agent/protocol';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { ANTHROPIC, OPENAI, fakeProvider, type Step } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

describe('the context window', () => {
  it('comes from the person first, then the server, then what is known of the model', () => {
    expect(contextWindow({ ...OPENAI, modelId: 'gpt-4.1', contextTokens: 50_000 })).toEqual({ tokens: 50_000, source: 'yours' });
    expect(contextWindow({ ...OPENAI, modelId: 'm', detectedContext: { model: 'm', tokens: 262_144, how: 'x', at: 0 } })).toEqual({ tokens: 262_144, source: 'server' });
    // Detected for another model: not used.
    expect(contextWindow({ ...OPENAI, modelId: 'gpt-4o', detectedContext: { model: 'other', tokens: 9000, how: 'x', at: 0 } }).tokens).toBe(128_000);
    expect(contextWindow({ ...ANTHROPIC, modelId: 'claude-sonnet-4-5' }).source).toBe('known');
    expect(contextWindow({ id: 'l', type: 'local', name: 'x', apiKey: '', modelId: 'qwen3' })).toEqual({ tokens: 8192, source: 'default' });
  });

  it("reads the window out of servers' overflow errors", () => {
    expect(overflowWindow("This model's maximum context length is 32768 tokens. However, you requested 40000 tokens")).toEqual({ overflow: true, tokens: 32768 });
    expect(overflowWindow('the request exceeds the available context size (8192 tokens), try increasing it')).toEqual({ overflow: true, tokens: 8192 });
    expect(overflowWindow('prompt is too long: 250000 tokens > 200000 maximum').overflow).toBe(true);
    expect(overflowWindow('invalid tools field').overflow).toBe(false);
  });

  it('keeps a quarter of the window (at most the output limit) for the reply', () => {
    expect(budgetFor(8192, 3000)).toEqual({ reply: 2048, prompt: 3144 });
    expect(budgetFor(262_144, 5000)).toEqual({ reply: 16_384, prompt: 240_760 });
  });
});

/** A model that follows `steps`, and writes a summary whenever it is asked for one. */
function scripted(steps: Step[]): { script: Array<(body: Record<string, unknown>) => Step>; summaries: () => number } {
  let summaries = 0;
  let i = 0;
  const respond = (body: Record<string, unknown>): Step => {
    if (JSON.stringify(body).includes('You compress the conversation')) {
      summaries++;
      return { text: `SUMMARY ${summaries}: the user wants the big files read; some were read.` };
    }
    return steps[i++] ?? { text: 'out of script' };
  };
  return { script: Array.from({ length: 40 }, () => respond), summaries: () => summaries };
}

describe.each([
  ['anthropic', ANTHROPIC],
  ['openai', OPENAI],
] as const)('compaction over %s', (wire, base) => {
  it('summarizes older turns before the prompt passes the threshold, and keeps tool results with their calls', async () => {
    const provider: ProviderConfig = { ...base, contextTokens: 14_000 };
    const vfs = new Vfs();
    for (let n = 1; n <= 6; n++) vfs.writeFile(`/big${n}.txt`, `file ${n} line\n`.repeat(260));
    const steps: Step[] = [1, 2, 3, 4, 5, 6].map((n) => ({ calls: [{ name: 'read_file', input: { path: `big${n}.txt` } }] }));
    steps.push({ text: 'Read them all.' });
    const { script, summaries } = scripted(steps);
    const fake = fakeProvider(wire, script);
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => provider, projectSummary: () => 'Project: big files' });
    const events: AgentEvent[] = [];
    await agent.run('read every big file', (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Read them all.' });
    expect(summaries()).toBeGreaterThanOrEqual(1);
    const compacts = events.filter((e) => e.type === 'compact') as Array<Extract<AgentEvent, { type: 'compact' }>>;
    expect(compacts.length).toBeGreaterThanOrEqual(1);
    expect(compacts[0].after).toBeLessThan(compacts[0].before);
    expect(compacts[0].how).toBe('summary');
    // The whole conversation is kept for the chat; the model reads from the latest summary.
    const view = agent.view();
    expect(view[0]).toMatchObject({ role: 'user', summary: true, automatic: true });
    expect((view[0] as Extract<Turn, { role: 'user' }>).text).toContain('Project: big files');
    view.forEach((t, i) => {
      if (t.role === 'tool') expect(view[i - 1].role).toBe('assistant');
    });
    expect(agent.turns.length).toBeGreaterThan(view.length);
    // The last request carries the summary, not the first file's contents.
    const last = JSON.stringify(fake.bodies.at(-1));
    expect(last).toContain('SUMMARY');
    expect(last).not.toContain('file 1 line');
    // Every request stayed inside the window (a character is at least a third of a token here).
    for (const body of fake.bodies) expect(JSON.stringify(body).length / 3).toBeLessThan(14_000);
  });
});

describe('when the server says the prompt is too long', () => {
  it('remembers the window it states, compacts and tries again', async () => {
    const provider: ProviderConfig = { ...OPENAI, modelId: 'gpt-4o' };
    const vfs = new Vfs();
    for (let n = 1; n <= 3; n++) vfs.writeFile(`/big${n}.txt`, `file ${n} line\n`.repeat(300));
    const { script } = scripted([
      { calls: [{ name: 'read_file', input: { path: 'big1.txt' } }] },
      { calls: [{ name: 'read_file', input: { path: 'big2.txt' } }] },
      { calls: [{ name: 'read_file', input: { path: 'big3.txt' } }] },
      { error: { status: 400, body: JSON.stringify({ error: { message: "This model's maximum context length is 12000 tokens. However, your messages resulted in 13000 tokens." } }) } },
      { text: 'Done after compacting.' },
    ]);
    fakeProvider('openai', script);
    const windows: number[] = [];
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => provider, projectSummary: () => 'p', onWindow: (t) => windows.push(t) });
    const events: AgentEvent[] = [];
    await agent.run('read them', (e) => events.push(e));
    expect(windows).toEqual([12_000]);
    expect(events.some((e) => e.type === 'compact')).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Done after compacting.' });
  });
});

describe('a single huge message', () => {
  it('is not summarized away: the request stays, trimmed if need be', async () => {
    const provider: ProviderConfig = { ...OPENAI, contextTokens: 12_000 };
    const { script, summaries } = scripted([{ text: 'Got it.' }]);
    fakeProvider('openai', script);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run(`Please look at this log:\n${'2026-09-26 12:00:00 INFO something happened\n'.repeat(2500)}`, (e) => events.push(e));
    expect(summaries()).toBe(0);
    expect(agent.turns.some((t) => t.role === 'user' && t.summary)).toBe(false);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Got it.' });
  });
});
