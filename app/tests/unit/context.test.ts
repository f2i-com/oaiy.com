import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent, type SessionTool } from '../../src/agent/agent';
import { budgetFor, contextWindow, detectContextWindow, overflowWindow } from '../../src/agent/context';
import type { Turn } from '../../src/agent/protocol';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { ANTHROPIC, LOCAL, OPENAI, fakeProvider, type Step } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

const PICTURE = { mediaType: 'image/png' as const, data: 'iVBORw0KGgo=' };
const sse = (events: unknown[]) => events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('');
/** A streamed reply of plain text. */
const streamedText = (text: string) => () =>
  new Response(`${sse([{ choices: [{ index: 0, delta: { content: text } }] }, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }])}data: [DONE]\n\n`, { status: 200, headers: { 'content-type': 'text/event-stream' } });
/** A fetch that answers each request with the next of `answers`, keeping the bodies sent. */
function answering(answers: Array<() => Response>): Array<Record<string, unknown>> {
  const bodies: Array<Record<string, unknown>> = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    bodies.push(JSON.parse(String(init.body)) as Record<string, unknown>);
    const next = answers[bodies.length - 1];
    if (!next) throw new Error(`no answer for request ${bodies.length}`);
    return next();
  });
  return bodies;
}
const tooLong = (events: AgentEvent[]) => events.some((e) => e.type === 'status' && /too long for the model/.test(e.message));

describe('the context window', () => {
  it('comes from the person first, then the server, then what is known of the model', () => {
    expect(contextWindow({ ...OPENAI, modelId: 'gpt-4.1', contextTokens: 50_000 })).toEqual({ tokens: 50_000, source: 'yours' });
    expect(contextWindow({ ...OPENAI, modelId: 'm', detectedContext: { model: 'm', tokens: 262_144, how: 'x', at: 0 } })).toEqual({ tokens: 262_144, source: 'server' });
    // Detected for another model: not used.
    expect(contextWindow({ ...OPENAI, modelId: 'gpt-4o', detectedContext: { model: 'other', tokens: 9000, how: 'x', at: 0 } }).tokens).toBe(128_000);
    expect(contextWindow({ ...ANTHROPIC, modelId: 'claude-sonnet-4-5' }).source).toBe('known');
    expect(contextWindow({ id: 'l', type: 'local', name: 'x', apiKey: '', modelId: 'qwen3' })).toEqual({ tokens: 8192, source: 'default' });
  });

  it("is OAIY's own engine's default, for a model of OAIY's that has not said yet", async () => {
    // OAIY tells a model's window once the model is loaded. Planned for 8,192, and then 16,384, a first message could
    // not be sent (the instructions and tools alone, some 11,500 tokens, left too little), so the model was never
    // loaded and never said.
    const oaiy: ProviderConfig = { id: 'o', type: 'local', serverKind: 'oaiy', name: 'OAIY', apiKey: '', baseUrl: 'http://127.0.0.1:8080', modelId: 'qwen3.5-9b', followEngine: true };
    expect(contextWindow(oaiy)).toEqual({ tokens: 65_536, source: 'default' });
    expect(budgetFor(contextWindow(oaiy).tokens, 7000).prompt).toBeGreaterThan(1024 * 4);
    // the instructions and tools as they are now leave the conversation room
    expect(budgetFor(contextWindow(oaiy).tokens, 11_500).prompt).toBeGreaterThan(1024 * 16);
    // What OAIY says, once it does, and what the person sets, still come first.
    expect(contextWindow({ ...oaiy, detectedContext: { model: 'qwen3.5-9b', tokens: 65_536, how: "OAIY's engine", at: 0 } }).tokens).toBe(65_536);
    expect(contextWindow({ ...oaiy, contextTokens: 32_000 })).toEqual({ tokens: 32_000, source: 'yours' });

    // Asked before the model is loaded, OAIY's discovery names no window: a guess, to be asked again after a reply.
    const answers: Array<Record<string, unknown>> = [{ service: 'oaiy-studio', llm: { context_tokens: null } }, { service: 'oaiy-studio', llm: { context_tokens: 16_384 } }];
    vi.stubGlobal('fetch', async (url: string) => {
      expect(String(url)).toBe('http://127.0.0.1:8080/v1/discovery');
      return new Response(JSON.stringify(answers.shift()), { status: 200, headers: { 'content-type': 'application/json' } });
    });
    expect(await detectContextWindow(oaiy)).toEqual({ tokens: 65_536, how: 'OAIY, before the model is loaded', guess: true });
    expect(await detectContextWindow(oaiy)).toEqual({ tokens: 16_384, how: "OAIY's engine" });
  });

  it("reads the window out of servers' overflow errors", () => {
    expect(overflowWindow("This model's maximum context length is 32768 tokens. However, you requested 40000 tokens")).toEqual({ overflow: true, tokens: 32768 });
    // oaiy-llm-server's own (a model on the eGPU whose card holds less than it was asked to)
    expect(overflowWindow("the prompt is 91093 tokens, and m27's maximum context length is 81920 tokens here")).toEqual({ overflow: true, tokens: 81920 });
    expect(overflowWindow('the request exceeds the available context size (8192 tokens), try increasing it')).toEqual({ overflow: true, tokens: 8192 });
    expect(overflowWindow('prompt is too long: 250000 tokens > 200000 maximum').overflow).toBe(true);
    expect(overflowWindow('invalid tools field').overflow).toBe(false);
    // oaiy-llm-server's for a prompt its pictures take past the context: as it words it now (as one of text), and as an
    // engine installed before still does (Gemma and GLM; Qwen and MiniCPM-V)
    expect(overflowWindow("the prompt with its pictures is 70000 tokens, and the model's maximum context length is 65536 tokens here")).toEqual({ overflow: true, tokens: 65_536 });
    expect(overflowWindow('the prompt with its pictures (70000 tokens) is past the context (65536)')).toEqual({ overflow: true, tokens: 65_536 });
    expect(overflowWindow('images exceed configured context')).toEqual({ overflow: true, tokens: null });
  });

  it('keeps a quarter of the window (a fifth of a small one; at most the output limit) for the reply', () => {
    expect(budgetFor(8192, 3000)).toEqual({ reply: 1638, prompt: 3554 });
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

  it("does the same for OAIY's engine refusing a prompt its pictures take past the context: by its code, and by an older engine's words", async () => {
    const provider: ProviderConfig = { ...LOCAL, serverKind: 'oaiy', modelId: 'qwen3-vl', contextTokens: 32_000 };
    const refuse = (message: string, code: string) => () => new Response(JSON.stringify({ error: { message, type: 'invalid_request_error', code } }), { status: 400, headers: { 'content-type': 'application/json' } });
    for (const [message, code] of [
      // (as the engine words it now, as one of text, with that code)
      ["the prompt with its pictures is 33000 tokens, and the model's maximum context length is 30000 tokens here", 'context_length_exceeded'],
      // (as an engine installed before words it: a bad request, by its words alone)
      ['the prompt with its pictures (33000 tokens) is past the context (30000)', 'invalid_request'],
      ['images exceed configured context', 'invalid_request'],
    ]) {
      const bodies = answering([refuse(message, code), streamedText('A red fox.')]);
      const windows: number[] = [];
      const events: AgentEvent[] = [];
      await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => 'p', onWindow: (t) => windows.push(t) }).run('what is in this picture?', (e) => events.push(e), undefined, [PICTURE]);
      expect(tooLong(events), message).toBe(true);
      expect(bodies, message).toHaveLength(2);
      expect(events.at(-1), message).toMatchObject({ type: 'done', text: 'A red fox.' });
      // the window it states is learned (a model that does not take pictures it is not taken for)
      expect(windows, message).toEqual(/30000/.test(message) ? [30_000] : []);
      expect(events.some((e) => e.type === 'status' && /does not take images/.test(e.message)), message).toBe(false);
    }
  });

  it('does the same for an overflow said inside a stream that had begun', async () => {
    const provider: ProviderConfig = { ...LOCAL, serverKind: 'other', modelId: 'm', contextTokens: 32_000 };
    const words = 'the request exceeds the available context size (8192 tokens), try increasing it';
    // An older llama.cpp: a 200 and an event stream, then an error event for a prompt past its context.
    for (const error of [{ code: 400, message: words, type: 'exceed_context_size_error' }, { message: words }]) {
      const bodies = answering([() => new Response(sse([{ error }]), { status: 200, headers: { 'content-type': 'text/event-stream' } }), streamedText('ok')]);
      const windows: number[] = [];
      const events: AgentEvent[] = [];
      await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => 'p', onWindow: (t) => windows.push(t) }).run('hello', (e) => events.push(e));
      expect(tooLong(events), JSON.stringify(error)).toBe(true);
      expect(bodies).toHaveLength(2);
      expect(windows).toEqual([8192]);
      expect(events.at(-1)).toMatchObject({ type: 'done', text: 'ok' });
    }
  });
});

describe('a single huge message', () => {
  const LOG = `Please look at this log:\n${'2026-09-26 12:00:00 INFO something happened\n'.repeat(2500)}and tell me what failed last.`;

  it('is not summarized away: the request stays, its start and its end, cut to fit', async () => {
    const provider: ProviderConfig = { ...OPENAI, contextTokens: 12_000 };
    const { script, summaries } = scripted([{ text: 'Got it.' }]);
    const fake = fakeProvider('openai', script);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run(LOG, (e) => events.push(e));
    expect(summaries()).toBe(0);
    expect(agent.turns.some((t) => t.role === 'user' && t.summary)).toBe(false);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Got it.' });
    // What was sent fits the window (at 3.5 characters a token, as the agent estimates), and keeps both ends of the log.
    const sent = JSON.stringify(fake.bodies[0]);
    expect(sent.length / 3.5).toBeLessThan(12_000);
    expect(sent).toContain('Please look at this log');
    expect(sent).toContain('tell me what failed last.');
    expect(sent).toMatch(/\[… \d+ characters left out here to fit the model's context …\]/);
    // The conversation keeps it whole, for the chat.
    expect(agent.turns[0].role === 'user' && agent.turns[0].text.endsWith('tell me what failed last.')).toBe(true);
  });

  it('is answered by a server that refuses a prompt past its window, and so is a Try again', async () => {
    const window = 12_000;
    const provider: ProviderConfig = { ...OPENAI, contextTokens: window };
    // (a server that counts 3.5 characters a token, and refuses a prompt over its window as OpenAI words it)
    const respond = (body: Record<string, unknown>): Step => {
      const tokens = Math.round(JSON.stringify(body).length / 3.5);
      return tokens > window ? { error: { status: 400, body: JSON.stringify({ error: { message: `This model's maximum context length is ${window} tokens. However, your messages resulted in ${tokens} tokens.` } }) } } : { text: 'Got it.' };
    };
    const fake = fakeProvider('openai', Array.from({ length: 6 }, () => respond));
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => provider, projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run(LOG, (e) => events.push(e));
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Got it.' });
    expect(fake.bodies).toHaveLength(1);
    const again: AgentEvent[] = [];
    await agent.run('', (e) => again.push(e), undefined, [], [], true);
    expect(again.at(-1)).toMatchObject({ type: 'done', text: 'Got it.' });
    expect(fake.bodies).toHaveLength(2);
  });
});

describe('compaction and the request being worked on', () => {
  /** A model that follows `main`, and answers every summary request with `summary`. */
  function model(summary: string, main: Step[]) {
    const isSummary = (body: Record<string, unknown>) => JSON.stringify(body).includes('You compress the conversation');
    let i = 0;
    const fake = fakeProvider('openai', Array.from({ length: 20 }, () => (body: Record<string, unknown>): Step => (isSummary(body) ? { text: summary } : main[i++] ?? { text: 'out of script' })));
    return { fake, summaries: () => fake.bodies.filter(isSummary).length, main: () => fake.bodies.filter((b) => !isSummary(b)) };
  }
  /** A tool of the conversation whose one result is big (a server's log). */
  const bigLog: SessionTool = { spec: { name: 'fetch_log', description: 'Fetch the server log.', parameters: { type: 'object', properties: {} } }, run: async () => 'GET /login 200 12ms\n'.repeat(2300) };
  // A request whose last words matter, longer than the 300 characters a request without a summary was cut to.
  const REQUEST = `Fix the sign-in form in src/login.html. ${'Keep the layout and the colours exactly as they are; '.repeat(6)}and last of all: the submit button must read exactly "Sign in now".`;
  const agentFor = (window: number) => new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...OPENAI, contextTokens: window }), projectSummary: () => 'p', sessionTools: [bigLog] });

  it('leaves a request with one big step after it as it is: nothing to summarize, the big result is cut', async () => {
    for (const request of [REQUEST, 'Tidy the log.']) {
      const { fake, summaries, main } = model('SUMMARY: the user wants the sign-in form fixed.', [{ calls: [{ name: 'fetch_log', input: {} }] }, { text: 'done' }]);
      const agent = agentFor(16_000);
      const events: AgentEvent[] = [];
      await agent.run(request, (e) => events.push(e));
      expect(events.at(-1)).toMatchObject({ type: 'done', text: 'done' });
      // (the request summarized on its own was a request paid for, and a longer prompt: "Tidy the log." became a summary)
      expect(summaries()).toBe(0);
      expect(events.some((e) => e.type === 'compact')).toBe(false);
      expect(agent.view()[0]).toMatchObject({ role: 'user', text: expect.stringContaining(request) });
      const last = JSON.stringify(main().at(-1));
      expect(last).toContain(request.includes('Sign in now') ? 'Sign in now' : 'Tidy the log.');
      expect(last).toContain('[cut to fit the model');
      expect(JSON.stringify(main().at(-1)).length / 3).toBeLessThan(16_000);
      expect(fake.bodies).toHaveLength(2);
    }
  });

  it('quotes the request whole in the instructions once it is summarized, whether the model wrote the summary or could not', async () => {
    for (const summary of ['SUMMARY: the user wants the sign-in form fixed; notes.md was written.', '']) {
      const { summaries, main } = model(summary, [
        { calls: [{ name: 'append_file', input: { path: 'notes.md', content: `${'A note on the sign-in form. '.repeat(150)}\n` } }] },
        { calls: [{ name: 'fetch_log', input: {} }] },
        { text: 'done' },
      ]);
      const agent = agentFor(16_000);
      const events: AgentEvent[] = [];
      await agent.run(REQUEST, (e) => events.push(e));
      const compact = events.find((e) => e.type === 'compact') as Extract<AgentEvent, { type: 'compact' }>;
      expect(compact).toMatchObject({ how: summary ? 'summary' : 'trimmed' });
      expect(summaries()).toBe(1);
      expect(events.at(-1)).toMatchObject({ type: 'done', text: 'done' });
      // The request's turn was summarized: the model reads it, word for word, in its instructions.
      const view = agent.view();
      expect(view[0]).toMatchObject({ role: 'user', summary: true });
      expect(JSON.stringify(view)).not.toContain('Fix the sign-in form');
      const system = String((main().at(-1)!.messages as Array<{ content: unknown }>)[0].content);
      expect(system).toContain(`The user's request being worked on, word for word, for reference (the part of the conversation it was in was summarized to make room):\n"""\n${REQUEST}\n"""`);
      if (!summary) {
        // Made without the model: the files written, appended ones among them; the request not cut to its first 300 characters.
        const note = (view[0] as Extract<Turn, { role: 'user' }>).text;
        expect(note).toContain('Files changed: notes.md');
        expect(note).not.toContain('Fix the sign-in form');
      }
      // Each tool result still follows the call it answers.
      view.forEach((t, i) => {
        if (t.role === 'tool') expect(view[i - 1].role).toBe('assistant');
      });
    }
  });
});

describe('the characters-per-token calibration', () => {
  /** A server that counts a prompt's text at 4 characters a token (its pictures' data aside) and each picture at 256 tokens. */
  function server(replies: Array<Record<string, unknown>>): void {
    let n = 0;
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
      const raw = String(init.body);
      const pictures = (raw.match(/data:image\//g) ?? []).length;
      const promptTokens = Math.round(raw.replace(/data:image\/[a-z]+;base64,[A-Za-z0-9+/=]*/g, '').length / 4) + 256 * pictures;
      const delta = replies[n++] ?? { content: 'done' };
      return new Response(`${sse([{ choices: [{ index: 0, delta }] }, { choices: [{ index: 0, delta: {}, finish_reason: delta.tool_calls ? 'tool_calls' : 'stop' }] }, { choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: 5 } }])}data: [DONE]\n\n`, { status: 200, headers: { 'content-type': 'text/event-stream' } });
    });
  }
  async function calibrated(pictures: number): Promise<number> {
    server([{ tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{"path":"src/app.js"}' } }] }, { content: 'done' }]);
    const vfs = new Vfs();
    vfs.writeFile('/src/app.js', 'const greeting = "hello";\nconsole.log(greeting);\n'.repeat(40), { parents: true });
    const agent = new Agent({ vfs, gate: new NetGate(), provider: () => ({ ...OPENAI, contextTokens: 128_000 }), projectSummary: () => 'p' });
    await agent.run('Compare these pictures with what src/app.js draws.', () => {}, undefined, Array.from({ length: pictures }, () => PICTURE));
    return (agent as unknown as { charsPerToken: number }).charsPerToken;
  }

  it('is not thrown off by the pictures in view: each costs what the server makes of it, not characters', async () => {
    const none = await calibrated(0);
    // (calibrated on text: about the server's 4 characters a token)
    expect(none).toBeGreaterThan(3);
    for (const pictures of [4, 8]) {
      const ratio = await calibrated(pictures);
      // (pictures counted as characters, it came out at twice the text's: a text prompt then estimated at half its size)
      expect(ratio / none, `${pictures} pictures`).toBeLessThan(1.1);
    }
  });
});
