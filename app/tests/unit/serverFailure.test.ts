// A request the provider's server failed of its own (an error inside its stream, a status of 500 or more) is asked again
// before an unattended run ends on it: after a pause, then with the conversation compacted, and only then does the run
// end. OAIY's engine, its GPU out of room for Nanbeige 4.3 33B's prompt, said only "native Qwen inference failed", and a
// run making a snake game ended there.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { LOCAL } from './fakeProvider';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const ENGINE: ProviderConfig = { ...LOCAL, serverKind: 'oaiy', name: 'OAIY', baseUrl: 'http://127.0.0.1:17999', modelId: 'nanbeige4.3-33b', contextTokens: 32_000 };

const sse = (events: unknown[]) => `${events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('')}data: [DONE]\n\n`;
const streamed = (events: unknown[]) => () => new Response(sse(events), { status: 200, headers: { 'content-type': 'text/event-stream' } });
const reply = (text: string) => streamed([{ choices: [{ index: 0, delta: { content: text } }] }, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }]);
const calls = (name: string, input: Record<string, unknown>) =>
  streamed([{ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: `call_${name}_${JSON.stringify(input).length}`, type: 'function', function: { name, arguments: JSON.stringify(input) } }] } }] }, { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }]);
/** OAIY's engine failing once its stream has begun, in its words (crates/oaiy-llm-server/src/api.rs). */
const engineFailed = streamed([{ error: { message: 'native Qwen inference failed', type: 'server_error', code: 'server_error' } }]);
const answer = (status: number, body: unknown) => () => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

/** A fetch that answers each request of the run with the next of `answers`, and a summary request with a summary. */
function serve(answers: Array<() => Response>) {
  const bodies: Array<Record<string, unknown>> = [];
  let summaries = 0;
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    if (JSON.stringify(body).includes('You compress the conversation')) {
      summaries++;
      return reply('SUMMARY: the user wants the notes read; the first ones were read.')();
    }
    bodies.push(body);
    const next = answers[bodies.length - 1];
    if (!next) throw new Error(`no answer for request ${bodies.length}`);
    return next();
  });
  return { bodies, summaries: () => summaries };
}

/** Run the agent on the test's clock: its pauses pass as the clock is moved on. */
async function runOnTheClock(agent: Agent, prompt: string, signal?: AbortSignal): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  let over = false;
  const run = agent.run(prompt, (e) => events.push(e), signal).finally(() => {
    over = true;
  });
  for (let i = 0; i < 200 && !over; i++) await vi.advanceTimersByTimeAsync(250);
  await run;
  return events;
}

const statuses = (events: AgentEvent[]) => events.filter((e): e is Extract<AgentEvent, { type: 'status' }> => e.type === 'status').map((e) => e.message);

describe("a request the server failed of its own", () => {
  it('is asked again after a short pause, and the run goes on: an error inside the stream, or a 500 before it', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    for (const [what, failure, said] of [
      ["OAIY's engine, inside its stream", engineFailed, 'native Qwen inference failed'],
      ['another server, inside its stream, with no code', streamed([{ error: { message: 'CUDA error: out of memory' } }]), 'CUDA error: out of memory'],
      ['a 500 before any stream', answer(500, { error: { message: 'engine crashed', type: 'server_error' } }), 'engine crashed'],
      ['a 502 from a proxy', answer(502, { error: { message: 'bad gateway' } }), 'bad gateway'],
    ] as const) {
      const { bodies } = serve([failure, reply('The snake game is ready.')]);
      const events: AgentEvent[] = [];
      let over = false;
      const run = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => 'p' })
        .run('create a snake game', (e) => events.push(e))
        .finally(() => {
          over = true;
        });
      // not at once: after a pause of a few seconds
      await vi.advanceTimersByTimeAsync(2000);
      expect(bodies, what).toHaveLength(1);
      for (let i = 0; i < 40 && !over; i++) await vi.advanceTimersByTimeAsync(250);
      await run;
      expect(bodies, what).toHaveLength(2);
      expect(statuses(events), what).toContain(`The provider failed (${said}); retrying in 3 s`);
      expect(events.some((e) => e.type === 'error'), what).toBe(false);
      expect(events.at(-1), what).toMatchObject({ type: 'done', text: 'The snake game is ready.' });
    }
  });

  it('failing so again, is asked once more with the conversation compacted, and a third failure ends the run in its words', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const vfs = () => {
      const files = new Vfs();
      for (let n = 1; n <= 4; n++) files.writeFile(`/notes${n}.txt`, `note ${n}: the snake moves one cell a tick\n`.repeat(160));
      return files;
    };
    const reads = [1, 2, 3, 4].map((n) => calls('read_file', { path: `notes${n}.txt` }));

    // The third try answers: the run goes on, from the compacted conversation.
    const { bodies, summaries } = serve([...reads, engineFailed, engineFailed, reply('Read them all.')]);
    const events = await runOnTheClock(new Agent({ vfs: vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => 'p' }), 'read the notes');
    expect(statuses(events)).toEqual(expect.arrayContaining([
      'The provider failed (native Qwen inference failed); retrying in 3 s',
      'The provider failed again (native Qwen inference failed); compacting the conversation and retrying once more',
    ]));
    const compacted = events.findIndex((e) => e.type === 'compact');
    expect(compacted).toBeGreaterThan(events.findIndex((e) => e.type === 'status' && /failed again/.test(e.message)));
    expect(summaries()).toBe(1);
    expect(bodies).toHaveLength(7);
    // the third try carried the summary, not the first notes word for word
    expect(JSON.stringify(bodies[6])).toContain('SUMMARY: the user wants the notes read');
    expect(JSON.stringify(bodies[6])).not.toContain('note 1: the snake');
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'Read them all.' });

    // The third fails too: the run ends on the server's words, to be tried again by the person, after three tries of it.
    const again = serve([...reads, engineFailed, engineFailed, engineFailed, reply('never asked')]);
    const ended = await runOnTheClock(new Agent({ vfs: vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => 'p' }), 'read the notes');
    expect(again.bodies).toHaveLength(7);
    expect(ended.at(-1)).toMatchObject({ type: 'error', message: expect.stringContaining('native Qwen inference failed'), retry: true });
  });

  it('is not asked again once Stop is pressed in the pause', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { bodies } = serve([engineFailed, reply('never asked')]);
    const stop = new AbortController();
    const events: AgentEvent[] = [];
    const run = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => 'p' }).run('create a snake game', (e) => events.push(e), stop.signal);
    for (let i = 0; i < 20 && !statuses(events).some((s) => /retrying in/.test(s)); i++) await vi.advanceTimersByTimeAsync(100);
    stop.abort();
    await vi.advanceTimersByTimeAsync(5000);
    await run;
    expect(statuses(events)).toContain('The provider failed (native Qwen inference failed); retrying in 3 s');
    expect(bodies).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'notice', message: 'Stopped.' });
  });

  it("is told apart from what is not the server's own failure: a refused request, or a reply stopped for repeating itself", async () => {
    // A client's error (4xx): said at once, as before.
    const refused = serve([answer(400, { error: { message: 'tools: unknown field "strict"', type: 'invalid_request_error' } }), reply('never asked')]);
    const events: AgentEvent[] = [];
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => 'p' }).run('hi', (e) => events.push(e));
    expect(refused.bodies).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ type: 'error', message: expect.stringContaining('unknown field') });
    expect(statuses(events).some((s) => /retrying in|compacting the conversation and retrying/.test(s))).toBe(false);

    // A reply the engine stopped for repeating itself, said inside the stream: asked again once, told what it repeated (its
    // own branch), and then not taken for the server failing.
    const said = 'generation_repetition: the reply went on repeating the same 4 tokens ("dirs.indexOf(", 32 times: 128 tokens) in a tool call and was stopped';
    const repeating = streamed([{ error: { message: said, type: 'server_error', code: 'generation_repetition' } }]);
    const looped = serve([repeating, repeating, reply('never asked')]);
    const told: AgentEvent[] = [];
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ENGINE, projectSummary: () => 'p' }).run('hi', (e) => told.push(e));
    expect(looped.bodies).toHaveLength(2);
    expect(told.at(-1)).toMatchObject({ type: 'error', message: expect.stringContaining('generation_repetition') });
    expect(statuses(told).some((s) => /The provider failed/.test(s))).toBe(false);
  });
});
