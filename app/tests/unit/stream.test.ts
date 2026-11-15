// The Agent's streamed replies (agent/providers/stream.ts, agent/protocol.ts): read as the same reply sent whole is, and
// a reply whose stream cannot be read asked again without streaming.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Agent, type AgentEvent } from '../../src/agent/agent';
import { sendTurn } from '../../src/agent/protocol';
import type { ProviderConfig } from '../../src/agent/providers/types';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { ANTHROPIC, LOCAL, OPENAI } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

const data = (v: unknown) => `data: ${JSON.stringify(v)}\n\n`;
const json = (body: unknown, type = 'application/json') => () => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': type } });
/** A reply's bytes as a stream, cut every `every` bytes (a character may be split between two chunks). */
const streamed = (text: string, every = 5) => () => {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream<Uint8Array>({
    start(c) {
      for (let i = 0; i < bytes.length; i += every) c.enqueue(bytes.subarray(i, i + every));
      c.close();
    },
  }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
};
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
const said = (events: AgentEvent[], type: 'text' | 'thinking') => events.filter((e) => e.type === type).map((e) => (e as { delta: string }).delta).join('');

describe('a reply whose stream cannot be read', () => {
  const completion = { choices: [{ index: 0, message: { role: 'assistant', content: 'hello there' }, finish_reason: 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 2 } };

  it('is asked for once more without streaming and read whole: the run ends with it, and the chat shows it', async () => {
    // A gateway that answers a streamed request with the whole reply, labelled as text: "the streamed reply had no
    // events", the run said, of a server that had answered; and Try again asked the same way.
    const bodies = answering([json(completion, 'text/plain; charset=utf-8'), json(completion, 'text/plain; charset=utf-8')]);
    const events: AgentEvent[] = [];
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => ({ ...OPENAI, baseUrl: 'https://gateway.example/v1' }), projectSummary: () => 'p' }).run('hi', (e) => events.push(e));
    // (stream_options goes with a stream only)
    expect(bodies.map((b) => [b.stream, 'stream_options' in b])).toEqual([[true, true], [false, false]]);
    expect(events.at(-1)).toMatchObject({ type: 'done', text: 'hello there' });
    expect(said(events, 'text')).toBe('hello there');
  });

  it('is asked for so on either wire, a stream with an event that is not JSON too; one that still cannot be read is the error', async () => {
    const notJson = streamed('data: keep-alive\n\n');
    answering([notJson, json({ choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] })]);
    expect((await sendTurn(OPENAI, 's', [{ role: 'user', text: 'hi' }], [])).text).toBe('ok');
    const anthropic = answering([json({ type: 'message', role: 'assistant', content: [{ type: 'text', text: 'ok too' }], stop_reason: 'end_turn', usage: { input_tokens: 5, output_tokens: 2 } }, 'text/plain'), json({ type: 'message', role: 'assistant', content: [{ type: 'text', text: 'ok too' }], stop_reason: 'end_turn', usage: { input_tokens: 5, output_tokens: 2 } })]);
    expect((await sendTurn(ANTHROPIC, 's', [{ role: 'user', text: 'hi' }], [])).text).toBe('ok too');
    expect(anthropic.map((b) => b.stream)).toEqual([true, false]);
    answering([notJson, notJson]);
    await expect(sendTurn(OPENAI, 's', [{ role: 'user', text: 'hi' }], [])).rejects.toThrow('AI response was not valid JSON');
  });
});

describe("a reply's thought (OpenAI-compatible: OAIY's engine, Ollama, LM Studio, llama.cpp)", () => {
  const thoughtStream = `${data({ choices: [{ index: 0, delta: { reasoning_content: 'First I check ' } }] })}${data({ choices: [{ index: 0, delta: { reasoning_content: 'the file.' } }] })}${data({ choices: [{ index: 0, delta: { content: 'Done.' } }] })}${data({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}data: [DONE]\n\n`;
  const whole = json({ choices: [{ index: 0, message: { role: 'assistant', content: 'Done.', reasoning_content: 'First I check the file.' }, finish_reason: 'stop' }] });
  const engine: ProviderConfig = { ...LOCAL, serverKind: 'oaiy' };

  it('streamed, is kept with the reply as a whole one keeps it, and saved with the turn the chat draws again', async () => {
    answering([whole, streamed(thoughtStream)]);
    expect((await sendTurn(engine, 's', [{ role: 'user', text: 'hi' }], [])).thinking).toBe('First I check the file.');
    expect((await sendTurn(engine, 's', [{ role: 'user', text: 'hi' }], [])).thinking).toBe('First I check the file.');
    answering([streamed(thoughtStream)]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => engine, projectSummary: () => 'p' });
    const events: AgentEvent[] = [];
    await agent.run('hi', (e) => events.push(e));
    // (shown once, as it streamed)
    expect(said(events, 'thinking')).toBe('First I check the file.');
    expect(agent.savedTurns().find((t) => t.role === 'assistant')).toMatchObject({ text: 'Done.', thinking: 'First I check the file.' });
  });

  it('sent whole, is shown as a streamed one is, once, with the reply', async () => {
    answering([whole]);
    const events: AgentEvent[] = [];
    await new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => engine, projectSummary: () => 'p' }).run('hi', (e) => events.push(e));
    expect([said(events, 'thinking'), said(events, 'text')]).toEqual(['First I check the file.', 'Done.']);
  });
});

describe('tool calls across stream chunks (OpenAI-compatible)', () => {
  const call = (tc: Record<string, unknown>) => data({ choices: [{ index: 0, delta: { tool_calls: [tc] } }] });
  const end = `${data({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] })}data: [DONE]\n\n`;
  const send = async (text: string, every = 5) => {
    answering([streamed(text, every)]);
    return sendTurn({ ...OPENAI, baseUrl: 'https://x.example/v1' }, 's', [{ role: 'user', text: 'go' }], [{ name: 'write_file', description: '', parameters: {} }]);
  };

  it('are told apart by their indexes, interleaved, their arguments split mid-character', async () => {
    const reply = await send(
      call({ index: 0, id: 'a', type: 'function', function: { name: 'write_file', arguments: '' } }) +
        call({ index: 1, id: 'b', type: 'function', function: { name: 'write_file', arguments: '{"path":"b.txt",' } }) +
        call({ index: 0, function: { arguments: '{"path":"a.txt","content":"caf' } }) +
        call({ index: 1, function: { arguments: '"content":"😀 ok"}' } }) +
        call({ index: 0, function: { arguments: 'é 😀"}' } }) +
        end,
      3,
    );
    expect(reply.calls.map((c) => [c.id, c.input])).toEqual([['a', { path: 'a.txt', content: 'café 😀' }], ['b', { path: 'b.txt', content: '😀 ok' }]]);
  });

  it('with no index, by their ids', async () => {
    const reply = await send(call({ id: 'a', function: { name: 'write_file', arguments: '{"path":' } }) + call({ function: { arguments: '"a.txt"}' } }) + call({ id: 'b', function: { name: 'write_file', arguments: '{"path":"b.txt"}' } }) + end);
    expect(reply.calls.map((c) => c.input)).toEqual([{ path: 'a.txt' }, { path: 'b.txt' }]);
  });

  it('sent whole, each with its own id, at one index, are as many calls (the later fragments of that index the last one\'s)', async () => {
    // (they were one call, the second's id and the two argument objects run together: neither ran)
    const reply = await send(
      call({ index: 0, id: 'a', type: 'function', function: { name: 'write_file', arguments: '{"path":"a.txt"}' } }) +
        call({ index: 0, id: 'b', type: 'function', function: { name: 'write_file', arguments: '{"path":"b.txt"}' } }) +
        call({ index: 0, id: 'c', type: 'function', function: { name: 'write_file', arguments: '{"path":' } }) +
        call({ index: 0, function: { arguments: '"c.txt"}' } }) +
        end,
    );
    expect(reply.calls.map((c) => [c.id, c.input, c.parseError])).toEqual([['a', { path: 'a.txt' }, undefined], ['b', { path: 'b.txt' }, undefined], ['c', { path: 'c.txt' }, undefined]]);
  });
});
