// A scripted provider behind a fake `fetch`, answering in either wire shape
// as a server-sent-event stream. Modelled on softn Studio's agentHarness.
import { vi } from 'vitest';
import type { ProviderConfig } from '../../src/agent/providers/types';

export interface Step {
  text?: string;
  calls?: Array<{ name: string; input: Record<string, unknown> }>;
}

export const ANTHROPIC: ProviderConfig = { id: 'a', type: 'anthropic', name: 'Anthropic', apiKey: 'k', modelId: 'model-under-test' };
export const OPENAI: ProviderConfig = { id: 'o', type: 'openai', name: 'OpenAI', apiKey: 'k', modelId: 'model-under-test' };
export const LOCAL: ProviderConfig = { id: 'l', type: 'local', serverKind: 'ollama', name: 'Ollama', apiKey: '', modelId: 'qwen3', baseUrl: 'http://localhost:11434' };

function sse(events: Array<Record<string, unknown> | '[DONE]'>, named = false): string {
  return events
    .map((e) => (e === '[DONE]' ? 'data: [DONE]\n\n' : `${named ? `event: ${String(e.type)}\n` : ''}data: ${JSON.stringify(e)}\n\n`))
    .join('');
}

function anthropicStream(step: Step, n: number): string {
  const events: Array<Record<string, unknown>> = [{ type: 'message_start', message: { usage: { input_tokens: 10, output_tokens: 0 } } }];
  let index = 0;
  if (step.text) {
    events.push({ type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of step.text.match(/.{1,7}/gs) ?? []) events.push({ type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    events.push({ type: 'content_block_stop', index });
    index++;
  }
  for (const [i, call] of (step.calls ?? []).entries()) {
    events.push({ type: 'content_block_start', index, content_block: { type: 'tool_use', id: `toolu_${n}_${i}`, name: call.name, input: {} } });
    const json = JSON.stringify(call.input);
    for (const piece of json.match(/.{1,9}/gs) ?? []) events.push({ type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: piece } });
    events.push({ type: 'content_block_stop', index });
    index++;
  }
  events.push({ type: 'message_delta', delta: { stop_reason: step.calls?.length ? 'tool_use' : 'end_turn' }, usage: { output_tokens: 5 } });
  events.push({ type: 'message_stop' });
  return sse(events, true);
}

function openAIStream(step: Step, n: number): string {
  const events: Array<Record<string, unknown> | '[DONE]'> = [];
  for (const piece of step.text?.match(/.{1,7}/gs) ?? []) events.push({ choices: [{ index: 0, delta: { content: piece } }] });
  for (const [i, call] of (step.calls ?? []).entries()) {
    events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: `call_${n}_${i}`, type: 'function', function: { name: call.name, arguments: '' } }] } }] });
    for (const piece of JSON.stringify(call.input).match(/.{1,9}/gs) ?? []) {
      events.push({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, function: { arguments: piece } }] } }] });
    }
  }
  events.push({ choices: [{ index: 0, delta: {}, finish_reason: step.calls?.length ? 'tool_calls' : 'stop' }] });
  events.push({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5 } });
  events.push('[DONE]');
  return sse(events);
}

export interface FakeProvider {
  bodies: Array<Record<string, unknown>>;
  urls: string[];
  headers: Array<Record<string, string>>;
}

/** Install a fake fetch that answers each request with the next scripted step. */
export function fakeProvider(wire: 'anthropic' | 'openai', script: Array<Step | ((body: Record<string, unknown>) => Step)>): FakeProvider {
  const record: FakeProvider = { bodies: [], urls: [], headers: [] };
  let n = 0;
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    record.bodies.push(body);
    record.urls.push(String(url));
    record.headers.push(Object.fromEntries(new Headers(init.headers).entries()));
    const next = script[n];
    if (!next) throw new Error(`the script has no step ${n + 1}`);
    const step = typeof next === 'function' ? next(body) : next;
    const text = wire === 'anthropic' ? anthropicStream(step, n) : openAIStream(step, n);
    n++;
    const bytes = new TextEncoder().encode(text);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < bytes.length; i += 13) controller.enqueue(bytes.subarray(i, i + 13));
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  });
  return record;
}
