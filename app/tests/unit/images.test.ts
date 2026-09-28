import { afterEach, describe, expect, it, vi } from 'vitest';
import { sendTurn, type Turn } from '../../src/agent/protocol';
import { ANTHROPIC, OPENAI, fakeProvider } from './fakeProvider';

afterEach(() => vi.unstubAllGlobals());

const pixel = { mediaType: 'image/png' as const, data: 'iVBORw0KGgo=' };
const turns: Turn[] = [
  { role: 'user', text: 'What is in this picture?', images: [pixel] },
  { role: 'assistant', text: '', calls: [{ id: 'c1', name: 'view_image', input: { path: 'uploads/a.png', x: 10 } }] },
  { role: 'tool', results: [{ id: 'c1', name: 'view_image', content: 'uploads/a.png: 100×100 px', isError: false, images: [pixel] }] },
];

describe('images reach the model in each wire shape', () => {
  it('Anthropic: image blocks in the user turn and inside the tool result', async () => {
    const fake = fakeProvider('anthropic', [{ text: 'a cat' }]);
    await sendTurn(ANTHROPIC, 'system', turns, []);
    const messages = fake.bodies[0].messages as Array<{ role: string; content: Array<Record<string, unknown>> }>;
    expect(messages[0].content[1]).toEqual({ type: 'image', source: { type: 'base64', media_type: 'image/png', data: pixel.data } });
    const toolResult = messages[2].content[0] as { type: string; content: Array<Record<string, unknown>> };
    expect(toolResult.type).toBe('tool_result');
    expect(toolResult.content[0]).toEqual({ type: 'text', text: 'uploads/a.png: 100×100 px' });
    expect(toolResult.content[1]).toMatchObject({ type: 'image' });
  });

  it('OpenAI-compatible: image_url parts, and a tool\'s images in a user message after the tool message', async () => {
    const fake = fakeProvider('openai', [{ text: 'a cat' }]);
    await sendTurn(OPENAI, 'system', turns, []);
    const messages = fake.bodies[0].messages as Array<{ role: string; content: unknown }>;
    expect(messages[1]).toEqual({ role: 'user', content: [{ type: 'text', text: 'What is in this picture?' }, { type: 'image_url', image_url: { url: `data:image/png;base64,${pixel.data}` } }] });
    expect(messages[3]).toMatchObject({ role: 'tool', tool_call_id: 'c1', content: 'uploads/a.png: 100×100 px' });
    expect(messages[4]).toMatchObject({ role: 'user', content: [{ type: 'text' }, { type: 'image_url' }] });
    expect(messages).toHaveLength(5);
  });

  it('a turn without images is sent as plain text', async () => {
    const fake = fakeProvider('openai', [{ text: 'hi' }]);
    await sendTurn(OPENAI, 'system', [{ role: 'user', text: 'hello' }], []);
    expect((fake.bodies[0].messages as Array<{ content: unknown }>)[1].content).toBe('hello');
  });
});
