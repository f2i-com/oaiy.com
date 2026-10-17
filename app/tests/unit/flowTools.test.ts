import { describe, expect, it } from 'vitest';
import { Agent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { flowSessionTools, flowToolSpec, keepFlowFiles, listFlowTools, readFlowTool, runOutcome } from '../../src/desktop/flowTools';
import { Desktop } from '../../src/desktop/bridge';
import { OPENAI, fakeProvider } from './fakeProvider';

const greeting = {
  name: 'Greeting card',
  oaiyTool: { name: 'Make Greeting', description: 'Makes a welcome line for a person.' },
  nodes: [
    { id: 'in1', type: 'input_text', data: { label: 'person' } },
    { id: 'in2', type: 'input_file', data: { label: 'letter head' } },
    { id: 'tpl1', type: 'template', data: { template: 'Hello {{input}}' } },
  ],
  edges: [],
};

describe("flows as the agent's tools", () => {
  it('reads a flow made a tool: its name, what it is for, and its inputs by label', () => {
    const tool = readFlowTool('tool-greeting', greeting)!;
    expect(tool).toMatchObject({ id: 'tool-greeting', name: 'make_greeting', flowName: 'Greeting card', inputs: [{ label: 'person', type: 'input_text' }, { label: 'letter head', type: 'input_file' }] });
    expect(readFlowTool('smoke', { name: 'smoke', nodes: [] })).toBeNull();
    const spec = flowToolSpec(tool, new Set(['read_file']));
    expect(spec.name).toBe('make_greeting');
    expect(spec.parameters).toMatchObject({ required: ['person', 'letter_head'] });
    expect(JSON.stringify(spec.parameters)).toContain('The path of a file on this computer');
    // A name the agent already has gets flow_ in front.
    expect(flowToolSpec({ ...tool, name: 'read_file' }, new Set(['read_file'])).name).toBe('flow_read_file');
  });

  it('hands back what the flow returned, or why it failed', () => {
    expect(runOutcome({ status: 'succeeded', output: { engine: 'zipp', output: 'Hello Lance', results: {} } })).toBe('Hello Lance');
    expect(runOutcome({ status: 'succeeded', output: { a: 1 } })).toContain('"a": 1');
    expect(runOutcome({ status: 'running', runId: 'r1' })).toContain('still running');
    expect(() => runOutcome({ status: 'failed', error: { code: 'node_failed', message: 'the template broke' } })).toThrow('the flow failed: the template broke');
  });

  it('the agent uses a flow tool: the flow runs on the desktop with its inputs by label', async () => {
    const runs: Array<{ flowId: string; input: Record<string, unknown> }> = [];
    const desktop = {
      flows: async () => [{ id: 'tool-greeting', name: 'Greeting card' }, { id: 'smoke', name: 'smoke' }],
      flow: async (id: string) => (id === 'tool-greeting' ? greeting : { name: 'smoke', nodes: [] }),
      runFlow: async (flowId: string, input: Record<string, unknown>) => {
        runs.push({ flowId, input });
        return { status: 'succeeded', output: { output: `Hello ${String(input.person)}, welcome!` } };
      },
    } as unknown as Desktop;
    const tools = flowSessionTools(await listFlowTools(desktop), () => desktop, new Set());
    expect(tools.map((t) => t.spec.name)).toEqual(['make_greeting']);
    fakeProvider('openai', [
      (body) => {
        expect(JSON.stringify(body.tools)).toContain('make_greeting');
        return { calls: [{ name: 'make_greeting', input: { person: 'Sam', letter_head: 'C:/head.png' } }] };
      },
      (body) => {
        expect(JSON.stringify(body.messages)).toContain('Hello Sam, welcome!');
        return { text: 'It says: Hello Sam, welcome!' };
      },
    ]);
    const agent = new Agent({ vfs: new Vfs(), gate: new NetGate(), provider: () => OPENAI, projectSummary: () => '', sessionTools: () => tools });
    const done: string[] = [];
    await agent.run('Greet Sam with my flow', (e) => {
      if (e.type === 'done') done.push(e.text);
    });
    expect(runs).toEqual([{ flowId: 'tool-greeting', input: { person: 'Sam', 'letter head': 'C:/head.png' } }]);
    expect(done).toEqual(['It says: Hello Sam, welcome!']);
  });

  it("keeps a flow's pictures in the project and hands the model their paths, not their bytes", async () => {
    const vfs = new Vfs();
    const file = 'http://127.0.0.1:17972/api/ai/engine/gateway/files/images/2026-10-17/fox.png';
    const asked: string[] = [];
    const fetchFile = async (url: string) => {
      asked.push(url);
      return url === file ? new Uint8Array([137, 80, 78, 71]) : null;
    };
    const out = {
      image: file,
      original: 'data:image/png;base64,iVBORw0K',
      pictures: { made: { file, seed: 7 } },
      note: 'https://example.com/about.html',
      elsewhere: 'https://example.com/cat.png',
    };
    const kept = await keepFlowFiles(out, 'make_and_check_image', fetchFile, vfs);
    expect(kept.saved).toEqual(['flows/make_and_check_image/fox.png', 'flows/make_and_check_image/make_and_check_image-2.png']);
    expect(kept.value).toEqual({
      image: 'flows/make_and_check_image/fox.png',
      original: 'flows/make_and_check_image/make_and_check_image-2.png',
      pictures: { made: { file: 'flows/make_and_check_image/fox.png', seed: 7 } },
      note: 'https://example.com/about.html',
      elsewhere: 'https://example.com/cat.png',
    });
    expect([...vfs.readBytes('/flows/make_and_check_image/fox.png')]).toEqual([137, 80, 78, 71]);
    // the same picture twice is fetched once; a page is not fetched at all; one that is not the engine's is left as it is
    expect(asked).toEqual([file, 'https://example.com/cat.png']);
    // a second run's picture of the same name does not overwrite the first
    expect((await keepFlowFiles({ image: file }, 'make_and_check_image', fetchFile, vfs)).saved).toEqual(['flows/make_and_check_image/fox-2.png']);
  });

  it("sends the desktop's token only for the engine's files on the desktop", async () => {
    const d = new Desktop('http://127.0.0.1:17972', 'secret');
    expect(await d.engineFile('https://example.com/api/ai/engine/gateway/files/a.png')).toBeNull();
    expect(await d.engineFile('http://127.0.0.1:17972/api/bridge/flows')).toBeNull();
    expect(await d.engineFile('http://127.0.0.1:17972/api/ai/engine/gateway/files/../../../bridge/flows')).toBeNull();
    expect(await d.engineFile('not a url')).toBeNull();
  });
});

describe('a flow run from the Agent', () => {
  const reply = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

  it('is started, then followed until it ends (no request kept waiting for minutes)', async () => {
    const seen: string[] = [];
    const states = ['running', 'running', 'succeeded'];
    const real = globalThis.fetch;
    globalThis.fetch = (async (url: string, init: RequestInit = {}) => {
      seen.push(`${init.method ?? 'GET'} ${url}`);
      if (init.method === 'POST') {
        expect(JSON.parse(String(init.body)).mode).toBe('async');
        return reply(201, { runId: 'run_1', status: 'queued' });
      }
      const status = states.shift() ?? 'succeeded';
      return reply(200, { runId: 'run_1', status, ...(status === 'succeeded' ? { output: { output: 'made' } } : {}) });
    }) as typeof fetch;
    try {
      const run = await new Desktop('http://127.0.0.1:17972', 'tok').runFlow('f', { Prompt: 'x' }, 60_000, undefined, 5);
      expect(run.status).toBe('succeeded');
      expect(runOutcome(run)).toBe('made');
      expect(seen).toEqual(['POST http://127.0.0.1:17972/api/bridge/runs', ...Array(3).fill('GET http://127.0.0.1:17972/api/bridge/runs/run_1')]);
    } finally {
      globalThis.fetch = real;
    }
  });

  it('is cancelled on the desktop when the Agent is stopped', async () => {
    const seen: string[] = [];
    const real = globalThis.fetch;
    globalThis.fetch = (async (url: string, init: RequestInit = {}) => {
      seen.push(`${init.method ?? 'GET'} ${url}`);
      return init.method === 'POST' && !url.endsWith('/cancel') ? reply(201, { runId: 'run_2', status: 'queued' }) : reply(200, { runId: 'run_2', status: 'running' });
    }) as typeof fetch;
    try {
      const stop = new AbortController();
      const running = new Desktop('http://127.0.0.1:17972', 'tok').runFlow('f', {}, 60_000, stop.signal, 5);
      await new Promise((r) => setTimeout(r, 30));
      stop.abort();
      await expect(running).rejects.toBeTruthy();
      await new Promise((r) => setTimeout(r, 5));
      expect(seen).toContain('POST http://127.0.0.1:17972/api/bridge/runs/run_2/cancel');
    } finally {
      globalThis.fetch = real;
    }
  });
});
