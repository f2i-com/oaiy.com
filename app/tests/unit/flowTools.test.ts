import { describe, expect, it } from 'vitest';
import { Agent } from '../../src/agent/agent';
import { NetGate } from '../../src/gate/netgate';
import { Vfs } from '../../src/vfs/vfs';
import { flowSessionTools, flowToolSpec, listFlowTools, readFlowTool, runOutcome } from '../../src/desktop/flowTools';
import type { Desktop } from '../../src/desktop/bridge';
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
});
