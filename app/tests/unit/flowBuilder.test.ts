import { describe, expect, it } from 'vitest';
import { NODE_TYPES, checkFlow, flowBuilderTools, layOut, nameTemplateInputs } from '../../src/desktop/flowBuilder';
import type { Desktop } from '../../src/desktop/bridge';

const signal = new AbortController().signal;

const greeting = {
  nodes: [
    { id: 'in1', type: 'input_text', data: { label: 'person' } },
    { id: 'tpl1', type: 'template', data: { template: 'Hello {{input}}!' } },
    { id: 'out1', type: 'output', data: { label: 'Output' } },
  ],
  edges: [
    { source: 'in1', sourceHandle: 'text', target: 'tpl1', targetHandle: 'input' },
    { source: 'tpl1', sourceHandle: 'result', target: 'out1', targetHandle: 'result' },
  ],
};

describe('the agent builds flows', () => {
  it("knows the editor's node types, from their own definitions", () => {
    expect(NODE_TYPES.length).toBeGreaterThan(40);
    const template = NODE_TYPES.find((n) => n.type === 'template')!;
    expect(template.outputs.map((o) => o.id)).toEqual(['result']);
    expect(NODE_TYPES.find((n) => n.type === 'ai_llm')!.inputs.map((i) => i.id)).toContain('prompt');
  });

  it("checks a flow against the node types: types, ids and handles", () => {
    expect(checkFlow(greeting.nodes, greeting.edges)).toEqual([]);
    const problems = checkFlow(
      [...greeting.nodes, { id: 'x', type: 'teleporter', data: {} }, { id: 'in1', type: 'input_text', data: {} }],
      [...greeting.edges, { source: 'tpl1', sourceHandle: 'text', target: 'out1', targetHandle: 'result' }, { source: 'nope', sourceHandle: 'a', target: 'out1', targetHandle: 'result' }],
    );
    expect(problems).toContain('node x: there is no node type "teleporter" (see flow_nodes)');
    expect(problems).toContain('two nodes are called in1');
    expect(problems.some((p) => p.startsWith('tpl1 (template) has no output "text"; its outputs: result'))).toBe(true);
    expect(problems).toContain('an edge comes from nope, which is not a node');
  });

  it('lays a flow out in columns from its inputs, keeping positions it was given', () => {
    const placed = layOut([...greeting.nodes, { id: 'p', type: 'output', data: {}, position: { x: 5, y: 6 } }], greeting.edges);
    const at = Object.fromEntries(placed.map((n) => [n.id, n.position]));
    expect(at.in1!.x).toBeLessThan(at.tpl1!.x);
    expect(at.tpl1!.x).toBeLessThan(at.out1!.x);
    expect(at.p).toEqual({ x: 5, y: 6 });
  });

  it("names a template's inputs after its handles, as many as it uses, so the editor knows its placeholders", () => {
    const [, tpl] = nameTemplateInputs(greeting.nodes, greeting.edges);
    expect(tpl.data).toMatchObject({ inputCount: 1, inputNames: ['input'] });
    const two = nameTemplateInputs([{ id: 't', type: 'template', data: { template: '{{input}} and {{input3}}' } }], []);
    expect(two[0].data.inputNames).toEqual(['input', 'input2', 'input3']);
    const named = { id: 't', type: 'template', data: { inputNames: ['who'] } };
    expect(nameTemplateInputs([named], [])[0]).toBe(named);
  });

  it('writes a checked flow to the desktop (a tool when asked), refuses a broken one, and runs one', async () => {
    const stored: Array<[string, Record<string, unknown>]> = [];
    const desktop = {
      putFlow: async (id: string, doc: Record<string, unknown>) => void stored.push([id, doc]),
      flows: async () => [{ id: 'greet-a-person', name: 'Greet a person' }],
      flow: async () => ({ name: 'Greet a person', ...greeting }),
      runFlow: async (_id: string, input: Record<string, unknown>) => ({ status: 'succeeded', output: { engine: 'zipp', output: `Hello ${String(input.person)}!` } }),
    } as unknown as Desktop;
    let looked = 0;
    const tools = Object.fromEntries(flowBuilderTools(() => desktop, async () => void looked++).map((t) => [t.spec.name, t]));
    expect(Object.keys(tools)).toEqual(['flow_nodes', 'flow_list', 'flow_read', 'flow_write', 'flow_run']);
    expect(await tools.flow_nodes.run({ types: ['template'] }, signal)).toContain('template — ');
    const out = await tools.flow_write.run({ name: 'Greet a person', ...greeting, tool: { name: 'greet', description: 'Greets someone' } }, signal);
    expect(out).toContain('Wrote flow greet-a-person ("Greet a person"): 3 nodes, 2 edges; its inputs: person');
    // Made a tool: the agent's tools are looked at again at once.
    expect(out).toContain('greet is one of your tools from your next step');
    expect(looked).toBe(1);
    const [id, doc] = stored[0];
    expect(id).toBe('greet-a-person');
    expect(doc.oaiyTool).toEqual({ name: 'greet', description: 'Greets someone' });
    expect((doc.nodes as Array<{ position?: unknown }>).every((n) => n.position)).toBe(true);
    await expect(tools.flow_write.run({ name: 'Broken', nodes: [{ id: 'a', type: 'teleporter' }], edges: [] }, signal)).rejects.toThrow('the flow was not written');
    expect(await tools.flow_run.run({ id: 'greet-a-person', input: { person: 'Sam' } }, signal)).toBe('Hello Sam!');
    expect(await tools.flow_list.run({}, signal)).toBe('greet-a-person: Greet a person');
  });
});
