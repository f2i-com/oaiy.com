/**
 * The agent builds and runs flows: the same flows the flow editor (Flows, in
 * OAIY's window) opens, stored on OAIY Desktop. It reads the node types
 * (`flow_nodes`, from the editor's own node definitions), the stored flows
 * (`flow_list`, `flow_read`), writes one (`flow_write`, checked against the
 * node types and laid out so the editor shows it tidily), and runs one
 * (`flow_run`, its inputs by their labels). A flow written with `tool` becomes
 * one of the agent's tools, as one made a tool in the editor does.
 */
import type { SessionTool } from '../agent/agent';
import type { Desktop } from './bridge';
import catalog from '../agent/flowNodes.json';

export interface NodeType {
  type: string;
  name: string;
  module: string;
  description: string;
  inputs: Array<{ id: string; type: string }>;
  outputs: Array<{ id: string; type: string }>;
  properties: Array<{ id: string; type: string; default?: unknown; options?: unknown[]; advanced?: boolean }>;
}

export const NODE_TYPES: NodeType[] = catalog as NodeType[];
const BY_TYPE = new Map(NODE_TYPES.map((n) => [n.type, n]));
/** Node types whose handles come from their settings, not their definition. */
const DYNAMIC = new Set(['macro']);

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

interface FlowNode {
  id: string;
  type: string;
  position?: { x: number; y: number };
  data: Record<string, unknown>;
}
interface FlowEdge {
  id?: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

/** What is wrong with a flow's nodes and edges, against the node types (empty: nothing). */
export function checkFlow(nodes: FlowNode[], edges: FlowEdge[]): string[] {
  const problems: string[] = [];
  const byId = new Map<string, FlowNode>();
  for (const n of nodes) {
    if (!n.id || typeof n.id !== 'string') problems.push('a node has no id');
    else if (byId.has(n.id)) problems.push(`two nodes are called ${n.id}`);
    else byId.set(n.id, n);
    if (!BY_TYPE.has(n.type)) problems.push(`node ${n.id}: there is no node type "${n.type}" (see flow_nodes)`);
  }
  for (const e of edges) {
    const from = byId.get(e.source);
    const to = byId.get(e.target);
    if (!from) problems.push(`an edge comes from ${e.source}, which is not a node`);
    if (!to) problems.push(`an edge goes to ${e.target}, which is not a node`);
    const fromType = from && BY_TYPE.get(from.type);
    const toType = to && BY_TYPE.get(to.type);
    if (fromType && !DYNAMIC.has(fromType.type) && !fromType.outputs.some((o) => o.id === e.sourceHandle)) {
      problems.push(`${e.source} (${fromType.type}) has no output "${e.sourceHandle}"; its outputs: ${fromType.outputs.map((o) => o.id).join(', ') || 'none'}`);
    }
    if (toType && !DYNAMIC.has(toType.type) && !toType.inputs.some((i) => i.id === e.targetHandle)) {
      problems.push(`${e.target} (${toType.type}) has no input "${e.targetHandle}"; its inputs: ${toType.inputs.map((i) => i.id).join(', ') || 'none'}`);
    }
  }
  return problems;
}

/** A template node's handles, in order: its placeholders are named after them ({{input}}, {{input2}}…). */
const TEMPLATE_INPUTS = ['input', 'input2', 'input3', 'input4', 'input5'];

/**
 * What the editor needs to show a template node as it runs: its inputs named
 * after its handles, as many as its edges and placeholders use (the editor's
 * own compiler does the same), so none of its placeholders shows as unknown.
 */
export function nameTemplateInputs(nodes: FlowNode[], edges: FlowEdge[]): FlowNode[] {
  return nodes.map((n) => {
    if (n.type !== 'template' || Array.isArray(n.data.inputNames)) return n;
    const used = [
      ...edges.filter((e) => e.target === n.id).map((e) => e.targetHandle),
      ...[...String(n.data.template ?? '').matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]),
    ];
    const count = Math.max(1, ...used.map((id) => TEMPLATE_INPUTS.indexOf(id) + 1));
    return { ...n, data: { ...n.data, inputCount: count, inputNames: TEMPLATE_INPUTS.slice(0, count) } };
  });
}

/** Positions for the editor: a column per step from the inputs, nodes stacked in each. */
export function layOut(nodes: FlowNode[], edges: FlowEdge[]): FlowNode[] {
  const depth = new Map<string, number>(nodes.map((n) => [n.id, 0]));
  for (let pass = 0; pass < nodes.length; pass++) {
    let moved = false;
    for (const e of edges) {
      const d = (depth.get(e.source) ?? 0) + 1;
      if (depth.has(e.target) && d > (depth.get(e.target) ?? 0) && d < nodes.length + 1) {
        depth.set(e.target, d);
        moved = true;
      }
    }
    if (!moved) break;
  }
  const row = new Map<number, number>();
  return nodes.map((n) => {
    if (n.position && Number.isFinite(n.position.x) && Number.isFinite(n.position.y)) return n;
    const col = depth.get(n.id) ?? 0;
    const r = row.get(col) ?? 0;
    row.set(col, r + 1);
    return { ...n, position: { x: 80 + col * 340, y: 80 + r * 220 } };
  });
}

/** A node type as the model reads it. */
function describe(n: NodeType, full: boolean): string {
  const handles = `in: ${n.inputs.map((i) => `${i.id}:${i.type}`).join(', ') || '-'}; out: ${n.outputs.map((o) => `${o.id}:${o.type}`).join(', ') || '-'}`;
  if (!full) return `${n.type} — ${n.name}: ${n.description} (${handles})`;
  const props = n.properties
    .filter((p) => !p.advanced)
    .map((p) => `${p.id}:${p.type}${p.default !== undefined ? `=${JSON.stringify(p.default)}` : ''}${p.options ? ` [${p.options.join('|')}]` : ''}`)
    .join('; ');
  return `${n.type} — ${n.name} (${n.module}): ${n.description}\n  ${handles}\n  data: ${props || '-'}`;
}

function connected(desktop: () => Desktop | null): Desktop {
  const d = desktop();
  if (!d) throw new Error('OAIY Desktop is not connected, so flows cannot be reached');
  return d;
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'flow';

export function flowBuilderTools(desktop: () => Desktop | null): SessionTool[] {
  return [
    {
      spec: {
        name: 'flow_nodes',
        description: 'The flow editor\'s node types: all of them in brief, or those named in full (their inputs, outputs and data settings). Read before writing a flow.',
        parameters: { type: 'object', properties: { types: { type: 'array', items: { type: 'string' }, description: 'Node types to show in full (omit for the list)' } } },
      },
      run: async (input) => {
        const wanted = Array.isArray(input.types) ? input.types.map(String) : [];
        if (!wanted.length) return `${NODE_TYPES.length} node types (flow_nodes with types for their settings):\n${NODE_TYPES.map((n) => describe(n, false)).join('\n')}`;
        return wanted.map((t) => (BY_TYPE.has(t) ? describe(BY_TYPE.get(t)!, true) : `${t}: no such node type`)).join('\n');
      },
    },
    {
      spec: { name: 'flow_list', description: 'The flows stored on OAIY Desktop (the ones the flow editor shows): id and name.', parameters: { type: 'object', properties: {} } },
      run: async (_input, signal) => {
        const flows = await connected(desktop).flows(signal);
        return flows.length ? flows.map((f) => `${f.id}: ${f.name}`).join('\n') : 'No flows are stored yet.';
      },
    },
    {
      spec: { name: 'flow_read', description: 'A stored flow, as JSON: its nodes (id, type, data) and edges (source, sourceHandle, target, targetHandle).', parameters: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } },
      run: async (input, signal) => {
        const doc = await connected(desktop).flow(String(input.id), signal);
        if (!isRecord(doc)) return 'No such flow.';
        const nodes = Array.isArray(doc.nodes) ? doc.nodes.filter(isRecord).map((n) => ({ id: n.id, type: n.type, data: n.data })) : [];
        return JSON.stringify({ name: doc.name, ...(doc.oaiyTool ? { tool: doc.oaiyTool } : {}), nodes, edges: doc.edges }, null, 1);
      },
    },
    {
      spec: {
        name: 'flow_write',
        description:
          'Write a flow to OAIY Desktop (made, or replaced when the id exists): it then opens in the flow editor, and flow_run runs it. Nodes are {id, type, data} (data holds the node\'s settings by their ids; an input node\'s data.label names its input); edges are {source, sourceHandle, target, targetHandle}, the handles being the node types\' output and input ids. It is checked against the node types first. Give tool {name, description} to make it one of your tools.',
        parameters: {
          type: 'object',
          required: ['name', 'nodes', 'edges'],
          properties: {
            id: { type: 'string', description: 'Its id (default: from the name)' },
            name: { type: 'string' },
            nodes: { type: 'array', items: { type: 'object' } },
            edges: { type: 'array', items: { type: 'object' } },
            tool: { type: 'object', description: '{name, description}: make it a tool of yours', properties: { name: { type: 'string' }, description: { type: 'string' } } },
          },
        },
      },
      run: async (input, signal) => {
        const nodes = (Array.isArray(input.nodes) ? input.nodes : []).filter(isRecord).map((n) => ({ id: String(n.id ?? ''), type: String(n.type ?? ''), data: isRecord(n.data) ? n.data : {}, ...(isRecord(n.position) ? { position: n.position as { x: number; y: number } } : {}) }));
        const edges = (Array.isArray(input.edges) ? input.edges : []).filter(isRecord).map((e, i) => ({ id: String(e.id ?? `e${i + 1}`), source: String(e.source ?? ''), sourceHandle: String(e.sourceHandle ?? ''), target: String(e.target ?? ''), targetHandle: String(e.targetHandle ?? '') }));
        if (!nodes.length) throw new Error('a flow needs nodes');
        const problems = checkFlow(nodes, edges);
        if (problems.length) throw new Error(`the flow was not written:\n- ${problems.join('\n- ')}`);
        const name = String(input.name);
        const id = typeof input.id === 'string' && input.id.trim() ? input.id.trim() : slug(name);
        const tool = isRecord(input.tool) && typeof input.tool.name === 'string' ? { name: input.tool.name, description: String(input.tool.description ?? '') } : null;
        await connected(desktop).putFlow(id, { name, ...(tool ? { oaiyTool: tool } : {}), nodes: layOut(nameTemplateInputs(nodes, edges), edges), edges }, signal);
        const inputs = nodes.filter((n) => n.type.startsWith('input_')).map((n) => String(n.data.label ?? n.id));
        return `Wrote flow ${id} ("${name}"): ${nodes.length} nodes, ${edges.length} edges${inputs.length ? `; its inputs: ${inputs.join(', ')}` : ''}. It is in the flow editor now${tool ? `, and ${tool.name} becomes one of your tools within a minute` : ''}. Run it with flow_run.`;
      },
    },
    {
      spec: {
        name: 'flow_run',
        description: 'Run a stored flow on OAIY Desktop and wait for what it returns (up to 2 minutes). input: its inputs by their labels.',
        parameters: { type: 'object', required: ['id'], properties: { id: { type: 'string' }, input: { type: 'object', description: 'The inputs, by label' } } },
      },
      run: async (input, signal) => {
        const result = await connected(desktop).runFlow(String(input.id), isRecord(input.input) ? input.input : {}, 120_000, signal);
        const status = String(result.status ?? '');
        if (status === 'succeeded') {
          const out = isRecord(result.output) && 'output' in result.output ? result.output.output : result.output;
          const text = typeof out === 'string' ? out : JSON.stringify(out ?? null, null, 1);
          return text.length > 8000 ? `${text.slice(0, 8000)}\n[cut at 8,000 characters]` : text;
        }
        if (status === 'queued' || status === 'running') return `Still running (run ${String(result.runId ?? '')}).`;
        const error = isRecord(result.error) ? String(result.error.message ?? result.error.code ?? '') : '';
        throw new Error(`the flow ${status || 'failed'}${error ? `: ${error}` : ''}`);
      },
    },
  ];
}
