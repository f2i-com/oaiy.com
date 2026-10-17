/**
 * Flows as the agent's tools. A flow made a tool in the flow editor is stored
 * on OAIY Desktop with a name and what it is for (`oaiyTool`); here it becomes
 * one of the agent's tools: its input nodes are the parameters (by their
 * labels, as the flow engine takes inputs), and using it runs the flow on the
 * desktop and hands back what it returned.
 *
 * A flow can also stand in front of one of the agent's own tools
 * (`oaiyToolHook: {tool, mode}`): run before it (to check, change or log the
 * call) or instead of it (the person's own way of doing it). See `ToolHook`.
 */
import type { SessionTool, ToolHook } from '../agent/agent';
import type { ToolSpec } from '../agent/protocol';
import type { Vfs } from '../vfs/vfs';
import type { Desktop } from './bridge';

export interface FlowTool {
  /** The flow's id in the desktop's store. */
  id: string;
  /** What the agent calls it. */
  name: string;
  description: string;
  flowName: string;
  /** The flow's inputs: their labels and what kind each is. */
  inputs: Array<{ label: string; type: string }>;
}

const INPUT_KINDS: Record<string, string> = {
  input_text: 'Text',
  input_file: 'The path of a file on this computer',
  input_folder: 'The path of a folder on this computer',
  input_audio: 'The path of an audio file on this computer',
  input_video: 'The path of a video file on this computer',
};

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

/** A stored flow as a tool, or null when it is not one. */
export function readFlowTool(id: string, doc: unknown): FlowTool | null {
  if (!isRecord(doc) || !isRecord(doc.oaiyTool)) return null;
  const tool = doc.oaiyTool;
  const name = typeof tool.name === 'string' ? tool.name.toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48) : '';
  if (!name) return null;
  const graph = isRecord(doc.graph) ? doc.graph : doc;
  const nodes = Array.isArray(graph.nodes) ? graph.nodes.filter(isRecord) : [];
  const inputs = nodes
    .filter((n) => typeof n.type === 'string' && n.type in INPUT_KINDS)
    .map((n) => ({ label: String((isRecord(n.data) && typeof n.data.label === 'string' && n.data.label.trim()) || n.id), type: String(n.type) }));
  return {
    id,
    name,
    description: typeof tool.description === 'string' && tool.description.trim() ? tool.description.trim() : `Runs the flow "${String(doc.name ?? id)}".`,
    flowName: String(doc.name ?? id),
    inputs,
  };
}

/** A parameter's name from an input's label (the flow still gets the label itself). */
function paramName(label: string): string {
  return label.trim().replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'input';
}

/** The tool as the model sees it. `taken`: the names already in use (a clash gets `flow_` in front). */
export function flowToolSpec(tool: FlowTool, taken: Set<string>): ToolSpec {
  const properties: Record<string, unknown> = {};
  for (const input of tool.inputs) properties[paramName(input.label)] = { type: 'string', description: `${INPUT_KINDS[input.type] ?? 'Text'}: the flow's "${input.label}" input` };
  return {
    name: taken.has(tool.name) ? `flow_${tool.name}` : tool.name,
    description: `${tool.description} (A flow of yours, "${tool.flowName}", run by OAIY.)`,
    parameters: { type: 'object', required: Object.keys(properties), properties },
  };
}

/** The flow's own output in what a run returned (the engine's job answer holds it beside how it ran). */
function flowOutput(result: Record<string, unknown>): unknown {
  return isRecord(result.output) && 'output' in result.output ? result.output.output : result.output;
}

/** What a run returned, for the model: its output, at most 8,000 characters. */
export function runOutcome(result: Record<string, unknown>): string {
  const status = String(result.status ?? '');
  if (status === 'succeeded') {
    const value = flowOutput(result);
    const out = typeof value === 'string' ? value : JSON.stringify(value ?? null, null, 1);
    return out.length > 8000 ? `${out.slice(0, 8000)}\n[cut at 8,000 characters]` : out;
  }
  if (status === 'queued' || status === 'running') return `The flow is still running (run ${String(result.runId ?? '')}); its result was not ready in time.`;
  const error = isRecord(result.error) ? String(result.error.message ?? result.error.code ?? '') : String(result.error ?? '');
  throw new Error(`the flow ${status || 'failed'}${error ? `: ${error}` : ''}`);
}

/** How long a flow tool is waited for: a flow may load a picture model, then another, then a language model again. */
export const FLOW_TOOL_WAIT_MS = 600_000;

const MEDIA_TYPES: Record<string, string> = {
  'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif',
  'video/mp4': 'mp4', 'video/webm': 'webm', 'audio/mpeg': 'mp3', 'audio/wav': 'wav', 'audio/ogg': 'ogg', 'audio/flac': 'flac',
};

function dataBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * The files in a flow's output kept in the project (`flows/<tool>/`): the
 * pictures OAIY's engine made, by the desktop URLs the flow gave for them, and
 * any data: URL. Each is replaced by its project path, so the agent reads
 * paths it can look at (view_image) or edit with (reference_images), and never
 * a picture's bytes. `fetchFile` answers null for a URL that is not the
 * engine's; such a value is left as it is.
 */
export async function keepFlowFiles(
  value: unknown,
  tool: string,
  fetchFile: (url: string) => Promise<Uint8Array | null>,
  vfs: Vfs,
): Promise<{ value: unknown; saved: string[] }> {
  const saved: string[] = [];
  const done = new Map<string, string>();
  const folder = `flows/${tool}`;
  const place = (name: string) => {
    let path = `${folder}/${name}`;
    for (let n = 2; vfs.exists(`/${path}`); n++) path = `${folder}/${name.replace(/(\.[a-z0-9]+)$/i, `-${n}$1`)}`;
    return path;
  };
  const keep = async (text: string): Promise<string> => {
    const known = done.get(text);
    if (known) return known;
    let path: string | null = null;
    const data = /^data:([a-z]+\/[a-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$/i.exec(text);
    if (data && MEDIA_TYPES[data[1].toLowerCase()]) {
      path = place(`${tool}-${saved.length + 1}.${MEDIA_TYPES[data[1].toLowerCase()]}`);
      vfs.writeFile(`/${path}`, dataBytes(data[2].replace(/\s+/g, '')), { parents: true });
    } else if (/^https?:\/\//i.test(text)) {
      let name = '';
      try {
        name = decodeURIComponent(new URL(text).pathname.split('/').pop() ?? '').replace(/[^A-Za-z0-9._-]+/g, '_');
      } catch {
        return text;
      }
      if (!/\.(png|jpe?g|webp|gif|mp4|webm|mp3|wav|ogg|flac)$/i.test(name)) return text;
      const bytes = await fetchFile(text);
      if (!bytes) return text;
      path = place(name);
      vfs.writeFile(`/${path}`, bytes, { parents: true });
    }
    if (!path) return text;
    saved.push(path);
    done.set(text, path);
    return path;
  };
  const walk = async (v: unknown, depth: number): Promise<unknown> => {
    if (typeof v === 'string') return keep(v);
    if (depth > 8) return v;
    if (Array.isArray(v)) {
      const out: unknown[] = [];
      for (const x of v) out.push(await walk(x, depth + 1));
      return out;
    }
    if (isRecord(v)) {
      const out: Record<string, unknown> = {};
      for (const [k, x] of Object.entries(v)) out[k] = await walk(x, depth + 1);
      return out;
    }
    return v;
  };
  return { value: await walk(value, 0), saved };
}

/**
 * The agent's tools for the flows made tools on the desktop. With `files` (the
 * project's), the pictures a flow hands back are kept there (see keepFlowFiles).
 */
export function flowSessionTools(tools: FlowTool[], desktop: () => Desktop | null, taken: Set<string>, files?: () => Vfs | null): SessionTool[] {
  return tools.map((tool) => ({
    spec: flowToolSpec(tool, taken),
    run: async (input, signal) => {
      const d = desktop();
      if (!d) throw new Error('OAIY Desktop is not connected, so flows cannot run');
      // The flow takes its inputs by their labels.
      const byLabel: Record<string, string> = {};
      for (const i of tool.inputs) byLabel[i.label] = String(input[paramName(i.label)] ?? '');
      const result = await d.runFlow(tool.id, byLabel, FLOW_TOOL_WAIT_MS, signal);
      const vfs = files?.();
      if (!vfs || result.status !== 'succeeded') return runOutcome(result);
      const kept = await keepFlowFiles(flowOutput(result), tool.name, (url) => d.engineFile(url, signal), vfs);
      const outcome = runOutcome({ status: 'succeeded', output: { output: kept.value } });
      return kept.saved.length
        ? `${outcome}\n\nSaved in the project: ${kept.saved.join(', ')} (look at a picture with view_image; show it to the person with present_file)`
        : outcome;
    },
  }));
}

/** A flow in front of one of the agent's tools. */
export interface FlowHook {
  /** The flow's id in the desktop's store. */
  id: string;
  /** The tool it stands in front of. */
  tool: string;
  mode: 'before' | 'instead';
  flowName: string;
  inputs: Array<{ label: string; type: string }>;
}

/** A stored flow as a hook, or null when it is not one. */
export function readFlowHook(id: string, doc: unknown): FlowHook | null {
  if (!isRecord(doc) || !isRecord(doc.oaiyToolHook)) return null;
  const hook = doc.oaiyToolHook;
  const tool = typeof hook.tool === 'string' ? hook.tool.trim() : '';
  const mode = hook.mode === 'instead' ? 'instead' : hook.mode === 'before' ? 'before' : null;
  if (!tool || !mode) return null;
  const graph = isRecord(doc.graph) ? doc.graph : doc;
  const nodes = Array.isArray(graph.nodes) ? graph.nodes.filter(isRecord) : [];
  const inputs = nodes
    .filter((n) => typeof n.type === 'string' && n.type in INPUT_KINDS)
    .map((n) => ({ label: String((isRecord(n.data) && typeof n.data.label === 'string' && n.data.label.trim()) || n.id), type: String(n.type) }));
  return { id, tool, mode, flowName: String(doc.name ?? id), inputs };
}

/**
 * What a hook's flow is given from a call, by its inputs' labels: an input
 * labelled like one of the call's parameters gets that parameter; `tool` gets
 * the tool's name; `input` (or `call`) gets the call's whole input as JSON.
 */
export function hookInputs(hook: FlowHook, call: { name: string; input: Record<string, unknown> }): Record<string, string> {
  const text = (v: unknown) => (typeof v === 'string' ? v : JSON.stringify(v ?? null));
  const out: Record<string, string> = {};
  for (const { label } of hook.inputs) {
    const key = label.trim();
    const param = Object.keys(call.input).find((k) => k.toLowerCase() === key.toLowerCase() || paramName(k).toLowerCase() === paramName(key).toLowerCase());
    if (param) out[label] = text(call.input[param]);
    else if (/^tool$/i.test(key)) out[label] = call.name;
    else if (/^(input|call)$/i.test(key)) out[label] = JSON.stringify(call.input);
    else out[label] = '';
  }
  return out;
}

export { beforeVerdict, type BeforeVerdict } from '../agent/hookVerdict';

/** The hooks as the agents take them: each runs its flow on the desktop. */
export function flowToolHooks(hooks: FlowHook[], desktop: () => Desktop | null): ToolHook[] {
  return hooks.map((hook) => ({
    tool: hook.tool,
    mode: hook.mode,
    flowName: hook.flowName,
    run: async (call, signal) => {
      const d = desktop();
      if (!d) throw new Error('OAIY Desktop is not connected, so the flow cannot run');
      return runOutcome(await d.runFlow(hook.id, hookInputs(hook, call), 120_000, signal));
    },
  }));
}

/** The flows made tools and the flows in front of tools, read from the desktop. */
export async function readFlowStore(desktop: Desktop, signal?: AbortSignal): Promise<{ tools: FlowTool[]; hooks: FlowHook[] }> {
  const tools: FlowTool[] = [];
  const hooks: FlowHook[] = [];
  for (const { id } of await desktop.flows(signal)) {
    const doc = await desktop.flow(id, signal).catch(() => null);
    const tool = readFlowTool(id, doc);
    if (tool && !tools.some((t) => t.name === tool.name)) tools.push(tool);
    const hook = readFlowHook(id, doc);
    // One flow in front of a tool at a time, each way.
    if (hook && !hooks.some((h) => h.tool === hook.tool && h.mode === hook.mode)) hooks.push(hook);
  }
  return { tools, hooks };
}

/** The flows made tools, read from the desktop. */
export async function listFlowTools(desktop: Desktop, signal?: AbortSignal): Promise<FlowTool[]> {
  return (await readFlowStore(desktop, signal)).tools;
}
