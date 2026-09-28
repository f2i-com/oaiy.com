/**
 * Flows as the agent's tools, from the flow editor in OAIY's window.
 *
 * OAIY's window gives its pages the desktop (`window.__OAIY_DESKTOP__`: its
 * address and a token). A flow made a tool is stored there, with a name and
 * what it is for; the agent in OAIY finds it, offers it as one of its tools
 * (the flow's input nodes are the tool's parameters, by their labels), and
 * runs it on the desktop when it uses it.
 *
 * A flow can instead stand in front of one of the agent's own tools, stored
 * with `oaiyToolHook: {tool, mode}`: `before` it (it is given the call and may
 * let it go ahead, change it or stop it) or `instead` of it (what it returns is
 * the tool's result).
 */
import type { Flow } from 'oaiy-core';

export interface AgentTool {
  name: string;
  description: string;
}

interface Desktop {
  origin: string;
  token: string;
}

/** The desktop, when this page is in OAIY's window. */
export function oaiyDesktop(): Desktop | null {
  const given = (window as unknown as { __OAIY_DESKTOP__?: { origin?: unknown; token?: unknown } }).__OAIY_DESKTOP__;
  return given && typeof given.origin === 'string' && typeof given.token === 'string' && given.token ? { origin: given.origin, token: given.token } : null;
}

/** A tool's name as the agent calls it: lower case, words joined by underscores. */
export function toolName(text: string): string {
  return text.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48) || 'flow_tool';
}

/** Where the desktop keeps a flow that is a tool. */
export function toolFlowId(flow: Flow): string {
  return `tool-${flow.id.replace(/[^A-Za-z0-9_-]+/g, '-')}`.slice(0, 96);
}

/** The agent's own tools a flow can stand in front of, with what each does. */
export const AGENT_TOOLS: Array<{ name: string; does: string }> = [
  { name: 'write_file', does: 'writes a file in the project' },
  { name: 'edit_file', does: 'changes part of a file' },
  { name: 'append_file', does: 'adds to the end of a file' },
  { name: 'delete_file', does: 'deletes a file' },
  { name: 'read_file', does: 'reads a file' },
  { name: 'web_fetch', does: 'fetches a web page' },
  { name: 'sandbox_shell', does: 'runs a shell command in the sandbox' },
  { name: 'code_run', does: 'runs code' },
  { name: 'generate_image', does: 'makes a picture' },
  { name: 'generate_video', does: 'makes a video' },
  { name: 'generate_speech', does: 'speaks text aloud' },
  { name: 'generate_music', does: 'makes music' },
  { name: 'send_text_message', does: 'sends a text message from the phone' },
  { name: 'request_appointment', does: 'asks for an appointment on a call or in a text' },
  { name: 'calendar_book', does: 'books an appointment' },
  { name: 'calendar_change', does: 'changes an appointment' },
  { name: 'lookup_business_data', does: "looks up the business's details on a call" },
  { name: 'end_call', does: 'ends a phone call' },
  { name: 'transcribe_audio', does: 'writes out a recording' },
  { name: 'flow_run', does: 'runs a flow' },
];

/** Where the desktop keeps a flow that stands in front of a tool. */
export function hookFlowId(flow: Flow): string {
  return `hook-${flow.id.replace(/[^A-Za-z0-9_-]+/g, '-')}`.slice(0, 96);
}

/** Store `flow` on the desktop in front of one of the agent's tools (again: it is updated). */
export async function publishAsHook(flow: Flow, hook: { tool: string; mode: 'before' | 'instead' }): Promise<void> {
  const desktop = oaiyDesktop();
  if (!desktop) throw new Error("Flows stand in front of the agent's tools in the OAIY app.");
  const tool = hook.tool.trim();
  if (!/^[a-z][a-z0-9_]{1,63}$/.test(tool)) throw new Error(`"${tool}" is not the name of one of the agent's tools`);
  const body = { name: flow.name, oaiyToolHook: { tool, mode: hook.mode, flowId: flow.id }, nodes: flow.graph.nodes, edges: flow.graph.edges };
  const resp = await fetch(`${desktop.origin}/api/bridge/flows/${encodeURIComponent(hookFlowId(flow))}`, {
    method: 'PUT',
    headers: { authorization: `Bearer ${desktop.token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`OAIY did not take it (HTTP ${resp.status}): ${(await resp.text()).slice(0, 200)}`);
}

/** How `flow` is given to the agent now: a tool of its own, in front of one, both, or neither. */
export async function agentUse(flow: Flow): Promise<{ tool?: { name: string; description: string }; hook?: { tool: string; mode: 'before' | 'instead' } }> {
  const desktop = oaiyDesktop();
  if (!desktop) return {};
  const read = async (id: string) => {
    const resp = await fetch(`${desktop.origin}/api/bridge/flows/${encodeURIComponent(id)}`, { headers: { authorization: `Bearer ${desktop.token}` } }).catch(() => null);
    return resp?.ok ? ((await resp.json().catch(() => null)) as Record<string, unknown> | null) : null;
  };
  const [tool, hook] = await Promise.all([read(toolFlowId(flow)), read(hookFlowId(flow))]);
  const t = tool?.oaiyTool as { name?: string; description?: string } | undefined;
  const h = hook?.oaiyToolHook as { tool?: string; mode?: string } | undefined;
  return {
    ...(t?.name ? { tool: { name: t.name, description: t.description ?? '' } } : {}),
    ...(h?.tool && (h.mode === 'before' || h.mode === 'instead') ? { hook: { tool: h.tool, mode: h.mode } } : {}),
  };
}

/** Take `flow` back from in front of the agent's tool. */
export async function withdrawHook(flow: Flow): Promise<void> {
  const desktop = oaiyDesktop();
  if (!desktop) return;
  await fetch(`${desktop.origin}/api/bridge/flows/${encodeURIComponent(hookFlowId(flow))}`, { method: 'DELETE', headers: { authorization: `Bearer ${desktop.token}` } });
}

/** Store `flow` on the desktop as a tool for the agent (again: the tool is updated). */
export async function publishAsTool(flow: Flow, tool: AgentTool): Promise<void> {
  const desktop = oaiyDesktop();
  if (!desktop) throw new Error('Flows become tools for the agent in the OAIY app.');
  const body = {
    name: flow.name,
    oaiyTool: { name: toolName(tool.name), description: tool.description.trim(), flowId: flow.id },
    nodes: flow.graph.nodes,
    edges: flow.graph.edges,
  };
  const resp = await fetch(`${desktop.origin}/api/bridge/flows/${encodeURIComponent(toolFlowId(flow))}`, {
    method: 'PUT',
    headers: { authorization: `Bearer ${desktop.token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`OAIY did not take the tool (HTTP ${resp.status}): ${(await resp.text()).slice(0, 200)}`);
}

/** Stop offering `flow` to the agent. */
export async function withdrawTool(flow: Flow): Promise<void> {
  const desktop = oaiyDesktop();
  if (!desktop) return;
  await fetch(`${desktop.origin}/api/bridge/flows/${encodeURIComponent(toolFlowId(flow))}`, {
    method: 'DELETE',
    headers: { authorization: `Bearer ${desktop.token}` },
  });
}
