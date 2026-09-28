/**
 * Flows as the agent's tools, from the flow editor in OAIY's window.
 *
 * OAIY's window gives its pages the desktop (`window.__OAIY_DESKTOP__`: its
 * address and a token). A flow made a tool is stored there, with a name and
 * what it is for; the agent in OAIY finds it, offers it as one of its tools
 * (the flow's input nodes are the tool's parameters, by their labels), and
 * runs it on the desktop when it uses it.
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
