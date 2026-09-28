/**
 * In OAIY's window the editor's flows and OAIY Desktop's are the same flows.
 *
 * The desktop keeps flows (its bridge's flow store): the agent in OAIY writes
 * them, runs them, and uses the ones made tools. Here, while the editor is
 * open in OAIY:
 * - the desktop's flows come into the editor's project (at start, then every
 *   15 seconds, so one the agent just wrote appears);
 * - a flow changed here is saved there a moment after the last edit (and, if
 *   it was made a tool, the tool's copy too); one deleted here is removed there.
 * Macros, demos and built-in flows stay the editor's own. Outside OAIY this
 * does nothing.
 */
import { useEffect, useRef } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Flow, OAIYProject } from 'oaiy-core';
import { oaiyDesktop, toolFlowId } from '../lib/oaiyAgentTools';

const PULL_MS = 15_000;
const SETTLE_MS = 1_200;

type Doc = { name?: string; nodes?: unknown[]; edges?: unknown[]; oaiyTool?: unknown };

/** What of a flow the desktop keeps: a change to any of it is sent. */
export function flowKey(f: Pick<Flow, 'name' | 'graph'>): string {
  return JSON.stringify([f.name, f.graph?.nodes ?? [], f.graph?.edges ?? []]);
}

/** Whether a flow of the editor's goes to the desktop. */
export function shared(f: Flow): boolean {
  return !f.isMacro && !f.isDemo && !f.isBuiltIn && !f.localOnly;
}

/** A desktop flow as a flow of the editor's. */
export function fromDoc(id: string, doc: Doc, now: string): Flow | null {
  if (!Array.isArray(doc.nodes)) return null;
  return { id, name: doc.name || id, createdAt: now, updatedAt: now, graph: { nodes: doc.nodes as Flow['graph']['nodes'], edges: (Array.isArray(doc.edges) ? doc.edges : []) as Flow['graph']['edges'] } };
}

export function useDesktopFlows(project: OAIYProject, setProject: Dispatch<SetStateAction<OAIYProject>>): void {
  const desktop = oaiyDesktop();
  const latest = useRef(project);
  latest.current = project;
  /** Each flow as last in step with the desktop. */
  const synced = useRef(new Map<string, string>());
  /** Ids on the desktop, as last listed. */
  const there = useRef(new Set<string>());
  /** Deleted here this session: not brought back by the next pull. */
  const deleted = useRef(new Set<string>());
  const pulledOnce = useRef(false);

  const call = async (method: string, path: string, body?: unknown): Promise<Response | null> => {
    if (!desktop) return null;
    try {
      return await fetch(`${desktop.origin}${path}`, {
        method,
        headers: { authorization: `Bearer ${desktop.token}`, ...(body ? { 'content-type': 'application/json' } : {}) },
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch {
      return null;
    }
  };

  // Bring the desktop's flows in.
  useEffect(() => {
    if (!desktop) return;
    let live = true;
    const pull = async () => {
      const resp = await call('GET', '/api/bridge/flows');
      if (!resp?.ok) return;
      const list = ((await resp.json().catch(() => ({}))) as { flows?: Array<{ flowId?: string }> }).flows ?? [];
      const ids = list.map((f) => String(f.flowId ?? '')).filter(Boolean);
      there.current = new Set(ids);
      const have = new Set(latest.current.flows.map((f) => f.id));
      // A tool's copy (`tool-…`) is the tool, not a second flow.
      const missing = ids.filter((id) => !id.startsWith('tool-') && !have.has(id) && !deleted.current.has(id));
      const now = new Date().toISOString();
      const fresh: Flow[] = [];
      for (const id of missing) {
        const doc = await call('GET', `/api/bridge/flows/${encodeURIComponent(id)}`);
        const flow = doc?.ok ? fromDoc(id, (await doc.json().catch(() => ({}))) as Doc, now) : null;
        if (flow) {
          fresh.push(flow);
          synced.current.set(flow.id, flowKey(flow));
        }
      }
      // What was there at the first pull and is here too starts in step.
      if (!pulledOnce.current) {
        for (const f of latest.current.flows) if (ids.includes(f.id) && !synced.current.has(f.id)) synced.current.set(f.id, '');
      }
      pulledOnce.current = true;
      if (live && fresh.length) {
        setProject((prev) => ({ ...prev, flows: [...prev.flows, ...fresh.filter((f) => !prev.flows.some((p) => p.id === f.id))] }));
      }
    };
    void pull();
    const timer = window.setInterval(() => void pull(), PULL_MS);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Send what changed here, and remove what was deleted here.
  useEffect(() => {
    if (!desktop) return;
    const timer = window.setTimeout(async () => {
      if (!pulledOnce.current) return;
      const current = new Map(project.flows.filter(shared).map((f) => [f.id, f]));
      for (const [id, f] of current) {
        const key = flowKey(f);
        if (synced.current.get(id) === key) continue;
        const saved = await call('PUT', `/api/bridge/flows/${encodeURIComponent(id)}`, { name: f.name, nodes: f.graph.nodes, edges: f.graph.edges });
        if (!saved?.ok) continue;
        synced.current.set(id, key);
        there.current.add(id);
        // Made a tool: its copy follows the flow, keeping its name and what it is for.
        const toolId = toolFlowId(f);
        if (there.current.has(toolId)) {
          const tool = await call('GET', `/api/bridge/flows/${encodeURIComponent(toolId)}`);
          const doc = tool?.ok ? ((await tool.json().catch(() => null)) as Doc | null) : null;
          if (doc?.oaiyTool) await call('PUT', `/api/bridge/flows/${encodeURIComponent(toolId)}`, { ...doc, name: f.name, nodes: f.graph.nodes, edges: f.graph.edges });
        }
      }
      for (const id of [...synced.current.keys()]) {
        if (current.has(id) || project.flows.some((f) => f.id === id)) continue;
        await call('DELETE', `/api/bridge/flows/${encodeURIComponent(id)}`);
        synced.current.delete(id);
        deleted.current.add(id);
      }
    }, SETTLE_MS);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.flows]);
}
