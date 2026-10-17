/**
 * In OAIY's window the editor's flows and OAIY Desktop's are the same flows.
 *
 * The desktop keeps flows (its bridge's flow store): the agent in OAIY writes
 * them, runs them, and uses the ones made tools. Here, while the editor is
 * open in OAIY:
 * - the desktop's flows come into the editor's project, at start and every 15
 *   seconds: a new one appears, and one the agent changed is updated here;
 * - a flow changed here is saved there a moment after the last edit (and, if
 *   it was made a tool, the tool's copy too); one deleted here is removed there,
 *   and one deleted there (unchanged here) is removed here.
 *
 * Which side changed is told three ways: each flow as it was when the two last
 * agreed is kept (in this page's storage, so a restart knows it too). A flow
 * changed on one side takes that side's; changed on both, this side's goes there.
 * A flow's name, description, tags, whether it is shared with apps, nodes and
 * edges go both ways; what else the desktop's document holds (a tool's
 * description the agent wrote) is kept as it is. Macros, demos and built-in
 * flows stay the editor's own. Outside OAIY this does nothing.
 */
import { useEffect, useRef } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Flow, OAIYProject } from 'oaiy-core';
import { hookFlowId, oaiyDesktop, toolFlowId } from '../lib/oaiyAgentTools';

const PULL_MS = 15_000;
const SETTLE_MS = 1_200;
// (`-2`: a flow's description, tags and sharing became part of what is compared)
const SYNCED_KEY = 'oaiy-desktop-flows-synced-2';

type Doc = {
  name?: string;
  description?: unknown;
  tags?: unknown;
  oaiyRemote?: unknown;
  nodes?: unknown[];
  edges?: unknown[];
  oaiyTool?: unknown;
  oaiyToolHook?: unknown;
};

/** What of a flow the desktop keeps: a change to any of it is sent. */
export function flowKey(f: Pick<Flow, 'name' | 'graph' | 'description' | 'tags' | 'remote'>): string {
  return JSON.stringify([f.name, f.description ?? '', f.tags ?? [], !!f.remote, f.graph?.nodes ?? [], f.graph?.edges ?? []]);
}

/** A flow as the desktop's document of it: `doc` (as it is stored there) with this flow's part put in. */
export function toDoc(f: Flow, doc: Doc | null = null): Doc {
  return {
    ...(doc ?? {}),
    name: f.name,
    description: f.description || undefined,
    tags: f.tags?.length ? f.tags : undefined,
    oaiyRemote: f.remote ? true : undefined,
    nodes: f.graph.nodes,
    edges: f.graph.edges,
  };
}

/** Whether a flow of the editor's goes to the desktop. */
export function shared(f: Flow): boolean {
  return !f.isMacro && !f.isDemo && !f.isBuiltIn && !f.localOnly;
}

/** A desktop flow as a flow of the editor's. */
export function fromDoc(id: string, doc: Doc, now: string): Flow | null {
  if (!Array.isArray(doc.nodes)) return null;
  const tags = Array.isArray(doc.tags) ? doc.tags.filter((t): t is string => typeof t === 'string') : [];
  return {
    id,
    name: doc.name || id,
    ...(typeof doc.description === 'string' && doc.description ? { description: doc.description } : {}),
    ...(tags.length ? { tags } : {}),
    ...(doc.oaiyRemote === true ? { remote: true } : {}),
    createdAt: now,
    updatedAt: now,
    graph: { nodes: doc.nodes as Flow['graph']['nodes'], edges: (Array.isArray(doc.edges) ? doc.edges : []) as Flow['graph']['edges'] },
  };
}

/** What a pull changes here: flows to add or replace, ids to remove, and the flows as the two sides now agree. */
export interface Pulled {
  put: Flow[];
  remove: string[];
  synced: Map<string, string>;
}

/**
 * The desktop's flows against the editor's, given each as the two last agreed.
 * Flows changed here are left for the push (it sends them); so are flows
 * deleted here (it removes them there).
 */
export function reconcile(there: Map<string, Flow>, here: Flow[], agreed: Map<string, string>): Pulled {
  const synced = new Map(agreed);
  const put: Flow[] = [];
  const remove: string[] = [];
  const local = new Map(here.map((f) => [f.id, f]));
  for (const [id, theirs] of there) {
    const last = agreed.get(id);
    const mine = local.get(id);
    const theirKey = flowKey(theirs);
    if (!mine) {
      // New there: it comes here. Known before and gone here: deleted here, the push removes it there.
      if (last === undefined) {
        put.push(theirs);
        synced.set(id, theirKey);
      }
      continue;
    }
    const myKey = flowKey(mine);
    if (myKey === theirKey) {
      synced.set(id, theirKey);
    } else if (last !== theirKey && (last === undefined || myKey === last)) {
      // Changed there, not here: theirs.
      put.push({ ...mine, name: theirs.name, description: theirs.description, tags: theirs.tags, remote: theirs.remote, graph: theirs.graph });
      synced.set(id, theirKey);
    }
    // Changed here (or both): the push sends this side's.
  }
  for (const [id, last] of agreed) {
    if (there.has(id)) continue;
    const mine = local.get(id);
    // Gone there: if unchanged here it was deleted there; changed here, the push sends it back.
    if (!mine || flowKey(mine) === last) {
      if (mine) remove.push(id);
      synced.delete(id);
    }
  }
  return { put, remove, synced };
}

function loadSynced(): Map<string, string> {
  try {
    const saved = JSON.parse(localStorage.getItem(SYNCED_KEY) || '{}') as Record<string, unknown>;
    return new Map(Object.entries(saved).filter((e): e is [string, string] => typeof e[1] === 'string'));
  } catch {
    return new Map();
  }
}

function saveSynced(synced: Map<string, string>): void {
  try {
    localStorage.setItem(SYNCED_KEY, JSON.stringify(Object.fromEntries(synced)));
  } catch {
    // storage full or unavailable: the next start treats every flow as unknown
  }
}

/**
 * `changedThere` hears the ids of flows here replaced by the desktop's (the
 * canvas redraws the open one).
 */
export function useDesktopFlows(project: OAIYProject, setProject: Dispatch<SetStateAction<OAIYProject>>, changedThere?: (ids: string[]) => void): void {
  const desktop = oaiyDesktop();
  const onChanged = useRef(changedThere);
  onChanged.current = changedThere;
  const latest = useRef(project);
  latest.current = project;
  /** Each flow as it was when the two sides last agreed. */
  const synced = useRef<Map<string, string>>(desktop ? loadSynced() : new Map());
  /** Ids on the desktop, as last listed. */
  const there = useRef(new Set<string>());
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

  // Bring the desktop's flows in: new ones, and ones changed there.
  useEffect(() => {
    if (!desktop) return;
    let live = true;
    const pull = async () => {
      const resp = await call('GET', '/api/bridge/flows');
      if (!resp?.ok) return;
      const list = ((await resp.json().catch(() => ({}))) as { flows?: Array<{ flowId?: string }> }).flows ?? [];
      const ids = list.map((f) => String(f.flowId ?? '')).filter(Boolean);
      there.current = new Set(ids);
      const now = new Date().toISOString();
      const flows = new Map<string, Flow>();
      // A tool's copy (`tool-…`) is the tool, and a hook's (`hook-…`) the hook, not second flows.
      for (const id of ids.filter((id) => !id.startsWith('tool-') && !id.startsWith('hook-'))) {
        const doc = await call('GET', `/api/bridge/flows/${encodeURIComponent(id)}`);
        const flow = doc?.ok ? fromDoc(id, (await doc.json().catch(() => ({}))) as Doc, now) : null;
        if (flow) flows.set(id, flow);
      }
      if (!live) return;
      const { put, remove, synced: agreed } = reconcile(flows, latest.current.flows.filter(shared), synced.current);
      synced.current = agreed;
      saveSynced(agreed);
      pulledOnce.current = true;
      if (put.length || remove.length) {
        setProject((prev) => {
          const replaced = new Map(put.map((f) => [f.id, f]));
          const kept = prev.flows.filter((f) => !remove.includes(f.id)).map((f) => replaced.get(f.id) ?? f);
          const added = put.filter((f) => !prev.flows.some((p) => p.id === f.id));
          return { ...prev, flows: [...kept, ...added] };
        });
        const replaced = put.filter((f) => latest.current.flows.some((p) => p.id === f.id)).map((f) => f.id);
        if (replaced.length) onChanged.current?.(replaced);
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
      // Not before the first pull: a flow changed there while the editor was closed comes here first.
      if (!pulledOnce.current) return;
      const current = new Map(project.flows.filter(shared).map((f) => [f.id, f]));
      for (const [id, f] of current) {
        const key = flowKey(f);
        if (synced.current.get(id) === key) continue;
        // Into the document as the desktop has it, so what the editor does not show (a description the agent
        // reads) is not lost.
        const stored = there.current.has(id) ? await call('GET', `/api/bridge/flows/${encodeURIComponent(id)}`) : null;
        const doc = stored?.ok ? ((await stored.json().catch(() => null)) as Doc | null) : null;
        const saved = await call('PUT', `/api/bridge/flows/${encodeURIComponent(id)}`, toDoc(f, doc));
        if (!saved?.ok) continue;
        synced.current.set(id, key);
        there.current.add(id);
        // Made a tool, or put in front of one: each copy follows the flow, keeping what it is to the agent.
        for (const copyId of [toolFlowId(f), hookFlowId(f)]) {
          if (!there.current.has(copyId)) continue;
          const copy = await call('GET', `/api/bridge/flows/${encodeURIComponent(copyId)}`);
          const doc = copy?.ok ? ((await copy.json().catch(() => null)) as Doc | null) : null;
          if (doc?.oaiyTool || doc?.oaiyToolHook) await call('PUT', `/api/bridge/flows/${encodeURIComponent(copyId)}`, { ...doc, name: f.name, nodes: f.graph.nodes, edges: f.graph.edges });
        }
      }
      for (const id of [...synced.current.keys()]) {
        if (current.has(id) || project.flows.some((f) => f.id === id)) continue;
        const gone = await call('DELETE', `/api/bridge/flows/${encodeURIComponent(id)}`);
        if (gone && (gone.ok || gone.status === 404)) synced.current.delete(id);
        // A deleted flow is no longer the agent's tool, nor in front of one.
        for (const copyId of [toolFlowId({ id } as Flow), hookFlowId({ id } as Flow)]) {
          if (there.current.has(copyId)) await call('DELETE', `/api/bridge/flows/${encodeURIComponent(copyId)}`);
        }
      }
      saveSynced(synced.current);
    }, SETTLE_MS);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.flows]);
}
