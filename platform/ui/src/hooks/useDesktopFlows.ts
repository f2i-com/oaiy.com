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
 * agreed is kept (in this page's storage, saved only with the project itself, so
 * a restart reads the two as they were together). A flow changed on one side
 * takes that side's; changed on both, this side's goes there. Only a flow
 * deleted here (Delete) is removed there: one missing here for another reason
 * (a new project, an import, a project that could not be read) comes back.
 * A flow's name, description, tags, whether it is shared with apps, whether it
 * is a macro, nodes and edges go both ways; what else the desktop's document
 * holds (a tool's description the agent wrote) is kept as it is. A macro goes
 * too (marked `isMacro`): a flow that uses one is run there with it (the
 * desktop hands the CLI a flow's subflows and macros by id), where it failed
 * "Subflow not found". Demos and built-in flows stay the editor's own. A node
 * naming the person's own service, or one of the desktop's, carries what it
 * takes to call it (`withNodeServices`): the CLI has none of the editor's lists.
 * A flow the desktop refuses to keep is said once (`subscribeDesktopFlowUnsaved`),
 * and tried again after the next pull. Outside OAIY this does nothing.
 */
import { useEffect, useRef } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Flow, OAIYProject } from 'oaiy-core';
import { hookFlowId, oaiyDesktop, toolFlowId } from '../lib/oaiyAgentTools';
import { withNodeServices, withoutNodeServices } from '../bundled-modules/core-service/contract';

const PULL_MS = 15_000;
const SETTLE_MS = 1_200;
// (`-2`: a flow's description, tags and sharing became part of what is compared; `-3`: whether it is a macro)
const SYNCED_KEY = 'oaiy-desktop-flows-synced-3';

type Doc = {
  name?: string;
  description?: unknown;
  tags?: unknown;
  oaiyRemote?: unknown;
  isMacro?: unknown;
  nodes?: unknown[];
  edges?: unknown[];
  oaiyTool?: unknown;
  oaiyToolHook?: unknown;
};

/** What of a flow the desktop keeps: a change to any of it is sent. */
export function flowKey(f: Pick<Flow, 'name' | 'graph' | 'description' | 'tags' | 'remote' | 'isMacro'>): string {
  return JSON.stringify([f.name, f.description ?? '', f.tags ?? [], !!f.remote, !!f.isMacro, f.graph?.nodes ?? [], f.graph?.edges ?? []]);
}

/** A flow as the desktop's document of it: `doc` (as it is stored there) with this flow's part put in. */
export function toDoc(f: Flow, doc: Doc | null = null): Doc {
  return {
    ...(doc ?? {}),
    name: f.name,
    description: f.description || undefined,
    tags: f.tags?.length ? f.tags : undefined,
    oaiyRemote: f.remote ? true : undefined,
    isMacro: f.isMacro ? true : undefined,
    nodes: withNodeServices(f.graph.nodes),
    edges: f.graph.edges,
  };
}

/** Whether a flow of the editor's goes to the desktop: a macro of the person's own does (the flows using it run there
 * with it); a built-in one does not (it comes with the editor), nor a flow from a shared link (untrusted here, it would
 * run there, the Agent's to run, unsandboxed). */
export function shared(f: Flow): boolean {
  return !f.isDemo && !f.isBuiltIn && !f.localOnly && (f as { untrusted?: unknown }).untrusted !== true;
}

/** A macro's inputs and outputs, from its Macro Input and Macro Output nodes (what the palette shows of it). */
export function macroMetadataOf(nodes: Flow['graph']['nodes']): NonNullable<Flow['macroMetadata']> {
  const named = (n: Flow['graph']['nodes'][number], field: string, fallback: string) => String((n.data as Record<string, unknown> | undefined)?.[field] || fallback);
  return {
    inputs: nodes
      .filter((n) => n.type === 'macro_input')
      .map((n) => {
        const data = (n.data ?? {}) as Record<string, unknown>;
        return { id: n.id, name: named(n, 'name', 'input'), type: named(n, 'inputType', 'any'), required: Boolean(data.required), defaultValue: data.defaultValue ? String(data.defaultValue) : undefined };
      }),
    outputs: nodes.filter((n) => n.type === 'macro_output').map((n) => ({ id: n.id, name: named(n, 'name', 'output'), type: named(n, 'outputType', 'any') })),
  };
}

/** A desktop flow as a flow of the editor's. */
export function fromDoc(id: string, doc: Doc, now: string): Flow | null {
  if (!Array.isArray(doc.nodes)) return null;
  const tags = Array.isArray(doc.tags) ? doc.tags.filter((t): t is string => typeof t === 'string') : [];
  // (the services copied in for the desktop are its, not the editor's: the editor reads its own lists)
  const nodes = withoutNodeServices(doc.nodes as Flow['graph']['nodes']);
  return {
    id,
    name: doc.name || id,
    ...(typeof doc.description === 'string' && doc.description ? { description: doc.description } : {}),
    ...(tags.length ? { tags } : {}),
    ...(doc.oaiyRemote === true ? { remote: true } : {}),
    ...(doc.isMacro === true ? { isMacro: true, macroMetadata: macroMetadataOf(nodes) } : {}),
    createdAt: now,
    updatedAt: now,
    graph: { nodes, edges: (Array.isArray(doc.edges) ? doc.edges : []) as Flow['graph']['edges'] },
  };
}

/** What a flow the desktop did not keep is said with (`subscribeDesktopFlowUnsaved`). */
export interface DesktopFlowUnsaved {
  id: string;
  name: string;
  /** What the desktop said, or that it could not be reached. */
  reason: string;
}

const UNSAVED_EVENT = 'oaiy:desktop-flow-unsaved';
/** Flows said not to be kept, each said once until a save of it works. */
const toldUnsaved = new Set<string>();

function tellUnsaved(f: Flow, reason: string): void {
  if (toldUnsaved.has(f.id)) return;
  toldUnsaved.add(f.id);
  try {
    window.dispatchEvent(new CustomEvent<DesktopFlowUnsaved>(UNSAVED_EVENT, { detail: { id: f.id, name: f.name, reason } }));
  } catch {
    // (no CustomEvent: nothing to say it with)
  }
}

/** Hear of a flow OAIY Desktop did not keep (OAIYApp says so); returns the way to stop. */
export function subscribeDesktopFlowUnsaved(fn: (detail: DesktopFlowUnsaved) => void): () => void {
  const listener = (e: Event) => fn((e as CustomEvent<DesktopFlowUnsaved>).detail);
  window.addEventListener(UNSAVED_EVENT, listener);
  return () => window.removeEventListener(UNSAVED_EVENT, listener);
}

/** What a refused save says: the desktop's own words, else its status. */
async function refusal(resp: Response | null): Promise<string> {
  if (!resp) return 'OAIY Desktop could not be reached';
  const body = (await resp.json().catch(() => null)) as { error?: { message?: unknown } } | null;
  const said = typeof body?.error?.message === 'string' ? body.error.message : '';
  return said ? `OAIY Desktop said: ${said}` : `OAIY Desktop answered ${resp.status}`;
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
 * deleted here (`deleted`: it removes them there). A flow known before and
 * missing here, not deleted here, comes back. `listed`: the ids the desktop
 * listed, a flow among them that could not be read (absent from `there`) is
 * left as it is on both sides, not taken for deleted there.
 */
export function reconcile(
  there: Map<string, Flow>,
  here: Flow[],
  agreed: Map<string, string>,
  deleted: ReadonlySet<string> = new Set(),
  listed?: ReadonlySet<string>,
): Pulled {
  const synced = new Map(agreed);
  const put: Flow[] = [];
  const remove: string[] = [];
  const local = new Map(here.map((f) => [f.id, f]));
  for (const [id, theirs] of there) {
    const last = agreed.get(id);
    const mine = local.get(id);
    const theirKey = flowKey(theirs);
    if (!mine) {
      // New there: it comes here. Deleted here: the push removes it there. Known before and missing here for
      // another reason (a new project, an import): it comes back, rather than being deleted there.
      if (!deleted.has(id)) {
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
      put.push({
        ...mine,
        name: theirs.name,
        description: theirs.description,
        tags: theirs.tags,
        remote: theirs.remote,
        isMacro: theirs.isMacro,
        macroMetadata: theirs.isMacro ? macroMetadataOf(theirs.graph.nodes) : undefined,
        graph: theirs.graph,
      });
      synced.set(id, theirKey);
    }
    // Changed here (or both): the push sends this side's.
  }
  for (const [id, last] of agreed) {
    if (there.has(id) || listed?.has(id)) continue;
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

/** The agreed flows waiting to be saved with the project's next write (`projectWritten`). */
let unsaved: Map<string, string> | null = null;
/** Flows deleted here (Delete) and not yet removed there. */
const deletedHere = new Set<string>();
/** Asked to forget what the two sides agreed (a new project, an import): done at the next push. */
let forgetAsked = false;

/** The project was written to this page's storage: what the two sides agreed is saved with it, never ahead of it. */
export function projectWritten(): void {
  if (!unsaved) return;
  saveSynced(unsaved);
  unsaved = null;
}

/**
 * Flow `id` was deleted here, by the person, or is to be kept here only ("Keep it in this window only"): it is
 * removed on the desktop too (a copy left there went stale, the agent and apps still seeing it).
 */
export function markDeletedHere(id: string): void {
  deletedHere.add(id);
}

/**
 * The project was replaced (a new one, or one imported): what the two sides last agreed no longer says anything,
 * so nothing missing here is deleted there, and the desktop's flows come back into the new project.
 */
export function forgetDesktopSync(): void {
  forgetAsked = true;
  deletedHere.clear();
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
  /** A push under way, and whether another was asked for meanwhile (it runs after, with what is newest then). */
  const pushing = useRef(false);
  const pushAgain = useRef(false);
  const pushRef = useRef<() => Promise<void>>(async () => {});

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

  const keep = () => {
    unsaved = new Map(synced.current);
  };

  /** Send what changed here, and remove what was deleted here: one at a time, again after if asked meanwhile. */
  const push = async (): Promise<void> => {
    if (!desktop) return;
    // Not before the first pull: a flow changed there while the editor was closed comes here first.
    if (!pulledOnce.current) return;
    if (pushing.current) {
      pushAgain.current = true;
      return;
    }
    pushing.current = true;
    try {
      do {
        pushAgain.current = false;
        if (forgetAsked) {
          forgetAsked = false;
          synced.current = new Map();
          keep();
        }
        const flows = latest.current.flows;
        const current = new Map(flows.filter(shared).map((f) => [f.id, f]));
        for (const [id, f] of current) {
          const key = flowKey(f);
          if (synced.current.get(id) === key) continue;
          // Into the document as the desktop has it, so what the editor does not show (a description the agent
          // reads) is not lost.
          const stored = there.current.has(id) ? await call('GET', `/api/bridge/flows/${encodeURIComponent(id)}`) : null;
          const doc = stored?.ok ? ((await stored.json().catch(() => null)) as Doc | null) : null;
          const saved = await call('PUT', `/api/bridge/flows/${encodeURIComponent(id)}`, toDoc(f, doc));
          // (not sent: said once, and tried again after the next pull, as nothing says the two agree)
          if (!saved?.ok) {
            tellUnsaved(f, await refusal(saved));
            continue;
          }
          toldUnsaved.delete(id);
          synced.current.set(id, key);
          there.current.add(id);
          // Made a tool, or put in front of one: each copy follows the flow, keeping what it is to the agent.
          for (const copyId of [toolFlowId(f), hookFlowId(f)]) {
            if (!there.current.has(copyId)) continue;
            const copy = await call('GET', `/api/bridge/flows/${encodeURIComponent(copyId)}`);
            const doc = copy?.ok ? ((await copy.json().catch(() => null)) as Doc | null) : null;
            if (doc?.oaiyTool || doc?.oaiyToolHook) await call('PUT', `/api/bridge/flows/${encodeURIComponent(copyId)}`, { ...doc, name: f.name, nodes: withNodeServices(f.graph.nodes), edges: f.graph.edges });
          }
        }
        // Deleted here, by the person: removed there too (and as the agent's tool, or what goes in front of one).
        for (const id of [...deletedHere]) {
          // (one here again, and kept in step: not removed)
          if (flows.some((f) => f.id === id && shared(f))) {
            deletedHere.delete(id);
            continue;
          }
          const gone = await call('DELETE', `/api/bridge/flows/${encodeURIComponent(id)}`);
          if (!gone || !(gone.ok || gone.status === 404)) continue;
          synced.current.delete(id);
          deletedHere.delete(id);
          for (const copyId of [toolFlowId({ id } as Flow), hookFlowId({ id } as Flow)]) {
            if (there.current.has(copyId)) await call('DELETE', `/api/bridge/flows/${encodeURIComponent(copyId)}`);
          }
        }
        keep();
      } while (pushAgain.current);
    } finally {
      pushing.current = false;
    }
  };
  pushRef.current = push;

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
      if (forgetAsked) {
        forgetAsked = false;
        synced.current = new Map();
      }
      // (left as they are: flows deleted here, for the push to remove there, and flows here that are not the
      // desktop's, a local-only one of the same id, which a flow there must not come in over)
      const leave = new Set([...deletedHere, ...latest.current.flows.filter((f) => !shared(f)).map((f) => f.id)]);
      const { put, remove, synced: agreed } = reconcile(flows, latest.current.flows.filter(shared), synced.current, leave, there.current);
      synced.current = agreed;
      keep();
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
      // (what changed here meanwhile, or did not reach the desktop before, is sent now)
      void pushRef.current();
    };
    void pull();
    const timer = window.setInterval(() => void pull(), PULL_MS);
    // (the window closing or hidden: what changed in the last moment is sent now, not after the pause)
    const flush = () => void pushRef.current();
    const hidden = () => {
      if (document.visibilityState === 'hidden') flush();
    };
    window.addEventListener('pagehide', flush);
    document.addEventListener('visibilitychange', hidden);
    return () => {
      live = false;
      window.clearInterval(timer);
      window.removeEventListener('pagehide', flush);
      document.removeEventListener('visibilitychange', hidden);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A moment after the last edit, what changed is sent.
  useEffect(() => {
    if (!desktop) return;
    const timer = window.setTimeout(() => void pushRef.current(), SETTLE_MS);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.flows]);
}
