/**
 * Flows brought in from outside (a shared link, an imported file) are given ids of their own here, so they never
 * replace a flow of the same id; the references between them (a Subflow node's `flowId`, a macro node's
 * `_macroWorkflowId`) are rewritten to the new ids, so a flow that calls another still calls the one it came with.
 */

type Node = { data?: Record<string, unknown> } & Record<string, unknown>;
type WithGraph = { id?: string; graph?: { nodes?: unknown[]; edges?: unknown[] } };

/** The keys of a node's data that name another flow. */
const FLOW_REFS = ['flowId', '_macroWorkflowId'] as const;

/** `flows` with the ids `ids` maps (old → new) rewritten in their nodes' references; each flow's own id is left to the caller. */
export function rewriteFlowRefs<T extends WithGraph>(flows: T[], ids: ReadonlyMap<string, string>): T[] {
  return flows.map((f) => {
    const nodes = f.graph?.nodes;
    if (!Array.isArray(nodes)) return f;
    let changed = false;
    const next = nodes.map((n) => {
      const node = n as Node;
      const data = node?.data;
      if (!data || typeof data !== 'object') return n;
      let out: Record<string, unknown> | null = null;
      for (const key of FLOW_REFS) {
        const ref = data[key];
        if (typeof ref === 'string' && ids.has(ref)) {
          out = out ?? { ...data };
          out[key] = ids.get(ref);
        }
      }
      if (!out) return n;
      changed = true;
      return { ...node, data: out };
    });
    return changed ? { ...f, graph: { ...f.graph, nodes: next } } : f;
  });
}

/** `flows` under new ids (`newId()` for each), their references to one another rewritten to match. */
export function withNewIds<T extends WithGraph & { id: string }>(flows: T[], newId: () => string): { flows: T[]; ids: Map<string, string> } {
  const ids = new Map<string, string>();
  for (const f of flows) if (f.id && !ids.has(f.id)) ids.set(f.id, newId());
  const renamed = flows.map((f) => ({ ...f, id: ids.get(f.id) ?? newId() }));
  return { flows: rewriteFlowRefs(renamed, ids), ids };
}

/**
 * Whether the flow `flowId` (its `graph`), or one it calls (a Subflow or macro, at any depth, among `flows`), is a
 * shared flow marked untrusted where it was brought in.
 */
export function reachesUntrusted(flowId: string | undefined, graph: { nodes?: unknown[] }, flows: ReadonlyArray<WithGraph & { id: string }>): boolean {
  const marked = new Set(flows.filter((f) => (f as { untrusted?: unknown }).untrusted === true).map((f) => f.id));
  if (!marked.size) return false;
  if (flowId && marked.has(flowId)) return true;
  const byId = new Map(flows.map((f) => [f.id, f]));
  const refs = (nodes: unknown[] | undefined): string[] =>
    (nodes ?? []).flatMap((n) => {
      const data = (n as Node)?.data ?? {};
      return FLOW_REFS.map((k) => data[k]).filter((r): r is string => typeof r === 'string' && r.length > 0);
    });
  const seen = new Set<string>();
  const queue = refs(graph.nodes);
  while (queue.length) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    if (marked.has(id)) return true;
    queue.push(...refs(byId.get(id)?.graph?.nodes));
  }
  return false;
}
