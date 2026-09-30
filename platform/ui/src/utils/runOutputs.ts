/**
 * The fields a node's DATA gets from a run (what an Output node shows, the picture an Image Viewer shows, a saved video's address).
 *
 * They are written into the live nodes by the job-queue run path and are never persisted into a saved flow (hooks/useWorkflow.ts leaves
 * them out of the comparison and of what is saved: a stale one revives on reload). A flow that arrives from outside carries whatever its
 * author wrote there, run or not, and opening it makes the nodes show it, which for an address is a request to it. So a flow that comes
 * in arrives without them, by each way one does in the editor: a project file (ProjectIO.parseImportedProject), a flow file added with
 * Import flows (usePackageManager.normalizeFlowData), a workflow file (WorkflowIO.parseWorkflowJson) and a shared link (?flow=, App.tsx).
 * A package (.oaiy) is not among them: loading one needs the desktop's own commands, which the editor's browser shim does not have in a
 * tab or in OAIY's window (usePackageManager says so). The nodes fill them again when the flow runs.
 */
export const RUN_OUTPUT_FIELDS: ReadonlySet<string> = new Set(['outputValue', 'imageUrl', 'videoUrl']);

interface NodeLike {
  data?: Record<string, unknown>;
}

/** The nodes with the run outputs left out of their data. What has none is returned as it is. */
export function withoutRunOutputs<T extends NodeLike>(nodes: readonly T[]): T[] {
  return nodes.map((node) => {
    const data = node && typeof node === 'object' ? node.data : undefined;
    if (!data || typeof data !== 'object' || !Object.keys(data).some((key) => RUN_OUTPUT_FIELDS.has(key))) return node;
    return { ...node, data: Object.fromEntries(Object.entries(data).filter(([key]) => !RUN_OUTPUT_FIELDS.has(key))) };
  });
}

/** A flow (`{ graph: { nodes, edges } }`) with the run outputs left out of its nodes. Anything that is not one is returned as it is. */
export function flowWithoutRunOutputs<T>(flow: T): T {
  const graph = flow && typeof flow === 'object' ? (flow as { graph?: { nodes?: unknown } }).graph : undefined;
  if (!graph || typeof graph !== 'object' || !Array.isArray(graph.nodes)) return flow;
  return { ...flow, graph: { ...graph, nodes: withoutRunOutputs(graph.nodes as NodeLike[]) } };
}
