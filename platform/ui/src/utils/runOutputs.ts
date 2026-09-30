/**
 * The fields a node's DATA gets from a run (what an Output node shows, the picture an Image Viewer shows, a saved video's address).
 *
 * They are written into the live nodes by the job-queue run path and are never persisted into a saved flow (hooks/useWorkflow.ts leaves
 * them out of the comparison and of what is saved: a stale one revives on reload). A flow that arrives from outside carries whatever its
 * author wrote there, run or not, and opening it makes the nodes show it, which for an address is a request to it: so a flow that
 * is imported, opened from a shared link or read from a package arrives without them. The nodes fill them again when the flow runs.
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
