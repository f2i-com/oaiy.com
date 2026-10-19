/**
 * A run's results on the canvas: what goes as a run starts, and what a run that did not finish leaves.
 *
 * A run that failed showed the run before's result, as though it were its own, and said nothing of which step failed
 * (hooks/useWorkflowExecution.ts calls these).
 */
import type { Node } from '@xyflow/react';
import { RUN_OUTPUT_FIELDS } from '../utils/runOutputs';

/** A node's state in a run, as its card shows it (`stopped`: where Stop found it, not a failure). */
export type NodeRunStatus = 'running' | 'completed' | 'error' | 'stopped';

/** A node as a run starts: what the last run gave it (its results, its run state, `_stale`) left out. */
export function beforeRun(node: Node): Node {
  const data = node.data as Record<string, unknown>;
  if (!Object.keys(data).some((key) => RUN_OUTPUT_FIELDS.has(key) || key === '_status' || key === '_stale')) return node;
  const kept = Object.fromEntries(Object.entries(data).filter(([key]) => !RUN_OUTPUT_FIELDS.has(key) && key !== '_status' && key !== '_stale'));
  return { ...node, data: kept };
}

/**
 * A node after a run that did not finish (failed, or `stopped`): one still running is stopped there (or simply not
 * running, when the run failed elsewhere), one that failed keeps its mark (`failedNodeId`: the one the run said failed,
 * marked even where the canvas did not hear it as it happened), and a result node says it has no result from this run
 * (`_stale`).
 */
export function afterUnfinishedRun(node: Node, stopped: boolean, failedNodeId?: string): Node {
  const data = node.data as Record<string, unknown>;
  const updates: Record<string, unknown> = {};
  if (node.id === failedNodeId) updates._status = 'error';
  else if (data._status === 'running') updates._status = stopped ? 'stopped' : undefined;
  if (RESULT_NODES.has(String(node.type))) updates._stale = true;
  return Object.keys(updates).length ? { ...node, data: { ...data, ...updates } } : node;
}

/** The nodes that show a run's result. */
const RESULT_NODES: ReadonlySet<string> = new Set(['output', 'image_view', 'image_save', 'video_save']);
