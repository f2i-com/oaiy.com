/**
 * Undo and redo on the canvas: what it was before each change (a node added, deleted, moved or edited, a wire
 * made or removed), kept as the canvas's own state, gone back over with ⌘Z and forward with ⇧⌘Z.
 */

import type { Edge, Node } from '@xyflow/react';
import { RUN_OUTPUT_FIELDS } from '../utils/runOutputs';

/** The underscore fields a flow keeps (a macro's link, a folded node): the rest are a run's or the canvas's own. */
const KEPT_UNDERSCORE = new Set(['_macroWorkflowId', '_macroName', '_macroInputs', '_macroOutputs', '_collapsed']);

/**
 * What of the canvas an undo goes back over, as a string to compare: the nodes (where, what, in which group, what is
 * typed into them) and the wires. Not the selection, a run's outputs or statuses, nor what is being measured. A
 * long value (a picture dropped on a node) is compared by its length and start.
 */
export function canvasKey(nodes: readonly Node[], edges: readonly Edge[]): string {
  return JSON.stringify(
    {
      n: nodes.map((n) => [
        n.id,
        n.type,
        Math.round(n.position.x),
        Math.round(n.position.y),
        n.parentId ?? null,
        n.style?.width ?? null,
        n.style?.height ?? null,
        Object.entries(n.data ?? {}).filter(([k]) => !RUN_OUTPUT_FIELDS.has(k) && (!k.startsWith('_') || KEPT_UNDERSCORE.has(k))),
      ]),
      e: edges.map((e) => [e.id, e.source, e.sourceHandle ?? null, e.target, e.targetHandle ?? null]),
    },
    (_k, v) => (typeof v === 'function' ? undefined : typeof v === 'string' && v.length > 512 ? `${v.length}:${v.slice(0, 64)}` : v),
  );
}

export interface CanvasState {
  nodes: Node[];
  edges: Edge[];
}

interface Entry extends CanvasState {
  key: string;
}

/** The canvas's states, past and undone, around the one it is in now. */
export class UndoHistory {
  private past: Entry[] = [];
  private future: Entry[] = [];
  private present: Entry | null = null;

  constructor(private readonly limit = 100) {}

  /**
   * The canvas as it is now. A change from the last state recorded is a new step (and what was undone is gone);
   * the same canvas (a selection, a run's outputs) is no step. Returns the state before the step, or null.
   */
  record(state: CanvasState): CanvasState | null {
    const key = canvasKey(state.nodes, state.edges);
    const entry = { ...state, key };
    if (!this.present || this.present.key === key) {
      this.present = entry;
      return null;
    }
    const before = this.present;
    this.past.push(before);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
    this.present = entry;
    return before;
  }

  /** Back one step from `current` (recorded first, so a change not yet settled is the one undone); null at the start. */
  undo(current: CanvasState): CanvasState | null {
    this.record(current);
    const back = this.past.pop();
    if (!back || !this.present) return null;
    this.future.push(this.present);
    this.present = back;
    return back;
  }

  /** Forward one undone step; null when there is none (or the canvas was changed since the undo). */
  redo(current: CanvasState): CanvasState | null {
    this.record(current);
    const next = this.future.pop();
    if (!next || !this.present) return null;
    this.past.push(this.present);
    this.present = next;
    return next;
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }
}

/**
 * `state` as the canvas takes it back: each node that is still there keeps what a run put on it (its outputs,
 * its status) and whether it is selected, so an undo changes what was edited and nothing else.
 */
export function restoreOnto(state: CanvasState, current: CanvasState): CanvasState {
  const now = new Map(current.nodes.map((n) => [n.id, n]));
  const nodes = state.nodes.map((n) => {
    const live = now.get(n.id);
    if (!live) return { ...n, selected: false };
    const carried = Object.fromEntries(
      Object.entries(live.data ?? {}).filter(([k]) => RUN_OUTPUT_FIELDS.has(k) || (k.startsWith('_') && !KEPT_UNDERSCORE.has(k))),
    );
    const data = { ...Object.fromEntries(Object.entries(n.data ?? {}).filter(([k]) => !RUN_OUTPUT_FIELDS.has(k))), ...carried };
    return { ...n, data, selected: live.selected ?? false };
  });
  const selectedEdges = new Set(current.edges.filter((e) => e.selected).map((e) => e.id));
  return { nodes, edges: state.edges.map((e) => ({ ...e, selected: selectedEdges.has(e.id) })) };
}
