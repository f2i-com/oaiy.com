/**
 * The canvas's undo and redo (lib/undoHistory.ts), kept as the canvas changes: a change is a step once it has
 * settled (not each frame of a drag, nor each key typed), and an undo takes in one not settled yet.
 */

import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import type { Edge, Node } from '@xyflow/react';
import { UndoHistory, restoreOnto, type CanvasState } from '../lib/undoHistory';

/** How long the canvas is still before a change is a step (typing in a node is one step, not one a letter). */
const SETTLE_MS = 400;

interface Options {
  nodes: Node[];
  edges: Edge[];
  setNodes: Dispatch<SetStateAction<Node[]>>;
  setEdges: Dispatch<SetStateAction<Edge[]>>;
  /** Told when a step removed nodes (not only boxes): how many, for an "Undo" offer. */
  onRemoved?: (count: number) => void;
}

export function useUndoHistory({ nodes, edges, setNodes, setEdges, onRemoved }: Options) {
  const history = useRef(new UndoHistory());
  const latest = useRef<CanvasState>({ nodes, edges });
  latest.current = { nodes, edges };
  const removedRef = useRef(onRemoved);
  removedRef.current = onRemoved;
  const [, redraw] = useState(0);

  const settle = useCallback((state: CanvasState) => {
    const before = history.current.record(state);
    if (!before) return;
    redraw((n) => n + 1);
    const here = new Set(state.nodes.map((n) => n.id));
    const gone = before.nodes.filter((n) => !here.has(n.id) && n.type !== 'group').length;
    if (gone > 0) removedRef.current?.(gone);
  }, []);

  useEffect(() => {
    // (not while a node is being dragged: its drop is the step)
    if (nodes.some((n) => n.dragging)) return;
    const timer = setTimeout(() => settle({ nodes, edges }), SETTLE_MS);
    return () => clearTimeout(timer);
  }, [nodes, edges, settle]);

  const apply = useCallback(
    (state: CanvasState | null) => {
      if (!state) return false;
      const next = restoreOnto(state, latest.current);
      setNodes(next.nodes);
      setEdges(next.edges);
      redraw((n) => n + 1);
      return true;
    },
    [setNodes, setEdges],
  );

  const undo = useCallback(() => apply(history.current.undo(latest.current)), [apply]);
  const redo = useCallback(() => apply(history.current.redo(latest.current)), [apply]);

  return { undo, redo, canUndo: history.current.canUndo, canRedo: history.current.canRedo };
}
