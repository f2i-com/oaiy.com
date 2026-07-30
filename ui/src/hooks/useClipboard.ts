/**
 * Clipboard Hook
 *
 * Manages copy/paste operations for workflow nodes and edges.
 * Uses localStorage for persistence across sessions.
 */

import { useCallback, useRef, type CSSProperties } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { Node, Edge } from '@xyflow/react';

const CLIPBOARD_KEY = 'oaiy_workflow_clipboard';

/** Persistent underscore fields that should be preserved during copy */
const PERSISTENT_UNDERSCORE_FIELDS = [
  '_macroWorkflowId',
  '_macroName',
  '_macroInputs',
  '_macroOutputs',
  '_collapsed',
];

/** Clipboard data structure */
export interface ClipboardData {
  nodes: Array<{
    id: string;
    type: string;
    position: { x: number; y: number };
    /** Grouping: only kept when the parent is also in the copy set (else dropped + position absolutized). */
    parentId?: string;
    extent?: 'parent';
    /** Group geometry lives in node.style.width/height; preserve it so groups paste at size. */
    style?: CSSProperties;
    width?: number;
    height?: number;
    data: Record<string, unknown>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    sourceHandle?: string;
    targetHandle?: string;
  }>;
  copiedAt: number;
}

export interface UseClipboardOptions {
  /** Current workflow nodes */
  nodes: Node[];
  /** Current workflow edges */
  edges: Edge[];
  /** Setter for nodes */
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  /** Setter for edges */
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  /** Optional log callback */
  addLog?: (entry: { source: string; message: string; type?: 'info' | 'error' | 'success' | 'node' }) => void;
}

export interface UseClipboardResult {
  /** Copy selected nodes and their connecting edges to clipboard */
  copySelected: () => boolean;
  /** Check if clipboard has valid data */
  hasClipboard: () => boolean;
  /** Get clipboard data */
  getClipboardData: () => ClipboardData | null;
  /** Paste nodes from clipboard at a given position */
  pasteClipboard: (pastePosition?: { x: number; y: number }) => boolean;
}

/**
 * Hook for managing workflow clipboard operations.
 */
export function useClipboard({
  nodes,
  edges,
  setNodes,
  setEdges,
  addLog,
}: UseClipboardOptions): UseClipboardResult {
  // Cascades successive keyboard pastes diagonally so repeated Ctrl+V doesn't stack
  // every copy at the identical spot (mirrors addNode's offset cascade).
  const pasteCounterRef = useRef(0);

  /**
   * Copy selected nodes and their connecting edges to clipboard.
   */
  const copySelected = useCallback((): boolean => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) {
      return false;
    }

    const selectedNodeIds = new Set(selectedNodes.map((n) => n.id));

    // Get edges that connect selected nodes (both ends must be selected)
    const selectedEdges = edges.filter(
      (e) => selectedNodeIds.has(e.source) && selectedNodeIds.has(e.target)
    );

    // Strip temporary status fields from node data, but keep persistent ones, and
    // capture grouping state (parentId/extent/style/width/height) so groups + their
    // children survive a copy/paste.
    const cleanedNodes = selectedNodes.map((n) => {
      const keepParent = !!(n.parentId && selectedNodeIds.has(n.parentId));
      let position = n.position;
      let parentId = n.parentId;
      let extent: 'parent' | undefined = n.extent === 'parent' ? 'parent' : undefined;
      if (n.parentId && !keepParent) {
        // Parent isn't in the copy set: the child's position is RELATIVE to it, so
        // convert to absolute and drop the parent link — otherwise paste would
        // reinterpret the small relative coords as absolute and detach the child.
        const parent = nodes.find((p) => p.id === n.parentId);
        if (parent) position = { x: n.position.x + parent.position.x, y: n.position.y + parent.position.y };
        parentId = undefined;
        extent = undefined;
      }
      return {
        id: n.id,
        type: n.type || 'unknown',
        position,
        ...(parentId ? { parentId } : {}),
        ...(extent ? { extent } : {}),
        ...(n.style ? { style: n.style as CSSProperties } : {}),
        ...(typeof n.width === 'number' ? { width: n.width } : {}),
        ...(typeof n.height === 'number' ? { height: n.height } : {}),
        data: Object.fromEntries(
          Object.entries(n.data as Record<string, unknown>).filter(([key]) => {
            // Keep persistent underscore fields
            if (PERSISTENT_UNDERSCORE_FIELDS.includes(key)) return true;
            // Strip other underscore-prefixed fields (like _status)
            if (key.startsWith('_')) return false;
            return true;
          })
        ),
      };
    });

    const clipboardData: ClipboardData = {
      nodes: cleanedNodes,
      edges: selectedEdges.map((e) => ({
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle || undefined,
        targetHandle: e.targetHandle || undefined,
      })),
      copiedAt: Date.now(),
    };

    try {
      localStorage.setItem(CLIPBOARD_KEY, JSON.stringify(clipboardData));
      addLog?.({
        source: 'System',
        message: `Copied ${selectedNodes.length} node${selectedNodes.length > 1 ? 's' : ''} and ${selectedEdges.length} connection${selectedEdges.length !== 1 ? 's' : ''}`,
        type: 'info',
      });
      return true;
    } catch {
      return false;
    }
  }, [nodes, edges, addLog]);

  /**
   * Check if clipboard has valid data.
   */
  const hasClipboard = useCallback((): boolean => {
    try {
      const data = localStorage.getItem(CLIPBOARD_KEY);
      if (!data) return false;
      const parsed = JSON.parse(data) as ClipboardData;
      return parsed.nodes && parsed.nodes.length > 0;
    } catch {
      return false;
    }
  }, []);

  /**
   * Get clipboard data (for checking before paste).
   */
  const getClipboardData = useCallback((): ClipboardData | null => {
    try {
      const data = localStorage.getItem(CLIPBOARD_KEY);
      if (!data) return null;
      return JSON.parse(data) as ClipboardData;
    } catch {
      return null;
    }
  }, []);

  /**
   * Paste nodes from clipboard at a given position.
   */
  const pasteClipboard = useCallback(
    (pastePosition?: { x: number; y: number }): boolean => {
      try {
        const data = localStorage.getItem(CLIPBOARD_KEY);
        if (!data) return false;

        const clipboardData = JSON.parse(data) as ClipboardData;
        if (!clipboardData.nodes || clipboardData.nodes.length === 0) return false;

        // Anchor bounds on TOP-LEVEL nodes only — a child's position is relative to
        // its parent, so including children would skew the offset anchor.
        const topLevel = clipboardData.nodes.filter((n) => !n.parentId);
        const anchorNodes = topLevel.length > 0 ? topLevel : clipboardData.nodes;
        const minX = Math.min(...anchorNodes.map((n) => n.position.x));
        const minY = Math.min(...anchorNodes.map((n) => n.position.y));

        // Explicit position (drag/drop) anchors there; keyboard paste cascades
        // diagonally (wrapping every 8) so repeated Ctrl+V doesn't stack copies.
        const step = pastePosition ? 0 : pasteCounterRef.current++ % 8;
        const offsetX = pastePosition ? pastePosition.x - minX : 50 + step * 30;
        const offsetY = pastePosition ? pastePosition.y - minY : 50 + step * 30;

        // Create ID mapping from old to new IDs
        const idMapping = new Map<string, string>();
        clipboardData.nodes.forEach((n) => {
          idMapping.set(n.id, uuidv4());
        });

        // Create new nodes with new IDs, remapped parents, and offset positions
        const newNodes: Node[] = clipboardData.nodes.map((n) => {
          const mappedParent = n.parentId ? idMapping.get(n.parentId) : undefined;
          const isChild = !!mappedParent;
          return {
            id: idMapping.get(n.id)!,
            type: n.type,
            // A child's position is relative to its re-parented parent (which
            // already got the offset) — don't offset it again or it drifts out.
            position: isChild
              ? { ...n.position }
              : { x: n.position.x + offsetX, y: n.position.y + offsetY },
            ...(mappedParent ? { parentId: mappedParent, extent: n.extent } : {}),
            ...(n.style ? { style: n.style } : {}),
            ...(typeof n.width === 'number' ? { width: n.width } : {}),
            ...(typeof n.height === 'number' ? { height: n.height } : {}),
            data: { ...n.data },
            selected: true, // Select newly pasted nodes
          };
        });

        // Create new edges with updated source/target IDs
        const newEdges: Edge[] = clipboardData.edges
          .filter((e) => idMapping.has(e.source) && idMapping.has(e.target))
          .map((e) => ({
            id: `e-${idMapping.get(e.source)}-${idMapping.get(e.target)}-${uuidv4().slice(0, 8)}`,
            source: idMapping.get(e.source)!,
            target: idMapping.get(e.target)!,
            sourceHandle: e.sourceHandle,
            targetHandle: e.targetHandle,
            selected: true, // Select newly pasted edges
          }));

        // Deselect existing nodes and edges, then add new ones
        setNodes((nds) => [...nds.map((n) => ({ ...n, selected: false })), ...newNodes]);
        setEdges((eds) => [...eds.map((e) => ({ ...e, selected: false })), ...newEdges]);

        addLog?.({
          source: 'System',
          message: `Pasted ${newNodes.length} node${newNodes.length > 1 ? 's' : ''} and ${newEdges.length} connection${newEdges.length !== 1 ? 's' : ''}`,
          type: 'info',
        });

        return true;
      } catch {
        return false;
      }
    },
    [setNodes, setEdges, addLog]
  );

  return {
    copySelected,
    hasClipboard,
    getClipboardData,
    pasteClipboard,
  };
}
