import { useCallback, useRef, useMemo, useState, useEffect } from 'react';
import type { Node } from '@xyflow/react';
import type { OnConnectStartParams } from '@xyflow/react';
import type { NodeType } from 'oaiy-core';
import type { ModuleNodeInfo } from './useModuleNodes';

/**
 * Quick connect state
 */
export interface QuickConnectState {
  isVisible: boolean;
  position: { x: number; y: number };
  sourceNodeId: string;
  sourceHandleId: string | null;
  sourceHandleType: 'source' | 'target';
  handleDataType: string;
  searchQuery: string;
  sourceHandlePosition: { x: number; y: number } | null;
}

/**
 * Options for useQuickConnect hook
 */
export interface UseQuickConnectOptions {
  /** Current workflow nodes */
  nodes: Node[];
  /** Available module node definitions */
  moduleNodes: ModuleNodeInfo[];
  /** Function to add a new node to the graph */
  addNode: (nodeType: NodeType, position: { x: number; y: number }) => string;
  /** Function to create an edge connection */
  onConnect: (connection: {
    source: string;
    sourceHandle: string | null;
    target: string;
    targetHandle: string | null;
  }) => void;
  /** React Flow instance ref for coordinate conversion */
  reactFlowInstance: React.MutableRefObject<{
    screenToFlowPosition: (position: { x: number; y: number }) => { x: number; y: number };
  } | null>;
  /** Delay in ms before showing quick connect popup (default: 2000) */
  showDelay?: number;
}

/**
 * Hook for managing quick connect functionality in the workflow builder.
 *
 * Quick connect allows users to drag a connection from a node handle and,
 * after holding for a short time, see a popup with compatible nodes to
 * quickly create and connect.
 */
export function useQuickConnect({
  nodes,
  moduleNodes,
  addNode,
  onConnect,
  reactFlowInstance,
  showDelay = 2000,
}: UseQuickConnectOptions) {
  // Quick-connect state for connection drag
  const [quickConnectState, setQuickConnectState] = useState<QuickConnectState | null>(null);
  const quickConnectTimerRef = useRef<NodeJS.Timeout | null>(null);
  const connectionDragPositionRef = useRef<{ x: number; y: number } | null>(null);
  const quickConnectVisibleRef = useRef(false);
  const [viewportChangeCounter, setViewportChangeCounter] = useState(0);

  // Track mouse position during connection drag
  const handleConnectionDragMove = useCallback((event: MouseEvent) => {
    connectionDragPositionRef.current = { x: event.clientX, y: event.clientY };
  }, []);

  // Handle connection drag start - start timer for popup
  const handleConnectStart = useCallback((_event: MouseEvent | TouchEvent, params: OnConnectStartParams) => {
    // Clear any existing timer and reset visibility
    if (quickConnectTimerRef.current) {
      clearTimeout(quickConnectTimerRef.current);
    }
    quickConnectVisibleRef.current = false;
    setQuickConnectState(null);

    // Track mouse position for popup placement
    document.addEventListener('mousemove', handleConnectionDragMove);

    // Get the source node and handle info
    const sourceNode = nodes.find(n => n.id === params.nodeId);
    if (!sourceNode) return;

    // Determine the handle data type from the node definition
    let handleDataType = 'any';
    const nodeType = sourceNode.type;
    const nodeDef = moduleNodes.find(n => n.definition.id === nodeType);
    if (nodeDef) {
      const handles = params.handleType === 'source' ? nodeDef.definition.outputs : nodeDef.definition.inputs;
      const handle = handles.find(h => h.id === params.handleId);
      if (handle) {
        handleDataType = handle.type;
      }
    }

    // Start timer to show quick-connect popup
    quickConnectTimerRef.current = setTimeout(() => {
      const pos = connectionDragPositionRef.current || { x: window.innerWidth / 2, y: window.innerHeight / 2 };

      // Get the source handle element position for drawing connection line
      let sourceHandlePosition: { x: number; y: number } | null = null;
      const handleSelector = `[data-nodeid="${params.nodeId}"][data-handleid="${params.handleId}"]`;
      const handleElement = document.querySelector(handleSelector);
      if (handleElement) {
        const rect = handleElement.getBoundingClientRect();
        sourceHandlePosition = {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        };
      }

      // Mark as visible in ref (for closure in handleConnectEnd)
      quickConnectVisibleRef.current = true;

      setQuickConnectState({
        isVisible: true,
        position: pos,
        sourceNodeId: params.nodeId || '',
        sourceHandleId: params.handleId || null,
        sourceHandleType: params.handleType || 'source',
        handleDataType,
        searchQuery: '',
        sourceHandlePosition,
      });
    }, showDelay);
  }, [nodes, moduleNodes, handleConnectionDragMove, showDelay]);

  // Handle connection drag end - clear timer but keep popup if visible
  const handleConnectEnd = useCallback(() => {
    // Clear the timer
    if (quickConnectTimerRef.current) {
      clearTimeout(quickConnectTimerRef.current);
      quickConnectTimerRef.current = null;
    }

    // Stop tracking mouse position
    document.removeEventListener('mousemove', handleConnectionDragMove);

    // If popup is NOT visible, reset state
    // If popup IS visible, keep it open so user can select a node
    if (!quickConnectVisibleRef.current) {
      setQuickConnectState(null);
    }
  }, [handleConnectionDragMove]);

  // Clean up pending timer + drag listener on unmount (e.g. keyed remount / flow switch mid-drag)
  useEffect(() => () => {
    if (quickConnectTimerRef.current) {
      clearTimeout(quickConnectTimerRef.current);
    }
    document.removeEventListener('mousemove', handleConnectionDragMove);
  }, [handleConnectionDragMove]);

  // Close popup
  const handleQuickConnectClose = useCallback(() => {
    quickConnectVisibleRef.current = false;
    setQuickConnectState(null);
  }, []);

  // Listen for clicks outside the popup to close it
  useEffect(() => {
    if (!quickConnectState?.isVisible) return;

    const handleDocumentClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement;

      // Don't close if clicking on the popup itself
      if (target.closest('[data-quick-connect-popup]')) return;

      // Don't close if clicking on a handle (allow manual connections)
      if (target.closest('.react-flow__handle')) return;

      // Don't close if clicking on a node (might be interacting with it)
      if (target.closest('.react-flow__node')) return;

      // Close the popup for clicks elsewhere
      handleQuickConnectClose();
    };

    // Use capture phase to catch clicks before they're handled
    document.addEventListener('mousedown', handleDocumentClick, true);

    return () => {
      document.removeEventListener('mousedown', handleDocumentClick, true);
    };
  }, [quickConnectState?.isVisible, handleQuickConnectClose]);

  // Filter compatible nodes based on handle type
  const quickConnectCompatibleNodes = useMemo(() => {
    if (!quickConnectState) return [];

    const { sourceHandleType, handleDataType } = quickConnectState;

    // Filter nodes that have compatible handles
    return moduleNodes.filter(nodeInfo => {
      // If dragging from output (source), we need nodes with inputs
      // If dragging from input (target), we need nodes with outputs
      const targetHandles = sourceHandleType === 'source'
        ? nodeInfo.definition.inputs
        : nodeInfo.definition.outputs;

      // Check if any handle is compatible
      return targetHandles.some(handle => {
        // 'any' type is compatible with everything
        if (handleDataType === 'any' || handle.type === 'any') return true;
        // Exact type match
        return handle.type === handleDataType;
      });
    });
  }, [quickConnectState, moduleNodes]);

  // Get current handle position (recalculated on each render to stay in sync with viewport)
  const quickConnectHandlePosition = useMemo(() => {
    if (!quickConnectState?.isVisible) return null;

    const handleSelector = `[data-nodeid="${quickConnectState.sourceNodeId}"][data-handleid="${quickConnectState.sourceHandleId}"]`;
    const handleElement = document.querySelector(handleSelector);
    if (handleElement) {
      const rect = handleElement.getBoundingClientRect();
      return {
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
      };
    }
    // Fallback to stored position if element not found
    return quickConnectState.sourceHandlePosition;
    // Intentionally omit reactFlowWrapper from deps - it's a ref that shouldn't trigger recalculation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quickConnectState?.isVisible, quickConnectState?.sourceNodeId, quickConnectState?.sourceHandleId, quickConnectState?.sourceHandlePosition, nodes, viewportChangeCounter]);

  // Handle node selection from popup
  const handleQuickConnectNodeSelect = useCallback((nodeType: NodeType) => {
    if (!quickConnectState || !reactFlowInstance.current) return;

    const { sourceNodeId, sourceHandleId, sourceHandleType, handleDataType } = quickConnectState;

    // Get position near the cursor
    const pos = connectionDragPositionRef.current || { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    const flowPosition = reactFlowInstance.current.screenToFlowPosition(pos);

    // Get node definition to find the right handle to connect to
    const nodeDef = moduleNodes.find(n => n.definition.id === nodeType);
    if (!nodeDef) return;

    // Find compatible handle on the new node
    const targetHandles = sourceHandleType === 'source'
      ? nodeDef.definition.inputs
      : nodeDef.definition.outputs;

    const compatibleHandle = targetHandles.find(handle => {
      if (handleDataType === 'any' || handle.type === 'any') return true;
      return handle.type === handleDataType;
    });

    if (!compatibleHandle) return;

    // Add the new node and get its ID
    const newNodeId = addNode(nodeType, flowPosition);

    // Create the edge connection after a tick to ensure node exists
    setTimeout(() => {
      if (sourceHandleType === 'source') {
        // Dragging from an output - connect to new node's input
        onConnect({
          source: sourceNodeId,
          sourceHandle: sourceHandleId,
          target: newNodeId,
          targetHandle: compatibleHandle.id,
        });
      } else {
        // Dragging from an input - connect new node's output to it
        onConnect({
          source: newNodeId,
          sourceHandle: compatibleHandle.id,
          target: sourceNodeId,
          targetHandle: sourceHandleId,
        });
      }
    }, 50);

    // Close the popup
    quickConnectVisibleRef.current = false;
    setQuickConnectState(null);
  }, [quickConnectState, moduleNodes, addNode, onConnect, reactFlowInstance]);

  // Trigger viewport change counter update (call this on viewport changes)
  const onViewportChange = useCallback(() => {
    setViewportChangeCounter(c => c + 1);
  }, []);

  // Update search query in quick connect popup
  const setSearchQuery = useCallback((query: string) => {
    setQuickConnectState(prev => prev ? { ...prev, searchQuery: query } : null);
  }, []);

  return {
    // State
    quickConnectState,
    quickConnectCompatibleNodes,
    quickConnectHandlePosition,

    // Handlers
    handleConnectStart,
    handleConnectEnd,
    handleQuickConnectClose,
    handleQuickConnectNodeSelect,
    onViewportChange,
    setSearchQuery,
  };
}
