import { useCallback, useRef, useMemo, useState, useEffect, useLayoutEffect } from 'react';
import { Layers, Map as MapIcon, Package as PackageIcon, Play, Plus, RefreshCw, RotateCcw, Save, ScrollText, SlidersHorizontal, Square, X } from 'lucide-react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ConnectionMode,
  type Node,
  type Edge,
  type Viewport,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { createLogger } from '../utils/logger';

const logger = createLogger('OAIYBuilder');

import { getNodeTypes, getRegistryVersion } from 'oaiy-ui-components';
import { edgeTypes } from './edges';
import NodePalette, { type MacroNodeData } from './panels/NodePalette';
import LogConsole from './panels/LogConsole';
import PropertiesPanel from './panels/PropertiesPanel';
import { SWATCH_CLASS } from './panels/nodeSwatches';
import { useWorkflow } from '../hooks/useWorkflow';
import { useStableHandlers } from '../hooks/useStableHandlers';
import { useQuickConnect } from '../hooks/useQuickConnect';
import { useWorkflowExecution } from '../hooks/useWorkflowExecution';
import { useModuleNodes } from '../hooks/useModuleNodes';
import { useEdgeScrolling } from '../hooks/useEdgeScrolling';
import { useNodeGrouping } from '../hooks/useNodeGrouping';
import { QuickConnectPopup } from './OAIYBuilder/QuickConnectPopup';
import type { PackageNodeInfo } from '../hooks/usePackageNodes';
import type { NodeType, WorkflowGraph, Flow, LLMEndpoint, ProjectConstant, ProjectSettings, ComfyUIAnalysis, OAIYPackageManifest } from 'oaiy-core';
import RunWorkflowModal, { hasInputNodes } from './ui/RunWorkflowModal';
import ComfyUIWorkflowDialog, { type ComfyUIWorkflowConfig } from 'oaiy-core/modules/core-image/ui/ComfyUIWorkflowDialog';
import { ServiceStartupDialog } from './PackageManager/ServiceStartupDialog';
import { CanvasContextMenu } from './OAIYBuilder/CanvasContextMenu';
import { makeIsValidConnection } from '../utils/edgeTypeValidation';
import TypedConnectionLine from './OAIYBuilder/TypedConnectionLine';

// Render-invariant literals hoisted to module scope so they keep a stable
// identity across renders (none close over render state). Recreating these
// inline forced ReactFlow + Background + MiniMap to treat every render as a
// prop change.
const CANVAS_STYLE = { backgroundColor: 'rgb(var(--bg-primary))' } as const;
const DEFAULT_EDGE_OPTIONS = {
  type: 'selectable',
  deletable: true,
  selectable: true,
  focusable: true,
} as const;
const CONNECTION_LINE_STYLE = { stroke: 'rgb(var(--accent-primary))', strokeWidth: 3 } as const;
const PRO_OPTIONS = { hideAttribution: true } as const;
// A small map, so it leaves the toolbar its room at the canvas's foot.
const MINIMAP_STYLE = { width: 168, height: 112 } as const;
const BACKGROUND_STYLE = { backgroundColor: 'rgb(var(--bg-primary))', color: 'rgb(var(--bg-tertiary))' } as const;

// Shared (non-per-node) dependencies that get spread into every node's data
// object. Used as the per-id data cache's invalidation key — if any of these
// change identity, every cached node data object must be rebuilt.
interface NodeDataSharedKey {
  handlers: ReturnType<typeof useStableHandlers>;
  availableFlows: Flow[];
  llmEndpoints: LLMEndpoint[];
  projectConstants: ProjectConstant[];
  projectSettings: ProjectSettings | undefined;
  onShowToast: ((message: string, type?: 'success' | 'error' | 'info' | 'warning') => void) | undefined;
  handleOpenComfyWorkflowDialog: (nodeId: string, analysis: ComfyUIAnalysis, fileName: string) => void;
}

const miniMapNodeColor = (node: Node): string => {
  switch (node.type) {
    case 'input_text': return '#22c55e';
    case 'input_file': return '#84cc16';
    case 'ai_llm': return '#a855f7';
    case 'logic_block': return '#3b82f6';
    case 'browser_request': return '#06b6d4';
    case 'memory': return '#06b6d4';
    case 'template': return '#f59e0b';
    case 'image_gen': return '#ec4899';
    case 'image_view': return '#6366f1';
    case 'image_save': return '#14b8a6';
    case 'image_combiner': return '#ec4899';
    case 'output': return '#10b981';
    default: return '#64748b';
  }
};

/**
 * The builder's own width (the canvas and its side panels, without the flows
 * rail or OAIY's sidebar) decides how the side panels behave, not the
 * window's: the same 1440px window leaves the canvas far less room beside the
 * web editor's sidebar than in OAIY's window.
 *
 * Under COMPACT_WIDTH the panels start closed and only one side is open at a
 * time; under NARROW_WIDTH a panel lies over the canvas's edge instead of
 * pushing it, so the canvas keeps its width.
 */
const COMPACT_WIDTH = 1100;
const NARROW_WIDTH = 640;
/** A first guess before the builder is measured (it is, before the first paint). */
const isCompactNow = (): boolean => typeof window !== 'undefined' && window.innerWidth < 1240;

// Per-panel open state on a wide window, persisted across reloads.
const loadBoolPref = (key: string, fallback: boolean): boolean => {
  try {
    const raw = localStorage.getItem(key);
    if (raw === 'true') return true;
    if (raw === 'false') return false;
  } catch { /* SSR/private-mode — fall through */ }
  return fallback;
};
const saveBoolPref = (key: string, val: boolean): void => {
  try { localStorage.setItem(key, val ? 'true' : 'false'); } catch { /* ignore quota */ }
};

// Props for integrated mode (with OAIYApp)
interface OAIYBuilderProps {
  initialGraph?: WorkflowGraph;
  onGraphChange?: (graph: WorkflowGraph) => void;
  availableFlows?: Flow[];
  /** Package macros (higher priority than project macros for execution) */
  packageMacros?: Flow[];
  llmEndpoints?: LLMEndpoint[];
  projectConstants?: ProjectConstant[];
  projectSettings?: ProjectSettings;
  onUpdateSettings?: (updates: Partial<ProjectSettings>) => void;
  // Flow identification for job queue
  flowId?: string;
  flowName?: string;
  /** Whether this flow is a macro */
  isMacro?: boolean;
  /** Whether this macro has been modified from original */
  isMacroModified?: boolean;
  /** Save macro changes callback */
  onSaveMacro?: () => void;
  /** Revert macro to original callback */
  onRevertMacro?: () => void;
  /** Run macro callback - opens the macro runner modal */
  onRunMacro?: () => void;
  /**
   * False while another section of the editor (Data, Queue, Settings…) is
   * showing over the canvas: it stays mounted, but keys meant for that page
   * (Backspace in a list, Ctrl+Shift+R) must not reach the flow.
   */
  active?: boolean;
  // Toast notifications
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
  // Package mode props
  packageMode?: {
    manifest: OAIYPackageManifest;
    flow: Flow;
    sourcePath: string;
  } | null;
  /**
   * Granted-permission subset captured at trust-confirm time. Forwarded
   * into the JobManager submit call so the runtime applies the
   * fail-closed permission gate to package flows. `null` means trusted
   * local flow (no gate).
   */
  packagePermissionContext?: {
    packageId: string;
    grantedPermissions: string[];
    packageTrustLevel: 'untrusted' | 'trusted' | 'verified' | 'blocked';
  } | null;
  onLoadPackage?: () => void;
  onClosePackage?: () => void;
  onReloadPackage?: () => void;
  // Package nodes for the node palette
  activePackageId?: string | null;
  packageNodes?: PackageNodeInfo[];
}

export default function OAIYBuilder({
  initialGraph,
  onGraphChange,
  availableFlows = [],
  llmEndpoints = [],
  projectConstants = [],
  projectSettings,
  onUpdateSettings,
  flowId = 'default',
  flowName = 'Untitled Flow',
  isMacro = false,
  isMacroModified = false,
  onSaveMacro,
  onRevertMacro,
  onRunMacro,
  active = true,
  onShowToast,
  packageMode,
  packagePermissionContext,
  onLoadPackage,
  onClosePackage,
  onReloadPackage,
  activePackageId,
  packageNodes,
}: OAIYBuilderProps = {}) {
  // Dynamic node types from registry - updated when plugins register new components
  const [nodeTypesVersion, setNodeTypesVersion] = useState(() => getRegistryVersion());
  const nodeTypes = useMemo(() => {
    const types = getNodeTypes();
    const pkgNodes = Object.keys(types).filter(k => k.startsWith('pkg:'));
    if (pkgNodes.length > 0) {
      logger.debug(`nodeTypes includes ${pkgNodes.length} package nodes`, { packageNodes: pkgNodes });
    }
    return types;
  }, [nodeTypesVersion]);

  // Poll for registry changes (plugins may load after component mounts).
  // The 10s cap is an ABSOLUTE deadline from mount: a functional state update
  // + empty deps mean a version bump no longer re-runs this effect (which
  // previously restarted both the interval and the stop-timer, so the cap
  // never fired while nodes kept registering and polling ran unbounded).
  useEffect(() => {
    const deadline = Date.now() + 10000;
    const checkRegistry = () => {
      const currentVersion = getRegistryVersion();
      setNodeTypesVersion((prev) => (currentVersion !== prev ? currentVersion : prev));
      if (Date.now() >= deadline) clearInterval(interval);
    };

    // Check immediately and then periodically until the deadline.
    const interval = setInterval(checkRegistry, 100);
    checkRegistry();

    return () => clearInterval(interval);
  }, []);

  // ComfyUI workflow dialog state (rendered here to escape React Flow transform context)
  const [comfyWorkflowDialogState, setComfyWorkflowDialogState] = useState<{
    nodeId: string;
    analysis: ComfyUIAnalysis;
    fileName: string;
  } | null>(null);

  // Handle ComfyUI workflow dialog open (called from ImageGenNode)
  const handleOpenComfyWorkflowDialog = useCallback((nodeId: string, analysis: ComfyUIAnalysis, fileName: string) => {
    setComfyWorkflowDialogState({ nodeId, analysis, fileName });
  }, []);

  // Keyboard shortcut for package reload (Ctrl+Shift+R)
  useEffect(() => {
    if (!packageMode || !onReloadPackage || !active) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        onReloadPackage();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [packageMode, onReloadPackage, active]);


  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    onNodesChange,
    onEdgesChange,
    onConnect: onConnectBase,
    addNode,
    updateNodeData,
    deleteSelected,
    copySelected,
    pasteClipboard,
    hasClipboard,
    autoLayout,
    getWorkflowGraph,
  } = useWorkflow({
    availableFlows,
    initialGraph,
    onGraphChange,
    projectSettings,
    onUpdateSettings,
  });

  // Ref to hold quick connect close function (set after hook initialization)
  const quickConnectCloseRef = useRef<(() => void) | null>(null);

  // Wrap onConnect to also close quick connect popup when a connection is made
  const onConnect = useCallback((connection: Parameters<typeof onConnectBase>[0]) => {
    onConnectBase(connection);
    // Close quick connect popup if open
    quickConnectCloseRef.current?.();
  }, [onConnectBase]);

  // Reject connections whose source/target handles declare incompatible
  // types (e.g., wiring a text source into a service's `image` input).
  // Re-built per render so the validator always sees the current nodes
  // — `isValidConnection` only fires during a drag, so the closure cost
  // is bounded to user interaction.
  const isValidConnection = useMemo(
    () => makeIsValidConnection(nodes),
    [nodes],
  );

  // Workflow execution hook - manages job submission, running, stopping, and completion
  const {
    isRunning,
    logs,
    showRunModal,
    showServiceDialog,
    flowTransitioning,
    runWorkflow,
    stopWorkflow,
    clearLogs,
    closeRunModal,
    confirmRunModal: handleRunModalConfirm,
    closeServiceDialog,
    proceedAfterServiceDialog,
    finishFlowTransition,
  } = useWorkflowExecution({
    flowId,
    flowName,
    isMacro,
    nodes,
    edges,
    getWorkflowGraph,
    setNodes,
    updateNodeData,
    onShowToast,
    hasInputNodes,
    packageMode,
    packagePermissionContext,
  });

  // Calculate selected nodes for multi-selection display
  const selectedNodes = useMemo(() => nodes.filter(n => n.selected), [nodes]);

  // Reference for reactFlowInstance (needed by flow transition effect)
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const reactFlowInstance = useRef<{
    screenToFlowPosition: (position: { x: number; y: number }) => { x: number; y: number };
    getViewport: () => { x: number; y: number; zoom: number };
    setViewport: (viewport: Viewport, options?: { duration?: number }) => void;
    fitView: (options?: { padding?: number; duration?: number; maxZoom?: number }) => void;
  } | null>(null);

  // Clear transitioning state once nodes are ready (after a microtask to ensure render)
  // This effect must come AFTER useWorkflow since it depends on 'nodes'
  useEffect(() => {
    if (flowTransitioning && nodes.length >= 0) {
      // Use double RAF to ensure:
      // 1. First RAF: nodes are rendered in DOM
      // 2. Second RAF: fitView is applied, then show canvas
      let innerRaf = 0;
      const rafId = requestAnimationFrame(() => {
        // Fit view to new nodes before showing
        if (reactFlowInstance.current) {
          reactFlowInstance.current.fitView({ padding: 0.2, duration: 0, maxZoom: 1 });
        }
        // Second RAF ensures fitView is painted before revealing
        innerRaf = requestAnimationFrame(() => {
          finishFlowTransition();
        });
      });
      // Cancel BOTH frames on cleanup so the inner one can't fire finishFlowTransition
      // after the deps change / unmount (OAIYBuilder remounts on flow switch).
      return () => {
        cancelAnimationFrame(rafId);
        cancelAnimationFrame(innerRaf);
      };
    }
  }, [flowTransitioning, nodes, finishFlowTransition]);

  // The canvas's two side panels, one family: the node palette on the left,
  // and the inspector on the right (the selected node's properties above the
  // execution log). On a wide window they are docked and remember whether
  // they were open. Below the xl width (1240px: Tailwind's xl and the shell's
  // rail rule) they start closed and only one side is open at a time, so the
  // canvas keeps its room; under 760px a panel lies over the canvas's edge.
  // The toolbar at the foot of the canvas opens and closes all three.
  const builderRef = useRef<HTMLDivElement>(null);
  const [builderWidth, setBuilderWidth] = useState(0);
  const compact = builderWidth > 0 ? builderWidth < COMPACT_WIDTH : isCompactNow();
  const narrow = builderWidth > 0 && builderWidth < NARROW_WIDTH;
  const [paletteOpen, setPaletteOpen] = useState<boolean>(() => !isCompactNow() && !loadBoolPref('oaiy.ui.paletteCollapsed', false));
  const [propsOpen, setPropsOpen] = useState<boolean>(() => !isCompactNow() && loadBoolPref('oaiy.ui.propertiesVisible', true));
  const [logOpen, setLogOpen] = useState<boolean>(() => !isCompactNow() && loadBoolPref('oaiy.ui.logPanelVisible', true));
  // Measured before the first paint: a compact builder opens with its panels
  // closed, a wide one as it was left.
  useLayoutEffect(() => {
    const el = builderRef.current;
    if (!el) return;
    const w = el.getBoundingClientRect().width;
    setBuilderWidth(w);
    if (w > 0) {
      const wide = w >= COMPACT_WIDTH;
      // An empty flow's first step is a node, so it opens with the palette.
      const empty = nodes.length === 0;
      setPaletteOpen(empty || (wide && !loadBoolPref('oaiy.ui.paletteCollapsed', false)));
      setPropsOpen(wide && loadBoolPref('oaiy.ui.propertiesVisible', true));
      setLogOpen(wide && loadBoolPref('oaiy.ui.logPanelVisible', true));
    }
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => setBuilderWidth(entries[0]?.contentRect.width ?? 0));
    ro.observe(el);
    return () => ro.disconnect();
    // Once, when the flow opens (the builder remounts per flow).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Growing narrow with both sides open (the rail opened, the window
  // shrank): the palette gives way, so one side is open at a time.
  useEffect(() => {
    if (compact && paletteOpen && (propsOpen || logOpen)) setPaletteOpen(false);
    // Only on crossing into compact, not on every toggle (the toggles keep it true).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compact]);
  const setPalette = useCallback((open: boolean) => {
    setPaletteOpen(open);
    if (compact) {
      if (open) { setPropsOpen(false); setLogOpen(false); }
    } else {
      saveBoolPref('oaiy.ui.paletteCollapsed', !open);
    }
  }, [compact]);
  const setProps = useCallback((open: boolean) => {
    setPropsOpen(open);
    if (compact) {
      if (open) setPaletteOpen(false);
    } else {
      saveBoolPref('oaiy.ui.propertiesVisible', open);
    }
  }, [compact]);
  const setLog = useCallback((open: boolean) => {
    setLogOpen(open);
    if (compact) {
      if (open) setPaletteOpen(false);
    } else {
      saveBoolPref('oaiy.ui.logPanelVisible', open);
    }
  }, [compact]);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [showMiniMap, setShowMiniMap] = useState(true); // Toggle minimap visibility
  // The canvas's own width: the map gives way to the toolbar when it is
  // narrow, and the toolbar drops its words when narrower still.
  const [canvasWidth, setCanvasWidth] = useState(0);

  // Get module nodes for quick-connect filtering
  const { nodes: moduleNodes } = useModuleNodes();

  // Quick-connect hook for drag-to-create node functionality
  const {
    quickConnectState,
    quickConnectCompatibleNodes,
    quickConnectHandlePosition,
    handleConnectStart,
    handleConnectEnd,
    handleQuickConnectClose,
    handleQuickConnectNodeSelect,
    onViewportChange: onQuickConnectViewportChange,
    setSearchQuery: setQuickConnectSearchQuery,
  } = useQuickConnect({
    nodes,
    moduleNodes,
    addNode,
    onConnect,
    reactFlowInstance,
  });

  // Keep quick connect close ref in sync with the hook's close function
  useEffect(() => {
    quickConnectCloseRef.current = handleQuickConnectClose;
  }, [handleQuickConnectClose]);

  // Use stable handlers to avoid recreating functions on each render
  const handlers = useStableHandlers(updateNodeData);

  const handleAutoLayoutHorizontal = useCallback(() => {
    autoLayout('LR');
    setContextMenu(null);
  }, [autoLayout]);

  const handleAutoLayoutVertical = useCallback(() => {
    autoLayout('TB');
    setContextMenu(null);
  }, [autoLayout]);

  // Node grouping hook
  const {
    handleGroupSelected: groupSelectedBase,
    handleUngroupSelected: ungroupSelectedBase,
  } = useNodeGrouping({ nodes, setNodes, onShowToast });

  // Wrap handlers to also close context menu
  const handleGroupSelected = useCallback(() => {
    groupSelectedBase();
    setContextMenu(null);
  }, [groupSelectedBase]);

  const handleUngroupSelected = useCallback(() => {
    ungroupSelectedBase();
    setContextMenu(null);
  }, [ungroupSelectedBase]);

  // Collapse/Expand all nodes — single batched setNodes pass instead of a
  // per-node updateNodeData forEach (each call ran its own O(N) setNodes,
  // making this O(N^2)).
  const handleCollapseAll = useCallback(() => {
    setNodes((nds) => nds.map((n) => ({ ...n, data: { ...n.data, _collapsed: true } })));
    setContextMenu(null);
  }, [setNodes]);

  const handleExpandAll = useCallback(() => {
    setNodes((nds) => nds.map((n) => ({ ...n, data: { ...n.data, _collapsed: false } })));
    setContextMenu(null);
  }, [setNodes]);

  // Context menu handler for right-click on canvas
  const handlePaneContextMenu = useCallback((event: MouseEvent | React.MouseEvent) => {
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY });
  }, []);

  // Kept as the canvas click hook: the floating workflow menu it used to
  // close is gone, and nothing else opens on canvas click today.
  const handleCanvasClick = useCallback(() => {}, []);

  // Handle clicking on the pane (background) to deselect nodes and close menus
  const handlePaneClick = useCallback(() => {
    if (contextMenu) setContextMenu(null);
  }, [contextMenu]);

  // Handle edge click to select it
  const handleEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      // Toggle selection on the clicked edge
      onEdgesChange([
        {
          id: edge.id,
          type: 'select',
          selected: !edge.selected,
        },
      ]);
    },
    [onEdgesChange]
  );

  // Wrap nodes with stable onChange handlers - handlers object is stable
  // so this only re-runs when nodes array changes
  // During flow transition, mark nodes as hidden to prevent visible jump
  //
  // Per-id data cache: rebuilding the ~160-field `data` object for EVERY node
  // on every `nodes` change gives each node a fresh `data` identity, defeating
  // the React.memo on the node components — a drag (which only mutates
  // `node.position`) would otherwise re-render every node. We cache the built
  // `data` keyed by node id and reuse it unless THAT node's own `node.data`
  // reference changed, `flowTransitioning` flipped, or any shared dependency
  // changed since the last build.
  const nodeDataCacheRef = useRef<Map<string, {
    dataKey: unknown;
    flowTransitioning: boolean;
    sharedKey: NodeDataSharedKey;
    data: Record<string, unknown>;
  }>>(new Map());
  const nodesWithHandlers = useMemo(() => {
    const cache = nodeDataCacheRef.current;
    // Shared dependencies — if any of these change identity, every node's data
    // must be rebuilt (they're spread into every node's data object).
    const sharedKey: NodeDataSharedKey = { handlers, availableFlows, llmEndpoints, projectConstants, projectSettings, onShowToast, handleOpenComfyWorkflowDialog };
    const liveIds = new Set<string>();
    const result = nodes.map((node: Node) => {
      liveIds.add(node.id);
      const cached = cache.get(node.id);
      if (
        cached &&
        cached.dataKey === node.data &&
        cached.flowTransitioning === flowTransitioning &&
        cached.sharedKey.handlers === sharedKey.handlers &&
        cached.sharedKey.availableFlows === sharedKey.availableFlows &&
        cached.sharedKey.llmEndpoints === sharedKey.llmEndpoints &&
        cached.sharedKey.projectConstants === sharedKey.projectConstants &&
        cached.sharedKey.projectSettings === sharedKey.projectSettings &&
        cached.sharedKey.onShowToast === sharedKey.onShowToast &&
        cached.sharedKey.handleOpenComfyWorkflowDialog === sharedKey.handleOpenComfyWorkflowDialog
      ) {
        // Reuse the cached data object so its identity stays stable; only the
        // node wrapper (position etc.) is fresh from React Flow.
        return { ...node, hidden: flowTransitioning, data: cached.data };
      }
      const data = buildNodeData(node);
      cache.set(node.id, { dataKey: node.data, flowTransitioning, sharedKey, data });
      return { ...node, hidden: flowTransitioning, data };
    });
    // Evict cache entries for nodes that no longer exist.
    for (const id of cache.keys()) {
      if (!liveIds.has(id)) cache.delete(id);
    }
    return result;

    // The per-node data object builder. Extracted so the cache-hit path can
    // skip it entirely.
    function buildNodeData(node: Node): Record<string, unknown> {
      return {
        ...node.data,
        // Use stable handlers bound to node.id
        onChange: handlers.onChange(node.id),
        onFileLoad: handlers.onFileLoad(node.id),
        onModelChange: handlers.onModelChange(node.id),
        onSystemPromptChange: handlers.onSystemPromptChange(node.id),
        onEndpointChange: handlers.onEndpointChange(node.id),
        onApiKeyChange: handlers.onApiKeyChange(node.id),
        onHeadersChange: handlers.onHeadersChange(node.id),
        onImageFormatChange: handlers.onImageFormatChange(node.id),
        onRequestFormatChange: handlers.onRequestFormatChange(node.id),
        onEnableThinkingChange: handlers.onEnableThinkingChange(node.id),
        onContextLengthChange: handlers.onContextLengthChange(node.id),
        onMaxTokensChange: handlers.onMaxTokensChange(node.id),
        onCodeChange: handlers.onCodeChange(node.id),
        onMethodChange: handlers.onMethodChange(node.id),
        onUrlChange: handlers.onUrlChange(node.id),
        onBodyChange: handlers.onBodyChange(node.id),
        onKeyChange: handlers.onKeyChange(node.id),
        onModeChange: handlers.onModeChange(node.id),
        onDefaultValueChange: handlers.onDefaultValueChange(node.id),
        onLabelChange: handlers.onLabelChange(node.id),
        onShowToast: onShowToast,
        onNegativePromptChange: handlers.onNegativePromptChange(node.id),
        onSeedChange: handlers.onSeedChange(node.id),
        onWorkflowTemplateChange: handlers.onWorkflowTemplateChange(node.id),
        onFilenameChange: handlers.onFilenameChange(node.id),
        onFormatChange: handlers.onFormatChange(node.id),
        onInputCountChange: handlers.onInputCountChange(node.id),
        onTemplateChange: handlers.onTemplateChange(node.id),
        onInputNamesChange: handlers.onInputNamesChange(node.id),
        onIterationsChange: handlers.onIterationsChange(node.id),
        onLoopModeChange: handlers.onLoopModeChange(node.id),
        onLoopNameChange: handlers.onLoopNameChange(node.id),
        // Loop End handlers
        onStopConditionChange: handlers.onStopConditionChange(node.id),
        onStopValueChange: handlers.onStopValueChange(node.id),
        onStopFieldChange: handlers.onStopFieldChange(node.id),
        onOperatorChange: handlers.onOperatorChange(node.id),
        onCompareValueChange: handlers.onCompareValueChange(node.id),
        // Subflow node specific
        onFlowSelect: handlers.onFlowSelect(node.id),
        onInputMappingsChange: handlers.onInputMappingsChange(node.id),
        availableFlows: availableFlows,
        // Endpoint selection for AI LLM nodes
        onEndpointIdChange: handlers.onEndpointIdChange(node.id),
        llmEndpoints: llmEndpoints,
        // Provider and API key constant handlers
        onProviderChange: handlers.onProviderChange(node.id),
        onApiKeyConstantChange: handlers.onApiKeyConstantChange(node.id),
        // Project constants and settings (for defaults and autocomplete)
        projectConstants: projectConstants,
        projectSettings: projectSettings,
        // Image gen specific
        onApiFormatChange: handlers.onApiFormatChange(node.id),
        onSizeChange: handlers.onSizeChange(node.id),
        onQualityChange: handlers.onQualityChange(node.id),
        onOutputFormatChange: handlers.onOutputFormatChange(node.id),
        onBackgroundChange: handlers.onBackgroundChange(node.id),
        onAspectRatioChange: handlers.onAspectRatioChange(node.id),
        // Browser Session handlers
        onBrowserProfileChange: handlers.onBrowserProfileChange(node.id),
        onSessionModeChange: handlers.onSessionModeChange(node.id),
        onCustomUserAgentChange: handlers.onCustomUserAgentChange(node.id),
        onCustomHeadersChange: handlers.onCustomHeadersChange(node.id),
        onInitialCookiesChange: handlers.onInitialCookiesChange(node.id),
        onViewportWidthChange: handlers.onViewportWidthChange(node.id),
        onViewportHeightChange: handlers.onViewportHeightChange(node.id),
        // Browser Request handlers
        onBodyTypeChange: handlers.onBodyTypeChange(node.id),
        onResponseFormatChange: handlers.onResponseFormatChange(node.id),
        onFollowRedirectsChange: handlers.onFollowRedirectsChange(node.id),
        onMaxRedirectsChange: handlers.onMaxRedirectsChange(node.id),
        onWaitForSelectorChange: handlers.onWaitForSelectorChange(node.id),
        onWaitTimeoutChange: handlers.onWaitTimeoutChange(node.id),
        // Browser Extract handlers
        onExtractionTypeChange: handlers.onExtractionTypeChange(node.id),
        onSelectorChange: handlers.onSelectorChange(node.id),
        onPatternChange: handlers.onPatternChange(node.id),
        onExtractTargetChange: handlers.onExtractTargetChange(node.id),
        onAttributeNameChange: handlers.onAttributeNameChange(node.id),
        onMaxLengthChange: handlers.onMaxLengthChange(node.id),
        // Browser Control handlers
        onActionChange: handlers.onActionChange(node.id),
        onValueChange: handlers.onValueChange(node.id),
        onScrollDirectionChange: handlers.onScrollDirectionChange(node.id),
        onScrollAmountChange: handlers.onScrollAmountChange(node.id),
        // Database handlers
        onOperationChange: handlers.onOperationChange(node.id),
        onStorageTypeChange: handlers.onStorageTypeChange(node.id),
        onCollectionNameChange: handlers.onCollectionNameChange(node.id),
        onFilterJsonChange: handlers.onFilterJsonChange(node.id),
        onTableNameChange: handlers.onTableNameChange(node.id),
        onWhereClauseChange: handlers.onWhereClauseChange(node.id),
        onRawSqlChange: handlers.onRawSqlChange(node.id),
        onLimitChange: handlers.onLimitChange(node.id),
        onAutoCreateTableChange: handlers.onAutoCreateTableChange(node.id),
        onTableSchemaChange: handlers.onTableSchemaChange(node.id),
        onColumnMappingsChange: handlers.onColumnMappingsChange(node.id),
        // Text-to-Speech handlers
        onVoiceChange: handlers.onVoiceChange(node.id),
        onCustomSpeakerIdChange: handlers.onCustomSpeakerIdChange(node.id),
        onSpeedChange: handlers.onSpeedChange(node.id),
        // Collapsible node handlers
        onCollapsedChange: handlers.onCollapsedChange(node.id),
        // Folder Input handlers
        onPathChange: handlers.onPathChange(node.id),
        onRecursiveChange: handlers.onRecursiveChange(node.id),
        onIncludePatternsChange: handlers.onIncludePatternsChange(node.id),
        onExcludePatternsChange: handlers.onExcludePatternsChange(node.id),
        onMaxFilesChange: handlers.onMaxFilesChange(node.id),
        onBrowse: handlers.onBrowse(node.id),
        // File Read handlers
        onReadAsChange: handlers.onReadAsChange(node.id),
        onCsvHasHeaderChange: handlers.onCsvHasHeaderChange(node.id),
        // Text Chunker handlers
        onChunkSizeChange: handlers.onChunkSizeChange(node.id),
        onOverlapChange: handlers.onOverlapChange(node.id),
        // Video Frame Extractor handlers
        onIntervalSecondsChange: handlers.onIntervalSecondsChange(node.id),
        onStartTimeChange: handlers.onStartTimeChange(node.id),
        onEndTimeChange: handlers.onEndTimeChange(node.id),
        onMaxFramesChange: handlers.onMaxFramesChange(node.id),
        onBatchSizeChange: handlers.onBatchSizeChange(node.id),
        // Video Input handlers
        onVideoLoad: handlers.onVideoLoad(node.id),
        // File Write handlers
        onTargetPathChange: handlers.onTargetPathChange(node.id),
        onFilenamePatternChange: handlers.onFilenamePatternChange(node.id),
        onContentTypeChange: handlers.onContentTypeChange(node.id),
        onCreateDirectoriesChange: handlers.onCreateDirectoriesChange(node.id),
        onBrowseFolder: handlers.onBrowseFolder(node.id),
        // Vectorize node handlers
        onOutputPathChange: handlers.onOutputPathChange(node.id),
        onColorCountChange: handlers.onColorCountChange(node.id),
        onSmoothnessChange: handlers.onSmoothnessChange(node.id),
        onMinAreaChange: handlers.onMinAreaChange(node.id),
        onRemoveBackgroundChange: handlers.onRemoveBackgroundChange(node.id),
        onOptimizeChange: handlers.onOptimizeChange(node.id),
        // ComfyUI workflow handlers
        onComfyWorkflowChange: handlers.onComfyWorkflowChange(node.id),
        onComfyWorkflowNameChange: handlers.onComfyWorkflowNameChange(node.id),
        onComfyPrimaryPromptNodeIdChange: handlers.onComfyPrimaryPromptNodeIdChange(node.id),
        onComfyImageInputNodeIdsChange: handlers.onComfyImageInputNodeIdsChange(node.id),
        onComfyImageInputConfigsChange: handlers.onComfyImageInputConfigsChange(node.id),
        onComfySeedModeChange: handlers.onComfySeedModeChange(node.id),
        onComfyFixedSeedChange: handlers.onComfyFixedSeedChange(node.id),
        // Video parameter handlers
        onComfyFrameCountNodeIdChange: handlers.onComfyFrameCountNodeIdChange(node.id),
        onComfyFrameCountChange: handlers.onComfyFrameCountChange(node.id),
        onComfyResolutionNodeIdChange: handlers.onComfyResolutionNodeIdChange(node.id),
        onComfyWidthChange: handlers.onComfyWidthChange(node.id),
        onComfyHeightChange: handlers.onComfyHeightChange(node.id),
        onComfyFrameRateNodeIdChange: handlers.onComfyFrameRateNodeIdChange(node.id),
        onComfyFrameRateChange: handlers.onComfyFrameRateChange(node.id),
        // Wan2GP handlers
        onWan2gpModelChange: handlers.onWan2gpModelChange(node.id),
        onWan2gpStepsChange: handlers.onWan2gpStepsChange(node.id),
        onWan2gpDurationChange: handlers.onWan2gpDurationChange(node.id),
        onWan2gpVramChange: handlers.onWan2gpVramChange(node.id),
        onWan2gpSeedChange: handlers.onWan2gpSeedChange(node.id),
        onWan2gpRandomSeedChange: handlers.onWan2gpRandomSeedChange(node.id),
        onWan2gpResolutionChange: handlers.onWan2gpResolutionChange(node.id),
        onWan2gpSamplerChange: handlers.onWan2gpSamplerChange(node.id),
        // Dynamic image input count
        onImageInputCountChange: handlers.onImageInputCountChange(node.id),
        // mmproj path picker (AI LLM node, Internal mode)
        onMmprojPathChange: handlers.onMmprojPathChange(node.id),
        // ComfyUI workflow dialog opener (opens dialog at OAIYBuilder level to escape transform context)
        onOpenComfyWorkflowDialog: (analysis: ComfyUIAnalysis, fileName: string) => handleOpenComfyWorkflowDialog(node.id, analysis, fileName),
      };
    }
  }, [nodes, handlers, availableFlows, llmEndpoints, projectConstants, projectSettings, handleOpenComfyWorkflowDialog, onShowToast, flowTransitioning]);

  // Make all edges selectable and use custom edge type
  // Also filter to only show edges with valid source and target nodes (prevents flicker during flow loading)
  // During flow transition, return empty array to completely hide edges until nodes are positioned
  const edgesWithOptions = useMemo(() => {
    // Hide all edges during flow transition to prevent flicker
    if (flowTransitioning) {
      return [];
    }
    const nodeIds = new Set(nodes.map(n => n.id));
    return edges
      .filter((edge: Edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge: Edge) => ({
        ...edge,
        type: 'selectable',
        selectable: true,
        deletable: true,
        focusable: true,
      }));
  }, [edges, nodes, flowTransitioning]);

  // Calculate smart pan bounds based on node positions
  // This limits panning to the area where nodes exist plus generous padding
  const translateExtent = useMemo((): [[number, number], [number, number]] => {
    if (nodes.length === 0) {
      // Default bounds when no nodes - allow reasonable movement
      return [[-2000, -2000], [2000, 2000]];
    }

    // Calculate bounding box of all nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const node of nodes) {
      const x = node.position.x;
      const y = node.position.y;
      // Estimate node size (most nodes are ~300x200)
      const width = (node.measured?.width ?? node.width ?? 300);
      const height = (node.measured?.height ?? node.height ?? 200);

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + width);
      maxY = Math.max(maxY, y + height);
    }

    // Add very generous padding (3000px) around the node bounds
    // This allows plenty of space to pan and add new nodes while preventing infinite scrolling
    const padding = 3000;
    return [
      [minX - padding, minY - padding],
      [maxX + padding, maxY + padding]
    ];
  }, [nodes]);

  // Store viewport to persist zoom level across flow switches
  // Using a ref so it persists across re-renders and flow changes
  const savedViewportRef = useRef<Viewport | null>(null);
  const hasInitializedViewport = useRef(false);

  // Track viewport changes to save the current zoom/pan state
  const handleViewportChange = useCallback((viewport: Viewport) => {
    savedViewportRef.current = viewport;
    // Notify quick connect hook of viewport changes to update handle position
    onQuickConnectViewportChange();
  }, [onQuickConnectViewportChange]);

  // Reset hasInitializedViewport when flow changes so we can set viewport on first load
  useEffect(() => {
    hasInitializedViewport.current = false;
  }, [flowId]);

  // Edge scrolling - pan when mouse moves near edges of the canvas
  useEdgeScrolling({
    wrapperRef: reactFlowWrapper,
    reactFlowInstance,
    translateExtent,
  });

  // Cascade offset for click-to-add so successive palette clicks don't stack
  // every node at the exact viewport centre (hiding the earlier ones). Drag
  // -from-palette is unaffected — it drops at the cursor. Wraps every 8 to stay
  // near the centre.
  const clickAddOffsetRef = useRef(0);

  // Handle adding node from palette click
  const handleAddNode = useCallback(
    (type: NodeType) => {
      // Add at center of viewport
      if (reactFlowInstance.current) {
        // screenToFlowPosition already accounts for pan AND zoom, so use its result
        // directly. Subtracting the viewport translation again (a screen-pixel value
        // from a flow coordinate) misplaced every palette-click node after any pan —
        // matching the correct pattern already used by handleAddNodeAtPosition/Macro.
        const bounds = reactFlowWrapper.current?.getBoundingClientRect();
        const position = reactFlowInstance.current.screenToFlowPosition({
          x: bounds ? bounds.left + bounds.width / 2 : window.innerWidth / 2,
          y: bounds ? bounds.top + bounds.height / 2 : window.innerHeight / 2,
        });
        const k = (clickAddOffsetRef.current++ % 8) * 36;
        addNode(type, { x: position.x - 128 + k, y: position.y - 64 + k });
      } else {
        addNode(type);
      }
    },
    [addNode]
  );

  // Handle adding node at specific screen position (from drag and drop)
  const handleAddNodeAtPosition = useCallback(
    (type: NodeType, screenX: number, screenY: number) => {
      if (reactFlowInstance.current) {
        const position = reactFlowInstance.current.screenToFlowPosition({
          x: screenX,
          y: screenY,
        });
        addNode(type, position);
      } else {
        addNode(type);
      }
    },
    [addNode]
  );

  // Filter macros from available flows
  const macros = useMemo(() => {
    return availableFlows.filter(flow => flow.isMacro);
  }, [availableFlows]);

  // Handle adding macro from palette click
  const handleAddMacro = useCallback(
    (macroData: MacroNodeData) => {
      const extraData = {
        _macroWorkflowId: macroData._macroWorkflowId,
        _macroName: macroData._macroName,
        _macroInputs: macroData._macroInputs,
        _macroOutputs: macroData._macroOutputs,
      };

      if (reactFlowInstance.current) {
        const bounds = reactFlowWrapper.current?.getBoundingClientRect();
        const position = reactFlowInstance.current.screenToFlowPosition({
          x: bounds ? bounds.left + bounds.width / 2 : window.innerWidth / 2,
          y: bounds ? bounds.top + bounds.height / 2 : window.innerHeight / 2,
        });
        addNode('macro' as NodeType, { x: position.x - 128, y: position.y - 64 }, extraData);
      } else {
        addNode('macro' as NodeType, undefined, extraData);
      }
    },
    [addNode]
  );

  // Handle adding macro at specific screen position (from drag and drop)
  const handleAddMacroAtPosition = useCallback(
    (macroData: MacroNodeData, screenX: number, screenY: number) => {
      const extraData = {
        _macroWorkflowId: macroData._macroWorkflowId,
        _macroName: macroData._macroName,
        _macroInputs: macroData._macroInputs,
        _macroOutputs: macroData._macroOutputs,
      };

      if (reactFlowInstance.current) {
        const position = reactFlowInstance.current.screenToFlowPosition({
          x: screenX,
          y: screenY,
        });
        addNode('macro' as NodeType, position, extraData);
      } else {
        addNode('macro' as NodeType, undefined, extraData);
      }
    },
    [addNode]
  );

  // Keyboard shortcuts
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      // Don't handle shortcuts when typing in inputs, textareas, or contenteditable elements
      const target = event.target as HTMLElement;
      const isEditing =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable ||
        target.closest('.monaco-editor'); // Monaco editor

      if (isEditing) {
        return;
      }

      // Copy: Ctrl+C / Cmd+C
      if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
        event.preventDefault();
        copySelected();
        return;
      }

      // Paste: Ctrl+V / Cmd+V
      if ((event.ctrlKey || event.metaKey) && event.key === 'v') {
        event.preventDefault();
        // Get mouse position relative to the flow canvas for paste location
        // If we don't have a good position, paste with default offset
        pasteClipboard();
        return;
      }

      // Select All: Ctrl+A / Cmd+A
      if ((event.ctrlKey || event.metaKey) && event.key === 'a') {
        event.preventDefault();
        setNodes((nds) => nds.map((n) => ({ ...n, selected: true })));
        setEdges((eds) => eds.map((e) => ({ ...e, selected: true })));
        return;
      }

      // Group: Ctrl+G / Cmd+G
      if ((event.ctrlKey || event.metaKey) && event.key === 'g') {
        event.preventDefault();
        handleGroupSelected();
        return;
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        deleteSelected();
      }
    },
    [deleteSelected, copySelected, pasteClipboard, setNodes, setEdges, handleGroupSelected]
  );

  // Handle ComfyUI workflow dialog confirmation
  const handleComfyWorkflowDialogConfirm = useCallback((config: ComfyUIWorkflowConfig) => {
    if (comfyWorkflowDialogState) {
      const { nodeId, fileName } = comfyWorkflowDialogState;
      updateNodeData(nodeId, {
        comfyWorkflow: config.workflowJson,
        comfyWorkflowName: fileName,
        comfyPrimaryPromptNodeId: config.primaryPromptNodeId,
        comfyImageInputNodeIds: config.imageInputNodeIds,
        comfyImageInputConfigs: config.imageInputConfigs,
        comfyAllImageNodeIds: config.allImageNodeIds, // All image nodes for bypassing unselected ones
        comfySeedMode: config.seedMode,
        comfyFixedSeed: config.fixedSeed,
      });
      setComfyWorkflowDialogState(null);
    }
  }, [comfyWorkflowDialogState, updateNodeData]);

  // Handle ComfyUI workflow dialog cancel
  const handleComfyWorkflowDialogCancel = useCallback(() => {
    setComfyWorkflowDialogState(null);
  }, []);

  // Watch the canvas's width (the side panels and the window both change it).
  useEffect(() => {
    const el = reactFlowWrapper.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => setCanvasWidth(entries[0]?.contentRect.width ?? 0));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Drawn over the canvas's foot: the map only while there is room beside the toolbar.
  const mapRoom = canvasWidth === 0 || canvasWidth >= 520;
  const tightToolbar = canvasWidth > 0 && canvasWidth < 560;
  const inspectorOpen = propsOpen || logOpen;

  return (
    <div
      ref={builderRef}
      className={`oaiy-builder${narrow ? ' narrow' : ''}${paletteOpen ? ' left-open' : ''}${propsOpen || logOpen ? ' right-open' : ''}`}
      style={{ backgroundColor: 'rgb(var(--bg-primary))' }}
      onKeyDown={onKeyDown}
      tabIndex={0}
    >
      {/* Left: the node palette, docked beside the canvas. */}
      {paletteOpen && (
        <NodePalette
          onAddNode={handleAddNode}
          onAddNodeAtPosition={handleAddNodeAtPosition}
          onClose={() => setPalette(false)}
          closeOnAdd={narrow}
          macros={macros}
          onAddMacro={handleAddMacro}
          onAddMacroAtPosition={handleAddMacroAtPosition}
          activePackageId={activePackageId}
          packageNodes={packageNodes}
        />
      )}

      {/* Center - Canvas */}
      <div
        ref={reactFlowWrapper}
        className="relative h-full min-w-0 flex-1"
        onClick={handleCanvasClick}
        onContextMenu={(e) => e.preventDefault()}
      >
        {/* Loading overlay for flow transitions */}
        {flowTransitioning && (
          <div className="oaiy-canvas-busy">
            <div>
              <span className="oaiy-spinner" aria-hidden="true" />
              <span>Opening {flowName}…</span>
            </div>
          </div>
        )}

        <ReactFlow
          nodes={nodesWithHandlers}
          edges={edgesWithOptions}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onEdgeClick={handleEdgeClick}
          onInit={(instance) => {
            reactFlowInstance.current = instance;
            // Restore saved viewport or fit view on first load
            if (!hasInitializedViewport.current) {
              hasInitializedViewport.current = true;
              if (savedViewportRef.current) {
                // Restore the saved viewport (preserves zoom level)
                instance.setViewport(savedViewportRef.current, { duration: 0 });
              } else if (nodes.length > 0) {
                // First time: fit view to show all nodes
                instance.fitView({ padding: 0.2, duration: 0, maxZoom: 1 });
              } else {
                // Fitting empty bounds can zoom to the maximum before the first node exists.
                instance.setViewport({ x: 0, y: 0, zoom: 1 }, { duration: 0 });
              }
            }
            // Clear transitioning state after viewport is set - use RAF to ensure paint
            if (flowTransitioning) {
              requestAnimationFrame(() => finishFlowTransition());
            }
          }}
          onViewportChange={handleViewportChange}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          className={`touch-manipulation transition-opacity duration-100 ${flowTransitioning ? 'opacity-0' : 'opacity-100'}`}
          style={CANVAS_STYLE}
          defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
          edgesReconnectable
          connectOnClick={true}
          connectionLineStyle={CONNECTION_LINE_STYLE}
          connectionLineComponent={TypedConnectionLine}
          proOptions={PRO_OPTIONS}
          onPaneClick={handlePaneClick}
          onPaneContextMenu={handlePaneContextMenu}
          onConnectStart={handleConnectStart}
          onConnectEnd={handleConnectEnd}
          // Backspace deletes the selection only while the canvas is the page
          // showing: React Flow listens on the whole document.
          deleteKeyCode={active ? 'Backspace' : null}
          // Navigation: scroll wheel zooms, left/middle/right-drag to pan
          panOnScroll={false}
          panOnDrag
          zoomOnPinch
          zoomOnScroll
          zoomActivationKeyCode={null}
          preventScrolling
          // Multi-select: Ctrl+drag creates selection box (overrides pan when Ctrl held)
          selectionOnDrag
          selectionKeyCode="Control"
          multiSelectionKeyCode="Control"
          elementsSelectable
          selectNodesOnDrag={false}
          // Auto-pan when dragging nodes or connections near edges
          autoPanOnNodeDrag
          autoPanOnConnect
          autoPanSpeed={8}
          // Require minimum movement before starting drag (prevents accidental drags)
          nodeDragThreshold={3}
          // Allow connections even when not perfectly aligned
          connectionMode={ConnectionMode.Loose}
          // Zoom settings
          minZoom={0.1}
          maxZoom={2}
        // No pan limits - allow free panning in any direction
        >
          <Background color="currentColor" style={BACKGROUND_STYLE} gap={20} size={1} />
          {/* A new flow's hint. Pointer-events none, so it never blocks the
              canvas; it points at the toolbar's Nodes and the right-click menu. */}
          {nodes.length === 0 && !flowTransitioning && (
            <div className="oaiy-canvas-hint">
              <div>
                <i><Plus size={18} /></i>
                <p>This flow is empty</p>
                <small>
                  {narrow
                    ? 'Tap Nodes below to add the first one.'
                    : 'Add a node from Nodes below: click it, or drag it onto the canvas. Right-click the canvas for more.'}
                </small>
              </div>
            </div>
          )}
          <Controls position="bottom-left" />
          {/* The map, while there is room for it beside the toolbar. */}
          {mapRoom && (
            <div className="oaiy-minimap">
              {showMiniMap ? (
                <>
                  <button
                    type="button"
                    onClick={() => setShowMiniMap(false)}
                    className="oaiy-icon-btn"
                    title="Hide the map"
                    aria-label="Hide the map"
                  >
                    <X size={12} />
                  </button>
                  <MiniMap nodeColor={miniMapNodeColor} style={MINIMAP_STYLE} />
                </>
              ) : (
                <button type="button" onClick={() => setShowMiniMap(true)} className="oaiy-map-toggle" title="Show the map">
                  <MapIcon size={14} /> Map
                </button>
              )}
            </div>
          )}
        </ReactFlow>

        {/* Right-click Context Menu */}
        {contextMenu && (
          <CanvasContextMenu
            position={contextMenu}
            nodes={nodes}
            hasClipboard={hasClipboard()}
            onCopy={copySelected}
            onPaste={pasteClipboard}
            onGroupSelected={handleGroupSelected}
            onUngroupSelected={handleUngroupSelected}
            onAutoLayoutHorizontal={handleAutoLayoutHorizontal}
            onAutoLayoutVertical={handleAutoLayoutVertical}
            onCollapseAll={handleCollapseAll}
            onExpandAll={handleExpandAll}
            onClose={() => setContextMenu(null)}
            screenToFlowPosition={reactFlowInstance.current?.screenToFlowPosition}
          />
        )}

        {/* Quick-Connect Popup - shown when dragging connection for 3+ seconds */}
        {quickConnectState && (
          <QuickConnectPopup
            state={quickConnectState}
            handlePosition={quickConnectHandlePosition}
            compatibleNodes={quickConnectCompatibleNodes}
            onNodeSelect={handleQuickConnectNodeSelect}
            onClose={handleQuickConnectClose}
            onSearchChange={setQuickConnectSearchQuery}
          />
        )}

        {/* The canvas's toolbar, the same at every width: the palette, the
            properties, Run (or Stop), and the log. A macro's own actions join
            it while one is open. */}
        <div className={`oaiy-toolbar${tightToolbar ? ' tight' : ''}`} role="toolbar" aria-label="Canvas">
          <button
            type="button"
            onClick={() => setPalette(!paletteOpen)}
            className={`oaiy-tool${paletteOpen ? ' on' : ''}`}
            aria-pressed={paletteOpen}
            aria-label={paletteOpen ? 'Hide the node palette' : 'Show the node palette'}
            title={paletteOpen ? 'Hide the nodes' : 'Add a node: click one, or drag it onto the canvas'}
          >
            <Plus size={16} />
            <span>Nodes</span>
          </button>
          <button
            type="button"
            onClick={() => setProps(!propsOpen)}
            className={`oaiy-tool${propsOpen ? ' on' : ''}`}
            aria-pressed={propsOpen}
            aria-label={propsOpen ? 'Hide the properties' : 'Show the properties'}
            title={selectedNodes.length === 0 ? 'Properties: select a node to set it up' : 'The selected node’s properties'}
          >
            <SlidersHorizontal size={16} />
            <span>Properties</span>
          </button>
          <span className="sep" aria-hidden="true" />

          {isMacro && (
            <span className="oaiy-pill accent" title="This flow is a macro: other flows use it as a node">
              <Layers size={11} /> macro
            </span>
          )}
          {isMacro && isMacroModified && onSaveMacro && (
            <button type="button" onClick={onSaveMacro} className="btn" title="Save the macro's changes" aria-label="Save macro changes">
              <Save size={14} /> Save
            </button>
          )}
          {isMacro && isMacroModified && onRevertMacro && (
            <button type="button" onClick={onRevertMacro} className="btn btn-warning" title="Go back to the original macro" aria-label="Revert to original macro">
              <RotateCcw size={14} /> Revert
            </button>
          )}

          {isRunning ? (
            <button type="button" onClick={stopWorkflow} className="btn btn-danger solid" aria-label="Stop Workflow" title="Stop the flow">
              <Square size={13} fill="currentColor" /> Stop
            </button>
          ) : isMacro ? (
            <button
              type="button"
              onClick={onRunMacro}
              disabled={!onRunMacro}
              className="btn btn-primary"
              aria-label="Run Macro"
              title="Run this macro with inputs of your own"
            >
              <Play size={13} fill="currentColor" /> Run macro
            </button>
          ) : (
            <button
              type="button"
              onClick={runWorkflow}
              disabled={nodes.length === 0}
              className="btn btn-primary"
              aria-label={nodes.length === 0 ? 'Run Workflow (disabled — empty flow)' : 'Run Workflow'}
              title={nodes.length === 0 ? 'Add at least one node before running' : 'Run the flow'}
            >
              <Play size={13} fill="currentColor" /> Run
            </button>
          )}

          <span className="sep" aria-hidden="true" />
          <button
            type="button"
            onClick={() => setLog(!logOpen)}
            className={`oaiy-tool${logOpen ? ' on' : ''}`}
            aria-pressed={logOpen}
            aria-label={logOpen ? 'Hide the log' : 'Show the log'}
            title="The execution log"
          >
            <ScrollText size={16} />
            <span>Log</span>
            {logs.length > 0 && !logOpen && <em>{logs.length > 99 ? '99+' : logs.length}</em>}
          </button>
        </div>

        {/* A package's flow: which package, and its reload and close. */}
        {packageMode && (
          <div className="oaiy-canvas-banner">
            <PackageIcon size={14} />
            <span>{packageMode.manifest.name}</span>
            <small>v{packageMode.manifest.version}</small>
            {onReloadPackage && (
              <button type="button" onClick={onReloadPackage} className="oaiy-icon-btn sm" title="Reload package (Ctrl+Shift+R)" aria-label="Reload package">
                <RefreshCw size={13} />
              </button>
            )}
            {onClosePackage && (
              <button type="button" onClick={onClosePackage} className="oaiy-icon-btn sm" title="Close package" aria-label="Close package">
                <X size={13} />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Right: the inspector, the palette's sibling — the selected node's
          properties above the execution log. */}
      {inspectorOpen && (
        <aside className="oaiy-side right" aria-label="Inspector">
          <div className="oaiy-side-split">
            {propsOpen && (
              <section className="grow">
                <div className="oaiy-side-head">
                  <h2>
                    Properties
                    {selectedNodes.length > 1 && <small>{selectedNodes.length} selected</small>}
                  </h2>
                  <div className="oaiy-side-tools">
                    <button type="button" onClick={() => setProps(false)} className="oaiy-icon-btn" title="Hide the properties" aria-label="Hide the properties">
                      <X size={15} />
                    </button>
                  </div>
                </div>
                <div className="oaiy-side-body">
                  {selectedNodes.length > 1 ? (
                    <div className="flex flex-col gap-2 p-3">
                      <p className="oaiy-help faint px-1">Ctrl+C copies them, Delete removes them. Pick one to set it up.</p>
                      <ul className="m-0 flex list-none flex-col gap-1 p-0">
                        {selectedNodes.map((node) => {
                          const def = node.data.__definition as { name?: string; color?: string; icon?: string } | undefined;
                          const nodeName = def?.name || node.type || 'Unknown';
                          const nodeColor = def?.color || 'slate';
                          return (
                            <li key={node.id}>
                              <button
                                type="button"
                                className="oaiy-node-item w-full text-left"
                                onClick={() => {
                                  // Deselect all others and select just this one
                                  setNodes(nds => nds.map(n => ({ ...n, selected: n.id === node.id })));
                                }}
                              >
                                <span className={`h-6 w-1.5 shrink-0 rounded-full ${SWATCH_CLASS[nodeColor] ?? SWATCH_CLASS.slate}`} />
                                <span className="oaiy-node-text">
                                  <strong>{nodeName}</strong>
                                  <small className="font-mono">{node.id}</small>
                                </span>
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  ) : (
                    <PropertiesPanel
                      selectedNode={selectedNodes[0] || null}
                      updateNodeData={updateNodeData}
                    />
                  )}
                </div>
              </section>
            )}
            {logOpen && (
              <section className={`log${propsOpen ? '' : ' alone'}`}>
                <LogConsole logs={logs} onClear={clearLogs} onClose={() => setLog(false)} />
              </section>
            )}
          </div>
        </aside>
      )}

      {/* ComfyUI Workflow Configuration Dialog (rendered here to escape React Flow transform context) */}
      {comfyWorkflowDialogState && (
        <ComfyUIWorkflowDialog
          analysis={comfyWorkflowDialogState.analysis}
          onConfirm={handleComfyWorkflowDialogConfirm}
          onCancel={handleComfyWorkflowDialogCancel}
        />
      )}

      {/* Run workflow modal - review/modify inputs before running */}
      <RunWorkflowModal
        isOpen={showRunModal}
        nodes={nodes}
        onRun={handleRunModalConfirm}
        onCancel={closeRunModal}
      />

      {/* Service startup dialog for package flows */}
      {packageMode && showServiceDialog && (
        <ServiceStartupDialog
          isOpen={showServiceDialog}
          packageId={packageMode.manifest.id}
          manifest={packageMode.manifest}
          sourcePath={packageMode.sourcePath}
          onServicesReady={proceedAfterServiceDialog}
          onCancel={closeServiceDialog}
          onSkip={proceedAfterServiceDialog}
        />
      )}
    </div>
  );
}
