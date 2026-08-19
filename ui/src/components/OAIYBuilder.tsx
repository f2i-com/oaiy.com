import { useCallback, useRef, useMemo, useState, useEffect } from 'react';
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
import PropertiesPanel, { SWATCH_CLASS } from './panels/PropertiesPanel';
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
import ConfirmDialog from './ui/ConfirmDialog';
import RunWorkflowModal, { hasInputNodes } from './ui/RunWorkflowModal';
import ComfyUIWorkflowDialog, { type ComfyUIWorkflowConfig } from 'oaiy-core/modules/core-image/ui/ComfyUIWorkflowDialog';
import { ServiceStartupDialog } from './PackageManager/ServiceStartupDialog';
import { CanvasContextMenu } from './OAIYBuilder/CanvasContextMenu';
import { makeIsValidConnection } from '../utils/edgeTypeValidation';
import { useTheme } from '../contexts/ThemeContext';
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
const CONNECTION_LINE_STYLE = { stroke: '#3b82f6', strokeWidth: 3 } as const;
const PRO_OPTIONS = { hideAttribution: true } as const;
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
  // For showing DataViewer within the builder frame
  showDataViewer?: boolean;
  dataViewerComponent?: React.ReactNode;
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
  showDataViewer = false,
  dataViewerComponent,
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
    if (!packageMode || !onReloadPackage) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        onReloadPackage();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [packageMode, onReloadPackage]);


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
    fitView: (options?: { padding?: number; duration?: number }) => void;
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
          reactFlowInstance.current.fitView({ padding: 0.2, duration: 0 });
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

  const [paletteOpen, setPaletteOpen] = useState(false);
  // Below lg the properties panel is undocked, and until now it had NO mobile
  // counterpart at all -- `hidden lg:flex` and nothing else -- so a phone user
  // could add a node and run a flow but could never set an endpoint, a prompt
  // or a model on it. This is that counterpart: the same panel in a sheet.
  const [propsSheetOpen, setPropsSheetOpen] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);
  // Per-panel collapse state, persisted across reloads. The three
  // builder side-panels (palette, properties, execution log) each have
  // an independent visibility flag so power users can reclaim screen
  // real-estate without going full-screen. Defaults: palette + props
  // visible, log visible. Reads happen once at mount via the lazy
  // initializer; toggles write through to localStorage.
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
  const [paletteCollapsed, setPaletteCollapsed] = useState<boolean>(() =>
    loadBoolPref('oaiy.ui.paletteCollapsed', false));
  const togglePaletteCollapsed = useCallback(() =>
    setPaletteCollapsed((p) => { saveBoolPref('oaiy.ui.paletteCollapsed', !p); return !p; }), []);
  const [propertiesVisible, setPropertiesVisible] = useState<boolean>(() =>
    loadBoolPref('oaiy.ui.propertiesVisible', true));
  const togglePropertiesVisible = useCallback(() =>
    setPropertiesVisible((p) => { saveBoolPref('oaiy.ui.propertiesVisible', !p); return !p; }), []);
  const [logPanelVisible, setLogPanelVisible] = useState<boolean>(() =>
    loadBoolPref('oaiy.ui.logPanelVisible', true));
  // Mirror the log toggle through the same persistence helper so the
  // existing close button updates localStorage too.
  useEffect(() => { saveBoolPref('oaiy.ui.logPanelVisible', logPanelVisible); }, [logPanelVisible]);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [showMiniMap, setShowMiniMap] = useState(true); // Toggle minimap visibility
  // The minimap mask must follow the theme — a hardcoded dark mask reads as
  // a heavy navy block over the white canvas in light mode.
  const { resolvedTheme } = useTheme();

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
        const position = reactFlowInstance.current.screenToFlowPosition({
          x: window.innerWidth / 2 - 128,
          y: window.innerHeight / 2,
        });
        const k = (clickAddOffsetRef.current++ % 8) * 36;
        addNode(type, { x: position.x + k, y: position.y + k });
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
        const position = reactFlowInstance.current.screenToFlowPosition({
          x: window.innerWidth / 2 - 128,
          y: window.innerHeight / 2,
        });
        addNode('macro' as NodeType, position, extraData);
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

  // MiniMap mask must follow the theme; memoize so the string only changes
  // when the resolved theme does (recreated inline it churned MiniMap props).
  const miniMapMaskColor = useMemo(
    () => (resolvedTheme === 'dark' ? 'rgba(15, 23, 42, 0.8)' : 'rgba(226, 232, 240, 0.7)'),
    [resolvedTheme],
  );

  return (
    <div className="flex h-full w-full overflow-hidden" style={{ backgroundColor: 'rgb(var(--bg-primary))' }} onKeyDown={onKeyDown} tabIndex={0}>
      {/* Left Panel - Node Palette (hidden on mobile by default, and hidden when showing DataViewer)
          When collapsed, only a thin rail with an expand chevron remains
          so the user can recover the canvas width and still get back. */}
      {!showDataViewer && (
        paletteCollapsed ? (
          <div className="hidden lg:flex lg:h-full lg:w-8 lg:flex-shrink-0 border-r border-slate-200 dark:border-slate-700 bg-white/30 dark:bg-slate-900/30">
            <button
              type="button"
              onClick={togglePaletteCollapsed}
              className="w-full h-full flex flex-col items-center justify-start pt-3 gap-2 text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Show node palette"
              aria-label="Show node palette"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <span className="text-[10px] [writing-mode:vertical-rl] tracking-wider uppercase select-none">Palette</span>
            </button>
          </div>
        ) : (
          <div className="hidden lg:flex lg:h-full lg:flex-shrink-0 relative">
            <NodePalette
              onAddNode={handleAddNode}
              onAddNodeAtPosition={handleAddNodeAtPosition}
              isOpen={true}
              onClose={() => { }}
              macros={macros}
              onAddMacro={handleAddMacro}
              onAddMacroAtPosition={handleAddMacroAtPosition}
              activePackageId={activePackageId}
              packageNodes={packageNodes}
            />
            <button
              type="button"
              onClick={togglePaletteCollapsed}
              className="absolute top-2 right-1 z-10 p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              title="Collapse node palette"
              aria-label="Collapse node palette"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          </div>
        )
      )}

      {/* Mobile Properties sheet — the counterpart to the `hidden lg:flex`
          dock on the right. A bottom sheet rather than a side drawer: a side
          drawer at 390px covers two thirds of the canvas, and the thing being
          configured is the node you just tapped, which you want to keep seeing.
          dvh, not vh, so iOS's collapsing URL bar cannot push the actions off
          the bottom. */}
      {!showDataViewer && propsSheetOpen && (
        <div className="lg:hidden">
          <div
            className="fixed inset-0 bg-black/50 z-40"
            onClick={() => setPropsSheetOpen(false)}
            aria-hidden="true"
          />
          <div
            className="fixed inset-x-0 bottom-0 z-50 flex flex-col max-h-[70dvh] rounded-t-2xl border-t border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-2xl"
            role="dialog"
            aria-label="Node properties"
          >
            <div className="flex items-center justify-between px-3 py-2 border-b border-slate-200 dark:border-slate-800 flex-shrink-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="h-1 w-8 rounded-full bg-slate-300 dark:bg-slate-600" aria-hidden="true" />
                <span className="text-sm font-medium truncate">Properties</span>
              </div>
              <button
                type="button"
                onClick={() => setPropsSheetOpen(false)}
                className="btn btn-ghost btn-icon"
                aria-label="Close properties"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            {/* The panel is position-agnostic (`flex flex-col h-full`), so it
                needs no change to live here — only somewhere to scroll. */}
            <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain">
              <PropertiesPanel
                selectedNode={selectedNodes[0] || null}
                updateNodeData={updateNodeData}
              />
            </div>
          </div>
        </div>
      )}

      {/* Mobile Node Palette (drawer) */}
      {!showDataViewer && (
        <div className="lg:hidden">
          <NodePalette
            onAddNode={handleAddNode}
            onAddNodeAtPosition={handleAddNodeAtPosition}
            isOpen={paletteOpen}
            onClose={() => setPaletteOpen(false)}
            macros={macros}
            onAddMacro={handleAddMacro}
            onAddMacroAtPosition={handleAddMacroAtPosition}
            activePackageId={activePackageId}
            packageNodes={packageNodes}
          />
        </div>
      )}

      {/* Center - Canvas or Data Viewer */}
      <div
        ref={reactFlowWrapper}
        className="flex-1 h-full relative"
        onClick={handleCanvasClick}
        onContextMenu={(e) => e.preventDefault()}
      >
        {/* Loading overlay for flow transitions */}
        {flowTransitioning && (
          <div className="absolute inset-0 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm z-50 flex items-center justify-center animate-fadeIn">
            <div className="flex flex-col items-center gap-4">
              <div className="w-12 h-12 border-4 rounded-full animate-spin" style={{ borderColor: 'rgb(var(--accent-primary) / 0.3)', borderTopColor: 'rgb(var(--accent-primary))' }} />
              <div className="text-slate-600 dark:text-slate-300 text-sm font-medium">
                Loading {flowName}...
              </div>
            </div>
          </div>
        )}

        {showDataViewer && dataViewerComponent ? (
          <div className="h-full w-full overflow-hidden" style={{ backgroundColor: 'rgb(var(--bg-primary))' }}>
            {dataViewerComponent}
          </div>
        ) : (
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
                } else {
                  // First time: fit view to show all nodes
                  instance.fitView({ padding: 0.2, duration: 0 });
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
            {/* Empty-flow hint — only on a brand-new flow with no nodes.
                Pointer-events-none so it never gets in the way of the
                canvas; the palette + right-click menu both stay usable
                through it. Mobile gets a tap-friendly variant since the
                drag-and-drop instruction doesn't apply on touch. */}
            {nodes.length === 0 && !flowTransitioning && (
              <div className="pointer-events-none absolute inset-0 z-[5] flex items-center justify-center p-6 select-none">
                <div className="text-center max-w-xs">
                  <div className="mx-auto mb-3 w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: 'rgb(var(--color-bg-tertiary) / 0.6)' }}>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </div>
                  <p className="text-sm font-medium" style={{ color: 'rgb(var(--color-text-secondary))' }}>
                    This flow is empty
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
                    <span className="hidden lg:inline">Drag a node from the palette on the left, or right-click to add one.</span>
                    <span className="lg:hidden">Tap the menu icon to open the node palette.</span>
                  </p>
                </div>
              </div>
            )}
            <Controls
              className="!bg-white dark:!bg-slate-800 !border-slate-300 dark:!border-slate-700 !rounded-lg [&>button]:!bg-white dark:[&>button]:!bg-slate-800 [&>button]:!border-slate-300 dark:[&>button]:!border-slate-700 [&>button]:!text-slate-500 dark:[&>button]:!text-slate-300 [&>button:hover]:!bg-slate-100 dark:[&>button:hover]:!bg-slate-700"
              position="bottom-left"
            />
            {/* MiniMap with toggle */}
            <div className="hidden sm:block absolute bottom-4 right-4 z-10">
              {showMiniMap ? (
                <div className="relative">
                  <button
                    onClick={() => setShowMiniMap(false)}
                    className="absolute -top-1 -right-1 z-20 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-full p-0.5 transition-colors"
                    title="Hide minimap"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                  <MiniMap
                    className="!bg-white dark:!bg-slate-800 !border-slate-300 dark:!border-slate-700 !relative !m-0"
                    nodeColor={miniMapNodeColor}
                    maskColor={miniMapMaskColor}
                  />
                </div>
              ) : (
                <button
                  onClick={() => setShowMiniMap(true)}
                  className="bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg px-2 py-1.5 text-xs flex items-center gap-1.5 transition-colors"
                  title="Show minimap"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  Map
                </button>
              )}
            </div>
          </ReactFlow>
        )}

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

        {/* Mobile Bottom Toolbar - only show when not in data viewer */}
        {!showDataViewer && (
          <div className="lg:hidden absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm rounded-full px-2 py-1.5 border border-slate-300 dark:border-slate-700 shadow-lg" role="toolbar" aria-label="Mobile workflow controls">
            {/* Nodes Button */}
            <button
              onClick={() => setPaletteOpen(true)}
              className="btn btn-ghost btn-icon"
              title="Add Nodes"
              aria-label="Add Nodes"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
            {/* Properties — the only route to a node's settings below lg.
                Disabled rather than hidden when nothing is selected, so the
                control does not appear and vanish as selection changes. */}
            <button
              onClick={() => setPropsSheetOpen(true)}
              disabled={selectedNodes.length === 0}
              className="btn btn-ghost btn-icon disabled:opacity-40"
              title={selectedNodes.length === 0 ? 'Select a node first' : 'Node properties'}
              aria-label="Node properties"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>

            {/* Macro indicator for mobile */}
            {isMacro && (
              <div className="flex items-center justify-center w-8 h-8 bg-violet-100 dark:bg-violet-900/50 border border-violet-400 dark:border-violet-600 rounded-full" title="This is a macro">
                <svg className="w-4 h-4 text-violet-700 dark:text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                </svg>
              </div>
            )}

            {/* Run/Stop Button */}
            {isRunning ? (
              <button
                onClick={stopWorkflow}
                className="btn btn-danger btn-icon rounded-full"
                title="Stop Workflow"
                aria-label="Stop Workflow"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                </svg>
              </button>
            ) : isMacro ? (
              <button
                onClick={runWorkflow}
                className="btn btn-icon rounded-full bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 cursor-not-allowed"
                title="Macros cannot be run directly"
                aria-label="Macros cannot be run directly"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
              </button>
            ) : nodes.length === 0 ? (
              // Empty flow → disabled button with a tooltip explaining why,
              // saves the user a click+toast roundtrip vs the "Cannot run
              // empty workflow" warning that fires inside runWorkflow.
              <button
                disabled
                className="btn btn-icon rounded-full bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 cursor-not-allowed"
                title="Add at least one node before running"
                aria-label="Run Workflow (disabled — empty flow)"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
              </button>
            ) : (
              <button
                onClick={runWorkflow}
                className="btn btn-primary btn-icon rounded-full"
                title="Run Workflow"
                aria-label="Run Workflow"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
              </button>
            )}

            {/* Logs Button */}
            <button
              onClick={() => setLogsOpen(true)}
              className="btn btn-ghost btn-icon relative"
              title="View Logs"
              aria-label="View Logs"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              {logs.length > 0 && (
                <span className="absolute -top-1 -right-1 badge badge-blue text-[9px] min-w-4 h-4 justify-center">
                  {logs.length > 9 ? '9+' : logs.length}
                </span>
              )}
            </button>
          </div>
        )}

        {/* Desktop Floating Action Bar - only show when not in data viewer */}
        {!showDataViewer && (
          <div className={`hidden lg:flex absolute right-4 z-10 items-center gap-3 ${packageMode ? 'top-8' : 'top-4'}`} role="toolbar" aria-label="Workflow controls">
            {/* Macro controls - only show when modified */}
            {isMacro && isMacroModified && (
              <div className="flex items-center gap-2">
                {/* Save Changes button */}
                {onSaveMacro && (
                  <button
                    onClick={onSaveMacro}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-500 text-white px-3 py-2 rounded-lg shadow-lg text-sm font-medium"
                    title="Save macro changes"
                    aria-label="Save macro changes"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                    </svg>
                    Save
                  </button>
                )}
                {/* Revert button */}
                {onRevertMacro && (
                  <button
                    onClick={onRevertMacro}
                    className="flex items-center gap-2 bg-amber-600 hover:bg-amber-500 text-white px-3 py-2 rounded-lg shadow-lg text-sm font-medium"
                    title="Revert to original macro"
                    aria-label="Revert to original macro"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Revert
                  </button>
                )}
              </div>
            )}
            {/* Run/Stop Button */}
            {isRunning ? (
              <button
                onClick={stopWorkflow}
                className="btn btn-danger btn-lg shadow-lg hover:shadow-red-500/25"
                aria-label="Stop Workflow"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                </svg>
                Stop
              </button>
            ) : isMacro ? (
              <button
                onClick={onRunMacro}
                className="btn btn-lg shadow-lg bg-violet-600 hover:bg-violet-500 text-white hover:shadow-violet-500/25"
                aria-label="Run Macro"
                title="Run this macro with custom inputs"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
                Run Macro
              </button>
            ) : nodes.length === 0 ? (
              <button
                disabled
                className="btn btn-lg bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 cursor-not-allowed"
                title="Add at least one node before running"
                aria-label="Run Workflow (disabled — empty flow)"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
                Run
              </button>
            ) : (
              <button
                onClick={runWorkflow}
                className="btn btn-primary btn-lg shadow-lg hover:shadow-blue-500/25"
                aria-label="Run Workflow"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
                Run
              </button>
            )}
          </div>
        )}

        {/* Package Mode Banner - thin bar at top */}
        {packageMode && (
          <div className="absolute top-0 left-0 right-0 z-20 flex justify-center pointer-events-none">
            <div className="bg-purple-600/95 backdrop-blur-sm rounded-b-md shadow-lg px-3 py-1 flex items-center gap-2 pointer-events-auto">
              <svg className="w-3.5 h-3.5 text-purple-200 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
              <span className="text-white text-xs font-medium">{packageMode.manifest.name}</span>
              <span className="text-purple-200 text-[10px]">v{packageMode.manifest.version}</span>
              {onReloadPackage && (
                <button
                  onClick={onReloadPackage}
                  className="p-0.5 hover:bg-purple-500 rounded transition-colors text-purple-200 hover:text-white"
                  title="Reload package (Ctrl+Shift+R)"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              )}
              {onClosePackage && (
                <button
                  onClick={onClosePackage}
                  className="p-0.5 hover:bg-purple-500 rounded transition-colors text-purple-200 hover:text-white"
                  title="Close package"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        )}

      </div>

      {/* Right Panel - Properties & Log Console (stacked) */}
      {!showDataViewer && (
        <div className="hidden lg:flex flex-col h-full border-l border-slate-200 dark:border-slate-700 w-72 xl:w-80 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm z-20 transition-all duration-300">

          {/* Properties Panel or Selection List (Top, Flex-Grow)
              The collapse button lives in a thin header bar that's
              visible in both states so you always have a target to
              click. When collapsed, only the header remains and the
              log console takes the freed vertical space. */}
          {propertiesVisible ? (
            <div className="flex-1 min-h-0 overflow-hidden border-b border-slate-200 dark:border-slate-700/50 flex flex-col">
              <div className="flex items-center justify-between px-2 py-1 bg-slate-100/50 dark:bg-slate-800/30 border-b border-slate-200 dark:border-slate-700/50 flex-shrink-0">
                <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider px-1">
                  Properties
                </span>
                <button
                  type="button"
                  onClick={togglePropertiesVisible}
                  className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
                  title="Collapse properties"
                  aria-label="Collapse properties"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>
              <div className="flex-1 min-h-0 overflow-hidden">
            {selectedNodes.length > 1 ? (
              // Multi-selection list
              <div className="flex flex-col h-full bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
                {/* Header */}
                <div className="p-3 border-b border-slate-200 dark:border-slate-700/50 flex items-center gap-2 bg-slate-100/50 dark:bg-slate-800/30">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                  </svg>
                  <div>
                    <h2 className="font-bold text-slate-700 dark:text-slate-100 text-sm">{selectedNodes.length} Nodes Selected</h2>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400">Ctrl+C to copy, Delete to remove</div>
                  </div>
                </div>
                {/* Node list */}
                <div className="flex-1 overflow-y-auto p-2 space-y-1 custom-scrollbar">
                  {selectedNodes.map((node) => {
                    const def = node.data.__definition as { name?: string; color?: string; icon?: string } | undefined;
                    const nodeName = def?.name || node.type || 'Unknown';
                    const nodeColor = def?.color || 'slate';
                    return (
                      <div
                        key={node.id}
                        className="flex items-center gap-2 px-3 py-2 rounded-md bg-slate-100 dark:bg-slate-800/50 hover:bg-slate-200 dark:hover:bg-slate-700/50 transition-colors cursor-pointer"
                        onClick={() => {
                          // Deselect all others and select just this one
                          setNodes(nds => nds.map(n => ({ ...n, selected: n.id === node.id })));
                        }}
                      >
                        <div className={`w-2 h-6 rounded-full ${SWATCH_CLASS[nodeColor] ?? 'bg-slate-500'}`} />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm text-slate-700 dark:text-slate-200 truncate">{nodeName}</div>
                          <div className="text-[10px] text-slate-500 dark:text-slate-400 font-mono truncate">{node.id}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <PropertiesPanel
                selectedNode={selectedNodes[0] || null}
                updateNodeData={updateNodeData}
              />
            )}
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={togglePropertiesVisible}
              className="flex items-center justify-between px-3 py-1.5 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-xs flex-shrink-0"
              title="Show properties"
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <span>Properties</span>
              </span>
              {selectedNodes.length > 0 && (
                <span className="text-[10px] text-slate-400 dark:text-slate-500">
                  ({selectedNodes.length} selected)
                </span>
              )}
            </button>
          )}

          {/* Log Console (Bottom, Fixed or Flex)
              When properties is also collapsed, drop the h-1/3 cap
              and let the log flex into the freed space. */}
          {logPanelVisible ? (
            <div className={`${propertiesVisible ? 'h-1/3 min-h-[200px]' : 'flex-1 min-h-0'} flex flex-col border-t border-slate-300 dark:border-slate-700 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] dark:shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.3)]`}>
              {/* Re-using existing LogConsole component but embedded */}
              <LogConsole
                logs={logs}
                onClear={clearLogs}
                isOpen={true}
                onClose={() => setLogPanelVisible(false)}
                className="h-full !border-0 !rounded-none" // Override generic modal/panel styles if needed
              />
            </div>
          ) : (
            /* Collapsed log panel - small expand button */
            <button
              onClick={() => setLogPanelVisible(true)}
              className="flex items-center gap-2 px-3 py-1.5 bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-xs"
              title="Show execution log"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
              <span>Execution Log</span>
              <span className="text-slate-500 dark:text-slate-400">({logs.length})</span>
            </button>
          )}
        </div>
      )}

      {/* Mobile Log Console (drawer) */}
      {!showDataViewer && (
        <div className="lg:hidden">
          <LogConsole
            logs={logs}
            onClear={clearLogs}
            isOpen={logsOpen}
            onClose={() => setLogsOpen(false)}
          />
        </div>
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
