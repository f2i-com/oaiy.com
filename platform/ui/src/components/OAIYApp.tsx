import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import OAIYBuilder from './OAIYBuilder';
import FlowsSidebar from './panels/FlowsSidebar';
import { createLogger } from '../utils/logger';

const logger = createLogger('OAIYApp');
import DataViewer from './panels/DataViewer';
import SettingsPanel, { type SettingsPage } from './panels/SettingsPanel';
import QueuePanel, { QueueCount } from './panels/QueuePanel';
import ConfirmDialog from './ui/ConfirmDialog';
import MacroRunnerModal from './ui/MacroRunnerModal';
import { useProject } from '../hooks/useProject';
import { useToast } from './Toast';
import { usePackageWatcher } from '../hooks/usePackageWatcher';
import { usePackageNodes } from '../hooks/usePackageNodes';
import { usePackageManager, type LoadedPackage, type ActivePackageFlow } from '../hooks/usePackageManager';
import { JobQueueProvider, useJobQueue } from '../contexts/JobQueueContext';
import { ConfirmDialogProvider } from '../hooks/useConfirmDialog';
import ImportMenu from './ImportMenu';
import { ShellSidebar, ShellTopbar, ShellIconAction, ShellDock, ShellSections, shellNavItems, type EditorSection } from './chrome/ShellChrome';
import { Bot, HelpCircle, PanelLeft, Plus, Share2 } from 'lucide-react';
import AgentToolDialog from './dialogs/AgentToolDialog';
import NewFlowDialog from './dialogs/NewFlowDialog';
import { useTheme } from '../contexts/ThemeContext';
import { useMediaQuery } from '../hooks/useMediaQuery';
import {
  DESKTOP_API_BASE,
  getDesktopInfo,
  subscribeDesktopStatus,
  type DesktopInfo,
} from '../lib/desktopDetection';
import { subscribeStorageQuota } from '../lib/storageQuota';
import type { WorkflowGraph, GraphNode, Flow, LocalNetworkPermissionRequest, LocalNetworkPermissionResponse } from 'oaiy-core';
import LocalNetworkPermissionDialog from './dialogs/LocalNetworkPermissionDialog';
import { TrustDialog, DependencyDialog, PackageBrowser } from './PackageManager';
import ShareFlowDialog from './dialogs/ShareFlowDialog';
import { useBackendIntegration } from '../hooks/useBackendIntegration';
import type { QueuedRun, RunOutcome, FlowSnapshot } from '../lib/backendDispatcher';
import WelcomeWizard from './wizards/WelcomeWizard';
import { hasCompletedWizard } from '../lib/wizardPrefs';
import { saveService, listAllServices } from '../utils/serviceRegistry';
import { sanitizeProjectForExport } from '../utils/ProjectIO';
import type { CustomService } from 'oaiy-core/modules/core-service/examples';
import { v4 as uuidv4 } from 'uuid';

// Re-export types for backward compatibility
export type { LoadedPackage, ActivePackageFlow };

/** Persisted collapse state for the flows rail. */
const FLOWS_RAIL_KEY = 'oaiy.flowsRail';

// Connected FlowsSidebar that accesses JobQueue context
interface ConnectedFlowsSidebarProps extends Omit<React.ComponentProps<typeof FlowsSidebar>, 'isFlowRunning'> {
  flows: Flow[];
}

function ConnectedFlowsSidebar(props: ConnectedFlowsSidebarProps) {
  const { isFlowRunning } = useJobQueue();

  return <FlowsSidebar {...props} isFlowRunning={isFlowRunning} />;
}

// Small invisible bridge — captures the JobQueue context value into a
// ref so code outside the JobQueueProvider subtree (like the dispatcher
// executor in OAIYApp's body) can still reach submitJob/subscribeToJob.
// React's rules-of-hooks won't let us call useJobQueue() in OAIYApp's
// body because the provider wraps it; this bridge is the workaround.
type JobBridge = ReturnType<typeof useJobQueue>;
function JobQueueBridge({ handleRef }: { handleRef: React.MutableRefObject<JobBridge | null> }) {
  const value = useJobQueue();
  // Set on every render so the ref tracks the latest snapshot (the
  // closure object stays stable across renders unless context changes,
  // so this is essentially a one-time write).
  handleRef.current = value;
  return null;
}

export default function OAIYApp() {
  const {
    project,
    activeFlow,
    activeFlowId,
    setActiveFlowId,
    resetCounter,
    createFlow,
    updateFlow,
    deleteFlow,
    duplicateFlow,
    renameFlow,
    setFlowLocalOnly,
    updateFlowGraph,
    getAllFlows,
    exportProject,
    importProject,
    newProject,
    renameProject,
    updateSettings,
    getSettings,
    updateConstant,
    createConstant,
    deleteConstant,
    incrementResetCounter,
    saveAsMacro,
    hasMacroBeenModified,
    saveMacroChanges,
    revertMacroToOriginal,
    reloadMacros,
  } = useProject();

  const { addToast } = useToast();

  // Package nodes management
  const {
    loadPackageNodes,
    unloadPackageNodes,
    getAllPackageNodes,
    loadEmbeddedContent,
    unloadEmbeddedContent,
  } = usePackageNodes();

  // Collapsed-or-not survives a reload. A panel that silently reopens every
  // time you come back isn't collapsible in any useful sense — you'd re-collapse
  // it on every visit. Defaults to open so a first-time workspace shows its
  // flows; a malformed/absent value reads as open for the same reason.
  const [flowsSidebarOpen, setFlowsSidebarOpen] = useState(() => {
    // Below md this rail is a fixed OVERLAY, not a column, so "expanded"
    // means it covers the canvas AND the topbar — including the nav
    // hamburger, which made the menu unreachable by touch even though it
    // was drawn. From md up it is a docked column and should honour the
    // stored preference like any other week.
    try {
      if (typeof window !== 'undefined' && window.innerWidth < 760) return false;
      return localStorage.getItem(FLOWS_RAIL_KEY) !== 'collapsed';
    } catch {
      // Private mode / storage disabled — not a reason to fail to render.
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(FLOWS_RAIL_KEY, flowsSidebarOpen ? 'expanded' : 'collapsed');
    } catch {
      // Quota or a blocked store: losing the preference is survivable.
    }
  }, [flowsSidebarOpen]);
  // Below md the left rail is off-canvas (see index.css "phone: the rail
  // becomes a drawer"). Above md it is a grid column and this stays false.
  const [navOpen, setNavOpen] = useState(false);
  const isPhone = useMediaQuery('(width < 760px)');
  useEffect(() => { if (!isPhone) setNavOpen(false); }, [isPhone]);
  // The section showing in the main area (Workflows, Data, Queue, Packages,
  // Settings) and, in Settings, its page. Each is a view, not a popup: the
  // canvas stays mounted beneath the others, so coming back to Workflows finds
  // the same flow, viewport and selection.
  const [section, setSection] = useState<EditorSection>('workflows');
  const [settingsPage, setSettingsPage] = useState<SettingsPage>('services');
  const [newFlowOpen, setNewFlowOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  // In OAIY's window: the open flow given to the agent (a tool of its own, or before or instead of one of its tools).
  const [agentToolOpen, setAgentToolOpen] = useState(false);
  // First-run wizard. Auto-opens on the first ever load (no completed
  // flag in localStorage); can be re-opened any time from the header
  // help button. Skipping marks it completed too, so dismissing once
  // sticks.
  const [wizardOpen, setWizardOpen] = useState<boolean>(() => !hasCompletedWizard());
  const [editingFlowName, setEditingFlowName] = useState(false);
  const [flowNameValue, setFlowNameValue] = useState('');
  const flowNameInputRef = useRef<HTMLInputElement>(null);
  // Theme lives in ThemeContext; the topbar toggle just drives it.
  const { resolvedTheme, setTheme, followsOaiy } = useTheme();
  // OAIY Desktop presence feeds the sidebar's engine card AND the dock's LED, so
  // subscribe once here rather than in each.
  const [companion, setDesktop] = useState<DesktopInfo>(getDesktopInfo);
  useEffect(() => subscribeDesktopStatus(setDesktop), []);
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);

  // Package management hook
  const {
    loadedPackages,
    activePackageFlow,
    activePackage,
    activePackageFlowData,
    activePackageMacros,
    pendingPackage,
    pendingDependencies,
    changedPackage,
    getPackagePermissionContext,
    setActivePackageFlow,
    setPendingPackage,
    setPendingDependencies,
    clearChangedPackage,
    handleLoadPackage,
    handleLoadPackageFromPath,
    handlePackageTrustConfirm,
    handleDependencyLoaded,
    handleDependenciesSatisfied,
    handleContinueWithoutDeps,
    handleClosePackage,
    handleReloadPackage,
    handleSelectPackageFlow,
    handlePackageChanged,
  } = usePackageManager({
    projectFlows: project.flows,
    projectSettings: project.settings,
    onShowToast: addToast,
    setActiveFlowId,
    createFlow,
    updateFlow,
    getSettings,
    loadPackageNodes,
    unloadPackageNodes,
    loadEmbeddedContent,
    unloadEmbeddedContent,
  });

  // Macro runner modal state
  const [macroRunnerFlow, setMacroRunnerFlow] = useState<Flow | null>(null);

  // Local network permission dialog state
  const [permissionRequest, setPermissionRequest] = useState<LocalNetworkPermissionRequest | null>(null);
  const permissionResolverRef = useRef<((response: LocalNetworkPermissionResponse) => void) | null>(null);

  // Handle local network permission requests
  const handleLocalNetworkPermission = useCallback((request: LocalNetworkPermissionRequest): Promise<LocalNetworkPermissionResponse> => {
    return new Promise((resolve) => {
      permissionResolverRef.current = resolve;
      setPermissionRequest(request);
    });
  }, []);

  // Handle permission dialog response
  const handlePermissionResponse = useCallback((allowed: boolean, remember: boolean) => {
    if (permissionResolverRef.current) {
      permissionResolverRef.current({ allowed, remember });
      permissionResolverRef.current = null;
    }
    setPermissionRequest(null);
  }, []);

  // Handle flow name editing
  const startEditingFlowName = useCallback(() => {
    if (activeFlow) {
      setFlowNameValue(activeFlow.name);
      setEditingFlowName(true);
      setTimeout(() => flowNameInputRef.current?.select(), 0);
    }
  }, [activeFlow]);

  const finishEditingFlowName = useCallback(() => {
    if (activeFlowId && flowNameValue.trim()) {
      renameFlow(activeFlowId, flowNameValue.trim());
    }
    setEditingFlowName(false);
  }, [activeFlowId, flowNameValue, renameFlow]);

  // Debounce ref for macro auto-save
  const macroAutoSaveRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const MACRO_AUTOSAVE_DEBOUNCE_MS = 2000;
  // Flag to skip auto-save during revert/load operations
  const skipAutoSaveRef = useRef(false);

  // Handle flow graph updates from the builder
  const handleGraphChange = useCallback((graph: WorkflowGraph) => {
    if (activeFlowId) {
      updateFlowGraph(activeFlowId, graph);

      // Auto-save macros with debounce (unless we're in a skip state)
      const flow = project.flows.find(f => f.id === activeFlowId);
      if (flow?.isMacro && !skipAutoSaveRef.current) {
        // Clear pending auto-save
        if (macroAutoSaveRef.current) {
          clearTimeout(macroAutoSaveRef.current);
        }
        // Schedule new auto-save
        macroAutoSaveRef.current = setTimeout(async () => {
          await saveMacroChanges(activeFlowId);
          logger.debug(`Auto-saved macro "${flow.name}"`);
        }, MACRO_AUTOSAVE_DEBOUNCE_MS);
      }
    }
  }, [activeFlowId, updateFlowGraph, project.flows, saveMacroChanges]);

  // A new flow opens on the canvas, whichever section was showing.
  const handleCreateFlow = useCallback((name: string) => {
    createFlow(name);
    setActivePackageFlow(null);
    setSection('workflows');
  }, [createFlow, setActivePackageFlow]);

  // Loading a package's flow (from Packages, or its trust dialog) shows it.
  useEffect(() => {
    if (activePackageFlow) setSection((s) => (s === 'packages' ? 'workflows' : s));
  }, [activePackageFlow]);

  // Handle saving a flow as a macro
  const handleSaveAsMacro = useCallback((flowId: string) => {
    const result = saveAsMacro(flowId);
    if (result.success) {
      addToast(result.message, 'success');
    } else {
      addToast(result.message, 'error');
    }
  }, [saveAsMacro, addToast]);

  // Clean up macro auto-save timeout on unmount or when flow changes
  useEffect(() => {
    return () => {
      if (macroAutoSaveRef.current) {
        clearTimeout(macroAutoSaveRef.current);
      }
    };
  }, [activeFlowId]);

  // Handle editing a macro (navigate to it - all macros are now editable)
  const handleEditMacro = useCallback((flowId: string) => {
    setActiveFlowId(flowId);
    // No toast needed - auto-save is silent
  }, [setActiveFlowId]);

  // Handle saving macro changes (called when editing a macro)
  const handleSaveMacroChanges = useCallback(async (flowId: string) => {
    const saved = await saveMacroChanges(flowId);
    if (saved) {
      addToast('Macro changes saved', 'success');
    } else {
      addToast('Failed to save macro changes', 'error');
    }
  }, [saveMacroChanges, addToast]);

  // Handle reverting a macro to its original built-in version
  const handleRevertMacro = useCallback(async (flowId: string) => {
    // Clear any pending auto-save
    if (macroAutoSaveRef.current) {
      clearTimeout(macroAutoSaveRef.current);
      macroAutoSaveRef.current = null;
    }

    // Skip auto-save during revert to prevent re-saving the reverted state
    skipAutoSaveRef.current = true;

    const reverted = await revertMacroToOriginal(flowId);
    if (reverted) {
      // Force OAIYBuilder to reload with the reverted graph
      incrementResetCounter();
      addToast('Macro reverted to original version', 'success');
    } else {
      addToast('Failed to revert macro', 'error');
    }

    // Re-enable auto-save after a delay (enough for the graph to reload)
    setTimeout(() => {
      skipAutoSaveRef.current = false;
    }, 500);
  }, [revertMacroToOriginal, incrementResetCounter, addToast]);

  // Watch for package file changes
  usePackageWatcher({
    loadedPackages,
    checkInterval: 3000, // Check every 3 seconds
    onPackageChanged: handlePackageChanged,
  });

  // Handle selecting a user flow (clears package flow selection)
  const handleSelectUserFlow = useCallback(async (flowId: string) => {
    // Update UI immediately for instant feedback
    skipAutoSaveRef.current = true;
    setActiveFlowId(flowId);
    setActivePackageFlow(null); // Clear package flow selection

    // If switching away from a macro, save any pending changes in the background
    if (activeFlowId && activeFlowId !== flowId) {
      const currentFlow = project.flows.find(f => f.id === activeFlowId);
      if (currentFlow?.isMacro) {
        // Clear any pending auto-save
        if (macroAutoSaveRef.current) {
          clearTimeout(macroAutoSaveRef.current);
          macroAutoSaveRef.current = null;
        }
        // Save in background (don't await - UI already updated)
        saveMacroChanges(activeFlowId);
      }
    }

    // Re-enable auto-save after initial graph load
    setTimeout(() => {
      skipAutoSaveRef.current = false;
    }, 500);
  }, [activeFlowId, project.flows, saveMacroChanges, setActiveFlowId, setActivePackageFlow]);

  // Debug logging for flow display issues
  if (activePackageFlow) {
    logger.debug('Active package flow state', {
      packageId: activePackageFlow.packageId,
      flowId: activePackageFlow.flowId,
      packageFound: !!activePackage,
      packageFlowCount: activePackage?.flows.length,
      flowDataFound: !!activePackageFlowData,
      flowIds: activePackage?.flows.map(f => f.id),
      flowHasGraph: !!activePackageFlowData?.graph,
      flowGraphNodeCount: activePackageFlowData?.graph?.nodes?.length,
      flowGraphEdgeCount: activePackageFlowData?.graph?.edges?.length,
    });
  }
  // The previous "Apply Defaults to All Nodes" handler is gone along
  // with the Settings → Defaults service-picker UI it served. Each
  // service_call node now owns its `data.service` directly via the
  // Properties Panel — there's no project-level default to broadcast.

  // Ref to the JobQueue context value, populated by <JobQueueBridge/>
  // mounted inside the provider below.
  const jobQueueRef = useRef<JobBridge | null>(null);

  // Stable refs that the executeRun closure dereferences — keeps the
  // dispatcher useEffect from restarting on every flow change.
  const projectFlowsRef = useRef(project.flows);
  useEffect(() => { projectFlowsRef.current = project.flows; }, [project.flows]);
  const activeFlowIdRef = useRef(activeFlowId);
  useEffect(() => { activeFlowIdRef.current = activeFlowId; }, [activeFlowId]);

  // Backend integration — share state + dispatcher loop. Mount it once
  // here so it survives flow switches; the dispatcher only fires when
  // a share exists in localStorage AND the user has the toggle on.
  const backend = useBackendIntegration({
    executeRun: async (run: QueuedRun): Promise<RunOutcome> => {
      const jq = jobQueueRef.current;
      if (!jq) {
        return {
          status: 'error',
          error: 'JobQueue not ready — the dispatcher fired before the UI mounted. Try again in a moment.',
        };
      }
      // Resolve which flow to run. Prefer the share's bound flowId
      // (set when the share was created); fall back to the currently
      // active flow for shares loaded from an earlier session that
      // pre-date that field.
      const flowId = backend.share?.flowId ?? activeFlowIdRef.current ?? undefined;
      const flows = projectFlowsRef.current;
      const flow = flowId ? flows.find((f) => f.id === flowId) : undefined;
      if (!flow) {
        return {
          status: 'error',
          error: flowId
            ? `Shared flow "${flowId}" no longer exists in this browser.`
            : 'No active flow to execute. Open a flow before sharing.',
        };
      }
      if (!flow.graph || flow.graph.nodes.length === 0) {
        return { status: 'error', error: `Flow "${flow.name}" is empty (no nodes).` };
      }
      // Submit + subscribe. submitJob returns the new job id; the job
      // runs asynchronously through the existing JobManager pipeline,
      // exactly like a manual Run-button click does.
      const inputs = (run.inputs ?? {}) as Record<string, unknown>;
      addToast(`Remote run #${run.id} → ${flow.name} (${run.reason || 'no reason'})`, 'info');
      let jobId: string;
      try {
        jobId = jq.submitJob(flow.id, flow.graph, inputs, flow.name);
      } catch (e) {
        return { status: 'error', error: `Failed to enqueue job: ${(e as Error).message}` };
      }
      // Wait for completion via subscribe. Resolves on terminal status,
      // falls back to a long timeout so a stuck job doesn't hang the
      // dispatcher loop forever (the backend's run will eventually be
      // marked stale by its own cleanup).
      return new Promise<RunOutcome>((resolve) => {
        const TIMEOUT_MS = 10 * 60 * 1000; // 10 min
        const timer = window.setTimeout(() => {
          // Unsubscribe BEFORE aborting so the 'aborted' state change (notified
          // synchronously by abort()) can't re-enter this subscriber and overwrite
          // the timeout message. abort() frees the capacity-1 queue slot — otherwise
          // the abandoned job blocks every subsequent run. Cooperative: it propagates
          // the AbortSignal into the runtime; a pure synchronous hang can't be killed.
          unsub();
          jq.jobManager.abort(jobId);
          resolve({ status: 'error', error: 'Run exceeded 10 min timeout in browser.' });
        }, TIMEOUT_MS);
        const unsub = jq.subscribeToJob(jobId, (job) => {
          if (job.status === 'completed') {
            window.clearTimeout(timer);
            unsub();
            resolve({ status: 'done', result: (job.result as Record<string, unknown> | undefined) ?? {} });
          } else if (job.status === 'failed' || job.status === 'aborted') {
            window.clearTimeout(timer);
            unsub();
            resolve({
              status: 'error',
              error: job.error || `Job ${job.status}`,
            });
          }
        });
      });
    },
  });

  // Snapshot of the current project as a FlowSnapshot the dispatcher can ship.
  // Excludes localStorage-only run history and other non-portable state.
  //
  // SECURITY: a shared flow can be stored UNENCRYPTED (the password is
  // optional), so the node data MUST be scrubbed of inline secrets exactly
  // like the file-export path does — otherwise an API key pasted straight into
  // a service_call's apiKeyConstant / headers / bodyTemplate would leak as
  // plaintext to anyone with the view URL. sanitizeProjectForExport runs each
  // flow's graph through the same secret-stripping the export uses.
  const flowSnapshotForShare: FlowSnapshot = useMemo(
    () => ({
      flows: sanitizeProjectForExport(project).flows,
      settings: project.settings as unknown as Record<string, unknown>,
    }),
    [project],
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't hijack Ctrl+S / Ctrl+N when the user is typing into a field —
      // they almost certainly mean the browser's built-in save (or the
      // field's own behaviour), not "export the whole project". Without
      // this gate, hitting Ctrl+S inside a text node accidentally
      // downloads a JSON dump instead of saving the field. The flow
      // canvas itself doesn't focus a form element, so the global
      // shortcut still fires there.
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName;
      const editable = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t?.isContentEditable;
      if (editable) return;

      // Lowercase the key so the shortcuts still fire with Caps Lock on or
      // Shift held (e.key reflects the produced character — 'S'/'N' otherwise,
      // which silently no-ops and lets the browser's own Ctrl+S through).
      const key = e.key.toLowerCase();
      // Ctrl/Cmd + S exports the project as a JSON file to the browser's
      // download folder — there's no server-side "save" in the web build.
      if ((e.ctrlKey || e.metaKey) && key === 's') {
        e.preventDefault();
        exportProject();
        addToast(`Project "${project.name}" exported as JSON`, 'success');
      }
      // Ctrl/Cmd + N for new project
      if ((e.ctrlKey || e.metaKey) && key === 'n') {
        e.preventDefault();
        setShowNewProjectDialog(true);
      }
      // Ctrl/Cmd + B collapses/expands the flows rail — the convention every
      // editor uses for its file sidebar, and what both collapse buttons
      // advertise in their tooltip.
      if ((e.ctrlKey || e.metaKey) && key === 'b') {
        e.preventDefault();
        setFlowsSidebarOpen((open) => !open);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [exportProject, newProject, project.name, addToast]);

  // localStorage QuotaExceededError → sticky warning toast.
  //
  // useProject (project autosave + run history) and useWorkflow
  // (per-flow autosave) all wrap their setItem in try/catch — without
  // this listener the failure is dev-warn-only and the user keeps
  // editing for hours unaware that nothing is being persisted. The
  // helper throttles to one event per session, and we pass duration
  // 0 so the toast sticks until the user dismisses it.
  useEffect(() => {
    return subscribeStorageQuota(({ area }) => {
      addToast(
        `Browser storage is full (autosave hit its quota while saving ${area}). Recent edits may not survive a refresh — export the project (Ctrl/⌘+S), then delete flows you no longer need to make room.`,
        'warning',
        0,
      );
    });
  }, [addToast]);

  // The editor's sections: the web rail's entries, or in OAIY's window the
  // tabs at the top. One of them shows at a time, in the main area.
  const navItems = shellNavItems(section, setSection, { queue: <QueueCount /> });
  const newFlow = () => setNewFlowOpen(true);
  const openSettings = (page?: SettingsPage) => {
    if (page) setSettingsPage(page);
    setSection('settings');
  };
  // The rail lists flows: it is there for the canvas and a flow's data.
  const railSection = section === 'workflows' || section === 'data';
  const SECTION_LABEL: Record<EditorSection, string> = { workflows: 'Workflows', data: 'Data', queue: 'Queue', packages: 'Packages', settings: 'Settings' };

  return (
    <JobQueueProvider
      availableFlows={getAllFlows()}
      packageMacros={activePackageMacros}
      projectSettings={project.settings}
      constants={project.constants}
      onLocalNetworkPermission={handleLocalNetworkPermission}
      onUpdateSettings={updateSettings}
      onCreateFlow={createFlow}
      onDeleteFlow={deleteFlow}
      onUpdateFlow={updateFlow}
      onUpdateFlowGraph={updateFlowGraph}
      onReloadMacros={reloadMacros}
    >
    {/* Promise-style confirm replacement for native window.confirm.
        Mounted here so every modal/panel below can call useConfirmDialog()
        — including children of SettingsPanel + ShareFlowDialog. */}
    <ConfirmDialogProvider>
    {/* Captures jobQueue context into a ref the backend dispatcher
        executor (defined above this provider, where useJobQueue can't
        be called directly) can reach. */}
    <JobQueueBridge handleRef={jobQueueRef} />
    <div className={followsOaiy ? 'app-shell in-oaiy-shell' : 'app-shell'}>
      <a className="oaiy-skip" href="#oaiy-main">Skip to the canvas</a>

      {/* In OAIY's window its own sidebar is beside the editor: the sections are tabs in the topbar instead. */}
      {!followsOaiy && (
        <ShellSidebar
          items={navItems}
          navOpen={navOpen}
          isPhone={isPhone}
          onCloseNav={() => setNavOpen(false)}
          onNewFlow={newFlow}
          onOpenSettings={() => openSettings()}
          settingsActive={section === 'settings'}
          companionOnline={companion.available}
          companionDetail={
            companion.available
              ? `Desktop v${companion.version ?? '?'}`
              : 'Browser-only execution'
          }
        />
      )}

      <main id="oaiy-main" className="oaiy-workspace" inert={isPhone && navOpen}>
        <ShellTopbar
          navOpen={navOpen}
          onOpenNav={followsOaiy ? undefined : () => setNavOpen(true)}
          sections={
            followsOaiy ? (
              <ShellSections items={navItems} onNewFlow={newFlow} onOpenSettings={() => openSettings()} settingsActive={section === 'settings'} />
            ) : undefined
          }
          crumb={SECTION_LABEL[section]}
          theme={resolvedTheme}
          onSetTheme={followsOaiy ? undefined : setTheme}
          savedLabel={section === 'workflows' ? 'Saved locally' : undefined}
          chips={
            section === 'workflows' ? (
              <>
                {activePackageFlowData && activePackage ? (
                  <em className="oaiy-chip accent">{activePackage.manifest.name}</em>
                ) : null}
                {activeFlow?.localOnly && <em className="oaiy-chip ok">Local</em>}
                {backend.share && (
                  <em className="oaiy-chip accent">Shared · {backend.dispatchState}</em>
                )}
              </>
            ) : null
          }
          actions={
            <>
              {railSection && (
                <ShellIconAction
                  label="Toggle the flows rail"
                  title="Show or hide the flows (Ctrl+B)"
                  on={flowsSidebarOpen}
                  onClick={() => setFlowsSidebarOpen(!flowsSidebarOpen)}
                >
                  <PanelLeft size={16} />
                </ShellIconAction>
              )}
              {followsOaiy && section === 'workflows' && activeFlow && (
                <ShellIconAction
                  label="Give this flow to the agent"
                  title={`Give "${activeFlow.name}" to OAIY's agent: as a tool of its own, or before or instead of one of its tools`}
                  on={agentToolOpen}
                  onClick={() => setAgentToolOpen(true)}
                >
                  <Bot size={16} />
                </ShellIconAction>
              )}
              {section === 'workflows' && backend.enabled && (
                <ShellIconAction
                  label={backend.share ? 'Manage share' : 'Share this flow'}
                  title={
                    backend.share
                      ? `Shared · dispatcher ${backend.dispatchState}${
                          backend.dispatchDetail ? ` (${backend.dispatchDetail})` : ''
                        }`
                      : 'Share this flow'
                  }
                  on={!!backend.share}
                  onClick={() => setShareDialogOpen(true)}
                >
                  <Share2 size={16} />
                </ShellIconAction>
              )}
              <ImportMenu
                importProject={importProject}
                onImportFlows={() => { setSection('workflows'); void handleLoadPackage(); }}
                onExportProject={() => {
                  exportProject();
                  addToast(`Project "${project.name}" exported as JSON`, 'success');
                }}
                projectName={project.name}
                onShowToast={addToast}
              />
              <ShellIconAction
                label="Open the welcome wizard"
                title={'Welcome wizard / help\n\nShortcuts:\n  Ctrl/\u2318 + S \u2014 Export project as JSON\n  Ctrl/\u2318 + N \u2014 New project'}
                onClick={() => setWizardOpen(true)}
              >
                <HelpCircle size={16} />
              </ShellIconAction>
            </>
          }
        >
          {activePackageFlowData && activePackage ? (
            <h1>{activePackageFlowData.name}</h1>
          ) : (
            /* project / flow, both reachable — they are not alternatives. The
               project name is always editable in place; the flow name is a real
               button so it can be reached and activated from the keyboard. */
            <>
              <input
                className="oaiy-name oaiy-name-project"
                type="text"
                value={project.name}
                onChange={(e) => renameProject(e.target.value)}
                aria-label="Project name"
              />
              {activeFlow && railSection && (
                <>
                  <span className="oaiy-name-sep" aria-hidden="true">
                    /
                  </span>
                  {editingFlowName ? (
                    <input
                      ref={flowNameInputRef}
                      className="oaiy-name"
                      type="text"
                      value={flowNameValue}
                      onChange={(e) => setFlowNameValue(e.target.value)}
                      onBlur={finishEditingFlowName}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') finishEditingFlowName();
                        if (e.key === 'Escape') setEditingFlowName(false);
                      }}
                      aria-label="Flow name"
                      autoFocus
                    />
                  ) : (
                    <button
                      type="button"
                      className="oaiy-name oaiy-name-flow"
                      onClick={startEditingFlowName}
                      title={`${activeFlow.name} — rename this flow`}
                    >
                      {activeFlow.name}
                    </button>
                  )}
                </>
              )}
            </>
          )}
        </ShellTopbar>

        <section className="oaiy-view">
          <div className="flex h-full w-full min-w-0 min-h-0">
            {railSection && <ConnectedFlowsSidebar
              flows={project.flows}
              activeFlowId={activePackageFlow ? null : activeFlowId}
              onSelectFlow={handleSelectUserFlow}
              onDeleteFlow={deleteFlow}
              onDuplicateFlow={duplicateFlow}
              onRenameFlow={renameFlow}
              onSetFlowLocalOnly={setFlowLocalOnly}
              onSaveAsMacro={handleSaveAsMacro}
              onEditMacro={handleEditMacro}
              onSaveMacro={handleSaveMacroChanges}
              onRevertMacro={handleRevertMacro}
              hasMacroBeenModified={hasMacroBeenModified}
              isOpen={flowsSidebarOpen}
              onClose={() => setFlowsSidebarOpen(false)}
              onOpen={() => setFlowsSidebarOpen(true)}
              loadedPackages={loadedPackages}
              activePackageFlow={activePackageFlow}
              onSelectPackageFlow={handleSelectPackageFlow}
              onClosePackage={handleClosePackage}
            />}
            <div className="flex-1 relative min-h-0 min-w-0 oaiy-canvas-wrap">
            {/* The canvas: laid out and mounted under whichever section shows,
                so its flow, viewport and selection are there on the way back;
                not seen, and not reachable by keyboard, meanwhile. */}
            <div className={`oaiy-canvas-layer${section === 'workflows' ? '' : ' away'}`} inert={section !== 'workflows'} aria-hidden={section !== 'workflows' ? true : undefined}>

          {activePackageFlowData && activePackage ? (
            // Package flow view - show package flow with navigation
            <OAIYBuilder
              key={`package-${activePackageFlow!.packageId}-${activePackageFlow!.flowId}`}
              initialGraph={activePackageFlowData.graph}
              onGraphChange={() => {}} // Package flows are read-only for now
              availableFlows={activePackage.flows.filter(f => f.id !== activePackageFlow!.flowId)}
              packageMacros={activePackage.macros}
              llmEndpoints={project.llmEndpoints}
              projectConstants={project.constants}
              projectSettings={project.settings}
              onUpdateSettings={updateSettings}
              flowId={activePackageFlowData.id}
              flowName={activePackageFlowData.name}
              active={section === 'workflows'}
              onShowToast={addToast}
              // Package mode props - pass the active package info
              packageMode={{
                manifest: activePackage.manifest,
                flow: activePackageFlowData,
                sourcePath: activePackage.sourcePath,
              }}
              // Granted permission set captured at trust-confirm time —
              // OAIYBuilder forwards this into the JobManager submit
              // call so the runtime applies the fail-closed gate.
              packagePermissionContext={getPackagePermissionContext(activePackage.manifest.id)}
              onClosePackage={() => handleClosePackage(activePackage.manifest.id)}
              onReloadPackage={() => handleReloadPackage(activePackage.manifest.id)}
              onLoadPackage={handleLoadPackage}
              // Package nodes - show package nodes when viewing package flow
              activePackageId={activePackage.manifest.id}
              packageNodes={getAllPackageNodes()}
            />
          ) : activePackageFlow && !activePackageFlowData ? (
            // Package is selected but flow data not found - show error
            <div className="grid h-full w-full place-items-center p-6">
              <div className="oaiy-card max-w-md items-center text-center">
                <h2 className="m-0 text-[15px] font-semibold text-content-primary">That flow is not in the package</h2>
                <p className="oaiy-card-text">
                  The package has no flow "{activePackageFlow.flowId}".
                  {activePackage ? ` "${activePackage.manifest.name}" has ${activePackage.flows.length} flow(s).` : ''}
                </p>
                <div className="flex justify-center gap-2">
                  <button
                    onClick={() => {
                      setActivePackageFlow(null);
                      if (project.flows.length > 0) {
                        setActiveFlowId(project.flows[0].id);
                      }
                    }}
                    className="btn"
                  >
                    Close the package
                  </button>
                  {activePackage && activePackage.flows.length > 0 && (
                    <button
                      onClick={() => setActivePackageFlow({ packageId: activePackageFlow.packageId, flowId: activePackage.flows[0].id })}
                      className="btn btn-primary"
                    >
                      Open its first flow
                    </button>
                  )}
                </div>
              </div>
            </div>
          ) : activeFlow ? (
            <OAIYBuilder
              key={`${activeFlowId}-${resetCounter}`}
              initialGraph={activeFlow.graph}
              onGraphChange={handleGraphChange}
              availableFlows={getAllFlows().filter(f => f.id !== activeFlowId)}
              llmEndpoints={project.llmEndpoints}
              projectConstants={project.constants}
              projectSettings={project.settings}
              onUpdateSettings={updateSettings}
              flowId={activeFlow.id}
              flowName={activeFlow.name}
              isMacro={activeFlow.isMacro}
              isMacroModified={activeFlow.isMacro ? hasMacroBeenModified(activeFlow.id) : false}
              onSaveMacro={activeFlow.isMacro ? () => handleSaveMacroChanges(activeFlow.id) : undefined}
              /* Revert only applies to built-in macros (revert to the disk
                 version). User-created macros have no original to revert to,
                 so showing it just errored — gate on isBuiltIn. */
              onRevertMacro={activeFlow.isMacro && activeFlow.isBuiltIn ? () => handleRevertMacro(activeFlow.id) : undefined}
              onRunMacro={activeFlow.isMacro ? () => setMacroRunnerFlow(activeFlow) : undefined}
              active={section === 'workflows'}
              onShowToast={addToast}
              onLoadPackage={handleLoadPackage}
            />
          ) : (
            <div className="bg-dotgrid grid h-full w-full place-items-center p-6">
              <div className="oaiy-canvas-hint static">
                <div>
                  <i><Plus size={18} /></i>
                  <p>{project.flows.length === 0 ? 'No flows yet' : 'No flow open'}</p>
                  <small>
                    {project.flows.length === 0
                      ? 'Make your first flow, then add nodes to it from the palette.'
                      : 'Pick a flow from the rail, or make a new one.'}
                  </small>
                  <button onClick={newFlow} className="btn btn-primary mt-2 pointer-events-auto">
                    <Plus size={14} /> New flow
                  </button>
                </div>
              </div>
            </div>
          )}
            </div>
            {section === 'data' && (
              <DataViewer
                activeFlowId={activePackageFlowData && activePackage ? activePackageFlowData.id : activeFlow?.id}
                packageId={activePackageFlowData && activePackage ? activePackage.manifest.id : undefined}
                flowName={activePackageFlowData && activePackage ? activePackageFlowData.name : activeFlow?.name}
              />
            )}
            {section === 'queue' && (
              <QueuePanel
                onNavigateToFlow={(flowId) => {
                  setActivePackageFlow(null);
                  setActiveFlowId(flowId);
                  setSection('workflows');
                }}
              />
            )}
            {section === 'packages' && (
              <PackageBrowser
                onLoadPackage={handleLoadPackageFromPath}
                loadedPackageIds={new Set(loadedPackages.keys())}
                loaded={Array.from(loadedPackages.values()).map((p) => ({ id: p.manifest.id, name: p.manifest.name, version: p.manifest.version, flows: p.flows.length }))}
                onClosePackage={handleClosePackage}
                onLoadFile={() => void handleLoadPackage()}
              />
            )}
            {section === 'settings' && (
              <SettingsPanel
                page={settingsPage}
                onPageChange={setSettingsPage}
                settings={getSettings()}
                constants={project.constants || []}
                onUpdateSettings={updateSettings}
                onUpdateConstant={updateConstant}
                onCreateConstant={createConstant}
                onDeleteConstant={deleteConstant}
                onShowToast={addToast}
              />
            )}
            </div>
          </div>
        </section>

        {!followsOaiy && <ShellDock
          companionOnline={companion.available}
          endpointLabel={companion.available ? 'Local engine ready' : 'Local engine idle'}
          endpointUrl={backend.share ? backend.share.editUrl : DESKTOP_API_BASE}
          onCopyEndpoint={() => {
            const url = backend.share ? backend.share.editUrl : DESKTOP_API_BASE;
            void navigator.clipboard
              ?.writeText(url)
              .then(() => addToast('Endpoint URL copied.', 'success'))
              .catch(() => addToast('Could not access the clipboard.', 'error'));
          }}
          shared={!!backend.share}
          onManage={() => (backend.enabled ? setShareDialogOpen(true) : openSettings('general'))}
          manageLabel={backend.share ? 'Manage share' : 'Share'}
        />}
      </main>

      {agentToolOpen && activeFlow && (
        <AgentToolDialog flow={activeFlow} onClose={() => setAgentToolOpen(false)} onDone={(message) => addToast(message, 'success')} />
      )}

      <NewFlowDialog
        open={newFlowOpen}
        onClose={() => setNewFlowOpen(false)}
        onCreate={(name) => {
          setNewFlowOpen(false);
          handleCreateFlow(name);
        }}
      />

      {/* New project confirmation dialog */}
      <ConfirmDialog
        isOpen={showNewProjectDialog}
        title="Create New Project"
        message="This will create a new empty project. Any unsaved changes will be lost."
        confirmLabel="Create New"
        cancelLabel="Cancel"
        variant="warning"
        onConfirm={() => {
          setShowNewProjectDialog(false);
          newProject();
          addToast('New project created', 'success');
        }}
        onCancel={() => setShowNewProjectDialog(false)}
      />

      {/* The OAIY Agent panel lived here in the desktop build — removed
          for the web build, see the comment on the header-button slot
          above. External AI clients drive flows via the oaiy-api HTTP
          surface (POST /api/flows/{hash_edit}/runs). */}

      {/* Share Flow Dialog — backend create + URL display + password.
          onCreate is wrapped here to inject the active flowId so the
          dispatcher knows which flow to run when remote calls arrive. */}
      <ShareFlowDialog
        isOpen={shareDialogOpen}
        onClose={() => setShareDialogOpen(false)}
        share={backend.share}
        enabled={backend.enabled}
        onCreate={(snapshot, opts) => backend.createShare(snapshot, { ...opts, flowId: activeFlowId ?? undefined })}
        onForget={backend.forgetShare}
        snapshot={flowSnapshotForShare}
        defaultTitle={project.name}
      />

      {/* First-run wizard. Auto-opens on the very first load (no
          'oaiy.wizard.completed' flag) and re-openable from the help
          icon in the header. Dismissing it any way marks it completed
          so the user doesn't get prompted again. */}
      <WelcomeWizard
        isOpen={wizardOpen}
        onClose={() => setWizardOpen(false)}
        // Every non-built-in service the user already has. The wizard lists
        // them so a returning user can REUSE one for a starter flow instead of
        // re-creating it, and flags presets that are already configured.
        existingServices={listAllServices().filter((s) => !s.isBuiltIn)}
        // API-key constants that already hold a value, so the wizard can say
        // "already set — leave blank to reuse" for a shared key (e.g. add an
        // OpenAI chat service then GPT Image without re-typing OPENAI_API_KEY).
        configuredKeyConstants={(project.constants ?? [])
          .filter((c) => !!c.value && (c.isSecret || c.category === 'api_key'))
          .map((c) => c.key)}
        onSaveService={(svc) => {
          saveService(svc);
          addToast(`Service "${svc.name}" added`, 'success');
        }}
        onSaveApiKey={(constantName, value) => {
          // Wired into project constants as a secret. The runtime resolves
          // it via ctx.getConstant at run time — same as keys typed in
          // Settings → API Keys. Upsert by key so two services that share a
          // constant (e.g. OpenAI chat + GPT Image both use OPENAI_API_KEY)
          // update the one constant instead of creating duplicates.
          const existing = (project.constants ?? []).find((c) => c.key === constantName);
          if (existing) {
            updateConstant(existing.id, { value });
          } else {
            createConstant({
              name: constantName,
              key: constantName,
              value,
              category: 'api_key',
              isSecret: true,
            });
          }
        }}
        onCreateStarterFlow={(svc) => {
          // Build a runnable 3-node starter flow: input_text → service_call →
          // output. The web build folds AI LLM / Image Gen into the generic
          // Service Call node (image_gen isn't registered here), so EVERY
          // service — chat or image — is called the same way. For image
          // services the call returns base64 image data, which the output node
          // renders as a picture. Only the seeded prompt / labels differ.
          //
          // `service_call` isn't in the BuiltinNodeType string-literal union
          // (frozen in oaiy-core/types.ts); the module loader registers it by
          // string at runtime, so we widen the type field via
          // `as GraphNode['type']` without lying about which arm.
          const inputId = `input-${uuidv4().slice(0, 8)}`;
          const serviceId = `service-${uuidv4().slice(0, 8)}`;
          const outputId = `output-${uuidv4().slice(0, 8)}`;
          const isImage = !!svc.nodeTypes?.includes('image_gen') && !svc.nodeTypes?.includes('ai_llm');

          const starterGraph: WorkflowGraph = {
            nodes: [
              {
                id: inputId,
                type: 'input_text',
                position: { x: 80, y: 200 },
                // input_text stores its value under `value` (compiler, canvas
                // node, and Run modal all read data.value).
                data: {
                  label: 'Prompt',
                  value: isImage
                    ? 'A serene mountain lake at sunrise, ultra-detailed, photorealistic.'
                    : 'Tell me a one-line joke.',
                },
              },
              {
                id: serviceId,
                // The service_call compiler resolves endpoint / body / headers
                // / responsePath / apiKeyConstant from the linked service, so
                // just the service id is enough to wire it up.
                type: 'service_call' as GraphNode['type'],
                position: { x: 400, y: 200 },
                data: { service: svc.id, label: svc.name },
              },
              {
                id: outputId,
                type: 'output',
                position: { x: 720, y: 200 },
                data: { label: isImage ? 'Image' : 'Response' },
              },
            ],
            edges: [
              { id: `e-${inputId}-${serviceId}`, source: inputId, sourceHandle: 'text', target: serviceId, targetHandle: 'input' },
              { id: `e-${serviceId}-${outputId}`, source: serviceId, sourceHandle: 'response', target: outputId, targetHandle: 'result' },
            ],
          };
          const flowName = isImage ? `Images with ${svc.name}` : `Chat with ${svc.name}`;

          const starterFlow = createFlow(flowName, starterGraph);
          setActiveFlowId(starterFlow.id);
          // Show the canvas, whichever section was open when the wizard
          // was — otherwise there is just a toast and no flow in sight.
          setSection('workflows');
          addToast(`Created "${starterFlow.name}" — click Run to try it`, 'success');
        }}
      />

      {/* Local Network Permission Dialog */}
      {permissionRequest && (
        <LocalNetworkPermissionDialog
          request={permissionRequest}
          onResponse={handlePermissionResponse}
        />
      )}

      {/* Package Trust Dialog */}
      {pendingPackage && !pendingDependencies && (
        <TrustDialog
          manifest={pendingPackage.manifest}
          onConfirm={handlePackageTrustConfirm}
          onCancel={() => setPendingPackage(null)}
        />
      )}

      {/* Dependency Resolution Dialog */}
      {pendingDependencies && (
        <DependencyDialog
          isOpen={true}
          packageName={pendingDependencies.packageName}
          dependencies={pendingDependencies.dependencies}
          loadedPackages={loadedPackages}
          onPackageLoaded={handleDependencyLoaded}
          onAllLoaded={handleDependenciesSatisfied}
          onContinueAnyway={handleContinueWithoutDeps}
          onCancel={() => {
            setPendingDependencies(null);
            setPendingPackage(null);
          }}
        />
      )}

      {/* Package Changed Dialog */}
      <ConfirmDialog
        isOpen={changedPackage !== null}
        title="Package Updated"
        message={changedPackage ? `The package "${changedPackage.manifest.name}" has been modified. Would you like to reload it to see the changes?` : ''}
        confirmLabel="Reload"
        cancelLabel="Ignore"
        variant="info"
        onConfirm={() => {
          if (changedPackage) {
            handleReloadPackage(changedPackage.packageId);
          }
        }}
        onCancel={() => clearChangedPackage()}
      />

      {/* Macro Runner Modal */}
      {macroRunnerFlow && (
        <MacroRunnerModal
          macro={macroRunnerFlow}
          onClose={() => setMacroRunnerFlow(null)}
          onShowToast={addToast}
        />
      )}
    </div>
    </ConfirmDialogProvider>
    </JobQueueProvider>
  );
}
