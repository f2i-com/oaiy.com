import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import OAIYBuilder from './OAIYBuilder';
import FlowsSidebar from './panels/FlowsSidebar';
import { createLogger } from '../utils/logger';

const logger = createLogger('OAIYApp');
import DataViewer from './panels/DataViewer';
import SettingsPanel from './panels/SettingsPanel';
import QueuePanel from './panels/QueuePanel';
import ConfirmDialog from './ui/ConfirmDialog';
import MacroRunnerModal from './ui/MacroRunnerModal';
import { useProject } from '../hooks/useProject';
import { useToast } from './Toast';
import { usePackageWatcher } from '../hooks/usePackageWatcher';
import { usePackageNodes } from '../hooks/usePackageNodes';
import { usePackageManager, type LoadedPackage, type ActivePackageFlow } from '../hooks/usePackageManager';
import { JobQueueProvider, useJobQueue } from '../contexts/JobQueueContext';
import { ConfirmDialogProvider } from '../hooks/useConfirmDialog';
import ProjectImportButton from './ProjectImportButton';
import { ShellSidebar, ShellTopbar, ShellIconAction, ShellDock } from './chrome/ShellChrome';
import { Activity, HelpCircle, PanelLeft, Settings2, Share2 } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import {
  COMPANION_API_BASE,
  getCompanionInfo,
  subscribeCompanionStatus,
  type CompanionInfo,
} from '../lib/companionDetection';
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

type MainTab = 'builder' | 'data';

// Re-export types for backward compatibility
export type { LoadedPackage, ActivePackageFlow };

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

  const [flowsSidebarOpen, setFlowsSidebarOpen] = useState(true);
  const [settingsPanelOpen, setSettingsPanelOpen] = useState(false);
  // Track which Settings tab to land on. The agent panel's "Manage in
  // Settings…" affordance deep-links to 'models'; the toolbar gear
  // leaves it `undefined` so the panel restores the user's last tab.
  // SettingsPanel's web SettingsTab union — the desktop-only tabs
  // (plugins / api / models) were removed in the web build, so this
  // narrowed to match.
  const [settingsInitialTab, setSettingsInitialTab] = useState<
    'appearance' | 'defaults' | 'apikeys' | 'constants' | 'security' | 'services' | undefined
  >(undefined);
  const [queuePanelOpen, setQueuePanelOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  // First-run wizard. Auto-opens on the first ever load (no completed
  // flag in localStorage); can be re-opened any time from the header
  // help button. Skipping marks it completed too, so dismissing once
  // sticks.
  const [wizardOpen, setWizardOpen] = useState<boolean>(() => !hasCompletedWizard());
  const [editingFlowName, setEditingFlowName] = useState(false);
  const [flowNameValue, setFlowNameValue] = useState('');
  const flowNameInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<MainTab>('builder');
  // Theme lives in ThemeContext; the topbar toggle just drives it.
  const { resolvedTheme, setTheme } = useTheme();
  // Companion presence feeds the sidebar's engine card AND the dock's LED, so
  // subscribe once here rather than in each.
  const [companion, setCompanion] = useState<CompanionInfo>(getCompanionInfo);
  useEffect(() => subscribeCompanionStatus(setCompanion), []);
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
    showPackageBrowser,
    getPackagePermissionContext,
    setActivePackageFlow,
    setPendingPackage,
    setPendingDependencies,
    setShowPackageBrowser,
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

  // Handle creating a new flow
  const handleCreateFlow = useCallback((name: string) => {
    createFlow(name);
  }, [createFlow]);

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
        `Browser storage is full (autosave hit its quota while saving ${area}). Recent edits may not survive a refresh — export the project (Ctrl/⌘+S), then clear run history in Settings → Defaults to make room.`,
        'warning',
        0,
      );
    });
  }, [addToast]);

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
    <div className="app-shell">
      <a className="oaiy-skip" href="#oaiy-main">Skip to the canvas</a>

      <ShellSidebar
        view={activeTab}
        onSelectView={setActiveTab}
        onNewFlow={() => handleCreateFlow('Untitled flow')}
        onOpenQueue={() => setQueuePanelOpen(true)}
        onOpenPlugins={() => setShowPackageBrowser(true)}
        onOpenServices={() => {
          setSettingsInitialTab('services');
          setSettingsPanelOpen(true);
        }}
        onOpenSettings={() => setSettingsPanelOpen(true)}
        settingsActive={settingsPanelOpen}
        companionOnline={companion.available}
        companionDetail={
          companion.available
            ? `Companion v${companion.version ?? '?'}`
            : 'Browser-only execution'
        }
      />

      <main id="oaiy-main" className="oaiy-workspace">
        <ShellTopbar
          crumb={activeTab === 'data' ? 'Data' : 'Workflows'}
          theme={resolvedTheme}
          onSetTheme={setTheme}
          savedLabel={activeTab === 'builder' ? 'Saved locally' : undefined}
          chips={
            activeTab === 'builder' ? (
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
              <ShellIconAction
                label="Toggle the flows rail"
                on={flowsSidebarOpen}
                onClick={() => setFlowsSidebarOpen(!flowsSidebarOpen)}
              >
                <PanelLeft size={16} />
              </ShellIconAction>
              {activeTab === 'builder' && backend.enabled && (
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
              <ShellIconAction
                label="Toggle the job queue"
                on={queuePanelOpen}
                onClick={() => setQueuePanelOpen(!queuePanelOpen)}
              >
                <Activity size={16} />
              </ShellIconAction>
              <ProjectImportButton
                importProject={importProject}
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
              <ShellIconAction label="Settings" onClick={() => setSettingsPanelOpen(true)}>
                <Settings2 size={16} />
              </ShellIconAction>
            </>
          }
        >
          {activePackageFlowData && activePackage ? (
            <h1>{activePackageFlowData.name}</h1>
          ) : editingFlowName ? (
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
              autoFocus
            />
          ) : activeFlow ? (
            <h1
              onDoubleClick={startEditingFlowName}
              title="Double-click to rename this flow"
              style={{ cursor: 'text' }}
            >
              {activeFlow.name}
            </h1>
          ) : (
            <input
              className="oaiy-name"
              type="text"
              value={project.name}
              onChange={(e) => renameProject(e.target.value)}
              aria-label="Project name"
            />
          )}
        </ShellTopbar>

        <section className="oaiy-view">
          <div className="flex h-full w-full min-w-0 min-h-0">
            <ConnectedFlowsSidebar
              flows={project.flows}
              activeFlowId={activePackageFlow ? null : activeFlowId}
              onSelectFlow={handleSelectUserFlow}
              onCreateFlow={handleCreateFlow}
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
              loadedPackages={loadedPackages}
              activePackageFlow={activePackageFlow}
              onSelectPackageFlow={handleSelectPackageFlow}
              onClosePackage={handleClosePackage}
              onLoadPackage={handleLoadPackage}
              onOpenBrowser={() => setShowPackageBrowser(true)}
            />
            <div className="flex-1 relative min-h-0 oaiy-canvas-wrap">

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
              showDataViewer={activeTab === 'data'}
              dataViewerComponent={
                <DataViewer
                  activeFlowId={activePackageFlowData.id}
                  packageId={activePackage.manifest.id}
                  flowName={activePackageFlowData.name}
                />
              }
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
            <div className="h-full flex items-center justify-center bg-slate-50 dark:bg-transparent">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                  <svg className="w-8 h-8 text-red-500 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <h2 className="text-xl font-medium text-slate-700 dark:text-slate-300 mb-2">Flow Not Found</h2>
                <p className="text-slate-500 dark:text-slate-400 mb-4">
                  The selected flow "{activePackageFlow.flowId}" could not be found in the package.
                  {activePackage ? ` The package "${activePackage.manifest.name}" contains ${activePackage.flows.length} flow(s).` : ''}
                </p>
                <div className="flex gap-2 justify-center">
                  {activePackage && activePackage.flows.length > 0 && (
                    <button
                      onClick={() => setActivePackageFlow({ packageId: activePackageFlow.packageId, flowId: activePackage.flows[0].id })}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors"
                    >
                      View First Flow
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setActivePackageFlow(null);
                      if (project.flows.length > 0) {
                        setActiveFlowId(project.flows[0].id);
                      }
                    }}
                    className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                  >
                    Close Package
                  </button>
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
              showDataViewer={activeTab === 'data'}
              dataViewerComponent={
                <DataViewer
                  activeFlowId={activeFlow.id}
                  flowName={activeFlow.name}
                />
              }
              onShowToast={addToast}
              onLoadPackage={handleLoadPackage}
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-dotgrid p-6" style={{ backgroundColor: 'rgb(var(--color-bg-canvas))' }}>
              <div className="text-center max-w-sm">
                <h2 className="font-display text-4xl mb-3" style={{ fontWeight: 400, letterSpacing: '-0.025em', color: 'rgb(var(--color-text-primary))' }}>
                  An empty canvas
                </h2>
                <p className="mb-6 text-sm" style={{ color: 'rgb(var(--color-text-tertiary))' }}>
                  {project.flows.length === 0
                    ? 'Start by creating your first flow — then drag nodes from the palette to build it.'
                    : 'Pick a flow from the sidebar, or start a new one.'}
                </p>
                <button
                  onClick={() => createFlow('New Flow')}
                  className="btn btn-primary btn-md"
                >
                  {project.flows.length === 0 ? 'Create your first flow' : 'Create Flow'}
                </button>
              </div>
            </div>
          )}
            </div>
          </div>
        </section>

        <ShellDock
          companionOnline={companion.available}
          endpointLabel={companion.available ? 'Local engine ready' : 'Local engine idle'}
          endpointUrl={backend.share ? backend.share.editUrl : COMPANION_API_BASE}
          onCopyEndpoint={() => {
            const url = backend.share ? backend.share.editUrl : COMPANION_API_BASE;
            void navigator.clipboard
              ?.writeText(url)
              .then(() => addToast('Endpoint URL copied.', 'success'))
              .catch(() => addToast('Could not access the clipboard.', 'error'));
          }}
          shared={!!backend.share}
          onManage={() => (backend.enabled ? setShareDialogOpen(true) : setSettingsPanelOpen(true))}
          manageLabel={backend.share ? 'Manage share' : 'Share'}
        />
      </main>

      {/* Settings Panel */}
      <SettingsPanel
        isOpen={settingsPanelOpen}
        onClose={() => {
          setSettingsPanelOpen(false);
          setSettingsInitialTab(undefined);
        }}
        settings={getSettings()}
        constants={project.constants || []}
        onUpdateSettings={updateSettings}
        onUpdateConstant={updateConstant}
        onCreateConstant={createConstant}
        onDeleteConstant={deleteConstant}
        onShowToast={addToast}
        initialTab={settingsInitialTab}
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

      {/* Queue Panel */}
      <QueuePanel
        isOpen={queuePanelOpen}
        onClose={() => setQueuePanelOpen(false)}
        onNavigateToFlow={(flowId) => {
          setActiveFlowId(flowId);
          setQueuePanelOpen(false);
        }}
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
          // Switch to Builder if the user is sitting on the Data Viewer when
          // the wizard opens — they'd otherwise just see a toast and wonder
          // where the flow went.
          setActiveTab('builder');
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

      {/* Package Browser Dialog */}
      <PackageBrowser
        isOpen={showPackageBrowser}
        onClose={() => setShowPackageBrowser(false)}
        onLoadPackage={handleLoadPackageFromPath}
        loadedPackageIds={new Set(loadedPackages.keys())}
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
