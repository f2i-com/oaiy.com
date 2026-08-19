import { useState, useCallback, useMemo } from 'react';
import type { Flow, OAIYPackageManifest } from 'oaiy-core';
import ConfirmDialog from '../ui/ConfirmDialog';
import PackageServicesPanel from './PackageServicesPanel';
import { isWebBuild } from '../../lib/platform';

// Browsing the filesystem for installable .oaiy packages is a desktop-only
// capability (the web build has no real filesystem / native unzip). Compute
// once so the package-browser affordances are hidden in the browser, where
// they'd otherwise open an always-empty dialog. Importing a .json flow via
// "Load from File" still works and stays available.
const PACKAGE_BROWSE_AVAILABLE = !isWebBuild();

// Flow category for filtering
type FlowCategory = 'all' | 'user' | 'macros';

// Loaded package type (imported from OAIYApp)
interface LoadedPackage {
  manifest: OAIYPackageManifest;
  sourcePath: string;
  flows: Flow[];
}

// Active package flow reference
interface ActivePackageFlow {
  packageId: string;
  flowId: string;
}

interface FlowsSidebarProps {
  flows: Flow[];
  activeFlowId: string | null;
  onSelectFlow: (flowId: string) => void;
  onCreateFlow: (name: string) => void;
  onDeleteFlow: (flowId: string) => void;
  onDuplicateFlow: (flowId: string) => void;
  onRenameFlow: (flowId: string, name: string) => void;
  onSetFlowLocalOnly: (flowId: string, localOnly: boolean) => void;
  onSaveAsMacro?: (flowId: string) => void;
  onEditMacro?: (flowId: string) => void;
  onSaveMacro?: (flowId: string) => void;
  onRevertMacro?: (flowId: string) => void;
  hasMacroBeenModified?: (flowId: string) => boolean;
  isOpen: boolean;
  onClose: () => void;
  /** Expand from the collapsed rail. */
  onOpen: () => void;
  // Job tracking props
  isFlowRunning?: (flowId: string) => boolean;
  // Multi-package props
  loadedPackages?: Map<string, LoadedPackage>;
  activePackageFlow?: ActivePackageFlow | null;
  onSelectPackageFlow?: (packageId: string, flowId: string) => void;
  onClosePackage?: (packageId: string) => void;
  onLoadPackage?: () => void;
  onOpenBrowser?: () => void;
  // Export as package
  onExportAsPackage?: (flowId: string) => void;
}

export default function FlowsSidebar({
  flows,
  activeFlowId,
  onSelectFlow,
  onCreateFlow,
  onDeleteFlow,
  onDuplicateFlow,
  onRenameFlow,
  onSetFlowLocalOnly,
  onSaveAsMacro,
  onEditMacro,
  onSaveMacro,
  onRevertMacro,
  hasMacroBeenModified,
  isOpen,
  onClose,
  onOpen,
  isFlowRunning,
  loadedPackages,
  activePackageFlow,
  onSelectPackageFlow,
  onClosePackage,
  onLoadPackage,
  onOpenBrowser,
  onExportAsPackage,
}: FlowsSidebarProps) {
  const [newFlowName, setNewFlowName] = useState('');
  const [editingFlowId, setEditingFlowId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [contextMenuFlowId, setContextMenuFlowId] = useState<string | null>(null);
  const [expandedServices, setExpandedServices] = useState<Set<string>>(new Set());

  const toggleServiceExpanded = useCallback((packageId: string) => {
    setExpandedServices(prev => {
      const next = new Set(prev);
      if (next.has(packageId)) {
        next.delete(packageId);
      } else {
        next.add(packageId);
      }
      return next;
    });
  }, []);
  const [contextMenuPos, setContextMenuPos] = useState({ x: 0, y: 0 });
  const [deleteConfirm, setDeleteConfirm] = useState<{ flowId: string; flowName: string } | null>(null);
  const [activeCategory, setActiveCategory] = useState<FlowCategory>('all');

  // Categorize flows
  const categorizedFlows = useMemo(() => {
    const user: Flow[] = [];
    const macros: Flow[] = [];

    for (const flow of flows) {
      if (flow.isMacro) {
        macros.push(flow);
      } else {
        user.push(flow);
      }
    }

    return { user, macros };
  }, [flows]);

  // Filter flows based on category
  const filteredFlows = useMemo(() => {
    switch (activeCategory) {
      case 'user':
        return categorizedFlows.user;
      case 'macros':
        return categorizedFlows.macros;
      case 'all':
      default:
        return flows;
    }
  }, [flows, activeCategory, categorizedFlows]);

  // Category counts
  const categoryCounts = useMemo(() => ({
    all: flows.length,
    user: categorizedFlows.user.length,
    macros: categorizedFlows.macros.length,
  }), [flows, categorizedFlows]);

  const handleCreateFlow = useCallback(() => {
    if (newFlowName.trim()) {
      onCreateFlow(newFlowName.trim());
      setNewFlowName('');
    }
  }, [newFlowName, onCreateFlow]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCreateFlow();
    }
  }, [handleCreateFlow]);

  const handleStartRename = useCallback((flow: Flow) => {
    setEditingFlowId(flow.id);
    setEditingName(flow.name);
    setContextMenuFlowId(null);
  }, []);

  const handleFinishRename = useCallback(() => {
    if (editingFlowId && editingName.trim()) {
      onRenameFlow(editingFlowId, editingName.trim());
    }
    setEditingFlowId(null);
    setEditingName('');
  }, [editingFlowId, editingName, onRenameFlow]);

  const handleContextMenu = useCallback((e: React.MouseEvent, flowId: string) => {
    e.preventDefault();
    setContextMenuFlowId(flowId);
    // Clamp so the menu (up to ~300px tall for a modified macro) stays in the
    // viewport — right-clicking a flow near the bottom edge would otherwise push
    // its lower items (incl. Delete) off-screen and out of reach.
    const menuW = 200;
    const menuH = 300;
    const m = 8;
    setContextMenuPos({
      x: Math.max(m, Math.min(e.clientX, window.innerWidth - menuW - m)),
      y: Math.max(m, Math.min(e.clientY, window.innerHeight - menuH - m)),
    });
  }, []);

  const handleDeleteRequest = useCallback((flowId: string) => {
    const flow = flows.find(f => f.id === flowId);
    if (flow) {
      setDeleteConfirm({ flowId, flowName: flow.name });
    }
    setContextMenuFlowId(null);
  }, [flows]);

  const handleDeleteConfirm = useCallback(() => {
    if (deleteConfirm) {
      onDeleteFlow(deleteConfirm.flowId);
      setDeleteConfirm(null);
    }
  }, [deleteConfirm, onDeleteFlow]);

  const handleDeleteCancel = useCallback(() => {
    setDeleteConfirm(null);
  }, []);

  const handleDuplicate = useCallback((flowId: string) => {
    onDuplicateFlow(flowId);
    setContextMenuFlowId(null);
  }, [onDuplicateFlow]);

  const handleSaveAsMacro = useCallback((flowId: string) => {
    onSaveAsMacro?.(flowId);
    setContextMenuFlowId(null);
  }, [onSaveAsMacro]);

  const handleEditMacro = useCallback((flowId: string) => {
    onEditMacro?.(flowId);
    setContextMenuFlowId(null);
  }, [onEditMacro]);

  const handleSaveMacro = useCallback((flowId: string) => {
    onSaveMacro?.(flowId);
    setContextMenuFlowId(null);
  }, [onSaveMacro]);

  const handleRevertMacro = useCallback((flowId: string) => {
    onRevertMacro?.(flowId);
    setContextMenuFlowId(null);
  }, [onRevertMacro]);

  const handleExportAsPackage = useCallback((flowId: string) => {
    onExportAsPackage?.(flowId);
    setContextMenuFlowId(null);
  }, [onExportAsPackage]);

  const getFlowIcon = (flow: Flow) => {
    // Macro icons - show modified indicator if macro has been edited
    if (flow.isMacro) {
      const isModified = hasMacroBeenModified?.(flow.id) ?? false;
      if (isModified) {
        // Modified macro - show edit indicator
        return (
          <div className="relative" title="Modified macro (click to edit, right-click to revert)">
            <svg className="w-4 h-4 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-label="Macro">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
            </svg>
            <svg className="w-2.5 h-2.5 text-amber-400 absolute -top-0.5 -right-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-label="Modified">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          </div>
        );
      }
      // Unmodified macro - show standard icon
      return (
        <svg className="w-4 h-4 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-label="Macro">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
        </svg>
      );
    }
    if (flow.localOnly) {
      return (
        <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-label="Local only">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
      );
    }
    return (
      <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  };

  // Close context menu when clicking outside
  const handleBackdropClick = useCallback(() => {
    setContextMenuFlowId(null);
  }, []);

  // Collapsed: a narrow rail rather than nothing.
  //
  // This used to `return null`, which made the panel vanish with no affordance
  // left behind — the only way back was a topbar icon nowhere near where the
  // panel had been, so collapsing it read as "I broke something". A rail keeps
  // the reopen control exactly where the panel was, and keeps the flow count
  // visible so the workspace still tells you it has flows in it.
  //
  // Only on md+. Below that the panel is an overlay over the canvas, so a
  // permanent rail would eat scarce width for a control the topbar already
  // provides — there it still hides completely, with the backdrop to dismiss.
  if (!isOpen) {
    return (
      <div
        className="hidden xl:flex w-11 shrink-0 flex-col items-center gap-3 py-3"
        style={{
          backgroundColor: 'rgb(var(--color-bg-elevated))',
          borderRight: '1px solid rgb(var(--color-border-primary))',
        }}
      >
        <button
          type="button"
          onClick={onOpen}
          className="p-1.5 rounded-md transition-colors hover:bg-slate-200 dark:hover:bg-slate-700"
          aria-label="Expand the flows panel"
          aria-expanded={false}
          title="Expand flows (Ctrl+B)"
          style={{ color: 'rgb(var(--color-text-secondary))' }}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
        </button>

        {/* Vertical label + count. `writing-mode` keeps it legible in 44px. */}
        <button
          type="button"
          onClick={onOpen}
          className="flex-1 flex items-center justify-center cursor-pointer group"
          aria-hidden
          tabIndex={-1}
        >
          <span
            className="text-[11px] font-medium tracking-wide whitespace-nowrap transition-colors group-hover:opacity-100"
            style={{
              writingMode: 'vertical-rl',
              transform: 'rotate(180deg)',
              color: 'rgb(var(--color-text-tertiary))',
            }}
          >
            Flows{flows.length > 0 ? ` · ${flows.length}` : ''}
          </span>
        </button>
      </div>
    );
  }

  return (
    <>
      {/* Mobile backdrop */}
      <div
        className="xl:hidden fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />

      {/* Sidebar */}
      {/* Between md and xl the app rail is a 74px COLUMN while this panel is a
          fixed overlay, so anchoring it at 0 laid it over the rail and the
          primary navigation looked cut in half. Below md the rail is itself a
          drawer, so 0 is correct there; at xl this is relative and left is
          moot. */}
      <div
        className={`
          fixed xl:relative inset-y-0 z-50 xl:z-auto
          left-0 md:left-[74px] xl:left-0
          w-[85vw] sm:w-64 max-w-64 flex flex-col
          transform transition-transform duration-200 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full xl:translate-x-0'}
        `}
        style={{
          backgroundColor: 'rgb(var(--color-bg-elevated))',
          borderRight: '1px solid rgb(var(--color-border-primary))',
        }}
      >
        {/* Loaded Packages Section */}
        {loadedPackages && loadedPackages.size > 0 && (
          <div className="border-b border-purple-300 dark:border-purple-500/30">
            {/* Section header */}
            <div className="px-3 py-2.5 flex items-center justify-between bg-gradient-to-r from-purple-100 to-purple-50 dark:from-purple-900/30 dark:to-purple-900/10">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-purple-500/20 dark:bg-purple-500/30 flex items-center justify-center">
                  <svg className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                  </svg>
                </div>
                <span className="text-xs font-semibold text-purple-700 dark:text-purple-200">Packages</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-purple-200 dark:bg-purple-500/30 text-purple-600 dark:text-purple-300 font-medium">{loadedPackages.size}</span>
              </div>
              <div className="flex items-center gap-1">
                {onOpenBrowser && PACKAGE_BROWSE_AVAILABLE && (
                  <button
                    onClick={onOpenBrowser}
                    className="p-1.5 hover:bg-purple-200 dark:hover:bg-purple-600/30 rounded-md transition-colors"
                    title="Browse packages"
                  >
                    <svg className="w-3.5 h-3.5 text-purple-500 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </button>
                )}
                {onLoadPackage && (
                  <button
                    onClick={onLoadPackage}
                    className="p-1.5 hover:bg-purple-200 dark:hover:bg-purple-600/30 rounded-md transition-colors"
                    title="Load package or flow file"
                  >
                    <svg className="w-3.5 h-3.5 text-purple-500 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
            {/* Package list */}
            <div className="p-2.5 space-y-2 max-h-[40vh] overflow-y-auto bg-slate-50/50 dark:bg-transparent">
              {Array.from(loadedPackages.entries()).map(([packageId, pkg]) => {
                const isPackageActive = activePackageFlow?.packageId === packageId;
                return (
                  <div
                    key={packageId}
                    className={`
                      rounded-lg overflow-hidden shadow-sm
                      ${isPackageActive
                        ? 'bg-purple-50 dark:bg-purple-900/30 border-2 border-purple-400 dark:border-purple-500/60'
                        : 'bg-white dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/50 hover:border-purple-300 dark:hover:border-purple-500/40'
                      }
                      transition-all duration-150
                    `}
                  >
                    {/* Package header */}
                    <div className="flex items-center gap-2.5 px-3 py-2.5 bg-gradient-to-r from-purple-50/80 to-transparent dark:from-purple-900/20 dark:to-transparent">
                      <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                        <svg className="w-4 h-4 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-slate-800 dark:text-purple-100 truncate">{pkg.manifest.name}</div>
                        <div className="text-[11px] text-slate-500 dark:text-purple-400/70">v{pkg.manifest.version}</div>
                      </div>
                      {onClosePackage && (
                        <button
                          onClick={() => onClosePackage(packageId)}
                          className="p-1.5 hover:bg-red-100 dark:hover:bg-red-600/30 rounded-md transition-colors group"
                          title="Close package"
                        >
                          <svg className="w-3.5 h-3.5 text-slate-400 group-hover:text-red-500 dark:group-hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                    {/* Package flows */}
                    <div className="px-2 pb-2 space-y-1">
                      {pkg.flows.map((flow) => {
                        const isActive = activePackageFlow?.packageId === packageId && activePackageFlow?.flowId === flow.id;
                        return (
                          <button
                            key={flow.id}
                            onClick={() => onSelectPackageFlow?.(packageId, flow.id)}
                            className={`
                              w-full flex items-center gap-2 px-2.5 py-2 rounded-md text-xs text-left transition-all
                              ${isActive
                                ? 'bg-purple-500 text-white shadow-sm'
                                : 'text-slate-600 dark:text-purple-200/80 hover:bg-purple-100 dark:hover:bg-purple-600/20'
                              }
                            `}
                          >
                            <svg className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-white' : 'text-purple-500 dark:text-purple-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span className="truncate font-medium">{flow.name}</span>
                          </button>
                        );
                      })}
                    </div>
                    {/* Package services */}
                    <PackageServicesPanel
                      packageId={packageId}
                      manifest={pkg.manifest}
                      sourcePath={pkg.sourcePath}
                      isExpanded={expandedServices.has(packageId)}
                      onToggle={() => toggleServiceExpanded(packageId)}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Header */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgb(var(--color-border-secondary))' }}>
          <div className="flex items-center justify-between mb-2.5">
            <h2
              className="font-display text-base"
              style={{
                fontWeight: 500,
                letterSpacing: '-0.01em',
                color: 'rgb(var(--color-text-primary))',
              }}
            >
              {loadedPackages && loadedPackages.size > 0 ? 'Your Flows' : 'Flows'}
            </h2>
            {/* Collapse, on md+. The topbar has a toggle too, but a control in
                the panel's own header is where you look when you want the panel
                out of the way. */}
            <button
              type="button"
              onClick={onClose}
              className="hidden xl:block p-1 hover:bg-slate-200 dark:hover:bg-slate-700 rounded transition-colors"
              aria-label="Collapse the flows panel"
              aria-expanded
              title="Collapse flows (Ctrl+B)"
            >
              <svg className="w-4 h-4 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </button>
            {/* Below md the panel is an overlay, so this dismisses rather than
                collapses — a distinct affordance for distinct behaviour. */}
            <button
              type="button"
              onClick={onClose}
              className="xl:hidden p-1 hover:bg-slate-200 dark:hover:bg-slate-700 rounded transition-colors"
              aria-label="Close sidebar"
            >
              <svg className="w-5 h-5 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          {/* Category Tabs */}
          <div className="flex gap-0.5 p-0.5 rounded-md" style={{ backgroundColor: 'rgb(var(--color-bg-tertiary) / 0.6)' }}>
            {([
              { id: 'all', label: 'All' },
              { id: 'user', label: 'User' },
              { id: 'macros', label: 'Macros' },
            ] as const).map(cat => (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id)}
                className="flex-1 px-2 py-1 text-xs rounded-[5px] transition-all font-medium"
                style={
                  activeCategory === cat.id
                    ? {
                        backgroundColor: 'rgb(var(--color-bg-elevated))',
                        color: cat.id === 'macros' ? 'rgb(var(--accent-secondary))' : 'rgb(var(--accent-primary))',
                        boxShadow: 'var(--shadow-sm)',
                      }
                    : { color: 'rgb(var(--color-text-tertiary))' }
                }
              >
                {cat.label}
                {categoryCounts[cat.id] > 0 && (
                  <span className="ml-1 text-[10px] opacity-60 font-mono">{categoryCounts[cat.id]}</span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* New Flow Input */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid rgb(var(--color-border-secondary))' }}>
          <div className="flex gap-2">
            <input
              type="text"
              value={newFlowName}
              onChange={(e) => setNewFlowName(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="New flow name..."
              aria-label="New flow name"
              className="input flex-1"
            />
            <button
              onClick={handleCreateFlow}
              disabled={!newFlowName.trim()}
              className="btn btn-primary btn-icon"
              aria-label="Create new flow"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>
        </div>

        {/* Flow List */}
        <div className="flex-1 overflow-y-auto">
          {filteredFlows.length === 0 ? (
            <div className="empty-state py-8">
              <svg className="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="empty-state-title">
                {activeCategory === 'all' ? 'No flows yet' : `No ${activeCategory} flows`}
              </p>
              <p className="empty-state-description">
                {activeCategory === 'all' ? 'Create your first flow above' : `No flows in this category`}
              </p>
            </div>
          ) : (
            <ul className="p-2 space-y-1">
              {filteredFlows.map((flow) => (
                <li key={flow.id}>
                  {editingFlowId === flow.id ? (
                    <input
                      type="text"
                      value={editingName}
                      onChange={(e) => setEditingName(e.target.value)}
                      onBlur={handleFinishRename}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleFinishRename();
                        if (e.key === 'Escape') {
                          setEditingFlowId(null);
                          setEditingName('');
                        }
                      }}
                      autoFocus
                      className="w-full px-2 py-1.5 text-sm bg-white dark:bg-slate-900 border border-blue-500 rounded text-slate-700 dark:text-slate-200 focus:outline-none"
                    />
                  ) : (
                    <div
                      role="button"
                      tabIndex={0}
                      className={`
                        group w-full flex items-center gap-2 px-2 h-8 rounded text-sm
                        transition-colors cursor-pointer
                        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-white dark:focus:ring-offset-slate-900
                        ${flow.id === activeFlowId
                          ? 'bg-blue-600/30 text-blue-700 dark:text-blue-200 border border-blue-500/50'
                          : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                        }
                      `}
                      onClick={() => onSelectFlow(flow.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onSelectFlow(flow.id);
                        }
                      }}
                      onContextMenu={(e) => handleContextMenu(e, flow.id)}
                      aria-label={`Select flow: ${flow.name}`}
                      aria-current={flow.id === activeFlowId ? 'true' : undefined}
                    >
                      {getFlowIcon(flow)}
                      <span className="flex-1 truncate">{flow.name}</span>
                      {/* Running indicator */}
                      {isFlowRunning?.(flow.id) && (
                        <span className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 bg-blue-600/30 rounded text-blue-300 border border-blue-500/30" title="Running">
                          <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
                          <span>Running</span>
                        </span>
                      )}
                      {!isFlowRunning?.(flow.id) && flow.tags && flow.tags.length > 0 && (
                        <span className="text-[10px] px-1.5 py-0.5 bg-slate-200 dark:bg-slate-600 rounded text-slate-500 dark:text-slate-400 group-hover:opacity-0 transition-opacity">
                          {flow.tags[0]}
                        </span>
                      )}
                      {/* Delete button - visible on hover */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteRequest(flow.id);
                        }}
                        className="flex p-1 hover:bg-red-600/30 rounded transition-all opacity-0 group-hover:opacity-100"
                        title="Delete flow"
                        aria-label="Delete flow"
                      >
                        <svg className="w-3.5 h-3.5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Context Menu */}
        {contextMenuFlowId && (
          <>
            <div
              className="fixed inset-0 z-50"
              onClick={handleBackdropClick}
            />
            <div
              className="fixed z-50 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg shadow-xl py-1 min-w-[140px]"
              style={{ left: contextMenuPos.x, top: contextMenuPos.y }}
            >
              <button
                onClick={() => handleStartRename(flows.find(f => f.id === contextMenuFlowId)!)}
                className="w-full px-3 py-1.5 text-left text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Rename
              </button>
              <button
                onClick={() => handleDuplicate(contextMenuFlowId)}
                className="w-full px-3 py-1.5 text-left text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Duplicate
              </button>
              {onExportAsPackage && (
                <button
                  onClick={() => handleExportAsPackage(contextMenuFlowId)}
                  className="w-full px-3 py-1.5 text-left text-sm text-purple-600 dark:text-purple-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Export as Package
                </button>
              )}
              <div className="border-t border-slate-200 dark:border-slate-700 my-1" />
              {/* Macro options */}
              {(() => {
                const flow = flows.find(f => f.id === contextMenuFlowId);
                if (!flow) return null;

                if (flow.isMacro) {
                  const isModified = hasMacroBeenModified?.(flow.id) ?? false;
                  return (
                    <>
                      {/* Edit Macro - always available */}
                      <button
                        onClick={() => handleEditMacro(contextMenuFlowId)}
                        className="w-full px-3 py-1.5 text-left text-sm text-violet-600 dark:text-violet-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                        Edit Macro
                      </button>
                      {/* Save Changes - shows for all macros */}
                      <button
                        onClick={() => handleSaveMacro(contextMenuFlowId)}
                        className="w-full px-3 py-1.5 text-left text-sm text-green-600 dark:text-green-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                        </svg>
                        Save Changes
                      </button>
                      {/* Revert to Original - only shows if modified */}
                      {isModified && (
                        <button
                          onClick={() => handleRevertMacro(contextMenuFlowId)}
                          className="w-full px-3 py-1.5 text-left text-sm text-amber-600 dark:text-amber-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                          Revert to Original
                        </button>
                      )}
                    </>
                  );
                } else {
                  // Not a macro - offer to save as macro
                  return (
                    <button
                      onClick={() => handleSaveAsMacro(contextMenuFlowId)}
                      className="w-full px-3 py-1.5 text-left text-sm text-violet-600 dark:text-violet-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                      </svg>
                      Save as Macro
                    </button>
                  );
                }
              })()}
              <div className="border-t border-slate-200 dark:border-slate-700 my-1" />
              <button
                onClick={() => {
                  const flow = flows.find(f => f.id === contextMenuFlowId)!;
                  onSetFlowLocalOnly(contextMenuFlowId, !flow.localOnly);
                  setContextMenuFlowId(null);
                }}
                className="w-full px-3 py-1.5 text-left text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                {flows.find(f => f.id === contextMenuFlowId)?.localOnly ? 'Allow Cloud' : 'Local Only'}
              </button>
              <div className="border-t border-slate-200 dark:border-slate-700 my-1" />
              <button
                onClick={() => handleDeleteRequest(contextMenuFlowId)}
                className="w-full px-3 py-1.5 text-left text-sm text-red-500 dark:text-red-400 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete
              </button>
            </div>
          </>
        )}

        {/* Delete Confirmation Dialog */}
        <ConfirmDialog
          isOpen={deleteConfirm !== null}
          title="Delete Flow"
          message={`Are you sure you want to delete "${deleteConfirm?.flowName}"? This action cannot be undone and all nodes in this flow will be permanently removed.`}
          confirmLabel="Delete Flow"
          cancelLabel="Cancel"
          variant="danger"
          onConfirm={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
        />

        {/* Load Package Buttons - shown when no packages loaded */}
        {(onLoadPackage || (onOpenBrowser && PACKAGE_BROWSE_AVAILABLE)) && (!loadedPackages || loadedPackages.size === 0) && (
          <div className="p-3 border-t border-slate-200 dark:border-slate-700 space-y-2">
            {onOpenBrowser && PACKAGE_BROWSE_AVAILABLE && (
              <button
                onClick={onOpenBrowser}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs text-purple-600 dark:text-purple-300 hover:text-purple-800 dark:hover:text-white bg-purple-100 dark:bg-purple-900/20 hover:bg-purple-200 dark:hover:bg-purple-600/30 rounded-lg transition-colors border border-purple-400/50 dark:border-purple-500/30"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Browse Packages
              </button>
            )}
            {onLoadPackage && (
              <button
                onClick={onLoadPackage}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white bg-slate-100 dark:bg-slate-800/50 hover:bg-slate-200 dark:hover:bg-slate-700/50 rounded-lg transition-colors border border-slate-300 dark:border-slate-600/30"
                title={PACKAGE_BROWSE_AVAILABLE ? 'Load .oaiy package or .json flow file' : 'Import a .json flow file'}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Load from File
              </button>
            )}
          </div>
        )}

        {/* The brand wordmark, tagline and © used to live here. The shell's
            own sidebar carries them now (see chrome/ShellChrome.tsx), so this
            rail stays a plain list of flows — repeating them two panels apart
            just read as a duplicate. */}
      </div>
    </>
  );
}
