import { useState, useCallback, useMemo, useRef } from 'react';
import {
  ChevronsLeft,
  ChevronsRight,
  Copy,
  FileText,
  HardDrive,
  Layers,
  MoreHorizontal,
  Package,
  Pencil,
  RotateCcw,
  Save,
  Search,
  Trash2,
  Wrench,
  X,
} from 'lucide-react';
import type { Flow, OAIYPackageManifest } from 'oaiy-core';
import { oaiyDesktop } from '../../lib/oaiyAgentTools';
import ConfirmDialog from '../ui/ConfirmDialog';
import Menu, { MenuItem } from '../ui/Menu';
import AgentToolDialog from '../dialogs/AgentToolDialog';
import { useToast } from '../Toast';
import PackageServicesPanel from './PackageServicesPanel';

/**
 * The flows rail: the project's flows (and a loaded package's), to open one.
 *
 * Making a flow is New flow in the editor's header, and importing one is its
 * Import menu, so neither is repeated here: the rail is the list, a search
 * when it grows, and each flow's own actions in its ⋯ menu (or a right-click).
 */

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
  // Export as package
  onExportAsPackage?: (flowId: string) => void;
}

/** Past this many flows the rail offers a search. */
const SEARCH_FROM = 6;

export default function FlowsSidebar({
  flows,
  activeFlowId,
  onSelectFlow,
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
  onExportAsPackage,
}: FlowsSidebarProps) {
  const [editingFlowId, setEditingFlowId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [menu, setMenu] = useState<{ flowId: string; anchor: HTMLElement | null; at: { x: number; y: number } | null } | null>(null);
  const [expandedServices, setExpandedServices] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);

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

  // Filter flows by category, then by the search.
  const filteredFlows = useMemo(() => {
    const inCategory = activeCategory === 'user' ? categorizedFlows.user : activeCategory === 'macros' ? categorizedFlows.macros : flows;
    const q = query.trim().toLowerCase();
    if (!q) return inCategory;
    return inCategory.filter((f) => f.name.toLowerCase().includes(q) || f.description?.toLowerCase().includes(q) || f.tags?.some((t) => t.toLowerCase().includes(q)));
  }, [flows, activeCategory, categorizedFlows, query]);

  // Category counts
  const categoryCounts = useMemo(() => ({
    all: flows.length,
    user: categorizedFlows.user.length,
    macros: categorizedFlows.macros.length,
  }), [flows, categorizedFlows]);

  const handleStartRename = useCallback((flow: Flow) => {
    setEditingFlowId(flow.id);
    setEditingName(flow.name);
    setMenu(null);
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
    setMenu({ flowId, anchor: null, at: { x: e.clientX, y: e.clientY } });
  }, []);

  const handleDeleteRequest = useCallback((flowId: string) => {
    const flow = flows.find(f => f.id === flowId);
    if (flow) {
      setDeleteConfirm({ flowId, flowName: flow.name });
    }
    setMenu(null);
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

  // In OAIY's window: a flow can be given to the agent, as a tool or in front of one.
  const { addToast } = useToast();
  const [agentFlow, setAgentFlow] = useState<Flow | null>(null);

  const flowIcon = (flow: Flow) => {
    if (flow.isMacro) {
      const isModified = hasMacroBeenModified?.(flow.id) ?? false;
      return <Layers size={15} aria-label={isModified ? 'Macro, changed' : 'Macro'} />;
    }
    if (flow.localOnly) return <HardDrive size={15} aria-label="Local only" />;
    return <FileText size={15} aria-hidden="true" />;
  };

  const menuFlow = menu ? flows.find((f) => f.id === menu.flowId) ?? null : null;
  const closeMenu = useCallback(() => setMenu(null), []);

  // Collapsed: a narrow rail rather than nothing, so the way back is where the
  // panel was, and the count still says the workspace has flows in it. Only on
  // md+; below that the panel is an overlay, and the topbar's toggle opens it.
  if (!isOpen) {
    return (
      <div className="oaiy-rail-collapsed">
        <button
          type="button"
          onClick={onOpen}
          className="oaiy-icon-btn"
          aria-label="Expand the flows panel"
          aria-expanded={false}
          title="Expand flows (Ctrl+B)"
        >
          <ChevronsRight size={16} />
        </button>
        <button type="button" onClick={onOpen} aria-hidden tabIndex={-1}>
          Flows{flows.length > 0 ? ` · ${flows.length}` : ''}
        </button>
      </div>
    );
  }

  return (
    <>
      {/* Below md the rail lies over the canvas: a scrim to dismiss it. */}
      <div className="oaiy-rail-scrim" onClick={onClose} />

      <div className="oaiy-rail" data-testid="flows-rail">
        {/* Loaded packages: their flows, above the project's own. */}
        {loadedPackages && loadedPackages.size > 0 && (
          <div className="oaiy-rail-section max-h-[42%] overflow-y-auto">
            <header>
              <span className="oaiy-label">
                <Package size={12} /> Packages <span className="font-mono">{loadedPackages.size}</span>
              </span>
            </header>
            {Array.from(loadedPackages.entries()).map(([packageId, pkg]) => {
              const isPackageActive = activePackageFlow?.packageId === packageId;
              return (
                <div key={packageId} className={`oaiy-pkg${isPackageActive ? ' active' : ''}`}>
                  <div className="oaiy-pkg-head">
                    <Package size={14} />
                    <strong title={pkg.manifest.name}>{pkg.manifest.name}</strong>
                    <small>v{pkg.manifest.version}</small>
                    {onClosePackage && (
                      <button
                        type="button"
                        onClick={() => onClosePackage(packageId)}
                        className="oaiy-icon-btn sm"
                        title="Close package"
                        aria-label={`Close ${pkg.manifest.name}`}
                      >
                        <X size={13} />
                      </button>
                    )}
                  </div>
                  <ul>
                    {pkg.flows.map((flow) => {
                      const isActive = activePackageFlow?.packageId === packageId && activePackageFlow?.flowId === flow.id;
                      return (
                        <li key={flow.id}>
                          <div
                            role="button"
                            tabIndex={0}
                            className={`oaiy-flow-item${isActive ? ' active' : ''}`}
                            onClick={() => onSelectPackageFlow?.(packageId, flow.id)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault();
                                onSelectPackageFlow?.(packageId, flow.id);
                              }
                            }}
                            aria-current={isActive ? 'true' : undefined}
                          >
                            <FileText size={14} />
                            <span>{flow.name}</span>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
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
        )}

        <div className="oaiy-rail-head">
          <div className="oaiy-rail-title">
            <h2>
              {loadedPackages && loadedPackages.size > 0 ? 'Your flows' : 'Flows'}
              <small>{flows.length}</small>
            </h2>
            {/* Collapse on md+; below md the panel is an overlay, so this dismisses it. */}
            <button
              type="button"
              onClick={onClose}
              className="oaiy-icon-btn"
              aria-label="Collapse the flows panel"
              aria-expanded
              title="Collapse flows (Ctrl+B)"
            >
              <ChevronsLeft size={16} />
            </button>
          </div>
          <div className="oaiy-seg" role="tablist" aria-label="Which flows">
            {([
              { id: 'all', label: 'All' },
              { id: 'user', label: 'Flows' },
              { id: 'macros', label: 'Macros' },
            ] as const).map(cat => (
              <button
                key={cat.id}
                type="button"
                role="tab"
                aria-selected={activeCategory === cat.id}
                onClick={() => setActiveCategory(cat.id)}
              >
                {cat.label}
                {categoryCounts[cat.id] > 0 && <small>{categoryCounts[cat.id]}</small>}
              </button>
            ))}
          </div>
          {(flows.length >= SEARCH_FROM || query) && (
            <div className="oaiy-search">
              <Search size={14} />
              <input
                ref={searchRef}
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Escape' && query) { e.stopPropagation(); setQuery(''); } }}
                placeholder="Find a flow"
                aria-label="Find a flow"
                className="oaiy-input oaiy-input-sm"
              />
              {query && (
                <button type="button" onClick={() => { setQuery(''); searchRef.current?.focus(); }} aria-label="Clear the search">
                  <X size={13} />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Flow List */}
        <div className="oaiy-rail-list">
          {filteredFlows.length === 0 ? (
            <div className="oaiy-empty bare">
              <FileText size={22} />
              <p className="oaiy-empty-title">
                {query ? 'No flow matches' : activeCategory === 'macros' ? 'No macros yet' : 'No flows yet'}
              </p>
              <p className="oaiy-empty-text">
                {query
                  ? 'Try another word, or clear the search.'
                  : activeCategory === 'macros'
                    ? 'Save a flow as a macro from its ⋯ menu to use it as a node.'
                    : 'Make one with New flow, or bring one in with Import.'}
              </p>
            </div>
          ) : (
            <ul>
              {filteredFlows.map((flow) => {
                const active = flow.id === activeFlowId;
                const running = isFlowRunning?.(flow.id);
                return (
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
                        aria-label="Flow name"
                        className="oaiy-input oaiy-flow-rename"
                      />
                    ) : (
                      <div
                        role="button"
                        tabIndex={0}
                        className={`oaiy-flow-item${active ? ' active' : ''}${flow.isMacro ? ' macro' : flow.localOnly ? ' local' : ''}`}
                        onClick={() => onSelectFlow(flow.id)}
                        onKeyDown={(e) => {
                          if (e.target !== e.currentTarget) return;
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            onSelectFlow(flow.id);
                          }
                          if (e.key === 'F2') {
                            e.preventDefault();
                            handleStartRename(flow);
                          }
                        }}
                        onContextMenu={(e) => handleContextMenu(e, flow.id)}
                        aria-label={`Select flow: ${flow.name}`}
                        aria-current={active ? 'true' : undefined}
                        data-flow-name={flow.name}
                        title={flow.name}
                      >
                        {flowIcon(flow)}
                        <span>{flow.name}</span>
                        {running ? (
                          <span className="oaiy-pill dot accent live" title="Running">running</span>
                        ) : flow.tags && flow.tags.length > 0 ? (
                          <span className="oaiy-pill">{flow.tags[0]}</span>
                        ) : null}
                        <button
                          type="button"
                          className="oaiy-flow-more"
                          onClick={(e) => {
                            e.stopPropagation();
                            setMenu({ flowId: flow.id, anchor: e.currentTarget, at: null });
                          }}
                          aria-label={`Actions for ${flow.name}`}
                          aria-haspopup="menu"
                          title="Rename, duplicate, delete…"
                        >
                          <MoreHorizontal size={15} />
                        </button>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* A flow's own actions: from its ⋯ button, or a right-click. */}
        <Menu open={!!menuFlow} onClose={closeMenu} anchor={menu?.anchor} at={menu?.at} align="start" label={menuFlow ? `${menuFlow.name} actions` : 'Flow actions'}>
          {menuFlow && (
            <>
              <MenuItem icon={<Pencil size={14} />} label="Rename" onSelect={() => handleStartRename(menuFlow)} />
              <MenuItem icon={<Copy size={14} />} label="Duplicate" onSelect={() => onDuplicateFlow(menuFlow.id)} />
              {oaiyDesktop() && (
                <MenuItem
                  icon={<Wrench size={14} />}
                  label="Give it to the agent…"
                  hint="As a tool, or before or instead of one of its tools"
                  onSelect={() => setAgentFlow(menuFlow)}
                />
              )}
              {onExportAsPackage && (
                <MenuItem icon={<Package size={14} />} label="Export as a package" onSelect={() => onExportAsPackage(menuFlow.id)} />
              )}
              <hr />
              {menuFlow.isMacro ? (
                <>
                  <MenuItem icon={<Pencil size={14} />} label="Edit the macro" onSelect={() => onEditMacro?.(menuFlow.id)} />
                  <MenuItem icon={<Save size={14} />} label="Save the macro" onSelect={() => onSaveMacro?.(menuFlow.id)} />
                  {(hasMacroBeenModified?.(menuFlow.id) ?? false) && (
                    <MenuItem icon={<RotateCcw size={14} />} label="Revert to the original" onSelect={() => onRevertMacro?.(menuFlow.id)} />
                  )}
                </>
              ) : (
                <MenuItem icon={<Layers size={14} />} label="Save as a macro" hint="Use it as a node in other flows" onSelect={() => onSaveAsMacro?.(menuFlow.id)} />
              )}
              <MenuItem
                icon={<HardDrive size={14} />}
                label={menuFlow.localOnly ? 'Allow cloud services' : 'Keep it local only'}
                onSelect={() => onSetFlowLocalOnly(menuFlow.id, !menuFlow.localOnly)}
              />
              <hr />
              <MenuItem icon={<Trash2 size={14} />} label="Delete…" danger onSelect={() => handleDeleteRequest(menuFlow.id)} />
            </>
          )}
        </Menu>

        {agentFlow && <AgentToolDialog flow={agentFlow} onClose={() => setAgentFlow(null)} onDone={(message) => addToast(message, 'success')} />}

        <ConfirmDialog
          isOpen={deleteConfirm !== null}
          title="Delete this flow?"
          message={`“${deleteConfirm?.flowName}” and all its nodes are deleted. This cannot be undone.`}
          confirmLabel="Delete flow"
          cancelLabel="Cancel"
          variant="danger"
          onConfirm={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
        />
      </div>
    </>
  );
}
