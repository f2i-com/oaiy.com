import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  ArrowDownAZ,
  ChevronDown,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Clock,
  Copy,
  FileText,
  Globe,
  HardDrive,
  Layers,
  MoreHorizontal,
  Package,
  Pencil,
  Plus,
  RotateCcw,
  Save,
  Search,
  Sparkles,
  Trash2,
  Wrench,
  X,
} from 'lucide-react';
import { shortcut } from '../../lib/keys';
import type { Flow, OAIYPackageManifest } from 'oaiy-core';
import { oaiyDesktop } from '../../lib/oaiyAgentTools';
import { shared as keptOnDesktop } from '../../hooks/useDesktopFlows';
import type { ExampleFlow } from '../../lib/exampleFlows';
import ConfirmDialog from '../ui/ConfirmDialog';
import Menu, { MenuItem } from '../ui/Menu';
import AgentToolDialog from '../dialogs/AgentToolDialog';
import { useToast } from '../Toast';
import PackageServicesPanel from './PackageServicesPanel';

/**
 * The flows rail: the project's flows (and a loaded package's), to open one.
 *
 * Making a flow is New flow in the editor's header, and importing one is its
 * Import menu, so neither is repeated here: the rail is the list, found in by
 * a search (name, description, tags) and its tags, in the order last changed
 * or by name; the examples OAIY ships under it, to add; and each flow's own
 * actions in its ⋯ menu (or a right-click).
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
  /** In OAIY: share a flow with apps (FormLogic, the API), or stop. */
  onSetFlowRemote?: (flowId: string, remote: boolean) => void;
  /** The examples OAIY ships, and adding one to the project. */
  examples?: ExampleFlow[];
  onAddExample?: (example: ExampleFlow) => void;
}

type FlowSort = 'recent' | 'name';
const SORT_KEY = 'oaiy.flowsSort';
const EXAMPLES_KEY = 'oaiy.flowsExamples';

/** A remembered choice of the rail's (this browser's, best effort). */
function remembered<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return allowed.includes(v as T) ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}
function remember(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // storage unavailable: the choice lasts this visit
  }
}

/** Tags the editor sets for its own bookkeeping, not ones a person browses by. */
const HIDDEN_TAGS = new Set(['user-created']);

/** Whether a flow (or example) matches a search: its name, description or a tag. */
function matches(f: { name: string; description?: string; tags?: string[] }, q: string): boolean {
  return f.name.toLowerCase().includes(q) || !!f.description?.toLowerCase().includes(q) || !!f.tags?.some((t) => t.toLowerCase().includes(q));
}

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
  onSetFlowRemote,
  examples = [],
  onAddExample,
}: FlowsSidebarProps) {
  const [editingFlowId, setEditingFlowId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [menu, setMenu] = useState<{ flowId: string; anchor: HTMLElement | null; at: { x: number; y: number } | null } | null>(null);
  const [expandedServices, setExpandedServices] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const [tag, setTag] = useState<string | null>(null);
  const [sort, setSort] = useState<FlowSort>(() => remembered(SORT_KEY, ['recent', 'name'] as const, 'recent'));
  const [examplesOpen, setExamplesOpen] = useState(() => remembered(EXAMPLES_KEY, ['open', 'closed'] as const, 'open') === 'open');
  const searchRef = useRef<HTMLInputElement>(null);
  // Where each flow sits in the list while it is on screen: opening a flow (which may save it) never moves it, nor does
  // an edit; the newest-changed first as the list is first drawn, a flow made since on top. Sorting again re-sorts.
  const placed = useRef<Map<string, number>>(new Map());
  const activeRef = useRef<HTMLDivElement>(null);

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
  const [deleteConfirm, setDeleteConfirm] = useState<{ flowId: string; flowName: string; onDesktop?: boolean } | null>(null);
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

  const inCategory = activeCategory === 'user' ? categorizedFlows.user : activeCategory === 'macros' ? categorizedFlows.macros : flows;

  // The tags to browse by: those of the flows in view, the most used first.
  const tags = useMemo(() => {
    const counts = new Map<string, number>();
    for (const f of inCategory) for (const t of f.tags ?? []) if (!HIDDEN_TAGS.has(t)) counts.set(t, (counts.get(t) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([t]) => t);
  }, [inCategory]);
  const activeTag = tag && tags.includes(tag) ? tag : null;

  // The flows' places (see `placed`): kept for the ones placed, the new ones on top, the gone forgotten.
  const ranks = useMemo(() => {
    const ranks = placed.current;
    for (const id of [...ranks.keys()]) if (!flows.some((f) => f.id === id)) ranks.delete(id);
    const unplaced = flows.filter((f) => !ranks.has(f.id)).sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    if (ranks.size === 0) {
      unplaced.forEach((f, i) => ranks.set(f.id, i));
    } else {
      let top = Math.min(...ranks.values());
      for (const f of [...unplaced].reverse()) ranks.set(f.id, --top);
    }
    return new Map(ranks);
  }, [flows, sort]);

  // Filter flows by category, the tag and the search; then in the chosen order.
  const filteredFlows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const kept = inCategory.filter((f) => (!activeTag || f.tags?.includes(activeTag)) && (!q || matches(f, q)));
    return sort === 'name'
      ? [...kept].sort((a, b) => a.name.localeCompare(b.name))
      : [...kept].sort((a, b) => (ranks.get(a.id) ?? 0) - (ranks.get(b.id) ?? 0));
  }, [inCategory, activeTag, query, sort, ranks]);

  // The open flow in view in the list (opened from elsewhere: the canvas, an example, a new flow).
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' });
  }, [activeFlowId]);

  // The examples not in the project yet (one added is a flow of the list), as the search and tag find them.
  const shownExamples = useMemo(() => {
    if (activeCategory === 'macros' || !onAddExample) return [];
    const q = query.trim().toLowerCase();
    return examples.filter((e) => !flows.some((f) => f.id === e.id) && (!activeTag || e.tags.includes(activeTag)) && (!q || matches(e, q)));
  }, [examples, flows, activeCategory, activeTag, query, onAddExample]);

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
      setDeleteConfirm({ flowId, flowName: flow.name, onDesktop: !!oaiyDesktop() && keptOnDesktop(flow) });
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
    if (flow.localOnly) return <HardDrive size={15} aria-label="Kept in this window only" />;
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
          title={`Expand flows (${shortcut('B')})`}
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
            <span className="oaiy-rail-tools">
            <button
              type="button"
              onClick={() => {
                const next = sort === 'recent' ? 'name' : 'recent';
                placed.current.clear();
                setSort(next);
                remember(SORT_KEY, next);
              }}
              className="oaiy-icon-btn"
              aria-label={sort === 'recent' ? 'Sorted by last changed: sort by name' : 'Sorted by name: sort by last changed'}
              title={sort === 'recent' ? 'Last changed first (click for A to Z)' : 'A to Z (click for last changed first)'}
              data-testid="flows-sort"
            >
              {sort === 'recent' ? <Clock size={15} /> : <ArrowDownAZ size={15} />}
            </button>
            {/* Collapse on md+; below md the panel is an overlay, so this dismisses it. */}
            <button
              type="button"
              onClick={onClose}
              className="oaiy-icon-btn"
              aria-label="Collapse the flows panel"
              aria-expanded
              title={`Collapse flows (${shortcut('B')})`}
            >
              <ChevronsLeft size={16} />
            </button>
            </span>
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
          {(flows.length > 0 || examples.length > 0 || query) && (
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
          {tags.length > 0 && (
            <div className="oaiy-tags" role="group" aria-label="Show the flows with a tag">
              {tags.slice(0, 12).map((t) => (
                <button key={t} type="button" aria-pressed={activeTag === t} onClick={() => setTag(activeTag === t ? null : t)}>
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Flow List */}
        <div className="oaiy-rail-list">
          {filteredFlows.length === 0 ? (
            <div className="oaiy-empty bare">
              <FileText size={22} />
              <p className="oaiy-empty-title">
                {query || activeTag ? 'No flow matches' : activeCategory === 'macros' ? 'No macros yet' : 'No flows yet'}
              </p>
              <p className="oaiy-empty-text">
                {query || activeTag
                  ? 'Try another word or tag, or clear the search.'
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
                        className={`oaiy-flow-item${active ? ' active' : ''}${flow.isMacro ? ' macro' : flow.localOnly ? ' local' : ''}${flow.description ? ' has-desc' : ''}`}
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
                        ref={active ? activeRef : undefined}
                        data-flow-name={flow.name}
                        title={flow.description ? `${flow.name}\n${flow.description}` : flow.name}
                      >
                        {flowIcon(flow)}
                        <span className="oaiy-flow-text">
                          <span>{flow.name}</span>
                          {flow.description && <small>{flow.description}</small>}
                        </span>
                        {flow.remote && <Globe size={13} className="oaiy-flow-shared" aria-label="Shared with apps" />}
                        {running ? (
                          <span className="oaiy-pill dot accent live" title="Running">running</span>
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

          {/* The examples OAIY ships that are not in the project (yet, or any more): one click adds one. */}
          {shownExamples.length > 0 && (
            <div className="oaiy-rail-group" data-testid="flows-examples">
              <button
                type="button"
                className="oaiy-rail-group-head"
                aria-expanded={examplesOpen}
                onClick={() => {
                  setExamplesOpen(!examplesOpen);
                  remember(EXAMPLES_KEY, examplesOpen ? 'closed' : 'open');
                }}
              >
                {examplesOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                Examples <small>{shownExamples.length}</small>
              </button>
              {examplesOpen && (
                <ul>
                  {shownExamples.map((example) => (
                    <li key={example.id}>
                      <div
                        role="button"
                        tabIndex={0}
                        className={`oaiy-flow-item example${example.description ? ' has-desc' : ''}`}
                        onClick={() => onAddExample?.(example)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            onAddExample?.(example);
                          }
                        }}
                        aria-label={`Add the example flow: ${example.name}`}
                        title={`Add it to your flows${example.description ? `\n${example.description}` : ''}`}
                      >
                        <Sparkles size={15} aria-hidden="true" />
                        <span className="oaiy-flow-text">
                          <span>{example.name}</span>
                          {example.description && <small>{example.description}</small>}
                        </span>
                        <span className="oaiy-flow-add"><Plus size={12} /> Add</span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
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
              {oaiyDesktop() && onSetFlowRemote && !menuFlow.isMacro && !menuFlow.localOnly && (
                <MenuItem
                  icon={<Globe size={14} />}
                  label={menuFlow.remote ? 'Stop apps running it' : 'Let apps run it'}
                  hint={menuFlow.remote
                    ? 'FormLogic and other apps can run it now'
                    : 'FormLogic and other apps, through the oaiy-flows connector'}
                  onSelect={() => onSetFlowRemote(menuFlow.id, !menuFlow.remote)}
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
              {/* What it does: whether OAIY Desktop keeps the flow (for the agent, the CLI and other apps). Nothing
                  else: a flow's services are its nodes'. Only where there is a desktop to keep it. */}
              {oaiyDesktop() && !menuFlow.isMacro && (
                <MenuItem
                  icon={<HardDrive size={14} />}
                  label={menuFlow.localOnly ? 'Keep it in OAIY Desktop too' : 'Keep it in this window only'}
                  hint={menuFlow.localOnly ? 'The agent, the CLI and other apps see it again' : 'Taken off OAIY Desktop: the agent, the CLI and other apps no longer see it'}
                  onSelect={() => {
                    onSetFlowLocalOnly(menuFlow.id, !menuFlow.localOnly);
                    addToast(
                      menuFlow.localOnly
                        ? `“${menuFlow.name}” is kept in OAIY Desktop again.`
                        : `“${menuFlow.name}” is kept in this window only: it is taken off OAIY Desktop, and changes here are no longer sent there.`,
                      'info',
                      6000,
                    );
                  }}
                />
              )}
              {!(menuFlow.isMacro && menuFlow.isBuiltIn) && (
                <>
                  <hr />
                  <MenuItem icon={<Trash2 size={14} />} label="Delete…" danger onSelect={() => handleDeleteRequest(menuFlow.id)} />
                </>
              )}
            </>
          )}
        </Menu>

        {agentFlow && <AgentToolDialog flow={agentFlow} onClose={() => setAgentFlow(null)} onDone={(message) => addToast(message, 'success')} />}

        <ConfirmDialog
          isOpen={deleteConfirm !== null}
          title="Delete this flow?"
          message={`“${deleteConfirm?.flowName}” and all its nodes are deleted${deleteConfirm?.onDesktop ? ', here and in OAIY Desktop (with the agent’s tool made from it, if there is one)' : ''}. This cannot be undone.`}
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
