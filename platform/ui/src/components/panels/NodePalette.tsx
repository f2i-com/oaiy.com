import { memo, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { ChevronRight, ChevronsDownUp, ChevronsUpDown, Layers, Package, Search, X } from 'lucide-react';
import type { NodeType, Flow } from 'oaiy-core';
import { useModuleNodes, getNodeColorClasses, getNodeIcon } from '../../hooks/useModuleNodes';
import type { PackageNodeInfo } from '../../hooks/usePackageNodes';
import type { ModuleNodeInfo } from '../../hooks/useModuleNodes';

// Interface for macro node data when adding a macro
export interface MacroNodeData {
  _macroWorkflowId: string;
  _macroName: string;
  _macroInputs: Array<{ id: string; name: string; type: string; required?: boolean; defaultValue?: string }>;
  _macroOutputs: Array<{ id: string; name: string; type: string }>;
}

/**
 * The node palette: a side panel docked beside the canvas (the inspector's
 * sibling on the left), with a search, and the node kinds in collapsible
 * categories with their counts. Click a node to add it at the middle of the
 * canvas, or drag it to where it goes.
 */
interface NodePaletteProps {
  onAddNode: (type: NodeType) => void;
  onAddNodeAtPosition?: (type: NodeType, x: number, y: number) => void;
  /** Hide the palette (its close button). */
  onClose: () => void;
  /** Close after a click adds a node: on a phone, where the panel lies over the canvas. */
  closeOnAdd?: boolean;
  // Macro support
  macros?: Flow[];
  onAddMacro?: (macroData: MacroNodeData) => void;
  onAddMacroAtPosition?: (macroData: MacroNodeData, x: number, y: number) => void;
  // Package node support
  activePackageId?: string | null;
  packageNodes?: PackageNodeInfo[];
}

// Global drag state for cross-component communication
let globalDragState: { isDragging: boolean; nodeType: NodeType | null; macroData: MacroNodeData | null } = {
  isDragging: false,
  nodeType: null,
  macroData: null,
};

// Persist expanded categories across flow navigation (module-level state)
let persistedExpandedCategories: Set<string> = new Set();
let persistedSearchQuery: string = '';

export function getGlobalDragState() {
  return globalDragState;
}

export function clearGlobalDragState() {
  globalDragState = { isDragging: false, nodeType: null, macroData: null };
}

function NodePalette({ onAddNode, onAddNodeAtPosition, onClose, closeOnAdd = false, macros, onAddMacro, onAddMacroAtPosition, activePackageId, packageNodes }: NodePaletteProps) {
  const { groupedNodes, isLoading, error, nodes } = useModuleNodes({
    activePackageId,
    packageNodes,
  });
  const [draggingType, setDraggingType] = useState<NodeType | null>(null);
  const [draggingMacro, setDraggingMacro] = useState<MacroNodeData | null>(null);
  const [dragPosition, setDragPosition] = useState<{ x: number; y: number } | null>(null);
  const dragStartPos = useRef<{ x: number; y: number } | null>(null);
  const mouseDownTime = useRef<number>(0);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(persistedExpandedCategories);
  const [searchQuery, setSearchQuery] = useState(persistedSearchQuery);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Documentation popover state
  const [hoveredNode, setHoveredNode] = useState<ModuleNodeInfo | null>(null);
  const [popoverPosition, setPopoverPosition] = useState<{ x: number; y: number; showAbove?: boolean; showLeft?: boolean } | null>(null);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const nodeElementRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const popoverRef = useRef<HTMLDivElement>(null);

  // Persist state changes
  useEffect(() => {
    persistedExpandedCategories = expandedCategories;
  }, [expandedCategories]);

  useEffect(() => {
    persistedSearchQuery = searchQuery;
  }, [searchQuery]);

  // Toggle category expansion
  const toggleCategory = useCallback((category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }, []);

  // Filter macros based on search query
  const filteredMacros = useMemo(() => {
    if (!macros || macros.length === 0) return [];
    if (!searchQuery.trim()) return macros;

    const query = searchQuery.toLowerCase();
    return macros.filter(macro =>
      macro.name.toLowerCase().includes(query) ||
      macro.description?.toLowerCase().includes(query)
    );
  }, [macros, searchQuery]);

  // Expand all categories
  const expandAll = useCallback(() => {
    const categories = new Set(groupedNodes.map(g => g.category));
    if (macros && macros.length > 0) {
      categories.add('Macros');
    }
    setExpandedCategories(categories);
  }, [groupedNodes, macros]);

  // Collapse all categories
  const collapseAll = useCallback(() => {
    setExpandedCategories(new Set());
  }, []);

  // Filter nodes based on search query
  const filteredGroupedNodes = useMemo(() => {
    if (!searchQuery.trim()) return groupedNodes;

    const query = searchQuery.toLowerCase();
    return groupedNodes
      .map(group => ({
        ...group,
        nodes: group.nodes.filter(node =>
          node.definition.name.toLowerCase().includes(query) ||
          node.definition.description?.toLowerCase().includes(query) ||
          node.definition.id.toLowerCase().includes(query)
        )
      }))
      .filter(group => group.nodes.length > 0);
  }, [groupedNodes, searchQuery]);

  // When searching, auto-expand categories with matches
  // This is intentional: we want to expand categories when search results change
  useEffect(() => {
    if (searchQuery.trim()) {
      const categories = new Set(filteredGroupedNodes.map(g => g.category));
      if (filteredMacros.length > 0) {
        categories.add('Macros');
      }
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: sync state with search results
      setExpandedCategories(categories);
    }
  }, [searchQuery, filteredGroupedNodes, filteredMacros]);

  // Store cleanup function ref to handle unmount during drag
  const cleanupRef = useRef<(() => void) | null>(null);

  // True once a real drag (>5px) occurred during the current interaction; consumed
  // by handleAddNode to suppress only the stray post-drag click. Replaces the old
  // `held > 200ms` heuristic, which silently dropped slow in-place clicks + ALL
  // keyboard activation.
  const dragOccurredRef = useRef(false);
  const handleMouseDown = useCallback((e: React.MouseEvent, type: NodeType) => {
    // Only start drag on left click
    if (e.button !== 0) return;
    dragStartPos.current = { x: e.clientX, y: e.clientY };
    mouseDownTime.current = Date.now();
    dragOccurredRef.current = false;

    // We'll start actual drag after a small movement threshold
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragStartPos.current) return;

      const dx = Math.abs(moveEvent.clientX - dragStartPos.current.x);
      const dy = Math.abs(moveEvent.clientY - dragStartPos.current.y);

      // Start drag if moved more than 5 pixels
      if (dx > 5 || dy > 5) {
        dragOccurredRef.current = true;
        setDraggingType(type);
        setDragPosition({ x: moveEvent.clientX, y: moveEvent.clientY });
        globalDragState = { isDragging: true, nodeType: type, macroData: null };
        document.removeEventListener('mousemove', handleMouseMove);
        document.addEventListener('mousemove', handleDragMove);
        document.addEventListener('mouseup', handleMouseUp);
      }
    };

    const handleDragMove = (moveEvent: MouseEvent) => {
      setDragPosition({ x: moveEvent.clientX, y: moveEvent.clientY });
    };

    const handleMouseUp = (upEvent: MouseEvent) => {
      cleanup();

      if (globalDragState.isDragging && globalDragState.nodeType) {
        // Drop the node at the current position
        if (onAddNodeAtPosition) {
          onAddNodeAtPosition(globalDragState.nodeType, upEvent.clientX, upEvent.clientY);
        }
      }

      setDraggingType(null);
      setDragPosition(null);
      dragStartPos.current = null;
      globalDragState = { isDragging: false, nodeType: null, macroData: null };
    };

    // Cleanup function to remove all listeners
    const cleanup = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mousemove', handleDragMove);
      document.removeEventListener('mouseup', handleMouseUp);
      cleanupRef.current = null;
    };

    // Store cleanup for unmount
    cleanupRef.current = cleanup;

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [onAddNodeAtPosition]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clean up any lingering event listeners
      cleanupRef.current?.();
      globalDragState = { isDragging: false, nodeType: null, macroData: null };
    };
  }, []);

  const handleAddNode = useCallback((type: NodeType) => {
    // Only add on click if not dragging
    if (draggingType) return;
    // Suppress only the stray click that follows a real drag started+ended on the
    // same node (consume the flag). A slow/deliberate in-place click still adds, as
    // does keyboard activation (which resets the flag before calling this).
    if (dragOccurredRef.current) {
      dragOccurredRef.current = false;
      return;
    }
    onAddNode(type);
    if (closeOnAdd) onClose();
  }, [draggingType, onAddNode, onClose, closeOnAdd]);

  // Get the label for the dragging type
  const getDragLabel = useCallback((type: NodeType) => {
    const node = nodes.find(n => n.definition.id === type);
    return node?.definition.name || type;
  }, [nodes]);

  // Handle node hover for documentation popover
  const handleNodeMouseEnter = useCallback((node: ModuleNodeInfo, element: HTMLDivElement) => {
    // Clear any existing timeout
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }

    // Only show popover if the node has documentation
    if (!node.definition.doc && !node.definition.description) {
      return;
    }

    // Set a small delay before showing the popover
    hoverTimeoutRef.current = setTimeout(() => {
      const rect = element.getBoundingClientRect();
      const popoverWidth = 320; // max-w-sm is 384px but content is usually less
      const estimatedPopoverHeight = 300; // Estimate for initial positioning

      // Check if popover should appear on the left
      const showLeft = rect.right + popoverWidth + 16 > window.innerWidth;

      // Check if popover should appear above the node
      const showAbove = rect.top + estimatedPopoverHeight > window.innerHeight - 20;

      setHoveredNode(node);
      setPopoverPosition({
        x: showLeft ? rect.left - 8 : rect.right + 8,
        y: rect.top,
        showAbove,
        showLeft,
      });
    }, 400);
  }, []);

  const handleNodeMouseLeave = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setHoveredNode(null);
    setPopoverPosition(null);
  }, []);

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  // Adjust popover position after it renders based on actual size
  useEffect(() => {
    if (popoverRef.current && popoverPosition) {
      const popover = popoverRef.current;
      const popoverRect = popover.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const windowWidth = window.innerWidth;

      let newY = popoverPosition.y;
      let newX = popoverPosition.x;

      // Adjust vertical position if popover overflows bottom
      if (popoverRect.bottom > windowHeight - 10) {
        // Position so bottom of popover is 10px from bottom of screen
        newY = windowHeight - popoverRect.height - 10;
      }

      // Ensure top doesn't go above viewport
      if (newY < 10) {
        newY = 10;
      }

      // Adjust horizontal position for left-side popovers
      if (popoverPosition.showLeft) {
        newX = popoverPosition.x - popoverRect.width;
        if (newX < 10) {
          newX = 10;
        }
      } else {
        // Ensure right-side popover doesn't overflow
        if (newX + popoverRect.width > windowWidth - 10) {
          newX = windowWidth - popoverRect.width - 10;
        }
      }

      // Only update if position actually changed
      if (newY !== popoverPosition.y || newX !== popoverPosition.x) {
        setPopoverPosition(prev => prev ? { ...prev, x: newX, y: newY } : null);
      }
    }
  }, [hoveredNode]); // Re-run when hoveredNode changes (popover content changes)

  // Helper to create macro node data from a flow
  const createMacroNodeData = useCallback((macro: Flow): MacroNodeData => {
    return {
      _macroWorkflowId: macro.id,
      _macroName: macro.name,
      _macroInputs: macro.macroMetadata?.inputs.map(inp => ({
        id: inp.id,
        name: inp.name,
        type: inp.type,
        required: inp.required,
        defaultValue: inp.defaultValue,
      })) || [],
      _macroOutputs: macro.macroMetadata?.outputs.map(out => ({
        id: out.id,
        name: out.name,
        type: out.type,
      })) || [],
    };
  }, []);

  // Handle macro mouse down (for drag-and-drop)
  const handleMacroMouseDown = useCallback((e: React.MouseEvent, macro: Flow) => {
    if (e.button !== 0) return;
    dragStartPos.current = { x: e.clientX, y: e.clientY };
    mouseDownTime.current = Date.now();
    dragOccurredRef.current = false;

    const macroData = createMacroNodeData(macro);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragStartPos.current) return;

      const dx = Math.abs(moveEvent.clientX - dragStartPos.current.x);
      const dy = Math.abs(moveEvent.clientY - dragStartPos.current.y);

      if (dx > 5 || dy > 5) {
        dragOccurredRef.current = true;
        setDraggingMacro(macroData);
        setDragPosition({ x: moveEvent.clientX, y: moveEvent.clientY });
        globalDragState = { isDragging: true, nodeType: null, macroData };
        document.removeEventListener('mousemove', handleMouseMove);
        document.addEventListener('mousemove', handleDragMove);
        document.addEventListener('mouseup', handleMouseUp);
      }
    };

    const handleDragMove = (moveEvent: MouseEvent) => {
      setDragPosition({ x: moveEvent.clientX, y: moveEvent.clientY });
    };

    const handleMouseUp = (upEvent: MouseEvent) => {
      cleanup();

      if (globalDragState.isDragging && globalDragState.macroData) {
        if (onAddMacroAtPosition) {
          onAddMacroAtPosition(globalDragState.macroData, upEvent.clientX, upEvent.clientY);
        }
      }

      setDraggingMacro(null);
      setDragPosition(null);
      dragStartPos.current = null;
      globalDragState = { isDragging: false, nodeType: null, macroData: null };
    };

    const cleanup = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mousemove', handleDragMove);
      document.removeEventListener('mouseup', handleMouseUp);
      cleanupRef.current = null;
    };

    cleanupRef.current = cleanup;

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [createMacroNodeData, onAddMacroAtPosition]);

  // Handle macro click (add at default position)
  const handleAddMacro = useCallback((macro: Flow) => {
    if (draggingMacro) return;
    if (dragOccurredRef.current) {
      dragOccurredRef.current = false;
      return;
    }
    if (onAddMacro) {
      onAddMacro(createMacroNodeData(macro));
    }
    if (closeOnAdd) onClose();
  }, [draggingMacro, onAddMacro, createMacroNodeData, onClose, closeOnAdd]);

  const total = filteredGroupedNodes.reduce((n, g) => n + g.nodes.length, 0) + filteredMacros.length;

  return (
    <aside className="oaiy-side left" aria-label="Node palette" data-testid="node-palette">
      <div className="oaiy-side-head">
        <h2>
          Nodes <small>{total}</small>
        </h2>
        <div className="oaiy-side-tools">
          <button type="button" onClick={expandAll} className="oaiy-icon-btn" title="Open every category" aria-label="Expand all categories">
            <ChevronsUpDown size={15} />
          </button>
          <button type="button" onClick={collapseAll} className="oaiy-icon-btn" title="Close every category" aria-label="Collapse all categories">
            <ChevronsDownUp size={15} />
          </button>
          <button type="button" onClick={onClose} className="oaiy-icon-btn" title="Hide the nodes" aria-label="Close node palette">
            <X size={15} />
          </button>
        </div>
      </div>

      <div className="oaiy-side-sub">
        <div className="oaiy-search">
          <Search size={14} />
          <input
            ref={searchInputRef}
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape' && searchQuery) { e.stopPropagation(); setSearchQuery(''); } }}
            placeholder="Search nodes"
            aria-label="Search nodes"
            className="oaiy-input"
          />
          {searchQuery && (
            <button type="button" onClick={() => setSearchQuery('')} aria-label="Clear search">
              <X size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Node List */}
      <div className="oaiy-side-body py-1">
        {isLoading ? (
          <div className="oaiy-empty bare">
            <span className="oaiy-spinner" aria-hidden="true" />
            <p className="oaiy-empty-text">Loading nodes…</p>
          </div>
        ) : error ? (
          <div className="oaiy-empty bare">
            <p className="oaiy-empty-title">The nodes did not load</p>
            <p className="oaiy-empty-text">{error}</p>
          </div>
        ) : filteredGroupedNodes.length === 0 && filteredMacros.length === 0 ? (
          <div className="oaiy-empty bare">
            <Search size={22} />
            <p className="oaiy-empty-title">{searchQuery ? 'No node matches' : 'No nodes'}</p>
            <p className="oaiy-empty-text">{searchQuery ? 'Try another word.' : 'None of the modules offers a node.'}</p>
          </div>
        ) : (
          filteredGroupedNodes.map((group) => {
            const isExpanded = expandedCategories.has(group.category);
            const isPackageCategory = group.category === 'Package' || group.nodes.some(n => n.isPackageNode);
            return (
              <div key={group.category} className={`oaiy-cat${isPackageCategory ? ' pkg' : ''}`}>
                <button
                  type="button"
                  onClick={() => toggleCategory(group.category)}
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${group.label} category`}
                >
                  <ChevronRight size={13} aria-hidden="true" />
                  {group.category === 'Package' && <Package size={13} aria-hidden="true" />}
                  <span title={group.label}>{group.label}</span>
                  <small>{group.nodes.length}</small>
                </button>

                {isExpanded && (
                  <ul>
                    {group.nodes.map((node) => {
                      const colors = getNodeColorClasses(node.definition.color);
                      const nodeType = node.definition.id as NodeType;
                      return (
                        <li key={node.definition.id}>
                          <div
                            ref={(el) => {
                              if (el) nodeElementRefs.current.set(node.definition.id, el);
                            }}
                            role="button"
                            tabIndex={0}
                            onMouseDown={(e) => handleMouseDown(e, nodeType)}
                            onClick={() => handleAddNode(nodeType)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault();
                                dragOccurredRef.current = false; // keyboard is never a drag
                                handleAddNode(nodeType);
                              }
                            }}
                            onMouseEnter={(e) => handleNodeMouseEnter(node, e.currentTarget)}
                            onMouseLeave={handleNodeMouseLeave}
                            onFocus={(e) => handleNodeMouseEnter(node, e.currentTarget)}
                            onBlur={handleNodeMouseLeave}
                            aria-label={`Add ${node.definition.name} node`}
                            className="oaiy-node-item"
                          >
                            {/* The node kind's own colour, as on the canvas. */}
                            <span className={`oaiy-node-icon ${colors.bg} ${colors.border} ${colors.text}`}>
                              {getNodeIcon(node.definition.icon)}
                            </span>
                            <span className="oaiy-node-text">
                              <strong>{node.definition.name}</strong>
                              <small>{node.definition.description}</small>
                            </span>
                            {node.isPackageNode && <span className="oaiy-node-tag" title="From a package">PKG</span>}
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            );
          })
        )}

        {/* Macros: flows saved to be used as nodes. */}
        {filteredMacros.length > 0 && (
          <div className="oaiy-cat">
            <button
              type="button"
              onClick={() => toggleCategory('Macros')}
              aria-expanded={expandedCategories.has('Macros')}
              aria-label={`${expandedCategories.has('Macros') ? 'Collapse' : 'Expand'} Macros category`}
            >
              <ChevronRight size={13} aria-hidden="true" />
              <span>Macros</span>
              <small>{filteredMacros.length}</small>
            </button>

            {expandedCategories.has('Macros') && (
              <ul>
                {filteredMacros.map((macro) => {
                  const inputCount = macro.macroMetadata?.inputs.length || 0;
                  const outputCount = macro.macroMetadata?.outputs.length || 0;
                  return (
                    <li key={macro.id}>
                      <div
                        role="button"
                        tabIndex={0}
                        onMouseDown={(e) => handleMacroMouseDown(e, macro)}
                        onClick={() => handleAddMacro(macro)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            dragOccurredRef.current = false; // keyboard is never a drag
                            handleAddMacro(macro);
                          }
                        }}
                        aria-label={`Add ${macro.name} macro`}
                        className="oaiy-node-item"
                      >
                        <span className="oaiy-node-icon border-signal-magenta/40 bg-signal-magenta/10 text-signal-magenta">
                          <Layers size={15} />
                        </span>
                        <span className="oaiy-node-text">
                          <strong>{macro.name}</strong>
                          <small>
                            {inputCount} input{inputCount !== 1 ? 's' : ''}, {outputCount} output{outputCount !== 1 ? 's' : ''}
                          </small>
                        </span>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}
      </div>

      <div className="oaiy-side-foot">Click a node to add it, or drag it onto the canvas.</div>

      {/* Drag preview - rendered via portal to avoid transform issues */}
      {(draggingType || draggingMacro) && dragPosition && createPortal(
        <div className="oaiy-drag-ghost" style={{ left: dragPosition.x + 12, top: dragPosition.y + 12 }}>
          {draggingMacro ? draggingMacro._macroName : (draggingType ? getDragLabel(draggingType) : '')}
        </div>,
        document.body
      )}

      {/* Documentation popover - rendered via portal */}
      {hoveredNode && popoverPosition && !draggingType && createPortal(
        <div
          ref={popoverRef}
          className="oaiy-popover"
          style={{
            left: popoverPosition.showLeft ? 'auto' : popoverPosition.x,
            right: popoverPosition.showLeft ? window.innerWidth - popoverPosition.x : 'auto',
            top: popoverPosition.y,
            maxHeight: 'calc(100vh - 20px)',
          }}
          onMouseEnter={() => {
            // Keep popover open when hovering over it
            if (hoverTimeoutRef.current) {
              clearTimeout(hoverTimeoutRef.current);
              hoverTimeoutRef.current = null;
            }
          }}
          onMouseLeave={handleNodeMouseLeave}
        >
          <div className="mb-1.5 flex items-center gap-2">
            <span className={`oaiy-node-icon grid h-6 w-6 place-items-center rounded-[var(--r-sm)] border ${getNodeColorClasses(hoveredNode.definition.color).bg} ${getNodeColorClasses(hoveredNode.definition.color).border} ${getNodeColorClasses(hoveredNode.definition.color).text}`}>
              {getNodeIcon(hoveredNode.definition.icon)}
            </span>
            <h4 className="m-0">{hoveredNode.definition.name}</h4>
          </div>
          {hoveredNode.definition.doc ? (
            <p className="m-0 whitespace-pre-wrap">{hoveredNode.definition.doc}</p>
          ) : hoveredNode.definition.description ? (
            <p className="m-0">{hoveredNode.definition.description}</p>
          ) : null}
          {hoveredNode.definition.inputs.length > 0 && (
            <>
              <span className="oaiy-label">Inputs</span>
              <div className="oaiy-chips">
                {hoveredNode.definition.inputs.map(inp => <span key={inp.id}>{inp.name}</span>)}
              </div>
            </>
          )}
          {hoveredNode.definition.outputs.length > 0 && (
            <>
              <span className="oaiy-label">Outputs</span>
              <div className="oaiy-chips">
                {hoveredNode.definition.outputs.map(out => <span key={out.id}>{out.name}</span>)}
              </div>
            </>
          )}
        </div>,
        document.body
      )}
    </aside>
  );
}

export default memo(NodePalette);
