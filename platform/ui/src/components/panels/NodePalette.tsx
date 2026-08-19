import { memo, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { createPortal } from 'react-dom';
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

interface NodePaletteProps {
  onAddNode: (type: NodeType) => void;
  onAddNodeAtPosition?: (type: NodeType, x: number, y: number) => void;
  isOpen: boolean;
  onClose: () => void;
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

function NodePalette({ onAddNode, onAddNodeAtPosition, isOpen, onClose, macros, onAddMacro, onAddMacroAtPosition, activePackageId, packageNodes }: NodePaletteProps) {
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
    // Close on mobile after adding
    if (window.innerWidth < 768) {
      onClose();
    }
  }, [draggingType, onAddNode, onClose]);

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
    if (window.innerWidth < 768) {
      onClose();
    }
  }, [draggingMacro, onAddMacro, createMacroNodeData, onClose]);

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={onClose}
        />
      )}

      {/* Palette panel */}
      <div
        className={`
          fixed lg:relative z-50 lg:z-auto
          h-full max-h-screen bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col
          transition-transform duration-300 ease-in-out
          w-full lg:w-48 xl:w-64
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Header */}
        <div className="p-3 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h2 className="text-slate-700 dark:text-slate-200 font-semibold text-sm">Node Palette</h2>
            </div>
            {/* Close button for mobile */}
            <button
              onClick={onClose}
              className="lg:hidden p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
              aria-label="Close node palette"
            >
              <svg className="w-5 h-5 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Search box */}
          <div className="relative mb-2">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search nodes..."
              aria-label="Search nodes"
              className="input pl-9 pr-9 py-2"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 rounded transition-all"
                aria-label="Clear search"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          {/* Expand/Collapse buttons */}
          <div className="flex gap-1.5">
            <button
              onClick={expandAll}
              className="btn btn-ghost btn-sm flex-1"
              title="Expand all categories"
            >
              Expand All
            </button>
            <button
              onClick={collapseAll}
              className="btn btn-ghost btn-sm flex-1"
              title="Collapse all categories"
            >
              Collapse All
            </button>
          </div>
        </div>

        {/* Node List */}
        <div className="flex-1 overflow-y-auto p-2 sm:p-3 space-y-2">
          {isLoading ? (
            <div className="empty-state py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-slate-600" style={{ borderTopColor: 'rgb(var(--accent-primary))' }}></div>
              <p className="text-slate-500 text-sm mt-3">Loading nodes...</p>
            </div>
          ) : error ? (
            <div className="empty-state py-8">
              <svg className="empty-state-icon text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <p className="empty-state-title text-red-400">Failed to load nodes</p>
              <p className="empty-state-description">{error}</p>
            </div>
          ) : filteredGroupedNodes.length === 0 ? (
            <div className="empty-state py-8 animate-fadeIn">
              <svg className="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p className="empty-state-title">
                {searchQuery ? 'No matches found' : 'No nodes available'}
              </p>
              <p className="empty-state-description">
                {searchQuery ? `Try searching for something else` : 'Check your module configuration'}
              </p>
            </div>
          ) : (
            filteredGroupedNodes.map((group) => {
              const isExpanded = expandedCategories.has(group.category);
              const isPackageCategory = group.category === 'Package';
              const hasPackageNodes = group.nodes.some(n => n.isPackageNode);
              return (
                <div key={group.category} className={`space-y-1 ${isPackageCategory ? 'mb-3' : ''}`}>
                  {/* Special header for Package category */}
                  {isPackageCategory ? (
                    <div className="bg-gradient-to-r from-purple-100 to-purple-50 dark:from-purple-900/40 dark:to-purple-900/20 rounded-lg p-2 mb-2 border border-purple-200 dark:border-purple-500/30">
                      <button
                        onClick={() => toggleCategory(group.category)}
                        className="w-full flex items-center gap-2 text-purple-700 dark:text-purple-300 hover:text-purple-800 dark:hover:text-purple-200 transition-colors"
                        aria-expanded={isExpanded}
                        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${group.label} category`}
                      >
                        <svg
                          className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          aria-hidden="true"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <div className="w-6 h-6 rounded-md bg-purple-500/20 flex items-center justify-center">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                          </svg>
                        </div>
                        <span className="text-xs font-semibold uppercase tracking-wide whitespace-nowrap truncate min-w-0" title={group.label}>{group.label}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-purple-200 dark:bg-purple-500/30 text-purple-600 dark:text-purple-300 font-medium">{group.nodes.length}</span>
                      </button>
                    </div>
                  ) : (
                  /* Category Header */
                  <button
                    onClick={() => toggleCategory(group.category)}
                    className={`w-full flex items-center gap-2 px-2 py-1.5 text-left transition-colors ${
                      hasPackageNodes
                        ? 'text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-200'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                    }`}
                    aria-expanded={isExpanded}
                    aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${group.label} category`}
                  >
                    <svg
                      className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                    <span className="text-xs font-medium uppercase tracking-wide whitespace-nowrap truncate min-w-0" title={group.label}>{group.label}</span>
                    <span className="text-slate-500 dark:text-slate-600 text-xs flex-shrink-0">({group.nodes.length})</span>
                  </button>
                  )}

                  {/* Category Nodes */}
                  {isExpanded && (
                    <div className="space-y-2 pl-1 animate-slideDown">
                      {group.nodes.map((node, index) => {
                        const colors = getNodeColorClasses(node.definition.color);
                        const nodeType = node.definition.id as NodeType;
                        return (
                          <div
                            key={node.definition.id}
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
                            className={`
                              ${colors.bg} ${colors.border}
                              border rounded-lg p-3 cursor-grab active:cursor-grabbing
                              hover:brightness-110 active:scale-[0.98] transition-all duration-150
                              flex items-center gap-3
                              select-none group
                              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-slate-900
                            `}
                            style={{ animationDelay: `${index * 30}ms` }}
                          >
                            <div className={`w-9 h-9 rounded-lg ${colors.bg} ${colors.border} border flex items-center justify-center ${colors.text} flex-shrink-0 transition-transform group-hover:scale-105 relative`}>
                              {getNodeIcon(node.definition.icon)}
                              {/* Package node indicator */}
                              {node.isPackageNode && (
                                <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-500 rounded-full border border-slate-900" title="Package node" />
                              )}
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className={`${colors.text} font-medium text-sm truncate flex items-center gap-1.5`}>
                                {node.definition.name}
                                {node.isPackageNode && (
                                  <span className="text-[9px] px-1 py-0.5 bg-purple-100 dark:bg-purple-500/20 text-purple-600 dark:text-purple-400 rounded font-normal">PKG</span>
                                )}
                              </div>
                              <div className="text-slate-500 dark:text-slate-400 text-xs truncate hidden sm:block leading-relaxed">{node.definition.description}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* Macros Category */}
          {filteredMacros.length > 0 && (
            <div className="space-y-1">
              {/* Macros Category Header */}
              <button
                onClick={() => toggleCategory('Macros')}
                className="w-full flex items-center gap-2 px-2 py-1.5 text-violet-600 dark:text-violet-400 hover:text-violet-700 dark:hover:text-violet-200 transition-colors"
                aria-expanded={expandedCategories.has('Macros')}
                aria-label={`${expandedCategories.has('Macros') ? 'Collapse' : 'Expand'} Macros category`}
              >
                <svg
                  className={`w-3 h-3 transition-transform ${expandedCategories.has('Macros') ? 'rotate-90' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                </svg>
                <span className="text-xs font-medium uppercase tracking-wide">Macros</span>
                <span className="text-slate-500 dark:text-slate-600 text-xs">({filteredMacros.length})</span>
              </button>

              {/* Macros List */}
              {expandedCategories.has('Macros') && (
                <div className="space-y-2 pl-1 animate-slideDown">
                  {filteredMacros.map((macro, index) => {
                    const inputCount = macro.macroMetadata?.inputs.length || 0;
                    const outputCount = macro.macroMetadata?.outputs.length || 0;
                    return (
                      <div
                        key={macro.id}
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
                        className={`
                          bg-violet-100 dark:bg-violet-900/30 border-violet-300 dark:border-violet-600
                          border rounded-lg p-3 cursor-grab active:cursor-grabbing
                          hover:brightness-110 active:scale-[0.98] transition-all duration-150
                          flex items-center gap-3
                          select-none group
                          focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-slate-900
                        `}
                        style={{ animationDelay: `${index * 30}ms` }}
                      >
                        <div className="w-9 h-9 rounded-lg bg-violet-200 dark:bg-violet-900/30 border-violet-300 dark:border-violet-600 border flex items-center justify-center text-violet-600 dark:text-violet-400 flex-shrink-0 transition-transform group-hover:scale-105">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                          </svg>
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-violet-700 dark:text-violet-400 font-medium text-sm truncate">{macro.name}</div>
                          <div className="text-slate-500 dark:text-slate-500 text-xs truncate hidden sm:block leading-relaxed">
                            {inputCount} input{inputCount !== 1 ? 's' : ''}, {outputCount} output{outputCount !== 1 ? 's' : ''}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Drag preview - rendered via portal to avoid transform issues */}
        {(draggingType || draggingMacro) && dragPosition && createPortal(
          <div
            className={`fixed z-[9999] pointer-events-none rounded-lg px-3 py-2 shadow-xl ${
              draggingMacro
                ? 'bg-violet-100 dark:bg-violet-900/80 border border-violet-400 dark:border-violet-600'
                : 'bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600'
            }`}
            style={{
              left: dragPosition.x + 12,
              top: dragPosition.y + 12,
            }}
          >
            <span className={`text-sm font-medium ${draggingMacro ? 'text-violet-700 dark:text-violet-200' : 'text-slate-700 dark:text-slate-200'}`}>
              {draggingMacro ? draggingMacro._macroName : (draggingType ? getDragLabel(draggingType) : '')}
            </span>
          </div>,
          document.body
        )}

        {/* Documentation popover - rendered via portal */}
        {hoveredNode && popoverPosition && !draggingType && createPortal(
          <div
            ref={popoverRef}
            className="fixed z-[9998] bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg shadow-xl max-w-sm animate-fadeIn"
            style={{
              left: popoverPosition.showLeft ? 'auto' : popoverPosition.x,
              right: popoverPosition.showLeft ? window.innerWidth - popoverPosition.x : 'auto',
              top: popoverPosition.y,
              maxHeight: 'calc(100vh - 20px)',
              overflowY: 'auto',
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
            <div className="p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-6 h-6 rounded flex items-center justify-center ${getNodeColorClasses(hoveredNode.definition.color).bg} ${getNodeColorClasses(hoveredNode.definition.color).text}`}>
                  {getNodeIcon(hoveredNode.definition.icon)}
                </div>
                <h4 className="text-slate-700 dark:text-slate-200 font-semibold text-sm">{hoveredNode.definition.name}</h4>
              </div>
              {hoveredNode.definition.doc ? (
                <p className="text-slate-600 dark:text-slate-400 text-xs leading-relaxed whitespace-pre-wrap">
                  {hoveredNode.definition.doc}
                </p>
              ) : hoveredNode.definition.description ? (
                <p className="text-slate-600 dark:text-slate-400 text-xs leading-relaxed">
                  {hoveredNode.definition.description}
                </p>
              ) : null}
              {hoveredNode.definition.inputs.length > 0 && (
                <div className="mt-2 pt-2 border-t border-slate-300 dark:border-slate-700">
                  <span className="text-slate-500 text-[10px] uppercase tracking-wide">Inputs</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {hoveredNode.definition.inputs.map(inp => (
                      <span key={inp.id} className="text-[10px] px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded">
                        {inp.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {hoveredNode.definition.outputs.length > 0 && (
                <div className="mt-2 pt-2 border-t border-slate-300 dark:border-slate-700">
                  <span className="text-slate-500 text-[10px] uppercase tracking-wide">Outputs</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {hoveredNode.definition.outputs.map(out => (
                      <span key={out.id} className="text-[10px] px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded">
                        {out.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>,
          document.body
        )}

        {/* Footer */}
        <div className="p-2 sm:p-3 border-t border-slate-800">
          <p className="text-slate-600 text-[10px] text-center hidden sm:block">
            Click or drag to add nodes
          </p>
          <p className="text-slate-600 text-[10px] text-center sm:hidden">
            Tap to add nodes
          </p>
        </div>
      </div>
    </>
  );
}

export default memo(NodePalette);
