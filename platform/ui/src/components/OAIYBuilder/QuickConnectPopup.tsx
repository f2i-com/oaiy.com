/**
 * QuickConnectPopup Component
 *
 * A popup that appears when dragging a connection, showing compatible nodes
 * that can be connected to the source handle.
 * Extracted from OAIYBuilder.tsx for maintainability.
 */

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Search, X, Zap } from 'lucide-react';
import { getNodeColorClasses, getNodeIcon, type ModuleNodeInfo } from '../../hooks/useModuleNodes';
import type { NodeType } from 'oaiy-core';

interface QuickConnectState {
  isVisible: boolean;
  position: { x: number; y: number };
  searchQuery: string;
}

interface QuickConnectPopupProps {
  /** Current quick connect state */
  state: QuickConnectState;
  /** Position of the source handle in screen coordinates */
  handlePosition: { x: number; y: number } | null;
  /** List of nodes compatible with the source handle */
  compatibleNodes: ModuleNodeInfo[];
  /** Callback when a node is selected */
  onNodeSelect: (nodeType: NodeType) => void;
  /** Callback when the popup should close */
  onClose: () => void;
  /** Callback to update search query */
  onSearchChange: (query: string) => void;
}

const MAX_SHOWN = 15;

export function QuickConnectPopup({
  state,
  handlePosition,
  compatibleNodes,
  onNodeSelect,
  onClose,
  onSearchChange,
}: QuickConnectPopupProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const itemRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [activeIndex, setActiveIndex] = useState(0);

  const matchingNodes = compatibleNodes.filter((node) => {
    if (!state.searchQuery) return true;
    const q = state.searchQuery.toLowerCase();
    return (
      node.definition.name.toLowerCase().includes(q) ||
      node.definition.id.toLowerCase().includes(q) ||
      node.category.toLowerCase().includes(q)
    );
  });
  const totalMatches = matchingNodes.length;
  const filteredNodes = matchingNodes.slice(0, MAX_SHOWN);

  useEffect(() => {
    setActiveIndex(0);
  }, [state.searchQuery, state.isVisible, compatibleNodes]);

  useEffect(() => {
    const activeEl = itemRefs.current[activeIndex];
    if (activeEl && typeof activeEl.scrollIntoView === 'function') {
      activeEl.scrollIntoView({ block: 'nearest' });
    }
  }, [activeIndex]);

  if (!state.isVisible) return null;

  const popupLeft = Math.max(8, Math.min(state.position.x, window.innerWidth - 300));
  const popupTop = Math.max(8, Math.min(state.position.y, window.innerHeight - 340));

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      onClose();
      return;
    }
    if (filteredNodes.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((prev) => (prev + 1) % filteredNodes.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((prev) => (prev - 1 + filteredNodes.length) % filteredNodes.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const chosen = filteredNodes[activeIndex] ?? filteredNodes[0];
      if (chosen) {
        onNodeSelect(chosen.definition.id as NodeType);
      }
    }
  };

  const activeOptionId = filteredNodes[activeIndex]
    ? `qc-opt-${filteredNodes[activeIndex].definition.id}`
    : undefined;

  const content = (
    <>
      {/* Connection line from source handle to popup */}
      {handlePosition && (
        <svg
          className="fixed inset-0 z-[9990] pointer-events-none"
          style={{ width: '100vw', height: '100vh' }}
          aria-hidden="true"
        >
          <line
            x1={handlePosition.x}
            y1={handlePosition.y}
            x2={popupLeft + 8}
            y2={popupTop + 40}
            style={{ stroke: 'rgb(var(--accent-primary))' }}
            strokeWidth="3"
            strokeDasharray="8 4"
            className="animate-pulse"
          />
          <circle
            cx={handlePosition.x}
            cy={handlePosition.y}
            r="6"
            style={{ fill: 'rgb(var(--accent-primary))' }}
          />
        </svg>
      )}

      {/* Quick-Connect Popup: the palette's rows, in a card at the pointer. */}
      <div
        data-quick-connect-popup
        role="dialog"
        aria-label="Connect a new node"
        className="fixed z-[10005] flex max-h-80 w-72 flex-col overflow-hidden rounded-[var(--r-ctl)] border bg-surface-secondary shadow-[var(--shadow-lg)]"
        style={{
          left: popupLeft,
          top: popupTop,
          borderColor: 'rgb(var(--accent-primary) / 0.5)',
        }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="flex flex-col gap-2 border-b border-edge-secondary px-3 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-[12px] font-semibold text-accent">
              <Zap size={13} aria-hidden="true" />
              Connect a new node
            </div>
            <button type="button" onClick={onClose} className="oaiy-icon-btn sm" title="Close" aria-label="Close">
              <X size={13} />
            </button>
          </div>
          <div className="oaiy-search">
            <Search size={13} />
            <input
              ref={inputRef}
              type="text"
              placeholder="Search nodes"
              aria-label="Search nodes"
              aria-autocomplete="list"
              aria-controls="qc-node-list"
              aria-activedescendant={activeOptionId}
              className="oaiy-input oaiy-input-sm"
              value={state.searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              autoFocus
            />
            {state.searchQuery && (
              <button
                type="button"
                onClick={() => {
                  onSearchChange('');
                  inputRef.current?.focus();
                }}
                aria-label="Clear search"
                title="Clear search"
              >
                <X size={12} />
              </button>
            )}
          </div>
        </div>

        {/* Compatible Nodes List */}
        <div id="qc-node-list" role="listbox" aria-label="Compatible nodes" className="max-h-52 overflow-y-auto p-1.5">
          {filteredNodes.map((node, index) => {
            const colorClasses = getNodeColorClasses(node.definition.color);
            const isActive = index === activeIndex;
            return (
              <button
                ref={(el) => { itemRefs.current[index] = el; }}
                id={`qc-opt-${node.definition.id}`}
                type="button"
                role="option"
                aria-selected={isActive}
                key={node.definition.id}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => onNodeSelect(node.definition.id as NodeType)}
                className={`oaiy-node-item w-full text-left${isActive ? ' bg-accent/15' : ''}`}
              >
                {/* The node kind's own colour, as on the canvas. */}
                <span className={`oaiy-node-icon ${colorClasses.bg} ${colorClasses.border} ${colorClasses.text}`}>
                  {getNodeIcon(node.definition.icon)}
                </span>
                <span className="oaiy-node-text">
                  <strong>{node.definition.name}</strong>
                  <small>{node.category}</small>
                </span>
              </button>
            );
          })}
          {filteredNodes.length === 0 && (
            <div className="px-3 py-4 text-center text-[12.5px] text-content-faint">
              {state.searchQuery ? 'No node matches' : 'No node can take this connection'}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div className="flex items-center justify-between gap-2 border-t border-edge-secondary px-3 py-1.5 text-[11px] text-content-faint">
          {totalMatches > MAX_SHOWN ? (
            <span>Showing {MAX_SHOWN} of {totalMatches} — refine search</span>
          ) : (
            <span>↑↓ choose · Enter connect · Esc cancel</span>
          )}
        </div>
      </div>
    </>
  );

  return typeof document !== 'undefined' ? createPortal(content, document.body) : content;
}
