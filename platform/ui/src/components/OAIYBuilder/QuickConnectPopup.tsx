/**
 * QuickConnectPopup Component
 *
 * A popup that appears when dragging a connection, showing compatible nodes
 * that can be connected to the source handle.
 * Extracted from OAIYBuilder.tsx for maintainability.
 */

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

export function QuickConnectPopup({
  state,
  handlePosition,
  compatibleNodes,
  onNodeSelect,
  onClose,
  onSearchChange,
}: QuickConnectPopupProps) {
  if (!state.isVisible) return null;

  const filteredNodes = compatibleNodes
    .filter(node =>
      !state.searchQuery ||
      node.definition.name.toLowerCase().includes(state.searchQuery.toLowerCase()) ||
      node.definition.id.toLowerCase().includes(state.searchQuery.toLowerCase())
    )
    .slice(0, 15); // Limit to 15 items

  return (
    <>
      {/* Connection line from source handle to popup */}
      {handlePosition && (
        <svg
          className="fixed inset-0 z-40 pointer-events-none"
          style={{ width: '100vw', height: '100vh' }}
        >
          <line
            x1={handlePosition.x}
            y1={handlePosition.y}
            x2={Math.min(state.position.x, window.innerWidth - 300) + 8}
            y2={Math.min(state.position.y, window.innerHeight - 340) + 40}
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
        className="fixed z-50 flex max-h-80 w-72 flex-col overflow-hidden rounded-[var(--r-ctl)] border bg-surface-secondary shadow-[var(--shadow-lg)]"
        style={{
          left: Math.min(state.position.x, window.innerWidth - 300),
          top: Math.min(state.position.y, window.innerHeight - 340),
          borderColor: 'rgb(var(--accent-primary) / 0.5)',
        }}
        onClick={(e) => e.stopPropagation()}
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
              type="text"
              placeholder="Search nodes"
              aria-label="Search nodes"
              className="oaiy-input oaiy-input-sm"
              value={state.searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              autoFocus
            />
          </div>
        </div>

        {/* Compatible Nodes List */}
        <div className="max-h-52 overflow-y-auto p-1.5">
          {filteredNodes.map(node => {
            const colorClasses = getNodeColorClasses(node.definition.color);
            return (
              <button
                type="button"
                key={node.definition.id}
                onClick={() => onNodeSelect(node.definition.id as NodeType)}
                className="oaiy-node-item w-full text-left"
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
        <div className="border-t border-edge-secondary px-3 py-1.5 text-[11px] text-content-faint">
          Click outside to cancel
        </div>
      </div>
    </>
  );
}
