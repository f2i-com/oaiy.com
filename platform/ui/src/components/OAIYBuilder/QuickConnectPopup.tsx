/**
 * QuickConnectPopup Component
 *
 * A popup that appears when dragging a connection, showing compatible nodes
 * that can be connected to the source handle.
 * Extracted from OAIYBuilder.tsx for maintainability.
 */

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

      {/* Quick-Connect Popup */}
      <div
        data-quick-connect-popup
        className="fixed z-50 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm border rounded-lg shadow-xl w-72 max-h-80 overflow-hidden"
        style={{
          left: Math.min(state.position.x, window.innerWidth - 300),
          top: Math.min(state.position.y, window.innerHeight - 340),
          borderColor: 'rgb(var(--accent-primary) / 0.5)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-3 py-2 border-b border-slate-300 dark:border-slate-700 bg-slate-100/80 dark:bg-slate-900/50">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-2 text-xs text-blue-400">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Quick Connect
            </div>
            <button
              onClick={onClose}
              className="text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              title="Close"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <input
            type="text"
            placeholder="Search nodes..."
            className="w-full px-2 py-1.5 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded text-sm text-slate-700 dark:text-slate-200 placeholder-slate-400 focus:outline-none focus:border-blue-500"
            value={state.searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            autoFocus
          />
        </div>

        {/* Compatible Nodes List */}
        <div className="max-h-52 overflow-y-auto">
          {filteredNodes.map(node => {
            const colorClasses = getNodeColorClasses(node.definition.color);
            return (
              <button
                key={node.definition.id}
                onClick={() => onNodeSelect(node.definition.id as NodeType)}
                className="w-full px-3 py-2 text-left hover:bg-slate-200/50 dark:hover:bg-slate-700/50 flex items-center gap-2 transition-colors"
              >
                <span className={`${colorClasses.text}`}>
                  {getNodeIcon(node.definition.icon)}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-slate-700 dark:text-slate-200 truncate">{node.definition.name}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 truncate">{node.category}</div>
                </div>
              </button>
            );
          })}
          {filteredNodes.length === 0 && (
            <div className="px-3 py-4 text-sm text-slate-500 dark:text-slate-400 text-center">
              {state.searchQuery ? 'No nodes match your search' : 'No compatible nodes found'}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div className="px-3 py-1.5 border-t border-slate-300 dark:border-slate-700 bg-slate-100/50 dark:bg-slate-900/30 text-xs text-slate-500 dark:text-slate-400">
          Click outside to cancel
        </div>
      </div>
    </>
  );
}
