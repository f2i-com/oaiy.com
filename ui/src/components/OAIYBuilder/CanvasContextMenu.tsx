/**
 * Canvas Context Menu Component
 *
 * Renders the right-click context menu for the workflow canvas.
 * Provides quick access to copy/paste, grouping, and layout operations.
 */

import { useCallback, useEffect, useRef } from 'react';
import type { Node } from '@xyflow/react';

export interface CanvasContextMenuProps {
  /** Current position of the context menu */
  position: { x: number; y: number };
  /** Current nodes in the workflow */
  nodes: Node[];
  /** Whether there's content in the clipboard */
  hasClipboard: boolean;
  /** Copy selected nodes */
  onCopy: () => void;
  /** Paste clipboard content at position */
  onPaste: (position?: { x: number; y: number }) => void;
  /** Group selected nodes */
  onGroupSelected: () => void;
  /** Ungroup selected group nodes */
  onUngroupSelected: () => void;
  /** Auto layout horizontally */
  onAutoLayoutHorizontal: () => void;
  /** Auto layout vertically */
  onAutoLayoutVertical: () => void;
  /** Collapse all nodes */
  onCollapseAll: () => void;
  /** Expand all nodes */
  onExpandAll: () => void;
  /** Close the context menu */
  onClose: () => void;
  /** Convert screen position to flow position */
  screenToFlowPosition?: (position: { x: number; y: number }) => { x: number; y: number };
}

export function CanvasContextMenu({
  position,
  nodes,
  hasClipboard,
  onCopy,
  onPaste,
  onGroupSelected,
  onUngroupSelected,
  onAutoLayoutHorizontal,
  onAutoLayoutVertical,
  onCollapseAll,
  onExpandAll,
  onClose,
  screenToFlowPosition,
}: CanvasContextMenuProps) {
  const hasSelectedNodes = nodes.some((n) => n.selected);
  const canGroup = nodes.filter((n) => n.selected).length >= 2;
  const canUngroup = nodes.some((n) => n.selected && n.type === 'group');

  const handlePaste = useCallback(() => {
    if (screenToFlowPosition) {
      const flowPosition = screenToFlowPosition(position);
      onPaste(flowPosition);
    } else {
      onPaste();
    }
    onClose();
  }, [position, onPaste, onClose, screenToFlowPosition]);

  const handleCopy = useCallback(() => {
    onCopy();
    onClose();
  }, [onCopy, onClose]);

  const handleGroup = useCallback(() => {
    onGroupSelected();
    onClose();
  }, [onGroupSelected, onClose]);

  const handleUngroup = useCallback(() => {
    onUngroupSelected();
    onClose();
  }, [onUngroupSelected, onClose]);

  const handleLayoutHorizontal = useCallback(() => {
    onAutoLayoutHorizontal();
    onClose();
  }, [onAutoLayoutHorizontal, onClose]);

  const handleLayoutVertical = useCallback(() => {
    onAutoLayoutVertical();
    onClose();
  }, [onAutoLayoutVertical, onClose]);

  const handleCollapse = useCallback(() => {
    onCollapseAll();
    onClose();
  }, [onCollapseAll, onClose]);

  const handleExpand = useCallback(() => {
    onExpandAll();
    onClose();
  }, [onExpandAll, onClose]);

  const menuRef = useRef<HTMLDivElement>(null);

  // Close the menu when Escape is pressed.
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Focus the first enabled menu item on mount.
  useEffect(() => {
    const firstEnabled = menuRef.current?.querySelector<HTMLButtonElement>(
      'button:not([disabled])'
    );
    firstEnabled?.focus();
  }, []);

  return (
    <>
      {/* Backdrop to close menu when clicking outside */}
      <div className="fixed inset-0 z-40" onClick={onClose} />

      {/* Context Menu */}
      <div
        ref={menuRef}
        role="menu"
        aria-orientation="vertical"
        className="fixed z-50 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl min-w-[200px] py-1"
        style={{ left: position.x, top: position.y }}
      >
        {/* Copy/Paste Section */}
        <button
          role="menuitem"
          onClick={handleCopy}
          disabled={!hasSelectedNodes}
          className={`w-full px-4 py-2 text-left text-sm flex items-center gap-3 ${
            hasSelectedNodes
              ? 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              : 'text-slate-400 dark:text-slate-500 cursor-not-allowed'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          Copy
          <span className="ml-auto text-xs text-slate-400 dark:text-slate-500">Ctrl+C</span>
        </button>
        <button
          role="menuitem"
          onClick={handlePaste}
          disabled={!hasClipboard}
          className={`w-full px-4 py-2 text-left text-sm flex items-center gap-3 ${
            hasClipboard
              ? 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              : 'text-slate-400 dark:text-slate-500 cursor-not-allowed'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          Paste
          <span className="ml-auto text-xs text-slate-400 dark:text-slate-500">Ctrl+V</span>
        </button>

        <div className="border-t border-slate-200 dark:border-slate-700 my-1" />

        {/* Group Section */}
        <button
          role="menuitem"
          onClick={handleGroup}
          disabled={!canGroup}
          className={`w-full px-4 py-2 text-left text-sm flex items-center gap-3 ${
            canGroup
              ? 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              : 'text-slate-400 dark:text-slate-500 cursor-not-allowed'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"
            />
          </svg>
          Group Selected
          <span className="ml-auto text-xs text-slate-400 dark:text-slate-500">Ctrl+G</span>
        </button>
        <button
          role="menuitem"
          onClick={handleUngroup}
          disabled={!canUngroup}
          className={`w-full px-4 py-2 text-left text-sm flex items-center gap-3 ${
            canUngroup
              ? 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              : 'text-slate-400 dark:text-slate-500 cursor-not-allowed'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
            />
          </svg>
          Ungroup
        </button>

        <div className="border-t border-slate-200 dark:border-slate-700 my-1" />

        {/* Layout Section */}
        <button
          role="menuitem"
          onClick={handleLayoutHorizontal}
          className="w-full px-4 py-2 text-left text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-3"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
            />
          </svg>
          Auto Layout (Horizontal)
        </button>
        <button
          role="menuitem"
          onClick={handleLayoutVertical}
          className="w-full px-4 py-2 text-left text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-3"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
          </svg>
          Auto Layout (Vertical)
        </button>

        <div className="border-t border-slate-200 dark:border-slate-700 my-1" />

        {/* Collapse/Expand Section */}
        <button
          role="menuitem"
          onClick={handleCollapse}
          className="w-full px-4 py-2 text-left text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-3"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
          Collapse All Nodes
        </button>
        <button
          role="menuitem"
          onClick={handleExpand}
          className="w-full px-4 py-2 text-left text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-3"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
            />
          </svg>
          Expand All Nodes
        </button>
      </div>
    </>
  );
}
