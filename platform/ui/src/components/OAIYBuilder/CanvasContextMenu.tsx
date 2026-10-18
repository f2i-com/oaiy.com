/**
 * Canvas Context Menu Component
 *
 * Renders the right-click context menu for the workflow canvas.
 * Provides quick access to copy/paste, grouping, and layout operations.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ClipboardPaste, Copy, FoldVertical, Group, LayoutGrid, Redo2, Rows3, Undo2, UnfoldVertical, Ungroup } from 'lucide-react';
import { shortcut } from '../../lib/keys';
import type { Node } from '@xyflow/react';

/** One item: the editor's menu row, with its shortcut at the end. */
function Item({ icon, label, shortcut, onClick, disabled = false }: {
  icon: ReactNode;
  label: string;
  shortcut?: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      className="items-center disabled:opacity-40"
      // The menu's rows set their cursor; a disabled one says so.
      style={disabled ? { cursor: 'not-allowed' } : undefined}
    >
      {icon}
      <span>{label}</span>
      {shortcut && <small className="ml-auto pl-3 font-mono">{shortcut}</small>}
    </button>
  );
}

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
  /** Undo and redo the last change on the canvas, where there is one */
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
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
  onUndo,
  onRedo,
  canUndo = false,
  canRedo = false,
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
  // At the pointer, moved in where it would run off the window (a right click near its right or bottom edge)
  const [at, setAt] = useState(position);
  useLayoutEffect(() => {
    const menu = menuRef.current;
    if (!menu) return;
    const { width, height } = menu.getBoundingClientRect();
    const margin = 8;
    setAt({
      x: Math.max(margin, Math.min(position.x, window.innerWidth - width - margin)),
      y: Math.max(margin, Math.min(position.y, window.innerHeight - height - margin)),
    });
  }, [position]);

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

      {/* Context Menu: the editor's menu, at the pointer. */}
      <div
        ref={menuRef}
        role="menu"
        aria-orientation="vertical"
        aria-label="Canvas"
        className="oaiy-menu"
        style={{ left: at.x, top: at.y }}
      >
        {/* Undo/Redo Section */}
        {onUndo && <Item icon={<Undo2 size={14} />} label="Undo" shortcut={shortcut('Z')} onClick={() => { onUndo(); onClose(); }} disabled={!canUndo} />}
        {onRedo && <Item icon={<Redo2 size={14} />} label="Redo" shortcut={shortcut('Z', { shift: true })} onClick={() => { onRedo(); onClose(); }} disabled={!canRedo} />}
        {(onUndo || onRedo) && <hr />}

        {/* Copy/Paste Section */}
        <Item icon={<Copy size={14} />} label="Copy" shortcut={shortcut('C')} onClick={handleCopy} disabled={!hasSelectedNodes} />
        <Item icon={<ClipboardPaste size={14} />} label="Paste" shortcut={shortcut('V')} onClick={handlePaste} disabled={!hasClipboard} />

        <hr />

        {/* Group Section */}
        <Item icon={<Group size={14} />} label="Group the selection" shortcut={shortcut('G')} onClick={handleGroup} disabled={!canGroup} />
        <Item icon={<Ungroup size={14} />} label="Ungroup" onClick={handleUngroup} disabled={!canUngroup} />

        <hr />

        {/* Layout Section */}
        <Item icon={<LayoutGrid size={14} />} label="Lay out left to right" onClick={handleLayoutHorizontal} />
        <Item icon={<Rows3 size={14} />} label="Lay out top to bottom" onClick={handleLayoutVertical} />

        <hr />

        {/* Collapse/Expand Section */}
        {/* (in the toolbar's words: Compact, Expand) */}
        <Item icon={<FoldVertical size={14} />} label="Make every node compact" onClick={handleCollapse} />
        <Item icon={<UnfoldVertical size={14} />} label="Expand every node" onClick={handleExpand} />
      </div>
    </>
  );
}
