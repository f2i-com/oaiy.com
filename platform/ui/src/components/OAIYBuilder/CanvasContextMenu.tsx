/**
 * Canvas Context Menu Component
 *
 * Renders the right-click context menu for the workflow canvas.
 * Provides quick access to copy/paste, grouping, and layout operations.
 */

import { useCallback, type ReactNode } from 'react';
import { ClipboardPaste, Copy, FoldVertical, Group, LayoutGrid, Redo2, Rows3, Undo2, UnfoldVertical, Ungroup } from 'lucide-react';
import { shortcut } from '../../lib/keys';
import Menu from '../ui/Menu';
import type { Node } from '@xyflow/react';

/** One item: the editor's menu row, with its shortcut at the end. */
function Item({ icon, label, shortcut: keyHint, onClick, disabled = false }: {
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
      className="items-center"
    >
      {icon}
      <span>{label}</span>
      {keyHint && <small className="ml-auto pl-3 font-mono">{keyHint}</small>}
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
  }, [position, onPaste, screenToFlowPosition]);

  return (
    <Menu open at={position} onClose={onClose} label="Canvas">
      {/* Undo/Redo Section */}
      {onUndo && <Item icon={<Undo2 size={14} />} label="Undo" shortcut={shortcut('Z')} onClick={onUndo} disabled={!canUndo} />}
      {onRedo && <Item icon={<Redo2 size={14} />} label="Redo" shortcut={shortcut('Z', { shift: true })} onClick={onRedo} disabled={!canRedo} />}
      {(onUndo || onRedo) && <hr />}

      {/* Copy/Paste Section */}
      <Item icon={<Copy size={14} />} label="Copy" shortcut={shortcut('C')} onClick={onCopy} disabled={!hasSelectedNodes} />
      <Item icon={<ClipboardPaste size={14} />} label="Paste" shortcut={shortcut('V')} onClick={handlePaste} disabled={!hasClipboard} />

      <hr />

      {/* Group Section */}
      <Item icon={<Group size={14} />} label="Group the selection" shortcut={shortcut('G')} onClick={onGroupSelected} disabled={!canGroup} />
      <Item icon={<Ungroup size={14} />} label="Ungroup" onClick={onUngroupSelected} disabled={!canUngroup} />

      <hr />

      {/* Layout Section */}
      <Item icon={<LayoutGrid size={14} />} label="Lay out left to right" onClick={onAutoLayoutHorizontal} />
      <Item icon={<Rows3 size={14} />} label="Lay out top to bottom" onClick={onAutoLayoutVertical} />

      <hr />

      {/* Collapse/Expand Section */}
      <Item icon={<FoldVertical size={14} />} label="Make every node compact" onClick={onCollapseAll} />
      <Item icon={<UnfoldVertical size={14} />} label="Expand every node" onClick={onExpandAll} />
    </Menu>
  );
}
