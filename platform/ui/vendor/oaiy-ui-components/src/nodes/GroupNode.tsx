import React, { memo, useState, useCallback } from 'react';
import { NodeResizer, type NodeProps } from '@xyflow/react';

// Group node colors - supports both light and dark mode
const GROUP_COLORS = {
  slate: { bg: 'rgba(71, 85, 105, 0.15)', border: 'rgb(71, 85, 105)', label: 'text-slate-700 dark:text-slate-300' },
  blue: { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgb(59, 130, 246)', label: 'text-blue-700 dark:text-blue-300' },
  purple: { bg: 'rgba(168, 85, 247, 0.15)', border: 'rgb(168, 85, 247)', label: 'text-purple-700 dark:text-purple-300' },
  green: { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgb(34, 197, 94)', label: 'text-green-700 dark:text-green-300' },
  orange: { bg: 'rgba(249, 115, 22, 0.15)', border: 'rgb(249, 115, 22)', label: 'text-orange-700 dark:text-orange-300' },
  pink: { bg: 'rgba(236, 72, 153, 0.15)', border: 'rgb(236, 72, 153)', label: 'text-pink-700 dark:text-pink-300' },
  cyan: { bg: 'rgba(6, 182, 212, 0.15)', border: 'rgb(6, 182, 212)', label: 'text-cyan-700 dark:text-cyan-300' },
};

type GroupColor = keyof typeof GROUP_COLORS;

export interface GroupNodeData {
  label?: string;
  color?: GroupColor;
  collapsed?: boolean;
  onChange?: (field: string, value: unknown) => void;
  [key: string]: unknown;  // Allow additional properties for React Flow compatibility
}

interface GroupNodeProps extends NodeProps {
  data: GroupNodeData;
}

export const GroupNode: React.FC<GroupNodeProps> = memo(({ data, selected }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(data.label || 'Group');

  const colorScheme = GROUP_COLORS[data.color || 'slate'];

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditing(true);
    setEditValue(data.label || 'Group');
  }, [data.label]);

  const handleBlur = useCallback(() => {
    setIsEditing(false);
    if (data.onChange && editValue !== data.label) {
      data.onChange('label', editValue);
    }
  }, [data, editValue]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      setIsEditing(false);
      if (data.onChange && editValue !== data.label) {
        data.onChange('label', editValue);
      }
    }
    if (e.key === 'Escape') {
      setIsEditing(false);
      setEditValue(data.label || 'Group');
    }
  }, [data, editValue]);

  return (
    <>
      {/* Resizer - only visible when selected */}
      <NodeResizer
        isVisible={selected}
        minWidth={150}
        minHeight={100}
        handleStyle={{
          width: 8,
          height: 8,
          borderRadius: 2,
          backgroundColor: colorScheme.border,
        }}
        lineStyle={{
          borderWidth: 1,
          borderColor: colorScheme.border,
        }}
      />

      {/* Group container */}
      <div
        className="w-full h-full rounded-[var(--r-card)] relative"
        style={{
          backgroundColor: colorScheme.bg,
          border: `2px dashed ${selected ? colorScheme.border : 'rgb(var(--color-border-strong))'}`,
          minWidth: 150,
          minHeight: 100,
        }}
      >
        {/* Label */}
        <div
          className="absolute -top-3.5 left-3 px-2 py-0.5 rounded-[var(--r-sm)] text-xs font-medium bg-[rgb(var(--color-bg-secondary))] border border-[rgb(var(--color-border-primary))] shadow-sm"
          style={{
            borderColor: selected ? colorScheme.border : undefined,
          }}
          onDoubleClick={handleDoubleClick}
        >
          {isEditing ? (
            <input
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onBlur={handleBlur}
              onKeyDown={handleKeyDown}
              autoFocus
              aria-label="Group name"
              className="bg-transparent outline-none text-xs w-24 text-[rgb(var(--color-text-primary))]"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className={colorScheme.label}>
              {data.label || 'Group'}
            </span>
          )}
        </div>

        {/* Accessible 18x18 color swatches - shown when selected */}
        {selected && (
          <div
            role="group"
            aria-label="Group color"
            className="nodrag absolute -top-3.5 right-3 flex items-center gap-1.5 rounded-full border border-[rgb(var(--color-border-primary))] bg-[rgb(var(--color-bg-secondary))] px-2 py-1 shadow-sm"
          >
            {(Object.keys(GROUP_COLORS) as GroupColor[]).map((color) => {
              const isCurrent = (data.color || 'slate') === color;
              return (
                <button
                  key={color}
                  type="button"
                  aria-label={`Set group color to ${color}`}
                  aria-pressed={isCurrent}
                  className={`w-[18px] h-[18px] rounded-full border-2 transition-transform focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[rgb(var(--accent-primary))] ${
                    isCurrent ? 'scale-110 ring-2 ring-[rgb(var(--accent-primary)/0.4)]' : 'hover:scale-110'
                  }`}
                  style={{
                    backgroundColor: GROUP_COLORS[color].border,
                    borderColor: isCurrent ? 'rgb(var(--color-bg-secondary))' : 'transparent',
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    data.onChange?.('color', color);
                  }}
                  title={`Color: ${color}`}
                />
              );
            })}
          </div>
        )}
      </div>
    </>
  );
});

GroupNode.displayName = 'GroupNode';

export default GroupNode;
