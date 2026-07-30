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
        className="w-full h-full rounded-xl relative"
        style={{
          backgroundColor: colorScheme.bg,
          border: `2px dashed ${selected ? colorScheme.border : 'rgba(100, 116, 139, 0.5)'}`,
          minWidth: 150,
          minHeight: 100,
        }}
      >
        {/* Label */}
        <div
          className="absolute -top-3 left-3 px-2 py-0.5 rounded text-xs font-medium bg-slate-100 dark:bg-slate-800"
          style={{
            borderColor: colorScheme.border,
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
              className="bg-transparent outline-none text-xs w-20 text-slate-700 dark:text-slate-200"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className={colorScheme.label}>
              {data.label || 'Group'}
            </span>
          )}
        </div>

        {/* Color indicator dots - shown when selected */}
        {selected && (
          <div className="absolute -top-3 right-3 flex gap-1">
            {(Object.keys(GROUP_COLORS) as GroupColor[]).map((color) => (
              <button
                key={color}
                className={`w-3 h-3 rounded-full border-2 transition-transform ${
                  data.color === color ? 'scale-125' : 'hover:scale-110'
                }`}
                style={{
                  backgroundColor: GROUP_COLORS[color].border,
                  borderColor: data.color === color ? 'white' : 'transparent',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  data.onChange?.('color', color);
                }}
                title={`Set color: ${color}`}
              />
            ))}
          </div>
        )}
      </div>
    </>
  );
});

GroupNode.displayName = 'GroupNode';

export default GroupNode;
