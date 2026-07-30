import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { useNodeResize, CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface InputTextNodeData {
  value?: string;
  _collapsed?: boolean;
  onChange?: (field: string, value: unknown) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface InputTextNodeProps {
  data: InputTextNodeData;
}

const InputTextIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
  </svg>
);

function InputTextNode({ data }: InputTextNodeProps) {
  const { size, handleResizeStart } = useNodeResize({
    initialWidth: 260,
    initialHeight: 140,
    constraints: { minWidth: 200, maxWidth: 500, minHeight: 120, maxHeight: 400 },
  });

  const onChangeRef = useRef(data.onChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onChangeRef.current = data.onChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    onChangeRef.current?.('value', e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const showBodyProperties = data.showBodyProperties !== false;
  const textareaHeight = Math.max(60, size.height - 80);

  // Safely convert value to string for display
  const displayValue = data.value
    ? (typeof data.value === 'object' ? JSON.stringify(data.value) : String(data.value))
    : '';

  const collapsedPreview = (
    <div className="text-slate-400 truncate text-[10px]">
      {displayValue ? displayValue.substring(0, 50) + (displayValue.length > 50 ? '...' : '') : <span className="italic text-slate-500">Empty</span>}
    </div>
  );

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'text', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  const resizeHandles = (
    <>
      <div
        className="nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize opacity-0 group-hover:opacity-100 hover:bg-green-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 'e')}
      />
      <div
        className="nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize opacity-0 group-hover:opacity-100 hover:bg-green-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 's')}
      />
      <div
        className="nodrag absolute bottom-1 right-1 w-3 h-3 cursor-nwse-resize opacity-0 group-hover:opacity-100 transition-opacity"
        onMouseDown={(e) => handleResizeStart(e, 'se')}
      >
        <svg className="w-3 h-3 text-slate-500" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22Z" />
        </svg>
      </div>
    </>
  );

  return (
    <CollapsibleNodeWrapper
      title="Input: Text"
      color="green"
      icon={InputTextIcon}
      width={size.width}
      collapsedWidth={140}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      outputHandles={outputHandles}
      resizeHandles={resizeHandles}
    >
      {showBodyProperties && (
        <div>
          <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Text Content</label>
          <textarea
            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 resize-none focus:outline-none focus:border-green-500"
            style={{ height: textareaHeight, resize: 'none' }}
            placeholder="Enter text..."
            value={displayValue}
            onChange={handleChange}
            onMouseDown={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(InputTextNode);
