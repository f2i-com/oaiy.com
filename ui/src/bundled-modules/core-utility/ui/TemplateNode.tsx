import { memo, useMemo, useRef, useEffect, useCallback } from 'react';
import { Position } from '@xyflow/react';
import { useNodeResize, CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface TemplateNodeData {
  template?: string;
  inputCount?: number;
  inputNames?: string[];
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onTemplateChange?: (value: string) => void;
  onInputCountChange?: (value: number) => void;
  onInputNamesChange?: (value: string[]) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface TemplateNodeProps {
  data: TemplateNodeData;
}

const TemplateIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
  </svg>
);

function TemplateNode({ data }: TemplateNodeProps) {
  const { size, handleResizeStart } = useNodeResize({
    initialWidth: 320,
    initialHeight: 300,
    constraints: { minWidth: 260, maxWidth: 600, minHeight: 250, maxHeight: 600 },
  });
  // Default to 2 inputs for backward compatibility with existing workflows
  // Flowplan compiler sets inputCount/inputNames explicitly for generated nodes
  const inputCount = data.inputCount ?? 2;
  const inputNames = data.inputNames || ['var1', 'var2'];

  const onTemplateChangeRef = useRef(data.onTemplateChange);
  const onInputCountChangeRef = useRef(data.onInputCountChange);
  const onInputNamesChangeRef = useRef(data.onInputNamesChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onTemplateChangeRef.current = data.onTemplateChange;
    onInputCountChangeRef.current = data.onInputCountChange;
    onInputNamesChangeRef.current = data.onInputNamesChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleTemplateChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onTemplateChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const templateVariables = useMemo(() => {
    if (!data.template) return [];
    const matches = data.template.match(/\{\{(\w+)\}\}/g);
    return matches ? [...new Set(matches.map(m => m.slice(2, -2)))] : [];
  }, [data.template]);

  const handleInputNameChange = useCallback((index: number, name: string) => {
    const newNames = [...inputNames];
    newNames[index] = name;
    onInputNamesChangeRef.current?.(newNames);
  }, [inputNames]);

  const handleAddInput = useCallback(() => {
    if (inputCount < 6) {
      const newCount = inputCount + 1;
      const newNames = [...inputNames, `var${newCount}`];
      onInputCountChangeRef.current?.(newCount);
      onInputNamesChangeRef.current?.(newNames);
    }
  }, [inputCount, inputNames]);

  const handleRemoveInput = useCallback(() => {
    if (inputCount > 1) {
      const newCount = inputCount - 1;
      const newNames = inputNames.slice(0, newCount);
      onInputCountChangeRef.current?.(newCount);
      onInputNamesChangeRef.current?.(newNames);
    }
  }, [inputCount, inputNames]);

  const textareaHeight = Math.max(80, size.height - 200);

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className="text-amber-400">{inputCount} vars</span>
      {templateVariables.length > 0 && (
        <span className="ml-1 text-[10px] font-mono">{`{{${templateVariables[0]}}}`}</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => {
    // Use standard handle IDs that match the node definition: input, input2, input3, input4, input5
    const standardIds = ['input', 'input2', 'input3', 'input4', 'input5'];
    return standardIds.slice(0, inputCount).map((id, index) => ({
      id,
      type: 'target' as const,
      position: Position.Left,
      color: '!bg-blue-500',
      label: inputNames[index] || id,
      labelColor: 'text-blue-400',
      size: 'md' as const,
    }));
  }, [inputNames, inputCount]);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'result', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  const resizeHandles = (
    <>
      <div
        className="nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize opacity-0 group-hover:opacity-100 hover:bg-amber-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 'e')}
      />
      <div
        className="nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize opacity-0 group-hover:opacity-100 hover:bg-amber-500/30 transition-all"
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
      title="Template"
      color="amber"
      icon={TemplateIcon}
      width={size.width}
      collapsedWidth={140}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
      resizeHandles={resizeHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          {/* Input Variables */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-slate-600 dark:text-slate-400 text-xs">Input Variables</label>
              <div className="flex gap-1">
                <button
                  onClick={handleRemoveInput}
                  disabled={inputCount <= 1}
                  className="w-5 h-5 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 text-xs"
                >
                  −
                </button>
                <button
                  onClick={handleAddInput}
                  disabled={inputCount >= 6}
                  className="w-5 h-5 flex items-center justify-center bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-slate-700 dark:text-slate-300 text-xs"
                >
                  +
                </button>
              </div>
            </div>
            <div className="flex flex-col gap-1">
              {Array.from({ length: inputCount }).map((_, index) => (
                <input
                  key={index}
                  type="text"
                  className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-1.5 py-0.5 text-xs text-amber-700 dark:text-amber-400 focus:outline-none focus:border-amber-500 font-mono"
                  value={inputNames[index] || ''}
                  onChange={(e) => handleInputNameChange(index, e.target.value)}
                  onMouseDown={(e) => e.stopPropagation()}
                  placeholder={`var${index + 1}`}
                />
              ))}
            </div>
          </div>

          {/* Template Editor */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
              Template <span className="text-slate-600">(use {`{{varName}}`})</span>
            </label>
            <textarea
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 resize-none focus:outline-none focus:border-amber-500 font-mono"
              style={{ height: textareaHeight }}
              placeholder={`{"prompt": "{{prompt}}"}`}
              value={data.template || ''}
              onChange={handleTemplateChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          {/* Variables Used */}
          {templateVariables.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {templateVariables.map((varName: string) => (
                <span
                  key={varName}
                  className={`px-1.5 py-0.5 text-[10px] rounded font-mono ${inputNames.includes(varName)
                    ? 'bg-green-900/50 text-green-400 border border-green-700'
                    : 'bg-red-900/50 text-red-400 border border-red-700'
                    }`}
                >
                  {`{{${varName}}}`}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(TemplateNode);
