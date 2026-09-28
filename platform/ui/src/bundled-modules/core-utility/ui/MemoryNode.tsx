import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig, type ValidationIssue } from 'oaiy-ui-components';

interface MemoryNodeData {
  mode?: 'read' | 'write';
  key?: string;
  defaultValue?: string;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onModeChange?: (value: 'read' | 'write') => void;
  onKeyChange?: (value: string) => void;
  onDefaultValueChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface MemoryNodeProps {
  data: MemoryNodeData;
}

const MemoryIcon = (
  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
    <path d="M3 12v3c0 1.657 3.134 3 7 3s7-1.343 7-3v-3c0 1.657-3.134 3-7 3s-7-1.343-7-3z" />
    <path d="M3 7v3c0 1.657 3.134 3 7 3s7-1.343 7-3V7c0 1.657-3.134 3-7 3S3 8.657 3 7z" />
    <path d="M17 5c0 1.657-3.134 3-7 3S3 6.657 3 5s3.134-3 7-3 7 1.343 7 3z" />
  </svg>
);

function MemoryNode({ data }: MemoryNodeProps) {
  const mode = data.mode || 'read';

  const onModeChangeRef = useRef(data.onModeChange);
  const onKeyChangeRef = useRef(data.onKeyChange);
  const onDefaultValueChangeRef = useRef(data.onDefaultValueChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onModeChangeRef.current = data.onModeChange;
    onKeyChangeRef.current = data.onKeyChange;
    onDefaultValueChangeRef.current = data.onDefaultValueChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleModeChange = useCallback((newMode: 'read' | 'write') => {
    onModeChangeRef.current?.(newMode);
  }, []);

  const handleKeyChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onKeyChangeRef.current?.(e.target.value);
  }, []);

  const handleDefaultValueChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onDefaultValueChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const validationIssues = useMemo(() => {
    const issues: ValidationIssue[] = [];
    if (!data.key) {
      issues.push({ field: 'Key', message: 'Required' });
    }
    return issues;
  }, [data.key]);

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className={mode === 'read' ? 'text-cyan-400' : 'text-orange-400'}>{mode.toUpperCase()}</span>
      {data.key && <span className="ml-1 font-mono text-[10px]">{data.key}</span>}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => {
    if (mode === 'write') {
      return [
        { id: 'value', type: 'target', position: Position.Left, color: '!bg-blue-500', label: 'value', labelColor: 'text-blue-400', size: 'lg' },
      ];
    }
    return [];
  }, [mode]);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'value', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Memory"
      color="cyan"
      icon={MemoryIcon}
      width={240}
      collapsedWidth={130}
      status={data._status}
      validationIssues={validationIssues}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          {/* Mode Toggle */}
          <div>
            <span className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Mode</span>
            <div className="flex gap-2">
              <button
                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${mode === 'read' ? 'bg-cyan-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-600'
                  }`}
                onClick={() => handleModeChange('read')}
              >
                Read
              </button>
              <button
                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${mode === 'write' ? 'bg-cyan-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-600'
                  }`}
                onClick={() => handleModeChange('write')}
              >
                Write
              </button>
            </div>
          </div>

          {/* Key Input */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Key Name</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              placeholder="my_variable"
              value={data.key || ''}
              onChange={handleKeyChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          {/* Default Value for Read mode */}
          {mode === 'read' && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Default (if not set)</label>
              <input
                type="text"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
                placeholder="default value"
                value={data.defaultValue || ''}
                onChange={handleDefaultValueChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(MemoryNode);
