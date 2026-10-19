import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig, type ValidationIssue } from 'oaiy-ui-components';

interface MemoryNodeData {
  /** What the run does (compiler.ts): `get` reads the key, `set` stores the input under it. */
  operation?: string;
  /** The card's own word for it before it set `operation` (kept in step for flows that have it). */
  mode?: 'read' | 'write';
  key?: string;
  _status?: 'running' | 'completed' | 'error' | 'stopped';
  _collapsed?: boolean;
  onModeChange?: (value: 'read' | 'write') => void;
  onKeyChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  onChange?: (field: string, value: unknown) => void;
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

/** Read or Write, from what the run does (`operation`), else the card's older `mode`. */
function modeOf(data: MemoryNodeData): 'read' | 'write' {
  if (data.operation) return ['set', 'append', 'write'].includes(data.operation) ? 'write' : 'read';
  return data.mode === 'write' ? 'write' : 'read';
}

function MemoryNode({ data }: MemoryNodeProps) {
  const mode = modeOf(data);

  const onModeChangeRef = useRef(data.onModeChange);
  const onKeyChangeRef = useRef(data.onKeyChange);
  const onChangeRef = useRef(data.onChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onModeChangeRef.current = data.onModeChange;
    onKeyChangeRef.current = data.onKeyChange;
    onChangeRef.current = data.onChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  // The run reads `operation` (get or set): the card set only its own `mode`, so Write stored nothing.
  const handleModeChange = useCallback((newMode: 'read' | 'write') => {
    onChangeRef.current?.('operation', newMode === 'write' ? 'set' : 'get');
    onModeChangeRef.current?.(newMode);
  }, []);

  const handleKeyChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onKeyChangeRef.current?.(e.target.value);
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
    <span>
      {mode === 'read' ? 'reads' : 'stores'} {data.key ? <span className="font-mono">{data.key}</span> : 'nothing yet: give it a key'}
    </span>
  );

  // Its input in both modes: Write stores it, and Read gives it on when nothing is stored under the key yet.
  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'value', type: 'target', position: Position.Left, color: '!bg-slate-400', size: 'lg' },
  ], []);

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
            <span className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Operation</span>
            <div className="flex gap-2">
              <button
                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${mode === 'read' ? 'bg-cyan-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-600'
                  }`}
                onClick={() => handleModeChange('read')}
                aria-pressed={mode === 'read'}
              >
                Read
              </button>
              <button
                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${mode === 'write' ? 'bg-cyan-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-600'
                  }`}
                onClick={() => handleModeChange('write')}
                aria-pressed={mode === 'write'}
              >
                Write
              </button>
            </div>
          </div>

          {/* Key Input */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Key</label>
            <input
              type="text"
              aria-label="Key"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              placeholder="my_variable"
              value={data.key || ''}
              onChange={handleKeyChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <p className="text-slate-500 dark:text-slate-400 text-[10px] leading-snug m-0">
            {mode === 'read'
              ? 'Gives what is stored under the key; nothing stored yet, it gives what comes in on Value.'
              : 'Stores what comes in on Value under the key, kept between runs, and passes it on.'}
          </p>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(MemoryNode);
