import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig, type ValidationIssue } from 'oaiy-ui-components';


type LoopMode = 'count' | 'foreach' | 'while_true';

interface LoopStartNodeData {
  iterations?: number;
  loopMode?: LoopMode;
  loopName?: string;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onIterationsChange?: (value: number) => void;
  onLoopModeChange?: (value: string) => void;
  onLoopNameChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface LoopStartNodeProps {
  data: LoopStartNodeData;
}

const LOOP_MODES = [
  { value: 'count', label: 'Count', description: 'Run N times' },
  { value: 'foreach', label: 'For Each', description: 'Iterate array' },
  { value: 'while_true', label: 'While True', description: 'Run until stop condition' },
] as const;

// Icon for the node header
const LoopIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

function LoopStartNode({ data }: LoopStartNodeProps) {
  const onIterationsChangeRef = useRef(data.onIterationsChange);
  const onLoopModeChangeRef = useRef(data.onLoopModeChange);
  const onLoopNameChangeRef = useRef(data.onLoopNameChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onIterationsChangeRef.current = data.onIterationsChange;
    onLoopModeChangeRef.current = data.onLoopModeChange;
    onLoopNameChangeRef.current = data.onLoopNameChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleLoopNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onLoopNameChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const loopMode = data.loopMode || 'count';
  const rawIterations = data.iterations ?? 3;
  const ceiling = loopMode === 'while_true' ? 1000 : 50;
  const iterations = Math.max(1, Math.min(ceiling, rawIterations));
  const isWarning = loopMode === 'count' && rawIterations > 20;
  const showBodyProperties = data.showBodyProperties !== false;

  const validationIssues = useMemo(() => {
    const issues: ValidationIssue[] = [];
    if (!data.loopName) {
      issues.push({ field: 'Name', message: 'Helps identify loops' });
    }
    return issues;
  }, [data.loopName]);

  const collapsedPreview = (
    <div className="text-slate-400">
      <div className={`font-mono text-[10px] ${loopMode === 'while_true' ? 'text-purple-400' : 'text-amber-400'}`}>
        {loopMode === 'count' && `FOR 1..${iterations}`}
        {loopMode === 'foreach' && 'FOR EACH'}
        {loopMode === 'while_true' && 'WHILE TRUE'}
      </div>
    </div>
  );

  // Input handles using HandleConfig
  const inputHandles = useMemo<HandleConfig[]>(() => {
    if (loopMode === 'count') {
      return [
        { id: 'count', type: 'target', position: Position.Left, color: '!bg-amber-500', label: 'count', labelColor: 'text-amber-400', size: 'md' },
      ];
    }
    return [
      { id: 'array', type: 'target', position: Position.Left, color: '!bg-blue-500', label: 'array', labelColor: 'text-blue-400', size: 'md' },
    ];
  }, [loopMode]);

  // Output handles using HandleConfig
  const outputHandles = useMemo<HandleConfig[]>(() => {
    const handles: HandleConfig[] = [
      { id: 'item', type: 'source', position: Position.Right, color: '!bg-green-500', label: loopMode === 'foreach' ? 'item' : 'i', labelColor: 'text-green-400', size: 'lg' },
    ];
    if (loopMode === 'foreach') {
      handles.push({ id: 'index', type: 'source', position: Position.Right, color: '!bg-amber-500', label: 'index', labelColor: 'text-amber-400', size: 'sm' });
    }
    return handles;
  }, [loopMode]);

  // Bottom handles for loop connection
  const bottomHandles = useMemo<HandleConfig[]>(() => [
    { id: 'loop', type: 'source', position: Position.Bottom, color: '!bg-amber-500', label: 'loop end', labelColor: 'text-amber-400', size: 'lg' },
  ], []);

  const title = data.loopName ? `Loop: ${data.loopName}` : 'Loop Start';

  return (
    <CollapsibleNodeWrapper
      title={title}
      color="amber"
      icon={LoopIcon}
      width={200}
      collapsedWidth={130}
      status={data._status}
      validationIssues={validationIssues}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
      bottomHandles={bottomHandles}
    >
      {showBodyProperties && (
        <>
          {/* Loop Name */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Name</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
              placeholder="main, outer..."
              value={data.loopName || ''}
              onChange={handleLoopNameChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          {/* Mode Dropdown */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Mode</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500"
              value={loopMode}
              onChange={(e: ChangeEvent<HTMLSelectElement>) => onLoopModeChangeRef.current?.(e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {LOOP_MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>{mode.label}</option>
              ))}
            </select>
          </div>

          {loopMode === 'count' && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Iterations</label>
              <input
                type="number"
                min={1}
                max={50}
                className={`nodrag nowheel w-full bg-white dark:bg-slate-900 border rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none font-mono ${isWarning ? 'border-orange-500 focus:border-orange-400' : 'border-slate-300 dark:border-slate-600 focus:border-amber-500'
                  }`}
                value={iterations}
                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                  const val = parseInt(e.target.value) || 1;
                  onIterationsChangeRef.current?.(Math.max(1, Math.min(50, val)));
                }}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}

          {loopMode === 'while_true' && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Max Iterations</label>
              <input
                type="number"
                min={1}
                max={1000}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500 font-mono"
                value={iterations}
                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                  const val = parseInt(e.target.value) || 100;
                  onIterationsChangeRef.current?.(Math.max(1, Math.min(1000, val)));
                }}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(LoopStartNode);
