import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


type StopCondition = 'none' | 'contains' | 'equals' | 'starts_with' | 'json_field';

interface LoopEndNodeData {
  collectedResults?: unknown[];
  stopCondition?: StopCondition;
  stopValue?: string;
  stopField?: string;
  loopName?: string; // Will be set from connected Loop Start
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onStopConditionChange?: (value: string) => void;
  onStopValueChange?: (value: string) => void;
  onStopFieldChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface LoopEndNodeProps {
  data: LoopEndNodeData;
}

const STOP_CONDITIONS = [
  { value: 'none', label: 'Run All Iterations', description: 'No early stop' },
  { value: 'contains', label: 'Result Contains', description: 'Stop when result contains text' },
  { value: 'equals', label: 'Result Equals', description: 'Stop when result equals value' },
  { value: 'starts_with', label: 'Result Starts With', description: 'Stop when result starts with text' },
  { value: 'json_field', label: 'JSON Field Equals', description: 'Stop when JSON field equals value' },
] as const;

const LoopEndIcon = (
  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
  </svg>
);

function LoopEndNode({ data }: LoopEndNodeProps) {
  const resultCount = data.collectedResults?.length ?? 0;
  const stopCondition = data.stopCondition || 'none';

  const onStopConditionChangeRef = useRef(data.onStopConditionChange);
  const onStopValueChangeRef = useRef(data.onStopValueChange);
  const onStopFieldChangeRef = useRef(data.onStopFieldChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onStopConditionChangeRef.current = data.onStopConditionChange;
    onStopValueChangeRef.current = data.onStopValueChange;
    onStopFieldChangeRef.current = data.onStopFieldChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleStopConditionChange = useCallback((e: ChangeEvent<HTMLSelectElement>) => {
    onStopConditionChangeRef.current?.(e.target.value);
  }, []);

  const handleStopValueChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onStopValueChangeRef.current?.(e.target.value);
  }, []);

  const handleStopFieldChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onStopFieldChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const showStopValue = stopCondition !== 'none';
  const showStopField = stopCondition === 'json_field';

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className={stopCondition !== 'none' ? 'text-purple-400' : 'text-amber-400'}>
        {stopCondition !== 'none' ? 'WHEN DONE' : 'END FOR'}
      </span>
      {resultCount > 0 && <span className="ml-1 text-[10px]">({resultCount})</span>}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'loop', type: 'target', position: Position.Top, color: '!bg-amber-500', label: 'loop', labelColor: 'text-amber-400', size: 'md' },
    { id: 'input', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'results', type: 'source', position: Position.Right, color: '!bg-green-500', label: '[ ]', labelColor: 'text-green-400', size: 'lg' },
  ], []);

  const titleExtra = (
    <>
      {data.loopName && (
        <span className="px-1.5 py-0.5 bg-amber-800 text-amber-300 text-[10px] rounded font-mono">
          {data.loopName}
        </span>
      )}
      {resultCount > 0 && (
        <span className="ml-auto px-1.5 py-0.5 bg-amber-900 text-amber-400 text-[10px] rounded">
          {resultCount}
        </span>
      )}
    </>
  );

  const showBodyProperties = data.showBodyProperties !== false;

  return (
    <CollapsibleNodeWrapper
      title="Loop End"
      color="amber"
      icon={LoopEndIcon}
      width={220}
      collapsedWidth={130}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
      titleExtra={titleExtra}
    >
      {showBodyProperties && (
        <>
          {/* Stop Condition */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Stop When</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500"
              value={stopCondition}
              onChange={handleStopConditionChange}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {STOP_CONDITIONS.map((cond) => (
                <option key={cond.value} value={cond.value}>
                  {cond.label}
                </option>
              ))}
            </select>
          </div>

          {/* JSON Field (for json_field condition) */}
          {showStopField && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Field Name</label>
              <input
                type="text"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
                placeholder="status, done, complete"
                value={data.stopField || ''}
                onChange={handleStopFieldChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}

          {/* Stop Value */}
          {showStopValue && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
                {stopCondition === 'json_field' ? 'Field Value' : 'Stop Value'}
              </label>
              <input
                type="text"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
                placeholder={stopCondition === 'json_field' ? 'true, done, complete' : 'DONE, finished, complete'}
                value={data.stopValue || ''}
                onChange={handleStopValueChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}

          <p className="text-slate-500 text-[10px]">
            {stopCondition === 'none'
              ? 'Collects results from each iteration into an array.'
              : 'Stops loop early when condition is met.'}
          </p>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(LoopEndNode);
