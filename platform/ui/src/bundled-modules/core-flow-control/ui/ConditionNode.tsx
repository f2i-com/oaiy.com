import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


type ConditionOperator = 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'greater' | 'less' | 'greater_eq' | 'less_eq' | 'is_empty' | 'not_empty';

interface ConditionNodeData {
  operator?: ConditionOperator;
  compareValue?: string;
  _status?: 'running' | 'completed' | 'error' | 'stopped';
  _collapsed?: boolean;
  onOperatorChange?: (value: ConditionOperator) => void;
  onCompareValueChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface ConditionNodeProps {
  data: ConditionNodeData;
}

const operators: { value: ConditionOperator; label: string }[] = [
  { value: 'equals', label: '== (equals)' },
  { value: 'not_equals', label: '!= (not equals)' },
  { value: 'contains', label: 'contains' },
  { value: 'not_contains', label: 'not contains' },
  { value: 'greater', label: '> (greater)' },
  { value: 'less', label: '< (less)' },
  { value: 'greater_eq', label: '>= (greater or eq)' },
  { value: 'less_eq', label: '<= (less or eq)' },
  { value: 'is_empty', label: 'is empty' },
  { value: 'not_empty', label: 'is not empty' },
];

const ConditionIcon = (
  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

function ConditionNode({ data }: ConditionNodeProps) {
  const onOperatorChangeRef = useRef(data.onOperatorChange);
  const onCompareValueChangeRef = useRef(data.onCompareValueChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onOperatorChangeRef.current = data.onOperatorChange;
    onCompareValueChangeRef.current = data.onCompareValueChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const operator = data.operator ?? 'equals';
  const needsCompareValue = !['is_empty', 'not_empty'].includes(operator);
  const showBodyProperties = data.showBodyProperties !== false;

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const operatorLabel = operators.find(op => op.value === operator)?.label || operator;

  const collapsedPreview = (
    <span>
      if the value {(operatorLabel.match(/\(([^)]+)\)/)?.[1] ?? operatorLabel).toLowerCase()}
      {data.compareValue ? <span className="font-mono"> {String(data.compareValue)}</span> : null}
    </span>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'input', type: 'target', position: Position.Left, color: '!bg-blue-500', label: 'value', labelColor: 'text-blue-700 dark:text-blue-400', size: 'lg' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'true', type: 'source', position: Position.Right, color: '!bg-green-500', label: 'true', labelColor: 'text-green-700 dark:text-green-400', size: 'md' },
    { id: 'false', type: 'source', position: Position.Right, color: '!bg-red-500', label: 'false', labelColor: 'text-red-700 dark:text-red-400', size: 'md' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Condition"
      color="cyan"
      icon={ConditionIcon}
      width={260}
      collapsedWidth={120}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {showBodyProperties && (
        <>
          {/* Operator */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Operator</label>
            <select
              aria-label="Operator"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
              value={operator}
              onChange={(e: ChangeEvent<HTMLSelectElement>) => onOperatorChangeRef.current?.(e.target.value as ConditionOperator)}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {operators.map((op) => (
                <option key={op.value} value={op.value}>
                  {op.label}
                </option>
              ))}
            </select>
          </div>

          {/* Compare Value */}
          {needsCompareValue && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Compare Value</label>
              <input
                type="text"
                aria-label="Compare Value"
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                placeholder="What to compare it with"
                value={data.compareValue || ''}
                onChange={(e: ChangeEvent<HTMLInputElement>) => onCompareValueChangeRef.current?.(e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          )}
        </>
      )}

      <p className="text-slate-500 dark:text-slate-400 text-[10px] leading-snug m-0">
        Passes the value on at True when this holds, else at False.
      </p>
    </CollapsibleNodeWrapper>
  );
}

export default memo(ConditionNode);
