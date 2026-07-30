import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface DatabaseQueryNodeData {
  collectionName?: string;
  filterJson?: string;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onCollectionNameChange?: (value: string) => void;
  onFilterJsonChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
}

interface DatabaseQueryNodeProps {
  data: DatabaseQueryNodeData;
}

const DatabaseIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
  </svg>
);

function DatabaseQueryNode({ data }: DatabaseQueryNodeProps) {
  const onCollectionNameChangeRef = useRef(data.onCollectionNameChange);
  const onFilterJsonChangeRef = useRef(data.onFilterJsonChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollectionNameChangeRef.current = data.onCollectionNameChange;
    onFilterJsonChangeRef.current = data.onFilterJsonChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleCollectionNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onCollectionNameChangeRef.current?.(e.target.value);
  }, []);

  const handleFilterJsonChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onFilterJsonChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const collapsedPreview = (
    <div className="text-slate-400">
      {data.collectionName ? (
        <span className="text-blue-400 font-mono text-[10px]">{data.collectionName}</span>
      ) : (
        <span className="italic text-slate-500">No name</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'filter', type: 'target', position: Position.Left, color: '!bg-orange-500', size: 'lg' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'result', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Read Data"
      color="blue"
      icon={DatabaseIcon}
      width={200}
      collapsedWidth={120}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Collection Name</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
              placeholder="my_data"
              value={data.collectionName || ''}
              onChange={handleCollectionNameChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Filter (JSON)</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-blue-500 font-mono"
              placeholder='{}'
              value={data.filterJson || ''}
              onChange={handleFilterJsonChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <p className="text-slate-500 text-[10px]">
            Reads data from the collection. Connect a filter or use JSON.
          </p>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(DatabaseQueryNode);
