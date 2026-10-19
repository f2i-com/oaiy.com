import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface DatabaseNodeData {
  /** The collection, as the node's definition (and Properties, and the run first) has it. */
  collection_name?: string;
  /** The card's own name for it before it wrote `collection_name` (kept in step). */
  collectionName?: string;
  _status?: 'running' | 'completed' | 'error' | 'stopped';
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onCollectionNameChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  onChange?: (field: string, value: unknown) => void;
}

interface DatabaseNodeProps {
  data: DatabaseNodeData;
}

const DatabaseIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
  </svg>
);

function DatabaseNode({ data }: DatabaseNodeProps) {
  const onCollectionNameChangeRef = useRef(data.onCollectionNameChange);
  const onChangeRef = useRef(data.onChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollectionNameChangeRef.current = data.onCollectionNameChange;
    onChangeRef.current = data.onChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  // The run reads `collection_name` before `collectionName`: the card wrote only the second, so a name set in
  // Properties (or by the agent) won over whatever was typed here, and the card showed "No name" for it.
  const collection = data.collection_name ?? data.collectionName ?? '';
  const handleCollectionNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    onChangeRef.current?.('collection_name', e.target.value);
    onCollectionNameChangeRef.current?.(e.target.value);
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const collapsedPreview = (
    <div className="text-slate-600 dark:text-slate-400">
      {collection ? (
        <span className="font-mono">{collection}</span>
      ) : (
        <span className="italic">No collection named yet</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'data', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'result', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Store Data"
      color="emerald"
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
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Collection</label>
            <input
              type="text"
              aria-label="Collection"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-emerald-500 font-mono"
              placeholder="my_data"
              value={collection}
              onChange={handleCollectionNameChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <p className="text-slate-500 dark:text-slate-400 text-[10px] leading-snug m-0">
            Stores what comes in on Data (any JSON) in this collection, kept between runs.
          </p>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(DatabaseNode);
