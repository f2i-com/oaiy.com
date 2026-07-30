import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position, useReactFlow, useNodeId, type NodeProps } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface MacroOutputNodeData {
    name?: string;
    outputType?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onCollapsedChange?: (value: boolean) => void;
}

const MacroOutputIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
    </svg>
);

const typeColors: Record<string, string> = {
    any: '!bg-slate-400',
    text: '!bg-emerald-500',
    number: '!bg-amber-500',
    image: '!bg-pink-500',
    video: '!bg-blue-500',
    audio: '!bg-purple-500',
    file: '!bg-orange-500',
};

function MacroOutputNode({ data }: NodeProps) {
    const nodeData = data as MacroOutputNodeData;
    const nodeId = useNodeId();
    const { updateNodeData } = useReactFlow();
    const onCollapsedChangeRef = useRef(nodeData.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = nodeData.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    const handleNameChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
        if (nodeId) updateNodeData(nodeId, { name: e.target.value });
    }, [nodeId, updateNodeData]);

    const handleTypeChange = useCallback((e: ChangeEvent<HTMLSelectElement>) => {
        if (nodeId) updateNodeData(nodeId, { outputType: e.target.value });
    }, [nodeId, updateNodeData]);

    const name = nodeData.name || 'output';
    const outputType = nodeData.outputType || 'any';

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-violet-400 font-medium">{name}</span>
            <span className="ml-1 text-slate-500">({outputType})</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        {
            id: 'value',
            type: 'target',
            position: Position.Left,
            color: typeColors[outputType] || '!bg-slate-400',
            size: 'lg',
            label: name
        },
    ], [outputType, name]);

    return (
        <CollapsibleNodeWrapper
            title="Macro Output"
            color="violet"
            icon={MacroOutputIcon}
            width={200}
            collapsedWidth={120}
            status={nodeData._status}
            isCollapsed={nodeData._collapsed}
            onCollapsedChange={handleCollapsedChange}
            collapsedPreview={collapsedPreview}
            inputHandles={inputHandles}
            outputHandles={[]}
        >
            {nodeData.showBodyProperties !== false && (
                <>
                    {/* Output Name */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Output Name</label>
                        <input
                            type="text"
                            value={name}
                            onChange={handleNameChange}
                            placeholder="output"
                            className="nodrag nowheel w-full px-2 py-1.5 text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded text-slate-800 dark:text-slate-200 focus:outline-none focus:border-violet-500"
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Output Type */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Type</label>
                        <select
                            value={outputType}
                            onChange={handleTypeChange}
                            className="nodrag nowheel w-full px-2 py-1.5 text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded text-slate-800 dark:text-slate-200 focus:outline-none focus:border-violet-500"
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="any">Any</option>
                            <option value="text">Text</option>
                            <option value="number">Number</option>
                            <option value="image">Image</option>
                            <option value="video">Video</option>
                            <option value="audio">Audio</option>
                            <option value="file">File</option>
                        </select>
                    </div>

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-violet-900/20 border border-violet-800/30 rounded text-xs text-violet-300">
                        <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Defines a macro output port</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(MacroOutputNode);
