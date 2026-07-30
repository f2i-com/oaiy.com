import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position, useReactFlow, useNodeId } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface VideoAvatarNodeData {
    image?: string;
    audio?: string;
    prompt?: string;
    apiUrl?: string;
    guidanceScale?: number;
    numInferenceSteps?: number;
    videoPath?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onCollapsedChange?: (value: boolean) => void;
}

interface VideoAvatarNodeProps {
    data: VideoAvatarNodeData;
}

const VideoAvatarIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
);

function VideoAvatarNode({ data }: VideoAvatarNodeProps) {
    const nodeId = useNodeId();
    const { updateNodeData } = useReactFlow();
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    // Create stable property update handlers using useReactFlow
    const handleApiUrlChange = useCallback((value: string) => {
        if (nodeId) updateNodeData(nodeId, { apiUrl: value });
    }, [nodeId, updateNodeData]);

    const handlePromptChange = useCallback((value: string) => {
        if (nodeId) updateNodeData(nodeId, { prompt: value });
    }, [nodeId, updateNodeData]);

    const handleGuidanceScaleChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { guidanceScale: value });
    }, [nodeId, updateNodeData]);

    const handleInferenceStepsChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { numInferenceSteps: value });
    }, [nodeId, updateNodeData]);

    const guidanceScale = data.guidanceScale ?? 5.0;
    const inferenceSteps = data.numInferenceSteps ?? 30;

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-purple-400">Ditto Avatar</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'image', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'lg', label: 'image' },
        { id: 'audio', type: 'target', position: Position.Left, color: '!bg-teal-500', size: 'lg', label: 'audio' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'source', position: Position.Right, color: '!bg-blue-500', size: 'lg', label: 'video' },
    ], []);

    return (
        <CollapsibleNodeWrapper
            title="Video Avatar"
            color="purple"
            icon={VideoAvatarIcon}
            width={320}
            collapsedWidth={140}
            status={data._status}
            isCollapsed={data._collapsed}
            onCollapsedChange={handleCollapsedChange}
            collapsedPreview={collapsedPreview}
            inputHandles={inputHandles}
            outputHandles={outputHandles}
        >
            {data.showBodyProperties !== false && (
                <>
                    {/* API URL */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">API URL</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            placeholder="http://127.0.0.1:8768/generate"
                            value={data.apiUrl || 'http://127.0.0.1:8768/generate'}
                            onChange={(e) => handleApiUrlChange(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Prompt (optional) */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Prompt (optional)</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            placeholder="A person talking naturally"
                            value={data.prompt || ''}
                            onChange={(e) => handlePromptChange(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Guidance Scale */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Guidance Scale</label>
                            <span className="text-xs text-purple-400">{guidanceScale.toFixed(1)}</span>
                        </div>
                        <input
                            type="range"
                            min="1"
                            max="15"
                            step="0.5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                            value={guidanceScale}
                            onChange={(e) => handleGuidanceScaleChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Inference Steps */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Inference Steps</label>
                            <span className="text-xs text-purple-400">{inferenceSteps}</span>
                        </div>
                        <input
                            type="range"
                            min="10"
                            max="100"
                            step="5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                            value={inferenceSteps}
                            onChange={(e) => handleInferenceStepsChange(parseInt(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-purple-900/20 border border-purple-800/30 rounded text-xs text-purple-300">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Generates lip-synced video using Ditto</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(VideoAvatarNode);
