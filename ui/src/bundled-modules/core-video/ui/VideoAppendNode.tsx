import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface VideoAppendNodeData {
    videos?: string[]; // Array input (from loops)
    video_1?: string;
    video_2?: string;
    video_3?: string;
    video_4?: string;
    outputPath?: string;
    filename?: string;
    format?: 'mp4' | 'webm';
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onFilenameChange?: (value: string) => void;
    onFormatChange?: (value: 'mp4' | 'webm') => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface VideoAppendNodeProps {
    data: VideoAppendNodeData;
}

const VideoAppendIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
    </svg>
);

function VideoAppendNode({ data }: VideoAppendNodeProps) {
    // Count from array or individual inputs
    const arrayCount = data.videos?.length || 0;
    const individualCount = [data.video_1, data.video_2, data.video_3, data.video_4].filter(Boolean).length;
    const totalCount = arrayCount > 0 ? arrayCount : individualCount;
    const usingArray = arrayCount > 0;

    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    const collapsedPreview = (
        <div className="text-slate-400">
            <span className="text-purple-400">{totalCount}</span>
            <span className="ml-1 text-[10px]">{usingArray ? 'array' : 'videos'}</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        // Array input (primary - for loops)
        { id: 'videos', type: 'target', position: Position.Left, color: '!bg-purple-500', size: 'lg', label: 'videos[]' },
        // Individual inputs (fallback)
        { id: 'video_1', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'sm', label: 'video 1' },
        { id: 'video_2', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'sm', label: 'video 2' },
        { id: 'video_3', type: 'target', position: Position.Left, color: '!bg-blue-400', size: 'sm', label: 'video 3' },
        { id: 'video_4', type: 'target', position: Position.Left, color: '!bg-blue-400', size: 'sm', label: 'video 4' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'source', position: Position.Right, color: '!bg-blue-500', size: 'lg' },
        { id: 'path', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'sm' },
    ], []);

    return (
        <CollapsibleNodeWrapper
            title="Video Append"
            color="purple"
            icon={VideoAppendIcon}
            width={280}
            collapsedWidth={130}
            status={data._status}
            isCollapsed={data._collapsed}
            onCollapsedChange={handleCollapsedChange}
            collapsedPreview={collapsedPreview}
            inputHandles={inputHandles}
            outputHandles={outputHandles}
        >
            {data.showBodyProperties !== false && (
                <>
                    {/* Connection Status */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs">
                        <svg className="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
                        </svg>
                        <span className="text-slate-400">
                            {usingArray ? (
                                <>
                                    <span className="text-purple-400 font-medium">{totalCount}</span> videos from array
                                </>
                            ) : (
                                <>
                                    <span className="text-purple-400 font-medium">{totalCount}</span> video{totalCount !== 1 ? 's' : ''} connected
                                </>
                            )}
                        </span>
                    </div>

                    {/* Filename */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Output Filename</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            placeholder="appended_video"
                            value={data.filename ?? ''}
                            onChange={(e) => data.onFilenameChange?.(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Format Select */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Format</label>
                        <select
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            value={data.format || 'mp4'}
                            onChange={(e) => data.onFormatChange?.(e.target.value as 'mp4' | 'webm')}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="mp4">MP4</option>
                            <option value="webm">WebM</option>
                        </select>
                    </div>

                    {/* Info */}
                    <div className="flex items-start gap-2 px-2 py-1.5 bg-purple-900/20 border border-purple-800/30 rounded text-xs text-purple-300">
                        <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Use <span className="text-purple-200">videos[]</span> for array input (from loops) or connect individual videos (1→2→3→4)</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(VideoAppendNode);
