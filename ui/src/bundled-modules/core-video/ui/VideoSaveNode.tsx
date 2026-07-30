import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';
import { pathToMediaUrl } from 'oaiy-core';


interface VideoSaveNodeData {
    videoUrl?: string;
    filename?: string;
    format?: 'mp4' | 'webm';
    path?: string;
    outputValue?: string; // Result path from workflow execution
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onFilenameChange?: (value: string) => void;
    onFormatChange?: (value: 'mp4' | 'webm') => void;
    onPathChange?: (value: string) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface VideoSaveNodeProps {
    data: VideoSaveNodeData;
}

const VideoSaveIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
);

function VideoSaveNode({ data }: VideoSaveNodeProps) {
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    // Get video source from either videoUrl or outputValue (result from workflow)
    const savedPath = data.outputValue || data.videoUrl || '';
    const hasVideo = savedPath !== '';

    // Convert local path to media server URL for playback
    const getVideoSrc = useMemo(() => {
        if (!savedPath) return '';
        // If already a URL, use as-is
        if (savedPath.startsWith('http') || savedPath.startsWith('data:')) return savedPath;
        // Convert local path to media server URL (uses dynamic port)
        return pathToMediaUrl(savedPath);
    }, [savedPath]);

    const collapsedPreview = (
        <div className="text-slate-400">
            <span className="text-orange-400">{data.format?.toUpperCase() || 'MP4'}</span>
            {data.filename && <span className="ml-1 text-[10px]">{data.filename}</span>}
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg', label: 'video' },
        { id: 'filename', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'sm', label: 'name' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'path', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
    ], []);

    const titleExtra = (
        <span className="ml-auto px-1.5 py-0.5 bg-orange-900 text-orange-400 text-[10px] rounded">
            AUTO
        </span>
    );

    return (
        <CollapsibleNodeWrapper
            title="Save Video"
            color="orange"
            icon={VideoSaveIcon}
            width={288}
            collapsedWidth={130}
            status={data._status}
            isCollapsed={data._collapsed}
            onCollapsedChange={handleCollapsedChange}
            collapsedPreview={collapsedPreview}
            inputHandles={inputHandles}
            outputHandles={outputHandles}
            titleExtra={titleExtra}
        >
            {data.showBodyProperties !== false && (
                <>
                    {/* Filename */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Filename</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                            placeholder="my_video"
                            value={data.filename ?? ''}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => data.onFilenameChange?.(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Format Select */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Format</label>
                        <select
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                            value={data.format || 'mp4'}
                            onChange={(e: ChangeEvent<HTMLSelectElement>) => data.onFormatChange?.(e.target.value as 'mp4' | 'webm')}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="mp4">MP4</option>
                            <option value="webm">WebM</option>
                        </select>
                    </div>

                    {/* Video Preview */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Preview</label>
                        <div className={`w-full bg-white dark:bg-slate-900 border rounded flex items-center justify-center overflow-hidden ${hasVideo ? 'border-orange-600' : 'border-slate-300 dark:border-slate-700 h-24'
                            }`}>
                            {hasVideo ? (
                                <video
                                    src={getVideoSrc}
                                    className="w-full rounded"
                                    controls
                                    playsInline
                                    style={{ maxHeight: '200px' }}
                                />
                            ) : (
                                <span className="text-slate-500 text-xs italic">No video</span>
                            )}
                        </div>
                    </div>

                    {/* Auto-save Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs text-slate-600 dark:text-slate-400">
                        <svg className="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Auto-saves during workflow execution</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(VideoSaveNode);
