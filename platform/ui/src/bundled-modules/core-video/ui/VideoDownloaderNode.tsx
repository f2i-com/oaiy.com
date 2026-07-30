import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface VideoDownloaderNodeData {
    url?: string;
    mode?: 'video' | 'audio';
    start?: number;
    end?: number | null;
    quality?: string;
    apiUrl?: string;
    filePath?: string;
    duration?: number;
    width?: number;
    height?: number;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onChange?: (field: string, value: unknown) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface VideoDownloaderNodeProps {
    data: VideoDownloaderNodeData;
}

const DownloadIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
);

function VideoDownloaderNode({ data }: VideoDownloaderNodeProps) {
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    const handleChange = useCallback((field: string, value: unknown) => {
        data.onChange?.(field, value);
    }, [data]);

    const mode = data.mode ?? 'video';
    const startTime = data.start ?? 0;
    const endTime = data.end ?? null;
    const quality = data.quality ?? 'best';

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className={mode === 'video' ? 'text-orange-400' : 'text-red-400'}>{mode}</span>
            <span className="mx-0.5">|</span>
            <span className="text-orange-300">{quality}</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'url', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg', label: 'url' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'source', position: Position.Right, color: '!bg-orange-500', size: 'lg', label: mode === 'video' ? 'video' : 'audio' },
        { id: 'duration', type: 'source', position: Position.Right, color: '!bg-purple-500', size: 'sm', label: 'duration', style: { top: '70%' } },
    ], [mode]);

    return (
        <CollapsibleNodeWrapper
            title="Video Downloader"
            color="orange"
            icon={DownloadIcon}
            width={320}
            collapsedWidth={150}
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
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                            placeholder="http://127.0.0.1:8771/download"
                            value={data.apiUrl || 'http://127.0.0.1:8771/download'}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('apiUrl', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Video URL */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Video URL</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                            placeholder="https://youtube.com/watch?v=..."
                            value={data.url || ''}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('url', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Mode Selection */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Download Mode</label>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                                    mode === 'video'
                                        ? 'bg-orange-600 text-white'
                                        : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                                }`}
                                onClick={() => handleChange('mode', 'video')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                Video
                            </button>
                            <button
                                type="button"
                                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                                    mode === 'audio'
                                        ? 'bg-red-600 text-white'
                                        : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                                }`}
                                onClick={() => handleChange('mode', 'audio')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                Audio Only
                            </button>
                        </div>
                    </div>

                    {/* Quality (video mode only) */}
                    {mode === 'video' && (
                        <div>
                            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Quality</label>
                            <select
                                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                                value={quality}
                                onChange={(e) => handleChange('quality', e.target.value)}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                <option value="best">Best Available</option>
                                <option value="1080">1080p</option>
                                <option value="720">720p</option>
                                <option value="480">480p</option>
                                <option value="360">360p</option>
                            </select>
                        </div>
                    )}

                    {/* Time Range */}
                    <div className="flex gap-2">
                        <div className="flex-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Start (s)</label>
                            <input
                                type="number"
                                min="0"
                                step="1"
                                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                                placeholder="0"
                                value={startTime}
                                onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('start', parseFloat(e.target.value) || 0)}
                                onMouseDown={(e) => e.stopPropagation()}
                            />
                        </div>
                        <div className="flex-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">End (s)</label>
                            <input
                                type="number"
                                min="0"
                                step="1"
                                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                                placeholder="(full)"
                                value={endTime ?? ''}
                                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                                    const val = e.target.value;
                                    handleChange('end', val === '' ? null : parseFloat(val));
                                }}
                                onMouseDown={(e) => e.stopPropagation()}
                            />
                        </div>
                    </div>

                    {/* Download Result Preview */}
                    {data.filePath && (
                        <div className="flex items-center gap-2 px-2 py-1.5 bg-orange-900/20 border border-orange-800/30 rounded">
                            <svg className="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span className="text-xs text-orange-300 truncate flex-1">
                                {data.duration ? `${data.duration.toFixed(1)}s` : 'Downloaded'}
                                {data.width && data.height && ` (${data.width}x${data.height})`}
                            </span>
                        </div>
                    )}

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs text-slate-600 dark:text-slate-400">
                        <svg className="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>YouTube, Vimeo, TikTok + 1000 more</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(VideoDownloaderNode);
