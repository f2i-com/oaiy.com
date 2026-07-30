import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface AudioFadeNodeData {
    fadeDuration?: number;
    fadeType?: 'exponential' | 'linear';
    fadeDirection?: 'in' | 'out';
    filename?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onChange?: (field: string, value: unknown) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface AudioFadeNodeProps {
    data: AudioFadeNodeData;
}

const FadeIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15.414a5 5 0 001.414 1.414M3.757 17.243a9 9 0 010-12.728" />
    </svg>
);

function AudioFadeNode({ data }: AudioFadeNodeProps) {
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

    const fadeDuration = data.fadeDuration ?? 10;
    const fadeType = data.fadeType ?? 'exponential';
    const fadeDirection = data.fadeDirection ?? 'out';

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-pink-400">{fadeDuration}s</span>
            <span className="mx-0.5">|</span>
            <span className="text-purple-400">{fadeType === 'exponential' ? 'exp' : 'lin'}</span>
            <span className="mx-0.5">|</span>
            <span className="text-blue-400">{fadeDirection}</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'target', position: Position.Left, color: '!bg-orange-500', size: 'lg', label: 'video' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'source', position: Position.Right, color: '!bg-orange-500', size: 'lg', label: 'video' },
        { id: 'path', type: 'source', position: Position.Right, color: '!bg-slate-400', size: 'sm', label: 'path' },
    ], []);

    return (
        <CollapsibleNodeWrapper
            title="Audio Fade"
            color="pink"
            icon={FadeIcon}
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
                    {/* Fade Direction */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Fade Direction</label>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                                    fadeDirection === 'out'
                                        ? 'bg-pink-600 text-white'
                                        : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                                }`}
                                onClick={() => handleChange('fadeDirection', 'out')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                Fade Out
                            </button>
                            <button
                                type="button"
                                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                                    fadeDirection === 'in'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                                }`}
                                onClick={() => handleChange('fadeDirection', 'in')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                Fade In
                            </button>
                        </div>
                    </div>

                    {/* Fade Type */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Fade Curve</label>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                                    fadeType === 'exponential'
                                        ? 'bg-purple-600 text-white'
                                        : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                                }`}
                                onClick={() => handleChange('fadeType', 'exponential')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                Exponential
                            </button>
                            <button
                                type="button"
                                className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                                    fadeType === 'linear'
                                        ? 'bg-cyan-600 text-white'
                                        : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600'
                                }`}
                                onClick={() => handleChange('fadeType', 'linear')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                Linear
                            </button>
                        </div>
                    </div>

                    {/* Fade Duration */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Fade Duration</label>
                            <span className="text-xs text-pink-400">{fadeDuration}s</span>
                        </div>
                        <input
                            type="range"
                            min="1"
                            max="30"
                            step="1"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                            value={fadeDuration}
                            onChange={(e) => handleChange('fadeDuration', parseInt(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                        <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                            <span>1s</span>
                            <span>30s</span>
                        </div>
                    </div>

                    {/* Filename */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Filename</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-pink-500"
                            placeholder="audio_faded"
                            value={data.filename ?? ''}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('filename', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs text-slate-600 dark:text-slate-400">
                        <svg className="w-4 h-4 text-pink-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Applies audio fade to video</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(AudioFadeNode);
