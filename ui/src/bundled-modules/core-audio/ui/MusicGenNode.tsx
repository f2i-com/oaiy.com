import { memo, useRef, useEffect, useCallback, useMemo, useState, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface MusicGenNodeData {
    prompt?: string;
    lyrics?: string;
    apiUrl?: string;
    duration?: number;
    inferSteps?: number;
    guidanceScale?: number;
    // HeartMuLa-specific settings
    temperature?: number;
    topk?: number;
    cfgScale?: number;
    seed?: number;
    filename?: string;
    audioPath?: string;
    service?: 'ace-step' | 'heartmula';
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onChange?: (field: string, value: unknown) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface MusicGenNodeProps {
    data: MusicGenNodeData;
}

const MusicIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
    </svg>
);

function MusicGenNode({ data }: MusicGenNodeProps) {
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);
    const [isPlaying, setIsPlaying] = useState(false);
    const audioRef = useRef<HTMLAudioElement | null>(null);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    const handleChange = useCallback((field: string, value: unknown) => {
        data.onChange?.(field, value);
    }, [data]);

    const duration = data.duration ?? 60;
    const service = data.service ?? 'ace-step';

    // Get the default API URL based on selected service
    const getDefaultApiUrl = useCallback((svc: string) => {
        return svc === 'heartmula'
            ? 'http://127.0.0.1:8767/generate'
            : 'http://127.0.0.1:8766/generate';
    }, []);

    // Handle service change - update API URL to default for new service
    const handleServiceChange = useCallback((newService: string) => {
        handleChange('service', newService);
        handleChange('apiUrl', getDefaultApiUrl(newService));
    }, [handleChange, getDefaultApiUrl]);

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-purple-400">{duration}s</span>
            <span className="ml-1 text-slate-500">{service === 'heartmula' ? 'HM' : 'ACE'}</span>
            <span className="ml-1 truncate">{data.prompt?.substring(0, 12) || 'music'}...</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'prompt', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'lg', label: 'prompt' },
        { id: 'lyrics', type: 'target', position: Position.Left, color: '!bg-pink-500', size: 'md', label: 'lyrics' },
        { id: 'duration', type: 'target', position: Position.Left, color: '!bg-green-500', size: 'sm', label: 'duration' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'audio', type: 'source', position: Position.Right, color: '!bg-purple-500', size: 'lg', label: 'audio' },
        { id: 'path', type: 'source', position: Position.Right, color: '!bg-slate-400', size: 'sm', label: 'path' },
    ], []);

    const handlePlayPause = () => {
        if (!data.audioPath) return;

        if (!audioRef.current) {
            audioRef.current = new Audio(data.audioPath);
            audioRef.current.onended = () => setIsPlaying(false);
        }

        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            audioRef.current.play();
            setIsPlaying(true);
        }
    };

    return (
        <CollapsibleNodeWrapper
            title="Music Gen"
            color="purple"
            icon={MusicIcon}
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
                    {/* Service Selector */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Music Service</label>
                        <select
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            value={service}
                            onChange={(e: ChangeEvent<HTMLSelectElement>) => handleServiceChange(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="ace-step">ACE-Step (Pop/EDM)</option>
                            <option value="heartmula">HeartMuLa (Vocal/Lyrics)</option>
                        </select>
                    </div>

                    {/* API URL */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">API URL</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            placeholder={getDefaultApiUrl(service)}
                            value={data.apiUrl || getDefaultApiUrl(service)}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('apiUrl', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Prompt */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Prompt (style tags)</label>
                        <textarea
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500 resize-none"
                            placeholder="pop, energetic, catchy melody, female vocal..."
                            rows={2}
                            value={data.prompt || ''}
                            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => handleChange('prompt', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Lyrics */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Lyrics (optional)</label>
                        <textarea
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500 resize-none"
                            placeholder="[verse]&#10;Your lyrics here...&#10;[chorus]&#10;Catchy chorus..."
                            rows={3}
                            value={data.lyrics || ''}
                            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => handleChange('lyrics', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Duration */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Duration</label>
                            <span className="text-xs text-purple-400">{duration}s</span>
                        </div>
                        <input
                            type="range"
                            min="10"
                            max="240"
                            step="5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                            value={duration}
                            onChange={(e) => handleChange('duration', parseInt(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                        <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                            <span>10s</span>
                            <span>4 min</span>
                        </div>
                    </div>

                    {/* Filename */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Filename</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-purple-500"
                            placeholder="music_output"
                            value={data.filename ?? ''}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('filename', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Audio Preview */}
                    {data.audioPath && (
                        <div className="flex items-center gap-2 px-2 py-1.5 bg-purple-900/20 border border-purple-800/30 rounded">
                            <button
                                type="button"
                                aria-label={isPlaying ? 'Pause music' : 'Play music'}
                                className="p-1.5 rounded bg-purple-600 hover:bg-purple-500 transition-colors"
                                onClick={handlePlayPause}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                {isPlaying ? (
                                    <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                                    </svg>
                                ) : (
                                    <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M8 5v14l11-7z" />
                                    </svg>
                                )}
                            </button>
                            <span className="text-xs text-purple-300 truncate flex-1">Music generated</span>
                        </div>
                    )}

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs text-slate-600 dark:text-slate-400">
                        <svg className="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>
                            {service === 'heartmula'
                                ? 'HeartMuLa: FP8 quantized, great for vocals'
                                : 'ACE-Step: Full precision, great for EDM/pop'}
                        </span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(MusicGenNode);
