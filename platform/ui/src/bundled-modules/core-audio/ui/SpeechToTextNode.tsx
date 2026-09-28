import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    speaker?: string;
}

interface SpeechToTextNodeData {
    apiUrl?: string;
    language?: string;
    enableWordTimestamps?: boolean;
    enableDiarization?: boolean;
    minSpeakers?: number;
    maxSpeakers?: number;
    // Output data
    transcriptText?: string;
    segments?: TranscriptSegment[];
    detectedLanguage?: string;
    duration?: number;
    // Node state
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    // Generic onChange handler (used by OAIYBuilder)
    onChange?: (field: string, value: unknown) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface SpeechToTextNodeProps {
    data: SpeechToTextNodeData;
}

const STTIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
);

function SpeechToTextNode({ data }: SpeechToTextNodeProps) {
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    // Generic change handler that uses data.onChange
    const handleChange = useCallback((field: string, value: unknown) => {
        data.onChange?.(field, value);
    }, [data]);

    const enableDiarization = data.enableDiarization ?? false;

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-violet-400">{data.language || 'auto-detect'}</span>
            {data.detectedLanguage && (
                <span className="ml-1 text-emerald-400">({data.detectedLanguage})</span>
            )}
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'media', type: 'target', position: Position.Left, color: '!bg-teal-500', size: 'lg', label: 'media' },
        { id: 'startTime', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'md', label: 'start' },
        { id: 'endTime', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'md', label: 'end' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'text', type: 'source', position: Position.Right, color: '!bg-amber-500', size: 'lg', label: 'text' },
        { id: 'segments', type: 'source', position: Position.Right, color: '!bg-blue-500', size: 'md', label: 'segments' },
        { id: 'language', type: 'source', position: Position.Right, color: '!bg-purple-500', size: 'sm', label: 'lang' },
        { id: 'duration', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'sm', label: 'duration' },
    ], []);

    // Format duration as mm:ss
    const formatDuration = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <CollapsibleNodeWrapper
            title="Speech to Text"
            color="violet"
            icon={STTIcon}
            width={300}
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
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-violet-500"
                            placeholder="http://127.0.0.1:8770/transcribe"
                            value={data.apiUrl || 'http://127.0.0.1:8770/transcribe'}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('apiUrl', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Language */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Language</label>
                        <select
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-violet-500"
                            value={data.language || ''}
                            onChange={(e: ChangeEvent<HTMLSelectElement>) => handleChange('language', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="">Auto-detect</option>
                            <option value="en">English</option>
                            <option value="es">Spanish</option>
                            <option value="fr">French</option>
                            <option value="de">German</option>
                            <option value="it">Italian</option>
                            <option value="pt">Portuguese</option>
                            <option value="nl">Dutch</option>
                            <option value="ru">Russian</option>
                            <option value="zh">Chinese</option>
                            <option value="ja">Japanese</option>
                            <option value="ko">Korean</option>
                            <option value="ar">Arabic</option>
                            <option value="hi">Hindi</option>
                        </select>
                    </div>

                    {/* Word Timestamps Toggle */}
                    <div className="flex items-center justify-between">
                        <label className="text-slate-600 dark:text-slate-400 text-xs">Word Timestamps</label>
                        <button
                            type="button"
                            role="switch"
                            aria-checked={data.enableWordTimestamps !== false}
                            aria-label="Word Timestamps"
                            className={`nodrag relative w-10 h-5 rounded-full transition-colors ${
                                data.enableWordTimestamps !== false ? 'bg-violet-600' : 'bg-slate-600'
                            }`}
                            onClick={() => handleChange('enableWordTimestamps', data.enableWordTimestamps === false)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <span
                                className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                                    data.enableWordTimestamps !== false ? 'left-5' : 'left-0.5'
                                }`}
                            />
                        </button>
                    </div>

                    {/* Speaker Diarization Toggle */}
                    <div className="flex items-center justify-between">
                        <label className="text-slate-600 dark:text-slate-400 text-xs">Speaker Diarization</label>
                        <button
                            type="button"
                            role="switch"
                            aria-checked={enableDiarization}
                            aria-label="Speaker Diarization"
                            className={`nodrag relative w-10 h-5 rounded-full transition-colors ${
                                enableDiarization ? 'bg-violet-600' : 'bg-slate-600'
                            }`}
                            onClick={() => handleChange('enableDiarization', !enableDiarization)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <span
                                className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                                    enableDiarization ? 'left-5' : 'left-0.5'
                                }`}
                            />
                        </button>
                    </div>

                    {/* Speaker count options (only when diarization is enabled) */}
                    {enableDiarization && (
                        <div className="flex gap-2">
                            <div className="flex-1">
                                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Min Speakers</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="20"
                                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-violet-500"
                                    placeholder="Auto"
                                    value={data.minSpeakers ?? ''}
                                    onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('minSpeakers', e.target.value ? parseInt(e.target.value) : null)}
                                    onMouseDown={(e) => e.stopPropagation()}
                                />
                            </div>
                            <div className="flex-1">
                                <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Max Speakers</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="20"
                                    className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-violet-500"
                                    placeholder="Auto"
                                    value={data.maxSpeakers ?? ''}
                                    onChange={(e: ChangeEvent<HTMLInputElement>) => handleChange('maxSpeakers', e.target.value ? parseInt(e.target.value) : null)}
                                    onMouseDown={(e) => e.stopPropagation()}
                                />
                            </div>
                        </div>
                    )}

                    {/* Diarization info */}
                    {enableDiarization && (
                        <div className="flex items-center gap-2 px-2 py-1.5 bg-amber-900/20 border border-amber-800/30 rounded text-xs text-amber-300">
                            <svg className="w-4 h-4 text-amber-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            <span>Requires HF_TOKEN in service config</span>
                        </div>
                    )}

                    {/* Results Preview (when completed) */}
                    {data.transcriptText && (
                        <div className="space-y-2">
                            {/* Duration and language */}
                            <div className="flex items-center justify-between text-xs">
                                {data.duration !== undefined && (
                                    <span className="text-slate-400">
                                        Duration: <span className="text-violet-400">{formatDuration(data.duration)}</span>
                                    </span>
                                )}
                                {data.detectedLanguage && (
                                    <span className="text-slate-400">
                                        Language: <span className="text-emerald-400">{data.detectedLanguage}</span>
                                    </span>
                                )}
                            </div>

                            {/* Transcript preview */}
                            <div className="bg-slate-100/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-700 rounded p-2">
                                <div className="text-xs text-slate-400 mb-1">Transcript Preview:</div>
                                <div className="nowheel text-xs text-slate-700 dark:text-slate-300 max-h-24 overflow-y-auto">
                                    {data.transcriptText.substring(0, 200)}
                                    {data.transcriptText.length > 200 && '...'}
                                </div>
                            </div>

                            {/* Segment count */}
                            {data.segments && data.segments.length > 0 && (
                                <div className="text-xs text-slate-400">
                                    <span className="text-blue-400">{data.segments.length}</span> segments detected
                                </div>
                            )}
                        </div>
                    )}

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs text-slate-600 dark:text-slate-400">
                        <svg className="w-4 h-4 text-violet-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Uses WhisperX for accurate transcription</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(SpeechToTextNode);
