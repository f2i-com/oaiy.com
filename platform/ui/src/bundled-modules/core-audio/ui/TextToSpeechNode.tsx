import { memo, useRef, useEffect, useCallback, useMemo, useState } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

type Provider = 'internal' | 'external';
type ResponseType = 'file_path' | 'url' | 'base64_wav' | 'base64_mp3' | 'binary_wav' | 'binary_mp3';

interface TextToSpeechNodeData {
    // Provider switch
    provider?: Provider;

    // Internal mode
    internalModel?: string;
    internalCfgScale?: number;
    internalDurationMultiplier?: number;
    internalDurationSeconds?: number;
    internalStreaming?: boolean;
    internalFastMode?: boolean;
    internalSeed?: number;

    // External mode
    externalPreset?: string;
    externalUrl?: string;
    externalMethod?: 'POST' | 'GET';
    externalHeaders?: string;
    externalApiKeyConstant?: string;
    externalBodyTemplate?: string;
    externalResponseType?: ResponseType;
    externalResponsePath?: string;

    // Shared
    description?: string;
    descriptionInput?: string;
    speaker?: string;
    language?: string;
    outputFormat?: 'wav' | 'mp3';
    filename?: string;

    audioPath?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;

    onChange?: (field: string, value: unknown) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface TextToSpeechNodeProps {
    data: TextToSpeechNodeData;
}

// Local TTS service templates. Picking one auto-fills url + method + headers +
// body template + response handling so the user doesn't hand-craft it. The
// flow-builder pivot (2026-05-20) is local-only: cloud services (ElevenLabs /
// OpenAI TTS) were removed — re-add an entry here with its api-key header if
// you ever want one back. "Custom" covers any other local HTTP TTS server.
type ExternalPreset = {
    id: string;
    label: string;
    url: string;
    method: 'POST' | 'GET';
    headers: string;
    body: string;
    responseType: ResponseType;
    responsePath: string;
    apiKeyHint: string;
};

const EXTERNAL_PRESETS: ExternalPreset[] = [
    {
        id: 'chatterbox-tts',
        label: 'Chatterbox TTS (local, port 8765)',
        url: 'http://127.0.0.1:8765/tts',
        method: 'POST',
        headers: '{"Content-Type": "application/json"}',
        body: '{"text": {{text}}, "audio_prompt_path": {{voiceRefPath}}, "language_id": {{language}}}',
        responseType: 'file_path',
        responsePath: 'audio_path',
        apiKeyHint: '',
    },
    {
        id: 'qwen3-tts',
        label: 'Qwen3 TTS (local, port 8772)',
        url: 'http://127.0.0.1:8772/tts',
        method: 'POST',
        headers: '{"Content-Type": "application/json"}',
        body: '{"text": {{text}}, "speaker": {{speaker}}, "language": {{language}}, "voice_description": {{description}}}',
        responseType: 'file_path',
        responsePath: 'audio_path',
        apiKeyHint: '',
    },
    {
        id: 'custom',
        label: 'Custom',
        url: 'http://127.0.0.1:8000/tts',
        method: 'POST',
        headers: '{"Content-Type": "application/json"}',
        body: '{"text": {{text}}}',
        responseType: 'file_path',
        responsePath: 'audio_path',
        apiKeyHint: '',
    },
];

const TTSIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
    </svg>
);

function TextToSpeechNode({ data }: TextToSpeechNodeProps) {
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);
    const [isPlaying, setIsPlaying] = useState(false);
    const [showExternalAdvanced, setShowExternalAdvanced] = useState(false);
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

    // External-only since the flow-builder pivot (2026-05-20): the bundled
    // DramaBox (plugin-tts) backend was removed, so TTS always goes through an
    // external HTTP endpoint. `provider` is pinned to 'external' (the Internal
    // UI block was deleted); the `Provider` union is kept so re-adding Internal
    // later is a localized change.
    const provider: Provider = 'external';
    const preset = data.externalPreset ?? 'chatterbox-tts';
    const selectedPreset = useMemo(
        () => EXTERNAL_PRESETS.find(p => p.id === preset) ?? EXTERNAL_PRESETS[0],
        [preset],
    );

    // When the user picks a different preset, auto-fill the dependent
    // fields (URL / method / headers / body / response handling / api key
    // hint). They can still tweak any one of them afterwards.
    const handlePresetChange = useCallback((newId: string) => {
        handleChange('externalPreset', newId);
        const p = EXTERNAL_PRESETS.find(x => x.id === newId);
        if (!p || p.id === 'custom') return;
        handleChange('externalUrl', p.url);
        handleChange('externalMethod', p.method);
        handleChange('externalHeaders', p.headers);
        handleChange('externalBodyTemplate', p.body);
        handleChange('externalResponseType', p.responseType);
        handleChange('externalResponsePath', p.responsePath);
        if (p.apiKeyHint) handleChange('externalApiKeyConstant', p.apiKeyHint);
    }, [handleChange]);

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-teal-400">
                {selectedPreset.label.split(' (')[0]}
            </span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'text', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'lg', label: 'text' },
        { id: 'descriptionInput', type: 'target', position: Position.Left, color: '!bg-purple-500', size: 'md', label: 'desc' },
        { id: 'audioPrompt', type: 'target', position: Position.Left, color: '!bg-teal-500', size: 'md', label: 'voice' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'audio', type: 'source', position: Position.Right, color: '!bg-teal-500', size: 'lg', label: 'audio' },
        { id: 'path', type: 'source', position: Position.Right, color: '!bg-slate-500', size: 'sm', label: 'path' },
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

    // (Internal/External TabButton removed — external-only node now.)

    const labelClass = 'text-slate-600 dark:text-slate-400 text-xs block mb-1';
    const inputClass = 'nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500';
    const textareaClass = `${inputClass} resize-none font-mono text-[11px]`;

    return (
        <CollapsibleNodeWrapper
            title="Text to Speech"
            color="teal"
            icon={TTSIcon}
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
                    {/* External-only (flow-builder pivot, 2026-05-20): bundled
                        DramaBox removed; the local-service dropdown below is the
                        only choice. No Internal/External toggle. */}

                    {/* ============== EXTERNAL ============== */}
                    {provider === 'external' && (
                        <>
                            <div>
                                <label className={labelClass}>Service</label>
                                <select
                                    className={inputClass}
                                    value={preset}
                                    onChange={(e) => handlePresetChange(e.target.value)}
                                    onMouseDown={(e) => e.stopPropagation()}
                                >
                                    {EXTERNAL_PRESETS.map(p => (
                                        <option key={p.id} value={p.id}>{p.label}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className={labelClass}>API URL</label>
                                <input type="text" className={inputClass}
                                    value={data.externalUrl ?? selectedPreset.url}
                                    onChange={(e) => handleChange('externalUrl', e.target.value)}
                                    onMouseDown={(e) => e.stopPropagation()} />
                            </div>

                            {(selectedPreset.apiKeyHint !== '' || (data.externalApiKeyConstant && data.externalApiKeyConstant !== '')) && (
                                <div>
                                    <label className={labelClass}>API Key (secret name)</label>
                                    <input type="text" className={inputClass}
                                        placeholder={selectedPreset.apiKeyHint || 'YOUR_API_KEY'}
                                        value={data.externalApiKeyConstant ?? selectedPreset.apiKeyHint}
                                        onChange={(e) => handleChange('externalApiKeyConstant', e.target.value)}
                                        onMouseDown={(e) => e.stopPropagation()} />
                                    <p className="text-[10px] text-slate-500 mt-1">From OAIY Secrets. Available as {'{{apiKey}}'} in headers/body.</p>
                                </div>
                            )}

                            <button
                                type="button"
                                className="text-xs text-teal-400 hover:text-teal-300 text-left"
                                onClick={() => setShowExternalAdvanced(v => !v)}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                {showExternalAdvanced ? '▾' : '▸'} Advanced (headers, body, response)
                            </button>
                            {showExternalAdvanced && (
                                <>
                                    <div>
                                        <label className={labelClass}>Method</label>
                                        <select className={inputClass} value={data.externalMethod ?? selectedPreset.method}
                                            onChange={(e) => handleChange('externalMethod', e.target.value as 'POST' | 'GET')}
                                            onMouseDown={(e) => e.stopPropagation()}
                                        >
                                            <option value="POST">POST</option>
                                            <option value="GET">GET</option>
                                        </select>
                                    </div>

                                    <div>
                                        <label className={labelClass}>Headers (JSON)</label>
                                        <textarea className={textareaClass} rows={2}
                                            value={data.externalHeaders ?? selectedPreset.headers}
                                            onChange={(e) => handleChange('externalHeaders', e.target.value)}
                                            onMouseDown={(e) => e.stopPropagation()} />
                                    </div>

                                    <div>
                                        <label className={labelClass}>Body (template)</label>
                                        <textarea className={textareaClass} rows={3}
                                            value={data.externalBodyTemplate ?? selectedPreset.body}
                                            onChange={(e) => handleChange('externalBodyTemplate', e.target.value)}
                                            onMouseDown={(e) => e.stopPropagation()} />
                                        <p className="text-[10px] text-slate-500 mt-1">{'{{text}} {{voiceRefPath}} {{description}} {{speaker}} {{language}}'} are JSON-escaped. {'{{textRaw}}'} for unquoted.</p>
                                    </div>

                                    <div>
                                        <label className={labelClass}>Response Type</label>
                                        <select className={inputClass} value={data.externalResponseType ?? selectedPreset.responseType}
                                            onChange={(e) => handleChange('externalResponseType', e.target.value as ResponseType)}
                                            onMouseDown={(e) => e.stopPropagation()}
                                        >
                                            <option value="binary_mp3">Binary MP3 (body is audio)</option>
                                            <option value="binary_wav">Binary WAV (body is audio)</option>
                                            <option value="base64_mp3">Base64 MP3 (in JSON field)</option>
                                            <option value="base64_wav">Base64 WAV (in JSON field)</option>
                                            <option value="url">URL (in JSON field)</option>
                                            <option value="file_path">Server file path (in JSON field)</option>
                                        </select>
                                    </div>

                                    {(data.externalResponseType ?? selectedPreset.responseType).startsWith('binary_') ? null : (
                                        <div>
                                            <label className={labelClass}>Response Path</label>
                                            <input type="text" className={inputClass}
                                                placeholder="audio_path or data.url or [0].audio"
                                                value={data.externalResponsePath ?? selectedPreset.responsePath}
                                                onChange={(e) => handleChange('externalResponsePath', e.target.value)}
                                                onMouseDown={(e) => e.stopPropagation()} />
                                        </div>
                                    )}
                                </>
                            )}
                        </>
                    )}

                    {/* ============== SHARED ============== */}
                    <div className="pt-2 border-t border-slate-700/50">
                        <label className={labelClass}>Voice Description (optional)</label>
                        <textarea
                            className={inputClass}
                            placeholder="A warm narrator, A cheerful young woman..."
                            rows={2}
                            value={data.description || ''}
                            onChange={(e) => handleChange('description', e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {provider === 'external' && (
                        <div>
                            <label className={labelClass}>Speaker / Voice ID</label>
                            <input type="text" className={inputClass}
                                placeholder="ElevenLabs voice ID, or 'alloy', etc."
                                value={data.speaker || ''}
                                onChange={(e) => handleChange('speaker', e.target.value)}
                                onMouseDown={(e) => e.stopPropagation()} />
                        </div>
                    )}

                    <div className="grid grid-cols-[1fr_auto_1fr] gap-2 items-end">
                        <div>
                            <label className={labelClass}>Language</label>
                            <select className={inputClass} value={data.language || 'Auto'}
                                onChange={(e) => handleChange('language', e.target.value)}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                <option value="Auto">Auto</option>
                                <option value="English">English</option>
                                <option value="Chinese">Chinese</option>
                                <option value="Japanese">Japanese</option>
                                <option value="Korean">Korean</option>
                                <option value="Spanish">Spanish</option>
                                <option value="French">French</option>
                                <option value="German">German</option>
                                <option value="Portuguese">Portuguese</option>
                            </select>
                        </div>
                        <div>
                            <label className={labelClass}>Format</label>
                            <select className={inputClass} value={data.outputFormat || 'wav'}
                                onChange={(e) => handleChange('outputFormat', e.target.value as 'wav' | 'mp3')}
                                onMouseDown={(e) => e.stopPropagation()}
                            >
                                <option value="wav">WAV</option>
                                <option value="mp3">MP3</option>
                            </select>
                        </div>
                        <div>
                            <label className={labelClass}>Filename</label>
                            <input type="text" className={inputClass} placeholder="tts_output"
                                value={data.filename ?? ''}
                                onChange={(e) => handleChange('filename', e.target.value)}
                                onMouseDown={(e) => e.stopPropagation()} />
                        </div>
                    </div>

                    {data.audioPath && (
                        <div className="flex items-center gap-2 px-2 py-1.5 bg-teal-900/20 border border-teal-800/30 rounded">
                            <button
                                type="button"
                                className="p-1.5 rounded bg-teal-600 hover:bg-teal-500 transition-colors"
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
                            <span className="text-xs text-teal-300 truncate flex-1">Audio generated</span>
                        </div>
                    )}
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(TextToSpeechNode);
