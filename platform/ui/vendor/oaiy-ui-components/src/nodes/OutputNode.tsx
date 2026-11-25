import React, { memo, useRef, useState, useEffect } from 'react';
import { Position } from '@xyflow/react';
import type { NodeDefinition } from 'oaiy-core';
import CollapsibleNodeWrapper, { type HandleConfig } from '../components/CollapsibleNodeWrapper';

// ============================================
// Types
// ============================================

export interface OutputNodeProps {
    id: string;
    data: Record<string, unknown> & {
        __definition?: NodeDefinition;
        result?: unknown;
        outputValue?: unknown;
        label?: string;
        _updateKey?: number;
        _collapsed?: boolean;
        _status?: 'running' | 'completed' | 'error' | 'stopped';
        onCollapsedChange?: (collapsed: boolean) => void;
    };
    selected?: boolean;
}

/** Allow only safe browser-loadable media URLs (never synthesize raw file:// URIs). */
function safeMediaSrc(raw: string): string | null {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    if (
        trimmed.startsWith('data:') ||
        trimmed.startsWith('blob:') ||
        trimmed.startsWith('https://') ||
        trimmed.startsWith('http://') ||
        trimmed.startsWith('/')
    ) {
        return trimmed;
    }
    return null;
}

// Helper to detect if a value is an audio file path
function isAudioPath(value: unknown): string | null {
    if (typeof value !== 'string') return null;
    const audioExtensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'];
    const lowerValue = value.toLowerCase();
    for (const ext of audioExtensions) {
        if (lowerValue.endsWith(ext)) {
            return value;
        }
    }
    return null;
}

// Helper to detect if a value is an image path/URL
function isImagePath(value: unknown): string | null {
    if (typeof value !== 'string') return null;
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp'];
    const lowerValue = value.toLowerCase();

    // Check for data URLs
    if (lowerValue.startsWith('data:image/')) {
        return value;
    }

    // Raw base64 image payload with no `data:` prefix — e.g. OpenAI's
    // gpt-image endpoint returns `data[0].b64_json`, which a Service Call node
    // surfaces verbatim. Detect by the format's magic bytes (base64-encoded)
    // and wrap as a data URL so the <img> below renders the picture. Cheap
    // prefix check first; only validate the (potentially large) base64 body
    // when a prefix matches.
    let mime: string | null = null;
    if (value.startsWith('iVBORw0KGgo')) mime = 'image/png';
    else if (value.startsWith('/9j/')) mime = 'image/jpeg';
    else if (value.startsWith('R0lGOD')) mime = 'image/gif';
    else if (value.startsWith('UklGR')) mime = 'image/webp';
    if (mime && value.length > 64 && /^[A-Za-z0-9+/=\s]+$/.test(value)) {
        return `data:${mime};base64,${value.replace(/\s/g, '')}`;
    }

    for (const ext of imageExtensions) {
        if (lowerValue.endsWith(ext)) {
            return value;
        }
    }
    return null;
}

// Helper to detect if a value is a video path
function isVideoPath(value: unknown): string | null {
    if (typeof value !== 'string') return null;
    const videoExtensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv'];
    const lowerValue = value.toLowerCase();
    for (const ext of videoExtensions) {
        if (lowerValue.endsWith(ext)) {
            return value;
        }
    }
    return null;
}

// ============================================
// OutputNode Component
// ============================================

// Custom comparison for memo - detect changes in result, status, collapse, selection, and _updateKey
const arePropsEqual = (prevProps: OutputNodeProps, nextProps: OutputNodeProps): boolean => {
    if (prevProps.id !== nextProps.id) return false;
    if (prevProps.selected !== nextProps.selected) return false;
    if (prevProps.data.result !== nextProps.data.result) return false;
    if (prevProps.data.outputValue !== nextProps.data.outputValue) return false;
    if (prevProps.data._updateKey !== nextProps.data._updateKey) return false;
    if (prevProps.data._collapsed !== nextProps.data._collapsed) return false;
    if (prevProps.data._status !== nextProps.data._status) return false;
    if (prevProps.data.label !== nextProps.data.label) return false;
    return true;
};

export const OutputNode: React.FC<OutputNodeProps> = memo(({ id: _id, data }) => {
    const definition = data.__definition;
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);

    // Get the result value (could be from input connection or stored)
    const resultValue = data.outputValue !== undefined ? data.outputValue : data.result;
    const label = (data.label as string) || 'Output';

    // Detect media types
    const audioPath = isAudioPath(resultValue);
    const imagePath = isImagePath(resultValue);
    const videoPath = isVideoPath(resultValue);
    const safeAudioSrc = audioPath ? safeMediaSrc(audioPath) : null;
    const safeImageSrc = imagePath ? safeMediaSrc(imagePath) : null;
    const safeVideoSrc = videoPath ? safeMediaSrc(videoPath) : null;

    // Handle play/pause
    const togglePlay = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            void audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
    };

    // Listen for audio end
    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const handleEnded = () => setIsPlaying(false);
        audio.addEventListener('ended', handleEnded);
        return () => audio.removeEventListener('ended', handleEnded);
    }, [safeAudioSrc]);

    // Default definition fallback
    const inputHandles: HandleConfig[] = [{
        id: 'result',
        type: 'target',
        position: Position.Left,
        color: 'port-any',
        size: 'sm',
        label: 'Result',
    }];

    // Format result for display
    const formatResult = (value: unknown): string => {
        if (value === undefined || value === null) return '—';
        if (typeof value === 'object') {
            try {
                return JSON.stringify(value, null, 2).slice(0, 200);
            } catch {
                return String(value);
            }
        }
        return String(value).slice(0, 200);
    };

    return (
        <CollapsibleNodeWrapper
            title={label}
            color="emerald"
            icon={
                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            }
            width={definition?.ui?.width || 220}
            status={data._status}
            isCollapsed={data._collapsed}
            onCollapsedChange={data.onCollapsedChange}
            inputHandles={inputHandles}
            outputHandles={[]}
        >
            <div className="flex flex-col gap-2">
                {/* Audio Player */}
                {audioPath && (
                    <div className="bg-[rgb(var(--color-bg-tertiary))] rounded-[var(--r-sm)] p-2">
                        <div className="flex items-center gap-2">
                            {safeAudioSrc && (
                                <button
                                    type="button"
                                    onClick={togglePlay}
                                    aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
                                    className="w-8 h-8 rounded-full bg-emerald-600 hover:bg-emerald-500 flex items-center justify-center transition-colors shrink-0"
                                >
                                    {isPlaying ? (
                                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                                            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                                        </svg>
                                    ) : (
                                        <svg className="w-4 h-4 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                                            <path d="M8 5v14l11-7z" />
                                        </svg>
                                    )}
                                </button>
                            )}
                            <span className="text-xs text-[rgb(var(--color-text-secondary))] truncate flex-1">
                                {audioPath.split(/[\\/]/).pop()}
                            </span>
                        </div>
                        {safeAudioSrc && (
                            <audio
                                ref={audioRef}
                                src={safeAudioSrc}
                                className="hidden"
                            />
                        )}
                    </div>
                )}

                {/* Image Preview */}
                {imagePath && (
                    <div className="bg-[rgb(var(--color-bg-tertiary))] rounded-[var(--r-sm)] overflow-hidden p-1">
                        {safeImageSrc ? (
                            <img
                                src={safeImageSrc}
                                alt="Result"
                                className="w-full h-20 object-contain"
                            />
                        ) : (
                            <span className="block truncate px-1 py-1 text-xs text-[rgb(var(--color-text-secondary))]">
                                {imagePath.split(/[\\/]/).pop()}
                            </span>
                        )}
                    </div>
                )}

                {/* Video Preview */}
                {videoPath && (
                    <div className="bg-[rgb(var(--color-bg-tertiary))] rounded-[var(--r-sm)] overflow-hidden p-1">
                        {safeVideoSrc ? (
                            <video
                                src={safeVideoSrc}
                                controls
                                className="w-full h-20"
                            />
                        ) : (
                            <span className="block truncate px-1 py-1 text-xs text-[rgb(var(--color-text-secondary))]">
                                {videoPath.split(/[\\/]/).pop()}
                            </span>
                        )}
                    </div>
                )}

                {/* Text Result (if not media) */}
                {!audioPath && !imagePath && !videoPath && resultValue !== undefined && (
                    <div className="p-2 bg-[rgb(var(--color-bg-tertiary))] rounded-[var(--r-sm)] text-xs text-[rgb(var(--color-text-primary))] max-h-20 overflow-y-auto font-mono">
                        {formatResult(resultValue)}
                    </div>
                )}

                {/* Empty state */}
                {resultValue === undefined && (
                    <div className="text-xs text-[rgb(var(--color-text-faint))] text-center py-2">
                        Connect a result
                    </div>
                )}
            </div>
        </CollapsibleNodeWrapper>
    );
}, arePropsEqual);

OutputNode.displayName = 'OutputNode';
