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
        label?: string;
        _updateKey?: number;
    };
    selected?: boolean;
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

// Custom comparison for memo - detect changes in result and _updateKey
const arePropsEqual = (prevProps: OutputNodeProps, nextProps: OutputNodeProps): boolean => {
    // Always re-render if id changes
    if (prevProps.id !== nextProps.id) return false;
    // Re-render if result changes
    if (prevProps.data.result !== nextProps.data.result) return false;
    // Re-render if _updateKey changes (forced update)
    if (prevProps.data._updateKey !== nextProps.data._updateKey) return false;
    // Re-render if label changes
    if (prevProps.data.label !== nextProps.data.label) return false;
    // Otherwise, consider props equal
    return true;
};

export const OutputNode: React.FC<OutputNodeProps> = memo(({ id: _id, data }) => {
    const definition = data.__definition;
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);

    // Get the result value (could be from input connection or stored)
    const resultValue = data.result;
    const label = (data.label as string) || 'Output';

    // Detect media types
    const audioPath = isAudioPath(resultValue);
    const imagePath = isImagePath(resultValue);
    const videoPath = isVideoPath(resultValue);

    // Handle play/pause
    const togglePlay = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
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
    }, [audioPath]);

    // Default definition fallback
    const inputHandles: HandleConfig[] = [{
        id: 'result',
        type: 'target',
        position: Position.Left,
        color: '!bg-gray-400',
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
            inputHandles={inputHandles}
            outputHandles={[]}
        >
            <div className="flex flex-col gap-2">
                {/* Audio Player */}
                {audioPath && (
                    <div className="bg-slate-200/50 dark:bg-slate-800/50 rounded-lg p-2">
                        <div className="flex items-center gap-2 mb-2">
                            <button
                                onClick={togglePlay}
                                className="w-8 h-8 rounded-full bg-emerald-500 hover:bg-emerald-400 flex items-center justify-center transition-colors"
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
                            <span className="text-xs text-slate-600 dark:text-slate-400 truncate flex-1">
                                {audioPath.split(/[\\/]/).pop()}
                            </span>
                        </div>
                        <audio
                            ref={audioRef}
                            src={`file://${audioPath.replace(/\\/g, '/')}`}
                            className="hidden"
                        />
                    </div>
                )}

                {/* Image Preview */}
                {imagePath && (
                    <div className="bg-slate-200/50 dark:bg-slate-800/50 rounded-lg overflow-hidden">
                        <img
                            src={imagePath.startsWith('data:') ? imagePath : `file://${imagePath.replace(/\\/g, '/')}`}
                            alt="Result"
                            className="w-full h-20 object-contain"
                        />
                    </div>
                )}

                {/* Video Preview */}
                {videoPath && (
                    <div className="bg-slate-200/50 dark:bg-slate-800/50 rounded-lg overflow-hidden">
                        <video
                            src={`file://${videoPath.replace(/\\/g, '/')}`}
                            controls
                            className="w-full h-20"
                        />
                    </div>
                )}

                {/* Text Result (if not media) */}
                {!audioPath && !imagePath && !videoPath && resultValue !== undefined && (
                    <div className="p-2 bg-slate-200/50 dark:bg-slate-800/50 rounded text-xs text-slate-600 dark:text-slate-300 max-h-20 overflow-y-auto font-mono">
                        {formatResult(resultValue)}
                    </div>
                )}

                {/* Empty state */}
                {resultValue === undefined && (
                    <div className="text-xs text-slate-500 dark:text-slate-400 text-center py-2">
                        Connect a result
                    </div>
                )}
            </div>
        </CollapsibleNodeWrapper>
    );
}, arePropsEqual);

OutputNode.displayName = 'OutputNode';
