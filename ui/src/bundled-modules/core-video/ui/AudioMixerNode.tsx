import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position, useReactFlow, useNodeId } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface AudioMixerNodeData {
    video?: string;
    audio?: string;
    outputPath?: string;
    videoVolume?: number;
    audioVolume?: number;
    replaceAudio?: boolean;
    filename?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onVideoVolumeChange?: (value: number) => void;
    onAudioVolumeChange?: (value: number) => void;
    onReplaceAudioChange?: (value: boolean) => void;
    onFilenameChange?: (value: string) => void;
    onCollapsedChange?: (value: boolean) => void;
}

interface AudioMixerNodeProps {
    data: AudioMixerNodeData;
}

const AudioMixerIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
    </svg>
);

function AudioMixerNode({ data }: AudioMixerNodeProps) {
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
    const handleVideoVolumeChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { videoVolume: value });
    }, [nodeId, updateNodeData]);

    const handleAudioVolumeChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { audioVolume: value });
    }, [nodeId, updateNodeData]);

    const handleReplaceAudioChange = useCallback((value: boolean) => {
        if (nodeId) updateNodeData(nodeId, { replaceAudio: value });
    }, [nodeId, updateNodeData]);

    const handleFilenameChange = useCallback((value: string) => {
        if (nodeId) updateNodeData(nodeId, { filename: value });
    }, [nodeId, updateNodeData]);

    const videoVol = data.videoVolume ?? 1.0;
    const audioVol = data.audioVolume ?? 1.0;

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-teal-400">V:{(videoVol * 100).toFixed(0)}%</span>
            <span className="ml-1">A:{(audioVol * 100).toFixed(0)}%</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg', label: 'video' },
        { id: 'audio', type: 'target', position: Position.Left, color: '!bg-teal-500', size: 'lg', label: 'audio' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'source', position: Position.Right, color: '!bg-blue-500', size: 'lg', label: 'video' },
    ], []);

    return (
        <CollapsibleNodeWrapper
            title="Audio Mixer"
            color="teal"
            icon={AudioMixerIcon}
            width={300}
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
                    {/* Video Volume */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Video Volume</label>
                            <span className="text-xs text-teal-400">{(videoVol * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                            value={videoVol}
                            onChange={(e) => handleVideoVolumeChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Audio Volume */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Audio Volume</label>
                            <span className="text-xs text-teal-400">{(audioVol * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                            value={audioVol}
                            onChange={(e) => handleAudioVolumeChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Replace Audio Toggle */}
                    <div className="flex items-center justify-between">
                        <label className="text-slate-600 dark:text-slate-400 text-xs">Replace Original Audio</label>
                        <button
                            type="button"
                            role="switch"
                            aria-checked={!!data.replaceAudio}
                            aria-label="Replace Original Audio"
                            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${data.replaceAudio ? 'bg-teal-600' : 'bg-slate-600'
                                }`}
                            onClick={() => handleReplaceAudioChange(!data.replaceAudio)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <span
                                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${data.replaceAudio ? 'translate-x-4' : 'translate-x-1'
                                    }`}
                            />
                        </button>
                    </div>

                    {/* Filename */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Output Filename</label>
                        <input
                            type="text"
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500"
                            placeholder="mixed_video"
                            value={data.filename ?? ''}
                            onChange={(e) => handleFilenameChange(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-teal-900/20 border border-teal-800/30 rounded text-xs text-teal-300">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>{data.replaceAudio ? 'Audio will replace video sound' : 'Audio will mix with video sound'}</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(AudioMixerNode);
