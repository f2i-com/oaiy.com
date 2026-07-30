import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position, useReactFlow, useNodeId } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface VideoPipNodeData {
    mainVideo?: string;
    pipVideo?: string;
    position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
    size?: number;
    margin?: number;
    shape?: 'rectangle' | 'rounded' | 'circle';
    startTime?: number;
    pipDuration?: number;
    mainVolume?: number;
    pipVolume?: number;
    outputValue?: string;
    _status?: 'running' | 'completed' | 'error';
    _collapsed?: boolean;
    showBodyProperties?: boolean;
    onCollapsedChange?: (value: boolean) => void;
}

interface VideoPipNodeProps {
    data: VideoPipNodeData;
}

const VideoPipIcon = (
    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z" />
        <rect x="12" y="10" width="6" height="5" rx="0.5" strokeWidth={2} />
    </svg>
);

// Visual preview component showing PiP position
function PipPreview({ position, size, shape }: { position: string; size: number; shape: string }) {
    const pipSize = Math.max(20, (size / 100) * 80);

    const getPosition = () => {
        switch (position) {
            case 'top-left': return { top: 4, left: 4 };
            case 'top-right': return { top: 4, right: 4 };
            case 'bottom-left': return { bottom: 4, left: 4 };
            case 'bottom-right': return { bottom: 4, right: 4 };
            default: return { bottom: 4, right: 4 };
        }
    };

    const getBorderRadius = () => {
        switch (shape) {
            case 'circle': return '50%';
            case 'rounded': return '4px';
            default: return '2px';
        }
    };

    return (
        <div className="relative w-full h-16 bg-slate-200 dark:bg-slate-800 rounded border border-slate-300 dark:border-slate-600 overflow-hidden">
            {/* Main video placeholder */}
            <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[8px] text-slate-500">Main Video</span>
            </div>
            {/* PiP overlay */}
            <div
                className="absolute bg-cyan-500/80 border border-cyan-400 flex items-center justify-center"
                style={{
                    width: pipSize,
                    height: pipSize * 0.75,
                    borderRadius: getBorderRadius(),
                    ...getPosition(),
                }}
            >
                <span className="text-[6px] text-white font-bold">PiP</span>
            </div>
        </div>
    );
}

function VideoPipNode({ data }: VideoPipNodeProps) {
    const nodeId = useNodeId();
    const { updateNodeData } = useReactFlow();
    const onCollapsedChangeRef = useRef(data.onCollapsedChange);

    useEffect(() => {
        onCollapsedChangeRef.current = data.onCollapsedChange;
    });

    const handleCollapsedChange = useCallback((collapsed: boolean) => {
        onCollapsedChangeRef.current?.(collapsed);
    }, []);

    const handlePositionChange = useCallback((value: string) => {
        if (nodeId) updateNodeData(nodeId, { position: value });
    }, [nodeId, updateNodeData]);

    const handleSizeChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { size: value });
    }, [nodeId, updateNodeData]);

    const handleMarginChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { margin: value });
    }, [nodeId, updateNodeData]);

    const handleShapeChange = useCallback((value: string) => {
        if (nodeId) updateNodeData(nodeId, { shape: value });
    }, [nodeId, updateNodeData]);

    const handleMainVolumeChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { mainVolume: value });
    }, [nodeId, updateNodeData]);

    const handlePipVolumeChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { pipVolume: value });
    }, [nodeId, updateNodeData]);

    const handleStartTimeChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { startTime: value });
    }, [nodeId, updateNodeData]);

    const handlePipDurationChange = useCallback((value: number) => {
        if (nodeId) updateNodeData(nodeId, { pipDuration: value });
    }, [nodeId, updateNodeData]);

    const position = data.position || 'bottom-right';
    const size = data.size ?? 25;
    const margin = data.margin ?? 20;
    const shape = data.shape || 'rectangle';
    const startTime = data.startTime ?? 0;
    const pipDuration = data.pipDuration ?? 0;
    const mainVolume = data.mainVolume ?? 1.0;
    const pipVolume = data.pipVolume ?? 1.0;

    // Calculate end time for display
    const endTime = pipDuration > 0 ? startTime + pipDuration : null;

    // Format time as MM:SS
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const collapsedPreview = (
        <div className="text-slate-600 dark:text-slate-400 text-[10px]">
            <span className="text-cyan-400">PiP</span>
            <span className="ml-1">{position}</span>
        </div>
    );

    const inputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'mainVideo', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg', label: 'main' },
        { id: 'pipVideo', type: 'target', position: Position.Left, color: '!bg-cyan-500', size: 'lg', label: 'pip' },
    ], []);

    const outputHandles = useMemo<HandleConfig[]>(() => [
        { id: 'video', type: 'source', position: Position.Right, color: '!bg-blue-500', size: 'lg', label: 'video' },
    ], []);

    return (
        <CollapsibleNodeWrapper
            title="Video PiP"
            color="cyan"
            icon={VideoPipIcon}
            width={300}
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
                    {/* Visual Preview */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Preview</label>
                        <PipPreview position={position} size={size} shape={shape} />
                    </div>

                    {/* Position Select */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Position</label>
                        <select
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
                            value={position}
                            onChange={(e: ChangeEvent<HTMLSelectElement>) => handlePositionChange(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="top-left">Top Left</option>
                            <option value="top-right">Top Right</option>
                            <option value="bottom-left">Bottom Left</option>
                            <option value="bottom-right">Bottom Right</option>
                        </select>
                    </div>

                    {/* Size Slider */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Size</label>
                            <span className="text-xs text-cyan-400">{size}%</span>
                        </div>
                        <input
                            type="range"
                            min="10"
                            max="50"
                            step="5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                            value={size}
                            onChange={(e) => handleSizeChange(parseInt(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Margin Slider */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Margin</label>
                            <span className="text-xs text-cyan-400">{margin}px</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            step="5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                            value={margin}
                            onChange={(e) => handleMarginChange(parseInt(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Shape Select */}
                    <div>
                        <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Shape</label>
                        <select
                            className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
                            value={shape}
                            onChange={(e: ChangeEvent<HTMLSelectElement>) => handleShapeChange(e.target.value)}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            <option value="rectangle">Rectangle</option>
                            <option value="rounded">Rounded</option>
                            <option value="circle">Circle</option>
                        </select>
                    </div>

                    {/* Audio Section Header */}
                    <div className="border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
                        <span className="text-slate-500 text-[10px] uppercase tracking-wide">Audio Mix</span>
                    </div>

                    {/* Main Video Volume */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Main Volume</label>
                            <span className="text-xs text-cyan-400">{(mainVolume * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            value={mainVolume}
                            onChange={(e) => handleMainVolumeChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* PiP Video Volume */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">PiP Volume</label>
                            <span className="text-xs text-cyan-400">{(pipVolume * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                            value={pipVolume}
                            onChange={(e) => handlePipVolumeChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Timing Section Header */}
                    <div className="border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
                        <span className="text-slate-500 text-[10px] uppercase tracking-wide">Timing</span>
                    </div>

                    {/* Start Time */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Start Time</label>
                            <span className="text-xs text-cyan-400">{formatTime(startTime)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="300"
                            step="0.5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                            value={startTime}
                            onChange={(e) => handleStartTimeChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* PiP Duration */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <label className="text-slate-600 dark:text-slate-400 text-xs">Duration</label>
                            <span className="text-xs text-cyan-400">{pipDuration === 0 ? 'Auto' : formatTime(pipDuration)}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="300"
                            step="0.5"
                            className="nodrag nowheel w-full h-2 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                            value={pipDuration}
                            onChange={(e) => handlePipDurationChange(parseFloat(e.target.value))}
                            onMouseDown={(e) => e.stopPropagation()}
                            onPointerDown={(e) => e.stopPropagation()}
                            onTouchStart={(e) => e.stopPropagation()}
                        />
                    </div>

                    {/* Calculated End Time Display */}
                    <div className="flex items-center justify-between px-2 py-1.5 bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded text-xs">
                        <span className="text-slate-400">PiP appears:</span>
                        <span className="text-cyan-400 font-mono">
                            {formatTime(startTime)} → {endTime !== null ? formatTime(endTime) : 'end of PiP'}
                        </span>
                    </div>

                    {/* Info */}
                    <div className="flex items-center gap-2 px-2 py-1.5 bg-cyan-900/20 border border-cyan-800/30 rounded text-xs text-cyan-300">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Overlay PiP video on main video</span>
                    </div>
                </>
            )}
        </CollapsibleNodeWrapper>
    );
}

export default memo(VideoPipNode);
