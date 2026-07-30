import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface VideoFrameExtractorNodeData {
  intervalSeconds?: number;
  startTime?: number;
  endTime?: number;
  maxFrames?: number;
  outputFormat?: 'png' | 'jpeg';
  batchSize?: number;  // 0 = extract all at once, >0 = extract in batches
  _videoName?: string;
  _frameCount?: number;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onIntervalSecondsChange?: (value: number) => void;
  onStartTimeChange?: (value: number) => void;
  onEndTimeChange?: (value: number) => void;
  onMaxFramesChange?: (value: number) => void;
  onOutputFormatChange?: (value: string) => void;
  onBatchSizeChange?: (value: number) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface VideoFrameExtractorNodeProps {
  data: VideoFrameExtractorNodeData;
}

const VideoIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

function VideoFrameExtractorNode({ data }: VideoFrameExtractorNodeProps) {
  const onIntervalSecondsChangeRef = useRef(data.onIntervalSecondsChange);
  const onStartTimeChangeRef = useRef(data.onStartTimeChange);
  const onEndTimeChangeRef = useRef(data.onEndTimeChange);
  const onMaxFramesChangeRef = useRef(data.onMaxFramesChange);
  const onOutputFormatChangeRef = useRef(data.onOutputFormatChange);
  const onBatchSizeChangeRef = useRef(data.onBatchSizeChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onIntervalSecondsChangeRef.current = data.onIntervalSecondsChange;
    onStartTimeChangeRef.current = data.onStartTimeChange;
    onEndTimeChangeRef.current = data.onEndTimeChange;
    onMaxFramesChangeRef.current = data.onMaxFramesChange;
    onOutputFormatChangeRef.current = data.onOutputFormatChange;
    onBatchSizeChangeRef.current = data.onBatchSizeChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleIntervalChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value > 0) {
      onIntervalSecondsChangeRef.current?.(value);
    }
  }, []);

  const handleStartTimeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value >= 0) {
      onStartTimeChangeRef.current?.(value);
    }
  }, []);

  const handleEndTimeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value >= 0) {
      onEndTimeChangeRef.current?.(value);
    }
  }, []);

  const handleMaxFramesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && value > 0) {
      onMaxFramesChangeRef.current?.(value);
    }
  }, []);

  const handleFormatChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onOutputFormatChangeRef.current?.(e.target.value);
  }, []);

  const handleBatchSizeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && value >= 0) {
      onBatchSizeChangeRef.current?.(value);
    }
  }, []);

  const intervalSeconds = data.intervalSeconds ?? 1.0;
  const startTime = data.startTime ?? 0;
  const endTime = data.endTime ?? 0;
  const maxFrames = data.maxFrames ?? 100;
  const outputFormat = data.outputFormat ?? 'jpeg';
  const batchSize = data.batchSize ?? 0;

  const collapsedPreview = (
    <div className="text-slate-600 dark:text-slate-400 text-[10px]">
      <span className="text-orange-400">{intervalSeconds}s</span>
      <span className="text-slate-500"> / </span>
      <span className="text-orange-300">{maxFrames} max</span>
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    {
      id: 'video', type: 'target', position: Position.Left,
      color: '!bg-orange-500', label: 'video', labelColor: 'text-orange-400', size: 'md'
    },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    {
      id: 'frames', type: 'source', position: Position.Right,
      color: '!bg-orange-500', label: 'frames', labelColor: 'text-orange-400', size: 'lg'
    },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Video Frames"
      color="orange"
      icon={VideoIcon}
      width={240}
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
          {/* Interval input */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
              Frame Interval (seconds)
            </label>
            <input
              type="number"
              min={0.1}
              step={0.1}
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
              value={intervalSeconds}
              onChange={handleIntervalChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
            <p className="text-slate-500 text-[10px] mt-0.5">Extract 1 frame every N seconds</p>
          </div>

          {/* Time range */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Start (s)</label>
              <input
                type="number"
                min={0}
                step={0.1}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                value={startTime}
                onChange={handleStartTimeChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">End (s)</label>
              <input
                type="number"
                min={0}
                step={0.1}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                value={endTime}
                onChange={handleEndTimeChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <p className="text-slate-500 text-[10px] mt-0.5">0 = full video</p>
            </div>
          </div>

          {/* Max frames and Batch size */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Max Frames</label>
              <input
                type="number"
                min={1}
                max={10000}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                value={maxFrames}
                onChange={handleMaxFramesChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Batch Size</label>
              <input
                type="number"
                min={0}
                max={100}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
                value={batchSize}
                onChange={handleBatchSizeChange}
                onFocus={(e) => e.target.select()}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          </div>
          <p className="text-slate-500 text-[10px]">Batch: 0 = all at once, &gt;0 = process in batches</p>

          {/* Output format */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Format</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-orange-500"
              value={outputFormat}
              onChange={handleFormatChange}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <option value="jpeg">JPEG (smaller)</option>
              <option value="png">PNG (lossless)</option>
            </select>
          </div>

          {/* Info */}
          <div className="text-slate-500 text-[10px] border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
            <div>Input: video file path</div>
            <div>Output: array of frame images</div>
          </div>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(VideoFrameExtractorNode);
