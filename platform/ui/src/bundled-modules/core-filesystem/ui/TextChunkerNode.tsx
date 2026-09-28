import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface TextChunkerNodeData {
  contentType?: 'raw' | 'json_array' | 'csv' | 'lines';
  chunkSize?: number;
  overlap?: number;
  csvHasHeader?: boolean;
  indexName?: string;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  _fileName?: string;  // File name from connected input (for auto-detection)
  _sourceFormat?: string;  // Format from File Read node (json, csv, lines, text)
  onContentTypeChange?: (value: string) => void;
  onChunkSizeChange?: (value: number) => void;
  onOverlapChange?: (value: number) => void;
  onCsvHasHeaderChange?: (value: boolean) => void;
  onIndexNameChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface TextChunkerNodeProps {
  data: TextChunkerNodeData;
}

const TextChunkerIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
  </svg>
);

const CONTENT_TYPES = [
  { value: 'raw', label: 'Raw Text', description: 'Character-based chunking with overlap' },
  { value: 'json_array', label: 'JSON Array', description: 'Batch array items together' },
  { value: 'csv', label: 'CSV Rows', description: 'Batch CSV rows together' },
  { value: 'lines', label: 'Lines', description: 'Batch lines together' },
] as const;

// Map file extensions to content types
const EXTENSION_MAP: Record<string, typeof CONTENT_TYPES[number]['value']> = {
  '.json': 'json_array',
  '.csv': 'csv',
  '.tsv': 'csv',
  '.txt': 'raw',
  '.md': 'raw',
  '.log': 'lines',
};

// Map File Read formats to content types
const FORMAT_MAP: Record<string, typeof CONTENT_TYPES[number]['value']> = {
  'json': 'json_array',
  'csv': 'csv',
  'lines': 'lines',
  'text': 'raw',
};

function TextChunkerNode({ data }: TextChunkerNodeProps) {
  const onContentTypeChangeRef = useRef(data.onContentTypeChange);
  const onChunkSizeChangeRef = useRef(data.onChunkSizeChange);
  const onOverlapChangeRef = useRef(data.onOverlapChange);
  const onCsvHasHeaderChangeRef = useRef(data.onCsvHasHeaderChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onContentTypeChangeRef.current = data.onContentTypeChange;
    onChunkSizeChangeRef.current = data.onChunkSizeChange;
    onOverlapChangeRef.current = data.onOverlapChange;
    onCsvHasHeaderChangeRef.current = data.onCsvHasHeaderChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleContentTypeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onContentTypeChangeRef.current?.(e.target.value);
  }, []);

  const handleChunkSizeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && value > 0) {
      onChunkSizeChangeRef.current?.(value);
    }
  }, []);

  const handleOverlapChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && value >= 0) {
      onOverlapChangeRef.current?.(value);
    }
  }, []);

  const handleCsvHasHeaderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onCsvHasHeaderChangeRef.current?.(e.target.checked);
  }, []);

  const chunkSize = data.chunkSize || 1000;
  const overlap = data.overlap ?? 100;
  const csvHasHeader = data.csvHasHeader !== false;

  // Auto-detect content type from file extension or source format
  const getDetectedType = (): typeof CONTENT_TYPES[number]['value'] | null => {
    // Priority 1: File Read's format setting
    if (data._sourceFormat && FORMAT_MAP[data._sourceFormat]) {
      return FORMAT_MAP[data._sourceFormat];
    }
    // Priority 2: File extension from filename
    if (data._fileName) {
      const ext = data._fileName.toLowerCase().match(/\.[^.]+$/)?.[0];
      if (ext && EXTENSION_MAP[ext]) {
        return EXTENSION_MAP[ext];
      }
    }
    return null;
  };

  const detectedType = getDetectedType();
  const isAutoDetected = !data.contentType && detectedType !== null;
  const contentType = data.contentType || detectedType || 'raw';
  const currentMode = CONTENT_TYPES.find(m => m.value === contentType) || CONTENT_TYPES[0];

  // Determine what chunk size means for the current mode
  const isRawTextMode = contentType === 'raw';
  const isJsonMode = contentType === 'json_array';
  const isCsvMode = contentType === 'csv';
  const isLinesMode = contentType === 'lines';

  // Show chunk size for all modes - different meaning per mode
  const showChunkSize = true;
  // Overlap only applies to raw text
  const showOverlap = isRawTextMode;
  // CSV header option
  const showCsvHeader = isCsvMode;

  // Get the appropriate label for chunk size
  const getChunkSizeLabel = () => {
    if (isRawTextMode) return 'Chunk Size (chars)';
    if (isJsonMode) return 'Items per Batch';
    if (isCsvMode) return 'Rows per Batch';
    if (isLinesMode) return 'Lines per Batch';
    return 'Chunk Size';
  };

  // Get the appropriate help text for chunk size
  const getChunkSizeHelp = () => {
    if (isRawTextMode) return 'Characters per chunk';
    if (isJsonMode) return '1 = each item separate, >1 = batch items';
    if (isCsvMode) return '1 = each row separate, >1 = batch rows';
    if (isLinesMode) return '1 = each line separate, >1 = batch lines';
    return 'Items per chunk';
  };

  const collapsedPreview = (
    <div className="text-slate-600 dark:text-slate-400 text-[10px]">
      <span className="text-amber-400">{currentMode.label}</span>
      {isAutoDetected && <span className="text-emerald-400 ml-1">●</span>}
      {showChunkSize && (
        <>
          <span className="text-slate-500"> · </span>
          <span className="text-amber-300">{chunkSize}</span>
        </>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'text', type: 'target', position: Position.Left, color: '!bg-amber-500', label: 'input', labelColor: 'text-amber-400', size: 'md' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'chunks', type: 'source', position: Position.Right, color: '!bg-amber-500', label: 'chunks', labelColor: 'text-amber-400', size: 'lg' },
  ], []);

  const showBodyProperties = data.showBodyProperties !== false;

  return (
    <CollapsibleNodeWrapper
      title="Text Chunker"
      color="amber"
      icon={TextChunkerIcon}
      width={260}
      collapsedWidth={140}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {showBodyProperties && (
        <>
          {/* Content Type */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-slate-600 dark:text-slate-400 text-xs">Content Type</label>
              {isAutoDetected && (
                <span className="text-emerald-400 text-[10px] flex items-center gap-1">
                  <span>●</span> Auto
                </span>
              )}
            </div>
            <select
              className={`nodrag nowheel w-full bg-white dark:bg-slate-900 border rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none ${isAutoDetected ? 'border-emerald-600/50 focus:border-emerald-500' : 'border-slate-300 dark:border-slate-600 focus:border-amber-500'
                }`}
              value={contentType}
              onChange={handleContentTypeChange}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {CONTENT_TYPES.map((mode) => (
                <option key={mode.value} value={mode.value}>{mode.label}</option>
              ))}
            </select>
            <p className="text-slate-500 text-[10px] mt-0.5">
              {isAutoDetected
                ? `Detected from ${data._sourceFormat ? 'File Read' : 'file extension'}`
                : currentMode.description
              }
            </p>
          </div>

          {/* Chunk Size / Batch Size */}
          {showChunkSize && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">
                {getChunkSizeLabel()}
              </label>
              <input
                type="number"
                min={1}
                step={isRawTextMode ? 100 : 1}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500"
                value={chunkSize}
                onChange={handleChunkSizeChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <p className="text-slate-500 text-[10px] mt-0.5">
                {getChunkSizeHelp()}
              </p>
            </div>
          )}

          {/* Overlap - only for raw text */}
          {showOverlap && (
            <div>
              <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Overlap (chars)</label>
              <input
                type="number"
                min={0}
                step={50}
                className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-amber-500"
                value={overlap}
                onChange={handleOverlapChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <p className="text-slate-500 text-[10px] mt-0.5">Overlap between chunks</p>
            </div>
          )}

          {/* CSV Has Header */}
          {showCsvHeader && (
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                className="nodrag w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-amber-500 focus:ring-amber-500 focus:ring-offset-0"
                checked={csvHasHeader}
                onChange={handleCsvHasHeaderChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <span className="text-slate-600 dark:text-slate-400 text-xs">CSV has header row</span>
            </div>
          )}

          {/* Mode-specific info */}
          <div className="text-slate-500 text-[10px] border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
            {isJsonMode && <div>Output: {chunkSize === 1 ? '{item, index}' : '{items[], index, count}'}</div>}
            {isCsvMode && <div>Output: {chunkSize === 1 ? '{data, row, index}' : '{rows[], index, count}'}</div>}
            {isLinesMode && <div>Output: {chunkSize === 1 ? '{text, index}' : '{lines[], index, count}'}</div>}
            {isRawTextMode && <div>Output: {'{'}text, start, end, index{'}'}</div>}
            <div className="text-slate-600 mt-1">
              Auto-detects type from .json, .csv, .txt extensions
            </div>
          </div>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(TextChunkerNode);
