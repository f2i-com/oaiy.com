import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface FileReadNodeData {
  readAs?: 'text' | 'base64' | 'json' | 'csv' | 'lines';
  csvHasHeader?: boolean;
  _fileName?: string;  // Current file name for preview
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onReadAsChange?: (value: string) => void;
  onCsvHasHeaderChange?: (value: boolean) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface FileReadNodeProps {
  data: FileReadNodeData;
}

const FileReadIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const READ_MODES = [
  { value: 'text', label: 'Text', description: 'Read as UTF-8 text' },
  { value: 'base64', label: 'Base64', description: 'Read as base64 data URL' },
  { value: 'json', label: 'JSON', description: 'Parse as JSON object/array' },
  { value: 'csv', label: 'CSV', description: 'Parse as CSV rows' },
  { value: 'lines', label: 'Lines', description: 'Split into lines array' },
] as const;

function FileReadNode({ data }: FileReadNodeProps) {
  const onReadAsChangeRef = useRef(data.onReadAsChange);
  const onCsvHasHeaderChangeRef = useRef(data.onCsvHasHeaderChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onReadAsChangeRef.current = data.onReadAsChange;
    onCsvHasHeaderChangeRef.current = data.onCsvHasHeaderChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleReadAsChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onReadAsChangeRef.current?.(e.target.value);
  }, []);

  const handleCsvHasHeaderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onCsvHasHeaderChangeRef.current?.(e.target.checked);
  }, []);

  const readAs = data.readAs || 'text';
  const csvHasHeader = data.csvHasHeader !== false;
  const showBodyProperties = data.showBodyProperties !== false;
  const currentMode = READ_MODES.find(m => m.value === readAs) || READ_MODES[0];

  // Color coding for different modes
  const getModeColor = () => {
    switch (readAs) {
      case 'json': return 'text-amber-400';
      case 'csv': return 'text-purple-400';
      case 'lines': return 'text-cyan-400';
      case 'base64': return 'text-emerald-400';
      default: return 'text-blue-400';
    }
  };

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className={`text-[10px] ${getModeColor()}`}>
        {currentMode.label}
      </span>
      {data._fileName && (
        <div className="text-slate-500 text-[10px] truncate">{data._fileName}</div>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'path', type: 'target', position: Position.Left, color: '!bg-emerald-500', label: 'path', labelColor: 'text-emerald-400', size: 'md' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'content', type: 'source', position: Position.Right, color: '!bg-emerald-500', label: 'content', labelColor: 'text-emerald-400', size: 'lg' },
    { id: 'info', type: 'source', position: Position.Right, color: '!bg-slate-500', label: 'info', labelColor: 'text-slate-400', size: 'sm' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="File Read"
      color="emerald"
      icon={FileReadIcon}
      width={220}
      collapsedWidth={120}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {showBodyProperties && (
        <>
          {/* Read Mode */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Read As</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
              value={readAs}
              onChange={handleReadAsChange}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {READ_MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>{mode.label}</option>
              ))}
            </select>
            <p className="text-slate-500 text-[10px] mt-0.5">{currentMode.description}</p>
          </div>

          {/* CSV Header Option - only show when CSV mode is selected */}
          {readAs === 'csv' && (
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                className="nodrag w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-purple-500 focus:ring-purple-500 focus:ring-offset-0"
                checked={csvHasHeader}
                onChange={handleCsvHasHeaderChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <span className="text-slate-600 dark:text-slate-400 text-xs">First row is header</span>
            </div>
          )}
        </>
      )}

      {/* Current File Preview */}
      {data._fileName && (
        <div className="flex items-center gap-2 px-2 py-1.5 bg-emerald-900/20 border border-emerald-600/30 rounded text-emerald-400 text-xs truncate">
          <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="truncate">{data._fileName}</span>
        </div>
      )}

      {/* Info */}
      <div className="text-slate-500 text-[10px] border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
        <div>Input: file path (string or FileInfo)</div>
        <div>Output: {readAs === 'json' ? 'parsed object/array' : readAs === 'csv' ? 'array of row objects' : readAs === 'lines' ? 'array of strings' : 'file content'}</div>
        <div className="text-slate-600 mt-1">Large files (&gt;10MB) stream via Text Chunker</div>
      </div>
    </CollapsibleNodeWrapper>
  );
}

export default memo(FileReadNode);
