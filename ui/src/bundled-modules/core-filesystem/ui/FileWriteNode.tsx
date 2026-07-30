import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface FileWriteNodeData {
  targetPath?: string;
  filenamePattern?: string;
  contentType?: 'text' | 'base64';
  createDirectories?: boolean;
  _lastWrittenPath?: string;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  onTargetPathChange?: (value: string) => void;
  onFilenamePatternChange?: (value: string) => void;
  onContentTypeChange?: (value: string) => void;
  onCreateDirectoriesChange?: (value: boolean) => void;
  onCollapsedChange?: (value: boolean) => void;
  onBrowseFolder?: () => void;
  showBodyProperties?: boolean;
}

interface FileWriteNodeProps {
  data: FileWriteNodeData;
}

const FileWriteIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

const CONTENT_TYPES = [
  { value: 'base64', label: 'Base64/Binary', description: 'For images and binary data' },
  { value: 'text', label: 'Text', description: 'For text files' },
] as const;

function FileWriteNode({ data }: FileWriteNodeProps) {
  const onTargetPathChangeRef = useRef(data.onTargetPathChange);
  const onFilenamePatternChangeRef = useRef(data.onFilenamePatternChange);
  const onContentTypeChangeRef = useRef(data.onContentTypeChange);
  const onCreateDirectoriesChangeRef = useRef(data.onCreateDirectoriesChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);
  const onBrowseFolderRef = useRef(data.onBrowseFolder);

  useEffect(() => {
    onTargetPathChangeRef.current = data.onTargetPathChange;
    onFilenamePatternChangeRef.current = data.onFilenamePatternChange;
    onContentTypeChangeRef.current = data.onContentTypeChange;
    onCreateDirectoriesChangeRef.current = data.onCreateDirectoriesChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
    onBrowseFolderRef.current = data.onBrowseFolder;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleTargetPathChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onTargetPathChangeRef.current?.(e.target.value);
  }, []);

  const handleFilenamePatternChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onFilenamePatternChangeRef.current?.(e.target.value);
  }, []);

  const handleContentTypeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onContentTypeChangeRef.current?.(e.target.value);
  }, []);

  const handleCreateDirectoriesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onCreateDirectoriesChangeRef.current?.(e.target.checked);
  }, []);

  const handleBrowseFolder = useCallback(() => {
    onBrowseFolderRef.current?.();
  }, []);

  const targetPath = data.targetPath || '';
  const filenamePattern = data.filenamePattern || '';
  const contentType = data.contentType || 'base64';
  const createDirectories = data.createDirectories ?? true;
  const showBodyProperties = data.showBodyProperties !== false;

  const displayName = filenamePattern || targetPath;
  const collapsedPreview = (
    <div className="text-slate-400">
      {displayName ? (
        <span className="text-teal-400 text-[10px] truncate block">{displayName.split(/[/\\]/).pop()}</span>
      ) : (
        <span className="italic text-slate-500 text-[10px]">No path</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'content', type: 'target', position: Position.Left, color: '!bg-teal-500', label: 'content', labelColor: 'text-teal-400', size: 'md' },
    { id: 'folder', type: 'target', position: Position.Left, color: '!bg-green-500', label: 'folder', labelColor: 'text-green-400', size: 'md' },
    { id: 'info', type: 'target', position: Position.Left, color: '!bg-slate-500', label: 'info', labelColor: 'text-slate-400', size: 'sm' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'path', type: 'source', position: Position.Right, color: '!bg-teal-500', label: 'path', labelColor: 'text-teal-400', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="File Write"
      color="teal"
      icon={FileWriteIcon}
      width={280}
      collapsedWidth={130}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {showBodyProperties && (
        <>
          {/* Output Folder (manual entry or via connection) */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Output Folder</label>
            <div className="flex gap-1">
              <input
                type="text"
                className="nodrag nowheel flex-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500 font-mono text-xs"
                placeholder="/output/folder or connect"
                value={targetPath}
                onChange={handleTargetPathChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <button
                onClick={handleBrowseFolder}
                className="nodrag px-2 py-1.5 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded text-slate-700 dark:text-slate-300 transition-colors"
                title="Browse for folder"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                </svg>
              </button>
            </div>
            <p className="text-slate-500 text-[10px] mt-0.5">Connect folder input or enter path manually</p>
          </div>

          {/* Filename Pattern */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Filename Pattern</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500 font-mono text-xs"
              placeholder="{{nameWithoutExt}}_output.txt"
              value={filenamePattern}
              onChange={handleFilenamePatternChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
            <p className="text-slate-500 text-[10px] mt-0.5">
              {'{{name}}'}, {'{{nameWithoutExt}}'}, {'{{ext}}'}, {'{{index}}'}
            </p>
          </div>

          {/* Content Type */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Content Type</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500"
              value={contentType}
              onChange={handleContentTypeChange}
              onMouseDown={(e) => e.stopPropagation()}
            >
              {CONTENT_TYPES.map((type) => (
                <option key={type.value} value={type.value}>{type.label}</option>
              ))}
            </select>
          </div>

          {/* Options */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={createDirectories}
              onChange={handleCreateDirectoriesChange}
              className="nodrag w-4 h-4 rounded bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-teal-500 focus:ring-teal-500 focus:ring-offset-0"
              onMouseDown={(e) => e.stopPropagation()}
            />
            <span className="text-slate-600 dark:text-slate-400 text-xs">Create directories if missing</span>
          </label>
        </>
      )}

      {/* Last Written */}
      {data._lastWrittenPath && (
        <div className="flex items-center gap-2 px-2 py-1.5 bg-teal-900/20 border border-teal-600/30 rounded text-teal-400 text-xs">
          <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          <span className="truncate">{data._lastWrittenPath}</span>
        </div>
      )}

      {/* Info */}
      <div className="text-slate-500 text-[10px] border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
        <div>Input: content + folder (optional) + file info for templating</div>
        <div>Output: written file path</div>
      </div>
    </CollapsibleNodeWrapper>
  );
}

export default memo(FileWriteNode);
