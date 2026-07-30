import { memo, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface InputFolderNodeData {
  path?: string;
  recursive?: boolean;
  includePatterns?: string;
  excludePatterns?: string;
  maxFiles?: number;
  _fileCount?: number;
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onPathChange?: (value: string) => void;
  onRecursiveChange?: (value: boolean) => void;
  onIncludePatternsChange?: (value: string) => void;
  onExcludePatternsChange?: (value: string) => void;
  onMaxFilesChange?: (value: number) => void;
  onCollapsedChange?: (value: boolean) => void;
  onBrowse?: () => void;
}

interface InputFolderNodeProps {
  data: InputFolderNodeData;
}

const FolderIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
  </svg>
);

function InputFolderNode({ data }: InputFolderNodeProps) {
  const onPathChangeRef = useRef(data.onPathChange);
  const onRecursiveChangeRef = useRef(data.onRecursiveChange);
  const onIncludePatternsChangeRef = useRef(data.onIncludePatternsChange);
  const onMaxFilesChangeRef = useRef(data.onMaxFilesChange);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);
  const onBrowseRef = useRef(data.onBrowse);

  useEffect(() => {
    onPathChangeRef.current = data.onPathChange;
    onRecursiveChangeRef.current = data.onRecursiveChange;
    onIncludePatternsChangeRef.current = data.onIncludePatternsChange;
    onMaxFilesChangeRef.current = data.onMaxFilesChange;
    onCollapsedChangeRef.current = data.onCollapsedChange;
    onBrowseRef.current = data.onBrowse;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handlePathChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onPathChangeRef.current?.(e.target.value);
  }, []);

  const handleRecursiveChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onRecursiveChangeRef.current?.(e.target.checked);
  }, []);

  const handleIncludePatternsChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onIncludePatternsChangeRef.current?.(e.target.value);
  }, []);

  const handleMaxFilesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value) || 100;
    onMaxFilesChangeRef.current?.(Math.max(1, Math.min(10000, val)));
  }, []);

  const handleBrowse = useCallback(() => {
    onBrowseRef.current?.();
  }, []);

  const path = data.path || '';
  const recursive = data.recursive ?? false;
  const includePatterns = data.includePatterns || '*.png, *.jpg, *.jpeg';
  const maxFiles = data.maxFiles || 100;
  const fileCount = data._fileCount;

  const collapsedPreview = (
    <div className="text-slate-400">
      {path ? (
        <div className="truncate">
          <span className="text-green-400 text-[10px]">{path.split(/[/\\]/).pop()}</span>
          {fileCount !== undefined && (
            <span className="text-slate-500 text-[10px] ml-1">({fileCount})</span>
          )}
        </div>
      ) : (
        <span className="italic text-slate-500 text-[10px]">No folder</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'path', type: 'target', position: Position.Left, color: '!bg-blue-500', label: 'path', labelColor: 'text-blue-400', size: 'md' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'files', type: 'source', position: Position.Right, color: '!bg-green-500', label: 'files', labelColor: 'text-green-400', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Folder Input"
      color="green"
      icon={FolderIcon}
      width={280}
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
          {/* Big clickable region — same affordance as Audio/Video/File
              input nodes. Click anywhere on the dotted box to open the
              folder picker (Tauri native on desktop; HTML
              webkitdirectory input on the web). Folder path appears below
              when set; the explicit text field remains for users who
              prefer to paste a path. */}
          <button
            type="button"
            onClick={handleBrowse}
            onMouseDown={(e) => e.stopPropagation()}
            className={`
              nodrag w-full h-24 border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all
              ${path
                ? 'border-green-600 bg-slate-100 dark:bg-slate-900'
                : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-900 hover:border-green-600'
              }
            `}
            title="Pick a folder"
          >
            {path ? (
              <div className="text-center px-2">
                <svg className="w-6 h-6 mx-auto text-green-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <p className="text-green-400 text-xs truncate max-w-full">
                  {path.split(/[/\\]/).pop() || path}
                </p>
                {fileCount !== undefined && (
                  <p className="text-slate-500 text-[10px] mt-0.5">{fileCount} file{fileCount !== 1 ? 's' : ''}</p>
                )}
              </div>
            ) : (
              <>
                <svg className="w-6 h-6 text-slate-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <p className="text-slate-600 dark:text-slate-400 text-xs">Click to pick a folder</p>
              </>
            )}
          </button>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Folder Path</label>
            <div className="flex gap-1">
              <input
                type="text"
                className="nodrag nowheel flex-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-green-500 font-mono text-xs"
                placeholder="/path/to/folder"
                value={path}
                onChange={handlePathChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
              <button
                onClick={handleBrowse}
                className="nodrag px-2 py-1.5 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded text-slate-700 dark:text-slate-300 transition-colors"
                title="Browse for folder"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
                </svg>
              </button>
            </div>
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Include Patterns</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-green-500 font-mono text-xs"
              placeholder="*.png, *.jpg"
              value={includePatterns}
              onChange={handleIncludePatternsChange}
              onMouseDown={(e) => e.stopPropagation()}
            />
            <p className="text-slate-500 text-[10px] mt-0.5">Comma-separated glob patterns</p>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={recursive}
                onChange={handleRecursiveChange}
                className="nodrag w-4 h-4 rounded bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-green-500 focus:ring-green-500 focus:ring-offset-0"
                onMouseDown={(e) => e.stopPropagation()}
              />
              <span className="text-slate-600 dark:text-slate-400 text-xs">Recursive</span>
            </label>

            <div className="flex items-center gap-2">
              <label className="text-slate-600 dark:text-slate-400 text-xs">Max:</label>
              <input
                type="number"
                min={1}
                max={10000}
                className="nodrag nowheel w-16 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-green-500 font-mono"
                value={maxFiles}
                onChange={handleMaxFilesChange}
                onMouseDown={(e) => e.stopPropagation()}
              />
            </div>
          </div>

          {fileCount !== undefined && (
            <div className="flex items-center gap-2 px-2 py-1.5 bg-green-900/20 border border-green-600/30 rounded text-green-400 text-xs">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>{fileCount} files found</span>
            </div>
          )}

          <div className="text-slate-500 text-[10px] border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
            Outputs array of file objects with path, name, ext, size
          </div>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(InputFolderNode);
