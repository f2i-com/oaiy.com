import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface ImageSaveNodeData {
  imageUrl?: string;
  filename?: string;
  format?: 'png' | 'jpg' | 'webp';
  _status?: 'running' | 'completed' | 'error';
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onFilenameChange?: (value: string) => void;
  onFormatChange?: (value: 'png' | 'jpg' | 'webp') => void;
  onCollapsedChange?: (value: boolean) => void;
}

interface ImageSaveNodeProps {
  data: ImageSaveNodeData;
}

const ImageSaveIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

function ImageSaveNode({ data }: ImageSaveNodeProps) {
  const hasImage = data.imageUrl && data.imageUrl !== '';
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const collapsedPreview = (
    <div className="text-slate-400">
      <span className="text-teal-400">{data.format?.toUpperCase() || 'PNG'}</span>
      {data.filename && <span className="ml-1 text-[10px]">{data.filename}</span>}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'image', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg', label: 'image' },
    { id: 'filename', type: 'target', position: Position.Left, color: '!bg-amber-500', size: 'sm', label: 'name' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'path', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  const titleExtra = (
    <span className="ml-auto px-1.5 py-0.5 bg-teal-900 text-teal-400 text-[10px] rounded">
      AUTO
    </span>
  );

  return (
    <CollapsibleNodeWrapper
      title="Save Image"
      color="teal"
      icon={ImageSaveIcon}
      width={288}
      collapsedWidth={130}
      status={data._status}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
      titleExtra={titleExtra}
    >
      {data.showBodyProperties !== false && (
        <>
          {/* Filename */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Filename</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500"
              placeholder="my_image"
              value={data.filename ?? ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => data.onFilenameChange?.(e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          {/* Format Select */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Format</label>
            <select
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500"
              value={data.format || 'png'}
              onChange={(e: ChangeEvent<HTMLSelectElement>) => data.onFormatChange?.(e.target.value as 'png' | 'jpg' | 'webp')}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <option value="png">PNG</option>
              <option value="jpg">JPG</option>
              <option value="webp">WebP</option>
            </select>
          </div>

          {/* Image Preview */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Preview</label>
            <div className={`w-full h-20 bg-white dark:bg-slate-900 border rounded flex items-center justify-center ${hasImage ? 'border-teal-600' : 'border-slate-300 dark:border-slate-700'
              }`}>
              {hasImage ? (
                <img
                  src={
                    data.imageUrl?.startsWith('data:') || data.imageUrl?.startsWith('http')
                      ? data.imageUrl
                      : data.imageUrl?.match(/^[A-Za-z]:[\\/]/)
                        ? `asset://localhost/${encodeURIComponent(data.imageUrl.replace(/\\/g, '/')).replace(/%2F/g, '/').replace(/%3A/g, ':')}`
                        : data.imageUrl
                  }
                  alt="Preview"
                  className="max-w-full max-h-full object-contain rounded"
                />
              ) : (
                <span className="text-slate-500 text-xs italic">No image</span>
              )}
            </div>
          </div>

          {/* Auto-save Info */}
          <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-100/50 dark:bg-slate-100/50 dark:bg-slate-900/50 rounded text-xs text-slate-600 dark:text-slate-400">
            <svg className="w-4 h-4 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Auto-saves during workflow execution</span>
          </div>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(ImageSaveNode);
