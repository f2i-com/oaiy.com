import { memo, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';
import { BLOCKED_MEDIA_WORDS, mediaUrlBlocked } from '../../../lib/localMedia';


interface ImageSaveNodeData {
  imageUrl?: string;
  /** Where it saves: what the run reads (compiler.ts) and Properties' File Path. */
  path?: string;
  /** An older card's Filename, which the run never read. */
  filename?: string;
  format?: 'png' | 'jpg' | 'webp';
  _status?: 'running' | 'completed' | 'error' | 'stopped';
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onFilenameChange?: (value: string) => void;
  onFormatChange?: (value: 'png' | 'jpg' | 'webp') => void;
  onCollapsedChange?: (value: boolean) => void;
  onChange?: (field: string, value: unknown) => void;
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

  // Where it saves (`path`, what the run reads: the card's Filename was never read, and its "name" input went nowhere).
  const path = data.path ?? '';
  const collapsedPreview = (
    <span className="block truncate" title={path || undefined}>
      {data.format?.toUpperCase() || 'PNG'} · {path ? path.split(/[/\\]/).pop() : 'to your Downloads folder'}
    </span>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'image', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'path', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Image Save"
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
    >
      {data.showBodyProperties !== false && (
        <>
          {/* Where */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">File Path</label>
            <input
              type="text"
              aria-label="File Path"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-500"
              placeholder="Empty: your Downloads folder"
              value={path}
              onChange={(e: ChangeEvent<HTMLInputElement>) => data.onChange?.('path', e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
            />
            <p className="text-slate-500 dark:text-slate-400 text-[10px] mt-0.5 mb-0">A whole path, like /Users/you/Pictures/lake.png (the format's ending is added if it has none)</p>
          </div>

          {/* Format Select */}
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Format</label>
            <select
              aria-label="Format"
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
              {hasImage && mediaUrlBlocked(data.imageUrl) ? (
                <span className="text-amber-600 dark:text-amber-400 text-[10px] p-2" data-blocked-media>{BLOCKED_MEDIA_WORDS}</span>
              ) : hasImage ? (
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
                <span className="text-slate-500 dark:text-slate-400 text-xs italic">Its picture shows here after a run</span>
              )}
            </div>
          </div>

          <p className="text-slate-500 dark:text-slate-400 text-[10px] leading-snug m-0">Saves each picture that reaches it as the flow runs, and gives on where it went.</p>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(ImageSaveNode);
