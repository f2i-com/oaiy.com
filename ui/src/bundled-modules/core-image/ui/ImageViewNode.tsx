import { memo, useState, useRef, useEffect, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';


interface ImageViewNodeData {
  imageUrl?: string;
  label?: string;
  _collapsed?: boolean;
  onLabelChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  showBodyProperties?: boolean;
}

interface ImageViewNodeProps {
  data: ImageViewNodeData;
}

const ImageViewIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
  </svg>
);

function ImageViewNode({ data }: ImageViewNodeProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [hasLoadError, setHasLoadError] = useState(false);
  const hasImage = data.imageUrl && data.imageUrl !== '';
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  // Re-arm the <img> on every new source URL: a transient load failure on one URL
  // would otherwise latch hasLoadError permanently (the `!hasLoadError` gate
  // unmounts the <img>, so onLoad can never fire to clear it), stranding every
  // subsequent valid image on the "Invalid image format" placeholder.
  useEffect(() => {
    setHasLoadError(false);
  }, [data.imageUrl]);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  // Check if it's a valid image source. Require a real http(s):// scheme (not a
  // bare 'http' prefix, which would also match junk like 'httpfoo') and a
  // `data:image/` MIME (not any `data:` payload), so an arbitrary string from
  // upstream node output can't be dropped straight into an <img src>.
  const imageUrl = data.imageUrl ?? '';
  const isValidImage = hasImage && (
    imageUrl.startsWith('data:image/') ||
    /^https?:\/\//i.test(imageUrl) ||
    imageUrl.startsWith('blob:')
  );

  const collapsedPreview = (
    <div className="text-slate-500 dark:text-slate-400">
      {hasImage ? (
        <span className="text-indigo-400">Image</span>
      ) : (
        <span className="italic text-slate-600 dark:text-slate-400">No image</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'image', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg' },
  ], []);

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'image', type: 'source', position: Position.Right, color: '!bg-green-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Image Viewer"
      color="indigo"
      icon={ImageViewIcon}
      width={isExpanded ? 384 : 288}
      collapsedWidth={140}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      outputHandles={outputHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Label</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-accent"
              placeholder="image_preview"
              value={data.label || ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => data.onLabelChange?.(e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          {/* Image Display */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-slate-600 dark:text-slate-400 text-xs">Preview</label>
              {hasImage && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 focus:outline-none focus:ring-1 focus:ring-indigo-500 rounded text-[10px] flex items-center gap-0.5"
                >
                  {isExpanded ? (
                    <>
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                      </svg>
                      Collapse
                    </>
                  ) : (
                    <>
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                      </svg>
                      Expand
                    </>
                  )}
                </button>
              )}
            </div>
            <div className={`w-full bg-white dark:bg-slate-900 border rounded flex items-center justify-center transition-all ${hasImage ? 'border-indigo-600' : 'border-slate-300 dark:border-slate-700'
              } ${isExpanded ? 'h-64' : 'h-32'}`}>
              {isValidImage && !hasLoadError ? (
                <img
                  src={data.imageUrl}
                  alt={data.label || 'Image'}
                  className="max-w-full max-h-full object-contain rounded"
                  onError={() => setHasLoadError(true)}
                  onLoad={() => setHasLoadError(false)}
                />
              ) : hasImage ? (
                <div className="text-center p-2">
                  <svg className="w-8 h-8 mx-auto text-slate-600 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span className="text-slate-600 dark:text-slate-400 text-xs">Invalid image format</span>
                </div>
              ) : (
                <div className="text-center p-2">
                  <svg className="w-8 h-8 mx-auto text-slate-600 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="text-slate-600 dark:text-slate-400 text-xs italic">Waiting for image...</span>
                </div>
              )}
            </div>
          </div>

          <p className="text-slate-600 dark:text-slate-400 text-[10px]">
            Displays image from input. Connect to Image Generator output.
          </p>
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(ImageViewNode);
