import { memo, useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface InputVideoNodeData {
  fileName?: string;
  filePath?: string;
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onVideoLoad?: (filePath: string, fileName: string) => void;
  onCollapsedChange?: (value: boolean) => void;
}

interface InputVideoNodeProps {
  data: InputVideoNodeData;
}

const MAX_VIDEO_SIZE = 10 * 1024 * 1024 * 1024; // 10GB

const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv'];

const VideoIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

function InputVideoNode({ data }: InputVideoNodeProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const isVideoFile = (file: File): boolean => {
    const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
    return file.type.startsWith('video/') || VIDEO_EXTENSIONS.includes(fileExt);
  };

  const handleFile = async (file: File) => {
    setError(null);

    if (!isVideoFile(file)) {
      setError('Please select a video file');
      return;
    }

    if (file.size > MAX_VIDEO_SIZE) {
      setError('Video too large (max 10GB)');
      return;
    }

    if (file.size === 0) {
      setError('File is empty');
      return;
    }

    // @ts-expect-error - path exists on File in Tauri context
    const filePath = file.path as string | undefined;
    const sanitizedName = file.name.replace(/[/\\]/g, '_');

    if (filePath) {
      data.onVideoLoad?.(filePath, sanitizedName);
      return;
    }

    // Browser fallback: dropped Files have no `.path`. Stash bytes as a
    // `blob:` URL — downstream paths flow through the same key, and the
    // web `read_file` shim fetches blob URLs back into raw bytes.
    //
    // The original filename is appended as a URL fragment so the
    // input_file runtime's extension-sniff classifies correctly (a bare
    // blob: URL has no extension → defaults to 'text' → binary video
    // gets UTF-8 corrupted). Picker + audio drop path do the same.
    try {
      const baseUrl = URL.createObjectURL(file);
      const safeName = encodeURIComponent(file.name || '');
      const blobUrl = safeName ? `${baseUrl}#${safeName}` : baseUrl;
      data.onVideoLoad?.(blobUrl, sanitizedName);
    } catch {
      setError('Could not read dropped video');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleClick = async () => {
    // Tauri 2 exposes invoke under `__TAURI_INTERNALS__`, not the
    // legacy `__TAURI__` global (see reference_plugin_ui_tauri_access).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const internals = (window as any).__TAURI_INTERNALS__;
    if (internals) {
      setIsLoading(true);
      setError(null);
      try {
        const result = await internals.invoke('plugin:oaiy-filesystem|pick_file', {
          filters: [{
            name: 'Video Files',
            extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', 'wmv', 'flv']
          }]
        }) as string | null;

        if (result) {
          const sanitizedName = result.split(/[/\\]/).pop() || 'video';
          data.onVideoLoad?.(result, sanitizedName);
        }
        setIsLoading(false);
      } catch (err) {
        setIsLoading(false);
        setError(`Failed to load video: ${err}`);
      }
    } else {
      // Fallback to HTML file input for browser-only dev mode
      fileInputRef.current?.click();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = ''; // allow re-selecting the same file (no onChange otherwise)
  };

  const hasVideo = data.fileName && data.filePath;

  const collapsedPreview = (
    <div className="text-slate-400">
      {hasVideo ? (
        <span className="text-orange-400 truncate block">{data.fileName}</span>
      ) : (
        <span className="italic text-slate-500">No video</span>
      )}
    </div>
  );

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'path', type: 'source', position: Position.Right, color: '!bg-orange-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Input: Video"
      color="orange"
      icon={VideoIcon}
      width={260}
      collapsedWidth={140}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      outputHandles={outputHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          <div
            onClick={handleClick}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`
              w-full h-24 border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all
              ${isDragging
                ? 'border-orange-500 bg-orange-900/30'
                : hasVideo
                  ? 'border-orange-600 bg-slate-100 dark:bg-slate-900'
                  : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-900 hover:border-orange-600'
              }
            `}
          >
            {isLoading ? (
              <div className="text-center">
                <svg className="w-6 h-6 mx-auto text-orange-500 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
            ) : hasVideo ? (
              <div className="text-center px-2">
                <svg className="w-6 h-6 mx-auto text-orange-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="text-orange-400 text-xs truncate max-w-full">{data.fileName}</p>
              </div>
            ) : (
              <>
                <svg className="w-6 h-6 text-slate-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="text-slate-600 dark:text-slate-400 text-xs">Click or drop video</p>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v,.wmv,.flv"
            onChange={handleInputChange}
            className="hidden"
          />
        </>
      )}

      {error && (
        <div className="flex items-center gap-2 px-2 py-1.5 bg-red-900/30 border border-red-600/50 rounded text-red-400 text-xs">
          <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* Info */}
      <div className="text-slate-500 text-[10px] border-t border-slate-300 dark:border-slate-700 pt-2 mt-1">
        <div>Output: video file path</div>
        <div>Use with Video Frames node</div>
      </div>
    </CollapsibleNodeWrapper>
  );
}

export default memo(InputVideoNode);
