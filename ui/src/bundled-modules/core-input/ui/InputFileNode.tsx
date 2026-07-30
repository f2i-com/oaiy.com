import { memo, useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Position } from '@xyflow/react';
import { CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';

interface InputFileNodeData {
  fileName?: string;
  fileType?: string;
  fileContent?: string;
  filePath?: string;
  imagePreview?: string;
  _collapsed?: boolean;
  showBodyProperties?: boolean;
  onFileLoad?: (content: string, fileName: string, type: string, preview?: string, filePath?: string) => void;
  onCollapsedChange?: (value: boolean) => void;
}

interface InputFileNodeProps {
  data: InputFileNodeData;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024;
// Threshold for loading file content into browser memory for preview
// Files larger than this will only pass the path to the backend
const UI_PREVIEW_LIMIT = 500 * 1024; // 500KB

const InputFileIcon = (
  <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

function InputFileNode({ data }: InputFileNodeProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeReaderRef = useRef<FileReader | null>(null);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  useEffect(() => {
    return () => {
      if (activeReaderRef.current) {
        activeReaderRef.current.abort();
      }
    };
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  const handleFile = async (file: File) => {
    if (activeReaderRef.current) {
      activeReaderRef.current.abort();
    }
    setError(null);

    const isImage = file.type.startsWith('image/');
    const isText = file.type.startsWith('text/') ||
      file.name.endsWith('.txt') ||
      file.name.endsWith('.md') ||
      file.name.endsWith('.json') ||
      file.name.endsWith('.csv');

    if (file.size > MAX_FILE_SIZE) {
      setError(`File too large (max 10MB)`);
      return;
    }

    if (file.size === 0) {
      setError('File is empty');
      return;
    }

    setIsLoading(true);

    // Get file path if available (Tauri adds .path property to File objects)
    // @ts-expect-error - path exists on File in Tauri context
    const filePath = file.path as string | undefined;

    // LARGE FILE OPTIMIZATION: If file exceeds preview limit, don't load content into memory
    // Just pass the file path to the backend which can handle it efficiently
    if (file.size > UI_PREVIEW_LIMIT && filePath) {
      setIsLoading(false);
      const sanitizedName = file.name.replace(/[/\\]/g, '_');
      const fileType = isImage ? 'image' : isText ? 'text' : 'binary';
      // Pass empty content but provide the file path for backend to read
      data.onFileLoad?.('', sanitizedName, `${fileType}_ref`, undefined, filePath);
      return;
    }

    const reader = new FileReader();
    activeReaderRef.current = reader;

    reader.onload = (e) => {
      setIsLoading(false);
      activeReaderRef.current = null;
      const content = e.target?.result as string;
      const sanitizedName = file.name.replace(/[/\\]/g, '_');
      if (isImage) {
        data.onFileLoad?.(content, sanitizedName, 'image', content, filePath);
      } else if (isText) {
        data.onFileLoad?.(content, sanitizedName, 'text', undefined, filePath);
      } else {
        data.onFileLoad?.(content, sanitizedName, 'binary', undefined, filePath);
      }
    };

    reader.onerror = () => {
      setIsLoading(false);
      activeReaderRef.current = null;
      setError(`Failed to read file`);
    };

    try {
      if (isImage || !isText) {
        reader.readAsDataURL(file);
      } else {
        reader.readAsText(file);
      }
    } catch {
      setIsLoading(false);
      activeReaderRef.current = null;
      setError(`Failed to read file`);
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
    // Prefer native Tauri file picker for guaranteed path access
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const tauri = (window as any).__TAURI__;
    if (tauri) {
      setIsLoading(true);
      try {
        const result = await tauri.core.invoke('plugin:oaiy-filesystem|pick_file', {
          filters: [{
            name: 'Supported Files',
            extensions: ['txt', 'md', 'json', 'csv', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'pdf']
          }]
        }) as string | null;

        if (result) {
          const sanitizedName = result.split(/[/\\]/).pop() || 'file';
          const ext = sanitizedName.split('.').pop()?.toLowerCase() || '';
          const isImage = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext);
          const isText = ['txt', 'md', 'json', 'csv'].includes(ext);
          const fileType = isImage ? 'image' : isText ? 'text' : 'binary';
          // Use file_ref type to indicate backend should read the file
          data.onFileLoad?.('', sanitizedName, `${fileType}_ref`, undefined, result);
        }
        setIsLoading(false);
      } catch (err) {
        setIsLoading(false);
        setError(`Failed to pick file: ${err}`);
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

  const hasFile = data.fileName && (data.fileContent || data.filePath);
  const isImage = data.fileType === 'image';

  const collapsedPreview = (
    <div className="text-slate-400">
      {hasFile ? (
        <span className="text-emerald-400 truncate block">{data.fileName}</span>
      ) : (
        <span className="italic text-slate-500">No file</span>
      )}
    </div>
  );

  const outputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'content', type: 'source', position: Position.Right, color: '!bg-emerald-500', size: 'lg' },
  ], []);

  return (
    <CollapsibleNodeWrapper
      title="Input: File"
      color="emerald"
      icon={InputFileIcon}
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
                ? 'border-emerald-500 bg-emerald-900/30'
                : hasFile
                  ? 'border-emerald-600 bg-slate-100 dark:bg-slate-900'
                  : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-900 hover:border-emerald-600'
              }
            `}
          >
            {isLoading ? (
              <div className="text-center">
                <svg className="w-6 h-6 mx-auto text-emerald-500 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              </div>
            ) : hasFile ? (
              isImage && data.imagePreview ? (
                <img src={data.imagePreview} alt={data.fileName} className="max-w-full max-h-full object-contain rounded" />
              ) : (
                <div className="text-center px-2">
                  <svg className="w-6 h-6 mx-auto text-emerald-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-emerald-400 text-xs truncate max-w-full">{data.fileName}</p>
                </div>
              )
            ) : (
              <>
                <svg className="w-6 h-6 text-slate-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="text-slate-600 dark:text-slate-400 text-xs">Drop image/text</p>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.txt,.md,.json,.csv,text/*"
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
    </CollapsibleNodeWrapper>
  );
}

export default memo(InputFileNode);
