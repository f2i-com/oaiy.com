import { memo, useState, useEffect, useRef, useCallback, useMemo, type ChangeEvent } from 'react';
import { Position } from '@xyflow/react';
import { useNodeResize, CollapsibleNodeWrapper, type HandleConfig } from 'oaiy-ui-components';
import { pathToMediaUrl } from 'oaiy-core';


interface OutputNodeData {
  label?: string;
  outputValue?: string | string[];
  _collapsed?: boolean;
  onLabelChange?: (value: string) => void;
  onCollapsedChange?: (value: boolean) => void;
  onShowToast?: (message: string, type?: 'success' | 'error' | 'info' | 'warning') => void;
  showBodyProperties?: boolean;
}

interface OutputNodeProps {
  data: OutputNodeData;
}

const isSafeUrl = (url: string): boolean => {
  if (!url || typeof url !== 'string') return false;
  const trimmed = url.trim().toLowerCase();
  return (
    trimmed.startsWith('http://') ||
    trimmed.startsWith('https://') ||
    trimmed.startsWith('data:image/') ||
    trimmed.startsWith('data:video/') ||
    trimmed.startsWith('data:audio/') ||
    trimmed.startsWith('blob:')
  );
};

// Raw base64 image payload with no `data:` prefix — e.g. OpenAI's gpt-image
// endpoint returns `data[0].b64_json`, which a Service Call surfaces verbatim.
// Recognise it by the image format's magic bytes (base64-encoded) and turn it
// into a data: URL so it renders inline instead of as a wall of text. Cheap
// prefix check first; only validate the (potentially large) base64 body when a
// prefix matches.
const rawBase64ImageDataUrl = (value: unknown): string | null => {
  if (typeof value !== 'string' || value.length < 64) return null;
  let mime: string | null = null;
  if (value.startsWith('iVBORw0KGgo')) mime = 'image/png';
  else if (value.startsWith('/9j/')) mime = 'image/jpeg';
  else if (value.startsWith('R0lGOD')) mime = 'image/gif';
  // 'UklGR' is base64('RIFF') — the container fourcc shared by WebP, WAV and
  // AVI, so the prefix alone can't tell them apart. Disambiguate by the 'WEBP'
  // format tag: its last three bytes ('E','B','P') sit at byte offset 9, which
  // is 3-byte aligned, so they ALWAYS encode to base64 chars 12-15 == 'RUJQ'
  // regardless of the 4-byte size field. WAV ('QVZF') and AVI ('Vkkg') differ,
  // so they fall through to null and audio/video detection takes over.
  else if (value.startsWith('UklGR') && value.slice(12, 16) === 'RUJQ') mime = 'image/webp';
  if (!mime) return null;
  if (!/^[A-Za-z0-9+/=\s]+$/.test(value)) return null;
  return `data:${mime};base64,${value.replace(/\s/g, '')}`;
};

const isImageUrl = (url: string): boolean => {
  if (rawBase64ImageDataUrl(url) !== null) return true;
  if (!isSafeUrl(url)) return false;
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:', 'blob:', 'data:'].includes(parsed.protocol)) return false;
  } catch {
    if (!url.startsWith('data:image/')) return false;
  }
  return (
    (url.includes('/view?filename=') && /\.(png|jpg|jpeg|gif|webp)/i.test(url)) ||
    /\.(png|jpg|jpeg|gif|webp)(\?|$)/i.test(url) ||
    url.startsWith('data:image/')
  );
};

const isVideoUrl = (url: string): boolean => {
  if (!isSafeUrl(url)) return false;
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:', 'blob:', 'data:'].includes(parsed.protocol)) return false;
  } catch {
    if (!url.startsWith('data:video/')) return false;
  }
  return (
    (url.includes('/view?filename=') && /\.(mp4|webm|mov|avi)/i.test(url)) ||
    /\.(mp4|webm|mov|avi)(\?|$)/i.test(url) ||
    url.startsWith('data:video/')
  );
};

const isAudioUrl = (url: string): boolean => {
  if (!isSafeUrl(url)) return false;
  try {
    const parsed = new URL(url);
    if (!['http:', 'https:', 'blob:', 'data:'].includes(parsed.protocol)) return false;
  } catch {
    if (!url.startsWith('data:audio/')) return false;
  }
  return (
    /\.(wav|mp3|ogg|flac|m4a|aac)(\?|$)/i.test(url) ||
    url.startsWith('data:audio/')
  );
};

// Check for local file paths (Windows or Unix)
const isAudioPath = (value: string): boolean => {
  if (!value || typeof value !== 'string') return false;
  // Windows path: C:\... or Unix path: /...
  const isFilePath = /^([A-Z]:[\\]|\/)/.test(value);
  if (!isFilePath) return false;
  return /\.(wav|mp3|ogg|flac|m4a|aac)$/i.test(value);
};

// Check for local video file paths (Windows or Unix)
const isVideoPath = (value: string): boolean => {
  if (!value || typeof value !== 'string') return false;
  // Windows path: C:\... or C:/... or Unix path: /...
  const isFilePath = /^([A-Z]:[\\/]|\/)/i.test(value);
  if (!isFilePath) return false;
  return /\.(mp4|webm|mov|avi|mkv)$/i.test(value);
};

const OutputIcon = (
  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
  </svg>
);

function OutputNode({ data }: OutputNodeProps) {
  const { size, handleResizeStart } = useNodeResize({
    initialWidth: 280,
    initialHeight: 260,
    constraints: { minWidth: 220, maxWidth: 600, minHeight: 200, maxHeight: 800 },
  });
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());
  const [copyFeedback, setCopyFeedback] = useState<'success' | 'error' | null>(null);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onCollapsedChangeRef = useRef(data.onCollapsedChange);

  useEffect(() => {
    onCollapsedChangeRef.current = data.onCollapsedChange;
  });

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const handleCollapsedChange = useCallback((collapsed: boolean) => {
    onCollapsedChangeRef.current?.(collapsed);
  }, []);

  // Helper to extract displayable value from various formats
  const extractDisplayValue = (v: unknown): string | null => {
    if (v === null || v === undefined) return null;
    if (typeof v === 'string') return v;
    // Handle video/audio output objects like { video: "path", path: "path" }
    if (typeof v === 'object' && v !== null) {
      const obj = v as Record<string, unknown>;
      // Prefer video, then audio, then path property for media playback
      if (typeof obj.video === 'string') return obj.video;
      if (typeof obj.audio === 'string') return obj.audio;
      if (typeof obj.path === 'string') return obj.path;
      // Fallback to JSON stringify for other objects
      return JSON.stringify(v);
    }
    return String(v);
  };

  const outputArray = Array.isArray(data.outputValue)
    ? data.outputValue.map(extractDisplayValue).filter((v): v is string => v !== null && v !== '')
    : data.outputValue
      ? [extractDisplayValue(data.outputValue)].filter((v): v is string => v !== null && v !== '')
      : [];
  const hasOutput = outputArray.length > 0;
  const isMultiple = outputArray.length > 1;

  useEffect(() => {
    if (outputArray.length > 0) {
      setSelectedImageIndex((prev: number) => prev >= outputArray.length ? 0 : prev);
    }
  }, [outputArray.length]);

  useEffect(() => {
    setImageErrors(new Set());
  }, [data.outputValue]);

  const isImageArray = hasOutput && outputArray.every(isImageUrl);
  const isSingleImage = hasOutput && !isMultiple && isImageUrl(outputArray[0]);
  const isVideoArray = hasOutput && outputArray.every((v) => isVideoUrl(v) || isVideoPath(v));
  const isSingleVideo = hasOutput && !isMultiple && (isVideoUrl(outputArray[0]) || isVideoPath(outputArray[0]));
  const isAudioArray = hasOutput && outputArray.every((v) => isAudioUrl(v) || isAudioPath(v));
  const isSingleAudio = hasOutput && !isMultiple && (isAudioUrl(outputArray[0]) || isAudioPath(outputArray[0]));
  const safeIndex = Math.min(selectedImageIndex, Math.max(0, outputArray.length - 1));
  const displayValue = isMultiple ? (outputArray[safeIndex] || '') : (outputArray[0] || '');

  const handleCopy = async () => {
    if (hasOutput) {
      const textContent = Array.isArray(data.outputValue)
        ? JSON.stringify(data.outputValue, null, 2)
        : typeof data.outputValue === 'object'
          ? JSON.stringify(data.outputValue, null, 2)
          : String(data.outputValue || '');
      try {
        await navigator.clipboard.writeText(textContent);
        setCopyFeedback('success');
        if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
        copyTimeoutRef.current = setTimeout(() => setCopyFeedback(null), 2000);
      } catch {
        setCopyFeedback('error');
        if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
        copyTimeoutRef.current = setTimeout(() => setCopyFeedback(null), 3000);
      }
    }
  };

  const handleSave = async () => {
    if (!hasOutput) return;

    // Check if output is videos
    if (isVideoArray || isSingleVideo) {
      let savedCount = 0;
      let failedCount = 0;

      // Download each video
      for (let i = 0; i < outputArray.length; i++) {
        const videoUrl = outputArray[i];
        if (!isSafeUrl(videoUrl)) continue;

        try {
          // Fetch the video
          const response = await fetch(videoUrl);
          const blob = await response.blob();

          // Determine file extension from URL or content type
          let ext = 'mp4';
          const urlMatch = videoUrl.match(/\.(mp4|webm|mov|avi)/i);
          if (urlMatch) {
            ext = urlMatch[1].toLowerCase();
          } else if (blob.type.includes('webm')) {
            ext = 'webm';
          } else if (blob.type.includes('avi')) {
            ext = 'avi';
          }

          // Create download link
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = blobUrl;
          const suffix = outputArray.length > 1 ? `_${i + 1}` : '';
          a.download = `${data.label || 'output'}${suffix}.${ext}`;
          a.click();
          URL.revokeObjectURL(blobUrl);
          savedCount++;

          // Small delay between downloads to avoid browser blocking
          if (i < outputArray.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 200));
          }
        } catch (err) {
          console.error(`Failed to download video ${i + 1}:`, err);
          failedCount++;
        }
      }

      // Show toast notification
      if (data.onShowToast) {
        if (failedCount === 0) {
          const msg = savedCount === 1
            ? 'Video saved to Downloads'
            : `${savedCount} videos saved to Downloads`;
          data.onShowToast(msg, 'success');
        } else if (savedCount === 0) {
          data.onShowToast('Failed to save videos', 'error');
        } else {
          data.onShowToast(`Saved ${savedCount} videos, ${failedCount} failed`, 'warning');
        }
      }
    } else if (isImageArray || isSingleImage) {
      // Check if output is images
      let savedCount = 0;
      let failedCount = 0;

      // Download each image
      for (let i = 0; i < outputArray.length; i++) {
        // Raw base64 (no data: prefix) → data URL so fetch() + download works.
        const imageUrl = rawBase64ImageDataUrl(outputArray[i]) ?? outputArray[i];
        if (!isSafeUrl(imageUrl)) continue;

        try {
          // Fetch the image
          const response = await fetch(imageUrl);
          const blob = await response.blob();

          // Determine file extension from URL or content type
          let ext = 'png';
          const urlMatch = imageUrl.match(/\.(png|jpg|jpeg|gif|webp)/i);
          if (urlMatch) {
            ext = urlMatch[1].toLowerCase();
          } else if (blob.type.includes('jpeg') || blob.type.includes('jpg')) {
            ext = 'jpg';
          } else if (blob.type.includes('gif')) {
            ext = 'gif';
          } else if (blob.type.includes('webp')) {
            ext = 'webp';
          }

          // Create download link
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = blobUrl;
          const suffix = outputArray.length > 1 ? `_${i + 1}` : '';
          a.download = `${data.label || 'output'}${suffix}.${ext}`;
          a.click();
          URL.revokeObjectURL(blobUrl);
          savedCount++;

          // Small delay between downloads to avoid browser blocking
          if (i < outputArray.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 200));
          }
        } catch (err) {
          console.error(`Failed to download image ${i + 1}:`, err);
          failedCount++;
        }
      }

      // Show toast notification
      if (data.onShowToast) {
        if (failedCount === 0) {
          const msg = savedCount === 1
            ? 'Image saved to Downloads'
            : `${savedCount} images saved to Downloads`;
          data.onShowToast(msg, 'success');
        } else if (savedCount === 0) {
          data.onShowToast('Failed to save images', 'error');
        } else {
          data.onShowToast(`Saved ${savedCount} images, ${failedCount} failed`, 'warning');
        }
      }
    } else {
      // Save as text file
      const textContent = Array.isArray(data.outputValue)
        ? JSON.stringify(data.outputValue, null, 2)
        : typeof data.outputValue === 'object'
          ? JSON.stringify(data.outputValue, null, 2)
          : String(data.outputValue || '');
      const blob = new Blob([textContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${data.label || 'output'}.txt`;
      a.click();
      URL.revokeObjectURL(url);

      // Show toast notification
      if (data.onShowToast) {
        data.onShowToast(`Saved ${data.label || 'output'}.txt to Downloads`, 'success');
      }
    }
  };

  const contentHeight = Math.max(60, size.height - 160);

  const collapsedPreview = (
    <div className="text-slate-400">
      {hasOutput ? (
        isVideoArray || isSingleVideo ? (
          <span className="text-orange-400">Video{isMultiple ? ` (${outputArray.length})` : ''}</span>
        ) : isImageArray || isSingleImage ? (
          <span className="text-pink-400">Image{isMultiple ? ` (${outputArray.length})` : ''}</span>
        ) : (
          <span className="truncate text-[10px]">{displayValue.substring(0, 40)}...</span>
        )
      ) : (
        <span className="italic text-slate-500">Waiting...</span>
      )}
    </div>
  );

  const inputHandles = useMemo<HandleConfig[]>(() => [
    { id: 'result', type: 'target', position: Position.Left, color: '!bg-blue-500', size: 'lg' },
  ], []);

  const resizeHandles = (
    <>
      <div
        className="nodrag absolute top-0 right-0 w-2 h-full cursor-ew-resize opacity-0 group-hover:opacity-100 hover:bg-emerald-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 'e')}
      />
      <div
        className="nodrag absolute bottom-0 left-0 w-full h-2 cursor-ns-resize opacity-0 group-hover:opacity-100 hover:bg-emerald-500/30 transition-all"
        onMouseDown={(e) => handleResizeStart(e, 's')}
      />
      <div
        className="nodrag absolute bottom-1 right-1 w-3 h-3 cursor-nwse-resize opacity-0 group-hover:opacity-100 transition-opacity"
        onMouseDown={(e) => handleResizeStart(e, 'se')}
      >
        <svg className="w-3 h-3 text-slate-500" viewBox="0 0 24 24" fill="currentColor">
          <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22Z" />
        </svg>
      </div>
    </>
  );

  return (
    <CollapsibleNodeWrapper
      title="Output"
      color="emerald"
      icon={OutputIcon}
      width={size.width}
      collapsedWidth={140}
      isCollapsed={data._collapsed}
      onCollapsedChange={handleCollapsedChange}
      collapsedPreview={collapsedPreview}
      inputHandles={inputHandles}
      resizeHandles={resizeHandles}
    >
      {data.showBodyProperties !== false && (
        <>
          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Label</label>
            <input
              type="text"
              className="nodrag nowheel w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded px-2 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
              placeholder="final_result"
              value={data.label || ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => data.onLabelChange?.(e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
            />
          </div>

          <div>
            <label className="text-slate-600 dark:text-slate-400 text-xs block mb-1">Result</label>
            {isVideoArray || isSingleVideo ? (
              <div
                className="bg-slate-100 dark:bg-slate-900 border border-orange-600 rounded overflow-hidden"
                style={{ height: contentHeight }}
              >
                {(() => {
                  // Convert local file path to media server URL
                  const isPath = isVideoPath(displayValue);
                  const videoSrc = isPath ? pathToMediaUrl(displayValue) : displayValue;

                  const isValidSrc = isSafeUrl(displayValue) || isPath;
                  return isValidSrc ? (
                    <video
                      key={displayValue}
                      src={videoSrc}
                      controls
                      className="w-full h-full object-contain"
                      onError={() => setImageErrors((prev: Set<number>) => new Set([...prev, 0]))}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-slate-500 text-xs p-2">
                      <span className="truncate">{displayValue}</span>
                    </div>
                  );
                })()}
              </div>
            ) : isImageArray || isSingleImage ? (
              <div
                className="bg-slate-100 dark:bg-slate-900 border border-emerald-600 rounded overflow-hidden"
                style={{ height: contentHeight }}
              >
                {isMultiple ? (
                  <div className="grid grid-cols-2 gap-1 p-1 overflow-auto h-full">
                    {outputArray.map((url, index) => (
                      <button
                        key={`img-${index}`}
                        onClick={() => setSelectedImageIndex(index)}
                        className={`relative aspect-square rounded border overflow-hidden ${index === safeIndex ? 'border-pink-500 ring-1 ring-pink-500/50' : 'border-slate-300 dark:border-slate-600'
                          }`}
                      >
                        {imageErrors.has(index) ? (
                          <div className="w-full h-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center text-slate-500">
                            <span className="text-xs">Error</span>
                          </div>
                        ) : (isSafeUrl(url) || rawBase64ImageDataUrl(url)) ? (
                          <img
                            src={rawBase64ImageDataUrl(url) ?? url}
                            alt={`Image ${index + 1}`}
                            className="w-full h-full object-cover"
                            onError={() => setImageErrors((prev: Set<number>) => new Set([...prev, index]))}
                          />
                        ) : null}
                      </button>
                    ))}
                  </div>
                ) : (isSafeUrl(displayValue) || rawBase64ImageDataUrl(displayValue)) ? (
                  <img
                    src={rawBase64ImageDataUrl(displayValue) ?? displayValue}
                    alt="Output"
                    className="w-full h-full object-contain"
                    onError={() => setImageErrors((prev: Set<number>) => new Set([...prev, 0]))}
                  />
                ) : null}
              </div>
            ) : isAudioArray || isSingleAudio ? (
              <div
                className="bg-slate-100 dark:bg-slate-900 border border-teal-600 rounded overflow-hidden p-3"
                style={{ minHeight: 60 }}
              >
                {isMultiple ? (
                  <div className="space-y-2 overflow-auto max-h-32">
                    {outputArray.map((audioPath, index) => {
                      // Convert Windows path to Tauri asset URL
                      const audioSrc = isAudioPath(audioPath)
                        ? `asset://localhost/${encodeURIComponent(audioPath).replace(/%5C/g, '/').replace(/%3A/g, ':')}`
                        : audioPath;
                      const fileName = audioPath.split(/[\\/]/).pop() || `Audio ${index + 1}`;
                      return (
                        <div key={`audio-${index}`} className="flex items-center gap-2 bg-slate-200/50 dark:bg-slate-800/50 p-2 rounded">
                          <audio
                            controls
                            className="h-8 flex-1"
                            style={{ filter: 'invert(1) hue-rotate(180deg)' }}
                          >
                            <source src={audioSrc} type="audio/wav" />
                          </audio>
                          <span className="text-[10px] text-slate-400 truncate max-w-[80px]">{fileName}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <audio
                      key={displayValue}
                      controls
                      className="h-8 flex-1"
                      style={{ filter: 'invert(1) hue-rotate(180deg)' }}
                    >
                      <source
                        src={isAudioPath(displayValue)
                          ? `asset://localhost/${encodeURIComponent(displayValue).replace(/%5C/g, '/').replace(/%3A/g, ':')}`
                          : displayValue}
                        type="audio/wav"
                      />
                    </audio>
                    <span className="text-[10px] text-slate-400 truncate max-w-[80px]">
                      {displayValue.split(/[\\/]/).pop()}
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <textarea
                readOnly
                value={hasOutput
                  ? (Array.isArray(data.outputValue)
                    ? JSON.stringify(data.outputValue, null, 2)
                    : typeof data.outputValue === 'object'
                      ? JSON.stringify(data.outputValue, null, 2)
                      : String(data.outputValue || ''))
                  : ''}
                placeholder="Waiting for execution..."
                style={{ height: contentHeight }}
                className={`nodrag nowheel w-full bg-white dark:bg-slate-900 border rounded px-2 py-2 text-xs font-mono resize-none ${hasOutput ? 'border-emerald-600 text-emerald-700 dark:text-emerald-300' : 'border-slate-300 dark:border-slate-700 text-slate-500'
                  }`}
                onMouseDown={(e) => e.stopPropagation()}
              />
            )}
          </div>

          {hasOutput && (
            <div className="flex gap-1.5">
              <button
                onClick={handleCopy}
                className={`flex-1 px-2 py-1.5 text-xs rounded flex items-center justify-center gap-1 ${copyFeedback === 'success' ? 'bg-green-700 text-white' :
                  copyFeedback === 'error' ? 'bg-red-700 text-white' :
                    'bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'
                  }`}
              >
                {copyFeedback === 'success' ? 'Copied!' : copyFeedback === 'error' ? 'Failed' : 'Copy'}
              </button>
              <button
                onClick={handleSave}
                className="flex-1 px-2 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white text-xs rounded"
              >
                Save
              </button>
            </div>
          )}
        </>
      )}
    </CollapsibleNodeWrapper>
  );
}

export default memo(OutputNode);
