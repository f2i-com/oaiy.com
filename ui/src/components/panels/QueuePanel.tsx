/**
 * Queue Panel
 *
 * Displays the job queue status including active jobs, pending jobs,
 * and history. Provides controls for aborting jobs and configuring
 * queue behavior.
 */

import { memo, useCallback, useState, useEffect, useRef } from 'react';
import { useJobQueue } from '../../contexts/JobQueueContext';
import type { Job, JobConfig } from 'oaiy-core';
import { CopyLink } from '../ui/CopyButton';
import { useFocusTrap } from '../../hooks/useFocusTrap';

/**
 * Format a value for display (handles various types)
 */
function formatValue(value: unknown, depth = 0): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') {
    // Truncate long strings
    if (value.length > 500) {
      return `"${value.slice(0, 500)}..." (${value.length} chars)`;
    }
    return `"${value}"`;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    if (depth > 2) return `[Array(${value.length})]`;
    const items = value.slice(0, 5).map(v => formatValue(v, depth + 1));
    const suffix = value.length > 5 ? `, ... +${value.length - 5} more` : '';
    return `[${items.join(', ')}${suffix}]`;
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value as object);
    if (keys.length === 0) return '{}';
    if (depth > 2) return `{Object(${keys.length} keys)}`;
    return `{${keys.slice(0, 3).join(', ')}${keys.length > 3 ? ', ...' : ''}}`;
  }
  return String(value);
}

/**
 * Check if a URL is safe to load
 */
function isSafeUrl(url: string): boolean {
  if (!url || typeof url !== 'string') return false;
  const trimmed = url.trim().toLowerCase();
  return (
    trimmed.startsWith('http://') ||
    trimmed.startsWith('https://') ||
    trimmed.startsWith('data:image/') ||
    trimmed.startsWith('blob:')
  );
}

/**
 * Check if a string is an image URL
 */
function isImageUrl(url: string): boolean {
  if (!isSafeUrl(url)) return false;
  return (
    url.includes('/view?filename=') ||
    /\.(png|jpg|jpeg|gif|webp)(\?|$)/i.test(url) ||
    url.startsWith('data:image/')
  );
}

/**
 * Check if a value contains image URLs
 * Prioritizes __output__ field for workflow results to avoid duplicates
 */
function extractImageUrls(value: unknown): string[] {
  if (typeof value === 'string' && isImageUrl(value)) {
    return [value];
  }
  if (Array.isArray(value)) {
    return value.filter((v): v is string => typeof v === 'string' && isImageUrl(v));
  }
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>;

    // Prioritize __output__ field if it exists (this is the canonical workflow output)
    if ('__output__' in obj && obj.__output__ !== undefined) {
      const output = obj.__output__;
      if (typeof output === 'string' && isImageUrl(output)) {
        return [output];
      }
      if (Array.isArray(output)) {
        return output.filter((v): v is string => typeof v === 'string' && isImageUrl(v));
      }
    }

    // Fallback: search all values (but deduplicate)
    const urls = new Set<string>();
    for (const v of Object.values(obj)) {
      if (typeof v === 'string' && isImageUrl(v)) {
        urls.add(v);
      } else if (Array.isArray(v)) {
        for (const item of v) {
          if (typeof item === 'string' && isImageUrl(item)) {
            urls.add(item);
          }
        }
      }
    }
    return Array.from(urls);
  }
  return [];
}

interface QueuePanelProps {
  isOpen: boolean;
  onClose: () => void;
  /** Optional callback when user clicks on a flow to navigate to it */
  onNavigateToFlow?: (flowId: string) => void;
}

/**
 * Format timestamp to readable time
 */
function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Format duration in ms to readable string
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

/**
 * Status badge component
 */
const StatusBadge = memo(function StatusBadge({ status }: { status: Job['status'] }) {
  const styles: Record<Job['status'], string> = {
    pending: 'badge bg-slate-200 text-slate-700 dark:bg-slate-500/20 dark:text-slate-300',
    running: 'badge badge-blue',
    awaiting_ai: 'badge badge-purple',
    completed: 'badge badge-green',
    failed: 'badge badge-red',
    aborted: 'badge badge-amber',
  };

  const labels: Record<Job['status'], string> = {
    pending: 'Pending',
    running: 'Running',
    awaiting_ai: 'Awaiting AI',
    completed: 'Done',
    failed: 'Failed',
    aborted: 'Stopped',
  };

  return (
    <span className={styles[status]}>
      {labels[status]}
    </span>
  );
});

/**
 * Job output viewer component
 */
const JobOutputViewer = memo(function JobOutputViewer({ result }: { result: unknown }) {
  const [copyFeedback, setCopyFeedback] = useState(false);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());

  const handleCopy = useCallback(() => {
    const text = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 1500);
    }).catch(() => {
      // Clipboard access may fail in some contexts, silently ignore
    });
  }, [result]);

  // Check for images in the result
  const imageUrls = extractImageUrls(result);
  const hasImages = imageUrls.length > 0;

  // Render images if found
  const renderImages = () => {
    if (imageUrls.length === 1) {
      return (
        <div className="bg-slate-200 dark:bg-slate-800 rounded overflow-hidden">
          {!imageErrors.has(0) && isSafeUrl(imageUrls[0]) ? (
            <img
              src={imageUrls[0]}
              alt="Output"
              className="w-full h-auto max-h-40 object-contain"
              onError={() => setImageErrors(prev => new Set([...prev, 0]))}
            />
          ) : (
            <div className="h-20 flex items-center justify-center text-slate-500 text-xs">
              Failed to load image
            </div>
          )}
        </div>
      );
    }

    // Multiple images - show grid with selection
    return (
      <div className="space-y-2">
        {/* Enlarged preview of the selected thumbnail — otherwise clicking a
            thumbnail moved the highlight ring but did nothing else (dead affordance). */}
        {isSafeUrl(imageUrls[selectedImageIndex]) && !imageErrors.has(selectedImageIndex) && (
          <div className="bg-slate-200 dark:bg-slate-800 rounded overflow-hidden">
            <img
              src={imageUrls[selectedImageIndex]}
              alt={`Image ${selectedImageIndex + 1}`}
              className="w-full h-auto max-h-40 object-contain"
              onError={() => setImageErrors(prev => new Set([...prev, selectedImageIndex]))}
            />
          </div>
        )}
        <div className="grid grid-cols-3 gap-1">
          {imageUrls.slice(0, 9).map((url, index) => (
            <button
              key={index}
              onClick={() => setSelectedImageIndex(index)}
              className={`relative aspect-square rounded border overflow-hidden ${
                index === selectedImageIndex ? 'border-pink-500 ring-1 ring-pink-500/50' : 'border-slate-600'
              }`}
            >
              {imageErrors.has(index) ? (
                <div className="w-full h-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center text-slate-500">
                  <span className="text-[8px]">Error</span>
                </div>
              ) : isSafeUrl(url) ? (
                <img
                  src={url}
                  alt={`Image ${index + 1}`}
                  className="w-full h-full object-cover"
                  onError={() => setImageErrors(prev => new Set([...prev, index]))}
                />
              ) : null}
            </button>
          ))}
        </div>
        {imageUrls.length > 9 && (
          <div className="text-slate-500 text-[10px] text-center">
            +{imageUrls.length - 9} more images
          </div>
        )}
      </div>
    );
  };

  // Render the result based on type
  const renderResult = () => {
    if (result === null || result === undefined) {
      return <span className="text-slate-500 italic">No output</span>;
    }

    // If we have images, show them first
    if (hasImages) {
      return renderImages();
    }

    if (typeof result === 'string') {
      // Check if it's an image URL that wasn't detected
      if (isImageUrl(result)) {
        return renderImages();
      }
      return (
        <pre className="text-slate-700 dark:text-slate-300 text-[10px] whitespace-pre-wrap break-words font-mono">
          {result.length > 2000 ? `${result.slice(0, 2000)}...\n\n(truncated, ${result.length} total chars)` : result}
        </pre>
      );
    }

    if (typeof result === 'object') {
      // For objects, show a formatted view
      const entries = Object.entries(result as Record<string, unknown>);
      if (entries.length === 0) {
        return <span className="text-slate-500 italic">Empty result</span>;
      }

      return (
        <div className="space-y-1">
          {entries.slice(0, 10).map(([key, value]) => {
            // Check if this value is an image URL
            const valueImageUrls = extractImageUrls(value);
            if (valueImageUrls.length > 0) {
              return (
                <div key={key} className="flex flex-col">
                  <span className="text-cyan-700 dark:text-cyan-400 text-[10px] font-medium">{key}:</span>
                  <div className="pl-2 mt-1">
                    {valueImageUrls.length === 1 && isSafeUrl(valueImageUrls[0]) ? (
                      <img
                        src={valueImageUrls[0]}
                        alt={key}
                        className="w-full max-h-32 object-contain rounded border border-slate-600"
                      />
                    ) : (
                      <span className="text-pink-400 text-[10px]">
                        {valueImageUrls.length} image{valueImageUrls.length > 1 ? 's' : ''}
                      </span>
                    )}
                  </div>
                </div>
              );
            }

            return (
              <div key={key} className="flex flex-col">
                <span className="text-cyan-700 dark:text-cyan-400 text-[10px] font-medium">{key}:</span>
                <span className="text-slate-700 dark:text-slate-300 text-[10px] pl-2 break-words font-mono">
                  {formatValue(value)}
                </span>
              </div>
            );
          })}
          {entries.length > 10 && (
            <div className="text-slate-500 text-[10px] italic">
              +{entries.length - 10} more fields...
            </div>
          )}
        </div>
      );
    }

    return <span className="text-slate-700 dark:text-slate-300 text-[10px]">{String(result)}</span>;
  };

  return (
    <div className="mt-2 bg-slate-100 dark:bg-slate-900/80 rounded p-2 border border-slate-300/50 dark:border-slate-700/50">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-slate-500 dark:text-slate-400 text-[10px] font-medium">
          Output{hasImages ? ` (${imageUrls.length} image${imageUrls.length > 1 ? 's' : ''})` : ''}
        </span>
        <button
          onClick={handleCopy}
          className="text-[10px] text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
          title="Copy to clipboard"
        >
          {copyFeedback ? '✓ Copied' : 'Copy'}
        </button>
      </div>
      <div className="max-h-48 overflow-y-auto">
        {renderResult()}
      </div>
    </div>
  );
});

/**
 * Individual job item component
 */
interface JobItemProps {
  job: Job;
  onAbort?: () => void;
  onNavigate?: () => void;
  showAbort?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

const JobItem = memo(function JobItem({ job, onAbort, onNavigate, showAbort, isExpanded, onToggleExpand }: JobItemProps) {
  // Track elapsed time for running jobs with a timer
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    // Only run timer for active jobs
    if (job.status !== 'running' && job.status !== 'pending') return;
    if (!job.startedAt) return;

    const interval = setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => clearInterval(interval);
  }, [job.status, job.startedAt]);

  const duration = job.completedAt && job.startedAt
    ? job.completedAt - job.startedAt
    : job.startedAt
      ? now - job.startedAt
      : 0;

  const hasOutput = job.status === 'completed' && job.result !== undefined;

  return (
    <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg p-2.5 space-y-1.5">
      {/* Header row */}
      <div className="flex items-center justify-between gap-2">
        <button
          onClick={onNavigate}
          className="text-slate-700 dark:text-slate-200 text-sm font-medium truncate hover:text-blue-600 dark:hover:text-blue-400 transition-colors text-left"
          title={`Navigate to ${job.flowName}`}
        >
          {job.flowName || 'Untitled Flow'}
        </button>
        <StatusBadge status={job.status} />
      </div>

      {/* Info row */}
      <div className="flex items-center justify-between text-[10px] text-slate-500">
        <span>{formatTime(job.submittedAt)}</span>
        {duration > 0 && (
          <span className="flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {formatDuration(duration)}
          </span>
        )}
      </div>

      {/* Error message if failed */}
      {job.status === 'failed' && job.error && (
        <div className="text-red-700 dark:text-red-400 text-[10px] bg-red-100 dark:bg-red-950/50 border border-red-200 dark:border-transparent rounded px-1.5 py-1 flex items-start justify-between gap-1">
          <span className="truncate flex-1" title={job.error}>{job.error}</span>
          <CopyLink text={job.error} label="Copy" className="shrink-0" />
        </div>
      )}

      {/* Actions */}
      {showAbort && (job.status === 'running' || job.status === 'pending') && (
        <button
          onClick={onAbort}
          className="w-full mt-1 px-2 py-1 text-[10px] font-medium text-red-700 dark:text-red-400 bg-red-100 dark:bg-red-950/50 hover:bg-red-200 dark:hover:bg-red-900/50 rounded transition-colors"
        >
          {job.status === 'running' ? 'Stop' : 'Cancel'}
        </button>
      )}

      {/* View Output button for completed jobs */}
      {hasOutput && onToggleExpand && (
        <button
          onClick={onToggleExpand}
          className="w-full mt-1 px-2 py-1 text-[10px] font-medium text-cyan-700 dark:text-cyan-400 bg-cyan-100 dark:bg-cyan-950/50 hover:bg-cyan-200 dark:hover:bg-cyan-900/50 rounded transition-colors flex items-center justify-center gap-1"
        >
          <svg
            className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          {isExpanded ? 'Hide Output' : 'View Output'}
        </button>
      )}

      {/* Expanded output view */}
      {isExpanded && hasOutput && (
        <JobOutputViewer result={job.result} />
      )}
    </div>
  );
});

/**
 * Queue configuration component
 */
interface QueueConfigProps {
  config: JobConfig;
  onChange: (config: Partial<JobConfig>) => void;
}

const QueueConfig = memo(function QueueConfig({ config, onChange }: QueueConfigProps) {
  const handleModeChange = useCallback((newMode: 'sequential' | 'parallel') => {
    if (newMode === 'parallel') {
      // When switching to parallel, ensure maxConcurrency is at least 2
      onChange({
        mode: newMode,
        maxConcurrency: Math.max(config.maxConcurrency, 2)
      });
    } else {
      onChange({ mode: newMode });
    }
  }, [config.maxConcurrency, onChange]);

  return (
    <div className="space-y-2">
      {/* Mode selector */}
      <div className="flex items-center justify-between">
        <label className="text-slate-500 dark:text-slate-400 text-xs">Mode</label>
        <select
          value={config.mode}
          onChange={(e) => handleModeChange(e.target.value as 'sequential' | 'parallel')}
          className="bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 text-xs px-2 py-1 rounded border border-slate-300 dark:border-slate-700 focus:border-blue-500 focus:outline-none"
        >
          <option value="sequential">Sequential</option>
          {/* Re-enabled: module runtimes now take ctx per-call via createMethods(ctx)
              and cross-module calls route through the per-job ctx.modules accessor, so
              concurrent jobs no longer share ctx, per-job state, or module references.
              (See JobManager.processQueue + cli/test/{module-ctx,cross-module}-isolation.) */}
          <option value="parallel">Parallel</option>
        </select>
      </div>

      {/* Concurrency selector (only for parallel mode) */}
      {config.mode === 'parallel' && (
        <div className="flex items-center justify-between">
          <label className="text-slate-500 dark:text-slate-400 text-xs">Max Concurrent</label>
          <select
            value={config.maxConcurrency}
            onChange={(e) => onChange({ maxConcurrency: parseInt(e.target.value, 10) })}
            className="bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 text-xs px-2 py-1 rounded border border-slate-300 dark:border-slate-700 focus:border-blue-500 focus:outline-none"
          >
            {[2, 3, 4, 5, 6, 8].map((n) => (
              <option key={n} value={n}>{n} jobs</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
});

/**
 * Queue Panel component
 */
function QueuePanel({ isOpen, onClose, onNavigateToFlow }: QueuePanelProps) {
  const { jobManager, activeJobs, queuedJobs, history, config, setConfig, clearHistory } = useJobQueue();
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);
  const drawerRef = useRef<HTMLDivElement>(null);
  useFocusTrap(drawerRef, isOpen);

  const handleAbort = useCallback((jobId: string) => {
    jobManager.abort(jobId);
  }, [jobManager]);

  const handleNavigate = useCallback((flowId: string) => {
    if (onNavigateToFlow) {
      onNavigateToFlow(flowId);
    }
  }, [onNavigateToFlow]);

  const handleToggleExpand = useCallback((jobId: string) => {
    setExpandedJobId(prev => prev === jobId ? null : jobId);
  }, []);

  // Escape-to-close while open. Window-level so it fires regardless
  // of focus inside the drawer. Matches the other modal dialogs.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  const totalActive = activeJobs.length + queuedJobs.length;

  // Hide panel when closed (consistent with FlowsSidebar behavior)
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop — covers the canvas so the panel reads as a modal
          drawer, not a layout shift. Click-to-dismiss matches every
          other overlay in the app. */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />

      {/* Panel — fixed/overlay on every breakpoint so it never pushes
          the canvas around. Slides in from the right edge. */}
      <div
        ref={drawerRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="queue-panel-title"
        className="
          fixed z-50 right-0 top-0
          h-full bg-white dark:bg-slate-950 border-l border-slate-200 dark:border-slate-800 flex flex-col
          w-72 sm:w-80 md:w-96 lg:w-[28rem] shadow-2xl
        "
      >
        {/* Header */}
        <div className="p-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div>
              <h2 id="queue-panel-title" className="text-slate-700 dark:text-slate-200 font-semibold text-sm flex items-center gap-2">
                Job Queue
                {totalActive > 0 && (
                  <span className="w-5 h-5 bg-blue-600 text-white text-[10px] rounded-full flex items-center justify-center">
                    {totalActive}
                  </span>
                )}
              </h2>
              <p className="text-slate-400 dark:text-slate-500 text-xs">
                {config.mode === 'sequential' ? 'Sequential' : `Parallel (${config.maxConcurrency})`}
              </p>
            </div>
          </div>
          {/* Close button */}
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
            aria-label="Close queue panel"
          >
            <svg className="w-5 h-5 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          {/* Active Jobs */}
          {activeJobs.length > 0 && (
            <section>
              <h3 className="text-slate-500 dark:text-slate-400 text-xs font-medium mb-2 flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                Running ({activeJobs.length})
              </h3>
              <div className="space-y-2">
                {activeJobs.map((job) => (
                  <JobItem
                    key={job.id}
                    job={job}
                    showAbort
                    onAbort={() => handleAbort(job.id)}
                    onNavigate={() => handleNavigate(job.flowId)}
                  />
                ))}
              </div>
            </section>
          )}

          {/* Queued Jobs */}
          {queuedJobs.length > 0 && (
            <section>
              <h3 className="text-slate-500 dark:text-slate-400 text-xs font-medium mb-2">
                Pending ({queuedJobs.length})
              </h3>
              <div className="space-y-2">
                {queuedJobs.map((job) => (
                  <JobItem
                    key={job.id}
                    job={job}
                    showAbort
                    onAbort={() => handleAbort(job.id)}
                    onNavigate={() => handleNavigate(job.flowId)}
                  />
                ))}
              </div>
            </section>
          )}

          {/* Empty state for active/queued */}
          {activeJobs.length === 0 && queuedJobs.length === 0 && (
            <div className="empty-state py-8">
              <svg className="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <p className="empty-state-title">No active jobs</p>
              <p className="empty-state-description">Run a flow to see it here</p>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <section>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-slate-500 dark:text-slate-400 text-xs font-medium">
                  History ({history.length})
                </h3>
                {/* Confirm before wiping — one-click clear used to lose
                    the whole run history. Native confirm() is fine for a
                    destructive action that's rarely taken. */}
                <button
                  onClick={() => {
                    if (history.length === 0) return;
                    const msg = history.length === 1
                      ? 'Clear 1 history entry?'
                      : `Clear all ${history.length} history entries?`;
                    if (confirm(msg)) clearHistory();
                  }}
                  className="btn btn-ghost btn-sm text-[10px]"
                  title="Clear all completed/failed run history"
                >
                  Clear
                </button>
              </div>
              <div className="space-y-2">
                {history.slice(0, 20).map((job) => (
                  <JobItem
                    key={job.id}
                    job={job}
                    onNavigate={() => handleNavigate(job.flowId)}
                    isExpanded={expandedJobId === job.id}
                    onToggleExpand={() => handleToggleExpand(job.id)}
                  />
                ))}
                {history.length > 20 && (
                  <div className="text-slate-600 text-xs text-center py-1">
                    +{history.length - 20} more
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Configuration */}
          <section className="border-t border-slate-200 dark:border-slate-800 pt-3">
            <h3 className="text-slate-500 dark:text-slate-400 text-xs font-medium mb-2">Settings</h3>
            <QueueConfig config={config} onChange={setConfig} />
          </section>
        </div>

        {/* Status Bar */}
        <div className="p-2 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-[10px] text-slate-500 dark:text-slate-600">
          <span>Job Queue</span>
          <span className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${activeJobs.length > 0 ? 'bg-blue-500 animate-pulse' : 'bg-green-500'}`} />
            {activeJobs.length > 0 ? 'Processing' : 'Idle'}
          </span>
        </div>
      </div>
    </>
  );
}

export default memo(QueuePanel);
