/**
 * Queue — the flows this editor is running, waiting to run, and has run this
 * session (the JobQueue), with how it runs them. A page in the editor's main
 * area, like Data and Settings.
 *
 * It is this editor's own queue, kept in memory: a reload starts it empty. In
 * OAIY's window every run on the machine (triggers, the agent, other clients)
 * is in the dashboard's Run history, beside this editor's tab.
 */

import { memo, useCallback, useState, useEffect } from 'react';
import { ChevronDown, Clock, Copy, ListChecks, Square } from 'lucide-react';
import { useJobQueue } from '../../contexts/JobQueueContext';
import type { Job, JobConfig } from 'oaiy-core';
import { CopyLink } from '../ui/CopyButton';
import SectionPage, { Card, EmptyState } from '../chrome/SectionPage';
import { useConfirmDialog } from '../../hooks/useConfirmDialog';
import { oaiyDesktop } from '../../lib/oaiyAgentTools';

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
  /** Open a flow on the canvas (a job's name is a link to its flow). */
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

/** A job's state, as the dashboard's status pills. */
const StatusBadge = memo(function StatusBadge({ status }: { status: Job['status'] }) {
  const tone: Record<Job['status'], string> = {
    pending: 'oaiy-pill dot',
    running: 'oaiy-pill dot accent live',
    awaiting_ai: 'oaiy-pill dot info live',
    completed: 'oaiy-pill dot ok',
    failed: 'oaiy-pill dot err',
    aborted: 'oaiy-pill dot warn',
  };

  const labels: Record<Job['status'], string> = {
    pending: 'waiting',
    running: 'running',
    awaiting_ai: 'awaiting AI',
    completed: 'done',
    failed: 'failed',
    aborted: 'stopped',
  };

  return <span className={tone[status]}>{labels[status]}</span>;
});

/**
 * What a finished job returned: its pictures, or its fields.
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

  const failed = <div className="grid h-20 place-items-center text-xs text-content-faint">Could not load the picture</div>;

  // Render images if found
  const renderImages = () => {
    if (imageUrls.length === 1) {
      return (
        <div className="overflow-hidden rounded-[var(--r-sm)] bg-surface-tertiary">
          {!imageErrors.has(0) && isSafeUrl(imageUrls[0]) ? (
            <img
              src={imageUrls[0]}
              alt="Output"
              className="h-auto max-h-56 w-full object-contain"
              onError={() => setImageErrors(prev => new Set([...prev, 0]))}
            />
          ) : failed}
        </div>
      );
    }

    // Multiple images - show grid with selection
    return (
      <div className="space-y-2">
        {/* Enlarged preview of the selected thumbnail. */}
        {isSafeUrl(imageUrls[selectedImageIndex]) && !imageErrors.has(selectedImageIndex) && (
          <div className="overflow-hidden rounded-[var(--r-sm)] bg-surface-tertiary">
            <img
              src={imageUrls[selectedImageIndex]}
              alt={`Image ${selectedImageIndex + 1}`}
              className="h-auto max-h-56 w-full object-contain"
              onError={() => setImageErrors(prev => new Set([...prev, selectedImageIndex]))}
            />
          </div>
        )}
        <div className="grid grid-cols-6 gap-1">
          {imageUrls.slice(0, 12).map((url, index) => (
            <button
              key={index}
              onClick={() => setSelectedImageIndex(index)}
              aria-label={`Show picture ${index + 1}`}
              className={`relative aspect-square overflow-hidden rounded-[var(--r-sm)] border ${
                index === selectedImageIndex ? 'border-accent ring-1 ring-accent/50' : 'border-edge-primary'
              }`}
            >
              {imageErrors.has(index) ? (
                <div className="grid h-full w-full place-items-center bg-surface-tertiary text-[9px] text-content-faint">Error</div>
              ) : isSafeUrl(url) ? (
                <img
                  src={url}
                  alt={`Image ${index + 1}`}
                  className="h-full w-full object-cover"
                  onError={() => setImageErrors(prev => new Set([...prev, index]))}
                />
              ) : null}
            </button>
          ))}
        </div>
        {imageUrls.length > 12 && (
          <div className="text-center text-[11px] text-content-faint">+{imageUrls.length - 12} more pictures</div>
        )}
      </div>
    );
  };

  // Render the result based on type
  const renderResult = () => {
    if (result === null || result === undefined) {
      return <span className="italic text-content-faint">No output</span>;
    }

    if (hasImages) {
      return renderImages();
    }

    if (typeof result === 'string') {
      if (isImageUrl(result)) {
        return renderImages();
      }
      return (
        <pre className="m-0 whitespace-pre-wrap break-words font-mono text-[11px] text-content-secondary">
          {result.length > 2000 ? `${result.slice(0, 2000)}...\n\n(truncated, ${result.length} total chars)` : result}
        </pre>
      );
    }

    if (typeof result === 'object') {
      const entries = Object.entries(result as Record<string, unknown>);
      if (entries.length === 0) {
        return <span className="italic text-content-faint">Empty result</span>;
      }

      return (
        <div className="space-y-1.5">
          {entries.slice(0, 10).map(([key, value]) => {
            const valueImageUrls = extractImageUrls(value);
            if (valueImageUrls.length > 0) {
              return (
                <div key={key} className="flex flex-col gap-1">
                  <span className="font-mono text-[11px] font-semibold text-signal-cyan">{key}</span>
                  {valueImageUrls.length === 1 && isSafeUrl(valueImageUrls[0]) ? (
                    <img
                      src={valueImageUrls[0]}
                      alt={key}
                      className="max-h-40 w-full rounded-[var(--r-sm)] border border-edge-primary object-contain"
                    />
                  ) : (
                    <span className="text-[11px] text-content-secondary">
                      {valueImageUrls.length} picture{valueImageUrls.length > 1 ? 's' : ''}
                    </span>
                  )}
                </div>
              );
            }

            return (
              <div key={key} className="flex flex-col">
                <span className="font-mono text-[11px] font-semibold text-signal-cyan">{key}</span>
                <span className="break-words pl-2 font-mono text-[11px] text-content-secondary">
                  {formatValue(value)}
                </span>
              </div>
            );
          })}
          {entries.length > 10 && (
            <div className="text-[11px] italic text-content-faint">+{entries.length - 10} more fields…</div>
          )}
        </div>
      );
    }

    return <span className="text-[11px] text-content-secondary">{String(result)}</span>;
  };

  return (
    <div className="mt-2 rounded-[var(--r-ctl)] border border-edge-primary bg-surface-tertiary/60 p-2.5">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="oaiy-label">
          Output{hasImages ? ` · ${imageUrls.length} picture${imageUrls.length > 1 ? 's' : ''}` : ''}
        </span>
        <button onClick={handleCopy} className="btn btn-ghost btn-sm" title="Copy to the clipboard">
          <Copy size={12} />
          {copyFeedback ? 'Copied' : 'Copy'}
        </button>
      </div>
      <div className="max-h-64 overflow-y-auto">
        {renderResult()}
      </div>
    </div>
  );
});

interface JobItemProps {
  job: Job;
  onAbort?: () => void;
  onNavigate?: () => void;
  showAbort?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

/** One job: its flow (a link to it), when, how long, its state, and its result. */
const JobItem = memo(function JobItem({ job, onAbort, onNavigate, showAbort, isExpanded, onToggleExpand }: JobItemProps) {
  // Track elapsed time for running jobs with a timer
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
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
  const active = job.status === 'running' || job.status === 'pending';

  return (
    <li className="px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <button
            onClick={onNavigate}
            className="min-w-0 truncate text-left text-[13px] font-semibold text-content-primary hover:text-accent"
            title={`Open ${job.flowName || 'this flow'} on the canvas`}
          >
            {job.flowName || 'Untitled flow'}
          </button>
          <span className="flex items-center gap-2 font-mono text-[11px] text-content-faint">
            {formatTime(job.submittedAt)}
            {duration > 0 && (
              <span className="inline-flex items-center gap-1">
                <Clock size={11} />
                {formatDuration(duration)}
              </span>
            )}
          </span>
        </div>
        <StatusBadge status={job.status} />
        {showAbort && active && (
          <button onClick={onAbort} className="btn btn-danger btn-sm">
            <Square size={11} />
            {job.status === 'running' ? 'Stop' : 'Cancel'}
          </button>
        )}
        {hasOutput && onToggleExpand && (
          <button onClick={onToggleExpand} className="btn btn-sm" aria-expanded={!!isExpanded}>
            <ChevronDown size={13} className={isExpanded ? 'rotate-180 transition-transform' : 'transition-transform'} />
            {isExpanded ? 'Hide output' : 'Output'}
          </button>
        )}
      </div>

      {job.status === 'failed' && job.error && (
        <div className="oaiy-banner mt-2">
          <span className="min-w-0 flex-1 break-words" title={job.error}>{job.error}</span>
          <CopyLink text={job.error} label="Copy" className="shrink-0" />
        </div>
      )}

      {isExpanded && hasOutput && <JobOutputViewer result={job.result} />}
    </li>
  );
});

interface QueueConfigProps {
  config: JobConfig;
  onChange: (config: Partial<JobConfig>) => void;
}

/** How the queue runs flows: one at a time, or several. */
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
    <div className="oaiy-form-grid narrow">
      <label className="oaiy-field">
        <span>Runs</span>
        <select
          value={config.mode}
          onChange={(e) => handleModeChange(e.target.value as 'sequential' | 'parallel')}
          className="oaiy-select"
        >
          <option value="sequential">One at a time</option>
          {/* Module runtimes take ctx per call, so concurrent jobs share no
              state (see JobManager.processQueue). */}
          <option value="parallel">Several at once</option>
        </select>
      </label>
      {config.mode === 'parallel' && (
        <label className="oaiy-field">
          <span>At most</span>
          <select
            value={config.maxConcurrency}
            onChange={(e) => onChange({ maxConcurrency: parseInt(e.target.value, 10) })}
            className="oaiy-select"
          >
            {[2, 3, 4, 5, 6, 8].map((n) => (
              <option key={n} value={n}>{n} flows</option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
});

/** A count for the Queue tab: how many flows are running or waiting. */
export function QueueCount() {
  const { activeJobs, queuedJobs } = useJobQueue();
  const n = activeJobs.length + queuedJobs.length;
  if (n === 0) return null;
  return <em className="oaiy-tab-count" aria-label={`${n} running or waiting`}>{n}</em>;
}

function QueuePanel({ onNavigateToFlow }: QueuePanelProps) {
  const { jobManager, activeJobs, queuedJobs, history, config, setConfig, clearHistory } = useJobQueue();
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);
  const confirm = useConfirmDialog();

  const handleAbort = useCallback((jobId: string) => {
    jobManager.abort(jobId);
  }, [jobManager]);

  const handleNavigate = useCallback((flowId: string) => {
    onNavigateToFlow?.(flowId);
  }, [onNavigateToFlow]);

  const handleToggleExpand = useCallback((jobId: string) => {
    setExpandedJobId(prev => prev === jobId ? null : jobId);
  }, []);

  const handleClear = useCallback(async () => {
    if (history.length === 0) return;
    const ok = await confirm({
      title: 'Clear the finished runs?',
      message: history.length === 1
        ? 'The one finished run and its output are taken off this list.'
        : `All ${history.length} finished runs and their outputs are taken off this list.`,
      confirmLabel: 'Clear',
      variant: 'danger',
    });
    if (ok) clearHistory();
  }, [history.length, confirm, clearHistory]);

  const busy = activeJobs.length > 0;
  const inOaiy = oaiyDesktop() !== null;

  return (
    <SectionPage
      kicker="Flows"
      title="Queue"
      description={
        inOaiy
          ? "Flows this editor is running, waiting to run, and has run since it opened. Every run on this machine is in OAIY's Run history."
          : 'Flows this editor is running, waiting to run, and has run since it opened.'
      }
      actions={
        <span className={busy ? 'oaiy-pill dot accent live' : 'oaiy-pill dot ok'}>
          {busy ? `running ${activeJobs.length}` : 'idle'}
        </span>
      }
      testId="queue-page"
    >
      {activeJobs.length === 0 && queuedJobs.length === 0 ? (
        <Card title="Now">
          <EmptyState icon={<ListChecks size={26} />} title="Nothing is running">
            Press Run on the canvas: the flow is listed here while it runs, and under Finished with what it returned.
          </EmptyState>
        </Card>
      ) : (
        <>
          {activeJobs.length > 0 && (
            <Card title="Running" count={activeJobs.length} flush>
              <ul className="oaiy-rows m-0 list-none p-0">
                {activeJobs.map((job) => (
                  <JobItem key={job.id} job={job} showAbort onAbort={() => handleAbort(job.id)} onNavigate={() => handleNavigate(job.flowId)} />
                ))}
              </ul>
            </Card>
          )}
          {queuedJobs.length > 0 && (
            <Card title="Waiting" count={queuedJobs.length} flush>
              <ul className="oaiy-rows m-0 list-none p-0">
                {queuedJobs.map((job) => (
                  <JobItem key={job.id} job={job} showAbort onAbort={() => handleAbort(job.id)} onNavigate={() => handleNavigate(job.flowId)} />
                ))}
              </ul>
            </Card>
          )}
        </>
      )}

      {history.length > 0 && (
        <Card
          title="Finished"
          count={history.length}
          flush
          actions={<button onClick={handleClear} className="btn btn-ghost btn-sm" title="Take every finished run off this list">Clear</button>}
        >
          <ul className="oaiy-rows m-0 list-none p-0">
            {history.slice(0, 50).map((job) => (
              <JobItem
                key={job.id}
                job={job}
                onNavigate={() => handleNavigate(job.flowId)}
                isExpanded={expandedJobId === job.id}
                onToggleExpand={() => handleToggleExpand(job.id)}
              />
            ))}
          </ul>
          {history.length > 50 && (
            <p className="oaiy-help faint px-4 py-2">and {history.length - 50} earlier</p>
          )}
        </Card>
      )}

      <Card title="How flows run">
        <p className="oaiy-card-text">
          One at a time keeps a flow's models and services to itself; several at once finishes a batch sooner when the machine can take it.
        </p>
        <QueueConfig config={config} onChange={setConfig} />
      </Card>
    </SectionPage>
  );
}

export default memo(QueuePanel);
