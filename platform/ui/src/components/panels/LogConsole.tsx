import { memo, useEffect, useRef, useCallback, useMemo } from 'react';
import type { LogEntry } from 'oaiy-core';
import { CopyButton } from '../ui/CopyButton';

interface LogConsoleProps {
  logs: LogEntry[];
  onClear?: () => void;
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

// Each level gets a badge (background + text) and the default row text
// colour. Both sides of the variant are paired so light + dark mode get
// the same level encoding with 4.5:1 contrast. Dark side is original;
// light side added so the panel is legible on the white workspace.
const typeStyles: Record<string, { badge: string; text: string }> = {
  info:    { badge: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',             text: 'text-slate-700 dark:text-slate-300' },
  error:   { badge: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',                 text: 'text-red-700 dark:text-red-300' },
  success: { badge: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',         text: 'text-green-700 dark:text-green-300' },
  node:    { badge: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',     text: 'text-slate-700 dark:text-slate-300' },
  output:  { badge: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200', text: 'text-emerald-700 dark:text-emerald-300' },
};

// Memoized log entry component for better performance with large log lists
interface LogEntryRowProps {
  log: LogEntry;
  formatTime: (timestamp: number) => string;
}

const LogEntryRow = memo(function LogEntryRow({ log, formatTime }: LogEntryRowProps) {
  const style = typeStyles[log.type || 'info'];
  const isNode = log.source !== 'System';

  return (
    <div className="flex gap-1.5 sm:gap-2 items-start">
      {/* Timestamp - hidden on very small screens */}
      <span className="text-slate-600 shrink-0 w-14 sm:w-16 hidden xs:inline">
        {formatTime(log.timestamp)}
      </span>

      {/* Source Badge */}
      <span
        className={`badge shrink-0 ${log.source === 'Output'
          ? 'badge-green'
          : isNode
            ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
            : 'badge-blue'
          }`}
      >
        {log.source === 'Output' ? 'OUT' : isNode ? 'NODE' : 'SYS'}
      </span>

      {/* Message */}
      <span className={`${log.source === 'Output' ? 'text-emerald-700 dark:text-emerald-300' : style.text} whitespace-pre-wrap break-all text-[11px] sm:text-xs`}>
        {log.message}
        {/* Blinking cursor for streaming */}
        {log.isStreaming && (
          <span className="inline-block w-1.5 h-3 bg-green-500 ml-0.5 align-middle animate-blink" />
        )}
      </span>
    </div>
  );
});

function LogConsole({ logs, onClear, isOpen, onClose, className = '' }: LogConsoleProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether the user is pinned to the bottom — tracked from the scroll event (not
  // measured post-render) so stick-to-bottom only resumes when they're near the
  // bottom; otherwise scrolling up to read history gets yanked back down mid-run.
  const isPinnedRef = useRef(true);
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (el) isPinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);

  // Auto-scroll to bottom when new logs arrive — only if already pinned there.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && isPinnedRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  const formatTime = useCallback((timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  }, []);

  // Format all logs as text for copying
  const logsAsText = useMemo(() => {
    return logs.map(log => {
      const time = formatTime(log.timestamp);
      const source = log.source === 'Output' ? 'OUT' : log.source === 'System' ? 'SYS' : 'NODE';
      return `[${time}] [${source}] ${log.message}`;
    }).join('\n');
  }, [logs, formatTime]);

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-40"
          onClick={onClose}
        />
      )}

      {/* Log panel */}
      <div
        className={`
          fixed md:relative z-50 md:z-auto right-0 top-0
          h-full bg-white dark:bg-slate-950 border-l border-slate-200 dark:border-slate-800 flex flex-col
          transition-transform duration-300 ease-in-out
          w-72 sm:w-80 md:w-72 lg:w-80
          ${isOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-0'}
          ${className}
        `}
      >
        {/* Header */}
        <div className="p-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div>
              <h2 className="text-slate-700 dark:text-slate-200 font-semibold text-sm">Execution Log</h2>
              <p className="text-slate-400 dark:text-slate-500 text-xs">{logs.length} entries</p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            {logs.length > 0 && (
              <CopyButton text={logsAsText} label="Copy" size="sm" />
            )}
            {onClear && (
              <button
                onClick={onClear}
                className="btn btn-ghost btn-sm"
              >
                Clear
              </button>
            )}
            {/* Hide/collapse button */}
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
              title="Hide execution log"
              aria-label="Hide execution log"
            >
              <svg aria-hidden="true" className="w-4 h-4 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Log Entries */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          role="log"
          aria-live="polite"
          aria-relevant="additions"
          aria-label="Execution log"
          className="flex-1 overflow-y-auto p-2 sm:p-3 space-y-2 font-mono text-xs"
        >
          {logs.length === 0 ? (
            <div className="empty-state py-12">
              <svg className="empty-state-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h6a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
              </svg>
              <p className="empty-state-title">No logs yet</p>
              <p className="empty-state-description">Run a workflow to see output</p>
            </div>
          ) : (
            logs.map((log) => (
              <LogEntryRow key={log.id} log={log} formatTime={formatTime} />
            ))
          )}
        </div>

        {/* Status Bar */}
        <div className="p-2 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-[10px] text-slate-500 dark:text-slate-600">
          <span>Workflow Runtime</span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            Ready
          </span>
        </div>
      </div>
    </>
  );
}

export default memo(LogConsole);
