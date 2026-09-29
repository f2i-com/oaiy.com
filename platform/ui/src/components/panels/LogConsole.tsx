import { memo, useEffect, useRef, useCallback, useMemo } from 'react';
import { ScrollText, X } from 'lucide-react';
import type { LogEntry } from 'oaiy-core';
import { CopyButton } from '../ui/CopyButton';

/**
 * The execution log: what the flow's run said, line by line. It sits in the
 * inspector under the node's properties (or alone there), with the same head
 * as every side panel: its name and count, then Copy, Clear and hide.
 */
interface LogConsoleProps {
  logs: LogEntry[];
  onClear?: () => void;
  onClose: () => void;
}

interface LogEntryRowProps {
  log: LogEntry;
  formatTime: (timestamp: number) => string;
}

// Memoized log entry component for better performance with large log lists
const LogEntryRow = memo(function LogEntryRow({ log, formatTime }: LogEntryRowProps) {
  const isOutput = log.source === 'Output';
  const isNode = log.source !== 'System' && !isOutput;
  // Where the line came from, as a pill: the run's output, a node, or the system.
  const pill = isOutput ? 'oaiy-pill ok' : isNode ? 'oaiy-pill accent' : 'oaiy-pill info';
  const kind = isOutput ? 'output' : log.type || 'info';

  return (
    <div className={`oaiy-log-line ${kind}`}>
      <time>{formatTime(log.timestamp)}</time>
      <span className={pill}>{isOutput ? 'OUT' : isNode ? 'NODE' : 'SYS'}</span>
      <span>
        {log.message}
        {/* Blinking cursor for streaming */}
        {log.isStreaming && <span className="oaiy-log-cursor" />}
      </span>
    </div>
  );
});

function LogConsole({ logs, onClear, onClose }: LogConsoleProps) {
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
      <div className="oaiy-side-head">
        <h2>
          Log <small>{logs.length}</small>
        </h2>
        <div className="oaiy-side-tools">
          {logs.length > 0 && <CopyButton text={logsAsText} label="Copy" size="sm" />}
          {onClear && logs.length > 0 && (
            <button type="button" onClick={onClear} className="btn btn-ghost btn-sm">
              Clear
            </button>
          )}
          <button type="button" onClick={onClose} className="oaiy-icon-btn" title="Hide the log" aria-label="Hide execution log">
            <X size={15} />
          </button>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-label="Execution log"
        className="oaiy-log"
      >
        {logs.length === 0 ? (
          <div className="oaiy-empty bare">
            <ScrollText size={20} />
            <p className="oaiy-empty-title">Nothing yet</p>
            <p className="oaiy-empty-text">Run the flow, and what it says shows here.</p>
          </div>
        ) : (
          logs.map((log) => <LogEntryRow key={log.id} log={log} formatTime={formatTime} />)
        )}
      </div>
    </>
  );
}

export default memo(LogConsole);
