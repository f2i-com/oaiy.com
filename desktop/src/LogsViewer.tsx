import { useEffect, useRef, useState } from 'react';
import type { LogLine } from './api';

/**
 * Reusable log tail viewer. Polls a loader fn every 1.5s and auto-scrolls
 * to the bottom when new lines arrive. Closing it stops the poller, so
 * idle services don't burn CPU.
 *
 * One component, two consumers: service install/run logs (via
 * services.logs) and Python install/venv logs (via python.logs). The
 * shape is the same so we don't need bespoke components per source.
 */
interface Props {
  /** Loader called every poll tick. Throw on transient errors — we
   *  swallow and keep polling. */
  load: () => Promise<LogLine[]>;
  /** Pretty header text. */
  title: string;
  onClose?: () => void;
}

export default function LogsViewer({ load, title, onClose }: Props) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      // Tray app: skip polling while the window is hidden — nobody's watching.
      if (document.hidden) return;
      try {
        const next = await load();
        if (!cancelled) {
          setLines(next);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    };
    tick();
    const id = setInterval(tick, 1500);
    const onVis = () => {
      if (!document.hidden) tick();
    };
    document.addEventListener('visibilitychange', onVis);
    return () => {
      cancelled = true;
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [load]);

  // Move focus into the panel when it opens, so keyboard users land here.
  useEffect(() => {
    rootRef.current?.focus();
  }, []);

  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    // Auto-scroll only when user is at (or very near) the bottom — lets
    // them scroll up to read older lines without being yanked down.
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    autoScrollRef.current = atBottom;
  };

  return (
    <div
      className="logs-panel"
      ref={rootRef}
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.key === 'Escape' && onClose) onClose();
      }}
    >
      <div className="logs-header">
        <span className="logs-title">{title}</span>
        {onClose && (
          <button className="logs-close" onClick={onClose} aria-label="close logs">
            ✕
          </button>
        )}
      </div>
      <div
        className="logs-body"
        ref={scrollRef}
        onScroll={onScroll}
        role="log"
        aria-live="polite"
        aria-label={`${title} output`}
      >
        {error && <div className="logs-error">⚠ {error}</div>}
        {lines.length === 0 && !error && (
          <div className="logs-empty">No output yet.</div>
        )}
        {lines.map((line, i) => (
          <div
            key={`${line.timestamp}-${i}`}
            className={`logs-line logs-line-${line.stream}`}
          >
            <span className="logs-time">
              {new Date(line.timestamp).toLocaleTimeString()}
            </span>
            <span className="logs-text">{line.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
