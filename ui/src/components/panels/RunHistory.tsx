import { useState, useCallback, useEffect, useRef } from 'react';
import type { RunRecord } from 'oaiy-core';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface RunHistoryProps {
  runs: RunRecord[];
  onReplayRun?: (run: RunRecord) => void;
  onClearHistory: () => void;
  isOpen: boolean;
  onClose: () => void;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return 'Today';
  } else if (date.toDateString() === yesterday.toDateString()) {
    return 'Yesterday';
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function RunHistory({
  runs,
  onReplayRun,
  onClearHistory,
  isOpen,
  onClose,
}: RunHistoryProps) {
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // On narrow viewports this panel is an overlay drawer; on md+ it's an
  // inline sidebar. Only the overlay case should be a modal (focus trap +
  // dialog role) — trapping/labelling the inline sidebar would be wrong.
  const [isOverlay, setIsOverlay] = useState(
    () => typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(max-width: 767px)').matches
      : false,
  );

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia('(max-width: 767px)');
    const handler = (e: MediaQueryListEvent) => setIsOverlay(e.matches);
    setIsOverlay(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Escape closes the panel (harmless on md+; matches drawer behaviour).
  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onClose]);

  // Trap focus only in the overlay (mobile) case.
  useFocusTrap(panelRef, isOpen && isOverlay);

  const handleToggleExpand = useCallback((runId: string) => {
    setExpandedRunId((prev) => (prev === runId ? null : runId));
  }, []);

  const getStatusIcon = (status: RunRecord['status']) => {
    switch (status) {
      case 'completed':
        return (
          <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'aborted':
        return (
          <svg className="w-4 h-4 text-yellow-600 dark:text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
          </svg>
        );
      case 'running':
        return (
          <svg className="w-4 h-4 text-blue-600 dark:text-blue-400 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        );
    }
  };

  const getStatusLabel = (status: RunRecord['status']) => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'error': return 'Failed';
      case 'aborted': return 'Aborted';
      case 'running': return 'Running';
    }
  };

  // Group runs by date
  const groupedRuns = runs.reduce<Record<string, RunRecord[]>>((acc, run) => {
    const dateKey = formatDate(run.startedAt);
    if (!acc[dateKey]) acc[dateKey] = [];
    acc[dateKey].push(run);
    return acc;
  }, {});

  if (!isOpen) return null;

  return (
    <>
      {/* Mobile backdrop */}
      <div
        className="md:hidden fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        ref={panelRef}
        role={isOverlay ? 'dialog' : undefined}
        aria-modal={isOverlay ? true : undefined}
        aria-label={isOverlay ? 'Run History' : undefined}
        className={`
        fixed md:relative inset-y-0 right-0 z-50 md:z-auto
        w-[85vw] sm:w-80 max-w-80 bg-white dark:bg-slate-800 border-l border-slate-200 dark:border-slate-700 flex flex-col
        transform transition-transform duration-200 ease-in-out
        ${isOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-0'}
      `}>
        {/* Header */}
        <div className="p-3 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Run History</h2>
          <div className="flex items-center gap-2">
            {runs.length > 0 && (
              <button
                onClick={() => {
                  if (confirm(`Clear all ${runs.length} run history entr${runs.length === 1 ? 'y' : 'ies'}?`)) onClearHistory();
                }}
                className="text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              >
                Clear
              </button>
            )}
            <button
              onClick={onClose}
              className="md:hidden p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded transition-colors"
              aria-label="Close panel"
            >
              <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Run List */}
        <div className="flex-1 overflow-y-auto">
          {runs.length === 0 ? (
            <div className="p-4 text-center text-slate-500 text-sm">
              No runs yet. Execute a flow to see history.
            </div>
          ) : (
            <div className="p-2">
              {Object.entries(groupedRuns).map(([date, dateRuns]) => (
                <div key={date} className="mb-4">
                  <div className="text-xs font-medium text-slate-500 px-2 mb-1">{date}</div>
                  <ul className="space-y-1">
                    {dateRuns.map((run) => (
                      <li key={run.id}>
                        <button
                          onClick={() => handleToggleExpand(run.id)}
                          className={`
                            w-full p-2 rounded text-left transition-colors
                            ${expandedRunId === run.id
                              ? 'bg-slate-100 dark:bg-slate-700'
                              : 'hover:bg-slate-100 dark:hover:bg-slate-700/50'
                            }
                          `}
                        >
                          <div className="flex items-center gap-2">
                            {getStatusIcon(run.status)}
                            <span className="flex-1 text-sm text-slate-800 dark:text-slate-200 truncate">
                              {run.flowName}
                            </span>
                            <span className="text-xs text-slate-500">
                              {formatTime(run.startedAt)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 mt-1 ml-6">
                            <span className={`text-xs ${
                              run.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                              run.status === 'error' ? 'text-red-600 dark:text-red-400' :
                              run.status === 'aborted' ? 'text-yellow-600 dark:text-yellow-400' :
                              'text-blue-600 dark:text-blue-400'
                            }`}>
                              {getStatusLabel(run.status)}
                            </span>
                            {run.duration !== undefined && (
                              <span className="text-xs text-slate-500">
                                {formatDuration(run.duration)}
                              </span>
                            )}
                          </div>
                        </button>

                        {/* Expanded Details */}
                        {expandedRunId === run.id && (
                          <div className="bg-slate-100 dark:bg-slate-900 rounded mx-1 p-2 mt-1 text-xs">
                            {/* Node Timeline */}
                            {run.nodeStatuses.length > 0 && (
                              <div className="mb-3">
                                <div className="text-slate-400 mb-1">Nodes:</div>
                                <div className="space-y-1 max-h-32 overflow-y-auto">
                                  {run.nodeStatuses.map((node, i) => (
                                    <div key={i} className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
                                      <span className={`w-2 h-2 rounded-full ${
                                        node.status === 'completed' ? 'bg-green-400' :
                                        node.status === 'error' ? 'bg-red-400' :
                                        node.status === 'skipped' ? 'bg-slate-500' :
                                        'bg-blue-400'
                                      }`} />
                                      <span className="flex-1 truncate">{node.nodeId}</span>
                                      {node.completedAt && node.startedAt && (
                                        <span className="text-slate-500">
                                          {formatDuration(node.completedAt - node.startedAt)}
                                        </span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Inputs */}
                            {run.inputs && Object.keys(run.inputs).length > 0 && (
                              <div className="mb-2">
                                <div className="text-slate-400 mb-1">Inputs:</div>
                                <pre className="bg-white dark:bg-slate-800 p-1.5 rounded text-[10px] text-slate-700 dark:text-slate-300 overflow-x-auto max-h-20">
                                  {JSON.stringify(run.inputs, null, 2)}
                                </pre>
                              </div>
                            )}

                            {/* Outputs */}
                            {run.outputs && Object.keys(run.outputs).length > 0 && (
                              <div className="mb-2">
                                <div className="text-slate-400 mb-1">Outputs:</div>
                                <pre className="bg-white dark:bg-slate-800 p-1.5 rounded text-[10px] text-slate-700 dark:text-slate-300 overflow-x-auto max-h-20">
                                  {JSON.stringify(run.outputs, null, 2)}
                                </pre>
                              </div>
                            )}

                            {/* Actions */}
                            <div className="flex gap-2 mt-2">
                              {onReplayRun && (
                                <button
                                  onClick={() => onReplayRun(run)}
                                  className="flex-1 px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-500 transition-colors"
                                >
                                  Replay
                                </button>
                              )}
                              <button
                                onClick={() => {
                                  const blob = new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' });
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement('a');
                                  a.href = url;
                                  a.download = `run-${run.id}.json`;
                                  a.click();
                                  URL.revokeObjectURL(url);
                                }}
                                className="flex-1 px-2 py-1 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded text-xs hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                              >
                                Export
                              </button>
                            </div>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
