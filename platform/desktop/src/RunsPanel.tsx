import { useCallback, useEffect, useState } from 'react';
import { CircleAlert, RefreshCw } from 'lucide-react';
import { bridge, type RunHistory, type RunRecord, type RunStatus } from './api';
import DeadLetters from './DeadLetters';
import { peek, put } from './useCached';

/**
 * Run history — what happened, and why it failed.
 *
 * The ledger has always kept every run (20k of them, durable across restarts),
 * but until now the only way in was `GET /api/bridge/runs/:id`. That made a
 * failure diagnosable only by whoever happened to be holding its id: if a
 * trigger fired a flow at 3am and it failed, nothing on this machine would ever
 * tell you. This panel is the read surface over what was already recorded.
 *
 * Failures are the default filter for the same reason — a list dominated by
 * successes buries the one row anyone opened this screen to find.
 */

const POLL_MS = 5000;

const FAILURE_STATES: RunStatus[] = ['failed', 'timed_out', 'cancelled'];

type Filter = 'failures' | 'all';

/** Status → the badge class the design system already defines. */
function badgeFor(status: RunStatus): string {
  switch (status) {
    case 'succeeded':
      return 'badge badge-ok';
    case 'failed':
    case 'timed_out':
      return 'badge badge-err';
    case 'running':
    case 'queued':
      return 'badge badge-pending';
    default:
      return 'badge badge-neutral';
  }
}

const LABEL: Record<RunStatus, string> = {
  queued: 'queued',
  running: 'running',
  succeeded: 'succeeded',
  failed: 'failed',
  timed_out: 'timed out',
  cancelled: 'cancelled',
};

/** Local time, seconds included — run history is read at minute granularity. */
function when(iso: string | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/** How long it ran, from the two timestamps the ledger already stores. */
function duration(rec: RunRecord): string | null {
  if (!rec.startedAt || !rec.finishedAt) return null;
  const ms = new Date(rec.finishedAt).getTime() - new Date(rec.startedAt).getTime();
  if (!Number.isFinite(ms) || ms < 0) return null;
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export default function RunsPanel() {
  const [filter, setFilter] = useState<Filter>('failures');
  const [data, setData] = useState<RunHistory | null>(() => peek('runHistory') ?? null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await bridge.runs(filter === 'failures' ? FAILURE_STATES : 'all', 50);
      setData(next);
      put('runHistory', next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [filter]);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const counts = data?.byStatus ?? {};
  const failed = (counts.failed ?? 0) + (counts.timed_out ?? 0);
  const runs = data?.runs ?? [];

  return (
    <div className="panel">
      {error && <div className="banner banner-err">⚠ {error}</div>}

      {/* Above the run list: an event that never became a run is a worse
          failure than a run that failed, because nothing recorded it tried. */}
      <DeadLetters />

      <section className="service-section">
        <div className="section-title-row">
          <h3 className="section-title">
            {filter === 'failures' ? 'Failed runs' : 'All runs'}
          </h3>
          <div className="seg">
            <button
              className={filter === 'failures' ? 'active' : ''}
              onClick={() => setFilter('failures')}
            >
              Failures{failed > 0 ? ` (${failed})` : ''}
            </button>
            <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>
              All{data ? ` (${data.total})` : ''}
            </button>
          </div>
          <button className="btn-tiny" onClick={() => void refresh()} title="Refresh now">
            <RefreshCw size={13} />
          </button>
        </div>

        {data === null ? (
          <p className="form-hint">Loading runs…</p>
        ) : runs.length === 0 ? (
          <div className="empty-state">
            {filter === 'failures'
              ? data.total === 0
                ? 'No flow has run on this machine yet.'
                : `Nothing has failed — all ${data.total} recorded runs finished cleanly.`
              : 'No runs recorded yet.'}
          </div>
        ) : (
          <ul className="run-list">
            {runs.map((r) => {
              const open = expanded === r.runId;
              const dur = duration(r);
              return (
                <li key={r.runId} className="run-row">
                  <div className="run-head">
                    <span className={badgeFor(r.status)}>{LABEL[r.status]}</span>
                    <strong className="run-flow">{r.flowId ?? '(inline graph)'}</strong>
                    <span className="run-meta">
                      {r.callerProduct}
                      {r.triggerEvent ? ` · ${r.triggerEvent}` : ''}
                      {dur ? ` · ${dur}` : ''}
                    </span>
                    <span className="run-when">{when(r.finishedAt ?? r.reservedAt)}</span>
                  </div>

                  {/* The reason, on the row, unexpanded. Making someone click to
                      find out why a run failed is the problem this panel fixes. */}
                  {r.error && (
                    <p className="run-error">
                      <CircleAlert size={13} /> <code>{r.error.code}</code> {r.error.message}
                    </p>
                  )}

                  <button
                    className="btn-tiny run-more"
                    onClick={() => setExpanded(open ? null : r.runId)}
                  >
                    {open ? 'Less' : 'Details'}
                  </button>

                  {open && (
                    <dl className="run-detail">
                      <dt>Run</dt>
                      <dd>
                        <code>{r.runId}</code>
                      </dd>
                      <dt>Correlation</dt>
                      <dd>
                        <code>{r.correlationId}</code>
                      </dd>
                      <dt>Mode</dt>
                      <dd>
                        {r.mode}
                        {r.runtime ? ` · ${r.runtime}` : ''}
                      </dd>
                      <dt>Reserved</dt>
                      <dd>{when(r.reservedAt)}</dd>
                      {r.startedAt && (
                        <>
                          <dt>Started</dt>
                          <dd>{when(r.startedAt)}</dd>
                        </>
                      )}
                      {r.error?.nodeId && (
                        <>
                          <dt>Failed at</dt>
                          <dd>
                            <code>{r.error.nodeId}</code>
                          </dd>
                        </>
                      )}
                      {r.error?.capability && (
                        <>
                          <dt>Capability</dt>
                          <dd>
                            <code>{r.error.capability}</code>
                          </dd>
                        </>
                      )}
                      {r.error?.detail && (
                        <>
                          <dt>Detail</dt>
                          <dd>
                            <pre className="run-detail-text">{r.error.detail}</pre>
                          </dd>
                        </>
                      )}
                    </dl>
                  )}
                </li>
              );
            })}
          </ul>
        )}

        {data && data.runs.length >= 50 && (
          <p className="form-hint">
            Showing the 50 most recent. {data.total} runs are retained in total.
          </p>
        )}
      </section>
    </div>
  );
}
