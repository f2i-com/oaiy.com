import { useCallback, useEffect, useState } from 'react';
import { CircleAlert, RefreshCw, Trash2 } from 'lucide-react';
import { bridge, type RunHistory, type RunRecord, type RunStatus } from './api';
import DeadLetters from './DeadLetters';
import { useToast } from './Toasts';
import { invalidate, peek, put } from './useCached';

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

/**
 * What an empty failures list actually proves — which is less than it looks.
 *
 * This used to interpolate `total`, but that is the whole ledger (`l.len()`),
 * queued and running rows included, so it asserted that runs which had not
 * started had "finished cleanly". Nothing ages a queued run out — the desktop
 * worker deliberately leaves `queued`-mode runs for an external claimer — so
 * with no claimer running, that false all-clear is permanent, not a blink, and
 * this sentence is the only thing on the failures screen that speaks to overall
 * run health. byStatus is on the same response and breaks the ledger down per
 * status, so `succeeded` is the entry that is actually about finished work —
 * the map itself tallies every state, live runs included.
 */
function noFailuresMessage(data: RunHistory): string {
  if (data.total === 0) return 'No flow has run on this machine yet.';
  const counts = data.byStatus;
  // A response (or a cached snapshot) without byStatus leaves total as the only
  // number there is; saying 0 would be a worse answer than the old overcount.
  if (!counts) return `Nothing has failed — all ${data.total} recorded runs finished cleanly.`;
  const clean = counts.succeeded ?? 0;
  const inFlight = (counts.queued ?? 0) + (counts.running ?? 0);
  if (inFlight === 0) return `Nothing has failed — all ${clean} recorded runs finished cleanly.`;
  return `Nothing has failed — ${clean} finished cleanly, ${inFlight} still queued or running.`;
}

export default function RunsPanel() {
  const [filter, setFilter] = useState<Filter>('failures');
  // Keyed BY FILTER. One shared key meant a revisit could paint the cached
  // "all" snapshot under the "Failed runs" heading — a list of green succeeded
  // badges below a title claiming they failed. And if the refresh then failed,
  // it stayed that way indefinitely behind an error banner that said nothing
  // about the list being wrong.
  const [data, setData] = useState<RunHistory | null>(
    () => peek(`runHistory:${filter}`) ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const toast = useToast();

  const refresh = useCallback(async () => {
    try {
      const next = await bridge.runs(filter === 'failures' ? FAILURE_STATES : 'all', 50);
      setData(next);
      put(`runHistory:${filter}`, next);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [filter]);

  useEffect(() => {
    // Repaint from THIS filter's cache the moment the filter changes, so the
    // heading and the rows can never disagree while the refetch is in flight.
    setData(peek<RunHistory>(`runHistory:${filter}`) ?? null);
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh, filter]);

  const clear = async () => {
    // Spell out both halves: what survives, and the one consequence that is
    // not obvious — a consumer retrying an old idempotency key gets a fresh
    // run instead of the recorded result, so identical work can run again.
    if (
      !confirm(
        'Clear finished runs from this history?\n\nQueued and running work is kept. This cannot be undone, and an app that retries a request from before the clear will run it again rather than get the old result.',
      )
    )
      return;
    setClearing(true);
    try {
      const { cleared } = await bridge.clearRuns();
      // Drop BOTH filters' caches: the other tab's cached page still lists rows
      // that no longer exist, and switching to it would show them as current.
      invalidate('runHistory:failures');
      invalidate('runHistory:all');
      await refresh();
      toast.push({
        kind: 'success',
        title: cleared === 0 ? 'Nothing to clear' : `Cleared ${cleared} finished run${cleared === 1 ? '' : 's'}`,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setClearing(false);
    }
  };

  const counts = data?.byStatus ?? {};
  // Summed over the same constant the query uses. Hand-listing the states here
  // omitted `cancelled`, which the filter does fetch, so the tab read
  // "Failures (2)" above a list of five rows — a count and a list drawn from one
  // response, disagreeing, on the panel whose whole job is being trustworthy
  // about failures. Worse at the zero end: with only cancelled runs the badge
  // vanished entirely and the tab read a bare "Failures" over a full list.
  const failed = FAILURE_STATES.reduce((n, s) => n + (counts[s] ?? 0), 0);
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
          {/* Disabled while empty rather than hidden: a button that appears and
              vanishes with the list is harder to find than one that is always
              in the same place. */}
          <button
            className="btn-tiny btn-danger"
            onClick={() => void clear()}
            disabled={clearing || !data || data.total === 0}
            title="Clear finished runs"
            aria-label="Clear finished runs"
          >
            <Trash2 size={13} />
          </button>
        </div>

        {data === null ? (
          <p className="form-hint">Loading runs…</p>
        ) : runs.length === 0 ? (
          <div className="empty-state">
            {filter === 'failures' ? noFailuresMessage(data) : 'No runs recorded yet.'}
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
