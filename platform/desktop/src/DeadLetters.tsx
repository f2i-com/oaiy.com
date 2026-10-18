import { useCallback, useRef, useState } from 'react';
import { ChevronDown, Inbox, Loader2, RotateCw, X } from 'lucide-react';
import { bridge, type DeadLetter } from './api';
import { JsonBlock, fullWhen, whenText } from './RunParts';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * Events that arrived and produced no work.
 *
 * Two things put an event here: the host shed it under a flood (the queue is
 * bounded on purpose), or every binding that matched failed to fire for an
 * operational reason. Both used to be invisible — one logged to a ring that was
 * itself overwriting, the other forgotten on restart.
 *
 * Redrive re-dispatches against the CURRENT bindings rather than replaying a
 * decision, so fixing the binding and redriving works, and a loop-guard refusal
 * is refused again rather than forced through.
 */

const POLL_MS = 6000;

const KIND: Record<DeadLetter['reason']['kind'], string> = {
  shed: 'dropped',
  not_reserved: 'not run',
  not_delivered: 'not sent to the linked account',
};

export default function DeadLetters() {
  const [items, setItems] = useState<DeadLetter[] | null>(null);
  /** How many there are in all: the list holds the newest hundred. */
  const [total, setTotal] = useState(0);
  /** What is being asked about each entry, by its id: Redrive says "Redriving…" for a redrive alone. One entry's
   *  answer frees that entry only, so another's slow redrive cannot be pressed again meanwhile. */
  const [busy, setBusy] = useState<Record<string, 'redrive' | 'dismiss'>>({});
  const mark = (id: string, action: 'redrive' | 'dismiss' | null) =>
    setBusy((prev) => {
      const next = { ...prev };
      if (action) next[id] = action;
      else delete next[id];
      return next;
    });
  // Section-level, not per row: a successful redrive REMOVES its row, so a note
  // pinned to it would vanish with the thing it was confirming.
  const [note, setNote] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  // The latest request's number: a poll asked before a redrive, answered after it, would put the redriven entry back.
  const asked = useRef(0);

  const refresh = useCallback(async () => {
    const mine = ++asked.current;
    try {
      const r = await bridge.deadLetters();
      if (mine !== asked.current) return;
      setItems(r.deadLetters);
      setTotal(r.total ?? r.deadLetters.length);
    } catch {
      // The panel above already surfaces a dead bridge; don't say it twice.
    }
  }, []);

  // (not while the window is hidden in the tray)
  useVisiblePoll(() => void refresh(), POLL_MS);

  const redrive = async (item: DeadLetter) => {
    mark(item.id, 'redrive');
    setNote(null);
    try {
      const r = await bridge.redrive(item.id);
      // A redrive that fails again is a legitimate outcome, not an error — say
      // what happened rather than just "failed".
      setNote(
        r.reserved
          ? item.reason.kind === 'not_delivered'
            ? `${item.event} is waiting to be sent to the linked account again.`
            : `Redrive reserved a run for ${item.event}.`
          : `${item.event} is still not running — ${r.outcomes.join('; ') || 'no binding matched'}`,
      );
      await refresh();
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      mark(item.id, null);
    }
  };

  const dismiss = async (item: DeadLetter) => {
    if (!window.confirm(`Dismiss ${item.event}? It is taken off this list and never run.`)) return;
    mark(item.id, 'dismiss');
    setNote(null);
    try {
      await bridge.dismissDeadLetter(item.id);
      await refresh();
    } catch (e) {
      setNote(`${item.event} could not be dismissed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      mark(item.id, null);
    }
  };

  const noteLine = note && (
    <div className="datadir-note datadir-row" role="status">
      <span>{note}</span>
      <button className="btn-tiny" onClick={() => setNote(null)}>
        <X size={13} /> Close
      </button>
    </div>
  );

  // Nothing here is the normal state — an empty section would be noise; but what the last redrive did stays said
  // after its entry has gone.
  if (items === null || (items.length === 0 && !note)) return null;
  if (items.length === 0) {
    return <section className="service-section">{noteLine}</section>;
  }

  const now = Date.now();
  return (
    <section className="service-section">
      <div className="section-title-row">
        <h3 className="section-title">
          <Inbox size={14} /> Events that produced no work ({total > items.length ? `${items.length} of ${total}` : items.length})
        </h3>
      </div>
      <p className="form-hint">
        These arrived but never became a run. Fix the binding, then redrive — loop
        guards still apply, so a redrive can legitimately fail again.
      </p>
      {noteLine}

      <ul className="run-list">
        {items.map((d) => {
          const shown = open === d.id;
          return (
            <li key={d.id} className={`run-row${shown ? ' is-open' : ''}`} data-run-id={d.id}>
              <div className="run-line">
                <button type="button" className="run-head dead-head" aria-expanded={shown} aria-controls={`dead-${d.id}`} onClick={() => setOpen(shown ? null : d.id)}>
                  <span className="run-state">
                    <span className="badge badge-err">{KIND[d.reason.kind] ?? d.reason.kind}</span>
                  </span>
                  <span className="run-name">
                    <strong className="run-flow">{d.event}</strong>
                  </span>
                  <span className="run-meta">
                    {d.source}
                    {d.attempts > 0 ? ` · ${d.attempts} redrive${d.attempts === 1 ? '' : 's'}` : ''}
                  </span>
                  <time className="run-when" title={fullWhen(d.recordedAtMs)}>
                    {whenText(d.recordedAtMs, now)}
                  </time>
                  <ChevronDown size={14} className="run-chevron" aria-hidden />
                </button>
                <button className="btn-tiny" disabled={!!busy[d.id]} onClick={() => void redrive(d)}>
                  {busy[d.id] === 'redrive' ? <Loader2 size={13} className="spin" /> : <RotateCw size={13} />} {busy[d.id] === 'redrive' ? 'Redriving…' : 'Redrive'}
                </button>
                <button className="btn-tiny" disabled={!!busy[d.id]} onClick={() => void dismiss(d)}>
                  <X size={13} /> {busy[d.id] === 'dismiss' ? 'Dismissing…' : 'Dismiss'}
                </button>
              </div>

              <div className="run-error">
                <p>
                  <span className="run-error-text">
                    {d.reason.kind === 'shed'
                      ? 'The host event queue was full, so this was dropped before dispatch.'
                      : (d.reason.detail ?? 'No binding reserved a run.')}
                  </span>
                </p>
              </div>

              {d.lastOutcome && <p className="run-held form-hint">Last redrive: {d.lastOutcome}</p>}

              {shown && (
                <div className="run-body" id={`dead-${d.id}`}>
                  <dl className="run-detail">
                    <dt>Arrived</dt>
                    <dd>{fullWhen(d.recordedAtMs)}</dd>
                    {d.lastAttemptMs && (
                      <>
                        <dt>Last redrive</dt>
                        <dd>{fullWhen(d.lastAttemptMs)}</dd>
                      </>
                    )}
                    <dt>From</dt>
                    <dd>
                      <code>{d.source}</code>
                    </dd>
                    <dt>Entry</dt>
                    <dd>
                      <code>{d.id}</code>
                    </dd>
                  </dl>
                  <JsonBlock value={d.envelope} summary="The event as it arrived" />
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
