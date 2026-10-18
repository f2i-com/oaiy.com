import { useCallback, useState } from 'react';
import { Inbox, RotateCw, X } from 'lucide-react';
import { bridge, type DeadLetter } from './api';
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

function when(ms: number): string {
  const d = new Date(ms);
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString();
}

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

  const refresh = useCallback(async () => {
    try {
      const r = await bridge.deadLetters();
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

  // Nothing here is the normal state — an empty section would be noise; but what the last redrive did stays said
  // after its entry has gone.
  if (items === null || (items.length === 0 && !note)) return null;
  if (items.length === 0) {
    return (
      <section className="service-section">
        <div className="datadir-note">
          <span>{note}</span>
          <button className="btn-tiny" onClick={() => setNote(null)}>
            <X size={13} /> Close
          </button>
        </div>
      </section>
    );
  }

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
      {note && <div className="datadir-note"><span>{note}</span></div>}

      <ul className="run-list">
        {items.map((d) => (
          <li key={d.id} className="run-row">
            <div className="run-head">
              <span className="badge badge-err">
                {d.reason.kind === 'shed' ? 'dropped' : d.reason.kind === 'not_delivered' ? 'not sent to the linked account' : 'not run'}
              </span>
              <strong className="run-flow">{d.event}</strong>
              <span className="run-meta">
                {d.source}
                {d.attempts > 0 ? ` · ${d.attempts} redrive${d.attempts === 1 ? '' : 's'}` : ''}
              </span>
              <span className="run-when">{when(d.recordedAtMs)}</span>
            </div>

            <p className="run-error">
              <span className="run-error-text">
                {d.reason.kind === 'shed'
                  ? 'The host event queue was full, so this was dropped before dispatch.'
                  : (d.reason.detail ?? 'No binding reserved a run.')}
              </span>
            </p>

            {d.lastOutcome && (
              <p className="form-hint">Last redrive: {d.lastOutcome}</p>
            )}

            <div className="form-actions">
              <button className="btn-tiny" disabled={!!busy[d.id]} onClick={() => void redrive(d)}>
                <RotateCw size={13} /> {busy[d.id] === 'redrive' ? 'Redriving…' : 'Redrive'}
              </button>
              <button className="btn-tiny" disabled={!!busy[d.id]} onClick={() => void dismiss(d)}>
                <X size={13} /> {busy[d.id] === 'dismiss' ? 'Dismissing…' : 'Dismiss'}
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
