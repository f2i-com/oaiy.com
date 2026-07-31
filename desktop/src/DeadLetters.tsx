import { useCallback, useEffect, useState } from 'react';
import { Inbox, RotateCw, X } from 'lucide-react';
import { bridge, type DeadLetter } from './api';

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
  const [busy, setBusy] = useState<string | null>(null);
  // Section-level, not per row: a successful redrive REMOVES its row, so a note
  // pinned to it would vanish with the thing it was confirming.
  const [note, setNote] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await bridge.deadLetters();
      setItems(r.deadLetters);
    } catch {
      // The panel above already surfaces a dead bridge; don't say it twice.
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const redrive = async (item: DeadLetter) => {
    setBusy(item.id);
    setNote(null);
    try {
      const r = await bridge.redrive(item.id);
      // A redrive that fails again is a legitimate outcome, not an error — say
      // what happened rather than just "failed".
      setNote(
        r.reserved
          ? `Redrive reserved a run for ${item.event}.`
          : `${item.event} is still not running — ${r.outcomes.join('; ') || 'no binding matched'}`,
      );
      await refresh();
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const dismiss = async (item: DeadLetter) => {
    setBusy(item.id);
    try {
      await bridge.dismissDeadLetter(item.id);
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  // Nothing here is the normal state — an empty section would be noise.
  if (items === null || items.length === 0) return null;

  return (
    <section className="service-section">
      <div className="section-title-row">
        <h3 className="section-title">
          <Inbox size={14} /> Events that produced no work ({items.length})
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
                {d.reason.kind === 'shed' ? 'dropped' : 'not run'}
              </span>
              <strong className="run-flow">{d.event}</strong>
              <span className="run-meta">
                {d.source}
                {d.attempts > 0 ? ` · ${d.attempts} redrive${d.attempts === 1 ? '' : 's'}` : ''}
              </span>
              <span className="run-when">{when(d.recordedAtMs)}</span>
            </div>

            <p className="run-error">
              {d.reason.kind === 'shed'
                ? 'The host event queue was full, so this was dropped before dispatch.'
                : (d.reason.detail ?? 'No binding reserved a run.')}
            </p>

            {d.lastOutcome && (
              <p className="form-hint">Last redrive: {d.lastOutcome}</p>
            )}

            <div className="form-actions">
              <button className="btn-tiny" disabled={busy === d.id} onClick={() => void redrive(d)}>
                <RotateCw size={13} /> {busy === d.id ? 'Redriving…' : 'Redrive'}
              </button>
              <button className="btn-tiny" disabled={busy === d.id} onClick={() => void dismiss(d)}>
                <X size={13} /> Dismiss
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
