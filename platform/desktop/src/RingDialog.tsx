import { useCallback, useEffect, useRef, useState } from 'react';
import { Phone, PhoneOff } from 'lucide-react';
import { ring as api, type ActiveRing, type RingAction, type RingNotice } from './api';
import { readableNumber } from './contactsModel';
import { openSetup } from './useSetupState';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * A caller wants to speak to you. While the receptionist is trying to reach the
 * owner, this desktop rings (a notification, and this dialog): who is calling, what
 * they last said, and how long it rings. The owner answers on a Companion (this
 * computer cannot carry a call's audio, so there is nothing to accept here). What the
 * dialog offers is to decline and have the receptionist take a message, which asks the
 * phone to withdraw the request (it shows as stopping until the phone answers, a couple
 * of seconds at most), and "Not now", which only puts the box away: the devices go on ringing.
 */

/** How often the desktop is asked whether anyone is ringing. */
export const POLL_MS = 1_000;

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Whole seconds left to ring at `now` (the desktop's clock), never below zero. */
export function secondsLeft(ring: Pick<ActiveRing, 'expiresAt'>, now: number): number {
  return Math.max(0, Math.ceil((ring.expiresAt - now) / 1000));
}

export default function RingDialog({ on = true }: { on?: boolean }) {
  const [rings, setRings] = useState<ActiveRing[]>([]);
  /** Callers who asked for the owner when no device was set up to take a transfer: told, and dismissed by the owner. */
  const [notices, setNotices] = useState<RingNotice[]>([]);
  const [dismissed, setDismissed] = useState<string[]>([]);
  /** The desktop's clock less this window's, when it last answered: the countdown does not trust this window's clock. */
  const offset = useRef(0);
  const [tick, setTick] = useState(0);
  const [busy, setBusy] = useState<RingAction | null>(null);
  /** Rings the owner put away with "Not now": not shown again, though they go on ringing on their devices. */
  const [putAway, setPutAway] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const look = useCallback(() => {
    if (!on) return;
    api.active().then(
      ({ rings: list, notices: told }) => {
        if (list.length) offset.current = list[0].now - Date.now();
        setRings(list);
        setNotices(told);
      },
      () => {
        /* the desktop is away: what was shown stays until it says */
      },
    );
  }, [on]);
  useVisiblePoll(look, POLL_MS);
  useEffect(() => {
    if (!on) {
      setRings([]);
      setNotices([]);
    }
  }, [on]);

  // The countdown: twice a second while something rings.
  useEffect(() => {
    if (!rings.length) return;
    const id = window.setInterval(() => setTick((n) => n + 1), 500);
    return () => window.clearInterval(id);
  }, [rings.length]);

  const now = Date.now() + offset.current;
  const ringing = rings.filter((r) => r.expiresAt > now && !putAway.includes(r.id));
  const shown = ringing[0];

  const act = async (action: RingAction) => {
    if (!shown) return;
    setBusy(action);
    setError(null);
    try {
      // The phone is asked to withdraw the request: the ring stays, stopping, until it answers (the next look shows it).
      await api.respond(shown.id, action);
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(null);
      look();
    }
  };

  const told = notices.filter((n) => !dismissed.includes(n.id));
  const dismiss = (id: string) => {
    setDismissed((ids) => [...ids.slice(-20), id]);
    void api.dismissNotice(id).catch(() => {
      /* it was already gone */
    });
  };
  const noticeStack = told.length > 0 && (
    <div className="ring-notices" role="status" aria-live="polite">
      {told.map((n) => (
        <div className="ring-notice" key={n.id}>
          <p>
            <strong>{n.callerName || readableNumber(n.callerNumber) || 'A caller who hid their number'}</strong> asked for you.
          </p>
          <p>{n.text}</p>
          <div className="ring-notice-actions">
            <button type="button" className="btn-tiny" onClick={() => openSetup({ plugin: 'aokie', step: 'pair' })}>
              Set up a Companion
            </button>
            <button type="button" className="btn-tiny" onClick={() => dismiss(n.id)}>
              Dismiss
            </button>
          </div>
        </div>
      ))}
    </div>
  );

  void tick;
  if (!on) return null;
  if (!shown) return noticeStack || null;
  const left = secondsLeft(shown, now);
  const total = Math.max(1, Math.round((shown.expiresAt - shown.startedAt) / 1000));
  const who = shown.callerName || readableNumber(shown.callerNumber) || 'A caller who hid their number';
  const said = shown.said.filter((s) => s.trim()).slice(-2);
  const note = shown.note;
  return (
    <>
    {noticeStack}
    <div className="ring-overlay">
      <div className="ring-dialog" role="alertdialog" aria-modal="true" aria-labelledby="ring-title" aria-describedby="ring-who">
        <h2 id="ring-title">
          <Phone size={16} aria-hidden /> A caller wants to speak to you
        </h2>
        <div className="ring-who" id="ring-who">
          <strong>{who}</strong>
          {shown.callerName && shown.callerNumber && <span>{readableNumber(shown.callerNumber)}</span>}
        </div>
        {said.length > 0 && (
          <blockquote className="ring-said" aria-label="What they said">
            {said.map((line, i) => (
              <span key={i}>
                {i > 0 && <br />}“{line}”
              </span>
            ))}
          </blockquote>
        )}
        <div className="ring-clock" role="timer" aria-live="off">
          <span>Ringing{shown.devices.length ? `: ${shown.devices.join(', ')}` : ''}</span>
          <div className="ring-bar" aria-hidden>
            <i style={{ width: `${Math.min(100, (left / total) * 100)}%` }} />
          </div>
          <span aria-label={`${left} seconds left`}>{left}s</span>
        </div>
        <p className="ring-note">Answer on your Companion. This computer cannot take the call.</p>
        <div className="ring-actions">
          <button type="button" className="button secondary" disabled={busy !== null || shown.stopping} onClick={() => void act('decline')}>
            <PhoneOff size={14} /> Decline and take a message
          </button>
          <button type="button" className="button secondary" onClick={() => setPutAway((ids) => [...ids.slice(-20), shown.id])}>
            Not now
          </button>
        </div>
        {note && (
          <p className="ring-note" role="status">
            {note}
          </p>
        )}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
      </div>
    </div>
    </>
  );
}
