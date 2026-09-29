import { useCallback, useEffect, useRef, useState } from 'react';
import { MessageSquareText, Phone, PhoneOff } from 'lucide-react';
import { ring as api, type ActiveRing, type RingAction } from './api';
import { readableNumber } from './contactsModel';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * A caller wants to speak to you. While the receptionist is trying to reach the
 * owner, this desktop rings (a notification, and this dialog): who is calling, what
 * they last said, and how long it rings. The owner can take it (the Companion on this
 * computer or a phone takes the call: this computer cannot carry its audio), decline,
 * or have the receptionist take a message instead. What the owner does here is only
 * ever a request: the receptionist is told nothing until the phone says how it came out,
 * except when the owner declines, which sends the caller to the message offer at once.
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
  /** The desktop's clock less this window's, when it last answered: the countdown does not trust this window's clock. */
  const offset = useRef(0);
  const [tick, setTick] = useState(0);
  const [busy, setBusy] = useState<RingAction | null>(null);
  /** Rings the owner sent away here: not shown again if a look that was already on its way still lists them. */
  const [sentAway, setSentAway] = useState<string[]>([]);
  /** What each ring was told when the owner acted on it here. */
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  const look = useCallback(() => {
    if (!on) return;
    api.active().then(
      (list) => {
        if (list.length) offset.current = list[0].now - Date.now();
        setRings(list);
      },
      () => {
        /* the desktop is away: what was shown stays until it says */
      },
    );
  }, [on]);
  useVisiblePoll(look, POLL_MS);
  useEffect(() => {
    if (!on) setRings([]);
  }, [on]);

  // The countdown: twice a second while something rings.
  useEffect(() => {
    if (!rings.length) return;
    const id = window.setInterval(() => setTick((n) => n + 1), 500);
    return () => window.clearInterval(id);
  }, [rings.length]);

  const now = Date.now() + offset.current;
  const ringing = rings.filter((r) => r.expiresAt > now && !sentAway.includes(r.id));
  const shown = ringing[0];

  const act = async (action: RingAction) => {
    if (!shown) return;
    setBusy(action);
    setError(null);
    try {
      const r = await api.respond(shown.id, action);
      if (action === 'accept') {
        // Taking it is the Companion's: this says what was asked of it, and the ring goes on until the phone says.
        setNotes((n) => ({ ...n, [shown.id]: r.note }));
      } else {
        setSentAway((ids) => [...ids.slice(-20), shown.id]);
        setRings((list) => list.filter((x) => x.id !== shown.id));
      }
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(null);
      look();
    }
  };

  void tick;
  if (!on || !shown) return null;
  const left = secondsLeft(shown, now);
  const total = Math.max(1, Math.round((shown.expiresAt - shown.startedAt) / 1000));
  const who = shown.callerName || readableNumber(shown.callerNumber) || 'A caller who hid their number';
  const said = shown.said.filter((s) => s.trim()).slice(-2);
  const note = notes[shown.id] || shown.note;
  return (
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
        <div className="ring-actions">
          <button type="button" className="button" disabled={busy !== null} onClick={() => void act('accept')} autoFocus>
            <Phone size={14} /> Accept
          </button>
          <button type="button" className="button secondary" disabled={busy !== null} onClick={() => void act('decline')}>
            <PhoneOff size={14} /> Decline
          </button>
          <button type="button" className="button secondary" disabled={busy !== null} onClick={() => void act('message')}>
            <MessageSquareText size={14} /> Take a message instead
          </button>
        </div>
        {!shown.canAccept && !note && (
          <p className="ring-note">This computer cannot carry the call’s audio: to take it, answer on your Companion (on this computer or your phone).</p>
        )}
        {note && <p className="ring-note">{note}</p>}
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
