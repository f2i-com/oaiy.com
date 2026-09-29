import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Inbox, Phone, Search, Trash2, Undo2, UserRound, X } from 'lucide-react';
import { isNotFound, messages as api, type CallerMessage } from './api';
import { numberKey, readableNumber, saidWhen } from './contactsModel';
import { useToast } from './Toasts';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * Messages: what the receptionist took when the owner could not be reached.
 * Each one says who (the name they gave, else their number), what they want the
 * owner to know, and where to ring them back. A message is new until the page
 * has been in view with it for a few seconds (or the owner acts on it), then
 * seen, and handled when the owner says so; a handled one can be made new
 * again, and any can be deleted. The receptionist makes messages, on a call:
 * there is no way to write one here.
 */

/** How long a new message is on screen before it counts as seen. */
export const SEEN_AFTER_MS = 4_000;

type Filter = 'open' | 'handled' | 'all';
const FILTERS: Array<[Filter, string]> = [
  ['open', 'To do'],
  ['handled', 'Handled'],
  ['all', 'All'],
];

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Who a message is from: the name they gave, else their number, else that they hid it. */
export function whoOf(m: Pick<CallerMessage, 'name' | 'from' | 'callback'>): string {
  const number = readableNumber(m.from);
  return m.name || number || 'A caller who hid their number';
}

/** Whether a message (its words, name or numbers) matches what was typed. */
export function matchesMessage(m: CallerMessage, q: string): boolean {
  const t = q.trim().toLowerCase();
  if (!t) return true;
  const digits = t.replace(/\D/g, '');
  const tail = digits.slice(-9);
  return (
    m.name.toLowerCase().includes(t) ||
    m.message.toLowerCase().includes(t) ||
    (digits.length >= 3 && [m.from, m.callback].some((n) => n.replace(/\D/g, '').includes(tail)))
  );
}

export default function MessagesPanel({ onOpenContact, onOpenAgent }: { onOpenContact?: (key: string) => void; onOpenAgent?: () => void }) {
  const toast = useToast();
  const [list, setList] = useState<CallerMessage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** This desktop keeps no messages (it is older than them). */
  const [missing, setMissing] = useState(false);
  const [filter, setFilter] = useState<Filter>('open');
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.list();
      setList(r.messages ?? []);
      setError(null);
      setMissing(false);
    } catch (e) {
      if (isNotFound(e)) setMissing(true);
      else setError(errText(e));
    }
  }, []);
  // Asked again now and then: calls bring messages in while this is open.
  useVisiblePoll(() => void load(), 10_000);

  const change = useCallback(
    async (id: string, state: CallerMessage['state']) => {
      setBusy(id);
      try {
        const next = await api.mark(id, state);
        setList((prev) => prev?.map((m) => (m.id === id ? next : m)) ?? prev);
      } catch (e) {
        toast.push({ kind: 'error', title: 'Could not change the message', body: errText(e) });
      } finally {
        setBusy(null);
      }
    },
    [toast],
  );

  const remove = async (id: string) => {
    setBusy(id);
    try {
      await api.remove(id);
      setList((prev) => prev?.filter((m) => m.id !== id) ?? prev);
      setConfirming(null);
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not delete the message', body: errText(e) });
    } finally {
      setBusy(null);
    }
  };

  // A new message is seen once it has been on screen a few seconds.
  const newIds = useMemo(() => (list ?? []).filter((m) => m.state === 'new').map((m) => m.id).join(','), [list]);
  const seenTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    window.clearTimeout(seenTimer.current);
    if (!newIds) return;
    seenTimer.current = window.setTimeout(() => {
      if (document.hidden) return;
      for (const id of newIds.split(',')) void change(id, 'seen');
    }, SEEN_AFTER_MS);
    return () => window.clearTimeout(seenTimer.current);
  }, [newIds, change]);

  const shown = useMemo(
    () => (list ?? []).filter((m) => (filter === 'all' ? true : filter === 'handled' ? m.state === 'handled' : m.state !== 'handled')).filter((m) => matchesMessage(m, q)),
    [list, filter, q],
  );
  const todo = (list ?? []).filter((m) => m.state !== 'handled').length;

  if (missing) {
    return (
      <div className="panel messages-page">
        <p className="form-hint">This OAIY does not keep messages yet: update it to have the receptionist take them.</p>
      </div>
    );
  }

  return (
    <div className="panel messages-page">
      <div className="messages-toolbar">
        <div className="seg-tabs" role="tablist" aria-label="Which messages">
          {FILTERS.map(([id, label]) => (
            <button type="button" role="tab" key={id} aria-selected={filter === id} className={filter === id ? 'active' : undefined} onClick={() => setFilter(id)}>
              {label}
              {id === 'open' && todo > 0 && <em className="nav-count">{todo}</em>}
            </button>
          ))}
        </div>
        <label className="messages-search">
          <Search size={14} aria-hidden />
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search names, numbers and words" aria-label="Search messages" />
        </label>
      </div>

      {error && <p className="form-error" role="alert">Could not read the messages: {error}</p>}
      {list === null && !error && <p className="form-hint">Loading…</p>}
      {list !== null && shown.length === 0 && (
        <div className="messages-empty">
          <Inbox size={22} aria-hidden />
          <p>
            {list.length === 0
              ? 'No messages yet. When the receptionist cannot put someone through to you, it asks what they want you to know and keeps it here.'
              : filter === 'open' && !q.trim()
                ? 'Nothing waiting for you.'
                : 'No message matches.'}
          </p>
        </div>
      )}

      <ul className="message-list">
        {shown.map((m) => {
          const key = numberKey(m.from);
          return (
            <li key={m.id} className={`message-card${m.state === 'new' ? ' is-new' : ''}${m.state === 'handled' ? ' is-handled' : ''}${m.urgency === 'urgent' ? ' is-urgent' : ''}`} data-message={m.id}>
              <header>
                <strong className="message-who">{whoOf(m)}</strong>
                {m.urgency === 'urgent' && <em className="message-flag">Urgent</em>}
                {m.state === 'new' && <em className="message-flag message-new">New</em>}
                <time dateTime={m.at} title={new Date(m.at).toLocaleString()}>
                  {saidWhen(m.at)} {new Date(m.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                </time>
              </header>
              {/* Plain text: what a caller said is never markup. */}
              <p className="message-text">{m.message}</p>
              <footer>
                {m.callback ? (
                  <a className="message-number" href={`tel:${m.callback.replace(/[^\d+]/g, '')}`} title="Ring them back (opens your phone app)">
                    <Phone size={13} aria-hidden /> {readableNumber(m.callback)}
                  </a>
                ) : (
                  <span className="form-hint">No number to ring back</span>
                )}
                {m.wantsCallback && <span className="form-hint">Wants a call back</span>}
                <span className="message-actions">
                  {key && onOpenContact && (
                    <button type="button" className="btn-tiny" onClick={() => onOpenContact(key)} title="Their contact, with your notes and what the receptionist remembered">
                      <UserRound size={12} /> Contact
                    </button>
                  )}
                  {onOpenAgent && (
                    <button type="button" className="btn-tiny" onClick={onOpenAgent} title="The Agent, where their calls and texts are">
                      Agent
                    </button>
                  )}
                  {m.state === 'handled' ? (
                    <button type="button" className="btn-tiny" disabled={busy === m.id} onClick={() => void change(m.id, 'new')}>
                      <Undo2 size={12} /> Not handled
                    </button>
                  ) : (
                    <button type="button" className="btn-tiny" disabled={busy === m.id} onClick={() => void change(m.id, 'handled')}>
                      <Check size={12} /> Handled
                    </button>
                  )}
                  {confirming === m.id ? (
                    <>
                      <button type="button" className="btn-tiny danger" disabled={busy === m.id} onClick={() => void remove(m.id)}>
                        <Trash2 size={12} /> Delete it
                      </button>
                      <button type="button" className="btn-tiny" onClick={() => setConfirming(null)} aria-label="Keep it">
                        <X size={12} />
                      </button>
                    </>
                  ) : (
                    <button type="button" className="btn-tiny" onClick={() => setConfirming(m.id)} aria-label={`Delete the message from ${whoOf(m)}`}>
                      <Trash2 size={12} />
                    </button>
                  )}
                </span>
              </footer>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
