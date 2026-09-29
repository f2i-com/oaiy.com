import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from 'react';
import { CalendarDays, CalendarPlus, Check, ChevronLeft, ChevronRight, Columns3, List, Pencil, RefreshCw, X } from 'lucide-react';
import { calendar, type Appointment, type AppointmentStatus, type CalendarSettings, type CalendarSync } from './api';
import { AppointmentDetails, BookingForm, StatusPill, sayLongDay, type BookingValues } from './CalendarBooking';
import {
  DAYS,
  MONTHS,
  STATUS_TEXT,
  addDays,
  dateOf,
  endOf,
  gridHours,
  hhmm,
  layoutDay,
  minutesOf,
  mondayOf,
  parseYmd,
  sayMinutes,
  sayRange,
  sayTime,
  sayWeek,
  sayWhen,
  SOURCE_TEXT,
  snapDown,
  timeOf,
  toHHMM,
  weekdayOf,
  ymd,
} from './calendarModel';
import { refetchRequests, useWaitingRequests } from './calendarRequests';
import { describeSync } from './syncStatus';
import { useToast } from './Toasts';
import { moduleOn, useModules } from './useModules';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * The Calendar: the week (or a day, or a list) of appointments, and the
 * requests waiting for someone to confirm them, which the phone takes on calls
 * and the agent in texts. A click on an empty time starts a booking there; a
 * click on an appointment opens it in the side panel, where a request is
 * confirmed, declined or changed. The hours and services it offers are on
 * Hours & Services. It all lives on this machine; FormLogic syncs with it when
 * linked.
 */

type Mode = 'week' | 'day' | 'list';
type Side =
  | { kind: 'details'; a: Appointment; edit?: boolean }
  | { kind: 'new'; date: string; time: string | null };

const POLL_MS = 10_000;
/** One hour of the grid. */
const HOUR_PX = 56;
/** How far ahead the list reaches. */
const LIST_DAYS = 60;
const MODE_KEY = 'oaiy.calendarView';
const NARROW = '(max-width: 900px)';

function readMode(): Mode {
  try {
    const m = window.localStorage.getItem(MODE_KEY);
    return m === 'day' || m === 'list' ? m : 'week';
  } catch {
    return 'week';
  }
}
function writeMode(m: Mode) {
  try {
    window.localStorage.setItem(MODE_KEY, m);
  } catch {
    /* storage can be unavailable: the week opens next time */
  }
}

/** Whether the window is narrow (the week becomes a day). */
function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(NARROW).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(NARROW);
    if (!mq) return;
    const on = () => setNarrow(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return narrow;
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;

export default function CalendarPanel({ onOpenHours }: { onOpenHours?: () => void }) {
  const phoneOn = moduleOn(useModules(), 'phone') === true;
  const toast = useToast();
  const narrow = useNarrow();
  const [chosenMode, setChosenMode] = useState<Mode>(readMode);
  const mode: Mode = chosenMode === 'week' && narrow ? 'day' : chosenMode;
  const [anchor, setAnchor] = useState(() => new Date());
  const [now, setNow] = useState(() => new Date());
  const [settings, setSettings] = useState<CalendarSettings | null>(null);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sync, setSync] = useState<CalendarSync | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [side, setSide] = useState<Side | null>(null);
  const [filter, setFilter] = useState<'all' | AppointmentStatus>('all');
  const requests = useWaitingRequests(true);
  const opener = useRef<HTMLElement | null>(null);

  const week = useMemo(() => mondayOf(anchor), [anchor]);
  const today = ymd(now);
  const [from, to] = useMemo(
    () => (mode === 'list' ? [today, ymd(addDays(parseYmd(today), LIST_DAYS))] : [ymd(week), ymd(addDays(week, 7))]),
    [mode, today, week],
  );

  const refresh = useCallback(async () => {
    try {
      const got = await calendar.get(from, to);
      setSettings(got.settings);
      setAppointments(got.appointments);
      setError(null);
    } catch (e) {
      setError(errText(e));
    }
    setSync(await calendar.syncStatus().catch(() => null));
  }, [from, to]);

  useEffect(() => void refresh(), [refresh]);
  useVisiblePoll(() => void refresh(), POLL_MS);
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  const pickMode = (m: Mode) => {
    setChosenMode(m);
    writeMode(m);
  };

  const openSide = (next: Side) => {
    if (!side) opener.current = document.activeElement as HTMLElement | null;
    setSide(next);
  };
  const closeSide = useCallback(() => {
    setSide(null);
    const back = opener.current;
    opener.current = null;
    if (back?.isConnected) requestAnimationFrame(() => back.focus());
  }, []);
  // Escape closes the side panel.
  useEffect(() => {
    if (!side) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeSide();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [side, closeSide]);

  // The appointment shown stays current as the calendar is asked again.
  const everyone = useMemo(() => {
    const byId = new Map<string, Appointment>();
    for (const a of [...appointments, ...(requests ?? [])]) byId.set(a.id, a);
    return byId;
  }, [appointments, requests]);
  const shown = side?.kind === 'details' ? (everyone.get(side.a.id) ?? side.a) : null;

  const textConfirmation = async (a: Appointment, start: string) => {
    const who = settings?.business.trim() ? ` with ${settings.business.trim()}` : '';
    await calendar.text(a.phone, `Hi${a.name ? ` ${a.name}` : ''}, your ${a.service || 'appointment'}${who} is confirmed for ${sayWhen(start)}. See you then!`);
  };

  const afterChange = async () => {
    await Promise.all([refresh(), refetchRequests()]);
  };

  /** Confirm, decline, cancel, mark done; a confirmation is texted to them when asked to. */
  const decide = async (a: Appointment, status: AppointmentStatus, textThem: boolean) => {
    try {
      const updated = await calendar.update(a.id, { status });
      setSide((s) => (s?.kind === 'details' && s.a.id === a.id ? { kind: 'details', a: updated } : s));
      if (status === 'confirmed' && textThem && a.phone) {
        await textConfirmation(a, updated.start)
          .then(() => toast.push({ kind: 'success', title: 'Confirmed', body: `${a.name || 'They'} will get a text at ${a.phone}.` }))
          .catch((e) => toast.push({ kind: 'error', title: 'Confirmed, but the text did not go', body: errText(e) }));
      } else {
        const title =
          status === 'confirmed' ? (a.status === 'requested' ? 'Confirmed' : 'Booked again') : status === 'done' ? 'Marked done' : STATUS_TEXT[status];
        toast.push({ kind: 'success', title, body: `${a.service || 'Appointment'}, ${sayWhen(updated.start)}` });
      }
      await afterChange();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not change it', body: errText(e) });
    }
  };

  const saveChange = async (a: Appointment, v: BookingValues, confirm: boolean) => {
    try {
      const start = `${v.date}T${v.time}`;
      const updated = await calendar.update(a.id, { start, service: v.service, name: v.name, phone: v.phone, notes: v.notes, ...(confirm ? { status: 'confirmed' as const } : {}) });
      setSide({ kind: 'details', a: updated });
      setAnchor(parseYmd(v.date));
      if (confirm && phoneOn && settings?.textConfirmations && updated.phone) {
        await textConfirmation(updated, updated.start)
          .then(() => toast.push({ kind: 'success', title: 'Changed and confirmed', body: `${updated.name || 'They'} will get a text at ${updated.phone}.` }))
          .catch((e) => toast.push({ kind: 'error', title: 'Confirmed, but the text did not go', body: errText(e) }));
      } else {
        toast.push({ kind: 'success', title: confirm ? 'Changed and confirmed' : 'Changed', body: `${updated.service || 'Appointment'}, ${sayWhen(updated.start)}` });
      }
      await afterChange();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not save it', body: errText(e) });
    }
  };

  const book = async (v: BookingValues) => {
    try {
      const a = await calendar.create({ service: v.service, date: v.date, time: v.time, name: v.name, phone: v.phone, notes: v.notes, source: 'manual', status: 'confirmed' });
      toast.push({ kind: 'success', title: 'Booked', body: `${a.service || 'Appointment'}, ${sayWhen(a.start)}` });
      setAnchor(parseYmd(dateOf(a.start)));
      setSide({ kind: 'details', a });
      await afterChange();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not book it', body: errText(e) });
    }
  };

  const remove = async (a: Appointment) => {
    if (!window.confirm('Delete this appointment? Cancelling it keeps a record; deleting does not.')) return;
    try {
      await calendar.remove(a.id);
      toast.push({ kind: 'success', title: 'Deleted', body: `${a.service || 'Appointment'}, ${sayWhen(a.start)}` });
      closeSide();
      await afterChange();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not delete it', body: errText(e) });
    }
  };

  const syncNow = async () => {
    setSyncing(true);
    try {
      setSync(await calendar.syncNow());
      await afterChange();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not sync now', body: errText(e) });
    } finally {
      setSyncing(false);
    }
  };

  const step = mode === 'day' ? 1 : 7;
  const days = useMemo(() => (mode === 'day' ? [ymd(anchor)] : Array.from({ length: 7 }, (_, i) => ymd(addDays(week, i)))), [mode, anchor, week]);
  const inGrid = useMemo(() => appointments.filter((a) => days.includes(dateOf(a.start))), [appointments, days]);
  const [firstHour, lastHour] = useMemo(() => gridHours(settings, inGrid), [settings, inGrid]);
  const title = mode === 'list' ? 'Coming up' : mode === 'day' ? sayLongDay(ymd(anchor)) + (anchor.getFullYear() !== now.getFullYear() ? ` ${anchor.getFullYear()}` : '') : sayWeek(week);
  const onThisWeek = mode === 'day' ? ymd(anchor) === today : ymd(week) === ymd(mondayOf(now));
  const waiting = requests ?? [];

  const newAt = (date: string, time: string | null) => openSide({ kind: 'new', date: date < today ? today : date, time });

  return (
    <div className={`panel cal-page${side ? ' has-side' : ''}`}>
      {error && <div className="banner banner-err">The calendar could not be read: {error}</div>}
      {sync?.linked && (sync.problems?.length ?? 0) > 0 && (
        <div className="banner banner-err" role="status">
          FormLogic would not take {sync.problems!.length === 1 ? 'an appointment' : plural(sync.problems!.length, 'appointment')}:{' '}
          {sync.problems!
            .slice(0, 3)
            .map((p) => {
              const a = everyone.get(p.id);
              return `${a ? `${a.service || 'Appointment'}, ${sayWhen(a.start)}` : 'one'} (${p.message})`;
            })
            .join('; ')}
          . It is kept here, and sent again once it is changed.
        </div>
      )}

      <div className="cal-toolbar">
        <div className="cal-toolbar-nav">
          {mode !== 'list' && (
            <>
              <div className="cal-stepper">
                <button type="button" className="icon-button" aria-label={mode === 'day' ? 'The day before' : 'The week before'} onClick={() => setAnchor((d) => addDays(d, -step))}>
                  <ChevronLeft size={16} />
                </button>
                <button type="button" className="btn-tiny" disabled={onThisWeek} onClick={() => setAnchor(new Date())}>
                  Today
                </button>
                <button type="button" className="icon-button" aria-label={mode === 'day' ? 'The day after' : 'The week after'} onClick={() => setAnchor((d) => addDays(d, step))}>
                  <ChevronRight size={16} />
                </button>
              </div>
            </>
          )}
          <h2 className="cal-title">{title}</h2>
        </div>
        <div className="seg-tabs cal-modes" role="tablist" aria-label="Show the calendar as">
          {(narrow ? (['day', 'list'] as Mode[]) : (['week', 'day', 'list'] as Mode[])).map((m) => {
            const Icon = m === 'week' ? Columns3 : m === 'day' ? CalendarDays : List;
            return (
              <button type="button" role="tab" key={m} aria-selected={mode === m} className={mode === m ? 'active' : undefined} onClick={() => pickMode(m)}>
                <Icon size={14} />
                <span>{m === 'week' ? 'Week' : m === 'day' ? 'Day' : 'List'}</span>
              </button>
            );
          })}
        </div>
        <div className="cal-toolbar-end">
          <SyncChip sync={sync} syncing={syncing} onSync={() => void syncNow()} />
          <button type="button" className="btn btn-primary" disabled={!settings} onClick={() => newAt(mode === 'list' ? today : ymd(mode === 'day' ? anchor : week < parseYmd(today) ? now : week), null)}>
            <CalendarPlus size={14} /> New appointment
          </button>
        </div>
      </div>

      {waiting.length > 0 && !(mode === 'list' && filter === 'requested') && settings && (
        <RequestsStrip
          requests={waiting}
          settings={settings}
          canText={phoneOn}
          onDecide={decide}
          onOpen={(a, edit) => {
            setAnchor(parseYmd(dateOf(a.start)));
            openSide({ kind: 'details', a, edit });
          }}
          onShowAll={() => {
            pickMode('list');
            setFilter('requested');
          }}
        />
      )}

      <div className={`cal-main${side ? ' has-side' : ''}`}>
        <div className="cal-board">
          {!settings && !error ? (
            <div className="empty-state">Reading the calendar…</div>
          ) : !settings ? null : mode === 'list' ? (
            <AgendaList
              items={appointments}
              filter={filter}
              onFilter={setFilter}
              today={today}
              selectedId={shown?.id ?? null}
              onOpen={(a) => openSide({ kind: 'details', a })}
            />
          ) : (
            <>
              {mode === 'day' && (
                <DayStrip
                  week={week}
                  anchor={ymd(anchor)}
                  today={today}
                  settings={settings}
                  appointments={appointments}
                  onPick={(d) => setAnchor(parseYmd(d))}
                />
              )}
              <WeekGrid
                days={days}
                settings={settings}
                appointments={inGrid}
                today={today}
                now={now}
                firstHour={firstHour}
                lastHour={lastHour}
                selectedId={shown?.id ?? null}
                onSlot={newAt}
                onOpen={(a) => openSide({ kind: 'details', a })}
                onOpenDay={(d) => {
                  setAnchor(parseYmd(d));
                  pickMode('day');
                }}
              />
              <div className="cal-foot">
                <span className="cal-legend" aria-label="What the colours mean">
                  <i className="is-confirmed" /> Confirmed
                  <i className="is-requested" /> Request
                  <i className="is-done" /> Done
                  <i className="is-cancelled" /> Cancelled
                  <i className="is-closed" /> Closed
                </span>
                <span className="cal-foot-hint">Click an empty time to book it.</span>
                {onOpenHours && (
                  <button type="button" className="btn-link" onClick={onOpenHours}>
                    Change the hours and services
                  </button>
                )}
              </div>
            </>
          )}
        </div>

        {side && settings && (
          <aside className="cal-side" aria-labelledby="cal-side-title">
            <div className="cal-side-head">
              <div className="cal-side-heading">
                {side.kind === 'new' ? <span className="cal-side-kicker">New appointment</span> : shown && <StatusPill status={shown.status} />}
                <h2 id="cal-side-title" tabIndex={-1}>
                  {side.kind === 'new' ? 'Book a time' : side.edit ? `Change: ${shown?.service || 'appointment'}` : shown?.service || 'Appointment'}
                </h2>
              </div>
              <button type="button" className="icon-button" aria-label="Close the panel" title="Close (Esc)" onClick={closeSide}>
                <X size={16} />
              </button>
            </div>
            <div className="cal-side-body">
              {side.kind === 'new' ? (
                <BookingForm
                  key={`new:${side.date}:${side.time ?? ''}`}
                  settings={settings}
                  initial={{ date: side.date, time: side.time ?? undefined }}
                  wanted={side.time}
                  submitLabel="Book it"
                  onSubmit={(v) => book(v)}
                  onCancel={closeSide}
                />
              ) : shown && side.edit ? (
                <BookingForm
                  key={`edit:${shown.id}`}
                  settings={settings}
                  appointment={shown}
                  initial={{ service: shown.service, date: dateOf(shown.start), time: timeOf(shown.start), name: shown.name, phone: shown.phone, notes: shown.notes }}
                  submitLabel="Save the change"
                  onSubmit={(v, confirm) => saveChange(shown, v, confirm)}
                  onCancel={() => setSide({ kind: 'details', a: shown })}
                />
              ) : shown ? (
                <AppointmentDetails
                  a={shown}
                  settings={settings}
                  canText={phoneOn}
                  onDecide={decide}
                  onChange={() => setSide({ kind: 'details', a: shown, edit: true })}
                  onDelete={() => void remove(shown)}
                />
              ) : null}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

// ---- FormLogic, quietly ---------------------------------------------------------------------

function SyncChip({ sync, syncing, onSync }: { sync: CalendarSync | null; syncing: boolean; onSync: () => void }) {
  const view = describeSync(sync, null);
  if (!view || !sync) return null;
  const detail = [
    sync.error,
    sync.nextAttemptAt && sync.state !== 'synced' ? `Next try ${new Date(sync.nextAttemptAt).toLocaleTimeString()}` : null,
    sync.at ? `Last sync: ${sync.pulled} in, ${sync.pushed} out` : null,
    'The calendar works without FormLogic; changes made here are sent when it can be reached.',
  ]
    .filter(Boolean)
    .join('\n');
  return (
    <span className={`cal-sync is-${view.tone}`} title={detail}>
      <i aria-hidden />
      <span>
        FormLogic · {view.headline}
        {view.detail ? <small> {view.detail}</small> : null}
      </span>
      <button type="button" className="icon-button" aria-label={syncing ? 'Syncing with FormLogic' : 'Sync with FormLogic now'} title="Sync now" disabled={syncing} onClick={onSync}>
        <RefreshCw size={13} className={syncing ? 'spin' : undefined} />
      </button>
    </span>
  );
}

// ---- the requests waiting ----------------------------------------------------------------

const SHOWN_REQUESTS = 3;

function RequestsStrip({
  requests,
  settings,
  canText,
  onDecide,
  onOpen,
  onShowAll,
}: {
  requests: Appointment[];
  settings: CalendarSettings;
  canText: boolean;
  onDecide: (a: Appointment, s: AppointmentStatus, text: boolean) => Promise<void>;
  onOpen: (a: Appointment, edit?: boolean) => void;
  onShowAll: () => void;
}) {
  const texts = canText && settings.textConfirmations;
  return (
    <section className="cal-requests" aria-labelledby="cal-requests-title">
      <div className="cal-requests-head">
        <h3 className="section-title" id="cal-requests-title">
          {requests.length === 1 ? 'A request is waiting' : `${requests.length} requests are waiting`}
        </h3>
        <p className="form-hint">
          Asked for on calls and in texts, and kept for them until you answer.{texts ? ' Confirming one texts them.' : ''}
        </p>
        {requests.length > SHOWN_REQUESTS && (
          <button type="button" className="btn-link" onClick={onShowAll}>
            See all {requests.length}
          </button>
        )}
      </div>
      <ul className="cal-request-list">
        {requests.slice(0, SHOWN_REQUESTS).map((a) => (
          <RequestRow key={a.id} a={a} textByDefault={texts} canText={canText} onDecide={onDecide} onOpen={onOpen} />
        ))}
      </ul>
    </section>
  );
}

function RequestRow({
  a,
  textByDefault,
  canText,
  onDecide,
  onOpen,
}: {
  a: Appointment;
  textByDefault: boolean;
  canText: boolean;
  onDecide: (a: Appointment, s: AppointmentStatus, text: boolean) => Promise<void>;
  onOpen: (a: Appointment, edit?: boolean) => void;
}) {
  const [textThem, setTextThem] = useState(textByDefault && !!a.phone);
  const [busy, setBusy] = useState(false);
  const go = async (status: AppointmentStatus) => {
    setBusy(true);
    try {
      await onDecide(a, status, status === 'confirmed' && textThem);
    } finally {
      setBusy(false);
    }
  };
  const date = dateOf(a.start);
  const d = parseYmd(date);
  return (
    <li className="cal-request">
      <button type="button" className="cal-request-main" onClick={() => onOpen(a)} title="See the request">
        <span className="cal-request-date" aria-hidden>
          <small>{DAYS[weekdayOf(d)]}</small>
          <strong>{d.getDate()}</strong>
          <small>{MONTHS[d.getMonth()]}</small>
        </span>
        <span className="cal-request-text">
          <strong>
            {a.service || 'Appointment'} <span>{sayRange(timeOf(a.start), endOf(a))}</span>
          </strong>
          <small>
            {[a.name || 'No name given', a.phone, `asked ${SOURCE_TEXT[a.source] ?? `from ${a.source}`}`].filter(Boolean).join(' · ')}
          </small>
        </span>
      </button>
      <div className="cal-request-actions">
        {canText && a.phone && (
          <label className="cal-check" title="Text them that it is confirmed">
            <input type="checkbox" checked={textThem} onChange={(e) => setTextThem(e.target.checked)} /> Text them
          </label>
        )}
        <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void go('confirmed')}>
          <Check size={14} /> Confirm
        </button>
        <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => void go('declined')}>
          Decline
        </button>
        <button type="button" className="icon-button" disabled={busy} aria-label={`Change the ${a.service || 'request'}`} title="Change it" onClick={() => onOpen(a, true)}>
          <Pencil size={14} />
        </button>
      </div>
    </li>
  );
}

// ---- the week (or a day) ----------------------------------------------------------------

function WeekGrid({
  days,
  settings,
  appointments,
  today,
  now,
  firstHour,
  lastHour,
  selectedId,
  onSlot,
  onOpen,
  onOpenDay,
}: {
  days: string[];
  settings: CalendarSettings;
  appointments: Appointment[];
  today: string;
  now: Date;
  firstHour: number;
  lastHour: number;
  selectedId: string | null;
  onSlot: (date: string, time: string) => void;
  onOpen: (a: Appointment) => void;
  onOpenDay: (date: string) => void;
}) {
  const [ghost, setGhost] = useState<{ date: string; min: number } | null>(null);
  const stepMin = Math.max(15, settings.slotMinutes || 30);
  const ghostLen = settings.services[0]?.minutes || stepMin;
  const nowMin = now.getHours() * 60 + now.getMinutes();
  const hours = lastHour - firstHour;
  const y = (min: number) => ((min - firstHour * 60) / 60) * HOUR_PX;
  const minuteAt = (e: ReactMouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const min = firstHour * 60 + ((e.clientY - r.top) / HOUR_PX) * 60;
    return Math.max(firstHour * 60, Math.min(lastHour * 60 - stepMin, snapDown(min, stepMin)));
  };
  const onAppt = (e: ReactMouseEvent) => !!(e.target as HTMLElement).closest('.cal-appt');
  const style = { '--hour-px': `${HOUR_PX}px`, '--hours': hours, '--days': days.length } as CSSProperties;

  return (
    <div className={`cal-grid${days.length === 1 ? ' is-day' : ''}`} style={style}>
      <div className="cal-grid-head">
        <div className="cal-corner" />
        {days.map((date) => {
          const d = parseYmd(date);
          const spans = settings.hours[weekdayOf(d)] ?? [];
          return (
            <button
              type="button"
              key={date}
              className={`cal-dayhead${date === today ? ' is-today' : ''}${spans.length ? '' : ' is-closed'}${date < today ? ' is-past' : ''}`}
              onClick={() => onOpenDay(date)}
              disabled={days.length === 1}
              title={days.length === 1 ? undefined : `Open ${sayLongDay(date)}`}
            >
              <span className="cal-dayname">{DAYS[weekdayOf(d)]}</span>
              <strong>{d.getDate()}</strong>
              <small className={spans.length ? 'is-open' : undefined}>
                {spans.length ? spans.map((s) => sayRange(toHHMM(s.open) ?? s.open, toHHMM(s.close) ?? s.close)).join(', ') : 'Closed'}
              </small>
            </button>
          );
        })}
      </div>
      <div className="cal-grid-body">
        <div className="cal-times" aria-hidden>
          {Array.from({ length: hours }, (_, i) => (
            <span key={i} style={{ top: i * HOUR_PX }}>
              {i === 0 ? '' : `${(firstHour + i) % 12 || 12} ${firstHour + i < 12 ? 'am' : 'pm'}`}
            </span>
          ))}
        </div>
        {days.map((date) => {
          const d = parseYmd(date);
          const spans = settings.hours[weekdayOf(d)] ?? [];
          const mine = appointments.filter((a) => dateOf(a.start) === date);
          const lanes = layoutDay(mine);
          const past = date < today;
          const g = ghost?.date === date ? ghost.min : null;
          const ghostPast = date < today || (date === today && g !== null && g < nowMin);
          return (
            <div
              key={date}
              className={`cal-col${date === today ? ' is-today' : ''}${past ? ' is-past' : ''}`}
              onMouseMove={(e) => setGhost(onAppt(e) ? null : { date, min: minuteAt(e) })}
              onMouseLeave={() => setGhost(null)}
              onClick={(e) => {
                if (onAppt(e)) return;
                onSlot(date, hhmm(minuteAt(e)));
              }}
            >
              {spans.map((s, k) => {
                const o = minutesOf(toHHMM(s.open) ?? '');
                const c = minutesOf(toHHMM(s.close) ?? '');
                if (Number.isNaN(o) || Number.isNaN(c)) return null;
                return <div key={k} className="cal-open" style={{ top: y(o), height: y(c) - y(o) }} />;
              })}
              <div className="cal-lines" />
              {g !== null && !ghostPast && (
                <div className="cal-ghost" style={{ top: y(g), height: Math.max(22, (ghostLen / 60) * HOUR_PX - 3) }} aria-hidden>
                  <CalendarPlus size={12} /> {sayRange(hhmm(g), hhmm(g + ghostLen))}
                </div>
              )}
              {date === today && nowMin >= firstHour * 60 && nowMin <= lastHour * 60 && <div className="cal-now" style={{ top: y(nowMin) }} aria-hidden />}
              {mine.map((a) => {
                const start = minutesOf(timeOf(a.start));
                if (Number.isNaN(start)) return null;
                const lane = lanes.get(a.id) ?? { lane: 0, lanes: 1, cascade: false };
                const h = Math.max(24, (a.minutes / 60) * HOUR_PX - 3);
                const struck = a.status === 'cancelled' || a.status === 'declined';
                // Stacked: each one over the one before, indented; else side by side.
                const indent = lane.cascade ? Math.min(lane.lane * 24, 48) : (lane.lane * 100) / lane.lanes;
                const pos = {
                  top: y(start) + 1,
                  height: h,
                  left: `calc(${indent}% + 3px)`,
                  width: lane.cascade ? `calc(${100 - indent}% - 6px)` : `calc(${100 / lane.lanes}% - 6px)`,
                  zIndex: lane.cascade ? 2 + lane.lane : undefined,
                } as CSSProperties;
                const from = timeOf(a.start);
                return (
                  <button
                    type="button"
                    key={a.id}
                    className={`cal-appt is-${a.status}${h < 44 ? ' is-short' : ''}${lane.cascade && lane.lane ? ' is-stacked' : ''}${a.id === selectedId ? ' is-selected' : ''}`}
                    style={pos}
                    aria-label={`${a.service || 'Appointment'}, ${sayLongDay(date)} ${sayRange(from, endOf(a))}, ${a.name || 'no name'}: ${STATUS_TEXT[a.status]}`}
                    title={`${sayRange(from, endOf(a))} · ${a.service || 'Appointment'}${a.name ? ` · ${a.name}` : ''} · ${STATUS_TEXT[a.status]}`}
                    onClick={() => onOpen(a)}
                  >
                    <span className="cal-appt-top">
                      <span className="cal-appt-time">
                        <span className="is-long">{sayRange(from, endOf(a))}</span>
                        <span className="is-brief">{sayTime(from).replace(/ (am|pm)$/, '')}</span>
                      </span>
                      {a.status !== 'confirmed' && <span className="cal-appt-tag">{STATUS_TEXT[a.status]}</span>}
                    </span>
                    <strong className={struck ? 'is-struck' : undefined}>{a.service || 'Appointment'}</strong>
                    {a.name && <span className="cal-appt-who">{a.name}</span>}
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The day view's week: a chip a day, with how many are booked. */
function DayStrip({
  week,
  anchor,
  today,
  settings,
  appointments,
  onPick,
}: {
  week: Date;
  anchor: string;
  today: string;
  settings: CalendarSettings;
  appointments: Appointment[];
  onPick: (date: string) => void;
}) {
  return (
    <div className="cal-daystrip" role="tablist" aria-label="Day of the week">
      {Array.from({ length: 7 }, (_, i) => {
        const date = ymd(addDays(week, i));
        const d = parseYmd(date);
        const n = appointments.filter((a) => dateOf(a.start) === date && a.status !== 'cancelled' && a.status !== 'declined').length;
        const closed = (settings.hours[weekdayOf(d)] ?? []).length === 0;
        return (
          <button
            type="button"
            role="tab"
            key={date}
            aria-selected={date === anchor}
            className={`cal-daychip${date === anchor ? ' is-on' : ''}${date === today ? ' is-today' : ''}${closed ? ' is-off' : ''}`}
            onClick={() => onPick(date)}
          >
            <span>{DAYS[i]}</span>
            <strong>{d.getDate()}</strong>
            <small>{closed ? 'Closed' : n ? plural(n, 'booking') : 'Free'}</small>
          </button>
        );
      })}
    </div>
  );
}

// ---- the list ------------------------------------------------------------------------------

function AgendaList({
  items,
  filter,
  onFilter,
  today,
  selectedId,
  onOpen,
}: {
  items: Appointment[];
  filter: 'all' | AppointmentStatus;
  onFilter: (f: 'all' | AppointmentStatus) => void;
  today: string;
  selectedId: string | null;
  onOpen: (a: Appointment) => void;
}) {
  const count = (s: AppointmentStatus) => items.filter((a) => a.status === s).length;
  const shown = items.filter((a) => filter === 'all' || a.status === filter || (filter === 'cancelled' && a.status === 'declined'));
  const byDay = new Map<string, Appointment[]>();
  for (const a of [...shown].sort((x, y) => x.start.localeCompare(y.start))) {
    const d = dateOf(a.start);
    byDay.set(d, [...(byDay.get(d) ?? []), a]);
  }
  const tomorrow = ymd(addDays(parseYmd(today), 1));
  const filters: { id: 'all' | AppointmentStatus; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'requested', label: `Requests${count('requested') ? ` (${count('requested')})` : ''}` },
    { id: 'confirmed', label: 'Confirmed' },
    { id: 'cancelled', label: 'Cancelled' },
  ];
  return (
    <div className="cal-agenda">
      <div className="cal-agenda-head">
        <p className="form-hint">The next {LIST_DAYS} days.</p>
        <div className="seg-tabs" role="tablist" aria-label="Show">
          {filters.map((f) => (
            <button type="button" role="tab" key={f.id} aria-selected={filter === f.id} className={filter === f.id ? 'active' : undefined} onClick={() => onFilter(f.id)}>
              <span>{f.label}</span>
            </button>
          ))}
        </div>
      </div>
      {byDay.size === 0 ? (
        <div className="empty-state">
          <p>{filter === 'requested' ? 'No requests waiting.' : filter === 'all' ? 'Nothing booked yet.' : `Nothing ${STATUS_TEXT[filter].toLowerCase()}.`}</p>
          <p>Calls and texts bring requests here for you to confirm.</p>
        </div>
      ) : (
        [...byDay.entries()].map(([date, list]) => (
          <section key={date} className="cal-agenda-day">
            <h3>
              {sayLongDay(date)}
              {date === today ? <em>Today</em> : date === tomorrow ? <em>Tomorrow</em> : null}
            </h3>
            <ul>
              {list.map((a) => (
                <li key={a.id}>
                  <button type="button" className={`cal-agenda-row is-${a.status}${a.id === selectedId ? ' is-selected' : ''}`} onClick={() => onOpen(a)}>
                    <span className="cal-agenda-time">{sayRange(timeOf(a.start), endOf(a))}</span>
                    <span className="cal-agenda-what">
                      <strong className={a.status === 'cancelled' || a.status === 'declined' ? 'is-struck' : undefined}>{a.service || 'Appointment'}</strong>
                      <small>{[a.name || 'No name given', a.phone, sayMinutes(a.minutes)].filter(Boolean).join(' · ')}</small>
                    </span>
                    <StatusPill status={a.status} />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  );
}
