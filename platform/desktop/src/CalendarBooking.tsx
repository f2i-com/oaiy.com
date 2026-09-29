import { useEffect, useMemo, useRef, useState } from 'react';
import { Ban, CalendarPlus, Check, ChevronLeft, ChevronRight, CircleCheck, MessageSquare, Pencil, Phone, Trash2, Undo2, X } from 'lucide-react';
import { calendar, type Appointment, type AppointmentStatus, type CalendarSettings } from './api';
import {
  DAY_NAMES,
  MONTHS,
  SOURCE_TEXT,
  STATUS_TEXT,
  addDays,
  dateOf,
  endOf,
  groupTimes,
  mondayOf,
  parseYmd,
  pickTime,
  sayMinutes,
  sayRange,
  sayTime,
  timeOf,
  weekdayOf,
  ymd,
} from './calendarModel';

/**
 * The Calendar page's side panel: an appointment's details (and, for a
 * request, Confirm, Decline and Change), a change to one, or a new one. The
 * new one's time comes from the free-time picker: the days of a week, each
 * with its free times for the service chosen (the desktop's own answer: the
 * opening hours, the booking rules and what is already booked), grouped as
 * morning, afternoon and evening.
 */

/** "Thursday 1 October". */
export function sayLongDay(date: string): string {
  const d = parseYmd(date);
  return `${DAY_NAMES[weekdayOf(d)]} ${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

export function StatusPill({ status }: { status: AppointmentStatus }) {
  return <span className={`cal-pill is-${status}`}>{STATUS_TEXT[status]}</span>;
}

// ---- the free-time picker ------------------------------------------------------------

export interface SlotPickerProps {
  settings: CalendarSettings;
  /** The service (its length decides what fits), or none. */
  service: string;
  /** How long, when it is not the service's own length. */
  minutes?: number;
  date: string;
  time: string | null;
  /** The time aimed for when a day or service is picked (the slot clicked). */
  wanted?: string | null;
  /** An appointment being changed: its own time is offered too (it is taken only by itself). */
  current?: { date: string; time: string };
  onPick: (date: string, time: string | null) => void;
}

export function SlotPicker({ settings, service, minutes, date, time, wanted, current, onPick }: SlotPickerProps) {
  const today = ymd(new Date());
  const [week, setWeek] = useState(() => ymd(mondayOf(parseYmd(date))));
  const [free, setFree] = useState<Record<string, string[]> | null>(null);
  const [failed, setFailed] = useState(false);
  const [other, setOther] = useState(false);
  const aim = useRef<string | null>(wanted ?? time ?? null);

  // The strip follows the day chosen (a date typed, a slot clicked).
  useEffect(() => {
    const m = ymd(mondayOf(parseYmd(date)));
    setWeek((w) => (w === m ? w : m));
  }, [date]);

  useEffect(() => {
    let live = true;
    setFree(null);
    setFailed(false);
    calendar
      .free(week, 7, service || undefined, minutes)
      .then((r) => live && setFree(Object.fromEntries(r.days.map((d) => [d.date, d.times]))))
      .catch(() => {
        if (!live) return;
        setFree({});
        setFailed(true);
      });
    return () => {
      live = false;
    };
  }, [week, service, minutes]);

  const times = useMemo(() => {
    const t = free?.[date] ?? [];
    return current && current.date === date && !t.includes(current.time) ? [...t, current.time].sort() : t;
  }, [free, date, current]);

  // A day or service picked: keep the time when it is still free, else the nearest free one.
  useEffect(() => {
    if (!free || other) return;
    if (time && times.includes(time)) return;
    const next = pickTime(times, aim.current ?? time);
    if (next !== time) onPick(date, next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [free, times, date, other]);

  const days = Array.from({ length: 7 }, (_, i) => ymd(addDays(parseYmd(week), i)));
  const thisWeek = ymd(mondayOf(new Date()));
  const groups = groupTimes(times);
  const closed = (d: string) => (settings.hours[weekdayOf(parseYmd(d))] ?? []).length === 0;
  // The time aimed for (a slot clicked) is not free: say so, and that the nearest was picked.
  const missed = !!aim.current && !!time && aim.current !== time && !times.includes(aim.current);

  return (
    <div className="cal-picker">
      <div className="cal-picker-week">
        <button type="button" className="icon-button" aria-label="The week before" disabled={week <= thisWeek} onClick={() => setWeek(ymd(addDays(parseYmd(week), -7)))}>
          <ChevronLeft size={15} />
        </button>
        <div className="cal-daychips" role="radiogroup" aria-label="Day">
          {days.map((d) => {
            const past = d < today;
            const n = free?.[d]?.length ?? 0;
            const note = past ? 'Past' : closed(d) ? 'Closed' : free === null ? '…' : n ? `${n} free` : 'Full';
            const dd = parseYmd(d);
            return (
              <button
                type="button"
                key={d}
                role="radio"
                aria-checked={d === date}
                className={`cal-daychip${d === date ? ' is-on' : ''}${past || closed(d) ? ' is-off' : ''}${d === today ? ' is-today' : ''}`}
                disabled={past}
                onClick={() => onPick(d, null)}
              >
                <span>{DAY_NAMES[weekdayOf(dd)].slice(0, 3)}</span>
                <strong>{dd.getDate()}</strong>
                <small>{note}</small>
              </button>
            );
          })}
        </div>
        <button type="button" className="icon-button" aria-label="The week after" onClick={() => setWeek(ymd(addDays(parseYmd(week), 7)))}>
          <ChevronRight size={15} />
        </button>
      </div>

      <div className="cal-times-free" aria-live="polite">
        {free === null ? (
          <p className="form-hint">Finding free times…</p>
        ) : failed ? (
          <p className="card-warn">The free times could not be read. Pick a time yourself below.</p>
        ) : times.length === 0 ? (
          <p className="form-hint">{closed(date) ? `Closed on ${DAY_NAMES[weekdayOf(parseYmd(date))]}s.` : date < today ? 'That day is past.' : 'Nothing free that day for this.'} Pick another day, or another time below.</p>
        ) : (
          groups.map((g) => (
            <div key={g.part} className="cal-time-group">
              <span>{g.label}</span>
              <div role="radiogroup" aria-label={`${g.label} times`}>
                {g.times.map((t) => (
                  <button
                    type="button"
                    key={t}
                    role="radio"
                    aria-checked={!other && t === time}
                    className={`cal-chip${!other && t === time ? ' is-on' : ''}`}
                    onClick={() => {
                      aim.current = t;
                      setOther(false);
                      onPick(date, t);
                    }}
                  >
                    {sayTime(t)}
                    {current && current.date === date && current.time === t ? <em> now</em> : null}
                  </button>
                ))}
              </div>
            </div>
          ))
        )}
        {missed && !other && <p className="form-hint">{sayTime(aim.current!)} is not free for this; the nearest free time is picked.</p>}
      </div>

      <div className="cal-other-time">
        {other ? (
          <label className="form-row">
            <span>Another time</span>
            <input
              type="time"
              step={300}
              value={time ?? ''}
              onChange={(e) => onPick(date, e.target.value || null)}
              aria-describedby="cal-other-note"
            />
            <small id="cal-other-note" className="form-hint">
              Outside the hours or already taken: callers are not offered it, but you can book it here.
            </small>
          </label>
        ) : (
          <button type="button" className="btn-link" onClick={() => setOther(true)}>
            Another time…
          </button>
        )}
        {other && (
          <button type="button" className="btn-link" onClick={() => setOther(false)}>
            Back to the free times
          </button>
        )}
        <label className="cal-date-jump">
          <span>Or a date</span>
          <input type="date" value={date} min={today} onChange={(e) => e.target.value && onPick(e.target.value, null)} />
        </label>
      </div>
    </div>
  );
}

// ---- a booking: new, or a change -------------------------------------------------------

export interface BookingValues {
  service: string;
  date: string;
  time: string;
  name: string;
  phone: string;
  notes: string;
}

export function BookingForm({
  settings,
  initial,
  wanted,
  appointment,
  submitLabel,
  onSubmit,
  onCancel,
}: {
  settings: CalendarSettings;
  initial: Partial<BookingValues> & { date: string };
  wanted?: string | null;
  /** The appointment being changed (none for a new one). */
  appointment?: Appointment;
  submitLabel: string;
  /** `confirm`: a request changed and confirmed in one go. */
  onSubmit: (v: BookingValues, confirm: boolean) => Promise<void>;
  onCancel: () => void;
}) {
  const services = settings.services;
  const [service, setService] = useState(initial.service ?? services[0]?.name ?? '');
  const [date, setDate] = useState(initial.date);
  const [time, setTime] = useState<string | null>(initial.time ?? null);
  const [name, setName] = useState(initial.name ?? '');
  const [phone, setPhone] = useState(initial.phone ?? '');
  const [notes, setNotes] = useState(initial.notes ?? '');
  const [busy, setBusy] = useState(false);
  const chosen = services.find((s) => s.name === service);
  // A changed appointment keeps its own length until its service changes.
  const minutes = appointment && appointment.service === service ? appointment.minutes : chosen?.minutes;
  const current = appointment ? { date: dateOf(appointment.start), time: timeOf(appointment.start) } : undefined;
  const end = time && minutes ? endOf({ start: `${date}T${time}`, minutes }) : null;

  const submit = async (confirm: boolean) => {
    if (!time) return;
    setBusy(true);
    try {
      await onSubmit({ service, date, time, name, phone, notes }, confirm);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      className="cal-booking"
      onSubmit={(e) => {
        e.preventDefault();
        void submit(false);
      }}
    >
      <label className="form-row">
        <span>Service</span>
        {services.length ? (
          <select value={service} onChange={(e) => setService(e.target.value)}>
            {!chosen && service && <option value={service}>{service}</option>}
            {services.map((s) => (
              <option key={s.id || s.name} value={s.name}>
                {s.name} · {sayMinutes(s.minutes)}
                {s.price ? ` · ${s.price}` : ''}
              </option>
            ))}
          </select>
        ) : (
          <input value={service} placeholder="What it is for" onChange={(e) => setService(e.target.value)} />
        )}
        {minutes ? (
          <small className="cal-takes">
            Takes {sayMinutes(minutes)}
            {chosen?.description ? ` · ${chosen.description}` : ''}
          </small>
        ) : null}
      </label>

      <fieldset className="cal-when">
        <legend>When</legend>
        <SlotPicker
          settings={settings}
          service={service}
          minutes={appointment && appointment.service === service ? appointment.minutes : undefined}
          date={date}
          time={time}
          wanted={wanted ?? initial.time ?? null}
          current={current}
          onPick={(d, t) => {
            setDate(d);
            setTime(t);
          }}
        />
        <p className="cal-when-sum" aria-live="polite">
          {time ? (
            <>
              <strong>{sayLongDay(date)}</strong>, {end ? sayRange(time, end) : sayTime(time)}
            </>
          ) : (
            'Pick a time'
          )}
        </p>
      </fieldset>

      <div className="form-row-pair">
        <label className="form-row">
          <span>Name</span>
          <input value={name} autoComplete="off" placeholder="Who it is for" onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="form-row">
          <span>Phone</span>
          <input value={phone} type="tel" autoComplete="off" placeholder="For the confirmation text" onChange={(e) => setPhone(e.target.value)} />
        </label>
      </div>
      <label className="form-row">
        <span>Notes</span>
        <textarea rows={2} value={notes} placeholder="Anything to know before it" onChange={(e) => setNotes(e.target.value)} />
      </label>

      <div className="cal-side-actions">
        {appointment?.status === 'requested' && (
          <button type="button" className="btn btn-primary" disabled={!time || busy} onClick={() => void submit(true)}>
            <Check size={14} /> Save and confirm
          </button>
        )}
        <button type="submit" className={appointment?.status === 'requested' ? 'btn btn-secondary' : 'btn btn-primary'} disabled={!time || busy}>
          {!appointment && <CalendarPlus size={14} />} {busy ? 'Saving…' : submitLabel}
        </button>
        <button type="button" className="btn btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

// ---- an appointment's details ------------------------------------------------------------

export function AppointmentDetails({
  a,
  settings,
  canText,
  onDecide,
  onChange,
  onDelete,
}: {
  a: Appointment;
  settings: CalendarSettings;
  /** The phone is there to text from. */
  canText: boolean;
  onDecide: (a: Appointment, status: AppointmentStatus, textThem: boolean) => Promise<void>;
  onChange: () => void;
  onDelete: () => void;
}) {
  const [textThem, setTextThem] = useState(canText && settings.textConfirmations && !!a.phone);
  const [busy, setBusy] = useState<AppointmentStatus | null>(null);
  useEffect(() => setTextThem(canText && settings.textConfirmations && !!a.phone), [a.id, a.phone, canText, settings.textConfirmations]);
  const date = dateOf(a.start);
  const from = timeOf(a.start);
  const decide = async (status: AppointmentStatus, text = false) => {
    setBusy(status);
    try {
      await onDecide(a, status, text);
    } finally {
      setBusy(null);
    }
  };
  const created = new Date(a.createdAt);
  const source = SOURCE_TEXT[a.source] ?? `from ${a.source}`;
  const struck = a.status === 'cancelled' || a.status === 'declined';

  return (
    <div className={`cal-details is-${a.status}`}>
      <p className={`cal-details-when${struck ? ' is-struck' : ''}`}>
        <strong>{sayLongDay(date)}</strong>
        <span>
          {sayRange(from, endOf(a))} · {sayMinutes(a.minutes)}
        </span>
      </p>

      {a.status === 'requested' && (
        <div className="cal-ask">
          <p>
            Asked for {source}. It keeps this time for them until you confirm or decline it.
          </p>
          <div className="cal-side-actions">
            <button type="button" className="btn btn-primary" disabled={!!busy} onClick={() => void decide('confirmed', textThem)}>
              <Check size={14} /> {busy === 'confirmed' ? 'Confirming…' : 'Confirm'}
            </button>
            <button type="button" className="btn btn-secondary" disabled={!!busy} onClick={() => void decide('declined')}>
              <X size={14} /> Decline
            </button>
            <button type="button" className="btn btn-ghost" disabled={!!busy} onClick={onChange}>
              <Pencil size={13} /> Change
            </button>
          </div>
          {canText && a.phone ? (
            <label className="cal-check">
              <input type="checkbox" checked={textThem} onChange={(e) => setTextThem(e.target.checked)} />
              <MessageSquare size={13} /> Text {a.name || 'them'} the confirmation
            </label>
          ) : (
            <p className="form-hint">{a.phone ? 'The phone is off, so no text is sent.' : 'No number, so no text is sent.'}</p>
          )}
        </div>
      )}

      <dl className="cal-facts">
        <div>
          <dt>Who</dt>
          <dd>{a.name || <span className="cal-none">No name given</span>}</dd>
        </div>
        <div>
          <dt>Phone</dt>
          <dd>
            {a.phone ? (
              <a href={`tel:${a.phone}`} className="cal-tel">
                <Phone size={12} /> {a.phone}
              </a>
            ) : (
              <span className="cal-none">None</span>
            )}
          </dd>
        </div>
        {a.notes && (
          <div className="is-wide">
            <dt>Notes</dt>
            <dd className="cal-notes">{a.notes}</dd>
          </div>
        )}
        <div className="is-wide">
          <dt>Booked</dt>
          <dd>
            {a.source === 'manual' ? 'Here' : source.charAt(0).toUpperCase() + source.slice(1)}
            {Number.isNaN(created.getTime()) ? '' : `, ${created.getDate()} ${MONTHS[created.getMonth()]} at ${sayTime(`${created.getHours()}:${String(created.getMinutes()).padStart(2, '0')}`)}`}
            {a.callId ? '. The call is in the Agent’s call sessions.' : ''}
          </dd>
        </div>
      </dl>

      <div className="cal-side-actions cal-side-more">
        {a.status !== 'requested' && (
          <button type="button" className="btn btn-secondary" onClick={onChange}>
            <Pencil size={13} /> Change
          </button>
        )}
        {a.status === 'confirmed' && (
          <>
            <button type="button" className="btn btn-ghost" disabled={!!busy} onClick={() => void decide('done')}>
              <CircleCheck size={14} /> Mark done
            </button>
            <button type="button" className="btn btn-ghost" disabled={!!busy} onClick={() => void decide('cancelled')}>
              <Ban size={14} /> Cancel it
            </button>
          </>
        )}
        {struck && (
          <button type="button" className="btn btn-ghost" disabled={!!busy} onClick={() => void decide('confirmed')}>
            <Undo2 size={14} /> Book it again
          </button>
        )}
        <button type="button" className="btn-tiny btn-danger cal-delete" onClick={onDelete} title="Delete it: cancelling keeps a record, deleting does not" aria-label="Delete this appointment">
          <Trash2 size={13} /> Delete
        </button>
      </div>
    </div>
  );
}
