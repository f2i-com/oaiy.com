import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CalendarPlus, Check, ChevronLeft, ChevronRight, MessageSquare, Phone, Plus, RefreshCw, Trash2, Volume2, X } from 'lucide-react';
import { calendar, voices, type Appointment, type AppointmentStatus, type CalendarService, type CalendarSettings, type CalendarSync, type NewAppointment, type VoiceClip } from './api';
import { useToast } from './Toasts';
import { describeSync } from './syncStatus';

/**
 * The calendar: the week's appointments, the requests waiting for someone to
 * confirm them (the phone takes them on calls, the agent in texts), and the
 * business's hours and services, which the phone reads when a caller asks what
 * is free. It all lives on this machine; FormLogic syncs with it when linked.
 */

const POLL_MS = 10_000;
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
/** One hour of the week grid, in pixels. */
const HOUR_PX = 52;

type Tab = 'week' | 'requests' | 'settings';

const pad = (n: number) => String(n).padStart(2, '0');
const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const parseYmd = (s: string) => {
  const [y, m, d] = s.split('-').map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
};
const addDays = (d: Date, n: number) => new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
const mondayOf = (d: Date) => addDays(d, -((d.getDay() + 6) % 7));
const minutesOf = (hhmm: string) => {
  const [h, m] = hhmm.split(':').map(Number);
  return (h ?? 0) * 60 + (m ?? 0);
};
/** "9:30 am". */
function sayTime(hhmm: string): string {
  const m = minutesOf(hhmm);
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h % 12 || 12}${mm ? `:${pad(mm)}` : ''} ${h < 12 ? 'am' : 'pm'}`;
}
/** "Tue 29 Sep, 10 am". */
function sayWhen(start: string): string {
  const [date, time] = start.split('T');
  const d = parseYmd(date);
  return `${DAYS[(d.getDay() + 6) % 7]} ${d.getDate()} ${d.toLocaleString(undefined, { month: 'short' })}, ${sayTime(time ?? '00:00')}`;
}

const STATUS_LABEL: Record<AppointmentStatus, string> = {
  requested: 'requested',
  confirmed: 'confirmed',
  declined: 'declined',
  cancelled: 'cancelled',
  done: 'done',
};
function badgeFor(s: AppointmentStatus): string {
  return s === 'confirmed' ? 'badge badge-ok' : s === 'requested' ? 'badge badge-pending' : s === 'done' ? 'badge badge-neutral' : 'badge badge-err';
}
const SOURCE_LABEL: Record<string, string> = { call: 'a call', text: 'a text', agent: 'the agent', manual: 'here', formlogic: 'FormLogic' };

export default function CalendarPanel() {
  const toast = useToast();
  const [tab, setTab] = useState<Tab>('week');
  const [week, setWeek] = useState(() => mondayOf(new Date()));
  const [settings, setSettings] = useState<CalendarSettings | null>(null);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [requests, setRequests] = useState<Appointment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Appointment | null>(null);
  const [adding, setAdding] = useState<Partial<NewAppointment> | null>(null);
  const [sync, setSync] = useState<CalendarSync | null>(null);
  const [syncing, setSyncing] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [wk, all] = await Promise.all([calendar.get(ymd(week), ymd(addDays(week, 7))), calendar.get(ymd(new Date()))]);
      setSettings((s) => s ?? wk.settings);
      setAppointments(wk.appointments);
      setRequests(all.appointments.filter((a) => a.status === 'requested'));
      setSync(await calendar.syncStatus().catch(() => null));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [week]);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  /** Confirm or decline a request; a confirmation is texted to them when that is on and they have a number. */
  const decide = async (a: Appointment, status: AppointmentStatus, textThem: boolean) => {
    try {
      const updated = await calendar.update(a.id, { status });
      if (status === 'confirmed' && textThem && a.phone) {
        const who = settings?.business ? ` with ${settings.business}` : '';
        await calendar
          .text(a.phone, `Hi${a.name ? ` ${a.name}` : ''}, your ${a.service || 'appointment'}${who} is confirmed for ${sayWhen(updated.start)}. See you then!`)
          .then(() => toast.push({ kind: 'success', title: 'Confirmed', body: `Texted ${a.phone}.` }))
          .catch((e) => toast.push({ kind: 'error', title: 'Confirmed, but the text did not go', body: String(e instanceof Error ? e.message : e) }));
      } else {
        toast.push({ kind: 'success', title: status === 'confirmed' ? 'Confirmed' : `Marked ${STATUS_LABEL[status]}` });
      }
      setOpen(null);
      await refresh();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not change it', body: String(e instanceof Error ? e.message : e) });
    }
  };

  const syncView = useMemo(() => describeSync(sync, null), [sync]);
  const everyone = useMemo(() => [...appointments, ...requests], [appointments, requests]);
  const days = useMemo(() => Array.from({ length: 7 }, (_, i) => addDays(week, i)), [week]);
  // The grid spans the opening hours (with an hour either side), or 8 to 6.
  const [firstHour, lastHour] = useMemo(() => {
    const spans = settings?.hours.flat() ?? [];
    const starts = [...spans.map((s) => minutesOf(s.open)), ...appointments.map((a) => minutesOf(a.start.split('T')[1] ?? '09:00'))];
    const ends = [...spans.map((s) => minutesOf(s.close)), ...appointments.map((a) => minutesOf(a.start.split('T')[1] ?? '09:00') + a.minutes)];
    const first = starts.length ? Math.max(0, Math.floor(Math.min(...starts) / 60) - 1) : 8;
    const last = ends.length ? Math.min(24, Math.ceil(Math.max(...ends) / 60) + 1) : 18;
    return [first, Math.max(last, first + 4)];
  }, [settings, appointments]);
  const today = ymd(new Date());

  return (
    <div className="panel calendar">
      {error && <div className="banner banner-err">⚠ {error}</div>}
      {sync?.linked && (sync.problems?.length ?? 0) > 0 && (
        <div className="banner banner-err" role="status">
          FormLogic would not take {sync.problems!.length === 1 ? 'an appointment' : `${sync.problems!.length} appointments`}:{' '}
          {sync.problems!
            .slice(0, 3)
            .map((p) => {
              const a = everyone.find((x) => x.id === p.id);
              return `${a ? `${a.service || 'Appointment'}, ${sayWhen(a.start)}` : 'one'} (${p.message})`;
            })
            .join('; ')}
          . It is kept here, and sent again once it is changed.
        </div>
      )}

      <div className="section-title-row">
        <div className="seg">
          <button className={tab === 'week' ? 'active' : ''} onClick={() => setTab('week')}>
            Week
          </button>
          <button className={tab === 'requests' ? 'active' : ''} onClick={() => setTab('requests')}>
            Requests{requests.length ? ` (${requests.length})` : ''}
          </button>
          <button className={tab === 'settings' ? 'active' : ''} onClick={() => setTab('settings')}>
            Hours &amp; services
          </button>
        </div>
        {tab === 'week' && (
          <div className="cal-weeknav">
            <button className="btn-tiny" onClick={() => setWeek(addDays(week, -7))} aria-label="Previous week">
              <ChevronLeft size={14} />
            </button>
            <button className="btn-tiny" onClick={() => setWeek(mondayOf(new Date()))}>
              Today
            </button>
            <button className="btn-tiny" onClick={() => setWeek(addDays(week, 7))} aria-label="Next week">
              <ChevronRight size={14} />
            </button>
            <span className="cal-range">
              {week.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} – {addDays(week, 6).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}
            </span>
          </div>
        )}
        <button className="btn-tiny" onClick={() => void refresh()} title="Refresh now">
          <RefreshCw size={13} />
        </button>
        {syncView && sync && (
          <span
            className={`cal-sync${syncView.tone === 'err' || syncView.tone === 'warn' ? ' cal-sync-err' : ''}`}
            title={[
              sync.error,
              sync.nextAttemptAt && sync.state !== 'synced' ? `Next try ${new Date(sync.nextAttemptAt).toLocaleTimeString()}` : null,
              sync.at ? `Last sync: ${sync.pulled} in, ${sync.pushed} out` : null,
              'The calendar works without FormLogic; changes made here are sent when it can be reached.',
            ]
              .filter(Boolean)
              .join('\n')}
          >
            FormLogic: {syncView.headline} · {syncView.detail}
            <button
              className="btn-tiny"
              disabled={syncing}
              onClick={async () => {
                setSyncing(true);
                try {
                  setSync(await calendar.syncNow());
                  await refresh();
                } catch (e) {
                  toast.push({ kind: 'error', title: 'Could not sync now', body: String(e instanceof Error ? e.message : e) });
                } finally {
                  setSyncing(false);
                }
              }}
            >
              {syncing ? 'Syncing…' : 'Sync now'}
            </button>
          </span>
        )}
        <button className="btn btn-primary" onClick={() => setAdding({ date: today })}>
          <CalendarPlus size={14} /> New appointment
        </button>
      </div>

      {tab === 'week' && (
        <div className="cal-grid" style={{ ['--hours' as string]: lastHour - firstHour, ['--hour-px' as string]: `${HOUR_PX}px` }}>
          <div className="cal-times">
            <div className="cal-dayhead" />
            {Array.from({ length: lastHour - firstHour }, (_, i) => (
              <div key={i} className="cal-time">
                {sayTime(`${pad(firstHour + i)}:00`)}
              </div>
            ))}
          </div>
          {days.map((d, i) => {
            const date = ymd(d);
            const spans = settings?.hours[i] ?? [];
            const mine = appointments.filter((a) => a.start.startsWith(date) && a.status !== 'declined' && a.status !== 'cancelled');
            return (
              <div key={date} className={`cal-day${date === today ? ' cal-today' : ''}`}>
                <div className="cal-dayhead">
                  <span>{DAYS[i]}</span> <strong>{d.getDate()}</strong>
                  {spans.length === 0 && <em>closed</em>}
                </div>
                <div className="cal-body" onDoubleClick={() => setAdding({ date })}>
                  {spans.map((s, k) => (
                    <div key={k} className="cal-open" style={{ top: ((minutesOf(s.open) - firstHour * 60) / 60) * HOUR_PX, height: ((minutesOf(s.close) - minutesOf(s.open)) / 60) * HOUR_PX }} />
                  ))}
                  {mine.map((a) => {
                    const top = ((minutesOf(a.start.split('T')[1] ?? '00:00') - firstHour * 60) / 60) * HOUR_PX;
                    return (
                      <button key={a.id} className={`cal-appt cal-${a.status}`} style={{ top, height: Math.max(34, (a.minutes / 60) * HOUR_PX - 2) }} onClick={() => setOpen(a)} title={`${a.service} — ${a.name || 'no name'}`}>
                        <strong>{sayTime(a.start.split('T')[1] ?? '00:00')}</strong> {a.service}
                        {a.name && <span>{a.name}</span>}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {tab === 'requests' && (
        <section className="service-section">
          {requests.length === 0 ? (
            <div className="empty-state">No requests waiting. Appointments agreed on calls and in texts come here for you to confirm.</div>
          ) : (
            <ul className="run-list">
              {requests.map((a) => (
                <RequestRow key={a.id} a={a} textByDefault={settings?.textConfirmations ?? true} onDecide={decide} onOpen={() => setOpen(a)} />
              ))}
            </ul>
          )}
        </section>
      )}

      {tab === 'settings' && settings && (
        <>
          <SettingsForm
            settings={settings}
            onSaved={(s) => {
              setSettings(s);
              toast.push({ kind: 'success', title: 'Saved', body: 'The phone uses these hours and services from now on.' });
              void refresh();
            }}
          />
          <CallVoice business={settings.business} />
        </>
      )}

      {open && <AppointmentDialog a={open} settings={settings} onClose={() => setOpen(null)} onDecide={decide} onChanged={() => void refresh()} />}
      {adding && settings && (
        <NewAppointmentDialog
          start={adding}
          settings={settings}
          onClose={() => setAdding(null)}
          onCreated={(a) => {
            setAdding(null);
            setWeek(mondayOf(parseYmd(a.start.split('T')[0])));
            toast.push({ kind: 'success', title: 'Booked', body: `${a.service}, ${sayWhen(a.start)}` });
            void refresh();
          }}
        />
      )}
    </div>
  );
}

function RequestRow({ a, textByDefault, onDecide, onOpen }: { a: Appointment; textByDefault: boolean; onDecide: (a: Appointment, s: AppointmentStatus, text: boolean) => void; onOpen: () => void }) {
  const [textThem, setTextThem] = useState(textByDefault && !!a.phone);
  return (
    <li className="run-row cal-request">
      <div className="run-head">
        <span className={badgeFor(a.status)}>{STATUS_LABEL[a.status]}</span>
        <strong className="run-flow">
          {a.service || 'Appointment'} · {sayWhen(a.start)}
        </strong>
        <span className="run-meta">
          {a.name || 'no name'}
          {a.phone ? ` · ${a.phone}` : ''} · from {SOURCE_LABEL[a.source] ?? a.source}
        </span>
      </div>
      {a.notes && <p className="form-hint">{a.notes}</p>}
      <div className="row-actions">
        <button className="btn btn-primary" onClick={() => onDecide(a, 'confirmed', textThem)}>
          <Check size={14} /> Confirm
        </button>
        <button className="btn btn-secondary" onClick={() => onDecide(a, 'declined', false)}>
          <X size={14} /> Decline
        </button>
        <button className="btn btn-ghost" onClick={onOpen}>
          Change…
        </button>
        {a.phone && (
          <label className="cal-check">
            <input type="checkbox" checked={textThem} onChange={(e) => setTextThem(e.target.checked)} /> <MessageSquare size={13} /> text them the confirmation
          </label>
        )}
      </div>
    </li>
  );
}

function AppointmentDialog({ a, settings, onClose, onDecide, onChanged }: { a: Appointment; settings: CalendarSettings | null; onClose: () => void; onDecide: (a: Appointment, s: AppointmentStatus, text: boolean) => void; onChanged: () => void }) {
  const toast = useToast();
  const [date, setDate] = useState(a.start.split('T')[0]);
  const [time, setTime] = useState(a.start.split('T')[1] ?? '09:00');
  const [service, setService] = useState(a.service);
  const [name, setName] = useState(a.name);
  const [phone, setPhone] = useState(a.phone);
  const [notes, setNotes] = useState(a.notes);
  const save = async () => {
    try {
      await calendar.update(a.id, { start: `${date}T${time}`, service, name, phone, notes });
      toast.push({ kind: 'success', title: 'Saved' });
      onChanged();
      onClose();
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not save', body: String(e instanceof Error ? e.message : e) });
    }
  };
  const remove = async () => {
    if (!window.confirm('Delete this appointment? (Cancelling keeps a record; deleting does not.)')) return;
    await calendar.remove(a.id).catch(() => {});
    onChanged();
    onClose();
  };
  return (
    <div className="cal-dialog-back" onClick={onClose}>
      <div className="cal-dialog" role="dialog" aria-label="Appointment" onClick={(e) => e.stopPropagation()}>
        <div className="section-title-row">
          <h3 className="section-title">{a.service || 'Appointment'}</h3>
          <span className={badgeFor(a.status)}>{STATUS_LABEL[a.status]}</span>
          <button className="btn-tiny" onClick={onClose} aria-label="Close">
            <X size={14} />
          </button>
        </div>
        <p className="form-hint">
          From {SOURCE_LABEL[a.source] ?? a.source}
          {a.callId ? ' (the call is in the agent’s call sessions)' : ''}. Booked {new Date(a.createdAt).toLocaleString()}.
        </p>
        <div className="form-row-pair">
          <label className="form-row">
            <span>DATE</span>
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
          </label>
          <label className="form-row">
            <span>TIME</span>
            <input type="time" value={time} step={300} onChange={(e) => setTime(e.target.value)} />
          </label>
        </div>
        <label className="form-row">
          <span>SERVICE</span>
          <input list="cal-services" value={service} onChange={(e) => setService(e.target.value)} />
          <datalist id="cal-services">{settings?.services.map((s) => <option key={s.name} value={s.name} />)}</datalist>
        </label>
        <div className="form-row-pair">
          <label className="form-row">
            <span>NAME</span>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="form-row">
            <span>PHONE</span>
            <input value={phone} onChange={(e) => setPhone(e.target.value)} />
          </label>
        </div>
        <label className="form-row">
          <span>NOTES</span>
          <textarea rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <div className="form-actions cal-dialog-actions">
          {a.status === 'requested' && (
            <button className="btn btn-primary" onClick={() => onDecide(a, 'confirmed', !!settings?.textConfirmations)}>
              <Check size={14} /> Confirm
            </button>
          )}
          <button className="btn btn-secondary" onClick={() => void save()}>
            Save changes
          </button>
          {a.status !== 'cancelled' && (
            <button className="btn btn-ghost" onClick={() => onDecide(a, 'cancelled', false)}>
              Cancel it
            </button>
          )}
          {a.status === 'confirmed' && (
            <button className="btn btn-ghost" onClick={() => onDecide(a, 'done', false)}>
              Mark done
            </button>
          )}
          <button className="btn-tiny btn-danger" onClick={() => void remove()} title="Delete" aria-label="Delete">
            <Trash2 size={13} />
          </button>
          {a.phone && (
            <a className="btn btn-ghost" href={`tel:${a.phone}`}>
              <Phone size={14} /> {a.phone}
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

function NewAppointmentDialog({ start, settings, onClose, onCreated }: { start: Partial<NewAppointment>; settings: CalendarSettings; onClose: () => void; onCreated: (a: Appointment) => void }) {
  const toast = useToast();
  const [service, setService] = useState(start.service ?? settings.services[0]?.name ?? '');
  const [date, setDate] = useState(start.date ?? ymd(new Date()));
  const [time, setTime] = useState(start.time ?? '');
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [notes, setNotes] = useState('');
  const [free, setFree] = useState<string[] | null>(null);
  useEffect(() => {
    let live = true;
    calendar
      .free(date, 1, service || undefined)
      .then((r) => live && setFree(r.days[0]?.times ?? []))
      .catch(() => live && setFree(null));
    return () => {
      live = false;
    };
  }, [date, service]);
  const create = async () => {
    try {
      onCreated(await calendar.create({ service, date, time, name, phone, notes, source: 'manual', status: 'confirmed' }));
    } catch (e) {
      toast.push({ kind: 'error', title: 'Could not book it', body: String(e instanceof Error ? e.message : e) });
    }
  };
  return (
    <div className="cal-dialog-back" onClick={onClose}>
      <div className="cal-dialog" role="dialog" aria-label="New appointment" onClick={(e) => e.stopPropagation()}>
        <div className="section-title-row">
          <h3 className="section-title">New appointment</h3>
          <button className="btn-tiny" onClick={onClose} aria-label="Close">
            <X size={14} />
          </button>
        </div>
        <label className="form-row">
          <span>SERVICE</span>
          <select value={service} onChange={(e) => setService(e.target.value)}>
            {settings.services.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} ({s.minutes} min)
              </option>
            ))}
          </select>
        </label>
        <div className="form-row-pair">
          <label className="form-row">
            <span>DATE</span>
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
          </label>
          <label className="form-row">
            <span>TIME</span>
            <input type="time" value={time} step={300} onChange={(e) => setTime(e.target.value)} />
          </label>
        </div>
        <div className="cal-free">
          {free === null ? null : free.length === 0 ? (
            <span className="form-hint">Nothing free that day.</span>
          ) : (
            free.slice(0, 24).map((t) => (
              <button key={t} className={`cal-chip${t === time ? ' active' : ''}`} onClick={() => setTime(t)}>
                {sayTime(t)}
              </button>
            ))
          )}
        </div>
        <div className="form-row-pair">
          <label className="form-row">
            <span>NAME</span>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="form-row">
            <span>PHONE</span>
            <input value={phone} onChange={(e) => setPhone(e.target.value)} />
          </label>
        </div>
        <label className="form-row">
          <span>NOTES</span>
          <textarea rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <div className="form-actions">
          <button className="btn btn-primary" disabled={!date || !time} onClick={() => void create()}>
            <CalendarPlus size={14} /> Book it
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * The voice the phone speaks in: a clip of someone speaking (MP3, WAV...),
 * cloned by OAIY's own speech engine on this machine's GPU. Each can be heard
 * first; a new one needs only its clip (what it says is heard from it).
 */
function CallVoice({ business }: { business: string }) {
  const toast = useToast();
  const [list, setList] = useState<VoiceClip[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [words, setWords] = useState('');
  const [adding, setAdding] = useState(false);
  const player = useRef<HTMLAudioElement | null>(null);
  const fail = (title: string, e: unknown) => toast.push({ kind: 'error', title, body: String(e instanceof Error ? e.message : e) });

  const load = useCallback(async () => {
    try {
      const r = await voices.list();
      setList(r.voices);
      setChosen(r.chosen);
    } catch (e) {
      fail('Voices not read', e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    void load();
    return () => player.current?.pause();
  }, [load]);

  const hear = async (voice: string) => {
    setSpeaking(voice);
    try {
      const blob = await voices.hear(voice, `Hi, thanks for calling${business.trim() ? ` ${business.trim()}` : ''}! How can I help you today?`);
      player.current?.pause();
      const audio = new Audio(URL.createObjectURL(blob));
      player.current = audio;
      await audio.play();
    } catch (e) {
      fail('Could not speak', e);
    } finally {
      setSpeaking(null);
    }
  };
  const choose = async (voice: string) => {
    try {
      setChosen((await voices.choose(voice)).chosen);
    } catch (e) {
      fail('Not chosen', e);
    }
  };
  const remove = async (voice: string) => {
    try {
      setChosen((await voices.remove(voice)).chosen);
      setList((l) => l.filter((v) => v.name !== voice));
    } catch (e) {
      fail('Not removed', e);
    }
  };
  const add = async () => {
    if (!file) return;
    setAdding(true);
    try {
      const r = await voices.add(name.trim() || file.name.replace(/\.[^.]+$/, ''), file, words, true);
      toast.push({ kind: 'success', title: 'Voice added', body: `Calls are answered as “${r.voice.name}” now. Hear it to check it sounds right.` });
      setFile(null);
      setName('');
      setWords('');
      await load();
    } catch (e) {
      fail('Voice not added', e);
    } finally {
      setAdding(false);
    }
  };

  return (
    <section className="service-section cal-settings cal-voice">
      <h3 className="section-title">Voice on calls</h3>
      <p className="form-hint">
        The phone speaks in the voice of a short clip: 5 to 30 seconds of one person speaking clearly, as an MP3 or WAV. OAIY copies the voice on this computer and hears what the clip says itself.
      </p>
      <ul className="cal-voices">
        {list.map((v) => (
          <li key={v.name} className={v.name === chosen ? 'chosen' : ''}>
            <label className="cal-check">
              <input type="radio" name="call-voice" checked={v.name === chosen} onChange={() => void choose(v.name)} /> <strong>{v.name}</strong>
            </label>
            <span className="form-hint">{v.name === chosen ? 'Answers calls' : `${Math.max(1, Math.round(v.bytes / 1024))} KB`}</span>
            <button className="btn btn-ghost" disabled={speaking !== null} onClick={() => void hear(v.name)}>
              <Volume2 size={14} /> {speaking === v.name ? 'Speaking…' : 'Hear it'}
            </button>
            <button className="btn-tiny btn-danger" disabled={list.length < 2} title={list.length < 2 ? 'The phone needs one voice' : 'Remove this voice'} aria-label={`Remove ${v.name}`} onClick={() => void remove(v.name)}>
              <Trash2 size={13} />
            </button>
          </li>
        ))}
      </ul>
      <div className="cal-voice-add">
        <label className="form-row">
          <span>A CLIP</span>
          <input type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.opus" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <label className="form-row">
          <span>NAME</span>
          <input value={name} placeholder={file ? file.name.replace(/\.[^.]+$/, '') : 'Front desk'} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="form-row cal-voice-words">
          <span>WHAT IT SAYS (OPTIONAL)</span>
          <input value={words} placeholder="Heard from the clip when left empty" onChange={(e) => setWords(e.target.value)} />
        </label>
        <button className="btn btn-primary" disabled={!file || adding} onClick={() => void add()}>
          <Plus size={14} /> {adding ? 'Adding…' : 'Add and use it'}
        </button>
      </div>
    </section>
  );
}

function SettingsForm({ settings, onSaved }: { settings: CalendarSettings; onSaved: (s: CalendarSettings) => void }) {
  const toast = useToast();
  const [s, setS] = useState<CalendarSettings>(settings);
  const [saving, setSaving] = useState(false);
  const setDay = (i: number, open: boolean, from?: string, to?: string) => {
    const hours = s.hours.map((d) => [...d]);
    hours[i] = open ? [{ open: from ?? hours[i][0]?.open ?? '09:00', close: to ?? hours[i][0]?.close ?? '17:00' }] : [];
    setS({ ...s, hours });
  };
  const setService = (i: number, patch: Partial<CalendarService>) => setS({ ...s, services: s.services.map((x, k) => (k === i ? { ...x, ...patch } : x)) });
  const save = async () => {
    setSaving(true);
    try {
      const saved = await calendar.saveSettings(s);
      setS(saved);
      onSaved(saved);
    } catch (e) {
      toast.push({ kind: 'error', title: 'Not saved', body: String(e instanceof Error ? e.message : e) });
    } finally {
      setSaving(false);
    }
  };
  return (
    <section className="service-section cal-settings">
      <label className="form-row">
        <span>Business name</span>
        <input value={s.business} placeholder="As the receptionist says it" onChange={(e) => setS({ ...s, business: e.target.value })} />
      </label>

      <h3 className="section-title">Opening hours</h3>
      <div className="cal-hours">
        {DAY_NAMES.map((day, i) => {
          const span = s.hours[i]?.[0];
          return (
            <div key={day} className="cal-hours-row">
              <label className="cal-check">
                <input type="checkbox" checked={!!span} onChange={(e) => setDay(i, e.target.checked)} /> {day}
              </label>
              {span ? (
                <>
                  <input type="time" value={span.open} onChange={(e) => setDay(i, true, e.target.value, span.close)} />
                  <span className="form-hint">to</span>
                  <input type="time" value={span.close} onChange={(e) => setDay(i, true, span.open, e.target.value)} />
                </>
              ) : (
                <span className="form-hint">closed</span>
              )}
            </div>
          );
        })}
      </div>

      <h3 className="section-title">Services</h3>
      <p className="form-hint">What callers can book, and how long each takes. The phone offers times that fit.</p>
      <div className="cal-services">
        {s.services.map((svc, i) => (
          <div key={i} className="cal-service-row">
            <input value={svc.name} placeholder="Name" onChange={(e) => setService(i, { name: e.target.value })} />
            <input type="number" min={5} step={5} value={svc.minutes} title="Minutes" onChange={(e) => setService(i, { minutes: Number(e.target.value) })} />
            <span className="form-hint">min</span>
            <input value={svc.price ?? ''} placeholder="Price (optional)" onChange={(e) => setService(i, { price: e.target.value })} />
            <input value={svc.description ?? ''} placeholder="What it is (optional)" onChange={(e) => setService(i, { description: e.target.value })} />
            <button className="btn-tiny btn-danger" onClick={() => setS({ ...s, services: s.services.filter((_, k) => k !== i) })} aria-label="Remove service">
              <Trash2 size={13} />
            </button>
          </div>
        ))}
        <button className="btn btn-ghost" onClick={() => setS({ ...s, services: [...s.services, { name: '', minutes: 30 }] })}>
          <Plus size={14} /> Add a service
        </button>
      </div>

      <h3 className="section-title">Booking</h3>
      <div className="form-row-pair">
        <label className="form-row">
          <span>TIMES EVERY (MIN)</span>
          <input type="number" min={5} step={5} value={s.slotMinutes} onChange={(e) => setS({ ...s, slotMinutes: Number(e.target.value) })} />
        </label>
        <label className="form-row">
          <span>NOTICE (MIN)</span>
          <input type="number" min={0} step={15} value={s.noticeMinutes} onChange={(e) => setS({ ...s, noticeMinutes: Number(e.target.value) })} />
        </label>
        <label className="form-row">
          <span>UP TO (DAYS AHEAD)</span>
          <input type="number" min={1} max={366} value={s.horizonDays} onChange={(e) => setS({ ...s, horizonDays: Number(e.target.value) })} />
        </label>
      </div>
      <label className="cal-check">
        <input type="checkbox" checked={s.textConfirmations} onChange={(e) => setS({ ...s, textConfirmations: e.target.checked })} /> Text people when their appointment is confirmed
      </label>
      <div className="form-actions">
        <button className="btn btn-primary" disabled={saving} onClick={() => void save()}>
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>
    </section>
  );
}
