import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ArrowDown, ArrowUp, CalendarDays, Coffee, Copy, Plus, Trash2, Undo2, Volume2, X } from 'lucide-react';
import { calendar, voices, type CalendarService, type CalendarSettings, type CalendarSpan, type VoiceClip } from './api';
import {
  DAY_NAMES,
  DURATIONS,
  STEPS,
  checkSettings,
  copyToOpenDays,
  normalSettings,
  openDay,
  sameHoursWeekdays,
  sameSettings,
  sayHours,
  sayMinutes,
  withBreak,
  ymd,
} from './calendarModel';
import { useToast } from './Toasts';
import { moduleOn, useModules } from './useModules';
import { useVisiblePoll } from './useVisiblePoll';

/**
 * Hours & Services: the business the receptionist speaks for. Its name, when
 * it is open, what callers can book and how long each takes, and the rules
 * the free times follow; then how the phone sounds. The phone reads these
 * when a caller asks what is free, and the setup wizard's "Your business"
 * step is this same form.
 */

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function HoursPanel({ onOpenCalendar }: { onOpenCalendar?: () => void }) {
  const phoneOn = moduleOn(useModules(), 'phone') === true;
  const toast = useToast();
  const [settings, setSettings] = useState<CalendarSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => {
    try {
      const today = ymd(new Date());
      const got = await calendar.get(today, today);
      setSettings(got.settings);
      setError(null);
    } catch (e) {
      setError(errText(e));
    }
  }, []);
  // Asked again now and then: the Agent may change them too (an open form keeps its changes).
  useVisiblePoll(() => void load(), 15_000);

  return (
    <div className="panel hours-page">
      {error && <div className="banner banner-err">The business settings could not be read: {error}</div>}
      {settings ? (
        <SettingsForm
          settings={settings}
          onSaved={(s) => {
            setSettings(s);
            toast.push({ kind: 'success', title: 'Saved', body: 'The receptionist offers these hours and services from now on.' });
          }}
          after={phoneOn ? <CallVoice business={settings.business} /> : null}
          onOpenCalendar={onOpenCalendar}
        />
      ) : (
        !error && <div className="empty-state">Reading the business settings…</div>
      )}
    </div>
  );
}

// ---- the form ------------------------------------------------------------------------------

let nextKey = 1;
const newKey = () => `svc-${nextKey++}`;

/**
 * The business, its hours, its services and its booking rules, saved
 * together. On the page an unsaved-changes bar stays in view while anything
 * has changed; in the setup wizard (`embedded`) the form sits in the step
 * without cards of its own, and its Save is always there.
 */
export function SettingsForm({
  settings,
  onSaved,
  embedded = false,
  after,
  onOpenCalendar,
}: {
  settings: CalendarSettings;
  onSaved: (s: CalendarSettings) => void;
  embedded?: boolean;
  /** Shown after the form (How the phone sounds), outside what Save sends. */
  after?: ReactNode;
  onOpenCalendar?: () => void;
}) {
  const toast = useToast();
  const [base, setBase] = useState(() => normalSettings(settings));
  const [s, setS] = useState(base);
  const [keys, setKeys] = useState<string[]>(() => base.services.map(newKey));
  const [custom, setCustom] = useState<Set<string>>(() => new Set());
  const [touched, setTouched] = useState<Set<string>>(() => new Set());
  const [attempted, setAttempted] = useState(false);
  const [removed, setRemoved] = useState<{ svc: CalendarService; key: string; index: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const root = useRef<HTMLDivElement | null>(null);
  const dirty = !sameSettings(s, base);
  const problems = useMemo(() => checkSettings(s), [s]);
  const invalid = problems.list.length > 0;

  // New settings from the desktop (saved elsewhere, or by the Agent): taken while nothing here has changed.
  const incoming = JSON.stringify(normalSettings(settings));
  useEffect(() => {
    const next = normalSettings(settings);
    if (sameSettings(next, base)) return;
    if (!dirty) {
      setBase(next);
      setS(next);
      setKeys(next.services.map(newKey));
    } else setBase(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incoming]);

  useEffect(() => {
    if (!removed) return;
    const id = window.setTimeout(() => setRemoved(null), 10_000);
    return () => window.clearTimeout(id);
  }, [removed]);
  useEffect(() => {
    if (!justSaved) return;
    const id = window.setTimeout(() => setJustSaved(false), 3000);
    return () => window.clearTimeout(id);
  }, [justSaved]);

  const setHours = (hours: CalendarSpan[][]) => setS((x) => ({ ...x, hours }));
  const setSpan = (day: number, k: number, patch: Partial<CalendarSpan>) =>
    setHours(s.hours.map((d, i) => (i === day ? d.map((sp, j) => (j === k ? { ...sp, ...patch } : sp)) : d)));
  const setService = (i: number, patch: Partial<CalendarService>) => setS((x) => ({ ...x, services: x.services.map((v, k) => (k === i ? { ...v, ...patch } : v)) }));
  const touch = (key: string) => setTouched((t) => (t.has(key) ? t : new Set(t).add(key)));
  const moveService = (i: number, by: -1 | 1) => {
    const j = i + by;
    if (j < 0 || j >= s.services.length) return;
    const swap = <T,>(list: T[]) => {
      const next = [...list];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    };
    setS((x) => ({ ...x, services: swap(x.services) }));
    setKeys(swap);
    requestAnimationFrame(() => root.current?.querySelector<HTMLButtonElement>(`[data-svc="${keys[i]}"] .${by < 0 ? 'svc-up' : 'svc-down'}`)?.focus());
  };
  const removeService = (i: number) => {
    setRemoved({ svc: s.services[i], key: keys[i], index: i });
    setS((x) => ({ ...x, services: x.services.filter((_, k) => k !== i) }));
    setKeys((k) => k.filter((_, n) => n !== i));
  };
  const undoRemove = () => {
    if (!removed) return;
    const at = Math.min(removed.index, s.services.length);
    setS((x) => ({ ...x, services: [...x.services.slice(0, at), removed.svc, ...x.services.slice(at)] }));
    setKeys((k) => [...k.slice(0, at), removed.key, ...k.slice(at)]);
    setRemoved(null);
  };
  const addService = () => {
    const key = newKey();
    setS((x) => ({ ...x, services: [...x.services, { name: '', minutes: 30, price: '', description: '' }] }));
    setKeys((k) => [...k, key]);
    requestAnimationFrame(() => root.current?.querySelector<HTMLInputElement>(`[data-svc="${key}"] .svc-name`)?.focus());
  };

  const discard = () => {
    setS(base);
    setKeys(base.services.map(newKey));
    setCustom(new Set());
    setTouched(new Set());
    setAttempted(false);
    setRemoved(null);
  };

  const save = async () => {
    if (invalid) {
      setAttempted(true);
      // The first problem, brought into view (clear of the save bar) and focused.
      requestAnimationFrame(() => {
        const first = root.current?.querySelector<HTMLElement>('[aria-invalid="true"]');
        first?.scrollIntoView?.({ block: 'center', behavior: 'smooth' });
        first?.focus({ preventScroll: true });
      });
      return;
    }
    setSaving(true);
    try {
      const saved = normalSettings(await calendar.saveSettings(s));
      setBase(saved);
      setS(saved);
      setAttempted(false);
      setTouched(new Set());
      setRemoved(null);
      setJustSaved(true);
      onSaved(saved);
    } catch (e) {
      toast.push({ kind: 'error', title: 'Not saved', body: errText(e) });
    } finally {
      setSaving(false);
    }
  };

  const showSvc = (key: string) => attempted || touched.has(key);
  const openDays = s.hours.filter((d) => d.length).length;
  const firstOpen = s.hours.findIndex((d) => d.length > 0);
  const greeting = s.business.trim() ? `“Thanks for calling ${s.business.trim()}, how can I help?”` : '“Thanks for calling, how can I help?”';

  return (
    <div ref={root} className={`hours-form${embedded ? ' is-embedded' : ''}`}>
      <section className="hours-card hours-business" aria-labelledby="hours-business-title">
        <h3 className="section-title" id="hours-business-title">
          Business
        </h3>
        <label className="form-row">
          <span>Its name</span>
          <input className="hours-business-name" value={s.business} placeholder="As the receptionist says it" onChange={(e) => setS({ ...s, business: e.target.value })} />
        </label>
        <p className="form-hint">
          Callers hear: {greeting}
        </p>
      </section>

      <section className="hours-card hours-open" aria-labelledby="hours-open-title">
        <div className="hours-card-head">
          <h3 className="section-title" id="hours-open-title">
            Opening hours
          </h3>
          <div className="hours-quick">
            <button type="button" className="btn-tiny" onClick={() => setHours(sameHoursWeekdays(s.hours))} title="Open Monday to Friday, all with the hours of the first weekday that is open">
              Same hours Mon–Fri
            </button>
            {firstOpen >= 0 && openDays > 1 && (
              <button type="button" className="btn-tiny" onClick={() => setHours(copyToOpenDays(s.hours, firstOpen))} title="Every open day takes these hours; closed days stay closed. Each day’s copy button does the same from that day.">
                Copy {DAY_NAMES[firstOpen]} to all open days
              </button>
            )}
          </div>
        </div>
        <p className="form-hint hours-summary">{openDays ? sayHours(s.hours) : 'Closed every day: callers are offered no times.'}</p>
        <ul className="hours-days">
          {DAY_NAMES.map((day, i) => {
            const spans = s.hours[i] ?? [];
            const open = spans.length > 0;
            const dayProblems = problems.hours[i] ?? {};
            const canBreak = open && withBreak(spans) !== null && spans.length < 3;
            return (
              <li key={day} className={`hours-day${open ? '' : ' is-closed'}`}>
                <label className="hours-day-name">
                  <input
                    type="checkbox"
                    role="switch"
                    className="switch"
                    checked={open}
                    aria-label={`Open on ${day}s`}
                    onChange={(e) => setHours(e.target.checked ? openDay(s.hours, i) : s.hours.map((d, k) => (k === i ? [] : d)))}
                  />
                  <span>{day}</span>
                </label>
                <div className="hours-spans">
                  {open ? (
                    spans.map((sp, k) => (
                      <div key={k} className="hours-span">
                        <input
                          type="time"
                          value={sp.open}
                          aria-label={`${day}${k ? ` (${k + 1})` : ''}: opens at`}
                          aria-invalid={!!dayProblems[k]}
                          onChange={(e) => setSpan(i, k, { open: e.target.value })}
                        />
                        <span className="hours-to">to</span>
                        <input
                          type="time"
                          value={sp.close}
                          aria-label={`${day}${k ? ` (${k + 1})` : ''}: closes at`}
                          aria-invalid={!!dayProblems[k]}
                          onChange={(e) => setSpan(i, k, { close: e.target.value })}
                        />
                        {k > 0 && (
                          <button type="button" className="icon-button" aria-label={`Take away ${day}'s hours after the break`} title="Take away these hours" onClick={() => setHours(s.hours.map((d, n) => (n === i ? d.filter((_, j) => j !== k) : d)))}>
                            <X size={14} />
                          </button>
                        )}
                        {dayProblems[k] && (
                          <small className="hours-error" role="alert">
                            {dayProblems[k]}
                          </small>
                        )}
                      </div>
                    ))
                  ) : (
                    <span className="hours-closed">Closed</span>
                  )}
                </div>
                {open && (
                  <div className="hours-day-tools">
                    {canBreak && (
                      <button type="button" className="icon-button" aria-label={`Add a break on ${day}`} title="Add a break (lunch)" onClick={() => setHours(s.hours.map((d, n) => (n === i ? withBreak(d)! : d)))}>
                        <Coffee size={14} />
                      </button>
                    )}
                    <button type="button" className="icon-button" aria-label={`Copy ${day}'s hours to all open days`} title="Copy to all open days" onClick={() => setHours(copyToOpenDays(s.hours, i))}>
                      <Copy size={14} />
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <section className="hours-card hours-services" aria-labelledby="hours-services-title">
        <div className="hours-card-head">
          <h3 className="section-title" id="hours-services-title">
            Services <span className="section-count">{s.services.length}</span>
          </h3>
        </div>
        <p className="form-hint">What callers can book and how long each takes. The receptionist offers times each one fits.</p>
        {s.services.length === 0 && (
          <div className="empty-state empty-state-sm">
            <p>No services yet.</p>
            <p>Callers are offered {sayMinutes(s.slotMinutes || 30)} times until there is one.</p>
          </div>
        )}
        <ol className="svc-list">
          {s.services.map((svc, i) => {
            const key = keys[i] ?? `svc-at-${i}`;
            const p = problems.services[i] ?? {};
            const show = showSvc(key);
            const isCustom = custom.has(key) || !(DURATIONS as readonly number[]).includes(svc.minutes);
            return (
              <li key={key} className="svc-row" data-svc={key}>
                <div className="svc-order">
                  <button type="button" className="icon-button svc-up" aria-label={`Move ${svc.name || 'this service'} up`} disabled={i === 0} onClick={() => moveService(i, -1)}>
                    <ArrowUp size={13} />
                  </button>
                  <button type="button" className="icon-button svc-down" aria-label={`Move ${svc.name || 'this service'} down`} disabled={i === s.services.length - 1} onClick={() => moveService(i, 1)}>
                    <ArrowDown size={13} />
                  </button>
                </div>
                <div className="svc-fields">
                  <label className="svc-field svc-field-name">
                    <span>Name</span>
                    <input
                      className="svc-name"
                      value={svc.name}
                      placeholder="e.g. Lawn mowing"
                      aria-invalid={show && !!p.name}
                      onBlur={() => touch(key)}
                      onChange={(e) => setService(i, { name: e.target.value })}
                    />
                    {show && p.name && <small className="hours-error">{p.name}</small>}
                  </label>
                  <div className="svc-field svc-field-time">
                    <label htmlFor={`${key}-len`}>Takes</label>
                    <div className="svc-length">
                      <select
                        id={`${key}-len`}
                        value={isCustom ? 'custom' : String(svc.minutes)}
                        aria-invalid={show && !!p.minutes}
                        onChange={(e) => {
                          if (e.target.value === 'custom') setCustom((c) => new Set(c).add(key));
                          else {
                            setCustom((c) => {
                              const n = new Set(c);
                              n.delete(key);
                              return n;
                            });
                            setService(i, { minutes: Number(e.target.value) });
                          }
                        }}
                      >
                        {DURATIONS.map((m) => (
                          <option key={m} value={m}>
                            {sayMinutes(m)}
                          </option>
                        ))}
                        <option value="custom">Other…</option>
                      </select>
                      {isCustom && (
                        <span className="svc-minutes">
                          <input
                            type="number"
                            min={1}
                            max={1440}
                            step={5}
                            value={Number.isFinite(svc.minutes) && svc.minutes > 0 ? svc.minutes : ''}
                            aria-label="Minutes"
                            aria-invalid={show && !!p.minutes}
                            onBlur={() => touch(key)}
                            onChange={(e) => setService(i, { minutes: e.target.value === '' ? 0 : Number(e.target.value) })}
                          />
                          <span>min</span>
                        </span>
                      )}
                    </div>
                    {show && p.minutes && <small className="hours-error">{p.minutes}</small>}
                  </div>
                  <label className="svc-field svc-field-price">
                    <span>Price</span>
                    <input value={svc.price ?? ''} placeholder="Optional, e.g. from $60" onChange={(e) => setService(i, { price: e.target.value })} />
                  </label>
                  <label className="svc-field svc-field-about">
                    <span>What it is</span>
                    <input value={svc.description ?? ''} placeholder="Optional: what the receptionist can say about it" onChange={(e) => setService(i, { description: e.target.value })} />
                  </label>
                </div>
                <button type="button" className="icon-button svc-remove" aria-label={`Remove ${svc.name || 'this service'}`} title="Remove" onClick={() => removeService(i)}>
                  <Trash2 size={14} />
                </button>
              </li>
            );
          })}
        </ol>
        {removed && (
          <div className="svc-undo" role="status">
            <span>Removed {removed.svc.name.trim() ? `“${removed.svc.name.trim()}”` : 'a service'}.</span>
            <button type="button" className="btn-tiny" onClick={undoRemove}>
              <Undo2 size={13} /> Undo
            </button>
          </div>
        )}
        <button type="button" className="btn btn-ghost svc-add" onClick={addService}>
          <Plus size={14} /> Add a service
        </button>
      </section>

      <section className="hours-card hours-rules" aria-labelledby="hours-rules-title">
        <h3 className="section-title" id="hours-rules-title">
          Booking rules
        </h3>
        <ul className="rules">
          <li>
            <label htmlFor="rule-step">Offer times every</label>
            <select id="rule-step" value={s.slotMinutes} aria-invalid={!!problems.slotMinutes} onChange={(e) => setS({ ...s, slotMinutes: Number(e.target.value) })}>
              {[...new Set([...STEPS, s.slotMinutes])]
                .filter((m) => Number.isFinite(m) && m > 0)
                .sort((a, b) => a - b)
                .map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
            </select>
            <span>minutes.</span>
          </li>
          <li>
            <label htmlFor="rule-notice">Need at least</label>
            <input
              id="rule-notice"
              type="number"
              min={0}
              step={15}
              value={Number.isFinite(s.noticeMinutes) ? s.noticeMinutes : ''}
              aria-invalid={!!problems.noticeMinutes}
              onChange={(e) => setS({ ...s, noticeMinutes: e.target.value === '' ? Number.NaN : Number(e.target.value) })}
            />
            <span>minutes’ notice.</span>
            {problems.noticeMinutes && <small className="hours-error">{problems.noticeMinutes}</small>}
          </li>
          <li>
            <label htmlFor="rule-ahead">Take bookings up to</label>
            <input
              id="rule-ahead"
              type="number"
              min={1}
              max={366}
              value={Number.isFinite(s.horizonDays) ? s.horizonDays : ''}
              aria-invalid={!!problems.horizonDays}
              onChange={(e) => setS({ ...s, horizonDays: e.target.value === '' ? Number.NaN : Number(e.target.value) })}
            />
            <span>days ahead.</span>
            {problems.horizonDays && <small className="hours-error">{problems.horizonDays}</small>}
          </li>
          <li>
            <label className="rules-switch">
              <input type="checkbox" role="switch" className="switch" checked={s.textConfirmations} onChange={(e) => setS({ ...s, textConfirmations: e.target.checked })} />
              <span>Text people when their booking is confirmed.</span>
            </label>
          </li>
        </ul>
        {onOpenCalendar && !embedded && (
          <button type="button" className="btn-link" onClick={onOpenCalendar}>
            <CalendarDays size={13} /> See the calendar
          </button>
        )}
      </section>

      {after}

      {(dirty || embedded || justSaved) && (
        <div className={`hours-savebar${dirty ? ' is-dirty' : ''}${attempted && invalid ? ' is-invalid' : ''}`} role="region" aria-label="Save">
          <div className="hours-savebar-text" aria-live="polite">
            {dirty ? (
              <>
                <strong>
                  <i aria-hidden /> Unsaved changes
                </strong>
                <small>
                  {invalid
                    ? `${problems.list.length === 1 ? 'One thing' : `${problems.list.length} things`} to fix first: ${problems.list.slice(0, 2).join('; ')}${problems.list.length > 2 ? '; …' : ''}`
                    : 'The receptionist keeps the saved ones until you save.'}
                </small>
              </>
            ) : justSaved ? (
              <strong className="setup-ok">Saved</strong>
            ) : (
              <small>These are saved. Change anything above, then save it here.</small>
            )}
          </div>
          {dirty && (
            <button type="button" className="btn btn-ghost" disabled={saving} onClick={discard}>
              Discard
            </button>
          )}
          {(dirty || embedded) && (
            <button type="button" className="btn btn-primary" disabled={saving} onClick={() => void save()}>
              {saving ? 'Saving…' : dirty ? 'Save changes' : 'Save'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ---- how the phone sounds -------------------------------------------------------------------

/**
 * The voice the phone speaks in: a clip of someone speaking (MP3, WAV...),
 * cloned by OAIY's own speech engine on this machine's GPU. Each can be heard
 * first; a new one needs only its clip (what it says is heard from it).
 * Changes here apply at once (not with the form's Save).
 */
export function CallVoice({ business }: { business: string }) {
  const toast = useToast();
  const [list, setList] = useState<VoiceClip[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [words, setWords] = useState('');
  const [adding, setAdding] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const player = useRef<HTMLAudioElement | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const fail = (title: string, e: unknown) => toast.push({ kind: 'error', title, body: errText(e) });

  const load = useCallback(async () => {
    try {
      const r = await voices.list();
      setList(r.voices);
      setChosen(r.chosen);
    } catch (e) {
      fail('Voices not read', e);
    } finally {
      setLoaded(true);
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
      if (fileInput.current) fileInput.current.value = '';
      await load();
    } catch (e) {
      fail('Voice not added', e);
    } finally {
      setAdding(false);
    }
  };

  return (
    <section className="hours-card hours-voice" aria-labelledby="hours-voice-title">
      <div className="hours-card-head">
        <h3 className="section-title" id="hours-voice-title">
          How the phone sounds
        </h3>
        <span className="hours-now">Changes here apply at once</span>
      </div>
      <p className="form-hint">
        The voice callers hear. It is copied, on this computer, from a short clip: 5 to 30 seconds of one person speaking clearly, as an MP3 or WAV.
      </p>
      <ul className="voice-list" role="radiogroup" aria-label="The voice on calls">
        {!loaded ? (
          <li className="form-hint">Reading the voices…</li>
        ) : list.length === 0 ? (
          <li className="form-hint">No voices yet: add a clip below.</li>
        ) : (
          list.map((v) => (
            <li key={v.name} className={v.name === chosen ? 'is-chosen' : undefined}>
              <label className="voice-pick">
                <input type="radio" name="call-voice" checked={v.name === chosen} onChange={() => void choose(v.name)} />
                <span>
                  <strong>{v.name}</strong>
                  <small>{v.name === chosen ? 'Answers calls' : `${Math.max(1, Math.round(v.bytes / 1024))} KB clip`}</small>
                </span>
              </label>
              <button type="button" className="btn btn-ghost" disabled={speaking !== null} onClick={() => void hear(v.name)}>
                <Volume2 size={14} /> {speaking === v.name ? 'Speaking…' : 'Hear it'}
              </button>
              <button
                type="button"
                className="icon-button"
                disabled={list.length < 2}
                title={list.length < 2 ? 'The phone needs one voice' : 'Remove this voice'}
                aria-label={`Remove ${v.name}`}
                onClick={() => void remove(v.name)}
              >
                <Trash2 size={14} />
              </button>
            </li>
          ))
        )}
      </ul>
      <div className="voice-add">
        <label className="form-row">
          <span>A clip</span>
          <input ref={fileInput} type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.opus" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <label className="form-row">
          <span>Its name</span>
          <input value={name} placeholder={file ? file.name.replace(/\.[^.]+$/, '') : 'Front desk'} onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="form-row voice-words">
          <span>What it says (optional)</span>
          <input value={words} placeholder="Heard from the clip when left empty" onChange={(e) => setWords(e.target.value)} />
        </label>
        <button type="button" className="btn btn-primary" disabled={!file || adding} onClick={() => void add()}>
          <Plus size={14} /> {adding ? 'Adding…' : 'Add and use it'}
        </button>
      </div>
    </section>
  );
}
