import { useCallback, useEffect, useState } from 'react';
import { CalendarClock, Cloud, CloudOff, Cpu, Inbox, Phone } from 'lucide-react';
import { calendar, engines, link, phone, type Appointment, type CalendarSync, type EnginesStatus, type LinkStatus } from './api';
import { describeSync } from './syncStatus';
import { moduleOn, useModules } from './useModules';

/**
 * Today, at the top of Overview: whether the phone is connected (and a call
 * live), whether the language model is loaded, the next appointments, the
 * requests waiting for someone to confirm them, and, when linked, how this
 * desktop stands with FormLogic (everything here works without it). Each tile
 * opens its page. The phone's tile and the calendar's are there (and asked
 * about) only while a plugin provides them.
 */

const POLL_MS = 10_000;

const pad = (n: number) => String(n).padStart(2, '0');
const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
/** When an appointment is, in two parts that each fit a tile's line: the day, and the time. */
function sayWhen(start: string): { day: string; clock: string } {
  const [date, time = '00:00'] = start.split('T');
  const [h, m] = time.split(':').map(Number);
  const clock = `${h % 12 || 12}${m ? `:${pad(m)}` : ''} ${h < 12 ? 'am' : 'pm'}`;
  const today = ymd(new Date());
  const tomorrow = ymd(new Date(Date.now() + 86_400_000));
  if (date === today) return { day: 'Today', clock };
  if (date === tomorrow) return { day: 'Tomorrow', clock };
  const d = new Date(`${date}T00:00`);
  return { day: d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' }).replace(',', ''), clock };
}

export default function TodayPanel({ onNavigate }: { onNavigate: (view: 'agent' | 'calendar' | 'engines' | 'connections') => void }) {
  const [phoneState, setPhoneState] = useState<{ connected: boolean; onCall: boolean } | null>(null);
  const [model, setModel] = useState<EnginesStatus | null>(null);
  const [upcoming, setUpcoming] = useState<Appointment[] | null>(null);
  const [sync, setSync] = useState<CalendarSync | null>(null);
  const [linked, setLinked] = useState<LinkStatus | null>(null);
  /** The phone and the calendar, while a plugin provides them. */
  const modules = useModules();
  const phoneOn = moduleOn(modules, 'phone') === true;
  const calendarOn = moduleOn(modules, 'calendar') === true;

  const refresh = useCallback(async () => {
    const off = Promise.reject(new Error('off'));
    off.catch(() => {});
    const [p, calls, e, cal, s, l] = await Promise.allSettled([
      phoneOn ? phone.status() : off,
      phoneOn ? phone.calls() : off,
      engines.status(),
      calendarOn ? calendar.get(ymd(new Date())) : off,
      calendarOn ? calendar.syncStatus() : off,
      link.status(),
    ]);
    setPhoneState(p.status === 'fulfilled' ? { connected: p.value.connected, onCall: calls.status === 'fulfilled' && calls.value.length > 0 } : null);
    setModel(e.status === 'fulfilled' ? e.value : null);
    if (cal.status === 'fulfilled') {
      const now = `${ymd(new Date())}T${pad(new Date().getHours())}:${pad(new Date().getMinutes())}`;
      setUpcoming(cal.value.appointments.filter((a) => (a.status === 'confirmed' || a.status === 'requested') && a.start >= now));
    }
    setSync(s.status === 'fulfilled' ? s.value : null);
    setLinked(l.status === 'fulfilled' ? l.value : null);
  }, [phoneOn, calendarOn]);
  const formlogic = describeSync(sync, linked);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const requests = (upcoming ?? []).filter((a) => a.status === 'requested');
  const next = (upcoming ?? []).find((a) => a.status === 'confirmed') ?? upcoming?.[0];
  const when = next ? sayWhen(next.start) : null;
  const llm = model?.llm;

  return (
    <section className="service-section">
      <div className="section-title-row">
        <h3 className="section-title">Today</h3>
      </div>
      <div className="overview-grid today-grid">
        {phoneOn && (
          <button className="overview-tile" onClick={() => onNavigate('agent')} title="Calls and texts are answered by the agent">
            <Phone size={16} aria-hidden />
            <strong className={phoneState?.connected ? 'ok' : 'warn'}>{phoneState === null ? '—' : phoneState.onCall ? 'On a call' : phoneState.connected ? 'Connected' : 'Not connected'}</strong>
            <small>The phone</small>
          </button>
        )}
        <button className="overview-tile" onClick={() => onNavigate('engines')} title={llm?.resident ? `${llm.resident}: ${llm.state}` : 'The engines'}>
          <Cpu size={16} aria-hidden />
          <strong className={llm?.state === 'ready' ? 'ok' : undefined}>{model === null ? '—' : !model.running ? 'Engines off' : llm?.state === 'ready' ? 'Model ready' : llm?.state === 'loading' ? 'Loading…' : 'Loads on use'}</strong>
          <small>{llm?.resident ?? 'Language model'}</small>
        </button>
        {calendarOn && (
          <>
            <button className="overview-tile" onClick={() => onNavigate('calendar')} title="The next appointment">
              <CalendarClock size={16} aria-hidden />
              {/* Today's is its time; a later one its day, with the time beside what it is. */}
              <strong>{upcoming === null ? '—' : when ? (when.day === 'Today' ? when.clock : when.day) : 'Nothing booked'}</strong>
              <small>
                {next && when
                  ? `${when.day === 'Today' ? 'Today' : when.clock} · ${next.service || 'Appointment'}${next.name ? `, ${next.name}` : ''}`
                  : 'Next appointment'}
              </small>
            </button>
            <button className="overview-tile" onClick={() => onNavigate('calendar')}>
              <Inbox size={16} aria-hidden />
              <strong className={requests.length ? 'warn' : undefined}>{upcoming === null ? '—' : requests.length}</strong>
              <small>{requests.length === 1 ? 'Request to confirm' : 'Requests to confirm'}</small>
            </button>
          </>
        )}
        {formlogic && (
          <button
            className="overview-tile"
            onClick={() => onNavigate('connections')}
            aria-label={`FormLogic: ${formlogic.headline}. ${formlogic.detail}`}
            title={[sync?.error, linked?.heartbeatError, linked?.outbox?.lastError, 'Calls, the calendar and the agent work without FormLogic; what changed here is sent when it can be reached.'].filter(Boolean).join('\n')}
          >
            {formlogic.headline === 'Offline' ? <CloudOff size={16} aria-hidden /> : <Cloud size={16} aria-hidden />}
            <strong className={formlogic.tone === 'ok' ? 'ok' : formlogic.tone === 'neutral' ? undefined : 'warn'}>{formlogic.headline}</strong>
            <small>FormLogic · {formlogic.detail}</small>
          </button>
        )}
      </div>
    </section>
  );
}
