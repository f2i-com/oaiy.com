import { useCallback, useEffect, useState } from 'react';
import { CalendarClock, Cpu, Inbox, Phone } from 'lucide-react';
import { calendar, engines, phone, type Appointment, type EnginesStatus } from './api';

/**
 * Today, at the top of Overview: whether the phone is connected (and a call
 * live), whether the language model is loaded, the next appointments, and the
 * requests waiting for someone to confirm them. Each tile opens its page.
 */

const POLL_MS = 10_000;

const pad = (n: number) => String(n).padStart(2, '0');
const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
function sayWhen(start: string): string {
  const [date, time = '00:00'] = start.split('T');
  const [h, m] = time.split(':').map(Number);
  const clock = `${h % 12 || 12}${m ? `:${pad(m)}` : ''} ${h < 12 ? 'am' : 'pm'}`;
  const today = ymd(new Date());
  const tomorrow = ymd(new Date(Date.now() + 86_400_000));
  if (date === today) return clock;
  if (date === tomorrow) return `tomorrow ${clock}`;
  const d = new Date(`${date}T00:00`);
  return `${d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}, ${clock}`;
}

export default function TodayPanel({ onNavigate }: { onNavigate: (view: 'agent' | 'calendar' | 'engines') => void }) {
  const [phoneState, setPhoneState] = useState<{ connected: boolean; onCall: boolean } | null>(null);
  const [model, setModel] = useState<EnginesStatus | null>(null);
  const [upcoming, setUpcoming] = useState<Appointment[] | null>(null);

  const refresh = useCallback(async () => {
    const [p, calls, e, cal] = await Promise.allSettled([phone.status(), phone.calls(), engines.status(), calendar.get(ymd(new Date()))]);
    setPhoneState(p.status === 'fulfilled' ? { connected: p.value.connected, onCall: calls.status === 'fulfilled' && calls.value.length > 0 } : null);
    setModel(e.status === 'fulfilled' ? e.value : null);
    if (cal.status === 'fulfilled') {
      const now = `${ymd(new Date())}T${pad(new Date().getHours())}:${pad(new Date().getMinutes())}`;
      setUpcoming(cal.value.appointments.filter((a) => (a.status === 'confirmed' || a.status === 'requested') && a.start >= now));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const requests = (upcoming ?? []).filter((a) => a.status === 'requested');
  const next = (upcoming ?? []).find((a) => a.status === 'confirmed') ?? upcoming?.[0];
  const llm = model?.llm;

  return (
    <section className="service-section">
      <div className="section-title-row">
        <h3 className="section-title">Today</h3>
      </div>
      <div className="overview-grid today-grid">
        <button className="overview-tile" onClick={() => onNavigate('agent')} title="Calls and texts are answered by the agent">
          <Phone size={16} aria-hidden />
          <strong className={phoneState?.connected ? 'ok' : 'warn'}>{phoneState === null ? '—' : phoneState.onCall ? 'On a call' : phoneState.connected ? 'Connected' : 'Not connected'}</strong>
          <small>The phone</small>
        </button>
        <button className="overview-tile" onClick={() => onNavigate('engines')} title={llm?.resident ? `${llm.resident}: ${llm.state}` : 'The engines'}>
          <Cpu size={16} aria-hidden />
          <strong className={llm?.state === 'ready' ? 'ok' : undefined}>{model === null ? '—' : !model.running ? 'Engines off' : llm?.state === 'ready' ? 'Model ready' : llm?.state === 'loading' ? 'Loading…' : 'Loads on use'}</strong>
          <small>{llm?.resident ?? 'Language model'}</small>
        </button>
        <button className="overview-tile" onClick={() => onNavigate('calendar')}>
          <CalendarClock size={16} aria-hidden />
          <strong>{upcoming === null ? '—' : next ? sayWhen(next.start) : 'Nothing booked'}</strong>
          <small>{next ? `${next.service || 'Appointment'}${next.name ? `, ${next.name}` : ''}` : 'Next appointment'}</small>
        </button>
        <button className="overview-tile" onClick={() => onNavigate('calendar')}>
          <Inbox size={16} aria-hidden />
          <strong className={requests.length ? 'warn' : undefined}>{upcoming === null ? '—' : requests.length}</strong>
          <small>{requests.length === 1 ? 'Request to confirm' : 'Requests to confirm'}</small>
        </button>
      </div>
    </section>
  );
}
