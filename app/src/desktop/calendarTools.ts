/**
 * The desktop's calendar as the agent's tools. A text thread's agent may see
 * the free times and ask for an appointment for the person texting (a request,
 * for staff to confirm, as a call's is); the main agent also lists, books and
 * changes appointments for the person it works for.
 */
import type { SessionTool } from '../agent/agent';
import type { Desktop } from './bridge';

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** "Tue 29 Sep". */
function sayDate(ymd: string): string {
  const [y, m, d] = ymd.split('-').map(Number);
  const date = new Date(y, (m ?? 1) - 1, d ?? 1);
  return `${DAYS[date.getDay()]} ${date.getDate()} ${date.toLocaleString('en', { month: 'short' })}`;
}

/** "9:30 am". */
export function sayTime(hhmm: string): string {
  const [h = 0, m = 0] = hhmm.split(':').map(Number);
  return `${h % 12 || 12}${m ? `:${String(m).padStart(2, '0')}` : ''} ${h < 12 ? 'am' : 'pm'}`;
}

const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

function connected(desktop: () => Desktop | null): Desktop {
  const d = desktop();
  if (!d) throw new Error('OAIY Desktop is not connected, so the calendar cannot be reached');
  return d;
}

/** The free times as the model reads them: a line per day. */
export async function freeTimes(d: Desktop, input: Record<string, unknown>, signal?: AbortSignal): Promise<string> {
  const from = typeof input.from === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(input.from) ? input.from : today();
  const days = Math.min(31, Math.max(1, Number(input.days) || 7));
  const service = typeof input.service === 'string' ? input.service : undefined;
  const free = await d.calendarFree(from, days, service, signal);
  const lines = free.days.filter((day) => day.times.length).map((day) => `${sayDate(day.date)} (${day.date}): ${day.times.map(sayTime).join(', ')}`);
  const what = `${free.service ?? 'an appointment'}, ${free.minutes} min`;
  return lines.length ? `Free times for ${what}:\n${lines.join('\n')}` : `Nothing free for ${what} from ${sayDate(from)} for ${days} day(s).`;
}

const FREE_SPEC = {
  name: 'calendar_free_times',
  description: 'The free appointment times in the business calendar (on OAIY Desktop), within its opening hours, for a service (by name) or the default length.',
  parameters: {
    type: 'object',
    properties: {
      from: { type: 'string', description: 'The first day, YYYY-MM-DD (default today)' },
      days: { type: 'number', description: 'How many days to look at (default 7, at most 31)' },
      service: { type: 'string', description: 'The service, by name (sets the length)' },
    },
  },
};

/** A text thread's calendar tools: the free times, and a request for the person texting. */
export function textCalendarTools(desktop: () => Desktop | null, phone: string, name: () => string): SessionTool[] {
  return [
    { spec: FREE_SPEC, run: async (input, signal) => freeTimes(connected(desktop), input, signal) },
    {
      spec: {
        name: 'request_appointment',
        description:
          'Ask for an appointment for the person texting, once they have clearly agreed to a day and time from the free times. It is a REQUEST that staff confirm (they are texted when it is): never say it is booked or confirmed.',
        parameters: {
          type: 'object',
          required: ['service', 'date', 'time'],
          properties: {
            service: { type: 'string' },
            date: { type: 'string', description: 'YYYY-MM-DD' },
            time: { type: 'string', description: 'HH:MM, 24-hour' },
            name: { type: 'string', description: 'Their name, if they gave it' },
            notes: { type: 'string', description: 'Anything staff should know' },
          },
        },
      },
      run: async (input, signal) => {
        const a = await connected(desktop).calendarCreate(
          { service: input.service, date: input.date, time: input.time, name: String(input.name ?? '') || name(), phone, notes: String(input.notes ?? ''), status: 'requested', source: 'text' },
          signal,
        );
        return `Requested: ${a.service || 'an appointment'} on ${String(a.start ?? '').replace('T', ' at ')}. Staff will confirm it; tell them so (it is not confirmed yet).`;
      },
    },
  ];
}

/** The main agent's calendar tools. */
export function calendarTools(desktop: () => Desktop | null): SessionTool[] {
  return [
    { spec: FREE_SPEC, run: async (input, signal) => freeTimes(connected(desktop), input, signal) },
    {
      spec: {
        name: 'calendar_list',
        description: 'The appointments in the business calendar from a day (default today) for some days: time, service, who, status (requested ones wait for confirmation), and each one\'s id.',
        parameters: { type: 'object', properties: { from: { type: 'string', description: 'YYYY-MM-DD' }, days: { type: 'number', description: 'default 7' } } },
      },
      run: async (input, signal) => {
        const from = typeof input.from === 'string' && input.from ? input.from : today();
        const [y, m, d] = from.split('-').map(Number);
        const end = new Date(y, (m ?? 1) - 1, (d ?? 1) + Math.min(92, Math.max(1, Number(input.days) || 7)));
        const to = `${end.getFullYear()}-${String(end.getMonth() + 1).padStart(2, '0')}-${String(end.getDate()).padStart(2, '0')}`;
        const { appointments } = await connected(desktop).calendar(from, to, signal);
        if (!appointments.length) return `No appointments from ${from} to before ${to}.`;
        return appointments
          .map((a) => {
            const [date, time = ''] = String(a.start).split('T');
            return `${sayDate(date)} ${sayTime(time)} · ${a.service || 'appointment'} (${a.minutes} min) · ${a.name || 'no name'}${a.phone ? `, ${a.phone}` : ''} · ${a.status}${a.notes ? ` · ${a.notes}` : ''} · id ${a.id}`;
          })
          .join('\n');
      },
    },
    {
      spec: {
        name: 'calendar_book',
        description: 'Put an appointment in the business calendar (confirmed, unless status is "requested"). Check calendar_free_times first.',
        parameters: {
          type: 'object',
          required: ['date', 'time'],
          properties: {
            service: { type: 'string' },
            date: { type: 'string', description: 'YYYY-MM-DD' },
            time: { type: 'string', description: 'HH:MM, 24-hour' },
            name: { type: 'string' },
            phone: { type: 'string' },
            notes: { type: 'string' },
            status: { type: 'string', enum: ['confirmed', 'requested'] },
          },
        },
      },
      run: async (input, signal) => {
        const a = await connected(desktop).calendarCreate({ ...input, source: 'agent' }, signal);
        return `Booked: ${a.service || 'appointment'} on ${String(a.start).replace('T', ' at ')} (${a.status}), id ${a.id}.`;
      },
    },
    {
      spec: {
        name: 'calendar_change',
        description: 'Change an appointment by its id (from calendar_list): confirm, decline, cancel or mark it done, move it (start YYYY-MM-DDTHH:MM), or change its details.',
        parameters: {
          type: 'object',
          required: ['id'],
          properties: {
            id: { type: 'string' },
            status: { type: 'string', enum: ['requested', 'confirmed', 'declined', 'cancelled', 'done'] },
            start: { type: 'string', description: 'YYYY-MM-DDTHH:MM' },
            service: { type: 'string' },
            name: { type: 'string' },
            phone: { type: 'string' },
            notes: { type: 'string' },
          },
        },
      },
      run: async (input, signal) => {
        const { id, ...change } = input;
        const a = await connected(desktop).calendarUpdate(String(id), change, signal);
        return `Changed: ${a.service || 'appointment'} on ${String(a.start).replace('T', ' at ')}, ${a.status}.`;
      },
    },
  ];
}
