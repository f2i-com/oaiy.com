/**
 * The desktop's calendar as the agent's tools. A text thread's agent may see
 * what is free and ask for an appointment for the person texting (a request,
 * for staff to confirm, as a call's is); the main agent also lists, books and
 * changes appointments for the person it works for.
 */
import type { SessionTool } from '../agent/agent';
import type { Desktop } from './bridge';
import { addDays, dayLabel, freeRule, parseClock, planDay, readRules, repeatNote, sayClock, sayDate, sayDay, sayDays, sayStarts, sayTime, stepOf, whyNot, type Audience } from './freeSummary';

export { sayTime } from './freeSummary';

/** YYYY-MM-DD. */
const ymd = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
const today = () => ymd(new Date());
/** This machine's time, `YYYY-MM-DDTHH:MM`, when the desktop does not say its own. */
const localNow = () => {
  const d = new Date();
  return `${ymd(d)}T${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

function connected(desktop: () => Desktop | null): Desktop {
  const d = desktop();
  if (!d) throw new Error('OAIY Desktop is not connected, so the calendar cannot be reached');
  return d;
}

/** A free-times question, as asked: the first day, how many days, the service, and one start time to check. */
function freeQuery(input: Record<string, unknown>): { from: string; days: number; service?: string; at: number | null } {
  const from = typeof input.from === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(input.from) ? input.from : today();
  const time = typeof input.time === 'string' ? input.time.trim() : '';
  const at = time ? parseClock(time) : null;
  if (time && at === null) throw new Error('time is HH:MM, 24-hour (e.g. 13:00)');
  // A time is checked on the day `from`.
  const days = at !== null ? 1 : Math.min(31, Math.max(1, Number(input.days) || 7));
  const service = typeof input.service === 'string' && input.service.trim() ? input.service.trim() : undefined;
  return { from, days, service, at };
}

/**
 * What is free, as the model reads it: a line per day, with the opening hours,
 * the booked times (times only), the free ranges and when the service can
 * start (see freeSummary), then how to use it, for `audience`.
 */
export async function freeTimes(d: Desktop, input: Record<string, unknown>, signal?: AbortSignal, audience: Audience = 'owner'): Promise<string> {
  const { from, days, service, at } = freeQuery(input);
  // The hours and bookings (from the day before, for one that runs past midnight), and the desktop's own free starts.
  const [cal, free] = await Promise.all([d.calendar(addDays(from, -1), addDays(from, days), signal), d.calendarFree(from, days, service, signal)]);
  const rules = readRules(cal.settings);
  const step = stepOf(rules, free.days);
  const now = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(cal.now ?? '') ? cal.now : localNow();
  const onDesk = now.slice(0, 10);
  const what = `a ${free.minutes}-min ${free.service ?? 'appointment'}`;
  const times = new Map(free.days.map((day) => [day.date, day.times]));
  const plans = Array.from({ length: days }, (_, i) => addDays(from, i)).map((date) => planDay(date, rules, cal.appointments ?? [], times.get(date), now, step));
  const roomy = plans.some((p) => p.starts.length);

  // The next day with room after a day: among the days asked about, else (looked up once) in the two weeks after them.
  let later: Array<{ date: string; times: string[] }> | null = null;
  const nextRoom = async (after: string): Promise<{ date: string; starts: number[] } | null> => {
    const near = plans.find((p) => p.date > after && p.starts.length);
    if (near) return { date: near.date, starts: near.starts };
    later ??= await d.calendarFree(addDays(from, days), 14, service, signal).then((f) => f.days, () => []);
    const found = later.find((day) => day.date > after && day.times.length);
    return found ? { date: found.date, starts: planDay(found.date, rules, [], found.times, now, step).starts } : null;
  };

  const lines: string[] = [];
  if (at !== null) {
    const p = plans[0];
    const asked = `${sayClock(at)} on ${dayLabel(p.date, onDesk)}`;
    if (p.starts.includes(at)) lines.push(`Yes: ${asked} is free for ${what}.`);
    else {
      const why = whyNot(p, at, free.minutes, step, audience);
      const before = [...p.starts].reverse().find((t) => t < at);
      const same = p.starts.find((t) => t > at);
      const next = same === undefined ? await nextRoom(p.date) : null;
      const after = same !== undefined ? sayClock(same) : next ? `${sayDate(next.date)} at ${sayClock(next.starts[0])}` : '';
      lines.push(
        `No: ${asked} isn't available for ${what}${why ? ` (${why})` : ''}. The nearest free starts: ${before !== undefined ? `${sayClock(before)} before it` : 'none earlier that day'}, and ${after ? `${after} after it` : 'none after it in the next two weeks'}.`,
      );
    }
  }
  if (service && !free.service) lines.push(`There is no service called "${service}", so these are for ${what}.`);
  lines.push(`What is free for ${what}:`);
  for (const p of plans) {
    if (p.ahead || p.closed) continue;
    // A day where nothing fits names the next day with room (all of them at the end, when none of the days asked has any).
    const next = !p.starts.length && roomy ? await nextRoom(p.date) : null;
    lines.push(sayDay(p, what, step, onDesk, next ? sayDate(next.date) : undefined));
  }
  const closed = plans.filter((p) => p.closed && !p.ahead).map((p) => p.date);
  const ahead = plans.filter((p) => p.ahead).map((p) => p.date);
  if (closed.length) lines.push(`Closed: ${sayDays(closed)}.`);
  if (ahead.length) lines.push(`Too far ahead to book yet: ${sayDays(ahead)}.`);
  if (!roomy) {
    const last = plans[plans.length - 1].date;
    const next = await nextRoom(last);
    if (next) lines.push(`Nothing fits ${what} on the ${days === 1 ? 'day' : 'days'} asked. The next day with room: ${dayLabel(next.date, onDesk)}, when ${what} can start ${sayStarts(next.starts, step)}.`);
    else if (!ahead.length) lines.push(`Nothing fits ${what} in the two weeks after ${sayDate(last)} either.`);
  }
  lines.push(freeRule(audience));
  return lines.join('\n');
}

/** How soon the same check again reads as a repeat. */
const REPEAT_MS = 2 * 60_000;

/** The free-times tool, for `audience`: a check that reads the same as one a moment ago says so. */
function freeTool(desktop: () => Desktop | null, audience: Audience): SessionTool {
  // This tool's own checks: one tool per conversation for a call or a text thread.
  const checks = new Map<string, { text: string; at: number }>();
  return {
    spec: {
      name: 'calendar_free_times',
      description: `What is free in the business calendar (on OAIY Desktop), a line per day: its opening hours, the times already booked (times only), the free ranges, and when the service can start. With time, whether that one start is free on the day from, and if not, the nearest free starts. ${freeRule(audience)}`,
      parameters: {
        type: 'object',
        properties: {
          from: { type: 'string', description: 'The first day, YYYY-MM-DD (default today)' },
          days: { type: 'number', description: 'How many days to look at (default 7, at most 31)' },
          service: { type: 'string', description: 'The service, by name (sets the length)' },
          time: { type: 'string', description: 'One start time to check on the day from, HH:MM 24-hour (e.g. 13:00)' },
        },
      },
    },
    run: async (input, signal) => {
      const text = await freeTimes(connected(desktop), input, signal, audience);
      // Always read afresh (a booking may have changed it), and say so only when it reads the same.
      const q = freeQuery(input);
      const key = JSON.stringify([q.from, q.days, q.service?.toLowerCase() ?? '', q.at]);
      const now = Date.now();
      for (const [k, c] of checks) if (now - c.at > REPEAT_MS) checks.delete(k);
      const last = checks.get(key);
      checks.set(key, { text, at: now });
      return last && last.text === text ? `${repeatNote(audience)}\n${text}` : text;
    },
  };
}

/** A call's calendar tool: what is free, worded for the caller (a call requests its appointment through the phone). */
export function callCalendarTools(desktop: () => Desktop | null): SessionTool[] {
  return [freeTool(desktop, 'caller')];
}

/** A text thread's calendar tools: what is free, and a request for the person texting. */
export function textCalendarTools(desktop: () => Desktop | null, phone: string, name: () => string): SessionTool[] {
  return [
    freeTool(desktop, 'texter'),
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
    freeTool(desktop, 'owner'),
    {
      spec: {
        name: 'calendar_list',
        description: 'The appointments in the business calendar from a day (default today) for some days: time, service, who, status (requested ones wait for confirmation), and each one\'s id.',
        parameters: { type: 'object', properties: { from: { type: 'string', description: 'YYYY-MM-DD' }, days: { type: 'number', description: 'default 7' } } },
      },
      run: async (input, signal) => {
        const from = typeof input.from === 'string' && input.from ? input.from : today();
        const to = addDays(from, Math.min(92, Math.max(1, Number(input.days) || 7)));
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
