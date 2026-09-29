import type { Appointment, AppointmentStatus, CalendarService, CalendarSettings, CalendarSpan } from './api';

/**
 * The calendar's pure parts, shared by the Calendar page and Hours & Services:
 * dates and times as the receptionist says them, where appointments sit in a
 * day, the free-time picker, the opening hours' quick actions, and the checks
 * the settings pass before they are sent (the same the desktop makes, so Save
 * never meets a refusal the page did not already show).
 */

export const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const;
export const DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as const;

// ---- dates -------------------------------------------------------------------------

export const pad = (n: number) => String(n).padStart(2, '0');
export const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
export function parseYmd(s: string): Date {
  const [y, m, d] = s.split('-').map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
}
export const addDays = (d: Date, n: number) => new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
/** Monday is 0. */
export const weekdayOf = (d: Date) => (d.getDay() + 6) % 7;
export const mondayOf = (d: Date) => addDays(d, -weekdayOf(d));

// ---- times -------------------------------------------------------------------------

/** "HH:MM" as minutes after midnight (NaN when it is not a time). */
export function minutesOf(hhmm: string): number {
  const m = /^(\d{1,2}):(\d{2})$/.exec(hhmm.trim());
  if (!m) return Number.NaN;
  const h = Number(m[1]);
  const mm = Number(m[2]);
  return h < 24 && mm < 60 ? h * 60 + mm : Number.NaN;
}

/** Minutes after midnight as "HH:MM" (within the day). */
export const hhmm = (minutes: number) => {
  const m = Math.max(0, Math.min(24 * 60 - 1, Math.round(minutes)));
  return `${pad(Math.floor(m / 60))}:${pad(m % 60)}`;
};

/**
 * A time as people write it ("09:00", "9:00", "10am", "3:30 pm") as "HH:MM",
 * or null. The desktop reads the same forms (the Agent may have written one).
 */
export function toHHMM(s: string): string | null {
  const t = s.trim().toLowerCase();
  const plain = minutesOf(t);
  if (!Number.isNaN(plain)) return hhmm(plain);
  const m = /^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$/.exec(t);
  if (!m) return null;
  const h = Number(m[1]);
  const mm = Number(m[2] ?? 0);
  if (h < 1 || h > 12 || mm > 59) return null;
  return hhmm(((h % 12) + (m[3] === 'pm' ? 12 : 0)) * 60 + mm);
}

/** "9:30 am", "2 pm". */
export function sayTime(t: string): string {
  const m = minutesOf(t);
  if (Number.isNaN(m)) return t;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h % 12 || 12}${mm ? `:${pad(mm)}` : ''} ${h < 12 ? 'am' : 'pm'}`;
}

/** "9 – 10:30 am", "11 am – 12 pm": the am/pm once when both ends share it. */
export function sayRange(from: string, to: string): string {
  const a = sayTime(from);
  const b = sayTime(to);
  const [aClock, aHalf] = a.split(' ');
  const [, bHalf] = b.split(' ');
  return aHalf === bHalf ? `${aClock} – ${b}` : `${a} – ${b}`;
}

export const dateOf = (start: string) => start.split('T')[0] ?? '';
export const timeOf = (start: string) => (start.split('T')[1] ?? '00:00').slice(0, 5);
/** The time an appointment ends, "HH:MM". */
export const endOf = (a: Pick<Appointment, 'start' | 'minutes'>) => hhmm(minutesOf(timeOf(a.start)) + a.minutes);

export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const;

/** "Tue 29 Sep". */
export function sayDay(date: string): string {
  const d = parseYmd(date);
  return `${DAYS[weekdayOf(d)]} ${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

/** "28 Sep – 4 Oct 2026", "5 – 11 Oct 2026". */
export function sayWeek(monday: Date): string {
  const sunday = addDays(monday, 6);
  const a = monday.getMonth() === sunday.getMonth() ? `${monday.getDate()}` : `${monday.getDate()} ${MONTHS[monday.getMonth()]}`;
  const yearA = monday.getFullYear() !== sunday.getFullYear() ? ` ${monday.getFullYear()}` : '';
  return `${a}${yearA} – ${sunday.getDate()} ${MONTHS[sunday.getMonth()]} ${sunday.getFullYear()}`;
}

/** "Tue 29 Sep, 10 am". */
export const sayWhen = (start: string) => `${sayDay(dateOf(start))}, ${sayTime(timeOf(start))}`;

/** "30 min", "1 hr", "1 hr 30 min". */
export function sayMinutes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '';
  const h = Math.floor(n / 60);
  const m = n % 60;
  if (!h) return `${m} min`;
  return m ? `${h} hr ${m} min` : `${h} hr`;
}

// ---- status ------------------------------------------------------------------------

export const STATUS_TEXT: Record<AppointmentStatus, string> = {
  requested: 'Request',
  confirmed: 'Confirmed',
  declined: 'Declined',
  cancelled: 'Cancelled',
  done: 'Done',
};

/** Whether it keeps its time from anyone else (the desktop's own rule). */
export const holdsTime = (s: AppointmentStatus) => s === 'requested' || s === 'confirmed';

export const SOURCE_TEXT: Record<string, string> = {
  call: 'on a call',
  text: 'by text',
  agent: 'by the agent',
  manual: 'here',
  formlogic: 'in FormLogic',
};

// ---- the week grid -----------------------------------------------------------------

/** The hours the grid spans: the opening hours and the appointments, with an hour either side (8 to 6 with neither). */
export function gridHours(settings: CalendarSettings | null, appointments: Pick<Appointment, 'start' | 'minutes'>[]): [number, number] {
  const spans = (settings?.hours ?? []).flat();
  const starts: number[] = [];
  const ends: number[] = [];
  for (const s of spans) {
    const o = minutesOf(toHHMM(s.open) ?? '');
    const c = minutesOf(toHHMM(s.close) ?? '');
    if (!Number.isNaN(o) && !Number.isNaN(c)) {
      starts.push(o);
      ends.push(c);
    }
  }
  for (const a of appointments) {
    const t = minutesOf(timeOf(a.start));
    if (Number.isNaN(t)) continue;
    starts.push(t);
    ends.push(t + a.minutes);
  }
  if (!starts.length) return [8, 18];
  const first = Math.max(0, Math.floor(Math.min(...starts) / 60) - 1);
  const last = Math.min(24, Math.ceil(Math.max(...ends) / 60) + 1);
  return [first, Math.min(24, Math.max(last, first + 4))];
}

/** Overlapping appointments that start at least this far apart are stacked, each indented over the one before. */
export const CASCADE_MINUTES = 20;

export interface DaySlot {
  lane: number;
  lanes: number;
  /**
   * Stacked: the overlapping ones start apart, so each is drawn over the one
   * before, indented, and the start of every one shows in full. Otherwise
   * (they start together) they sit side by side, `lane` of `lanes`.
   */
  cascade: boolean;
}

/** Where each appointment of a day sits: full width alone; stacked or side by side where they overlap. */
export function layoutDay<T extends Pick<Appointment, 'id' | 'start' | 'minutes'>>(items: T[]): Map<string, DaySlot> {
  const out = new Map<string, DaySlot>();
  const sorted = [...items]
    .map((a) => ({ a, from: minutesOf(timeOf(a.start)), to: minutesOf(timeOf(a.start)) + Math.max(1, a.minutes) }))
    .filter((x) => !Number.isNaN(x.from))
    .sort((x, y) => x.from - y.from || y.to - x.to);
  type Item = (typeof sorted)[number];
  let cluster: { x: Item; lane: number }[] = [];
  let laneEnds: number[] = [];
  let clusterEnd = -1;
  const close = () => {
    const apart = cluster.every((p, i) =>
      cluster.every((q, j) => i === j || q.x.from >= p.x.to || p.x.from >= q.x.to || Math.abs(p.x.from - q.x.from) >= CASCADE_MINUTES),
    );
    for (const c of cluster) out.set(c.x.a.id, { lane: c.lane, lanes: laneEnds.length, cascade: laneEnds.length > 1 && apart });
    cluster = [];
    laneEnds = [];
  };
  for (const x of sorted) {
    if (x.from >= clusterEnd) close();
    let lane = laneEnds.findIndex((end) => end <= x.from);
    if (lane < 0) {
      lane = laneEnds.length;
      laneEnds.push(x.to);
    } else laneEnds[lane] = x.to;
    cluster.push({ x, lane });
    clusterEnd = Math.max(clusterEnd, x.to);
  }
  close();
  return out;
}

/** A point in the grid, in minutes after midnight, down to the step before it. */
export const snapDown = (minutes: number, step: number) => Math.floor(minutes / Math.max(5, step)) * Math.max(5, step);

// ---- the free-time picker -----------------------------------------------------------

export type PartOfDay = 'morning' | 'afternoon' | 'evening';
export const PART_TEXT: Record<PartOfDay, string> = { morning: 'Morning', afternoon: 'Afternoon', evening: 'Evening' };

export function partOfDay(t: string): PartOfDay {
  const m = minutesOf(t);
  return m < 12 * 60 ? 'morning' : m < 17 * 60 ? 'afternoon' : 'evening';
}

/** Free times ("HH:MM", in order) grouped as morning, afternoon and evening; empty groups left out. */
export function groupTimes(times: string[]): { part: PartOfDay; label: string; times: string[] }[] {
  const parts: PartOfDay[] = ['morning', 'afternoon', 'evening'];
  return parts
    .map((part) => ({ part, label: PART_TEXT[part], times: times.filter((t) => partOfDay(t) === part) }))
    .filter((g) => g.times.length > 0);
}

/**
 * The free time to start on: `wanted` when it is free; otherwise the nearest
 * free one (the later on a tie); with nothing wanted, the first. Null with
 * nothing free.
 */
export function pickTime(times: string[], wanted?: string | null): string | null {
  if (!times.length) return null;
  if (!wanted) return times[0];
  if (times.includes(wanted)) return wanted;
  const w = minutesOf(wanted);
  if (Number.isNaN(w)) return times[0];
  let best = times[0];
  let bestGap = Infinity;
  for (const t of times) {
    const gap = Math.abs(minutesOf(t) - w);
    if (gap < bestGap || (gap === bestGap && minutesOf(t) > minutesOf(best))) {
      best = t;
      bestGap = gap;
    }
  }
  return best;
}

// ---- opening hours ------------------------------------------------------------------

type Hours = CalendarSpan[][];

const copySpans = (spans: CalendarSpan[]) => spans.map((s) => ({ ...s }));
const DEFAULT_SPAN: CalendarSpan = { open: '09:00', close: '17:00' };

/** Seven days, Monday first, each span as "HH:MM" (a time that cannot be read is kept as written). */
export function normalHours(hours: Hours | undefined): Hours {
  return Array.from({ length: 7 }, (_, i) =>
    (hours?.[i] ?? []).map((s) => ({ open: toHHMM(s.open) ?? s.open, close: toHHMM(s.close) ?? s.close })),
  );
}

/**
 * "Same hours Mon–Fri": Monday to Friday all open, with the hours of the
 * first weekday that is open (`from` when given), or 9 to 5. The weekend is
 * left as it is.
 */
export function sameHoursWeekdays(hours: Hours, from?: number): Hours {
  const source = from !== undefined && hours[from]?.length ? hours[from] : hours.slice(0, 5).find((d) => d.length) ?? [DEFAULT_SPAN];
  return hours.map((d, i) => (i < 5 ? copySpans(source) : copySpans(d)));
}

/** "Copy to all open days": every day that is open takes `day`'s hours; closed days stay closed. */
export function copyToOpenDays(hours: Hours, day: number): Hours {
  const source = hours[day] ?? [];
  if (!source.length) return hours.map(copySpans);
  return hours.map((d) => (d.length ? copySpans(source) : []));
}

/** A day opened: the hours of the nearest open day before it (or after), or 9 to 5. */
export function openDay(hours: Hours, day: number): Hours {
  const before = [...hours.slice(0, day)].reverse().find((d) => d.length);
  const after = hours.slice(day + 1).find((d) => d.length);
  return hours.map((d, i) => (i === day ? copySpans(before ?? after ?? [DEFAULT_SPAN]) : copySpans(d)));
}

/** A break in a day's last span: 12 to 1 when it spans midday, else an hour in its middle. Null when it is too short. */
export function withBreak(spans: CalendarSpan[]): CalendarSpan[] | null {
  const last = spans[spans.length - 1];
  if (!last) return null;
  const o = minutesOf(last.open);
  const c = minutesOf(last.close);
  if (Number.isNaN(o) || Number.isNaN(c) || c - o < 120) return null;
  let from = 12 * 60;
  if (!(o < from && from + 60 < c)) from = o + Math.floor((c - o - 60) / 2 / 30) * 30;
  return [...spans.slice(0, -1).map((s) => ({ ...s })), { open: last.open, close: hhmm(from) }, { open: hhmm(from + 60), close: last.close }];
}

const sameSpans = (a: CalendarSpan[], b: CalendarSpan[]) => a.length === b.length && a.every((s, i) => s.open === b[i].open && s.close === b[i].close);

/** The week as callers are told it: "Mon–Fri 8 am – 5 pm · Sat 9 am – 12 pm · Sun closed". */
export function sayHours(hours: Hours): string {
  const days = normalHours(hours);
  const parts: string[] = [];
  let i = 0;
  while (i < 7) {
    let j = i;
    while (j + 1 < 7 && sameSpans(days[j + 1], days[i])) j++;
    const name = i === j ? DAYS[i] : j === i + 1 ? `${DAYS[i]}, ${DAYS[j]}` : `${DAYS[i]}–${DAYS[j]}`;
    const when = days[i].length ? days[i].map((s) => sayRange(s.open, s.close)).join(', ') : 'closed';
    parts.push(`${name} ${when}`);
    i = j + 1;
  }
  return parts.join(' · ');
}

// ---- checks ------------------------------------------------------------------------

export interface ServiceProblems {
  name?: string;
  minutes?: string;
}

/** What is wrong with a service, if anything (the desktop needs a name and 1 to 1440 minutes). */
export function checkService(s: CalendarService): ServiceProblems {
  const out: ServiceProblems = {};
  if (!s.name.trim()) out.name = 'Give it a name';
  if (!Number.isFinite(s.minutes) || s.minutes <= 0) out.minutes = 'Say how long it takes';
  else if (!Number.isInteger(s.minutes) || s.minutes > 24 * 60) out.minutes = 'Whole minutes, up to a day (1440)';
  return out;
}

/** What is wrong with a day's spans, one message a span (null when it is fine). */
export function checkSpan(s: CalendarSpan): string | null {
  const o = minutesOf(s.open);
  const c = minutesOf(s.close);
  if (Number.isNaN(o) || Number.isNaN(c)) return 'Needs a time to open and to close';
  if (c <= o) return `Closes at ${sayTime(s.close)}, before it opens at ${sayTime(s.open)}`;
  return null;
}

export interface SettingsProblems {
  /** Day index → span index → message. */
  hours: Record<number, Record<number, string>>;
  services: Record<number, ServiceProblems>;
  slotMinutes?: string;
  noticeMinutes?: string;
  horizonDays?: string;
  /** Each problem in a sentence, in page order. */
  list: string[];
}

export function checkSettings(s: CalendarSettings): SettingsProblems {
  const out: SettingsProblems = { hours: {}, services: {}, list: [] };
  s.hours.forEach((day, d) =>
    day.forEach((span, k) => {
      const why = checkSpan(span);
      if (why) {
        (out.hours[d] ??= {})[k] = why;
        out.list.push(`${DAY_NAMES[d]}: ${why.charAt(0).toLowerCase()}${why.slice(1)}`);
      }
    }),
  );
  s.services.forEach((svc, i) => {
    const p = checkService(svc);
    if (p.name || p.minutes) {
      out.services[i] = p;
      const who = svc.name.trim() ? `“${svc.name.trim()}”` : `Service ${i + 1}`;
      out.list.push(`${who}: ${[p.name, p.minutes].filter(Boolean).join('; ').toLowerCase()}`);
    }
  });
  if (!Number.isInteger(s.slotMinutes) || s.slotMinutes < 5 || s.slotMinutes > 240) {
    out.slotMinutes = 'Every 5 to 240 minutes';
    out.list.push('Times offered: every 5 to 240 minutes');
  }
  if (!Number.isInteger(s.noticeMinutes) || s.noticeMinutes < 0) {
    out.noticeMinutes = 'Whole minutes, 0 or more';
    out.list.push('Notice: whole minutes, 0 or more');
  }
  if (!Number.isInteger(s.horizonDays) || s.horizonDays < 1 || s.horizonDays > 366) {
    out.horizonDays = '1 to 366 days';
    out.list.push('Bookings ahead: 1 to 366 days');
  }
  return out;
}

/** The settings as the form edits them: hours as "HH:MM", optional service fields as strings. */
export function normalSettings(s: CalendarSettings): CalendarSettings {
  return {
    ...s,
    business: s.business ?? '',
    hours: normalHours(s.hours),
    services: (s.services ?? []).map((x) => ({ ...x, description: x.description ?? '', price: x.price ?? '' })),
  };
}

/** Whether the form has changed anything. */
export const sameSettings = (a: CalendarSettings, b: CalendarSettings) => JSON.stringify(normalSettings(a)) === JSON.stringify(normalSettings(b));

/** Durations offered for a service (anything else is "Custom"). */
export const DURATIONS = [15, 30, 45, 60, 90, 120] as const;
/** Steps offered between times. */
export const STEPS = [5, 10, 15, 20, 30, 45, 60, 90, 120] as const;
