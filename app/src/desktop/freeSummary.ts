/**
 * What is free in the calendar, as a line per day that an agent can say in a
 * sentence: the opening hours, the times already booked (times only), the free
 * ranges, and when the service can start.
 *
 * A list of every free half hour read badly on a call: the agent offered the
 * first four and never reached the afternoon. Ranges ("any time from 10:30 am
 * to 4 pm") say the whole day at once. The start windows are grouped from the
 * desktop's own free start times, so notice, how far ahead and the booking
 * rules still come from the desktop, which stays the authority on what can be
 * booked; the hours and bookings only explain them.
 */

/** Who the summary is for: the business owner's own agent, or an agent talking with a customer. */
export type Audience = 'owner' | 'caller' | 'texter';

/** From and to, in minutes from the day's midnight. */
export type Range = [number, number];

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** "Tue 29 Sep". */
export function sayDate(ymd: string): string {
  const [y, m, d] = ymd.split('-').map(Number);
  const date = new Date(y, (m ?? 1) - 1, d ?? 1);
  return `${DAYS[date.getDay()]} ${date.getDate()} ${date.toLocaleString('en', { month: 'short' })}`;
}

/** "9:30 am". */
export function sayTime(hhmm: string): string {
  const [h = 0, m = 0] = hhmm.split(':').map(Number);
  return `${h % 12 || 12}${m ? `:${String(m).padStart(2, '0')}` : ''} ${h < 12 ? 'am' : 'pm'}`;
}

/** Minutes from midnight as said: 570 → "9:30 am". */
function clock(minutes: number): string {
  const m = ((minutes % 1440) + 1440) % 1440;
  return sayTime(`${Math.floor(m / 60)}:${m % 60}`);
}

/** "1–2 pm", "10:30–11 am", "8 am–1 pm": the first half's am or pm left out when both halves share it. */
function sayRange([from, to]: Range): string {
  const a = clock(from);
  const b = clock(to);
  return a.slice(-2) === b.slice(-2) ? `${a.slice(0, -3)}–${b}` : `${a}–${b}`;
}

/** "a, b and c". */
function sayList(items: string[], last = 'and'): string {
  return items.length < 2 ? (items[0] ?? '') : `${items.slice(0, -1).join(', ')} ${last} ${items[items.length - 1]}`;
}

/** "HH:MM" → minutes from midnight. */
function minutesOf(hhmm: unknown): number | null {
  const m = /^(\d{1,2}):(\d{2})/.exec(String(hhmm ?? '').trim());
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
}

/** A time as people write it: "13:00", "1pm", "1:30 pm", "13". Null when it is not one. */
export function parseClock(text: string): number | null {
  const s = text.trim().toLowerCase().replace(/\./g, '');
  const m = /^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/.exec(s);
  if (!m) return null;
  let h = Number(m[1]);
  const min = Number(m[2] ?? 0);
  if (min > 59) return null;
  if (m[3]) {
    if (h < 1 || h > 12) return null;
    h = (h % 12) + (m[3] === 'pm' ? 12 : 0);
  } else if (h > 23) return null;
  return h * 60 + min;
}

/** The day `ymd` falls on, as a count of days (for sums across months). */
function dayNumber(ymd: string): number {
  const [y, m, d] = ymd.split('-').map(Number);
  return Math.round(Date.UTC(y, (m ?? 1) - 1, d ?? 1) / 86_400_000);
}

/** `ymd` moved by `n` days. */
export function addDays(ymd: string, n: number): string {
  const [y, m, d] = ymd.split('-').map(Number);
  const t = new Date(y, (m ?? 1) - 1, (d ?? 1) + n);
  return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`;
}

/** Ranges put in order, overlapping and touching ones joined. */
function merge(ranges: Range[]): Range[] {
  const out: Range[] = [];
  for (const [a, b] of [...ranges].filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0])) {
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}

/** `spans` with `taken` cut out of them. */
function subtract(spans: Range[], taken: Range[]): Range[] {
  const out: Range[] = [];
  for (const [open, close] of spans) {
    let at = open;
    for (const [a, b] of taken) {
      if (b <= at || a >= close) continue;
      if (a > at) out.push([at, a]);
      at = Math.max(at, b);
    }
    if (at < close) out.push([at, close]);
  }
  return out;
}

/** The calendar settings the summary needs, read from what the desktop serves (anything missing is left unknown). */
export interface CalendarRules {
  /** Seven days, Monday first, each with its opening spans (none: closed); null when not known. */
  hours: Range[][] | null;
  /** The step between the start times offered; null when not known. */
  step: number | null;
  /** How soon from now a time may be offered. */
  notice: number;
}

export function readRules(settings: Record<string, unknown> | undefined): CalendarRules {
  const s = settings ?? {};
  const raw = Array.isArray(s.hours) && s.hours.length === 7 ? (s.hours as unknown[]) : null;
  const hours = raw?.map((day) =>
    (Array.isArray(day) ? day : [])
      .map((span) => [minutesOf((span as Record<string, unknown>)?.open), minutesOf((span as Record<string, unknown>)?.close)])
      .filter((r): r is Range => r[0] !== null && r[1] !== null && r[0] < r[1]),
  ) ?? null;
  const step = Number(s.slotMinutes);
  const notice = Number(s.noticeMinutes);
  return { hours, step: step > 0 ? Math.max(5, step) : null, notice: notice > 0 ? notice : 0 };
}

/** An appointment as the desktop lists it (only its time matters here). */
export interface Booking {
  start?: unknown;
  minutes?: unknown;
  status?: unknown;
}

/** Confirmed and requested appointments hold their time; cancelled, declined and done ones do not. */
function holdsTime(b: Booking): boolean {
  return b.status === 'confirmed' || b.status === 'requested';
}

/** One day, worked out. */
export interface DayPlan {
  date: string;
  /** Not open that day. */
  closed: boolean;
  /** Past how far ahead the desktop books: not in its free times at all. */
  ahead: boolean;
  /** The opening spans, or null when the hours are not known. */
  hours: Range[] | null;
  /** The booked time within the opening hours, merged. */
  booked: Range[];
  /** The rest of the opening hours, from the first time the notice allows. */
  free: Range[];
  /** The first start the notice allows, when it cuts into the day (today). */
  soonest: number | null;
  /** The desktop's free start times. */
  starts: number[];
}

/**
 * A day's hours, bookings and free starts, together. `now` is the desktop's
 * clock (`YYYY-MM-DDTHH:MM`); `times` the desktop's free starts for the day,
 * or undefined when the day is past how far ahead it books.
 */
export function planDay(date: string, rules: CalendarRules, bookings: Booking[], times: string[] | undefined, now: string, step: number): DayPlan {
  const [y, m, d] = date.split('-').map(Number);
  const weekday = (new Date(y, (m ?? 1) - 1, d ?? 1).getDay() + 6) % 7;
  const hours = rules.hours ? rules.hours[weekday] ?? [] : null;
  const midnight = dayNumber(date) * 1440;
  const inDay: Range[] = [];
  for (const b of bookings) {
    if (!holdsTime(b)) continue;
    const start = String(b.start ?? '');
    const at = minutesOf(start.slice(11));
    if (!/^\d{4}-\d{2}-\d{2}T/.test(start) || at === null) continue;
    const from = dayNumber(start.slice(0, 10)) * 1440 + at - midnight;
    inDay.push([from, from + Math.max(1, Number(b.minutes) || 0)]);
  }
  const bookedAll = merge(inDay);
  const open = hours ?? [[0, 1440] as Range];
  // Booked time is only what falls in the opening hours: a booking across a break shows as its two parts.
  const booked = merge(open.flatMap(([a, b]) => bookedAll.map(([x, y]): Range => [Math.max(a, x), Math.min(b, y)])));
  // The notice cuts the day at the first start it allows, on the slot grid (as the desktop offers them),
  // so a check a minute later reads the same.
  const nowDay = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(now) ? dayNumber(now.slice(0, 10)) * 1440 + (minutesOf(now.slice(11)) ?? 0) : null;
  const cut = nowDay === null ? null : nowDay + rules.notice - midnight;
  let cutInto = false;
  const bookable: Range[] = [];
  for (const [a, b] of open) {
    const first = cut !== null && cut > a ? a + Math.ceil((cut - a) / step) * step : a;
    cutInto ||= first > a;
    if (first < b) bookable.push([first, b]);
  }
  // The first time the notice allows: the start of what is left, or the day's close when nothing is.
  const soonest = cutInto ? (bookable[0]?.[0] ?? open[open.length - 1]?.[1] ?? 0) : null;
  return {
    date,
    closed: hours !== null && hours.length === 0,
    ahead: times === undefined,
    hours,
    booked: hours ? booked : [],
    free: hours ? subtract(bookable, booked) : [],
    soonest: hours ? soonest : null,
    starts: (times ?? []).map(minutesOf).filter((t): t is number => t !== null).sort((a, b) => a - b),
  };
}

/** The step between start times: the calendar's own, or the smallest gap between the free times, or half an hour. */
export function stepOf(rules: CalendarRules, days: Array<{ times: string[] }>): number {
  if (rules.step) return rules.step;
  let gap = Infinity;
  for (const day of days) {
    const t = day.times.map(minutesOf).filter((x): x is number => x !== null);
    for (let i = 1; i < t.length; i++) if (t[i] > t[i - 1]) gap = Math.min(gap, t[i] - t[i - 1]);
  }
  return Number.isFinite(gap) ? gap : 30;
}

/** Start times in runs, a step apart: [first, last] each. */
function windows(starts: number[], step: number): Range[] {
  const out: Range[] = [];
  for (const t of starts) {
    const last = out[out.length - 1];
    if (last && t - last[1] === step) last[1] = t;
    else out.push([t, t]);
  }
  return out;
}

/** "at 8 am, or any time from 10:30 am to 4 pm". */
export function sayStarts(starts: number[], step: number): string {
  const said = windows(starts, step).map(([a, b]) => (a === b ? `at ${clock(a)}` : `any time from ${clock(a)} to ${clock(b)}`));
  return said.length < 2 ? (said[0] ?? '') : `${said.slice(0, -1).join(', ')}, or ${said[said.length - 1]}`;
}

/** "Fri 2 Oct (2026-10-02)", with ", today" on today. */
export function dayLabel(date: string, today: string): string {
  return `${sayDate(date)} (${date}${date === today ? ', today' : ''})`;
}

/**
 * A day as one line. `what` is what is being booked ("a 60-min Lawn mowing");
 * `next` the next day with room, for a day where nothing fits.
 */
export function sayDay(plan: DayPlan, what: string, step: number, today: string, next?: string): string {
  const label = dayLabel(plan.date, today);
  const fits = plan.starts.length ? `${what} can start ${sayStarts(plan.starts, step)}` : '';
  const none = `nothing fits ${what}${next ? ` (the next day with room is ${next})` : ''}`;
  if (!plan.hours) return `${label}: ${fits || none}.`;
  const hours = sayList(plan.hours.map(sayRange));
  if (!plan.booked.length && plan.soonest === null) {
    return fits ? `${label}: free all day, ${hours} (${fits}).` : `${label}: free all day, ${hours}; ${none}.`;
  }
  const parts = [`open ${hours}`];
  if (plan.booked.length) parts.push(`booked ${sayList(plan.booked.map(sayRange))}`);
  const closes = plan.hours[plan.hours.length - 1]?.[1] ?? 0;
  const over = plan.soonest !== null && plan.soonest >= closes;
  if (plan.soonest !== null && !over) parts.push(`too soon before ${clock(plan.soonest)}`);
  if (plan.free.length) parts.push(`free ${sayList(plan.free.map(sayRange))}`);
  else parts.push(over ? `no more times${plan.date === today ? ' today' : ''} (too short notice)` : 'nothing free');
  return fits ? `${label}: ${parts.join('; ')} (${fits}).` : `${label}: ${parts.join('; ')}; ${none}.`;
}

/**
 * Days in one line: "Sat 3 Oct and Sun 4 Oct", or "from Sat 31 Oct to Tue 3 Nov"
 * for more than three in a row (weekends a week apart are each named: a range
 * would read as the days between closed too).
 */
export function sayDays(dates: string[]): string {
  const inARow = dates.every((d, i) => i === 0 || dayNumber(d) - dayNumber(dates[i - 1]) === 1);
  return dates.length > 3 && inARow ? `from ${sayDate(dates[0])} to ${sayDate(dates[dates.length - 1])}` : sayList(dates.map(sayDate));
}

/**
 * Why a start time on a day is not offered, in words any audience may hear
 * (a clash says so only to the owner). Empty when there is no plainer reason.
 */
export function whyNot(plan: DayPlan, at: number, minutes: number, step: number, audience: Audience): string {
  if (plan.ahead) return 'too far ahead to book yet';
  if (plan.closed) return 'closed that day';
  if (plan.hours) {
    const span = plan.hours.find(([a, b]) => at >= a && at < b);
    if (!span) return `outside the opening hours, ${sayList(plan.hours.map(sayRange))}`;
    if (at + minutes > span[1]) return `it would run past ${clock(span[1])}`;
    if (plan.soonest !== null && at < plan.soonest) return 'too short notice';
    if ((at - span[0]) % step) return `start times are every ${step} minutes`;
  }
  if (audience === 'owner' && plan.booked.some(([a, b]) => at < b && a < at + minutes)) return 'it overlaps a booking; calendar_list says whose';
  return '';
}

export { clock as sayClock };

/** Who the other person is, as the rule names them. */
const WHO: Record<Audience, string> = { owner: 'your person', caller: 'caller', texter: 'person texting' };

/** How to use the summary, for the agent reading it: the same words in the tool's description. */
export function freeRule(audience: Audience): string {
  const naturally = "Say ranges naturally ('any time after 10:30 in the morning, or in the afternoon') rather than listing every half hour.";
  if (audience === 'owner') {
    return `Booked times are shown only to work out what is free. This is your person's own calendar: calendar_list says who is booked, when they want to know. Offer times inside the free ranges. ${naturally} Times passed on to a customer are only what is free: never who else is booked, or what.`;
  }
  return `Offer times inside the free ranges. Never tell the ${WHO[audience]} about other bookings: not who, not what, not that a time is 'booked by someone'. Say a time isn't available and offer the nearest free one. ${naturally}`;
}

/** Said before a check that reads the same as one a moment ago, so the agent does not say it all again. */
export function repeatNote(audience: Audience): string {
  return audience === 'owner'
    ? '(the same as your check a moment ago: carry on from where you were)'
    : `(the same as your check a moment ago: carry on from where you were; don't repeat it to the ${WHO[audience]})`;
}
