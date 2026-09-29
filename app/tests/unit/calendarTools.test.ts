import { afterEach, describe, expect, it, vi } from 'vitest';
import { calendarTools, callCalendarTools, sayTime, textCalendarTools } from '../../src/desktop/calendarTools';
import type { Desktop } from '../../src/desktop/bridge';

type Span = { open: string; close: string };
type Appt = { id?: string; start: string; minutes: number; status: string; service?: string; name?: string; phone?: string; notes?: string };

const dayNum = (ymd: string) => Math.round(Date.UTC(+ymd.slice(0, 4), +ymd.slice(5, 7) - 1, +ymd.slice(8, 10)) / 86_400_000);
const hm = (t: string) => +t.slice(0, 2) * 60 + +t.slice(3, 5);
const abs = (start: string) => dayNum(start.slice(0, 10)) * 1440 + hm(start.slice(11, 16));
const ymdOf = (n: number) => new Date(n * 86_400_000).toISOString().slice(0, 10);
const hhmm = (m: number) => `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
const open = (a: string, b: string): Span[] => [{ open: a, close: b }];
/** Monday first: 8 to 5 on weekdays, closed at the weekend (the live calendar's hours). */
const WEEKDAYS = [open('08:00', '17:00'), open('08:00', '17:00'), open('08:00', '17:00'), open('08:00', '17:00'), open('08:00', '17:00'), [], []];

/**
 * A fake desktop with a calendar: its free times worked out as the desktop's
 * own (`Calendar::free` in calendar/mod.rs): opening spans stepped by the slot,
 * no clash with a time-holding appointment, none sooner than the notice, none
 * past how far ahead it books.
 */
function calendarDesktop(o: { now: string; appointments?: Appt[]; hours?: Span[][]; notice?: number; horizon?: number }) {
  const appointments = [...(o.appointments ?? [])];
  const settings = {
    business: 'Green Lawns',
    hours: o.hours ?? WEEKDAYS,
    services: [
      { id: 'lawn-mowing', name: 'Lawn mowing', minutes: 60 },
      { id: 'hedge-trimming', name: 'Hedge trimming', minutes: 90 },
      { id: 'quote-visit', name: 'Quote visit', minutes: 30 },
    ],
    slotMinutes: 30,
    noticeMinutes: o.notice ?? 60,
    horizonDays: o.horizon ?? 30,
  };
  const holds = (a: Appt) => a.status === 'confirmed' || a.status === 'requested';
  const desktop = {
    calendar: async (from: string, to: string) => ({ available: true, settings, now: o.now, appointments: appointments.filter((a) => a.start.slice(0, 10) >= from && a.start.slice(0, 10) < to) }),
    calendarFree: async (from: string, days: number, service?: string) => {
      const svc = settings.services.find((s) => s.name.toLowerCase() === service?.toLowerCase());
      const minutes = svc?.minutes ?? settings.slotMinutes;
      const earliest = abs(o.now) + settings.noticeMinutes;
      const lastDay = dayNum(o.now.slice(0, 10)) + settings.horizonDays;
      const taken = appointments.filter(holds).map((a) => [abs(a.start), abs(a.start) + a.minutes]);
      const out = [];
      for (let i = 0; i < days; i++) {
        const n = dayNum(from) + i;
        if (n > lastDay) break;
        const date = ymdOf(n);
        const times: string[] = [];
        for (const span of settings.hours[(new Date(`${date}T12:00:00Z`).getUTCDay() + 6) % 7]) {
          for (let t = n * 1440 + hm(span.open); t + minutes <= n * 1440 + hm(span.close); t += settings.slotMinutes) {
            if (t >= earliest && !taken.some(([a, b]) => t < b && a < t + minutes)) times.push(hhmm(t - n * 1440));
          }
        }
        out.push({ date, times });
      }
      return { minutes, service: svc?.name ?? null, days: out };
    },
    calendarCreate: async (a: Record<string, unknown>) => {
      const made = { id: `appt_${appointments.length + 1}`, start: `${a.date}T${a.time}`, minutes: 60, status: String(a.status ?? 'confirmed'), service: String(a.service ?? ''), name: String(a.name ?? '') };
      appointments.push(made);
      return made;
    },
    // PATCH /api/calendar/appointments/:id: the fields given.
    calendarUpdate: async (id: string, change: Record<string, unknown>) => {
      const a = appointments.find((x) => x.id === id);
      if (!a) throw new Error(`no appointment ${id}`);
      Object.assign(a, change);
      return a;
    },
  } as unknown as Desktop;
  return { desktop, appointments };
}

/** The line for one day in a summary. */
const dayLine = (text: string, date: string) => text.split('\n').find((l) => l.includes(`(${date}`)) ?? '';

const signal = new AbortController().signal;

afterEach(() => {
  vi.useRealTimers();
});

describe("the calendar as the agent's tools", () => {
  it('says times as people do', () => {
    expect([sayTime('09:00'), sayTime('09:30'), sayTime('12:00'), sayTime('14:15'), sayTime('00:00')]).toEqual(['9 am', '9:30 am', '12 pm', '2:15 pm', '12 am']);
  });

  it("a text thread sees what is free, and asks for an appointment for the person texting (a request, never confirmed)", async () => {
    const { desktop, appointments } = calendarDesktop({ now: '2026-09-29T09:00' });
    const [free, request] = textCalendarTools(() => desktop, '+61491570006', () => 'Lance');
    expect(free.spec.name).toBe('calendar_free_times');
    const times = await free.run({ from: '2026-10-02', days: 1, service: 'Lawn mowing' }, signal);
    expect(times).toContain('Fri 2 Oct (2026-10-02): free all day, 8 am–5 pm (a 60-min Lawn mowing can start any time from 8 am to 4 pm).');
    expect(request.spec.name).toBe('request_appointment');
    expect(request.spec.description).toContain('never say it is booked or confirmed');
    const out = await request.run({ service: 'Lawn mowing', date: '2026-10-02', time: '09:30' }, signal);
    expect(appointments[0]).toMatchObject({ service: 'Lawn mowing', start: '2026-10-02T09:30', name: 'Lance', status: 'requested' });
    expect(out).toContain('Staff will confirm it');
  });

  it('the main agent lists, books and changes appointments', async () => {
    const created: Array<Record<string, unknown>> = [];
    const changed: Array<[string, Record<string, unknown>]> = [];
    const desktop = {
      calendarCreate: async (a: Record<string, unknown>) => {
        created.push(a);
        return { id: 'appt_1', service: a.service, start: `${a.date}T${a.time}`, status: a.status ?? 'confirmed' };
      },
      calendarUpdate: async (id: string, change: Record<string, unknown>) => {
        changed.push([id, change]);
        return { id, service: 'Lawn mowing', start: '2026-10-01T10:00', status: change.status };
      },
      calendar: async () => ({
        settings: {},
        now: '',
        appointments: [{ id: 'appt_2', start: '2026-10-01T10:30', service: 'Quote visit', minutes: 30, name: 'Lance', phone: '+61491570006', status: 'requested', notes: '' }],
      }),
    } as unknown as Desktop;
    const tools = calendarTools(() => desktop);
    expect(tools.map((t) => t.spec.name)).toEqual(['calendar_free_times', 'calendar_list', 'calendar_book', 'calendar_change']);
    const list = await tools[1].run({ from: '2026-09-28', days: 7 }, signal);
    expect(list).toBe('Thu 1 Oct 10:30 am · Quote visit (30 min) · Lance, +61491570006 · requested · id appt_2');
    await tools[2].run({ date: '2026-10-02', time: '11:00', service: 'Lawn mowing', name: 'Sam' }, signal);
    expect(created[0]).toMatchObject({ date: '2026-10-02', time: '11:00', source: 'agent' });
    const out = await tools[3].run({ id: 'appt_2', status: 'confirmed' }, signal);
    expect(changed[0]).toEqual(['appt_2', { status: 'confirmed' }]);
    expect(out).toBe('Changed: Lawn mowing on 2026-10-01 at 10:00, confirmed.');
    await expect(calendarTools(() => null)[0].run({}, signal)).rejects.toThrow('not connected');
  });
});

describe("cancel_appointment: the caller's own, found by their number", () => {
  /** Lance's request on Thursday and confirmed booking on Friday; Sam's booking on Monday. */
  const booked = () =>
    calendarDesktop({
      now: '2026-09-29T09:00',
      appointments: [
        { id: 'a1', start: '2026-10-01T10:00', minutes: 60, status: 'requested', service: 'Lawn mowing', name: 'Lance', phone: '0491570006' },
        { id: 'a2', start: '2026-10-02T13:00', minutes: 60, status: 'confirmed', service: 'Lawn mowing', name: 'Lance', phone: '+61 491 570 006', notes: 'Side gate' },
        { id: 'a3', start: '2026-10-05T09:00', minutes: 90, status: 'confirmed', service: 'Hedge trimming', name: 'Sam', phone: '+61411111111' },
      ],
    });
  const cancel = (desktop: Desktop, phone: string) => callCalendarTools(() => desktop, () => phone)[1];

  it('a request not yet confirmed is cancelled at once', async () => {
    const { desktop, appointments } = booked();
    const tool = cancel(desktop, '+61491570006');
    expect(tool.spec.name).toBe('cancel_appointment');
    expect(tool.spec.description).toContain("don't check the calendar again");
    const out = await tool.run({ date: '2026-10-01' }, signal);
    expect(out).toBe("Cancelled: their Lawn mowing on Thu 1 Oct at 10 am (a request, not yet confirmed). Tell them it is cancelled. Don't check the calendar again.");
    expect(appointments[0].status).toBe('cancelled');
    expect(appointments[0].notes).toMatch(/^Cancelled by the caller on a call \(\w{3} \d+ \w{3}\)\.$/);
    expect(appointments.slice(1).map((a) => a.status)).toEqual(['confirmed', 'confirmed']);
  });

  it('a confirmed booking is passed to staff: the cancellation is asked for in a note on it, once', async () => {
    const { desktop, appointments } = booked();
    const tool = cancel(desktop, '0491 570 006');
    const out = await tool.run({ date: '2026-10-02', time: '13:00' }, signal);
    expect(out).toBe("Their Lawn mowing on Fri 2 Oct at 1 pm is confirmed, so staff cancel it: the cancellation is asked for (noted on it for the team). Tell them the team will confirm the cancellation. Don't check the calendar again.");
    expect(appointments[1].status).toBe('confirmed');
    expect(appointments[1].notes).toMatch(/^Side gate\nCancellation asked for by the caller on a call \(.+\): staff to cancel it and let them know\.$/);
    await tool.run({ date: '2026-10-02' }, signal);
    expect(appointments[1].notes!.match(/Cancellation asked for/g)).toHaveLength(1);
    // The person texting has it too, for their own number.
    const byText = textCalendarTools(() => desktop, '+61491570006', () => 'Lance')[2];
    expect(byText.spec.name).toBe('cancel_appointment');
    expect(byText.spec.description).toContain("person texting's own appointment");
    expect(await byText.run({ date: '2026-10-01' }, signal)).toContain('Cancelled: their Lawn mowing on Thu 1 Oct at 10 am');
    expect(appointments[0].notes).toMatch(/^Cancelled by the person texting by text/);
  });

  it("another person's booking is never touched or told of", async () => {
    const { desktop, appointments } = booked();
    const out = await cancel(desktop, '+61491570006').run({ date: '2026-10-05' }, signal);
    expect(out).toBe('They have no appointment on Mon 5 Oct (found by their number): nothing was cancelled. Tell them so, and ask which day they mean.');
    for (const secret of ['Sam', 'Hedge', '9 am']) expect(out).not.toContain(secret);
    expect(appointments[2]).toMatchObject({ status: 'confirmed' });
    expect(appointments[2].notes).toBeUndefined();
    // A hidden number finds nothing, whatever the day (a hidden caller's key is their call's id, which has digits).
    for (const hidden of ['', 'Private', 'call_1bdd37']) expect(await cancel(desktop, hidden).run({ date: '2026-10-01' }, signal)).toContain('Their number is hidden');
    expect(appointments.map((a) => a.status)).toEqual(['requested', 'confirmed', 'confirmed']);
  });

  it('no match: none that day, none at that time, or two to choose from', async () => {
    const { desktop, appointments } = calendarDesktop({
      now: '2026-09-29T09:00',
      appointments: [
        { id: 'b1', start: '2026-10-01T09:00', minutes: 30, status: 'requested', service: 'Quote visit', phone: '0491570006' },
        { id: 'b2', start: '2026-10-01T14:00', minutes: 60, status: 'confirmed', service: 'Lawn mowing', phone: '0491570006' },
        { id: 'b3', start: '2026-10-02T10:00', minutes: 60, status: 'cancelled', service: 'Lawn mowing', phone: '0491570006' },
      ],
    });
    const tool = cancel(desktop, '+61491570006');
    expect(await tool.run({ date: '2026-10-03' }, signal)).toBe('They have no appointment on Sat 3 Oct (found by their number): nothing was cancelled. Tell them so, and ask which day they mean.');
    // One already cancelled is not theirs to cancel again.
    expect(await tool.run({ date: '2026-10-02' }, signal)).toContain('They have no appointment on Fri 2 Oct');
    expect(await tool.run({ date: '2026-10-01' }, signal)).toBe('They have 2 that day: Quote visit on Thu 1 Oct at 9 am and Lawn mowing on Thu 1 Oct at 2 pm. Nothing was cancelled yet: ask which, then cancel_appointment with its time.');
    expect(await tool.run({ date: '2026-10-01', time: '11:00' }, signal)).toBe('They have none at 11 am on Thu 1 Oct: theirs that day are Quote visit on Thu 1 Oct at 9 am and Lawn mowing on Thu 1 Oct at 2 pm. Nothing was cancelled: ask which they mean.');
    expect(appointments.map((a) => a.status)).toEqual(['requested', 'confirmed', 'cancelled']);
    await expect(tool.run({ date: 'Thursday' }, signal)).rejects.toThrow('date is YYYY-MM-DD');
    // Named by its time: that one.
    expect(await tool.run({ date: '2026-10-01', time: '9:00' }, signal)).toContain('Cancelled: their Quote visit on Thu 1 Oct at 9 am');
    expect(appointments.map((a) => a.status)).toEqual(['cancelled', 'confirmed', 'cancelled']);
  });
});

describe('what is free, as a summary per day', () => {
  it('a day with bookings says the booked and free ranges, and when the service can start', async () => {
    const { desktop } = calendarDesktop({
      now: '2026-09-29T09:00',
      appointments: [
        { start: '2026-10-01T09:00', minutes: 90, status: 'confirmed', service: 'Hedge trimming', name: 'Sam' },
        { start: '2026-10-01T11:00', minutes: 60, status: 'requested', service: 'Lawn mowing', name: 'Kim' },
        // Cancelled and declined ones hold no time.
        { start: '2026-10-01T14:00', minutes: 60, status: 'cancelled', service: 'Lawn mowing', name: 'Ann' },
        { start: '2026-10-01T15:00', minutes: 60, status: 'declined', service: 'Lawn mowing', name: 'Bo' },
      ],
    });
    const out = await calendarTools(() => desktop)[0].run({ from: '2026-10-01', days: 1, service: 'Lawn mowing' }, signal);
    expect(out).toBe(
      [
        'What is free for a 60-min Lawn mowing:',
        'Thu 1 Oct (2026-10-01): open 8 am–5 pm; booked 9–10:30 am and 11 am–12 pm; free 8–9 am, 10:30–11 am and 12–5 pm (a 60-min Lawn mowing can start at 8 am, or any time from 12 pm to 4 pm).',
        "Booked times are shown only to work out what is free. This is your person's own calendar: calendar_list says who is booked, when they want to know. Offer times inside the free ranges. Say ranges naturally ('any time after 10:30 in the morning, or in the afternoon') rather than listing every half hour. Times passed on to a customer are only what is free: never who else is booked, or what.",
      ].join('\n'),
    );
  });

  it('a day with nothing booked is free all day', async () => {
    const { desktop } = calendarDesktop({ now: '2026-09-29T09:00' });
    const out = await callCalendarTools(() => desktop)[0].run({ from: '2026-10-02', days: 1, service: 'Lawn mowing' }, signal);
    expect(dayLine(out, '2026-10-02')).toBe('Fri 2 Oct (2026-10-02): free all day, 8 am–5 pm (a 60-min Lawn mowing can start any time from 8 am to 4 pm).');
  });

  it('closed days are folded into one line, and a closed-only ask names the next day with room', async () => {
    const { desktop } = calendarDesktop({ now: '2026-09-29T09:00' });
    const [free] = callCalendarTools(() => desktop);
    const week = await free.run({ from: '2026-10-02', days: 3, service: 'Lawn mowing' }, signal);
    expect(week.split('\n').slice(0, 3)).toEqual([
      'What is free for a 60-min Lawn mowing:',
      'Fri 2 Oct (2026-10-02): free all day, 8 am–5 pm (a 60-min Lawn mowing can start any time from 8 am to 4 pm).',
      'Closed: Sat 3 Oct and Sun 4 Oct.',
    ]);
    // Two weekends are each named (a range would take in the week between).
    const fortnight = await free.run({ from: '2026-10-02', days: 10, service: 'Lawn mowing' }, signal);
    expect(fortnight).toContain('\nClosed: Sat 3 Oct, Sun 4 Oct, Sat 10 Oct and Sun 11 Oct.\n');
    // Days in a row past how far ahead it books are one range.
    const ahead = calendarDesktop({ now: '2026-09-29T09:00', horizon: 5 }).desktop;
    const far = await callCalendarTools(() => ahead)[0].run({ from: '2026-10-02', days: 10, service: 'Lawn mowing' }, signal);
    expect(far).toContain('\nClosed: Sat 3 Oct and Sun 4 Oct.\nToo far ahead to book yet: from Mon 5 Oct to Sun 11 Oct.\n');
    const weekend = await free.run({ from: '2026-10-03', days: 2, service: 'Lawn mowing' }, signal);
    expect(weekend).toContain('Closed: Sat 3 Oct and Sun 4 Oct.\nNothing fits a 60-min Lawn mowing on the days asked. The next day with room: Mon 5 Oct (2026-10-05), when a 60-min Lawn mowing can start any time from 8 am to 4 pm.');
  });

  it('a split day (a lunch break) shows both spans, and a booking across the break as its two parts', async () => {
    const split = [...open('08:00', '12:00'), ...open('13:00', '17:00')];
    const hours = [split, split, split, split, split, [], []];
    const { desktop } = calendarDesktop({
      now: '2026-09-28T07:00',
      hours,
      appointments: [
        { start: '2026-09-30T10:00', minutes: 60, status: 'confirmed', service: 'Lawn mowing', name: 'Sam' },
        { start: '2026-10-01T11:30', minutes: 120, status: 'requested', service: 'Hedge trimming', name: 'Kim' },
      ],
    });
    const out = await calendarTools(() => desktop)[0].run({ from: '2026-09-30', days: 2, service: 'Lawn mowing' }, signal);
    expect(dayLine(out, '2026-09-30')).toBe(
      'Wed 30 Sep (2026-09-30): open 8 am–12 pm and 1–5 pm; booked 10–11 am; free 8–10 am, 11 am–12 pm and 1–5 pm (a 60-min Lawn mowing can start any time from 8 am to 9 am, at 11 am, or any time from 1 pm to 4 pm).',
    );
    expect(dayLine(out, '2026-10-01')).toBe(
      'Thu 1 Oct (2026-10-01): open 8 am–12 pm and 1–5 pm; booked 11:30 am–12 pm and 1–1:30 pm; free 8–11:30 am and 1:30–5 pm (a 60-min Lawn mowing can start any time from 8 am to 10:30 am, or any time from 1:30 pm to 4 pm).',
    );
  });

  it("today's notice cuts the morning at the first start it allows, and reads the same a minute later", async () => {
    const at = (now: string) => calendarDesktop({ now, appointments: [{ start: '2026-09-29T14:00', minutes: 60, status: 'confirmed' }] }).desktop;
    const first = await calendarTools(() => at('2026-09-29T09:10'))[0].run({ from: '2026-09-29', days: 1, service: 'Lawn mowing' }, signal);
    expect(dayLine(first, '2026-09-29')).toBe(
      'Tue 29 Sep (2026-09-29, today): open 8 am–5 pm; booked 2–3 pm; too soon before 10:30 am; free 10:30 am–2 pm and 3–5 pm (a 60-min Lawn mowing can start any time from 10:30 am to 1 pm, or any time from 3 pm to 4 pm).',
    );
    const later = await calendarTools(() => at('2026-09-29T09:25'))[0].run({ from: '2026-09-29', days: 1, service: 'Lawn mowing' }, signal);
    expect(later).toBe(first);
    // After hours: nothing more today.
    const evening = await calendarTools(() => at('2026-09-29T16:30'))[0].run({ from: '2026-09-29', days: 1, service: 'Lawn mowing' }, signal);
    expect(dayLine(evening, '2026-09-29')).toBe('Tue 29 Sep (2026-09-29, today): open 8 am–5 pm; booked 2–3 pm; no more times today (too short notice); nothing fits a 60-min Lawn mowing.');
    expect(evening).toContain('The next day with room: Wed 30 Sep (2026-09-30)');
  });

  it('a day where nothing fits says so, and names the next day with room', async () => {
    const { desktop } = calendarDesktop({
      now: '2026-09-29T09:00',
      appointments: [
        { start: '2026-10-01T08:00', minutes: 540, status: 'confirmed', service: 'Hedge trimming', name: 'Sam' },
        { start: '2026-10-02T08:00', minutes: 510, status: 'confirmed', service: 'Hedge trimming', name: 'Sam' },
      ],
    });
    const [free] = textCalendarTools(() => desktop, '+61400000000', () => 'Lee');
    // Thursday is full; Friday, asked with it, has room only too short for a mowing.
    const two = await free.run({ from: '2026-10-01', days: 2, service: 'Quote visit' }, signal);
    expect(dayLine(two, '2026-10-01')).toBe('Thu 1 Oct (2026-10-01): open 8 am–5 pm; booked 8 am–5 pm; nothing free; nothing fits a 30-min Quote visit (the next day with room is Fri 2 Oct).');
    expect(dayLine(two, '2026-10-02')).toBe('Fri 2 Oct (2026-10-02): open 8 am–5 pm; booked 8 am–4:30 pm; free 4:30–5 pm (a 30-min Quote visit can start at 4:30 pm).');
    // Friday alone, for a mowing: nothing fits, and the next day with room is after the weekend.
    const friday = await free.run({ from: '2026-10-02', days: 1, service: 'Lawn mowing' }, signal);
    expect(friday.split('\n').slice(0, 3)).toEqual([
      'What is free for a 60-min Lawn mowing:',
      'Fri 2 Oct (2026-10-02): open 8 am–5 pm; booked 8 am–4:30 pm; free 4:30–5 pm; nothing fits a 60-min Lawn mowing.',
      'Nothing fits a 60-min Lawn mowing on the day asked. The next day with room: Mon 5 Oct (2026-10-05), when a 60-min Lawn mowing can start any time from 8 am to 4 pm.',
    ]);
  });

  it('a specific time: yes, or no with the nearest free starts before and after', async () => {
    const { desktop } = calendarDesktop({ now: '2026-09-29T13:02', appointments: [{ start: '2026-10-02T13:00', minutes: 60, status: 'requested', service: 'Quote visit', name: 'Lance' }] });
    const [caller] = callCalendarTools(() => desktop);
    const yes = await caller.run({ from: '2026-10-02', time: '10:00', service: 'Lawn mowing' }, signal);
    expect(yes.split('\n')[0]).toBe('Yes: 10 am on Fri 2 Oct (2026-10-02) is free for a 60-min Lawn mowing.');
    const no = await caller.run({ from: '2026-10-02', time: '1 pm', service: 'Lawn mowing' }, signal);
    expect(no.split('\n')[0]).toBe("No: 1 pm on Fri 2 Oct (2026-10-02) isn't available for a 60-min Lawn mowing. The nearest free starts: 12 pm before it, and 2 pm after it.");
    // The day's summary follows, so the agent can offer ranges.
    expect(no).toContain('Fri 2 Oct (2026-10-02): open 8 am–5 pm; booked 1–2 pm;');
    const late = await caller.run({ from: '2026-10-02', time: '16:30', service: 'Lawn mowing' }, signal);
    expect(late.split('\n')[0]).toBe("No: 4:30 pm on Fri 2 Oct (2026-10-02) isn't available for a 60-min Lawn mowing (it would run past 5 pm). The nearest free starts: 4 pm before it, and Mon 5 Oct at 8 am after it.");
    // The owner's agent is told why: a booking (and where to see whose).
    const owner = await calendarTools(() => desktop)[0].run({ from: '2026-10-02', time: '13:00', service: 'Lawn mowing' }, signal);
    expect(owner.split('\n')[0]).toContain("isn't available for a 60-min Lawn mowing (it overlaps a booking; calendar_list says whose)");
    await expect(caller.run({ from: '2026-10-02', time: 'after lunch' }, signal)).rejects.toThrow('time is HH:MM');
  });

  it('the same check again within two minutes says so, unless the calendar changed', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-09-29T13:00:00'));
    const { desktop } = calendarDesktop({ now: '2026-09-29T13:00' });
    const [free, request] = textCalendarTools(() => desktop, '+61400000000', () => 'Lee');
    const ask = { from: '2026-10-02', days: 1, service: 'Lawn mowing' };
    const first = await free.run(ask, signal);
    expect(first.startsWith('(')).toBe(false);
    vi.setSystemTime(new Date('2026-09-29T13:01:00'));
    const again = await free.run({ ...ask, service: 'lawn mowing' }, signal);
    expect(again).toBe(`(the same as your check a moment ago: carry on from where you were; don't repeat it to the person texting)\n${first}`);
    // A booking in between: it reads differently, so no note.
    await request.run({ service: 'Lawn mowing', date: '2026-10-02', time: '09:00' }, signal);
    const changed = await free.run(ask, signal);
    expect(changed.startsWith('(')).toBe(false);
    expect(changed).toContain('booked 9–10 am');
    // Three minutes later it is a new check.
    vi.setSystemTime(new Date('2026-09-29T13:04:30'));
    expect((await free.run(ask, signal)).startsWith('(')).toBe(false);
    // A call's note speaks of the caller.
    const [call] = callCalendarTools(() => desktop);
    await call.run(ask, signal);
    expect((await call.run(ask, signal)).split('\n')[0]).toBe("(the same as your check a moment ago: carry on from where you were; don't repeat it to the caller)");
  });

  it("the phone's and a text thread's wording never names who is booked or for what", async () => {
    const { desktop } = calendarDesktop({
      now: '2026-09-29T09:00',
      appointments: [
        { start: '2026-10-01T10:00', minutes: 90, status: 'confirmed', service: 'Hedge trimming', name: 'Sam Smith', phone: '+61411111111', notes: 'gate code 4321' },
        { start: '2026-10-02T13:00', minutes: 30, status: 'requested', service: 'Quote visit', name: 'Lance', phone: '0491570006' },
      ],
    });
    const tools = [callCalendarTools(() => desktop)[0], textCalendarTools(() => desktop, '+61400000000', () => 'Lee')[0]];
    for (const tool of tools) {
      const said = [
        tool.spec.description,
        await tool.run({ from: '2026-09-29', days: 7, service: 'Lawn mowing' }, signal),
        await tool.run({ from: '2026-10-02', time: '13:00', service: 'Lawn mowing' }, signal),
        await tool.run({ from: '2026-10-01', time: '10:30' }, signal),
      ].join('\n');
      for (const secret of ['Sam', 'Smith', 'Lance', 'Hedge', 'Quote', 'gate', '4321', '61411111111', '0491570006', 'calendar_list', 'overlaps']) expect(said).not.toContain(secret);
      expect(said).toContain("Never tell the");
      expect(said).toContain("not that a time is 'booked by someone'");
      expect(said).toContain('Say ranges naturally');
    }
    expect(tools[0].spec.description).toContain('Never tell the caller about other bookings');
    expect(tools[1].spec.description).toContain('Never tell the person texting about other bookings');
  });

  it('the live calendar: Friday 2 October, with a request at 1 pm', async () => {
    const { desktop } = calendarDesktop({
      now: '2026-09-29T13:02',
      appointments: [
        { start: '2026-10-01T10:00', minutes: 60, status: 'requested', service: 'Lawn mowing', name: 'Lance' },
        { start: '2026-10-02T13:00', minutes: 60, status: 'requested', service: 'Lawn mowing', name: 'Lance' },
      ],
    });
    const out = await callCalendarTools(() => desktop)[0].run({ from: '2026-10-02', days: 1, service: 'Lawn mowing' }, signal);
    expect(out).toBe(
      [
        'What is free for a 60-min Lawn mowing:',
        'Fri 2 Oct (2026-10-02): open 8 am–5 pm; booked 1–2 pm; free 8 am–1 pm and 2–5 pm (a 60-min Lawn mowing can start any time from 8 am to 12 pm, or any time from 2 pm to 4 pm).',
        "Offer times inside the free ranges. Never tell the caller about other bookings: not who, not what, not that a time is 'booked by someone'. Say a time isn't available and offer the nearest free one. Say ranges naturally ('any time after 10:30 in the morning, or in the afternoon') rather than listing every half hour.",
      ].join('\n'),
    );
  });
});
