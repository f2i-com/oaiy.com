// The calendar's pure parts: the opening hours' quick actions, the checks a
// service and the settings pass before they are sent (the desktop's own), the
// free-time picker's choices, and where appointments sit in a day.
import { describe, expect, it } from 'vitest';
import type { CalendarSettings, CalendarSpan } from './api';
import {
  checkService,
  checkSettings,
  copyToOpenDays,
  gridHours,
  groupTimes,
  layoutDay,
  normalHours,
  normalSettings,
  openDay,
  pickTime,
  receptionistName,
  sameHoursWeekdays,
  sameSettings,
  sayGreeting,
  sayHours,
  sayMinutes,
  sayRange,
  sayWeek,
  snapDown,
  toHHMM,
  withBreak,
} from './calendarModel';

const span = (open: string, close: string): CalendarSpan => ({ open, close });
const WEEK = (): CalendarSpan[][] => [
  [span('08:00', '17:00')],
  [span('09:00', '17:00')],
  [],
  [span('10:00', '14:00')],
  [span('08:00', '17:00')],
  [span('09:00', '12:00')],
  [],
];

const settings = (patch: Partial<CalendarSettings> = {}): CalendarSettings => ({
  business: 'Green Lawns',
  receptionist: '',
  hours: WEEK(),
  services: [
    { id: 'mow', name: 'Lawn mowing', minutes: 60, price: 'from $60' },
    { id: 'hedge', name: 'Hedge trimming', minutes: 90 },
  ],
  slotMinutes: 30,
  noticeMinutes: 60,
  horizonDays: 30,
  textConfirmations: true,
  ...patch,
});

describe('the opening hours’ quick actions', () => {
  it('“Same hours Mon–Fri” opens every weekday with the first open weekday’s hours, and leaves the weekend', () => {
    const hours = sameHoursWeekdays(WEEK());
    expect(hours.slice(0, 5)).toEqual(Array.from({ length: 5 }, () => [span('08:00', '17:00')]));
    expect(hours[5]).toEqual([span('09:00', '12:00')]);
    expect(hours[6]).toEqual([]);
  });

  it('“Same hours Mon–Fri” can start from a given day, and falls back to 9 to 5 with no weekday open', () => {
    expect(sameHoursWeekdays(WEEK(), 3)[0]).toEqual([span('10:00', '14:00')]);
    const shut = [[], [], [], [], [], [span('09:00', '12:00')], []];
    expect(sameHoursWeekdays(shut).slice(0, 5)).toEqual(Array.from({ length: 5 }, () => [span('09:00', '17:00')]));
  });

  it('“Copy to all open days” gives every open day that day’s hours; closed days stay closed', () => {
    const hours = copyToOpenDays(WEEK(), 3);
    expect(hours).toEqual([
      [span('10:00', '14:00')],
      [span('10:00', '14:00')],
      [],
      [span('10:00', '14:00')],
      [span('10:00', '14:00')],
      [span('10:00', '14:00')],
      [],
    ]);
    // The copies are copies: changing one does not change another.
    hours[0][0].open = '07:00';
    expect(hours[1][0].open).toBe('10:00');
  });

  it('copies a day with a break as a whole, and copying a closed day changes nothing', () => {
    const week = WEEK();
    week[0] = [span('08:00', '12:00'), span('13:00', '17:00')];
    expect(copyToOpenDays(week, 0)[4]).toEqual([span('08:00', '12:00'), span('13:00', '17:00')]);
    expect(copyToOpenDays(WEEK(), 2)).toEqual(WEEK());
  });

  it('a day opened takes the hours of the open day before it', () => {
    expect(openDay(WEEK(), 2)[2]).toEqual([span('09:00', '17:00')]);
    expect(openDay([[], [], [], [], [], [], []], 6)[6]).toEqual([span('09:00', '17:00')]);
  });

  it('a break goes at midday when the day spans it, else in the middle; a short day has none', () => {
    expect(withBreak([span('08:00', '17:00')])).toEqual([span('08:00', '12:00'), span('13:00', '17:00')]);
    expect(withBreak([span('13:00', '19:00')])).toEqual([span('13:00', '15:30'), span('16:30', '19:00')]);
    expect(withBreak([span('09:00', '10:30')])).toBeNull();
  });

  it('says the week as callers hear it', () => {
    expect(sayHours(sameHoursWeekdays(WEEK()))).toBe('Mon–Fri 8 am – 5 pm · Sat 9 am – 12 pm · Sun closed');
    expect(sayHours(WEEK())).toBe('Mon 8 am – 5 pm · Tue 9 am – 5 pm · Wed closed · Thu 10 am – 2 pm · Fri 8 am – 5 pm · Sat 9 am – 12 pm · Sun closed');
    const lunch = WEEK();
    lunch[0] = [span('08:00', '12:00'), span('13:00', '17:00')];
    expect(sayHours(lunch).split(' · ')[0]).toBe('Mon 8 am – 12 pm, 1 – 5 pm');
  });
});

describe('what a service needs', () => {
  it('a name and a length in whole minutes, up to a day', () => {
    expect(checkService({ name: 'Lawn mowing', minutes: 60 })).toEqual({});
    expect(checkService({ name: '  ', minutes: 60 })).toEqual({ name: 'Give it a name' });
    expect(checkService({ name: 'Quote', minutes: 0 })).toEqual({ minutes: 'Say how long it takes' });
    expect(checkService({ name: 'Quote', minutes: Number.NaN })).toEqual({ minutes: 'Say how long it takes' });
    expect(checkService({ name: 'Quote', minutes: 1441 }).minutes).toMatch(/up to a day/);
    expect(checkService({ name: '', minutes: 0 })).toEqual({ name: 'Give it a name', minutes: 'Say how long it takes' });
  });

  it('the settings list each problem in page order, by day and by service', () => {
    const s = settings({
      hours: [[span('17:00', '09:00')], [span('09:00', '17:00')], [], [], [], [], []],
      services: [{ name: 'Lawn mowing', minutes: 60 }, { name: '', minutes: 30 }, { name: 'Hedge', minutes: 0 }],
      slotMinutes: 2,
    });
    const p = checkSettings(s);
    expect(p.hours[0][0]).toBe('Closes at 9 am, before it opens at 5 pm');
    expect(p.services[1]).toEqual({ name: 'Give it a name' });
    expect(p.services[2]).toEqual({ minutes: 'Say how long it takes' });
    expect(p.slotMinutes).toBeDefined();
    expect(p.list).toEqual([
      'Monday: closes at 9 am, before it opens at 5 pm',
      'Service 2: give it a name',
      '“Hedge”: say how long it takes',
      'Times offered: every 5 to 240 minutes',
    ]);
  });

  it('a well-made week has nothing to fix; a time left empty does', () => {
    expect(checkSettings(settings()).list).toEqual([]);
    const p = checkSettings(settings({ hours: [[span('', '17:00')], [], [], [], [], [], []] }));
    expect(p.list).toEqual(['Monday: needs a time to open and to close']);
  });

  it('times as the Agent may have written them are read, and count as unchanged', () => {
    expect(toHHMM('9:00')).toBe('09:00');
    expect(toHHMM('10am')).toBe('10:00');
    expect(toHHMM('3:30 pm')).toBe('15:30');
    expect(toHHMM('12 am')).toBe('00:00');
    expect(toHHMM('13 pm')).toBeNull();
    expect(normalHours([[span('9am', '5 pm')]])[0]).toEqual([span('09:00', '17:00')]);
    expect(sameSettings(settings({ hours: [[span('8am', '5pm')], ...WEEK().slice(1)] }), settings())).toBe(true);
  });
});

describe('the free-time picker', () => {
  const times = ['09:00', '09:30', '11:00', '13:00', '13:30', '17:30'];

  it('groups the free times as morning, afternoon and evening', () => {
    expect(groupTimes(times)).toEqual([
      { part: 'morning', label: 'Morning', times: ['09:00', '09:30', '11:00'] },
      { part: 'afternoon', label: 'Afternoon', times: ['13:00', '13:30'] },
      { part: 'evening', label: 'Evening', times: ['17:30'] },
    ]);
    expect(groupTimes([])).toEqual([]);
  });

  it('starts on the time clicked when it is free', () => {
    expect(pickTime(times, '11:00')).toBe('11:00');
  });

  it('otherwise on the nearest free time, the later one on a tie', () => {
    expect(pickTime(times, '10:00')).toBe('09:30');
    expect(pickTime(times, '12:00')).toBe('13:00');
    expect(pickTime(['09:00', '11:00'], '10:00')).toBe('11:00');
    expect(pickTime(times, '20:00')).toBe('17:30');
  });

  it('with nothing asked for, the first free time; with nothing free, none', () => {
    expect(pickTime(times)).toBe('09:00');
    expect(pickTime(times, null)).toBe('09:00');
    expect(pickTime([], '10:00')).toBeNull();
  });
});

describe('the week grid', () => {
  const at = (id: string, time: string, minutes: number) => ({ id, start: `2026-09-29T${time}`, minutes });

  it('stacks overlapping appointments that start apart, and leaves the rest full width', () => {
    const lanes = layoutDay([at('a', '09:00', 60), at('b', '09:30', 60), at('c', '10:00', 30), at('d', '11:00', 30)]);
    expect(lanes.get('a')).toEqual({ lane: 0, lanes: 2, cascade: true });
    expect(lanes.get('b')).toEqual({ lane: 1, lanes: 2, cascade: true });
    // c starts as a ends: it takes a's lane.
    expect(lanes.get('c')).toEqual({ lane: 0, lanes: 2, cascade: true });
    expect(lanes.get('d')).toEqual({ lane: 0, lanes: 1, cascade: false });
  });

  it('puts appointments that start together side by side', () => {
    const lanes = layoutDay([at('a', '09:00', 60), at('b', '09:10', 30), at('c', '09:45', 30)]);
    expect([...lanes.values()].every((s) => !s.cascade && s.lanes === 2)).toBe(true);
    // c starts after b ends: it takes b's lane, beside a.
    expect([lanes.get('a')!.lane, lanes.get('b')!.lane, lanes.get('c')!.lane]).toEqual([0, 1, 1]);
  });

  it('spans the opening hours and the appointments, with an hour either side', () => {
    expect(gridHours(settings(), [])).toEqual([7, 18]);
    expect(gridHours(settings(), [{ start: '2026-09-29T18:30', minutes: 60 }])).toEqual([7, 21]);
    expect(gridHours(null, [])).toEqual([8, 18]);
  });

  it('snaps a click to the step before it', () => {
    expect(snapDown(10 * 60 + 44, 30)).toBe(10 * 60 + 30);
    expect(snapDown(10 * 60 + 44, 15)).toBe(10 * 60 + 30);
  });

  it('says times, lengths and weeks plainly', () => {
    expect(sayRange('09:00', '10:30')).toBe('9 – 10:30 am');
    expect(sayRange('11:00', '12:00')).toBe('11 am – 12 pm');
    expect(sayMinutes(90)).toBe('1 hr 30 min');
    expect(sayMinutes(45)).toBe('45 min');
    expect(sayWeek(new Date(2026, 8, 28))).toBe('28 Sep – 4 Oct 2026');
    expect(sayWeek(new Date(2026, 9, 5))).toBe('5 – 11 Oct 2026');
  });
});

describe('who answers', () => {
  it('the receptionist is Aokie until it is given a name', () => {
    expect(receptionistName(settings())).toBe('Aokie');
    expect(receptionistName(settings({ receptionist: '   ' }))).toBe('Aokie');
    expect(receptionistName(settings({ receptionist: ' Sam ' }))).toBe('Sam');
  });

  it('the greeting says the business and the receptionist, or only the receptionist', () => {
    expect(sayGreeting(settings())).toBe('Thanks for calling Green Lawns, this is Aokie. How can I help?');
    expect(sayGreeting(settings({ business: ' ', receptionist: 'Sam' }))).toBe('Thanks for calling, this is Sam. How can I help?');
  });

  it('a name of more than 40 characters is a problem, listed first as the card is first', () => {
    expect(checkSettings(settings({ receptionist: 'A'.repeat(40) })).list).toEqual([]);
    // Characters, as the desktop counts them: not UTF-16 units, and not the spaces around it.
    expect(checkSettings(settings({ receptionist: ` ${'😀'.repeat(40)} ` })).receptionist).toBeUndefined();
    const p = checkSettings(settings({ receptionist: 'A'.repeat(41), slotMinutes: 2 }));
    expect(p.receptionist).toBe('At most 40 characters');
    expect(p.list).toEqual(['The receptionist’s name: at most 40 characters', 'Times offered: every 5 to 240 minutes']);
  });

  it('a desktop from before the name was kept reads as no name', () => {
    const older: Partial<CalendarSettings> = settings();
    delete older.receptionist;
    expect(normalSettings(older as CalendarSettings).receptionist).toBe('');
  });
});
