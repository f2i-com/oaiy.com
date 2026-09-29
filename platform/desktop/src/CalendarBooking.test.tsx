// The free-time picker and the booking form: a service's length fills in, a
// day's free times come from the desktop (its hours, its booking rules, what
// is booked), the time clicked is kept when free and the nearest free one is
// picked when not, and a booking sends what was chosen.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CalendarSettings } from './api';

const { free } = vi.hoisted(() => ({ free: vi.fn() }));
vi.mock('./api', () => ({ calendar: { free: (...a: unknown[]) => free(...a) } }));

import { BookingForm, SlotPicker } from './CalendarBooking';
import { addDays, mondayOf, ymd } from './calendarModel';

// Next week, so no day of it is past whatever day the tests run.
const MONDAY = ymd(addDays(mondayOf(new Date()), 7));
const day = (n: number) => ymd(addDays(new Date(`${MONDAY}T00:00`), n));

const SETTINGS: CalendarSettings = {
  business: 'Green Lawns',
  hours: [[{ open: '08:00', close: '17:00' }], [{ open: '08:00', close: '17:00' }], [{ open: '08:00', close: '17:00' }], [], [{ open: '08:00', close: '17:00' }], [], []],
  services: [
    { id: 'mow', name: 'Lawn mowing', minutes: 60, price: 'from $60' },
    { id: 'hedge', name: 'Hedge trimming', minutes: 90 },
  ],
  slotMinutes: 30,
  noticeMinutes: 60,
  horizonDays: 30,
  textConfirmations: true,
};

/** The desktop's answer: Monday and Tuesday have times, Wednesday is full, Thursday closed. */
const week = (times: Record<number, string[]>) => ({
  minutes: 60,
  days: Array.from({ length: 7 }, (_, i) => ({ date: day(i), times: times[i] ?? [] })),
});

let host: HTMLDivElement;
let root: Root;
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
const chips = () => [...host.querySelectorAll<HTMLButtonElement>('.cal-times-free .cal-chip')];
const onChip = () => host.querySelector('.cal-times-free .cal-chip.is-on')?.textContent;
const dayChip = (n: number) => host.querySelectorAll<HTMLButtonElement>('.cal-daychip')[n];

beforeEach(() => {
  vi.clearAllMocks();
  free.mockResolvedValue(week({ 0: ['09:00', '09:30', '10:30', '13:00', '17:30'], 1: ['08:00', '14:00'] }));
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

function Picker({ date, wanted, onPick }: { date: string; wanted?: string | null; onPick?: (d: string, t: string | null) => void }) {
  const [d, setD] = React.useState(date);
  const [t, setT] = React.useState<string | null>(wanted ?? null);
  return (
    <SlotPicker
      settings={SETTINGS}
      service="Lawn mowing"
      date={d}
      time={t}
      wanted={wanted}
      onPick={(nd, nt) => {
        setD(nd);
        setT(nt);
        onPick?.(nd, nt);
      }}
    />
  );
}

describe('the free-time picker', () => {
  it('asks for the week’s free times for the service, and shows each day’s', async () => {
    await act(async () => root.render(<Picker date={day(0)} wanted="09:30" />));
    await settle();
    expect(free).toHaveBeenCalledWith(MONDAY, 7, 'Lawn mowing', undefined);
    expect(dayChip(0).textContent).toContain('5 free');
    expect(dayChip(1).textContent).toContain('2 free');
    expect(dayChip(2).textContent).toContain('Full');
    expect(dayChip(3).textContent).toContain('Closed');
    // Grouped by part of the day.
    const groups = [...host.querySelectorAll('.cal-time-group > span')].map((s) => s.textContent);
    expect(groups).toEqual(['Morning', 'Afternoon', 'Evening']);
    expect(chips().map((c) => c.textContent)).toEqual(['9 am', '9:30 am', '10:30 am', '1 pm', '5:30 pm']);
  });

  it('keeps the time clicked when it is free', async () => {
    const onPick = vi.fn();
    await act(async () => root.render(<Picker date={day(0)} wanted="09:30" onPick={onPick} />));
    await settle();
    expect(onChip()).toBe('9:30 am');
    expect(onPick).not.toHaveBeenCalled();
  });

  it('picks the nearest free time when the one clicked is taken, and says so', async () => {
    const onPick = vi.fn();
    await act(async () => root.render(<Picker date={day(0)} wanted="11:00" onPick={onPick} />));
    await settle();
    expect(onPick).toHaveBeenLastCalledWith(day(0), '10:30');
    expect(onChip()).toBe('10:30 am');
    expect(host.textContent).toContain('11 am is not free for this; the nearest free time is picked.');
  });

  it('moves to another day, keeping to the same time of day where it can', async () => {
    const onPick = vi.fn();
    await act(async () => root.render(<Picker date={day(0)} wanted="13:00" onPick={onPick} />));
    await settle();
    await act(async () => dayChip(1).click());
    await settle();
    expect(onPick).toHaveBeenLastCalledWith(day(1), '14:00');
    expect(onChip()).toBe('2 pm');
  });

  it('says when a day has nothing free, and offers another time', async () => {
    const onPick = vi.fn();
    await act(async () => root.render(<Picker date={day(2)} onPick={onPick} />));
    await settle();
    expect(host.textContent).toContain('Nothing free that day for this.');
    expect(chips()).toHaveLength(0);
    expect(onPick).not.toHaveBeenCalledWith(day(2), expect.any(String));
    await act(async () => [...host.querySelectorAll<HTMLButtonElement>('.btn-link')].find((b) => b.textContent === 'Another time…')!.click());
    expect(host.querySelector('.cal-other-time input[type="time"]')).not.toBeNull();
  });
});

describe('a new booking', () => {
  it('fills in the service’s length, asks again for its times, and books what was chosen', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    await act(async () =>
      root.render(<BookingForm settings={SETTINGS} initial={{ date: day(0), time: '09:00' }} wanted="09:00" submitLabel="Book it" onSubmit={onSubmit} onCancel={vi.fn()} />),
    );
    await settle();
    expect(host.querySelector('.cal-takes')?.textContent).toContain('Takes 1 hr');
    expect(host.querySelector('.cal-when-sum')?.textContent).toContain('9 – 10 am');

    // Hedge trimming takes an hour and a half: its own free times are asked for.
    free.mockResolvedValue(week({ 0: ['08:00', '13:00'] }));
    const select = host.querySelector<HTMLSelectElement>('select')!;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')!.set!;
      set.call(select, 'Hedge trimming');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await settle();
    expect(free).toHaveBeenLastCalledWith(MONDAY, 7, 'Hedge trimming', undefined);
    expect(host.querySelector('.cal-takes')?.textContent).toContain('Takes 1 hr 30 min');
    // 9 am is not free for it any more: the nearest is.
    expect(onChip()).toBe('8 am');

    const name = host.querySelector<HTMLInputElement>('input[placeholder="Who it is for"]')!;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
      set.call(name, 'Lanes');
      name.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => host.querySelector<HTMLButtonElement>('button[type="submit"]')!.click());
    expect(onSubmit).toHaveBeenCalledWith({ service: 'Hedge trimming', date: day(0), time: '08:00', name: 'Lanes', phone: '', notes: '' }, false);
  });

  it('cannot be booked with no time chosen', async () => {
    free.mockResolvedValue(week({}));
    await act(async () =>
      root.render(<BookingForm settings={SETTINGS} initial={{ date: day(2) }} submitLabel="Book it" onSubmit={vi.fn()} onCancel={vi.fn()} />),
    );
    await settle();
    expect(host.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled).toBe(true);
    expect(host.querySelector('.cal-when-sum')?.textContent).toBe('Pick a time');
  });
});
