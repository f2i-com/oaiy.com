import { describe, expect, it } from 'vitest';
import { calendarTools, sayTime, textCalendarTools } from '../../src/desktop/calendarTools';
import type { Desktop } from '../../src/desktop/bridge';

function fakeDesktop() {
  const created: Array<Record<string, unknown>> = [];
  const changed: Array<[string, Record<string, unknown>]> = [];
  const desktop = {
    calendarFree: async (from: string, days: number, service?: string) => ({
      minutes: service ? 60 : 30,
      service: service ? 'Lawn mowing' : null,
      days: [
        { date: '2026-09-29', times: ['09:00', '09:30', '14:00'] },
        { date: '2026-09-30', times: [] },
      ],
      from,
      days_asked: days,
    }),
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
  return { desktop, created, changed };
}

const signal = new AbortController().signal;

describe("the calendar as the agent's tools", () => {
  it('says times as people do', () => {
    expect([sayTime('09:00'), sayTime('09:30'), sayTime('12:00'), sayTime('14:15'), sayTime('00:00')]).toEqual(['9 am', '9:30 am', '12 pm', '2:15 pm', '12 am']);
  });

  it("a text thread sees the free times, and asks for an appointment for the person texting (a request, never confirmed)", async () => {
    const { desktop, created } = fakeDesktop();
    const [free, request] = textCalendarTools(() => desktop, '+61491570006', () => 'Lance');
    expect(free.spec.name).toBe('calendar_free_times');
    const times = await free.run({ service: 'Lawn mowing', days: 3 }, signal);
    expect(times).toBe('Free times for Lawn mowing, 60 min:\nTue 29 Sep (2026-09-29): 9 am, 9:30 am, 2 pm');
    expect(request.spec.name).toBe('request_appointment');
    expect(request.spec.description).toContain('never say it is booked or confirmed');
    const out = await request.run({ service: 'Lawn mowing', date: '2026-09-29', time: '09:30' }, signal);
    expect(created[0]).toMatchObject({ service: 'Lawn mowing', date: '2026-09-29', time: '09:30', name: 'Lance', phone: '+61491570006', status: 'requested', source: 'text' });
    expect(out).toContain('Staff will confirm it');
  });

  it("the main agent lists, books and changes appointments", async () => {
    const { desktop, created, changed } = fakeDesktop();
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
