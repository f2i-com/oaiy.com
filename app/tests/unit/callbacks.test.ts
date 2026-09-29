import { describe, expect, it } from 'vitest';
import { AU_PATTERN, Callbacks, callsBack, isBlocked, retryAfter, type Callback, type Screening } from '../../src/callbacks';
import type { Desktop, DesktopEvent } from '../../src/desktop/bridge';
import { setLocalCountry } from '../../src/phoneNumbers';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import { callStartNote, isCallStart } from '../../src/sessions';

const ended = (data: Record<string, unknown>): DesktopEvent => ({ seq: 1, name: 'aokie.call.ended', source: 'aokie', correlationId: '', idempotencyKey: '', occurredAt: '', data });

function setup(screening: Screening | null = { acceptPattern: '', blockedNumbers: '', rejectPrivate: false }, refuse = '', route: boolean | null = true) {
  let saved: Callback[] = [];
  const project = { loadCallbacks: async () => saved, saveCallbacks: async (list: Callback[]) => void (saved = list) };
  const dials: Array<Record<string, unknown>> = [];
  const desktop = {
    command: async (_c: string, command: string, payload: Record<string, unknown>) => {
      if (refuse) throw new Error(refuse);
      dials.push({ command, ...payload });
      return { accepted: true };
    },
  };
  const settings: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, callBack: true, callBackFilter: 'answered' };
  let free = true;
  const callbacks = new Callbacks(project as never, () => settings, () => desktop as unknown as Desktop, () => free, async () => screening, () => {}, async () => route);
  return { callbacks, dials, settings, setFree: (f: boolean) => (free = f), saved: () => saved };
}

describe('missed calls, called back', () => {
  it('a missed call is rung back once the receptionist is free, after giving them a minute to ring again', async () => {
    const { callbacks, dials, setFree } = setup();
    const t0 = Date.now();
    await callbacks.event(ended({ from: '0491570006', outcome: 'missed', direction: 'inbound' }));
    await callbacks.tick(t0 + 10_000);
    expect(dials).toHaveLength(0);
    setFree(false);
    await callbacks.tick(t0 + 120_000);
    expect(dials).toHaveLength(0);
    setFree(true);
    await callbacks.tick(t0 + 120_000);
    expect(dials).toEqual([expect.objectContaining({ command: 'call.dial', number: '0491570006', openingLine: expect.stringContaining('returning your call') })]);
    expect(callbacks.calling('+61491570006')).toBeTruthy();
    // Answered: done. Only one call back rings at a time, and none after.
    await callbacks.event(ended({ from: '0491570006', outcome: 'completed', direction: 'outbound' }));
    await callbacks.tick(t0 + 10 * 60_000);
    expect(dials).toHaveLength(1);
    expect(callbacks.list[0]).toMatchObject({ state: 'done', note: 'Called back.' });
  });

  it('not answered, it is tried once more later, then left', async () => {
    const { callbacks, dials } = setup();
    const t0 = Date.now();
    await callbacks.missed('0400000001', t0);
    await callbacks.tick(t0 + 120_000);
    await callbacks.event(ended({ from: '0400000001', outcome: 'no_answer', direction: 'outbound' }));
    expect(callbacks.list[0]).toMatchObject({ state: 'waiting', tries: 1 });
    await callbacks.tick(Date.now() + 21 * 60_000);
    expect(dials).toHaveLength(2);
    await callbacks.event(ended({ from: '0400000001', outcome: 'no_answer', direction: 'outbound' }));
    expect(callbacks.list[0]).toMatchObject({ state: 'dropped' });
  });

  it('a caller who rings again and is answered, or texts, is not called back', async () => {
    const { callbacks, dials } = setup();
    const t0 = Date.now();
    await callbacks.missed('0400000002', t0);
    await callbacks.event(ended({ from: '+61400000002', outcome: 'completed', direction: 'inbound' }));
    await callbacks.missed('0400000003', t0);
    await callbacks.event({ ...ended({ from: '0400000003', body: 'hi' }), name: 'aokie.sms.received' });
    await callbacks.tick(t0 + 120_000);
    expect(dials).toHaveLength(0);
    expect(callbacks.list.map((c) => c.state)).toEqual(['done', 'done']);
  });

  it('only the numbers the filter allows are rung: never a blocked one', async () => {
    const screening: Screening = { acceptPattern: AU_PATTERN, blockedNumbers: '0400 000 004, +61 400 000 005', rejectPrivate: true };
    expect(callsBack('0491570006', 'answered', screening)).toBe(true);
    expect(callsBack('+61491570006', 'au', screening)).toBe(true);
    expect(callsBack('+14155550100', 'answered', screening)).toBe(false);
    expect(callsBack('+14155550100', 'any', screening)).toBe(true);
    expect(callsBack('+61400000004', 'any', screening)).toBe(false);
    expect(callsBack('0400000005', 'au', screening)).toBe(false);
    expect(callsBack('', 'any', screening)).toBe(false);
    const { callbacks, dials } = setup(screening);
    const t0 = Date.now();
    await callbacks.missed('+14155550100', t0);
    await callbacks.tick(t0 + 120_000);
    expect(dials).toHaveLength(0);
    expect(callbacks.list[0]).toMatchObject({ state: 'dropped', note: 'Not one of the numbers you call back.' });
  });

  it('a caller who hung up waiting in the queue is rung back with a sorry; one who hung up on hold is not', async () => {
    const { callbacks, dials } = setup();
    const t0 = Date.now();
    await callbacks.event(ended({ from: '0400000011', outcome: 'abandoned_in_queue', direction: 'inbound' }));
    await callbacks.event(ended({ from: '0400000012', outcome: 'abandoned_on_hold', direction: 'inbound' }));
    await callbacks.tick(t0 + 120_000);
    expect(dials).toEqual([expect.objectContaining({ number: '0400000011', openingLine: expect.stringContaining('Sorry you were kept waiting') })]);
    expect(callbacks.list.map((c) => c.number)).toEqual(['0400000011']);
  });

  it('a blocked number matches as Aokie matches it: its last nine digits, six or more', () => {
    const screening: Screening = { acceptPattern: '', blockedNumbers: '123456, 12345', rejectPrivate: false };
    expect(callsBack('123456', 'any', screening)).toBe(false);
    expect(callsBack('9912345', 'any', screening)).toBe(true);
  });

  it('a blocked person is blocked however either number is written: as Aokie matches, or as the same person (never less)', () => {
    setLocalCountry('AU');
    const blocked = '0491 570 006\n+44 20 7946 0958';
    for (const number of ['+61491570006', '0491570006', '0011 61 491 570 006', '61491570006', '+442079460958', '0011 44 20 7946 0958', '00442079460958']) {
      expect(isBlocked(number, blocked), number).toBe(true);
      expect(callsBack(number, 'any', { acceptPattern: '', blockedNumbers: blocked, rejectPrivate: false }), number).toBe(false);
    }
    expect(isBlocked('+61491570157', blocked)).toBe(false);
    // Aokie's own rule still blocks what it blocked (a number the country's rules cannot read, by its last nine digits).
    expect(isBlocked('07700 900123', '+44 7700 900123')).toBe(true);
    // A number whose last nine digits differ from its other form (a New Zealand landline) is blocked too.
    setLocalCountry('NZ');
    expect(isBlocked('09 123 4567', '+64 9 123 4567')).toBe(true);
    setLocalCountry('AU');
    // A hidden caller is never rung back, however the phone says it.
    expect(callsBack('Private', 'any', null)).toBe(false);
  });

  it('the filters read a number as before, and an Australian one however it is written', () => {
    setLocalCountry('AU');
    const answered: Screening = { acceptPattern: AU_PATTERN, blockedNumbers: '', rejectPrivate: false };
    for (const [number, au, asAokie] of [
      ['0491570006', true, true],
      ['+61491570006', true, true],
      ['61491570006', true, true],
      ['(02) 9876 5432', true, true],
      // Written with Australia's international prefix: Australian, though Aokie's pattern does not read it so.
      ['0011 61 491 570 006', true, false],
      ['+14155550100', false, false],
      ['+442079460958', false, false],
    ] as const) {
      expect(callsBack(number, 'au', null), number).toBe(au);
      expect(callsBack(number, 'answered', answered), number).toBe(asAokie);
      expect(callsBack(number, 'any', answered), number).toBe(true);
    }
  });

  it('a missed call is the same person as their text or their next call, however each number is written', async () => {
    setLocalCountry('AU');
    const { callbacks } = setup();
    const t0 = Date.now();
    await callbacks.missed('0491570006', t0);
    await callbacks.missed('+61 491 570 006', t0 + 1000);
    expect(callbacks.open).toHaveLength(1);
    await callbacks.event({ ...ended({ from: '+61491570006', body: 'Sorry I missed you' }), name: 'aokie.sms.received' });
    expect(callbacks.list[0]).toMatchObject({ state: 'done', note: 'They texted: the text conversation has them.' });
  });

  it("calls on Aokie's own voice: its follow-ups ring back, not OAIY; the route unknown, it waits", async () => {
    const t0 = Date.now();
    const own = setup(undefined, '', false);
    await own.callbacks.missed('0400000021', t0);
    await own.callbacks.tick(t0 + 120_000);
    expect(own.dials).toHaveLength(0);
    expect(own.callbacks.list[0]).toMatchObject({ state: 'dropped', note: expect.stringContaining("Aokie's own voice") });
    const unknown = setup(undefined, '', null);
    await unknown.callbacks.missed('0400000022', t0);
    await unknown.callbacks.tick(t0 + 120_000);
    expect(unknown.dials).toHaveLength(0);
    expect(unknown.callbacks.list[0]).toMatchObject({ state: 'waiting' });
  });

  it('a refusal from the phone (quiet hours, the daily cap) waits and tries again', async () => {
    const t0 = new Date(2026, 8, 28, 22, 0).getTime();
    expect(retryAfter('quiet hours: automated calls are not placed between 21:00 and 8:00', t0)).toBe(t0 + 30 * 60_000);
    expect(new Date(retryAfter('daily dial cap reached (20/20 today)', t0)).getDate()).toBe(29);
    const { callbacks, dials } = setup(undefined, 'quiet hours: automated calls are not placed between 21:00 and 8:00');
    await callbacks.missed('0400000006', t0);
    await callbacks.tick(t0 + 120_000);
    expect(dials).toHaveLength(0);
    expect(callbacks.list[0]).toMatchObject({ state: 'waiting', tries: 0, nextAt: t0 + 120_000 + 30 * 60_000 });
  });

  it("calling back off: nothing rings, the missed call is kept for when it is on", async () => {
    const { callbacks, dials, settings } = setup();
    settings.callBack = false;
    const t0 = Date.now();
    await callbacks.missed('0400000007', t0);
    await callbacks.tick(t0 + 120_000);
    expect(dials).toHaveLength(0);
    settings.callBack = true;
    await callbacks.tick(t0 + 120_000);
    expect(dials).toHaveLength(1);
  });

  it("a call back's agent is told it rang them back, and why", () => {
    const note = callStartNote('Lance (0491570006)', "Hi, it's the receptionist, returning your call.", 'Name: Lance', new Date(2026, 8, 29, 10, 5), new Date(2026, 8, 29, 9, 40).getTime());
    expect(note).toMatch(/^\[OAIY\] 📞 You rang Lance \(0491570006\) back, returning their missed call from .+; they answered .+\. You opened with: "Hi, it's the receptionist, returning your call\."/);
    expect(isCallStart({ role: 'user', text: note, automatic: true })).toBe(true);
    // A call the phone placed for another reason (a flow's call.dial): the agent is told it rang, and why.
    const placed = callStartNote('Sam (0400000009)', 'Hi Sam, it is the receptionist.', '', new Date(2026, 8, 29, 10, 5), undefined, { purpose: 'Remind them of tomorrow at 9.' });
    expect(placed).toMatch(/^\[OAIY\] 📞 You rang Sam \(0400000009\); they answered .+\. Why you rang: Remind them of tomorrow at 9\. You opened with: "Hi Sam, it is the receptionist\."/);
    expect(isCallStart({ role: 'user', text: placed, automatic: true })).toBe(true);
  });
});
