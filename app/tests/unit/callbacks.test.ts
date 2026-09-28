import { describe, expect, it } from 'vitest';
import { AU_PATTERN, Callbacks, callsBack, retryAfter, type Callback, type Screening } from '../../src/callbacks';
import type { Desktop, DesktopEvent } from '../../src/desktop/bridge';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import { callStartNote, isCallStart } from '../../src/sessions';

const ended = (data: Record<string, unknown>): DesktopEvent => ({ seq: 1, name: 'aokie.call.ended', source: 'aokie', correlationId: '', idempotencyKey: '', occurredAt: '', data });

function setup(screening: Screening | null = { acceptPattern: '', blockedNumbers: '', rejectPrivate: false }, refuse = '') {
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
  const callbacks = new Callbacks(project as never, () => settings, () => desktop as unknown as Desktop, () => free, async () => screening);
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
  });
});
