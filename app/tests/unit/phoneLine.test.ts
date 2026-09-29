import { describe, expect, it } from 'vitest';
import type { DesktopEvent } from '../../src/desktop/bridge';
import { CALM_MS, PhoneLine } from '../../src/phoneLine';
import { Callbacks, type Callback } from '../../src/callbacks';
import { DEFAULT_MESSAGE_SETTINGS, type MessageSettings } from '../../src/settings';
import type { Desktop } from '../../src/desktop/bridge';

const ev = (name: string, data: Record<string, unknown> = {}, correlationId = ''): DesktopEvent => ({ seq: 1, name, source: 'aokie', correlationId, idempotencyKey: '', occurredAt: '', data });

describe("the phone's one line", () => {
  it('is busy while a dial of ours rings, until its end, then calm for a minute', () => {
    const line = new PhoneLine();
    const t = 1_000_000;
    expect(line.idle(t, false)).toBe(true);
    line.take('outreach', t);
    expect(line.idle(t, false)).toBe(false);
    line.bind('call_1', 'op_1');
    line.event(ev('aokie.call.outbound.dialing', { callId: 'call_1' }, 'call_1'), t + 1000);
    expect(line.idle(t + 2000, false)).toBe(false);
    line.event(ev('aokie.call.ended', { callId: 'call_1', outcome: 'completed', direction: 'outbound' }), t + 60_000);
    expect(line.idle(t + 60_001, false)).toBe(false);
    expect(line.idle(t + 60_000 + CALM_MS, false)).toBe(true);
  });

  it('a call coming in, or one waiting, keeps it busy until it ends; so does a live call here', () => {
    const line = new PhoneLine();
    const t = 5_000_000;
    line.event(ev('aokie.call.incoming', { from: '+61400000001' }, 'call_in'), t);
    expect(line.idle(t + CALM_MS * 5, false)).toBe(false);
    line.event(ev('aokie.call.waiting', { callId: 'call_in', waitingCallId: 'call_w' }), t + 1000);
    line.event(ev('aokie.call.ended', { callId: 'call_in' }), t + 2000);
    expect(line.idle(t + 2000 + CALM_MS, false)).toBe(false);
    line.event(ev('aokie.call.ended', { callId: 'call_w' }), t + 3000);
    expect(line.idle(t + 3000 + CALM_MS, false)).toBe(true);
    expect(line.idle(t + 3000 + CALM_MS, true)).toBe(false);
  });

  it("another's dial (a call back's) is not ours, and blocks until it ends", () => {
    const line = new PhoneLine();
    line.event(ev('aokie.call.outbound.dialing', { callId: 'call_cb' }, 'call_cb'), 0);
    expect(line.idle(CALM_MS * 3, false)).toBe(false);
    line.event(ev('aokie.call.ended', { callId: 'call_cb' }), 10);
    expect(line.idle(10 + CALM_MS, false)).toBe(true);
  });

  it('a dial the radio dropped (matched by its operation id) frees the line; a dial never heard of again is let go after ten minutes', () => {
    const line = new PhoneLine();
    line.take('outreach', 0);
    line.bind('call_x', 'op_x');
    line.event(ev('aokie.hardware.error', { code: 'control_failed', action: 'call.dial', operationId: 'op_other' }), 100);
    expect(line.placed).not.toBeNull();
    line.event(ev('aokie.hardware.error', { code: 'control_failed', action: 'call.dial', operationId: 'op_x' }), 200);
    expect(line.placed).toBeNull();
    line.take('outreach', 1_000_000);
    expect(line.idle(1_000_000 + 9 * 60_000, false)).toBe(false);
    expect(line.idle(1_000_000 + 11 * 60_000, false)).toBe(true);
  });
});

describe('call backs and the line', () => {
  function setup(skip: (n: string) => boolean = () => false) {
    let saved: Callback[] = [];
    const project = { loadCallbacks: async () => saved, saveCallbacks: async (list: Callback[]) => void (saved = list) };
    const dials: Array<Record<string, unknown>> = [];
    const desktop = { command: async (_c: string, command: string, payload: Record<string, unknown>) => void dials.push({ command, ...payload }) };
    const settings: MessageSettings = { ...DEFAULT_MESSAGE_SETTINGS, callBack: true, callBackFilter: 'any' };
    const callbacks = new Callbacks(project as never, () => settings, () => desktop as unknown as Desktop, () => true, async () => null, () => {}, async () => true, skip);
    return { callbacks, dials, settings };
  }

  it('says when one is due or ringing (an outreach list waits for it)', async () => {
    const { callbacks, settings } = setup();
    const t0 = Date.now();
    expect(callbacks.due(t0)).toBe(false);
    await callbacks.missed('0400000001', t0);
    expect(callbacks.due(t0 + 10_000)).toBe(false);
    expect(callbacks.due(t0 + 120_000)).toBe(true);
    settings.callBack = false;
    expect(callbacks.due(t0 + 120_000)).toBe(false);
    settings.callBack = true;
    await callbacks.tick(t0 + 120_000);
    expect(callbacks.ringing()).toBe(true);
  });

  it('never rings a number on the do-not-contact list', async () => {
    const { callbacks, dials } = setup((n) => n.endsWith('0002'));
    const t0 = Date.now();
    await callbacks.missed('0400000002', t0);
    expect(callbacks.due(t0 + 120_000)).toBe(false);
    await callbacks.tick(t0 + 120_000);
    expect(dials).toEqual([]);
    expect(callbacks.list[0]).toMatchObject({ state: 'dropped', note: 'They asked not to be called.' });
  });
});
