// What an answer that comes late is about: the ring it was asked for. (The final review's proof of F5-2, and the late failure that came after it.)
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ActiveRing } from './api';

const api = vi.hoisted(() => ({ active: vi.fn(), respond: vi.fn(), dismiss: vi.fn(), openSetup: vi.fn() }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, ring: { ...real.ring, active: (...a: unknown[]) => api.active(...a), respond: (...a: unknown[]) => api.respond(...a), dismissNotice: (...a: unknown[]) => api.dismiss(...a) } };
});
vi.mock('./useSetupState', () => ({ openSetup: (...a: unknown[]) => api.openSetup(...a) }));

import RingDialog, { POLL_MS } from './RingDialog';

const NOW = Date.parse('2026-09-30T05:00:00Z');
const ringing = (r: Partial<ActiveRing> = {}): ActiveRing => ({
  id: 'assist_1', callId: 'call_1', callerName: 'Alex', callerNumber: '+61491570006', said: ['Can I speak to the owner?'],
  startedAt: NOW, expiresAt: NOW + 30_000, now: Date.now() + 5_000, devices: ['Pixel 6'], stopping: false, taken: false, note: '', ...r,
});
const serve = (...rings: Array<Partial<ActiveRing>>) => api.active.mockImplementation(async () => ({ rings: rings.map((r) => ringing(r)), notices: [] }));

let host: HTMLDivElement;
let root: Root;
const settle = async () => { for (let i = 0; i < 4; i++) await act(async () => {}); };
const decline = () => [...host.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Decline and take a message')!;
const later = (ms: number) => act(async () => { vi.advanceTimersByTime(ms); });

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'], now: NOW });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  api.active.mockResolvedValue({ rings: [], notices: [] });
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.useRealTimers();
});

describe('the ring dialog and an answer that comes late', () => {
  it('shows a decline that failed on the ring that is shown, and not on the next caller\'s ring', async () => {
    serve({ id: 'assist_A', callId: 'call_A', callerName: 'Alex' });
    api.respond.mockRejectedValue(new Error('404: that ring is over'));
    await act(async () => root.render(<RingDialog />));
    await settle();
    await act(async () => decline().click());
    await settle();
    expect(host.querySelector('[role=alert]')?.textContent).toContain('that ring is over');
    // Ring A is gone; a different caller rings a while later.
    serve({ id: 'assist_B', callId: 'call_B', callerName: 'Sam' });
    await later(POLL_MS + 10);
    await settle();
    expect(host.textContent).toContain('Sam');
    expect(host.querySelector('[role=alert]')).toBeNull();
  });

  it('does not carry the error of a decline that fails after another caller\'s ring is shown', async () => {
    serve({ id: 'assist_A', callId: 'call_A', callerName: 'Alex' });
    let fail: ((e: Error) => void) | undefined;
    api.respond.mockImplementation(() => new Promise((_, reject) => { fail = reject; })); // the desktop is slow to answer
    await act(async () => root.render(<RingDialog />));
    await settle();
    await act(async () => decline().click());
    await settle();
    // While it is on its way, ring A ends and another caller rings.
    serve({ id: 'assist_B', callId: 'call_B', callerName: 'Sam' });
    await later(POLL_MS + 10);
    await settle();
    expect(host.textContent).toContain('Sam');
    // ...then the answer for A comes, and it is an error.
    await act(async () => fail!(new Error('404: that ring is over')));
    await settle();
    expect(host.textContent).toContain('Sam');
    expect(host.querySelector('[role=alert]'), 'the error was about ring A').toBeNull();
  });

  it('does not hold another caller\'s ring while a decline for the first is on its way', async () => {
    serve({ id: 'assist_A', callId: 'call_A', callerName: 'Alex' });
    api.respond.mockImplementation(() => new Promise(() => {})); // never answered
    await act(async () => root.render(<RingDialog />));
    await settle();
    await act(async () => decline().click());
    await settle();
    expect(decline().disabled, 'ring A waits for its own answer').toBe(true);
    serve({ id: 'assist_B', callId: 'call_B', callerName: 'Sam' });
    await later(POLL_MS + 10);
    await settle();
    expect(host.textContent).toContain('Sam');
    expect(decline().disabled, 'ring B can be declined').toBe(false);
  });
});
