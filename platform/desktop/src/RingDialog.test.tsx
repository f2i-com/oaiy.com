// The ring dialog: a caller wants to speak to the owner. It shows nothing until this desktop says someone is being rung
// for; then who, what they last said (as plain text), which devices ring and a countdown by the desktop's clock; and
// Accept (a request to the Companion: the ring goes on until the phone says how it came out), Decline and Take a message
// instead (the ring is over here at once). A ring past its time is not shown. Same convention as the other tests:
// react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ActiveRing } from './api';

const api = vi.hoisted(() => ({ active: vi.fn(), respond: vi.fn() }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, ring: { ...real.ring, active: (...a: unknown[]) => api.active(...a), respond: (...a: unknown[]) => api.respond(...a) } };
});

import RingDialog, { POLL_MS, secondsLeft } from './RingDialog';

const NOW = Date.parse('2026-09-30T05:00:00Z');
// The desktop's clock is three hours ahead of this window's: the countdown goes by the desktop's.
const SKEW = 3 * 3600 * 1000;

/** A ring that began at NOW by the desktop's clock, which is `SKEW` ahead of this window's and 5 s in when first asked. */
const ringing = (r: Partial<ActiveRing> = {}): ActiveRing => ({
  id: 'assist_1',
  callId: 'call_1',
  callerName: 'Alex',
  callerNumber: '+61491570006',
  said: ['Can I speak to the owner?'],
  startedAt: NOW + SKEW,
  expiresAt: NOW + SKEW + 30_000,
  now: Date.now() + SKEW + 5_000,
  devices: ['this computer', 'Pixel 6'],
  canAccept: true,
  note: '',
  ...r,
});
/** The desktop answers with these rings, and its clock goes on as time passes. */
const serve = (...rings: Array<Partial<ActiveRing>>) => api.active.mockImplementation(async () => rings.map((r) => ringing(r)));

let host: HTMLDivElement;
let root: Root;
const text = () => host.textContent ?? '';
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
const button = (label: string) => [...host.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label)!;
const click = async (el: Element) => {
  await act(async () => (el as HTMLElement).click());
  await settle();
};
async function mount(props: { on?: boolean } = {}) {
  await act(async () => root.render(<RingDialog {...props} />));
  await settle();
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'], now: NOW });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  api.active.mockResolvedValue([]);
  api.respond.mockResolvedValue({ ok: true, note: '' });
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.useRealTimers();
});

describe('the ring dialog', () => {
  it('shows nothing while nobody is being rung for', async () => {
    await mount();
    expect(host.querySelector('.ring-dialog')).toBeNull();
    expect(api.active).toHaveBeenCalled();
  });

  it('says who is calling, what they last said, who else rings, and how long is left by the desktop’s clock', async () => {
    serve({});
    await mount();
    const dialog = host.querySelector('[role=alertdialog]')!;
    expect(dialog.querySelector('h2')?.textContent).toContain('A caller wants to speak to you');
    expect(dialog.textContent).toContain('Alex');
    expect(dialog.textContent).toContain('0491 570 006');
    expect(dialog.querySelector('.ring-said')?.textContent).toBe('“Can I speak to the owner?”');
    expect(dialog.textContent).toContain('Ringing: this computer, Pixel 6');
    // 30 s of ring, 5 s gone by the desktop's own clock: 25 s left, whatever this window's clock says.
    expect(host.querySelector('[role=timer]')?.textContent).toContain('25s');
    await act(async () => {
      vi.advanceTimersByTime(4_000);
    });
    expect(host.querySelector('[role=timer]')?.textContent).toMatch(/2[01]s/);
    expect(secondsLeft({ expiresAt: 10_000 }, 9_001)).toBe(1);
    expect(secondsLeft({ expiresAt: 10_000 }, 20_000)).toBe(0);
  });

  it('shows what a caller said as plain text, never as markup, and a hidden number as such', async () => {
    serve({ callerName: '', callerNumber: '', said: ['Let me speak to <b>the owner</b>'] });
    await mount();
    expect(host.querySelector('.ring-said')?.textContent).toBe('“Let me speak to <b>the owner</b>”');
    expect(host.querySelector('.ring-said b')).toBeNull();
    expect(text()).toContain('A caller who hid their number');
  });

  it('is asked again every second while the window is visible', async () => {
    await mount();
    const before = api.active.mock.calls.length;
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS * 3 + 10);
    });
    expect(api.active.mock.calls.length).toBeGreaterThanOrEqual(before + 3);
  });

  it('Accept asks for the call to be taken, says what it asked, and the ring goes on until the phone says how it came out', async () => {
    serve({});
    api.respond.mockResolvedValue({ ok: true, note: 'Asked the Companion to take the call.' });
    await mount();
    await click(button('Accept'));
    expect(api.respond).toHaveBeenCalledWith('assist_1', 'accept');
    expect(text()).toContain('Asked the Companion to take the call.');
    expect(host.querySelector('.ring-dialog')).not.toBeNull();
    expect(button('Accept').disabled).toBe(false);
  });

  it('Decline and Take a message instead end the ring here at once, and each is its own request', async () => {
    serve({});
    await mount();
    await click(button('Decline'));
    expect(api.respond).toHaveBeenLastCalledWith('assist_1', 'decline');
    expect(host.querySelector('.ring-dialog')).toBeNull();

    serve({ id: 'assist_2' });
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS + 10);
    });
    await settle();
    await click(button('Take a message instead'));
    expect(api.respond).toHaveBeenLastCalledWith('assist_2', 'message');
    expect(host.querySelector('.ring-dialog')).toBeNull();
  });

  it('does not show a ring that has run out, and shows the next while one runs', async () => {
    serve({ expiresAt: NOW + SKEW + 1_000, now: NOW + SKEW + 1_500 });
    await mount();
    expect(host.querySelector('.ring-dialog')).toBeNull();
    serve({ id: 'old', expiresAt: NOW + SKEW, now: NOW + SKEW + 100 }, { id: 'live' });
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS + 10);
    });
    await settle();
    await click(button('Decline'));
    expect(api.respond).toHaveBeenLastCalledWith('live', 'decline');
  });

  it('says, when this computer cannot carry the call, that it is taken on the Companion, and does not once it has said what it asked', async () => {
    serve({ canAccept: false });
    await mount();
    expect(text()).toContain('answer on your Companion');
    api.respond.mockResolvedValue({ ok: false, note: 'The Phone plugin cannot take a call for you: answer on the Companion.' });
    await click(button('Accept'));
    expect(text()).toContain('The Phone plugin cannot take a call for you');
    expect(text()).not.toContain('This computer cannot carry the call');
  });

  it('shows why a request failed, and is not stuck', async () => {
    serve({});
    api.respond.mockRejectedValue(new Error('the ring is over'));
    await mount();
    await click(button('Decline'));
    expect(host.querySelector('[role=alert]')?.textContent).toContain('the ring is over');
    expect(button('Decline').disabled).toBe(false);
  });

  it('is nothing at all while there is no phone', async () => {
    serve({});
    await mount({ on: false });
    expect(host.querySelector('.ring-dialog')).toBeNull();
    expect(api.active).not.toHaveBeenCalled();
  });
});
