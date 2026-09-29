// The ring dialog: a caller wants to speak to the owner. It shows nothing until this desktop says someone is being rung
// for; then who, what they last said (as plain text), which devices ring and a countdown by the desktop's clock; that
// the owner answers on a Companion (there is no Accept: this computer cannot carry the call); "Decline and take a message"
// (the phone is asked to withdraw the request, and the ring shows as stopping until it answers) and "Not now" (the box is
// put away and the devices go on ringing). A ring past its time is not shown. Same convention as the other tests:
// react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ActiveRing, RingNotice } from './api';

const api = vi.hoisted(() => ({ active: vi.fn(), respond: vi.fn(), dismiss: vi.fn(), openSetup: vi.fn() }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, ring: { ...real.ring, active: (...a: unknown[]) => api.active(...a), respond: (...a: unknown[]) => api.respond(...a), dismissNotice: (...a: unknown[]) => api.dismiss(...a) } };
});
vi.mock('./useSetupState', () => ({ openSetup: (...a: unknown[]) => api.openSetup(...a) }));

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
  stopping: false,
  note: '',
  ...r,
});
/** The desktop answers with these rings, and its clock goes on as time passes. */
const serve = (...rings: Array<Partial<ActiveRing>>) => api.active.mockImplementation(async () => ({ rings: rings.map((r) => ringing(r)), notices: [] }));
const notice = (n: Partial<RingNotice> = {}): RingNotice => ({
  id: 'notice_1',
  callId: 'call_1',
  callerName: 'Alex',
  callerNumber: '+61491570006',
  at: NOW,
  text: 'Someone asked for you. No device is set up to take a transfer, so they were offered a message.',
  ...n,
});

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
  api.active.mockResolvedValue({ rings: [], notices: [] });
  api.respond.mockResolvedValue({ ok: true, note: '' });
  api.dismiss.mockResolvedValue({ ok: true });
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

  it('says the owner answers on a Companion, and has no Accept: this computer cannot take the call', async () => {
    serve({});
    await mount();
    expect(text()).toContain('Answer on your Companion. This computer cannot take the call.');
    expect(button('Accept')).toBeUndefined();
    expect([...host.querySelectorAll('button')].map((b) => b.textContent?.trim())).toEqual(['Decline and take a message', 'Not now']);
  });

  it('Decline and take a message asks the phone to withdraw the request and shows it stopping until the phone answers', async () => {
    serve({});
    api.respond.mockResolvedValue({ ok: true, note: 'Asking your Companion to stop ringing. The receptionist will offer the caller a message.' });
    await mount();
    await click(button('Decline and take a message'));
    expect(api.respond).toHaveBeenCalledWith('assist_1', 'decline');
    // Nothing is decided here: the ring stays, stopping, and the next look says so.
    serve({ stopping: true, note: 'Asking your Companion to stop ringing. The receptionist will offer the caller a message.' });
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS + 10);
    });
    await settle();
    expect(host.querySelector('.ring-dialog')).not.toBeNull();
    expect(text()).toContain('Asking your Companion to stop ringing.');
    expect(button('Decline and take a message').disabled).toBe(true);
    // The phone answered: the ring is gone.
    api.active.mockResolvedValue({ rings: [], notices: [] });
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS + 10);
    });
    await settle();
    expect(host.querySelector('.ring-dialog')).toBeNull();
  });

  it('Not now only puts the box away: nothing is asked of the phone and the ring is not shown again', async () => {
    serve({});
    await mount();
    await click(button('Not now'));
    expect(api.respond).not.toHaveBeenCalled();
    expect(host.querySelector('.ring-dialog')).toBeNull();
    // The devices go on ringing, and a look that still lists it does not bring the box back.
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS * 2 + 10);
    });
    await settle();
    expect(host.querySelector('.ring-dialog')).toBeNull();
    // A different ring is shown.
    serve({ id: 'assist_2' });
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS + 10);
    });
    await settle();
    expect(host.querySelector('.ring-dialog')).not.toBeNull();
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
    await click(button('Decline and take a message'));
    expect(api.respond).toHaveBeenLastCalledWith('live', 'decline');
  });

  it('shows why a request failed, and is not stuck', async () => {
    serve({});
    api.respond.mockRejectedValue(new Error('the ring is over'));
    await mount();
    await click(button('Decline and take a message'));
    expect(host.querySelector('[role=alert]')?.textContent).toContain('the ring is over');
    expect(button('Decline and take a message').disabled).toBe(false);
  });

  it('tells the owner, without a ring, that someone asked for them when no device is set up, and lets them dismiss it or go and set one up', async () => {
    api.active.mockResolvedValue({ rings: [], notices: [notice()] });
    await mount();
    expect(host.querySelector('.ring-dialog')).toBeNull();
    const stack = host.querySelector('[role=status].ring-notices')!;
    expect(stack.textContent).toContain('Alex asked for you.');
    expect(stack.textContent).toContain('No device is set up to take a transfer, so they were offered a message.');
    expect(button('Accept')).toBeUndefined();
    await click(button('Set up a Companion'));
    expect(api.openSetup).toHaveBeenCalledWith({ plugin: 'aokie', step: 'pair' });
    await click(button('Dismiss'));
    expect(api.dismiss).toHaveBeenCalledWith('notice_1');
    expect(host.querySelector('.ring-notices')).toBeNull();
    // A look that was already on its way and still lists it does not bring it back.
    await act(async () => {
      vi.advanceTimersByTime(POLL_MS + 10);
    });
    await settle();
    expect(host.querySelector('.ring-notices')).toBeNull();
  });

  it('shows a notice as plain text and a hidden number as such, beside a ring that is going', async () => {
    api.active.mockResolvedValue({ rings: [ringing({})], notices: [notice({ callerName: '<b>Alex</b>', id: 'n2' }), notice({ callerName: '', callerNumber: '', id: 'n3' })] });
    await mount();
    expect(host.querySelector('.ring-dialog')).not.toBeNull();
    const stack = host.querySelector('.ring-notices')!;
    expect(stack.querySelector('b')).toBeNull();
    expect(stack.textContent).toContain('<b>Alex</b> asked for you.');
    expect(stack.textContent).toContain('A caller who hid their number asked for you.');
  });

  it('is nothing at all while there is no phone', async () => {
    serve({});
    await mount({ on: false });
    expect(host.querySelector('.ring-dialog')).toBeNull();
    expect(api.active).not.toHaveBeenCalled();
  });
});
