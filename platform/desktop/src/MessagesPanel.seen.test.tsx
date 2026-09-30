// When a new message counts as seen: once it has been on screen for a few seconds, never before, and never over what the owner did meanwhile.
// (The final review's four proofs of the seen-timer defects, kept as tests.)
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CallerMessage } from './api';

const api = vi.hoisted(() => ({ list: vi.fn(), mark: vi.fn(), remove: vi.fn() }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, messages: { list: (...a: unknown[]) => api.list(...a), mark: (...a: unknown[]) => api.mark(...a), remove: (...a: unknown[]) => api.remove(...a) } };
});

import MessagesPanel, { SEEN_AFTER_MS } from './MessagesPanel';
import { ToastProvider } from './Toasts';

const message = (m: Partial<CallerMessage> & Pick<CallerMessage, 'id'>): CallerMessage => ({
  at: '2026-09-30T02:15:03Z', callId: 'call_1', from: '+61491570006', name: '', callback: '+61491570006', message: 'Please ring me about Friday.',
  urgency: 'normal', wantsCallback: true, state: 'new', seenAt: null, handledAt: null, handledBy: null, ...m,
});

let host: HTMLDivElement;
let root: Root;
let stored: CallerMessage[];
const settle = async () => { for (let i = 0; i < 4; i++) await act(async () => {}); };
/** A tab of the page (its label may carry a count). */
const button = (label: string) => [...host.querySelectorAll<HTMLButtonElement>('button[role=tab]')].find((b) => b.textContent?.trim().startsWith(label))!;
const search = (text: string) => act(async () => {
  const box = host.querySelector<HTMLInputElement>('input[type=search]')!;
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(box, text);
  box.dispatchEvent(new Event('input', { bubbles: true }));
});
async function mount() {
  await act(async () => { root.render(<ToastProvider><MessagesPanel /></ToastProvider>); });
  await settle();
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  stored = [
    message({ id: 'msg_new', at: '2026-09-30T03:00:00Z', name: 'Sam', message: 'The gate is jammed.' }),
    message({ id: 'msg_old', at: '2026-09-28T09:00:00Z', name: 'Kim', state: 'handled', handledAt: '2026-09-28T10:00:00Z', handledBy: 'owner', message: 'Old news.' }),
  ];
  api.list.mockImplementation(async () => ({ messages: stored, total: stored.length, unread: stored.filter((m) => m.state === 'new').length }));
  api.mark.mockImplementation(async (id: string, state: CallerMessage['state']) => {
    stored = stored.map((m) => (m.id === id ? { ...m, state } : m));
    return stored.find((m) => m.id === id);
  });
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.useRealTimers();
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => false });
});

describe('when a new message is seen', () => {
  it('is not while it is not on screen: the Handled tab is open', async () => {
    await mount();
    await act(async () => button('Handled').click());
    await settle();
    expect(host.querySelector('[data-message="msg_new"]')).toBeNull();
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); });
    await settle();
    expect(api.mark).not.toHaveBeenCalled();
    // ...and it is once it is shown (the wait is from then).
    await act(async () => button('To do').click());
    await settle();
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); });
    await settle();
    expect(api.mark).toHaveBeenCalledWith('msg_new', 'seen');
  });

  it('is not while the search hides it, and is once the search shows it again', async () => {
    await mount();
    await search('zzz-nothing');
    expect(host.querySelector('[data-message="msg_new"]')).toBeNull();
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); });
    await settle();
    expect(api.mark).not.toHaveBeenCalled();
    await search('');
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); });
    await settle();
    expect(api.mark).toHaveBeenCalledWith('msg_new', 'seen');
  });

  it('is not after the owner handled it: a Handled clicked while the wait is about to end sends no later "seen" for it', async () => {
    let release: (() => void) | undefined;
    api.mark.mockImplementation(async (id: string, state: CallerMessage['state']) => {
      if (state === 'handled') await new Promise<void>((r) => (release = r)); // the desktop is a little slow to answer
      stored = stored.map((m) => (m.id === id ? { ...m, state } : m));
      return stored.find((m) => m.id === id);
    });
    await mount();
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS - 100); });
    const card = host.querySelector('[data-message="msg_new"]')!;
    const handled = [...card.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Handled')!;
    await act(async () => handled.click());
    await act(async () => { vi.advanceTimersByTime(200); }); // the wait ends while 'handled' is on its way
    release?.();
    await settle();
    const states = api.mark.mock.calls.filter((c) => c[0] === 'msg_new').map((c) => c[1]);
    expect(states).toEqual(['handled']);
  });

  it('is not undone by a "seen" that is still on its way: a message\'s buttons wait for its own change, not for the first that comes back', async () => {
    // Two new messages are seen together, and the desktop answers for one before the other. The one still on its way must not be handled meanwhile:
    // the "seen" that lands after the "handled" would make it new again.
    stored = [message({ id: 'msg_a', at: '2026-09-30T03:00:00Z', name: 'Sam' }), message({ id: 'msg_b', at: '2026-09-30T02:00:00Z', name: 'Kim' })];
    const answers = new Map<string, () => void>();
    api.mark.mockImplementation((id: string, state: CallerMessage['state']) => new Promise((resolve) => {
      answers.set(id, () => {
        stored = stored.map((m) => (m.id === id ? { ...m, state } : m));
        resolve(stored.find((m) => m.id === id));
      });
    }));
    await mount();
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); }); // "seen" is asked for both, and slow
    await settle();
    expect(api.mark.mock.calls.map((c) => c[0]).sort()).toEqual(['msg_a', 'msg_b']);
    const handled = (id: string) => [...host.querySelector(`[data-message="${id}"]`)!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Handled')!;
    expect([handled('msg_a').disabled, handled('msg_b').disabled]).toEqual([true, true]);
    await act(async () => { answers.get('msg_a')!(); });
    await settle();
    expect(handled('msg_b').disabled, 'msg_b is still being asked about').toBe(true);
    await act(async () => { answers.get('msg_b')!(); });
    await settle();
    expect(handled('msg_b').disabled).toBe(false);
  });

  it('is not lost when the wait ends while the window is hidden: it is seen once the window shows again', async () => {
    await mount();
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); });
    await settle();
    expect(api.mark).not.toHaveBeenCalled();
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => false });
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); window.dispatchEvent(new Event('focus')); });
    await settle();
    await act(async () => { vi.advanceTimersByTime(SEEN_AFTER_MS + 50); });
    await settle();
    expect(host.querySelector('[data-message="msg_new"]')).not.toBeNull();
    expect(api.mark).toHaveBeenCalledWith('msg_new', 'seen');
  });
});
