// Messages: what callers left the owner, newest first, with who they are and where to ring them back; new until
// they have been on screen a few seconds; handled, made new again, deleted after asking; searched by a number
// written any way; a link to the caller's contact; what a caller said shown as plain text; and a desktop older than
// messages (a 404). There is no way to write one here. Same convention as the other tests: react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CallerMessage } from './api';

const api = vi.hoisted(() => ({ list: vi.fn(), mark: vi.fn(), remove: vi.fn() }));
vi.mock('./api', async (importOriginal) => {
  const real = await importOriginal<typeof import('./api')>();
  return { ...real, messages: { list: (...a: unknown[]) => api.list(...a), mark: (...a: unknown[]) => api.mark(...a), remove: (...a: unknown[]) => api.remove(...a) } };
});

import MessagesPanel, { SEEN_AFTER_MS, matchesMessage, whoOf } from './MessagesPanel';
import { ToastProvider } from './Toasts';

const message = (m: Partial<CallerMessage> & Pick<CallerMessage, 'id'>): CallerMessage => ({
  at: '2026-09-30T02:15:03Z',
  callId: 'call_1',
  from: '+61491570006',
  name: '',
  callback: '+61491570006',
  message: 'Please ring me about Friday.',
  urgency: 'normal',
  wantsCallback: true,
  state: 'new',
  seenAt: null,
  handledAt: null,
  handledBy: null,
  ...m,
});

let host: HTMLDivElement;
let root: Root;
let stored: CallerMessage[];
const text = () => host.textContent ?? '';
const settle = async () => {
  for (let i = 0; i < 4; i++) await act(async () => {});
};
const button = (label: string, within: ParentNode = host) =>
  [...within.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label || b.getAttribute('aria-label') === label)!;
const click = async (el: Element) => {
  await act(async () => (el as HTMLElement).click());
  await settle();
};
const card = (id: string) => host.querySelector<HTMLElement>(`[data-message="${id}"]`)!;
const ids = () => [...host.querySelectorAll('[data-message]')].map((c) => c.getAttribute('data-message'));
const contactOpened = vi.fn();

async function mount() {
  await act(async () => {
    root.render(
      <ToastProvider>
        <MessagesPanel onOpenContact={contactOpened} />
      </ToastProvider>,
    );
  });
  await settle();
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  stored = [
    message({ id: 'msg_b', at: '2026-09-30T03:00:00Z', name: 'Sam', from: '+61491570156', callback: '0491 570 006', message: 'The gate is <b>jammed</b>.', urgency: 'urgent' }),
    message({ id: 'msg_a', at: '2026-09-29T09:00:00Z', name: 'Alex', state: 'seen', seenAt: '2026-09-29T10:00:00Z' }),
    message({ id: 'msg_c', at: '2026-09-28T09:00:00Z', from: '', callback: '', name: '', message: 'They did not say who.', wantsCallback: false, state: 'handled', handledAt: '2026-09-28T10:00:00Z', handledBy: 'owner' }),
  ];
  api.list.mockImplementation(async () => ({ messages: stored, total: stored.length, unread: stored.filter((m) => m.state === 'new').length }));
  api.mark.mockImplementation(async (id: string, state: CallerMessage['state']) => {
    stored = stored.map((m) => (m.id === id ? { ...m, state, seenAt: state === 'new' ? null : (m.seenAt ?? '2026-09-30T04:00:00Z') } : m));
    return stored.find((m) => m.id === id);
  });
  api.remove.mockImplementation(async (id: string) => {
    stored = stored.filter((m) => m.id !== id);
  });
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.useRealTimers();
});

describe('Messages', () => {
  it('lists what is left to do, newest first, with who they are and where to ring them back', async () => {
    await mount();
    expect(ids()).toEqual(['msg_b', 'msg_a']);
    const sam = card('msg_b');
    expect(sam.querySelector('.message-who')?.textContent).toBe('Sam');
    expect(sam.textContent).toContain('Urgent');
    expect(sam.textContent).toContain('New');
    expect(sam.querySelector('a.message-number')?.textContent).toContain('0491 570 006');
    expect(sam.querySelector('a.message-number')?.getAttribute('href')).toBe('tel:0491570006');
    // No name given: their number, written the way people read it.
    expect(whoOf(message({ id: 'x', from: '+61491570006' }))).toBe('0491 570 006');
    expect(whoOf(message({ id: 'x', from: '', name: '' }))).toBe('A caller who hid their number');
    expect(card('msg_a').textContent).toContain('Wants a call back');
    expect(text()).not.toContain('They did not say who.');
  });

  it('shows what a caller said as plain text, never as markup', async () => {
    await mount();
    const words = card('msg_b').querySelector('.message-text')!;
    expect(words.textContent).toBe('The gate is <b>jammed</b>.');
    expect(words.querySelector('b')).toBeNull();
  });

  it('counts a new message as seen once it has been on screen a few seconds', async () => {
    await mount();
    expect(card('msg_b').textContent).toContain('New');
    expect(api.mark).not.toHaveBeenCalled();
    await act(async () => {
      vi.advanceTimersByTime(SEEN_AFTER_MS + 50);
    });
    await settle();
    expect(api.mark).toHaveBeenCalledWith('msg_b', 'seen');
    expect(api.mark).toHaveBeenCalledTimes(1);
    expect(card('msg_b').querySelector('.message-new')).toBeNull();
  });

  it('marks a message handled, moves it to Handled, and can make it new again', async () => {
    await mount();
    await click(button('Handled', card('msg_a')));
    expect(api.mark).toHaveBeenCalledWith('msg_a', 'handled');
    expect(ids()).toEqual(['msg_b']);
    await click(button('Handled', host.querySelector('[role=tablist]')!));
    expect(ids()).toEqual(['msg_a', 'msg_c']);
    await click(button('Not handled', card('msg_c')));
    expect(api.mark).toHaveBeenCalledWith('msg_c', 'new');
    expect(ids()).toEqual(['msg_a']);
    // Everything is under All.
    await click(button('All'));
    expect(ids().sort()).toEqual(['msg_a', 'msg_b', 'msg_c']);
  });

  it('asks before it deletes a message, and keeps it when the owner changes their mind', async () => {
    await mount();
    await click(button('Delete the message from Sam'));
    expect(api.remove).not.toHaveBeenCalled();
    await click(button('Keep it'));
    expect(ids()).toContain('msg_b');
    await click(button('Delete the message from Sam'));
    await click(button('Delete it'));
    expect(api.remove).toHaveBeenCalledWith('msg_b');
    expect(ids()).toEqual(['msg_a']);
  });

  it('finds a message by name, words or a number written any way', async () => {
    await mount();
    await click(button('All'));
    const search = host.querySelector<HTMLInputElement>('input[type=search]')!;
    const type = async (v: string) => {
      await act(async () => {
        const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
        set.call(search, v);
        search.dispatchEvent(new Event('input', { bubbles: true }));
      });
    };
    await type('0491 570 156');
    expect(ids()).toEqual(['msg_b']);
    await type('alex');
    expect(ids()).toEqual(['msg_a']);
    await type('friday');
    expect(ids()).toEqual(['msg_a']);
    await type('nothing like it');
    expect(ids()).toEqual([]);
    expect(text()).toContain('No message matches.');
    expect(matchesMessage(message({ id: 'x', from: '+61491570006' }), '0491570006')).toBe(true);
    expect(matchesMessage(message({ id: 'x', from: '+61491570006' }), '12')).toBe(false);
  });

  it('opens the caller’s contact by their number key, and only when there is a number', async () => {
    await mount();
    await click(button('Contact', card('msg_b')));
    expect(contactOpened).toHaveBeenCalledWith('491570156');
    await click(button('All'));
    expect(button('Contact', card('msg_c'))).toBeUndefined();
  });

  it('says so when nothing is waiting, and when there are no messages at all', async () => {
    stored = [message({ id: 'only', state: 'handled' })];
    await mount();
    expect(text()).toContain('Nothing waiting for you.');
    await act(async () => root.unmount());
    root = createRoot(host);
    stored = [];
    await mount();
    expect(text()).toContain('No messages yet.');
  });

  it('says plainly when new messages are being refused because too many wait for the owner', async () => {
    await mount();
    expect(host.querySelector('[data-testid=messages-notice]')).toBeNull();
    await act(async () => root.unmount());
    root = createRoot(host);
    api.list.mockImplementation(async () => ({ messages: stored, total: stored.length, unread: 1, notice: '100 messages from callers who hid their number are waiting: new ones from hidden numbers are refused until you mark some as handled or delete them.' }));
    await mount();
    const notice = host.querySelector('[data-testid=messages-notice]')!;
    expect(notice.getAttribute('role')).toBe('status');
    expect(notice.textContent).toContain('hid their number');
    expect(notice.textContent).toContain('refused until you mark some as handled');
    // The messages are still there to handle.
    expect(ids()).toEqual(['msg_b', 'msg_a']);
  });

  it('cannot write a message, and says so plainly on a desktop that keeps none', async () => {
    await mount();
    expect(host.querySelector('textarea')).toBeNull();
    expect([...host.querySelectorAll('button')].some((b) => /new message|add|write/i.test(b.textContent ?? ''))).toBe(false);
    await act(async () => root.unmount());
    root = createRoot(host);
    api.list.mockRejectedValue(new Error('404: not found'));
    await mount();
    expect(text()).toContain('does not keep messages yet');
  });
});
