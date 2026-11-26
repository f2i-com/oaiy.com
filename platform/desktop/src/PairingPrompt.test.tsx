import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { pending, paired, approve, deny, revoke, push } = vi.hoisted(() => ({
  pending: vi.fn(), paired: vi.fn(), approve: vi.fn(), deny: vi.fn(), revoke: vi.fn(), push: vi.fn(),
}));
vi.mock('./api', () => ({ pairing: { pending, paired, approve, deny, revoke } }));
vi.mock('./Toasts', () => ({ useToast: () => ({ push }) }));
import PairingPrompt from './PairingPrompt';

let container: HTMLDivElement;
let root: Root;
const saved = { id: 'saved-1', product: 'FormLogic', label: 'My workspace', origin: 'https://workspace.example', createdAtMs: Date.parse('2026-09-12T01:02:03Z') };
const request = { pairingId: 'new-1', product: 'formlogic', label: 'New workspace', origin: 'https://new.example', code: '123456', status: 'pending', createdAtMs: saved.createdAtMs };

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  pending.mockResolvedValue({ pending: [] });
  paired.mockResolvedValue({ paired: [saved] });
  approve.mockResolvedValue({});
  deny.mockResolvedValue({});
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
});
async function mount() { await act(async () => root.render(<PairingPrompt />)); }
const buttonIn = (el: Element, label: string) =>
  Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.includes(label))!;

describe('PairingPrompt', () => {
  it('shows nothing when no request is waiting, even with apps connected', async () => {
    // The connected apps have their own section on Connections (and Overview).
    // Listing them here as well put a second copy above that page's heading.
    await mount();
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(container.innerHTML).toBe('');
    expect(paired).not.toHaveBeenCalled();
  });

  it('shows a request with the code to confirm, and approves only that request', async () => {
    pending.mockResolvedValue({ pending: [request] });
    await mount();
    const prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.textContent).toContain('An app wants to connect');
    expect(prompt.textContent).toContain('123456');
    expect(container.textContent).not.toContain('Connected apps');
    await act(async () => buttonIn(prompt, 'Approve').click());
    expect(approve).toHaveBeenCalledWith('new-1');
    expect(deny).not.toHaveBeenCalled();
    expect(revoke).not.toHaveBeenCalled();
    expect(push).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: 'Approved formlogic' }));
  });

  it('says who asks by the address the request came from, on a line of its own, and what it calls itself as its own words', async () => {
    // The label was the heading and the product stood in bold before the address: a page that called itself
    // "https://formlogic.com" read as that site, with its real address after it in grey.
    const posing = { ...request, origin: 'https://evil.example', label: 'https://formlogic.com' };
    pending.mockResolvedValue({ pending: [posing] });
    await mount();
    let prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.querySelector('.pairing-who')!.textContent).toBe('From https://evil.example');
    expect(prompt.querySelector('.pairing-who b')!.textContent).toBe('https://evil.example');
    expect(prompt.querySelector('.pairing-head')!.textContent).toBe('An app wants to connect');
    expect(prompt.textContent).toContain('It calls itself “https://formlogic.com” (formlogic).');
    // Nothing the asker wrote is in bold or in the heading: only the address is.
    expect(Array.from(prompt.querySelectorAll('b, strong')).map((el) => el.textContent)).toEqual(['An app wants to connect', 'https://evil.example']);

    // FormLogic gives its own address as its label: said once, as who asks.
    pending.mockResolvedValue({ pending: [{ ...request, origin: 'https://formlogic.com', label: 'https://formlogic.com' }] });
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.querySelector('.pairing-who')!.textContent).toBe('From https://formlogic.com');
    expect(prompt.textContent).toContain('It calls itself “formlogic”. It wants to run your flows');

    // A program on this computer has no address, and no label is no claim.
    pending.mockResolvedValue({ pending: [{ ...request, origin: undefined, label: undefined, product: 'oaiy-cli' }] });
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.querySelector('.pairing-who')!.textContent).toBe('From a program on this computer');
    expect(prompt.textContent).toContain('It calls itself “oaiy-cli”.');
  });

  it('says what is granted: an app may run flows, which can run programs; an Agent reaches the phone and settings too', async () => {
    pending.mockResolvedValue({ pending: [request] });
    await mount();
    let prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.textContent).toContain('https://new.example');
    expect(prompt.textContent).toContain('A flow can run programs on this computer');
    expect(prompt.textContent).not.toContain('OAIY Agent');
    pending.mockResolvedValue({ pending: [{ ...request, access: 'agent', origin: undefined }] });
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.textContent).toContain('a program on this computer');
    expect(prompt.textContent).toContain('wants to be your OAIY Agent');
    expect(prompt.textContent).toContain('calendar and contacts');
  });

  it('denies a request when asked', async () => {
    pending.mockResolvedValue({ pending: [request] });
    await mount();
    const prompt = container.querySelector('[role="alert"]')!;
    await act(async () => buttonIn(prompt, 'Deny').click());
    expect(deny).toHaveBeenCalledWith('new-1');
    expect(approve).not.toHaveBeenCalled();
  });

  it('keeps a request on screen when one poll fails', async () => {
    pending.mockResolvedValue({ pending: [request] });
    await mount();
    pending.mockRejectedValue(new Error('starting'));
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(container.querySelector('[role="alert"]')!.textContent).toContain('123456');
  });
});
