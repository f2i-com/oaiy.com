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

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  pending.mockResolvedValue({ pending: [] });
  paired.mockResolvedValue({ paired: [saved] });
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

describe('PairingPrompt connected app disclosure', () => {
  it('starts collapsed, preserves identifying details, and remains open across polling', async () => {
    await mount();
    const details = container.querySelector('details')!;
    const summary = container.querySelector('summary')!;
    expect(details.open).toBe(false);
    expect(summary.textContent).toContain('Connected apps (1)');
    expect(container.querySelector('.pairing-origin')!.textContent).toBe(saved.origin);
    expect(container.querySelector('time')!.dateTime).toBe('2026-09-12T01:02:03.000Z');
    await act(async () => summary.click());
    expect(details.open).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(details.open).toBe(true);
    expect(revoke).not.toHaveBeenCalled();
  });

  it('keeps a new pairing decision visible outside the collapsed saved apps', async () => {
    pending.mockResolvedValue({ pending: [{ pairingId: 'new-1', product: 'New workspace', origin: 'https://new.example', code: '123456', status: 'pending', createdAtMs: saved.createdAtMs }] });
    await mount();
    const prompt = container.querySelector('[role="alert"]')!;
    expect(prompt.closest('details')).toBeNull();
    expect(prompt.textContent).toContain('123456');
    const button = Array.from(prompt.querySelectorAll('button')).find(el => el.textContent?.includes('Approve'))!;
    await act(async () => button.click());
    expect(approve).toHaveBeenCalledWith('new-1');
    expect(revoke).not.toHaveBeenCalled();
  });

  it('only revokes the selected saved app when its explicit control is used', async () => {
    await mount();
    await act(async () => container.querySelector('summary')!.click());
    const button = container.querySelector<HTMLButtonElement>('button[aria-label^="Revoke"]')!;
    expect(button.getAttribute('aria-label')).toContain(saved.origin);
    await act(async () => button.click());
    expect(revoke).toHaveBeenCalledWith(saved.id);
  });
});
