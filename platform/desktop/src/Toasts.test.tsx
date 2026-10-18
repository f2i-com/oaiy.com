import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ToastProvider, useToast } from './Toasts';

let container: HTMLDivElement;
let root: Root;
let push: ReturnType<typeof useToast>['push'];

function Grab() {
  push = useToast().push;
  return null;
}

beforeEach(async () => {
  vi.useFakeTimers();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(<ToastProvider><Grab /></ToastProvider>));
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
});
const shown = () => [...container.querySelectorAll('.toast-title')].map((t) => t.textContent);

describe('Toasts', () => {
  it('keeps an error long enough to read, and holds any toast while the pointer is on it', async () => {
    await act(async () => {
      push({ kind: 'success', title: 'Saved' });
      push({ kind: 'error', title: 'It failed', body: 'a long reason' });
    });
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(shown()).toEqual(['It failed']);
    // held under the pointer past its time, then let go with what it had left
    const error = container.querySelector('.toast')!;
    await act(async () => { error.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); });
    await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
    expect(shown()).toEqual(['It failed']);
    await act(async () => { error.dispatchEvent(new MouseEvent('mouseout', { bubbles: true })); });
    await act(async () => { await vi.advanceTimersByTimeAsync(7000); });
    expect(shown()).toEqual([]);
  });
});
