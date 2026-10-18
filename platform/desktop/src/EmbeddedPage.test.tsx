// The box a page in a webview of its own is laid over. The desktop lays the webview where this component says
// the box is, and it was told only when the box changed size or the window did. A box that moves and keeps its
// size (what is above it grew, and the page could not shrink) left the webview where the box had been: on a Mac
// it was seen over the lower half of a section's tabs.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const h = vi.hoisted(() => ({ invoke: vi.fn((_command: string, _args?: unknown) => Promise.resolve()) }));
vi.mock('./api', () => ({ isTauri: () => true, tauriInvoke: h.invoke }));

import EmbeddedPage from './EmbeddedPage';

let host: HTMLDivElement;
let root: Root;
let box = { left: 222, top: 124, width: 1058, height: 642 };

beforeEach(() => {
  (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers();
  vi.stubGlobal('requestAnimationFrame', (run: FrameRequestCallback) => setTimeout(() => run(0), 0));
  vi.stubGlobal('cancelAnimationFrame', (id: number) => clearTimeout(id));
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  box = { left: 222, top: 124, width: 1058, height: 642 };
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({ ...box, right: box.left + box.width, bottom: box.top + box.height, x: box.left, y: box.top, toJSON: () => box }) as DOMRect);
  h.invoke.mockClear();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const shown = () => h.invoke.mock.calls.filter(([command]) => command === 'show_embedded').map(([, args]) => args as { page: string; x: number; y: number; width: number; height: number; seen: Record<string, number> });

describe('the box a page in its own webview is laid over', () => {
  it('says where it is once it is drawn, with what the page saw', async () => {
    act(() => root.render(<EmbeddedPage page="engines" />));
    await act(async () => { await vi.advanceTimersByTimeAsync(20); });
    const calls = shown();
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ page: 'engines', x: 222, y: 124, width: 1058, height: 642 });
    expect(Object.keys(calls[0].seen).sort()).toEqual(['height', 'ratio', 'scrollX', 'scrollY', 'width']);
  });

  it('is not said again while it stays where it is', async () => {
    act(() => root.render(<EmbeddedPage page="flows" />));
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(shown()).toHaveLength(1);
  });

  it('says where it is again when it has moved without changing size', async () => {
    act(() => root.render(<EmbeddedPage page="engines" />));
    await act(async () => { await vi.advanceTimersByTimeAsync(20); });
    box = { ...box, top: 152 };
    await act(async () => { await vi.advanceTimersByTimeAsync(700); });
    const calls = shown();
    expect(calls).toHaveLength(2);
    expect(calls[1]).toMatchObject({ x: 222, y: 152, width: 1058, height: 642 });
  });

  it('stops looking, and hides the page, when it is gone', async () => {
    act(() => root.render(<EmbeddedPage page="agent" />));
    await act(async () => { await vi.advanceTimersByTimeAsync(20); });
    act(() => root.render(<div />));
    expect(h.invoke.mock.calls.some(([command]) => command === 'hide_embedded')).toBe(true);
    box = { ...box, top: 300 };
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(shown()).toHaveLength(1);
  });

  it('hides the page again when a show is answered after the page was left', async () => {
    // The first show makes the webview, which takes a while: the hide sent as the page was left can be answered first.
    let answer: () => void = () => {};
    h.invoke.mockImplementation((command: string) => (command === 'show_embedded' ? new Promise<void>((r) => (answer = r)) : Promise.resolve()));
    try {
      act(() => root.render(<EmbeddedPage page="flows" />));
      await act(async () => { await vi.advanceTimersByTimeAsync(20); });
      act(() => root.render(<div />));
      const hides = () => h.invoke.mock.calls.filter(([command]) => command === 'hide_embedded').length;
      expect(hides()).toBe(1);
      await act(async () => answer());
      expect(hides()).toBe(2);
    } finally {
      h.invoke.mockImplementation(() => Promise.resolve());
    }
  });

  it('lays the page wanted now again when another page’s show is answered after it', async () => {
    let answer: () => void = () => {};
    h.invoke.mockImplementation((command: string, args?: unknown) => (command === 'show_embedded' && (args as { page: string }).page === 'flows' ? new Promise<void>((r) => (answer = r)) : Promise.resolve()));
    try {
      act(() => root.render(<EmbeddedPage page="flows" />));
      await act(async () => { await vi.advanceTimersByTimeAsync(20); });
      act(() => root.render(<EmbeddedPage page="agent" />));
      await act(async () => { await vi.advanceTimersByTimeAsync(20); });
      expect(shown().filter((c) => c.page === 'agent')).toHaveLength(1);
      await act(async () => answer());
      await act(async () => { await vi.advanceTimersByTimeAsync(20); });
      expect(shown().filter((c) => c.page === 'agent')).toHaveLength(2);
    } finally {
      h.invoke.mockImplementation(() => Promise.resolve());
    }
  });

  it('takes the webview away while a call rings over the page, and lays it where the box is again after', async () => {
    await act(async () => root.render(<EmbeddedPage page="agent" />));
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    expect(h.invoke).toHaveBeenLastCalledWith('show_embedded', expect.objectContaining({ page: 'agent' }));
    h.invoke.mockClear();
    await act(async () => root.render(<EmbeddedPage page="agent" covered />));
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(h.invoke.mock.calls.map((c) => c[0])).toContain('hide_embedded');
    expect(h.invoke.mock.calls.map((c) => c[0])).not.toContain('show_embedded');
    h.invoke.mockClear();
    await act(async () => root.render(<EmbeddedPage page="agent" />));
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    expect(h.invoke).toHaveBeenLastCalledWith('show_embedded', expect.objectContaining({ page: 'agent', x: 222, y: 124 }));
  });
});
