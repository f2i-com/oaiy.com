// The banner over every page while OAIY's API refuses this window for the lack of its key (WindowKeyBanner.tsx): a
// window that cannot show it is OAIY's own says so, with a way to ask again, instead of a dashboard that shows nothing.
//
// Same convention as the other panel tests: raw react-dom/client + act.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { API_BASE } from './api';
import WindowKeyBanner from './WindowKeyBanner';
import { forgetWindowKey, KEY_REFUSAL, WINDOW_KEY_ERROR, windowFetch, windowKeyProblem } from './windowKey';

const KEY = '5f3c9a0e1b7d4c2f8a6e0d9b3c1f7a5e2d4b6c8e0f1a3b5c7d9e1f2a4b6c8d0e';
const bridged = window as unknown as { __TAURI_INTERNALS__?: { invoke: (cmd: string) => Promise<unknown> } };
const refused = () => new Response(JSON.stringify({ error: KEY_REFUSAL }), { status: 403 });
const ok = () => new Response('{}', { status: 200 });

let host: HTMLDivElement;
let root: Root;
/** Whether the desktop hands the key over when asked. */
let gives: boolean;

beforeEach(async () => {
  gives = false;
  bridged.__TAURI_INTERNALS__ = { invoke: vi.fn(async () => { if (!gives) throw new Error('refused'); return KEY; }) };
  forgetWindowKey();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => root.render(<WindowKeyBanner />));
});

afterEach(async () => {
  // (leave no problem behind: the window has its key and is answered)
  gives = true;
  forgetWindowKey();
  vi.stubGlobal('fetch', vi.fn(async () => ok()));
  await act(async () => { await windowFetch(`${API_BASE}/api/config`); });
  expect(windowKeyProblem()).toBeNull();
  act(() => root.unmount());
  host.remove();
  delete bridged.__TAURI_INTERNALS__;
  vi.unstubAllGlobals();
});

describe('the banner of a window OAIY is refusing', () => {
  it('is not there while nothing is wrong', async () => {
    expect(host.textContent).toBe('');
    vi.stubGlobal('fetch', vi.fn(async () => ok()));
    await act(async () => { await windowFetch(`${API_BASE}/api/config`); });
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('says what is wrong and what to do, and goes when "Try again" gets the key', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => refused()));
    await act(async () => { await windowFetch(`${API_BASE}/api/config`); });
    const banner = host.querySelector('[role="alert"]')!;
    expect(banner.textContent).toContain(WINDOW_KEY_ERROR);
    expect(WINDOW_KEY_ERROR).toContain('Restart OAIY');
    const again = banner.querySelector('button')!;
    expect(again.textContent).toContain('Try again');
    // The desktop still gives none: it stays.
    await act(async () => again.click());
    expect(host.querySelector('[role="alert"]')).not.toBeNull();
    gives = true;
    await act(async () => host.querySelector<HTMLButtonElement>('[role="alert"] button')!.click());
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });
});
