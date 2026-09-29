// How the phone sounds: how long the greeting waits after a call connects.
// Read from the voices list (or its own route on a desktop whose list has no
// delay), saved once the slider rests, shown as the desktop kept it, and
// hidden on a desktop without the setting (a 404). `fetch` is stubbed, so the
// real client in api.ts is what is checked.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CallVoice } from './HoursPanel';
import { ToastProvider } from './Toasts';

type Answer = { status?: number; body?: unknown };
/** `METHOD /path` → its answer; anything else is a 404, as on a desktop without it. */
let routes: Record<string, (init?: RequestInit) => Answer>;
const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
  const key = `${init?.method ?? 'GET'} ${new URL(url).pathname}`;
  const answer = routes[key]?.(init) ?? { status: 404, body: { error: 'no such route' } };
  return new Response(JSON.stringify(answer.body ?? {}), { status: answer.status ?? 200 });
});

const VOICE = { name: 'Front desk', file: 'front-desk.wav', bytes: 240_000, written: true };
const list = (extra: object = {}) => () => ({ body: { voices: [VOICE], chosen: 'Front desk', ...extra } });

let host: HTMLDivElement;
let root: Root;
const text = () => host.textContent ?? '';
const slider = () => host.querySelector<HTMLInputElement>('#greeting-delay');
const shown = () => host.querySelector('.voice-delay-value')?.textContent;
const calls = (method: string, path: string) => fetchMock.mock.calls.filter(([url, init]) => (init?.method ?? 'GET') === method && new URL(url).pathname === path);
const puts = () => calls('PUT', '/api/voice/settings');

/** Moves the slider as a person does: the value, then the `input` event React listens for. */
async function slide(value: number) {
  const input = slider()!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, String(value));
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
/** Lets the slider rest (the save waits 400 ms), then lets the answer in. */
async function rest(ms = 400) {
  await act(async () => {
    vi.advanceTimersByTime(ms);
  });
  await act(async () => {});
}

async function mount() {
  await act(async () =>
    root.render(
      <ToastProvider>
        <CallVoice business="Green Lawns" />
      </ToastProvider>,
    ),
  );
  // The list, then (on an older list) the settings on their own.
  await act(async () => {});
  await act(async () => {});
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  fetchMock.mockClear();
  vi.stubGlobal('fetch', fetchMock);
  routes = {
    'GET /api/voice/voices': list({ greetingDelayMs: 1500 }),
    'GET /api/voice/settings': () => ({ body: { greetingDelayMs: 1500 } }),
    'PUT /api/voice/settings': (init) => ({ body: JSON.parse(String(init?.body)) }),
  };
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('Greet after the call connects', () => {
  it('reads the delay from the voices list and says it in seconds, with why', async () => {
    routes['GET /api/voice/voices'] = list({ greetingDelayMs: 2250 });
    await mount();
    expect(slider()!.value).toBe('2250');
    expect(slider()!.getAttribute('aria-valuetext')).toBe('2.25 seconds');
    expect([slider()!.min, slider()!.max, slider()!.step]).toEqual(['0', '5000', '250']);
    expect(shown()).toBe('2.25 s');
    expect(text()).toContain('Greet after the call connects');
    expect(text()).toContain('The greeting starts sooner if the caller speaks first. Calls the agent places wait for the other person’s hello.');
    // The list said it: not asked again.
    expect(calls('GET', '/api/voice/settings')).toHaveLength(0);
  });

  it('on a desktop whose list has no delay, reads it from the settings', async () => {
    routes['GET /api/voice/voices'] = list();
    routes['GET /api/voice/settings'] = () => ({ body: { greetingDelayMs: 750 } });
    await mount();
    expect(calls('GET', '/api/voice/settings')).toHaveLength(1);
    expect(slider()!.value).toBe('750');
    expect(shown()).toBe('0.75 s');
  });

  it('saves once the slider rests: one PUT with the last value', async () => {
    await mount();
    expect(shown()).toBe('1.5 s');
    await slide(1750);
    await slide(2500);
    await slide(3000);
    // Said at once, saved only when it rests.
    expect(shown()).toBe('3 s');
    await rest(399);
    expect(puts()).toHaveLength(0);
    await rest(1);
    expect(puts()).toHaveLength(1);
    const [url, init] = puts()[0];
    expect(url).toBe('http://127.0.0.1:17972/api/voice/settings');
    expect(init?.method).toBe('PUT');
    expect(JSON.parse(String(init?.body))).toEqual({ greetingDelayMs: 3000 });
    expect(shown()).toBe('3 s');
    expect(slider()!.value).toBe('3000');
  });

  it('shows what the desktop kept, not what was asked', async () => {
    // The desktop keeps it to 0 to 5 s and answers with what it kept.
    routes['PUT /api/voice/settings'] = () => ({ body: { greetingDelayMs: 1500 } });
    await mount();
    await slide(2000);
    expect(shown()).toBe('2 s');
    await rest();
    expect(JSON.parse(String(puts()[0][1]?.body))).toEqual({ greetingDelayMs: 2000 });
    expect(shown()).toBe('1.5 s');
    expect(slider()!.value).toBe('1500');
    expect(slider()!.getAttribute('aria-valuetext')).toBe('1.5 seconds');
  });

  it('is hidden on a desktop without it (a 404), and the voices are still there', async () => {
    routes['GET /api/voice/voices'] = list();
    delete routes['GET /api/voice/settings'];
    await mount();
    expect(calls('GET', '/api/voice/settings')).toHaveLength(1);
    expect(slider()).toBeNull();
    expect(text()).not.toContain('Greet after the call connects');
    expect(text()).toContain('Front desk');
  });

  it('is hidden when saving it finds no such route (a 404)', async () => {
    delete routes['PUT /api/voice/settings'];
    await mount();
    await slide(2500);
    await rest();
    expect(puts()).toHaveLength(1);
    expect(slider()).toBeNull();
    expect(text()).not.toContain('Greet after the call connects');
    expect(text()).not.toContain('Not saved');
  });

  it('a save that fails says so and goes back to the saved value', async () => {
    routes['PUT /api/voice/settings'] = () => ({ status: 500, body: { error: { code: 'bad_settings', message: 'The disk is full' } } });
    await mount();
    await slide(4000);
    await rest();
    expect(text()).toContain('Not saved');
    expect(text()).toContain('The disk is full');
    expect(shown()).toBe('1.5 s');
    expect(slider()!.value).toBe('1500');
  });

  it('a change still waiting when the page closes is saved anyway', async () => {
    await mount();
    await slide(500);
    expect(puts()).toHaveLength(0);
    act(() => root.unmount());
    expect(puts()).toHaveLength(1);
    expect(JSON.parse(String(puts()[0][1]?.body))).toEqual({ greetingDelayMs: 500 });
    // afterEach unmounts again: a fresh root so that is harmless.
    root = createRoot(host);
  });
});

describe('hearing a voice', () => {
  it('says the business and the receptionist, as callers hear them', async () => {
    routes['POST /api/voice/voices/Front%20desk/try'] = () => ({ body: {} });
    // jsdom plays nothing: the sample is taken as played.
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockReturnValue(undefined);
    await act(async () =>
      root.render(
        <ToastProvider>
          <CallVoice business="Green Lawns" receptionist="Sam" />
        </ToastProvider>,
      ),
    );
    await act(async () => {});
    await act(async () => {});
    const hear = [...host.querySelectorAll('button')].find((b) => b.textContent?.includes('Hear it'))!;
    await act(async () => hear.click());
    const tried = fetchMock.mock.calls.filter(([url, init]) => init?.method === 'POST' && String(url).endsWith('/try'));
    expect(tried).toHaveLength(1);
    expect(JSON.parse(String(tried[0][1]?.body))).toEqual({ text: 'Hi, thanks for calling Green Lawns, this is Sam. How can I help you today?' });
    expect(text()).not.toContain('Could not speak');
    // Closed (it stops the sample) while jsdom's player is still stood in for.
    act(() => root.unmount());
    root = createRoot(host);
    vi.restoreAllMocks();
  });
});
