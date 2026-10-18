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
import { resetModules } from './useModules';

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
  resetModules();
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

// What callers hear first is the phone's own `greeting` setting (Aokie's), read from it and saved to it through its
// connector: the page showed a greeting of its own, and "Hear it" played a third, while the phone said Aokie's.
describe('the greeting', () => {
  /** The phone module, provided by Aokie through its connector `aokie`. */
  const phone = () => ({
    body: { revision: 1, modules: [{ id: 'phone', name: 'Phone', enabled: true, builtin: true, provider: { pluginId: 'aokie', name: 'Aokie', state: 'running', connector: 'aokie', declared: true } }], contributions: {}, warnings: [] },
  });
  let phoneGreeting: string | undefined;
  const sent = () => calls('POST', '/api/bridge/connectors/aokie/request').map(([, init]) => JSON.parse(String(init?.body)));
  beforeEach(() => {
    phoneGreeting = undefined;
    routes['GET /api/modules'] = phone;
    routes['POST /api/bridge/connectors/aokie/request'] = (init) => {
      const { command, payload } = JSON.parse(String(init?.body));
      if (command === 'settings.get') return { body: { ok: true, result: { ok: true, data: { settings: phoneGreeting === undefined ? {} : { greeting: phoneGreeting } } } } };
      if (command === 'settings.set') {
        phoneGreeting = payload.greeting;
        return { body: { ok: true, result: { ok: true, data: { appliesAtReconnect: [] } } } };
      }
      return { status: 400, body: { error: 'no such command' } };
    };
  });
  async function mountWith(business = 'Green Lawns', receptionist = '') {
    await act(async () =>
      root.render(
        <ToastProvider>
          <CallVoice business={business} receptionist={receptionist} />
        </ToastProvider>,
      ),
    );
    for (let i = 0; i < 4; i++) await act(async () => {});
  }
  const box = () => host.querySelector<HTMLInputElement>('#call-greeting')!;
  const said = () => [...host.querySelectorAll('.voice-greeting-said dd')].map((d) => d.textContent);
  const button = (label: string) => [...host.querySelectorAll('button')].find((b) => b.textContent?.trim() === label);

  it('says Aokie’s own greeting while the phone has none set, and how a caller it knows is greeted', async () => {
    await mountWith();
    expect(sent()[0]).toMatchObject({ command: 'settings.get' });
    expect(box().value).toBe('');
    expect(box().placeholder).toBe('Hello, thanks for calling. How can I help you today?');
    expect(said()[0]).toContain('“Hello, thanks for calling. How can I help you today?”');
    expect(said()[1]).toContain('“Hello Sam, thanks for calling. How can I help you today?”');
  });

  it('says the phone’s greeting, and saves a new one to the phone', async () => {
    phoneGreeting = 'Green Lawns, Aokie speaking.';
    await mountWith();
    expect(box().value).toBe('Green Lawns, Aokie speaking.');
    expect(said()[1]).toContain('“Hi Sam! Green Lawns, Aokie speaking.”');
    expect(button('Save')!.disabled).toBe(true);
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(box(), 'Hi there, Green Lawns here.');
      box().dispatchEvent(new Event('input', { bubbles: true }));
    });
    // Said as it is written, before it is saved.
    expect(said()[1]).toContain('“Hi Sam, Green Lawns here.”');
    await act(async () => button('Save')!.click());
    await act(async () => {});
    expect(sent().at(-1)).toMatchObject({ command: 'settings.set', payload: { greeting: 'Hi there, Green Lawns here.' } });
    expect(phoneGreeting).toBe('Hi there, Green Lawns here.');
    expect(text()).toContain('Greeting saved');
  });

  it('offers a greeting with the business’s name while the one set does not say it', async () => {
    await mountWith('Green Lawns', 'Sam');
    const offer = [...host.querySelectorAll('button')].find((b) => b.textContent?.startsWith('use “'))!;
    expect(offer.textContent).toBe('use “Hello, thanks for calling Green Lawns, this is Sam. How can I help you today?”');
    // (the receptionist is Sam: the caller it knows is someone else)
    expect(said()[1]).toContain('“Hello Alex, thanks for calling. How can I help you today?”');
    await act(async () => offer.click());
    expect(box().value).toBe('Hello, thanks for calling Green Lawns, this is Sam. How can I help you today?');
    expect([...host.querySelectorAll('button')].some((b) => b.textContent?.startsWith('use “'))).toBe(false);
  });

  it('says so when the phone cannot be asked, and shows Aokie’s own', async () => {
    routes['POST /api/bridge/connectors/aokie/request'] = () => ({ body: { ok: true, result: { ok: false, error: { message: 'the plugin is stopped' } } } });
    await mountWith();
    expect(text()).toContain('The phone’s greeting could not be read: the plugin is stopped');
    expect(box().disabled).toBe(true);
    expect(said()[0]).toContain('“Hello, thanks for calling. How can I help you today?”');
  });

  it('“Hear it” says the greeting as it reads in the box', async () => {
    phoneGreeting = 'Green Lawns, Aokie speaking.';
    routes['POST /api/voice/voices/Front%20desk/try'] = () => ({ body: {} });
    // jsdom plays nothing: the sample is taken as played.
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockReturnValue(undefined);
    await mountWith();
    const hear = [...host.querySelectorAll('button')].find((b) => b.textContent?.includes('Hear it'))!;
    await act(async () => hear.click());
    const tried = fetchMock.mock.calls.filter(([url, init]) => init?.method === 'POST' && String(url).endsWith('/try'));
    expect(tried).toHaveLength(1);
    expect(JSON.parse(String(tried[0][1]?.body))).toEqual({ text: 'Green Lawns, Aokie speaking.' });
    expect(text()).not.toContain('Could not speak');
    // Closed (it stops the sample) while jsdom's player is still stood in for.
    act(() => root.unmount());
    root = createRoot(host);
    vi.restoreAllMocks();
  });
});

describe('hearing a voice', () => {

  it('asks again in the voice’s row before it removes it', async () => {
    const OTHER = { name: 'Night line', file: 'night.wav', bytes: 120_000, written: true };
    routes['GET /api/voice/voices'] = () => ({ body: { voices: [VOICE, OTHER], chosen: 'Front desk' } });
    routes['DELETE /api/voice/voices/Night%20line'] = () => ({ body: { chosen: 'Front desk' } });
    await mount();
    const button = (label: string) => [...host.querySelectorAll('button')].find((b) => (b.getAttribute('aria-label') ?? b.textContent?.trim()) === label)!;
    await act(async () => button('Remove Night line').click());
    expect(calls('DELETE', '/api/voice/voices/Night%20line')).toHaveLength(0);
    await act(async () => button('Keep').click());
    expect(text()).toContain('Night line');
    await act(async () => button('Remove Night line').click());
    await act(async () => button('Remove').click());
    await act(async () => {});
    expect(calls('DELETE', '/api/voice/voices/Night%20line')).toHaveLength(1);
    expect(text()).not.toContain('Night line');
  });

  it('says “Speaking…” until the sample has been said, and lets the clip go then', async () => {
    routes['POST /api/voice/voices/Front%20desk/try'] = () => ({ body: {} });
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockReturnValue(undefined);
    const revoke = vi.spyOn(URL, 'revokeObjectURL');
    await mount();
    const hear = [...host.querySelectorAll('button')].find((b) => b.textContent?.includes('Hear it'))!;
    await act(async () => hear.click());
    expect(text()).toContain('Speaking…');
    expect(revoke).not.toHaveBeenCalled();
    // the player says it has ended
    const audio = vi.mocked(HTMLMediaElement.prototype.play).mock.contexts[0] as HTMLAudioElement;
    await act(async () => {
      audio.dispatchEvent(new Event('ended'));
    });
    expect(text()).not.toContain('Speaking…');
    expect(revoke).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
    root = createRoot(host);
    vi.restoreAllMocks();
  });
});
