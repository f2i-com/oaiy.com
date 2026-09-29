import { describe, expect, it, vi } from 'vitest';
import { InstallController, isIosSafari, isStandalone, type InstallEnvironment, type InstallState } from '../../src/pwa/install';

/** The browser's install event, with a prompt that answers as told. */
class PromptEvent extends Event {
  prompt = vi.fn(async () => {});
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
  constructor(outcome: 'accepted' | 'dismissed' = 'accepted') {
    super('beforeinstallprompt', { cancelable: true });
    this.userChoice = Promise.resolve({ outcome });
  }
}

function setup(env: Partial<InstallEnvironment> = {}) {
  const events = new EventTarget();
  const add = vi.spyOn(events, 'addEventListener');
  const controller = new InstallController({ events, standalone: () => false, desktopWindow: false, iosSafari: false, ...env });
  const states: InstallState[] = [];
  controller.onChange((s) => states.push(s));
  return { events, add, controller, states };
}

describe('installing the app from the browser', () => {
  it('offers nothing until the browser says the app can be installed', () => {
    const { controller, states } = setup();
    expect(controller.state).toBe('unavailable');
    expect(states).toEqual([]);
  });

  it('offers the install once the browser says so, and keeps the browser own bar away', () => {
    const { events, controller, states } = setup();
    const event = new PromptEvent();
    events.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(controller.state).toBe('available');
    expect(states).toEqual(['available']);
  });

  it('installs when asked: the browser asks the person, and an accepted answer is installed', async () => {
    const { events, controller, states } = setup();
    const event = new PromptEvent('accepted');
    events.dispatchEvent(event);
    await expect(controller.install()).resolves.toBe('accepted');
    expect(event.prompt).toHaveBeenCalledTimes(1);
    expect(controller.state).toBe('installed');
    // The offer went while it was open, and the answer made it installed.
    expect(states).toEqual(['available', 'unavailable', 'installed']);
  });

  it('cannot use the same event twice', async () => {
    const { events, controller } = setup();
    const event = new PromptEvent('dismissed');
    events.dispatchEvent(event);
    await controller.install();
    await expect(controller.install()).resolves.toBe('unavailable');
    expect(event.prompt).toHaveBeenCalledTimes(1);
  });

  it('takes a dismissal as an answer: nothing is offered until the browser sends a new event', async () => {
    const { events, controller } = setup();
    events.dispatchEvent(new PromptEvent('dismissed'));
    await expect(controller.install()).resolves.toBe('dismissed');
    expect(controller.state).toBe('unavailable');
    events.dispatchEvent(new PromptEvent('accepted'));
    expect(controller.state).toBe('available');
  });

  it('says unavailable, and offers nothing more, when the browser refuses to open its prompt', async () => {
    const { events, controller } = setup();
    const event = new PromptEvent();
    event.prompt.mockRejectedValueOnce(new Error('not from a user gesture'));
    events.dispatchEvent(event);
    await expect(controller.install()).resolves.toBe('unavailable');
    expect(controller.state).toBe('unavailable');
  });

  it('has nothing to install before there is an offer', async () => {
    const { controller } = setup();
    await expect(controller.install()).resolves.toBe('unavailable');
  });

  it('is installed once the browser says it was, however it was done', () => {
    const { events, controller, states } = setup();
    events.dispatchEvent(new PromptEvent());
    events.dispatchEvent(new Event('appinstalled'));
    expect(controller.state).toBe('installed');
    expect(states).toEqual(['available', 'installed']);
  });

  it('offers nothing to a page that already runs as an installed app', async () => {
    const { events, controller } = setup({ standalone: () => true });
    expect(controller.state).toBe('installed');
    events.dispatchEvent(new PromptEvent());
    expect(controller.state).toBe('installed');
    await expect(controller.install()).resolves.toBe('unavailable');
  });

  it('is not part of OAIY own window: it listens to nothing, and offers nothing, not even the iOS sentence', async () => {
    const { events, add, controller, states } = setup({ desktopWindow: true, iosSafari: true });
    expect(add).not.toHaveBeenCalled();
    const event = new PromptEvent();
    events.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(controller.state).toBe('unavailable');
    expect(states).toEqual([]);
    await expect(controller.install()).resolves.toBe('unavailable');
  });

  it('gives iOS Safari, which never sends the event, the by-hand instructions', () => {
    expect(setup({ iosSafari: true }).controller.state).toBe('ios');
    // Already added to the home screen: nothing to say.
    expect(setup({ iosSafari: true, standalone: () => true }).controller.state).toBe('installed');
  });

  it('gives any other browser nothing when it never sends the event', () => {
    expect(setup({ iosSafari: false }).controller.state).toBe('unavailable');
  });

  it('stops listening when disposed', () => {
    const { events, controller } = setup();
    controller.dispose();
    events.dispatchEvent(new PromptEvent());
    expect(controller.state).toBe('unavailable');
  });
});

describe('telling iOS Safari from the rest', () => {
  const SAFARI_IPHONE = 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1';
  const MAC_SAFARI = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15';
  const cases: Array<[string, { userAgent: string; platform?: string; maxTouchPoints?: number }, boolean]> = [
    ['Safari on an iPhone', { userAgent: SAFARI_IPHONE, platform: 'iPhone', maxTouchPoints: 5 }, true],
    ['Safari on an iPad', { userAgent: SAFARI_IPHONE.replace('iPhone', 'iPad').replace('iPhone OS', 'OS'), platform: 'iPad', maxTouchPoints: 5 }, true],
    ['Safari on an iPad that asks for desktop sites', { userAgent: MAC_SAFARI, platform: 'MacIntel', maxTouchPoints: 5 }, true],
    ['Safari on a Mac (no touch screen)', { userAgent: MAC_SAFARI, platform: 'MacIntel', maxTouchPoints: 0 }, false],
    ['Chrome on an iPhone', { userAgent: SAFARI_IPHONE.replace('Version/17.5', 'CriOS/126.0.6478.153'), platform: 'iPhone', maxTouchPoints: 5 }, false],
    ['Firefox on an iPhone', { userAgent: SAFARI_IPHONE.replace('Version/17.5', 'FxiOS/127.0'), platform: 'iPhone', maxTouchPoints: 5 }, false],
    ['Edge on an iPhone', { userAgent: SAFARI_IPHONE.replace('Version/17.5', 'EdgiOS/126.2592.87 Version/17.0'), platform: 'iPhone', maxTouchPoints: 5 }, false],
    ['Chrome on Android', { userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36', platform: 'Linux armv81', maxTouchPoints: 5 }, false],
    ['Chrome on Windows', { userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36', platform: 'Win32', maxTouchPoints: 0 }, false],
    ['a browser that says nothing', { userAgent: '' }, false],
  ];
  it.each(cases)('%s', (_name, nav, expected) => {
    expect(isIosSafari(nav)).toBe(expected);
  });
});

describe('telling an installed app from a tab', () => {
  const media = (...on: string[]) => (query: string) => ({ matches: on.some((mode) => query === `(display-mode: ${mode})`) });
  it('is standalone in the app window display modes', () => {
    expect(isStandalone({ matchMedia: media('standalone') })).toBe(true);
    expect(isStandalone({ matchMedia: media('minimal-ui') })).toBe(true);
    expect(isStandalone({ matchMedia: media('window-controls-overlay') })).toBe(true);
    expect(isStandalone({ matchMedia: media('fullscreen') })).toBe(true);
  });
  it('is standalone when iOS says it was opened from the home screen', () => {
    expect(isStandalone({ matchMedia: media(), navigator: { standalone: true } })).toBe(true);
  });
  it('is not standalone in a browser tab', () => {
    expect(isStandalone({ matchMedia: media('browser') })).toBe(false);
    expect(isStandalone({ matchMedia: media(), navigator: { standalone: false } })).toBe(false);
    expect(isStandalone({})).toBe(false);
  });
  it('is not standalone when the browser cannot say', () => {
    expect(isStandalone({ matchMedia: () => { throw new Error('no'); } })).toBe(false);
  });
});
