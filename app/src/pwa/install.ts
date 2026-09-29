/**
 * Installing the app from the browser.
 *
 * Chromium browsers say when the page may be installed (`beforeinstallprompt`)
 * and that it was (`appinstalled`); the page keeps the event and shows its own
 * "Install app" button, which uses it once. Safari on iOS never fires it: the
 * only way there is Share, then Add to Home Screen, so it gets that one
 * sentence. Every other browser gets nothing, rather than instructions that
 * may be wrong.
 *
 * Nothing is offered inside OAIY's own window (the desktop app serves the page
 * itself), nor to a page that already runs as an installed app.
 *
 * The controller takes its surroundings as arguments so that it can be tried
 * with stubs; `startInstall` gives it the real ones. It must be started as the
 * page loads, before anything is awaited, or the event has come and gone.
 */

/** What Chromium sends when the app can be installed (not in the DOM typings). */
export interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform?: string }>;
}

/**
 * - `available`: the browser will install the app when asked.
 * - `ios`: Safari on iOS, where it is done by hand from the Share menu.
 * - `installed`: it runs as an installed app, or was just installed.
 * - `unavailable`: nothing to offer (yet).
 */
export type InstallState = 'available' | 'ios' | 'installed' | 'unavailable';

/** What asking the browser came to. */
export type InstallResult = 'accepted' | 'dismissed' | 'unavailable';

export interface InstallEnvironment {
  /** Where `beforeinstallprompt` and `appinstalled` arrive: the window. */
  events: Pick<EventTarget, 'addEventListener' | 'removeEventListener'>;
  /** The page already runs as an installed app (its own window). */
  standalone(): boolean;
  /** OAIY's own window, or another desktop shell: the app is not installed from there. */
  desktopWindow: boolean;
  /** Safari on iOS (or iPadOS). */
  iosSafari: boolean;
}

export class InstallController {
  private deferred: BeforeInstallPromptEvent | null = null;
  private installed = false;
  private readonly listeners = new Set<(state: InstallState) => void>();
  private readonly onPrompt = (event: Event): void => {
    // Keep it for the button, and keep the browser's own bar from showing as well.
    event.preventDefault();
    this.deferred = event as BeforeInstallPromptEvent;
    this.changed();
  };
  private readonly onInstalled = (): void => {
    this.deferred = null;
    this.installed = true;
    this.changed();
  };

  constructor(private readonly env: InstallEnvironment) {
    // In OAIY's window the page is not offered for installation and is not to hear about it.
    if (env.desktopWindow) return;
    env.events.addEventListener('beforeinstallprompt', this.onPrompt);
    env.events.addEventListener('appinstalled', this.onInstalled);
  }

  get state(): InstallState {
    if (this.env.desktopWindow) return 'unavailable';
    if (this.installed || this.env.standalone()) return 'installed';
    if (this.deferred) return 'available';
    if (this.env.iosSafari) return 'ios';
    return 'unavailable';
  }

  /** Called with the new state whenever it changes; returns the function that stops it. */
  onChange(listener: (state: InstallState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Ask the browser to install the app. The browser lets an event be used once,
   * so the offer is gone afterwards, whatever the answer; a dismissed one comes
   * back only when the browser sends a new event.
   */
  async install(): Promise<InstallResult> {
    const event = this.deferred;
    if (!event || this.state !== 'available') return 'unavailable';
    this.deferred = null;
    this.changed();
    try {
      await event.prompt();
      const choice = await event.userChoice;
      if (choice.outcome !== 'accepted') return 'dismissed';
      this.installed = true;
      this.changed();
      return 'accepted';
    } catch {
      return 'unavailable';
    }
  }

  dispose(): void {
    this.env.events.removeEventListener('beforeinstallprompt', this.onPrompt);
    this.env.events.removeEventListener('appinstalled', this.onInstalled);
    this.listeners.clear();
  }

  private changed(): void {
    const state = this.state;
    for (const listener of [...this.listeners]) listener(state);
  }
}

/** Safari on iOS or iPadOS: Chrome, Firefox, Edge and the others on those systems are not it. */
export function isIosSafari(nav: { userAgent: string; platform?: string; maxTouchPoints?: number }): boolean {
  const ua = nav.userAgent;
  // An iPad asking for desktop sites says it is a Mac, but a Mac has no touch screen.
  const ios = /iPhone|iPad|iPod/.test(ua) || (nav.platform === 'MacIntel' && (nav.maxTouchPoints ?? 0) > 1);
  if (!ios) return false;
  // Other browsers on iOS put their own name in the user agent beside Safari's.
  return /Safari\//.test(ua) && !/CriOS|FxiOS|EdgiOS|OPiOS|OPT\/|GSA\/|DuckDuckGo|YaBrowser|Brave|UCBrowser|SamsungBrowser/.test(ua);
}

const DISPLAY_MODES = ['standalone', 'fullscreen', 'minimal-ui', 'window-controls-overlay'];

/** The page runs as an installed app: in a window of its own, not in a browser tab. */
export function isStandalone(win: { matchMedia?: (query: string) => { matches: boolean }; navigator?: { standalone?: boolean } }): boolean {
  if (win.navigator?.standalone === true) return true;
  try {
    return DISPLAY_MODES.some((mode) => win.matchMedia?.(`(display-mode: ${mode})`).matches === true);
  } catch {
    return false;
  }
}

/** The controller for this page. `desktopWindow`: OAIY's own window (or the desktop shell), where nothing is offered. */
export function startInstall(win: Window, desktopWindow: boolean): InstallController {
  return new InstallController({
    events: win,
    standalone: () => isStandalone(win as unknown as Parameters<typeof isStandalone>[0]),
    desktopWindow,
    iosSafari: isIosSafari(win.navigator),
  });
}
