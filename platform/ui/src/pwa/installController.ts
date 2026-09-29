/**
 * "Install app": offered only when the browser offers it.
 *
 * Chrome and Edge fire `beforeinstallprompt` when the page can be installed (the manifest and
 * the worker are right, and it is not installed yet). The event is kept, its default (the
 * browser's own mini-bar) is cancelled, and the editor shows a button of its own that plays it.
 * The event can be played once. If the person dismisses it the browser fires a new one later,
 * and the button comes back only then.
 *
 * Never offered:
 *   - inside OAIY's own window (`__OAIY_DESKTOP__`): it is an app already;
 *   - in a window that is the installed app (display-mode: standalone, or iOS's navigator.standalone).
 *
 * The listener has to be on the page before the browser fires the event, which can be soon
 * after load: `start` is called from main.tsx, not from a component.
 */

/** The install prompt event (not in the DOM typings). */
export interface InstallPromptEvent {
  preventDefault(): void;
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

export interface InstallState {
  /** The browser has offered installation and the person has not answered. */
  available: boolean;
  /** The app was installed from here. */
  installed: boolean;
}

export interface InstallEnv {
  addEventListener(type: 'beforeinstallprompt', listener: (event: InstallPromptEvent) => void): void;
  addEventListener(type: 'appinstalled', listener: () => void): void;
  /** Whether this window is OAIY's own. */
  inOaiyWindow: boolean;
  /** Whether this window is the installed app. */
  standalone: boolean;
}

export type InstallOutcome = 'accepted' | 'dismissed' | 'unavailable';

export interface InstallController {
  getState(): InstallState;
  subscribe(listener: () => void): () => void;
  /** Start listening. Does nothing where installing is never offered. */
  start(env: InstallEnv): void;
  /** Show the browser's install dialog. */
  prompt(): Promise<InstallOutcome>;
}

export function createInstallController(): InstallController {
  let state: InstallState = { available: false, installed: false };
  let stashed: InstallPromptEvent | null = null;
  let started = false;
  const listeners = new Set<() => void>();
  const set = (next: InstallState) => {
    if (next.available === state.available && next.installed === state.installed) return;
    state = next;
    for (const listener of [...listeners]) listener();
  };

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    start(env) {
      if (started || env.inOaiyWindow || env.standalone) return;
      started = true;
      env.addEventListener('beforeinstallprompt', (event) => {
        event.preventDefault();
        stashed = event;
        set({ available: true, installed: false });
      });
      env.addEventListener('appinstalled', () => {
        stashed = null;
        set({ available: false, installed: true });
      });
    },
    async prompt() {
      const event = stashed;
      if (!event) return 'unavailable';
      // One play per event: the button goes until the browser offers again.
      stashed = null;
      set({ available: false, installed: state.installed });
      try {
        await event.prompt();
        return (await event.userChoice).outcome;
      } catch {
        return 'unavailable';
      }
    },
  };
}
