/**
 * The link a tab in a browser keeps to OAIY Desktop: what its person has said, so that the editor may look for the desktop when it
 * opens, and only then.
 *
 * OAIY's own window is given the desktop and looks for it as it always has. A tab is not: on a public address a look at 127.0.0.1
 * or the LAN is a permission prompt for the visitor (Chrome's Local Network Access) and shows the site what is on their network.
 * So a tab looks when its person presses Connect (components/ConnectDesktop.tsx), and, once the desktop has answered, when the
 * editor opens from then on: that is the link, and Disconnect forgets it. An engine address given in Settings (lib/engineEndpoint.ts,
 * a desktop on another machine or another port) is a link too: the person typed where the desktop is.
 *
 * Nothing here contacts anything.
 */
import { DEFAULT_ENGINE_BASE, getEngineBase, subscribeEngineBase } from './engineEndpoint';

const KEY = 'oaiy.desktopLinked';

const listeners = new Set<() => void>();
/** The link as this tab knows it, for a browser that will not keep it (storage blocked). */
let linked = false;

function notify(): void {
  for (const listener of listeners) {
    try {
      listener();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('[desktop-link] listener threw:', e);
    }
  }
}

/** Whether Connect was pressed here and the desktop answered (Disconnect forgets it). */
export function linkedByConnect(): boolean {
  if (linked) return true;
  try {
    return localStorage.getItem(KEY) === '1';
  } catch {
    return false;
  }
}

/** Whether this browser has been linked to a desktop by its person: by Connect, or by an address of its own given to the engine in Settings. */
export function hasSavedDesktopLink(): boolean {
  return getEngineBase() !== DEFAULT_ENGINE_BASE || linkedByConnect();
}

/** The desktop answered Connect: from now on the editor looks for it as it opens. */
export function rememberDesktopLink(): void {
  linked = true;
  try {
    localStorage.setItem(KEY, '1');
  } catch {
    // Storage is blocked: the link lasts as long as the tab does.
  }
  notify();
}

/** Disconnect: the editor asks nothing of this computer when it opens until Connect is pressed again. */
export function forgetDesktopLink(): void {
  linked = false;
  try {
    localStorage.removeItem(KEY);
  } catch {
    // nothing kept, nothing to forget
  }
  notify();
}

/** Called when the link is made, forgotten or its address changed. Returns an unsubscribe. */
export function subscribeDesktopLink(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

// An address given (or taken back) in Settings makes or breaks the link too.
subscribeEngineBase(notify);
