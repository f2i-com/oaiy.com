/**
 * What kind of page this is, read once and without a request (design 3.5): one of OAIY's own windows, the older bot.computer
 * desktop shell, or a tab in a browser.
 *
 * OAIY's own windows and the desktop shell belong to a person who has installed OAIY, so a page there may look for OAIY on this
 * computer as it always has. A tab does not: on a public address every look at 127.0.0.1 or a LAN address is, under Chrome 142+,
 * a permission prompt for the visitor, and it tells whoever reads the page's requests what is on their network. A tab looks only
 * where the person has said (a link they saved) or when they press a button (`looksOnLoad`).
 *
 * Only what the page's own globals say counts. Nothing here looks at the address of a dev server, at `import.meta.env`, or at
 * `isWebBuild()`: the flow editor OAIY shows in its window is the same shimmed build as the one on a web host, so the shim is no
 * sign of a browser tab.
 */

/** Which window the page is in. */
export type HostKind =
  /** OAIY's own window: the desktop gives the page its address and a token before the page runs. */
  | 'oaiy-window'
  /** The bot.computer desktop shell (Tauri): no desktop is given, but the person installed this app on this computer. */
  | 'desktop-shell'
  /** Anything else: a tab in a browser, on a public host, a dev server or a local copy. */
  | 'browser';

export interface Host {
  kind: HostKind;
  /**
   * `<meta name="oaiy-hosted">`: `cookie` when OAIY's own server serves the page and the person is signed in to it, `static` on a
   * static host, and null everywhere else (a dev server, a desktop-embedded page).
   */
  hosted: 'cookie' | 'static' | null;
  /** An automated browser (`navigator.webdriver`): read only by the seams the tests use, never to grant anything to a visitor. */
  automated: boolean;
}

/** The parts of the page's globals that `readHost` looks at. The page's own `globalThis` is one; a test gives its own. */
export interface HostEnv {
  location?: { hostname?: string; protocol?: string };
  navigator?: { webdriver?: boolean };
  document?: { querySelector?: (selector: string) => { getAttribute(name: string): string | null } | null };
  /** `{ origin, token, theme }`, given by OAIY's window before the page runs. */
  __OAIY_DESKTOP__?: unknown;
  /** Tauri's own bridge (the flow editor's browser shim defines a copy of it: see `__OAIY_WEB_SHIM__`). */
  __TAURI_INTERNALS__?: unknown;
  /** Set by the flow editor's browser shim, which is in OAIY's window too. */
  __OAIY_WEB_SHIM__?: unknown;
}

/** The addresses OAIY's windows are served from: `http://<name>.localhost` on Windows, `<scheme>://` elsewhere. */
const OAIY_HOSTNAMES = new Set(['oaiy.localhost', 'oaiyflows.localhost']);
const OAIY_PROTOCOLS = new Set(['oaiy:', 'oaiyflows:']);
const SHELL_HOSTNAMES = new Set(['botcomputer.localhost']);
const SHELL_PROTOCOLS = new Set(['botcomputer:']);

/** Whether OAIY's window has given the page its desktop: an address and a token, both text. */
function desktopGiven(given: unknown): boolean {
  if (!given || typeof given !== 'object') return false;
  const { origin, token } = given as { origin?: unknown; token?: unknown };
  return typeof origin === 'string' && typeof token === 'string' && token !== '';
}

function hostedOf(env: HostEnv): Host['hosted'] {
  try {
    const content = env.document?.querySelector?.('meta[name="oaiy-hosted"]')?.getAttribute('content');
    return content === 'cookie' || content === 'static' ? content : null;
  } catch {
    return null;
  }
}

/** The page's host from its globals: `readHost()` in a page, `readHost(fake)` in a test. Never throws. */
export function readHost(env: HostEnv = globalThis as HostEnv): Host {
  const hostname = env.location?.hostname ?? '';
  const protocol = env.location?.protocol ?? '';
  const ownWindow = desktopGiven(env.__OAIY_DESKTOP__) || OAIY_HOSTNAMES.has(hostname) || OAIY_PROTOCOLS.has(protocol);
  const shell = SHELL_HOSTNAMES.has(hostname) || SHELL_PROTOCOLS.has(protocol) || ('__TAURI_INTERNALS__' in env && env.__OAIY_WEB_SHIM__ !== true);
  return {
    kind: ownWindow ? 'oaiy-window' : shell ? 'desktop-shell' : 'browser',
    hosted: hostedOf(env),
    automated: env.navigator?.webdriver === true,
  };
}

/**
 * Whether a page here may look for OAIY on this computer when it loads, at the addresses OAIY listens on out of the box
 * (127.0.0.1:17972 and :8080). Only OAIY's own windows and the desktop shell: a tab looks at a link its person saved, or
 * when they press a button.
 */
export function looksOnLoad(host: Host): boolean {
  return host.kind !== 'browser';
}
