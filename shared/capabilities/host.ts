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
  location?: { hostname?: string; protocol?: string; port?: string; href?: string };
  navigator?: { webdriver?: boolean };
  document?: { querySelector?: (selector: string) => { getAttribute(name: string): string | null } | null };
  /** `{ origin, token, theme }`, given by OAIY's window before the page runs. */
  __OAIY_DESKTOP__?: unknown;
  /** Tauri's own bridge (the flow editor's browser shim defines a copy of it: see `__OAIY_WEB_SHIM__`). */
  __TAURI_INTERNALS__?: unknown;
  /** Set by the flow editor's browser shim, which is in OAIY's window too. */
  __OAIY_WEB_SHIM__?: unknown;
}

/**
 * The origins OAIY serves its own pages from, EXACTLY: `http://<name>.localhost` on Windows (no port: the desktop maps the scheme, it
 * does not listen), `<scheme>://localhost` elsewhere. Not "any port of that host name": a browser resolves every `*.localhost` name to
 * this computer, so a page at `http://oaiy.localhost:8000` is whatever answers on port 8000 of it (a link in a model's reply can open
 * one), and it is not OAIY's window.
 */
const OWN_ORIGINS: ReadonlyArray<readonly [protocol: string, hostname: string, kind: 'oaiy-window' | 'desktop-shell']> = [
  ['http:', 'oaiy.localhost', 'oaiy-window'],
  ['oaiy:', 'localhost', 'oaiy-window'],
  ['http:', 'oaiyflows.localhost', 'oaiy-window'],
  ['oaiyflows:', 'localhost', 'oaiy-window'],
  ['http:', 'botcomputer.localhost', 'desktop-shell'],
  ['botcomputer:', 'localhost', 'desktop-shell'],
];

/** Which of OAIY's own origins the page is at, or null. No port, and no user name or password in the address. */
function ownOrigin(location: HostEnv['location']): 'oaiy-window' | 'desktop-shell' | null {
  const hostname = location?.hostname ?? '';
  const protocol = location?.protocol ?? '';
  if ((location?.port ?? '') !== '') return null;
  if (typeof location?.href === 'string') {
    try {
      const url = new URL(location.href);
      if (url.username !== '' || url.password !== '') return null;
    } catch {
      return null;
    }
  }
  return OWN_ORIGINS.find(([p, h]) => p === protocol && h === hostname)?.[2] ?? null;
}

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
  const at = ownOrigin(env.location);
  const ownWindow = desktopGiven(env.__OAIY_DESKTOP__) || at === 'oaiy-window';
  const shell = at === 'desktop-shell' || ('__TAURI_INTERNALS__' in env && env.__OAIY_WEB_SHIM__ !== true);
  return Object.freeze({
    kind: ownWindow ? 'oaiy-window' : shell ? 'desktop-shell' : 'browser',
    hosted: hostedOf(env),
    automated: env.navigator?.webdriver === true,
  });
}
/**
 * Whether a page here may look for OAIY on this computer when it loads, at the addresses OAIY listens on out of the box
 * (127.0.0.1:17972 and :8080). Only OAIY's own windows and the desktop shell: a tab looks at a link its person saved, or
 * when they press a button.
 */
export function looksOnLoad(host: Host): boolean {
  return host.kind !== 'browser';
}

let page: Host | null = null;

/**
 * The window THIS page is in, read once, the first time anything asks (the apps ask as they load, before any flow, model-written page
 * or other script of the person's runs), and the same frozen answer after. A script that runs later and sets `__OAIY_DESKTOP__` or
 * `__TAURI_INTERNALS__` does not turn a tab into one of OAIY's windows, which is what would make the page look for OAIY on this
 * computer as it loads. `readHost(env)` is the pure reading, for tests.
 */
export function pageHost(): Host {
  return (page ??= readHost());
}

/** Read the page again (for the tests that give a page other globals; nothing in an app calls this). */
export function forgetPageHost(): void {
  page = null;
}
