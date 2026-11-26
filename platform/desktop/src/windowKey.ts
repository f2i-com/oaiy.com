/**
 * The key OAIY's own window sends with every request to OAIY's local API (`Authorization: Bearer`).
 *
 * On Windows this window's origin is `http://tauri.localhost`: a name, and any page served by whatever program listens
 * on this computer's port 80 has the same one. So the API takes that origin for OAIY's window only together with a key
 * the desktop hands this page and nobody else (its `window_token` command: refused to every other webview, and to this
 * one while it shows any other page). The key belongs to this run of OAIY: it is kept here in memory, never written
 * down, never put in an address, and never given to a plugin's screen (a frame of this page, which asks through
 * `pluginScreenRpc`; the request itself is made here).
 *
 * In a browser tab (the dev page, the tests) there is no native bridge and no key: requests go as they always did.
 * A window that cannot be given its key still sends its requests: where its origin is one no page can copy (a Mac,
 * Linux) the API answers them as before, and where it is refused for the lack of the key the dashboard says so
 * (`windowKeyProblem`), instead of every page failing in its own words.
 */
import { desktopLaunch } from './nativeLaunch';

/**
 * What the API answers a request that has one of OAIY's own windows' origins and no key (`NAME_ONLY_REFUSAL` in
 * `src-tauri/src/http.rs`: a test there holds the two to each other).
 */
export const KEY_REFUSAL = 'origin not allowed without the token OAIY gives its own windows';

/** What the person is told then, in place of the API's own words. */
export const WINDOW_KEY_ERROR =
  'This window could not show OAIY that it is OAIY’s own, so OAIY is not answering it. Restart OAIY. If it is still like this afterwards, install OAIY again.';

/** How long the desktop is given to hand the key over: a request then goes without it instead of waiting for good. */
const KEY_WAIT_MS = 5000;

type Invoke = (cmd: string, args?: unknown) => Promise<unknown>;

let asked: Promise<string | null> | null = null;
let problem: string | null = null;
const watchers = new Set<() => void>();

function host(): object | undefined {
  return typeof window === 'undefined' ? undefined : window;
}

/** The desktop's native bridge, which only OAIY's own window has. */
function bridge(): { invoke: Invoke } | null {
  const internals = (host() as { __TAURI_INTERNALS__?: { invoke?: unknown } } | undefined)?.__TAURI_INTERNALS__;
  return internals && typeof internals.invoke === 'function' ? (internals as { invoke: Invoke }) : null;
}

/** A key as the desktop makes them: printable, unbroken, and long enough to be one. */
function isKey(value: unknown): value is string {
  return typeof value === 'string' && /^[\x21-\x7e]{16,512}$/.test(value);
}

async function ask(native: { invoke: Invoke }): Promise<string | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const key = await Promise.race([
      native.invoke('window_token', {}),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('the desktop did not answer')), KEY_WAIT_MS);
      }),
    ]);
    return isKey(key) ? key : null;
  } catch {
    // (refused, not there, or too slow: the request goes without a key, and the API's answer decides what is said)
    return null;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/**
 * This window's key: asked of the desktop once and kept for as long as the page lives. `null` in a browser tab, and
 * where the desktop gave none (it is asked again at the next request). Never rejects.
 */
export function windowKey(): Promise<string | null> {
  const native = bridge();
  if (!native) return Promise.resolve(null);
  if (!asked) {
    const mine: Promise<string | null> = ask(native).then((key) => {
      if (key === null && asked === mine) asked = null;
      return key;
    });
    asked = mine;
  }
  return asked;
}

/** Forget the key, so that the next request asks the desktop for it again ("Try again"). */
export function forgetWindowKey(): void {
  asked = null;
}

function say(next: string | null): void {
  if (problem === next) return;
  problem = next;
  for (const tell of watchers) tell();
}

/** Why this window's requests are being refused, in words for the person; `null` while they are not. */
export function windowKeyProblem(): string | null {
  return problem;
}

/** Told when `windowKeyProblem` changes (a store for `useSyncExternalStore`). */
export function watchWindowKeyProblem(tell: () => void): () => void {
  watchers.add(tell);
  return () => {
    watchers.delete(tell);
  };
}

/** Ask for the key afresh; the problem goes once a request made with it is answered. */
export async function retryWindowKey(): Promise<void> {
  forgetWindowKey();
  if ((await windowKey()) !== null) say(null);
}

/** Whether `url` is this window's own API: the only place its key is ever sent. */
function isOwnApi(url: string): boolean {
  try {
    return new URL(url).origin === new URL(desktopLaunch(host()).apiBase).origin;
  } catch {
    return false;
  }
}

function plain(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};
  if (typeof Headers !== 'undefined' && headers instanceof Headers) {
    const out: Record<string, string> = {};
    headers.forEach((value, name) => {
      out[name] = value;
    });
    return out;
  }
  if (Array.isArray(headers)) return Object.fromEntries(headers);
  return { ...(headers as Record<string, string>) };
}

/** `init` with the key added: the caller's own headers kept, any `Authorization` of its own replaced. */
export function withWindowKey(init: RequestInit | undefined, key: string): RequestInit {
  const headers = plain(init?.headers);
  for (const name of Object.keys(headers)) {
    if (name.toLowerCase() === 'authorization') delete headers[name];
  }
  return { ...init, headers: { ...headers, Authorization: `Bearer ${key}` } };
}

/** Whether the API refused `response` for the lack of this window's key. */
async function refusedForTheKey(response: Response): Promise<boolean> {
  if (response.status !== 403 || typeof response.clone !== 'function') return false;
  try {
    const body: unknown = await response.clone().json();
    return (body as { error?: unknown } | null)?.error === KEY_REFUSAL;
  } catch {
    return false;
  }
}

/**
 * `fetch`, for a request to OAIY's local API: with this window's key when it has one. Every request the dashboard
 * makes to the API goes through here (a test reads the sources for a `fetch` that does not). A request to any other
 * address is passed on untouched: the key goes nowhere else.
 */
export function windowFetch(url: string, init?: RequestInit): Promise<Response> {
  const bare = () => (init === undefined ? fetch(url) : fetch(url, init));
  // (a browser tab, or another address: the request as it always was, made at once)
  if (!bridge() || !isOwnApi(url)) return bare();
  return withTheKey(url, init, bare);
}

async function withTheKey(url: string, init: RequestInit | undefined, bare: () => Promise<Response>): Promise<Response> {
  const key = await windowKey();
  const response = await (key !== null ? fetch(url, withWindowKey(init, key)) : bare());
  if (await refusedForTheKey(response)) {
    // (a key that was sent and not taken is no key: the next request asks the desktop again)
    forgetWindowKey();
    say(WINDOW_KEY_ERROR);
  } else if (key !== null) {
    say(null);
  }
  return response;
}
