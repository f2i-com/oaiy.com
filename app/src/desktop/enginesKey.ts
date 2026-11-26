/**
 * The key OAIY Desktop gives the Agent for the gateway of the engines it runs (port 8080, or the port the desktop says
 * they moved to), sent as `Authorization: Bearer` with every request the page makes there.
 *
 * Those engines answer a page at one of this computer's own names only with it: a name under `localhost`, which is
 * what the Agent's window in OAIY Desktop has on Windows (`http://oaiy.localhost`) and the OAIY Agent app
 * (`http://oaiyagent.localhost`), or a loopback address and port (the Agent served by `npm start`,
 * `http://localhost:5317`). Every browser sends such a name to this computer, so a page served by whatever program
 * listens on its port 80, opened as `http://oaiy.localhost/...` in the person's ordinary browser, has the Agent's
 * origin exactly; the key is what tells the two apart.
 *
 * Where it comes from: OAIY's own window gives it in the page's first line (`window.__OAIY_DESKTOP__.enginesKey`), and
 * the desktop tells a page paired with it when the page asks where the engines are (`GET /api/engines`, `gatewayKey`).
 * It is kept in memory alone, sent to the engines' gateway on this computer and nowhere else, and only on a request
 * that carries no `Authorization` of its own: an API key the person gave for OAIY goes as it always did.
 */

/** A key as the desktop makes them: printable, unbroken, and long enough to be one. */
export function isEnginesKey(value: unknown): value is string {
  return typeof value === 'string' && /^[\x21-\x7e]{16,512}$/.test(value);
}

/** The engines' gateway's usual address on this computer. */
const USUAL: readonly string[] = ['http://127.0.0.1:8080', 'http://localhost:8080'];

let key: string | null = null;
let gateways = new Set<string>(USUAL);
let installed = false;

/** Whether an origin is one of this computer's loopback addresses (where OAIY Desktop's engines are). */
const loopback = (origin: string): boolean => /^http:\/\/(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$/i.test(origin);

/** The engines' gateway is (also) at `url`'s origin, as the desktop says: taken where it is a loopback address alone. */
export function addEnginesGateway(url: string | null | undefined): void {
  if (!url) return;
  try {
    const origin = new URL(url).origin;
    if (loopback(origin)) gateways.add(origin);
  } catch {
    /* not an address */
  }
}

/** The URL a request is for, from what `fetch` was given. */
function urlOf(input: RequestInfo | URL): string {
  if (typeof input === 'string') return input;
  if (input instanceof URL) return input.href;
  return input.url;
}

/**
 * What `fetch(input, init)` is sent with: `init` with the key added where the request goes to the engines' gateway and
 * carries no `Authorization` of its own, and `init` as it was for any other request (or when there is no key).
 */
export function withEnginesKey(input: RequestInfo | URL, init?: RequestInit): RequestInit | undefined {
  if (!key) return init;
  let origin: string;
  try {
    // (a relative address is the page's own: never the gateway's)
    origin = new URL(urlOf(input)).origin;
  } catch {
    return init;
  }
  if (!gateways.has(origin)) return init;
  const own = init?.headers ?? (typeof Request !== 'undefined' && input instanceof Request ? input.headers : undefined);
  const headers = new Headers(own);
  if (headers.has('authorization')) return init;
  headers.set('Authorization', `Bearer ${key}`);
  return { ...init, headers };
}

/**
 * Keep `value` as the engines' key, when it is one, and from then on send it with every request this page makes to the
 * engines' gateway (`host.fetch` is wrapped, once). Anything that is not a key is no key: one kept before stays.
 */
export function useEnginesKey(value: unknown, host: { fetch: typeof fetch } = globalThis): void {
  if (!isEnginesKey(value)) return;
  key = value;
  if (installed) return;
  installed = true;
  const original = host.fetch.bind(host);
  host.fetch = ((input: RequestInfo | URL, init?: RequestInit) => original(input, withEnginesKey(input, init))) as typeof fetch;
}

/** For the tests: no key, the usual gateway, and the next `useEnginesKey` wraps `fetch` again. */
export function forgetEnginesKeyForTests(): void {
  key = null;
  gateways = new Set(USUAL);
  installed = false;
}
