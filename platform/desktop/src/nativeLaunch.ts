/** The normal desktop and plain-browser development endpoint stays unchanged. */
export const DEFAULT_API_BASE = 'http://127.0.0.1:17972';

export interface DesktopLaunch {
  readonly apiBase: string;
  readonly isolated: boolean;
}

const DEFAULT_LAUNCH: DesktopLaunch = Object.freeze({ apiBase: DEFAULT_API_BASE, isolated: false });
const CONTEXT_KEY = '__OAIY_DESKTOP_LAUNCH__';
const CONTEXT_ERROR = 'The isolated OAIY desktop launch context is invalid.';

/**
 * Native startup installs this context before the dashboard's modules run. It
 * contains a port only, never a URL or credential, and cannot be changed by the
 * page. Reading it synchronously keeps every imported API_BASE consistent before
 * React effects, plugin asset requests, and health polling can start.
 *
 * Browser URLs, query parameters and local storage do not select an endpoint.
 * A browser without Tauri's callable bridge ignores even a same-named property.
 * An invalid context in a native window fails closed rather than contacting the
 * ordinary installation by falling back to its port.
 */
export function desktopLaunch(host?: object): DesktopLaunch {
  if (!host) return DEFAULT_LAUNCH;
  const internals = (host as { __TAURI_INTERNALS__?: { invoke?: unknown } }).__TAURI_INTERNALS__;
  if (!internals || typeof internals.invoke !== 'function') return DEFAULT_LAUNCH;

  const descriptor = Object.getOwnPropertyDescriptor(host, CONTEXT_KEY);
  if (!descriptor) return DEFAULT_LAUNCH;
  if (!('value' in descriptor) || descriptor.writable || descriptor.configurable) {
    throw new Error(CONTEXT_ERROR);
  }

  const context: unknown = descriptor.value;
  if (!context || typeof context !== 'object' || !Object.isFrozen(context)) {
    throw new Error(CONTEXT_ERROR);
  }
  const prototype = Object.getPrototypeOf(context);
  const keys = Reflect.ownKeys(context);
  if ((prototype !== Object.prototype && prototype !== null)
    || keys.length !== 3
    || !['version', 'isolated', 'apiPort'].every((key) => keys.includes(key))) {
    throw new Error(CONTEXT_ERROR);
  }
  const fields = Object.getOwnPropertyDescriptors(context);
  if (Object.values(fields).some((field) => !('value' in field))) {
    throw new Error(CONTEXT_ERROR);
  }
  const { value: version } = fields.version;
  const { value: isolated } = fields.isolated;
  const { value: apiPort } = fields.apiPort;
  if (version !== 1 || isolated !== true || typeof apiPort !== 'number'
    || !Number.isInteger(apiPort) || apiPort < 1024 || apiPort > 65535
    || apiPort === 17972 || apiPort === 17973) {
    throw new Error(CONTEXT_ERROR);
  }
  return Object.freeze({ apiBase: `http://127.0.0.1:${apiPort}`, isolated: true });
}

/** Compatibility helper for callers that only need the endpoint. */
export function desktopApiBase(host?: object): string {
  return desktopLaunch(host).apiBase;
}
