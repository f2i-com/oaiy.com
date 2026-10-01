/** The only error metadata carried across the plugin screen boundary: a 1..64
 * character code, up to 1024 message characters and an optional positive safe
 * integer protocol version. Unknown well-formed codes remain plugin-owned. */
export interface PluginErrorDetails {
  code: string;
  message: string;
  version?: number;
}

const PLUGIN_FAILURE = 'The plugin could not complete this action.';
const HOST_FAILURE = 'The desktop could not complete this plugin request.';
const MESSAGE_LIMIT = 1024;

function ownValue(value: unknown, key: string): unknown {
  if (!value || typeof value !== 'object') return undefined;
  // Do not invoke accessors or read prototype properties while projecting an error.
  return Object.getOwnPropertyDescriptor(value, key)?.value;
}

function record(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function message(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback;
  return value.slice(0, MESSAGE_LIMIT).replace(/[\u0000-\u001f\u007f]/g, ' ').trim() || fallback;
}

/** Accept a closed scalar record, then copy it. No plugin object crosses the frame. */
function details(value: unknown, envelopeVersion?: unknown): PluginErrorDetails | undefined {
  if (!record(value)) return undefined;
  if (Reflect.ownKeys(value).some((key) => (key !== 'code' && key !== 'message' && key !== 'version') ||
    !Object.hasOwn(Object.getOwnPropertyDescriptor(value, key)!, 'value'))) return undefined;
  const code = ownValue(value, 'code');
  const text = ownValue(value, 'message');
  const hasVersion = Object.hasOwn(value, 'version');
  const errorVersion = ownValue(value, 'version');
  if (hasVersion && (typeof errorVersion !== 'number' || !Number.isSafeInteger(errorVersion) || errorVersion < 1)) return undefined;
  if (hasVersion && envelopeVersion !== undefined && errorVersion !== envelopeVersion) return undefined;
  const version = envelopeVersion === undefined ? errorVersion : envelopeVersion;
  if (typeof code !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$/.test(code) || typeof text !== 'string' || !message(text, '')) return undefined;
  if (version !== undefined && (typeof version !== 'number' || !Number.isSafeInteger(version) || version < 1)) return undefined;
  return { code, message: message(text, PLUGIN_FAILURE), ...(version === undefined ? {} : { version }) };
}

/** An explicit plugin refusal; code/version describe the plugin's protocol, not a retry policy. */
export class PluginCommandError extends Error {
  readonly code: string;
  readonly version?: number;

  constructor(error: PluginErrorDetails) {
    super(error.message);
    this.name = 'PluginCommandError';
    this.code = error.code;
    if (error.version !== undefined) this.version = error.version;
  }
}

/** Refusals can use the SDK error field or a versioned domain error in data. */
function refusal(envelope: Record<string, unknown>, fallback: string): Error {
  const data = ownValue(envelope, 'data');
  const nestedError = record(data) ? ownValue(data, 'error') : undefined;
  const topError = ownValue(envelope, 'error');
  // An explicitly present but invalid version (including an accessor) cannot
  // silently become an unversioned typed error.
  const dataVersion = record(data) && Object.hasOwn(data, 'version') ? ownValue(data, 'version') ?? null : undefined;
  const typed = details(nestedError, dataVersion) ?? details(topError);
  if (typed) return new PluginCommandError(typed);
  const legacy = typeof topError === 'string' ? topError : typeof nestedError === 'string' ? nestedError
    : ownValue(topError, 'message') ?? ownValue(nestedError, 'message');
  return new Error(message(legacy, fallback));
}

/** Unwrap the gateway/SDK envelopes without turning a refused action into a success. */
export function unwrapPluginCommandResponse(response: unknown): unknown {
  if (!record(response)) throw new Error(HOST_FAILURE);
  if (ownValue(response, 'ok') === false) throw refusal(response, HOST_FAILURE);
  if (ownValue(response, 'ok') !== true) throw new Error(HOST_FAILURE);
  const result = ownValue(response, 'result');
  if (!record(result)) {
    if (ownValue(result, 'ok') === false) throw new Error(PLUGIN_FAILURE);
    return result; // Older plugins can return a bare scalar/array.
  }
  const status = ownValue(result, 'ok');
  if (Object.hasOwn(result, 'ok') && typeof status !== 'boolean') throw new Error(PLUGIN_FAILURE);
  if (status === false) {
    throw refusal(result, PLUGIN_FAILURE);
  }
  // An ok:true data.error remains domain data for plugins that validate it themselves.
  return Object.hasOwn(result, 'data') ? ownValue(result, 'data') : result;
}

/** Retain the old error string and add optional bounded metadata for newer screens. */
export function serializePluginCallError(error: unknown): { error: string; errorDetails?: PluginErrorDetails } {
  const text = message(typeof error === 'string' ? error : ownValue(error, 'message'), 'Host call failed.');
  const version = ownValue(error, 'version');
  const typed = error instanceof PluginCommandError
    ? details({ code: ownValue(error, 'code'), message: text, ...(version === undefined ? {} : { version }) })
    : undefined;
  return { error: text, ...(typed ? { errorDetails: typed } : {}) };
}
