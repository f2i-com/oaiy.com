/**
 * The port protocol between an app (the Agent, the flow editor) and the providers origin (design 3.2), and the checks that
 * make it safe to speak.
 *
 * An app posts `hello` to the providers frame once, with a MessageChannel port; everything after goes over the port. The
 * frame accepts a `hello` only from an origin in its compiled list (`parseAppOrigins`) and only from its parent. The
 * operations an app may call are the ones in `OPS` and no others. There is no operation that returns, edits or redirects a
 * key, that adds, edits or deletes a provider, or that changes a base URL, a passphrase or the mode: those exist only in the
 * top-level providers window, where the address bar says whose form it is.
 *
 * Nothing here trusts a message. `parseRequest` builds a NEW object from the fields it knows and leaves the rest behind, so a
 * field the protocol does not define (a `baseUrl`, a `headers.authorization`, a `key`) never reaches the code that acts on
 * the request. Plain TypeScript with no DOM types beyond the message shapes: the holder, an app's client and the tests all
 * import it.
 */
import { isLoopbackHost } from '../providers/errors';
import type { ProviderSummary } from '../providers/types';

export const PROTOCOL_VERSION = 1;

/** Every operation an app may call. There is no other; a name outside this list is refused with `unknown-op`. */
export const OPS = ['abort', 'fetch', 'list', 'models', 'probe', 'setModel', 'status', 'test', 'ui.open'] as const;
export type Op = (typeof OPS)[number];

const OP_SET: ReadonlySet<string> = new Set(OPS);

// --- Limits -----------------------------------------------------------------

/**
 * The most a request body may EVER hold: the ceiling the checker enforces and a record's own limit cannot pass. What a request may hold
 * in practice is `DEFAULT_MAX_BODY_BYTES`, or what the record says (`limits.maxBodyBytes`), because a request that carries megabytes is
 * money, and the holder bounds volume, not cost.
 */
export const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** What a request body may hold when the record says nothing (1 MiB: a long conversation, an image of moderate size). */
export const DEFAULT_MAX_BODY_BYTES = 1024 * 1024;
/** The default number of bytes an app may send in an hour, across its requests. */
export const DEFAULT_BYTES_PER_HOUR = 64 * 1024 * 1024;
/** The embedded modal's Test and Load-models buttons call the provider too: they are an "app" of their own with their own, smaller hour. */
export const MODAL_APP = 'modal';
export const DEFAULT_MODAL_BUDGET_PER_HOUR = 60;
/** How long a request may take when it says nothing, and the least and most it may ask for (ms). */
export const DEFAULT_TIMEOUT_MS = 120_000;
export const MIN_TIMEOUT_MS = 1_000;
export const MAX_TIMEOUT_MS = 600_000;
/** How many requests one app connection may have open at once. */
export const MAX_IN_FLIGHT = 8;
/**
 * How much a connection may ask of the holder itself, apart from the provider (a compromised app can flood the port, and the holder's
 * cost of answering is what it must bound): operations being worked on at once (a further one is refused `busy` at once, not queued), a
 * rate for every operation of any kind (a burst, then a sustained number a second), connections an app may hold (the least recently
 * active is closed to make room), and how long a connection may be silent before it is closed (a MessagePort has no close event, so the
 * holder cannot tell a page that has gone from one that is quiet: it closes the quiet ones, and tells them, `{t:'closed'}`).
 */
export const MAX_PENDING_OPS = 16;
export const OPS_BURST = 100;
export const OPS_PER_SECOND = 50;
export const MAX_CONNECTIONS_PER_APP = 16;
export const IDLE_CLOSE_MS = 15 * 60 * 1000;
/** Refusals a connection is answered for in a second; past that a flood is dropped without an answer, which is cheaper still. */
export const REFUSALS_ANSWERED_PER_SECOND = 100;
/** The default number of provider requests an app may make in an hour. */
export const DEFAULT_BUDGET_PER_HOUR = 600;
export const BUDGET_WINDOW_MS = 60 * 60 * 1000;

const ID_PATTERN = /^[A-Za-z0-9_.:-]{1,96}$/;
const MODEL_MAX = 200;
const CONTROL = /[\u0000-\u001f\u007f]/;

// --- The handshake ----------------------------------------------------------

export interface HelloMessage {
  op: 'hello';
  v: number;
}

/** A `hello`: `{op:'hello', v}` with a positive integer version. Whatever else it carries (an `app` name, an `origin`) is not read: who is asking is what the browser says (`event.origin`). */
export function parseHello(data: unknown): HelloMessage | null {
  if (!isPlain(data) || data.op !== 'hello') return null;
  const v = data.v;
  if (typeof v !== 'number' || !Number.isInteger(v) || v < 1 || v > 1_000_000) return null;
  return { op: 'hello', v };
}

export interface HelloReply {
  t: 'hello';
  v: number;
  ops: readonly string[];
  /** Which app the holder took the page for, from its origin. */
  app: string;
}

// --- Which pages may speak --------------------------------------------------

/**
 * The origins the holder answers, from its `<meta name="oaiy-apps" content="agent=https://agent.example flows=https://flows.example">`
 * (written when the folders are assembled, never hard-coded). An entry must be an exact origin (`https:`, or `http:` on this
 * computer, for tests); a malformed one makes the whole list empty, so a mistake fails closed.
 */
export function parseAppOrigins(content: string | null | undefined): Map<string, string> {
  const origins = new Map<string, string>();
  if (typeof content !== 'string') return origins;
  for (const part of content.split(/\s+/).filter(Boolean)) {
    const eq = part.indexOf('=');
    const app = part.slice(0, eq);
    const origin = part.slice(eq + 1);
    if (eq <= 0 || !/^[a-z][a-z0-9-]{0,15}$/.test(app) || !isExactOrigin(origin) || origins.has(origin)) return new Map();
    origins.set(origin, app);
  }
  return origins;
}

/** Whether `text` is an origin exactly as a browser writes one: a scheme, a host and maybe a port, and nothing else; `https:`, or `http:` for this computer only. */
export function isExactOrigin(text: string): boolean {
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return false;
  }
  if (url.origin !== text || url.origin === 'null') return false;
  // A host name is letters, digits, dots and hyphens (or an IPv6 address in brackets): no `*`, no `%`, nothing a pattern could mean.
  if (!/^(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?|\[[0-9a-f:]+\])$/.test(url.hostname)) return false;
  if (url.protocol === 'https:') return true;
  return url.protocol === 'http:' && isLoopbackHost(url.hostname);
}

/** The app an origin belongs to, or null: the exact-match check the holder makes before it reads a message. */
export function appForOrigin(origins: ReadonlyMap<string, string>, origin: string): string | null {
  if (typeof origin !== 'string' || origin === 'null' || origin === '') return null;
  return origins.get(origin) ?? null;
}

// --- Requests ---------------------------------------------------------------

export type OpenTarget = 'pick' | 'manage';

/** A request body: text, or bytes. */
export type RequestBody = string | ArrayBuffer | ArrayBufferView;

export interface FetchRequest {
  op: 'fetch';
  provider: string;
  path: string;
  method: 'GET' | 'POST';
  query?: Array<[string, string]>;
  headers?: Array<[string, string]>;
  body?: RequestBody;
  timeoutMs: number;
}

export type PortRequest =
  | { op: 'list' }
  | { op: 'status' }
  | FetchRequest
  | { op: 'abort'; target: number }
  | { op: 'models'; provider: string }
  | { op: 'test'; provider: string }
  | { op: 'probe'; provider: string }
  | { op: 'setModel'; provider: string; model: string }
  | { op: 'ui.open'; target: OpenTarget };

export type ParseFailureCode = 'bad-request' | 'unknown-op';

export type ParsedRequest =
  | { ok: true; id: number; request: PortRequest }
  | { ok: false; id: number | null; code: ParseFailureCode; message: string };

function isPlain(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isId(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 0x7fffffff;
}

function providerId(value: unknown): string | null {
  return typeof value === 'string' && ID_PATTERN.test(value) ? value : null;
}

function bodyLength(body: RequestBody): number {
  return typeof body === 'string' ? body.length * 3 : body.byteLength; // UTF-8 is at most three bytes per UTF-16 unit
}

function pairs(value: unknown, max: number, nameOk: (name: string) => boolean): Array<[string, string]> | null {
  if (!Array.isArray(value) || value.length > max) return null;
  const out: Array<[string, string]> = [];
  for (const item of value) {
    if (!Array.isArray(item) || item.length !== 2) return null;
    const [name, val] = item as unknown[];
    if (typeof name !== 'string' || typeof val !== 'string' || !nameOk(name) || name.length > 64 || val.length > 1024) return null;
    out.push([name, val]);
  }
  return out;
}

/**
 * A request from a port, checked. Returns a NEW request object with only the fields the protocol defines, so a field it does
 * not define never travels on. `id` is read first so a refusal can name the request it answers.
 */
export function parseRequest(data: unknown): ParsedRequest {
  if (!isPlain(data)) return { ok: false, id: null, code: 'bad-request', message: 'A request is an object.' };
  const id = isId(data.id) ? data.id : null;
  const refuse = (code: ParseFailureCode, message: string): ParsedRequest => ({ ok: false, id, code, message });
  if (id === null) return refuse('bad-request', 'A request has a numeric id.');
  const op = data.op;
  if (typeof op !== 'string' || !OP_SET.has(op)) return refuse('unknown-op', 'There is no such operation.');

  switch (op as Op) {
    case 'list':
      return { ok: true, id, request: { op: 'list' } };
    case 'status':
      return { ok: true, id, request: { op: 'status' } };
    case 'abort': {
      if (!isId(data.target)) return refuse('bad-request', 'abort names the request to stop.');
      return { ok: true, id, request: { op: 'abort', target: data.target } };
    }
    case 'models':
    case 'test':
    case 'probe': {
      const provider = providerId(data.provider);
      if (provider === null) return refuse('bad-request', 'The provider is not named.');
      return { ok: true, id, request: { op, provider } as PortRequest };
    }
    case 'setModel': {
      const provider = providerId(data.provider);
      const model = data.model;
      if (provider === null || typeof model !== 'string' || model.length === 0 || model.length > MODEL_MAX || CONTROL.test(model) || model !== model.trim()) {
        return refuse('bad-request', 'setModel names a provider and a model.');
      }
      return { ok: true, id, request: { op: 'setModel', provider, model } };
    }
    case 'ui.open': {
      if (data.target !== 'pick' && data.target !== 'manage') return refuse('bad-request', 'ui.open opens pick or manage.');
      return { ok: true, id, request: { op: 'ui.open', target: data.target } };
    }
    case 'fetch': {
      const provider = providerId(data.provider);
      if (provider === null) return refuse('bad-request', 'The provider is not named.');
      if (typeof data.path !== 'string' || data.path.length > 64) return refuse('bad-request', 'The path is not a short string.');
      if (data.method !== 'GET' && data.method !== 'POST') return refuse('bad-request', 'The method is GET or POST.');
      const request: FetchRequest = { op: 'fetch', provider, path: data.path, method: data.method, timeoutMs: DEFAULT_TIMEOUT_MS };
      if (data.query !== undefined) {
        const query = pairs(data.query, 16, (n) => n.length > 0);
        if (query === null) return refuse('bad-request', 'The query is not a short list of name and value pairs.');
        request.query = query;
      }
      if (data.headers !== undefined) {
        // Names are checked against the allowed ones later (recordHeaders); here it is only a small list of pairs.
        const headers = pairs(data.headers, 16, (n) => n.length > 0);
        if (headers === null) return refuse('bad-request', 'The headers are not a short list of name and value pairs.');
        request.headers = headers;
      }
      if (data.body !== undefined && data.body !== null) {
        const body = data.body;
        const isBytes = body instanceof ArrayBuffer || ArrayBuffer.isView(body);
        if (typeof body !== 'string' && !isBytes) return refuse('bad-request', 'The body is text or bytes.');
        if (bodyLength(body as RequestBody) > MAX_BODY_BYTES) return refuse('bad-request', 'The body is too large.');
        request.body = body as RequestBody;
      }
      if (data.timeoutMs !== undefined) {
        const t = data.timeoutMs;
        if (typeof t !== 'number' || !Number.isFinite(t)) return refuse('bad-request', 'timeoutMs is a number.');
        request.timeoutMs = Math.min(MAX_TIMEOUT_MS, Math.max(MIN_TIMEOUT_MS, Math.floor(t)));
      }
      return { ok: true, id, request };
    }
  }
}

// --- Answers ----------------------------------------------------------------

export type ErrorCode =
  | ParseFailureCode
  | 'unknown-provider'
  | 'unknown-model'
  | 'bad-path'
  | 'bad-method'
  | 'bad-query'
  | 'bad-base'
  | 'bad-url'
  | 'bad-headers'
  | 'too-many'
  | 'busy'
  | 'too-large'
  | 'bad-body'
  | 'budget'
  | 'locked'
  | 'key-unreadable'
  | 'timeout'
  | 'aborted'
  | 'network'
  | 'redirect'
  | 'internal';

export interface ErrorBody {
  code: ErrorCode;
  message: string;
  /** Set for `budget`: when a request would be answered again. */
  retryAfterMs?: number;
}

/** What `status` says: the vault, and the on-device engine (none yet). No secret. */
export interface StatusBody {
  mode: 'device' | 'passphrase';
  locked: boolean;
  engine: { state: 'none' | 'idle' | 'loading' | 'ready' | 'error'; model: string | null; progress: number | null };
}

export type Reply = { id: number; ok: true; result: unknown } | { id: number; ok: false; error: ErrorBody };

/** The stream of a `fetch`: the status line, chunks of the body as they arrive, then `end` (or `error`). */
export type StreamEvent =
  | { id: number; t: 'head'; status: number; statusText: string; headers: Array<[string, string]> }
  | { id: number; t: 'chunk'; bytes: ArrayBuffer }
  | { id: number; t: 'end' }
  | { id: number; t: 'error'; error: ErrorBody };

/** A stream event before it is given the id of the request it answers. */
export type StreamBody = StreamEvent extends infer E ? (E extends unknown ? Omit<E, 'id'> : never) : never;

/**
 * Sent to an app without being asked. `changed` means the list of providers did. `closed` means the holder has dropped this connection
 * (it was quiet for too long, or the app opened too many): the app says hello again if it still wants one.
 */
export type Push = { t: 'changed' } | { t: 'closed'; reason: 'idle' | 'replaced' };

export type ListResult = ProviderSummary[];

/** The answer to `test` and `models`: what a person may act on. It never carries the key, and the provider's own words are redacted. */
export interface TestResult {
  ok: boolean;
  models?: Array<{ id: string; label?: string }>;
  /** Whether the model takes a `tools` array: `unknown` when it could not be told. */
  tools?: 'yes' | 'no' | 'unknown';
  /** What a person may do next, in words. Set when `ok` is false. */
  error?: { kind: string; message: string; status?: number };
}

export function errorBody(code: ErrorCode, message: string, extra: Partial<ErrorBody> = {}): ErrorBody {
  return { code, message, ...extra };
}
