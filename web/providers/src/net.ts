/**
 * What the holder does around a call to a provider: it never follows a redirect, it counts the request, and when a call never got
 * an answer it finds out why (design 4.2, step 4), so the person is told what to change and where.
 *
 * The browser reports a server that is down, a CORS refusal, a permission not given and mixed content as the same TypeError; a
 * second request, with `mode: 'no-cors'` and no credentials, tells the first two apart: an opaque answer means the server is up,
 * and no answer means it is not, or the browser did not let the call out.
 */
import { describeNetworkFailure, isBlockedMixedContent, isLocalAddress, type NetworkFailure } from '@oaiy/shared/providers/errors';
import type { ProviderRecord } from '@oaiy/shared/providers/types';

// Where an address is on this computer or this network is decided in shared/ (the adapters need it too).
export { isLocalAddress };

export interface PageInfo {
  protocol: string;
  origin: string;
}

export interface Failure {
  kind: NetworkFailure;
  message: string;
}

const PROBE_TIMEOUT_MS = 4000;

/** Why a call to `url` (a provider's address) failed without an answer. */
export async function classifyFailure(record: Pick<ProviderRecord, 'kind' | 'serverKind'>, url: string, page: PageInfo, fetchImpl: typeof fetch): Promise<Failure> {
  const local = isLocalAddress(url);
  const context = { url, pageOrigin: page.origin, serverKind: record.serverKind, local };
  if (isBlockedMixedContent(url, page.protocol)) return { kind: 'mixed-content', message: describeNetworkFailure('mixed-content', context) };
  let up = false;
  try {
    // No credentials, no headers of ours: nothing about the key or the request goes with it.
    await fetchImpl(new URL(url).origin + '/', { method: 'GET', mode: 'no-cors', credentials: 'omit', cache: 'no-store', referrerPolicy: 'no-referrer', signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    up = true;
  } catch {
    up = false;
  }
  if (!up) return { kind: 'unreachable', message: describeNetworkFailure('unreachable', context) };
  // Up. A service on the internet that is up and unreadable is most often a key it refused; a server on this computer or network
  // is most often one that was never told to allow this page.
  const kind: NetworkFailure = record.kind === 'external' && !local ? 'unreadable' : 'cors';
  return { kind, message: describeNetworkFailure(kind, context) };
}

/** A request that would have gone past what the app's hour allows. */
export class BudgetExhausted extends Error {
  readonly limit: number;
  readonly retryAfterMs: number;

  readonly reason: 'requests' | 'bytes';
  readonly byteLimit: number;

  constructor(limit: number, retryAfterMs: number, reason: 'requests' | 'bytes' = 'requests', byteLimit = 0) {
    super(budgetMessage(reason, limit, byteLimit));
    this.name = 'BudgetExhausted';
    this.limit = limit;
    this.retryAfterMs = retryAfterMs;
    this.reason = reason;
    this.byteLimit = byteLimit;
  }
}

/** What an app is told when its hour is used up: fixed wording, a function of which limit it hit. */
export function budgetMessage(reason: 'requests' | 'bytes', limit: number, byteLimit: number): string {
  const size = byteLimit >= 1024 * 1024 ? `${Math.round((byteLimit / 1024 / 1024) * 10) / 10} MiB` : `${Math.round(byteLimit / 1024)} KiB`;
  return reason === 'bytes'
    ? `This app has sent its ${size} for this hour. It can go on when the hour has passed, or the limit can be raised on the Providers page.`
    : `This app has made its ${limit} requests for this hour. It can go on when the hour has passed, or the limit can be raised on the Providers page.`;
}

/** What taking one request of an hour's allowance answers. */
export type TakeResult = { ok: true } | { ok: false; reason: 'requests' | 'bytes'; limit: number; byteLimit: number; retryAfterMs: number };

/** The size in bytes of a request body: text as UTF-8, bytes as they are, nothing as 0. */
export function bodyBytes(body: unknown): number {
  if (typeof body === 'string') return new TextEncoder().encode(body).byteLength;
  if (body instanceof ArrayBuffer || ArrayBuffer.isView(body)) return body.byteLength;
  return 0;
}

/**
 * A `fetch` for the holder's own calls to one provider (the model list, the connection test, the context probe): it goes only to the
 * provider's own origin, never follows a redirect, sends no credentials, and counts against the app's budget when `take` is given.
 * A call it refuses fails like a network error (a TypeError), as the browser's own refusals do.
 */
export function guardedFetch(
  record: Pick<ProviderRecord, 'baseUrl'>,
  fetchImpl: typeof fetch,
  take?: (bytes: number) => Promise<TakeResult>,
): typeof fetch {
  const origin = new URL(record.baseUrl).origin;
  return async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (new URL(url).origin !== origin) throw new TypeError('blocked: not the provider’s own address');
    if (take) {
      const taken = await take(bodyBytes(init?.body));
      if (!taken.ok) throw new BudgetExhausted(taken.limit, taken.retryAfterMs, taken.reason, taken.byteLimit);
    }
    const response = await fetchImpl(url, { ...init, redirect: 'manual', credentials: 'omit', cache: 'no-store', referrerPolicy: 'no-referrer' });
    if (response.type === 'opaqueredirect' || (response.status >= 300 && response.status < 400)) {
      void response.body?.cancel().catch(() => {});
      throw new TypeError('blocked: a redirect is not followed');
    }
    return response;
  };
}
