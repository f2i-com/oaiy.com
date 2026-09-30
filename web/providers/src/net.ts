/**
 * What the holder does around a call to a provider: it never follows a redirect, it counts the request, and when a call never got
 * an answer it finds out why (design 4.2, step 4), so the person is told what to change and where.
 *
 * The browser reports a server that is down, a CORS refusal, a permission not given and mixed content as the same TypeError; a
 * second request, with `mode: 'no-cors'` and no credentials, tells the first two apart: an opaque answer means the server is up,
 * and no answer means it is not, or the browser did not let the call out.
 */
import { describeNetworkFailure, isBlockedMixedContent, isLoopbackHost, type NetworkFailure } from '@oaiy/shared/providers/errors';
import type { ProviderRecord } from '@oaiy/shared/providers/types';

export interface PageInfo {
  protocol: string;
  origin: string;
}

export interface Failure {
  kind: NetworkFailure;
  message: string;
}

const PROBE_TIMEOUT_MS = 4000;

/** Whether the address is one on this computer or this network (where a browser may ask for a permission the person can give). */
export function isLocalAddress(url: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^\[|\]$/g, '');
    if (isLoopbackHost(host) || host.endsWith('.local')) return true;
    const m = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(host);
    if (!m) return false;
    const [a, b] = [Number(m[1]), Number(m[2])];
    return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254) || (a === 100 && b >= 64 && b <= 127);
  } catch {
    return false;
  }
}

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

  constructor(limit: number, retryAfterMs: number) {
    super(`The app has made its ${limit} requests for this hour.`);
    this.name = 'BudgetExhausted';
    this.limit = limit;
    this.retryAfterMs = retryAfterMs;
  }
}

/**
 * A `fetch` for the holder's own calls to one provider (the model list, the connection test, the context probe): it goes only to the
 * provider's own origin, never follows a redirect, sends no credentials, and counts against the app's budget when `take` is given.
 * A call it refuses fails like a network error (a TypeError), as the browser's own refusals do.
 */
export function guardedFetch(
  record: Pick<ProviderRecord, 'baseUrl'>,
  fetchImpl: typeof fetch,
  take?: () => Promise<{ ok: true } | { ok: false; limit: number; retryAfterMs: number }>,
): typeof fetch {
  const origin = new URL(record.baseUrl).origin;
  return async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (new URL(url).origin !== origin) throw new TypeError('blocked: not the provider’s own address');
    if (take) {
      const taken = await take();
      if (!taken.ok) throw new BudgetExhausted(taken.limit, taken.retryAfterMs);
    }
    const response = await fetchImpl(url, { ...init, redirect: 'manual', credentials: 'omit', cache: 'no-store', referrerPolicy: 'no-referrer' });
    if (response.type === 'opaqueredirect' || (response.status >= 300 && response.status < 400)) {
      void response.body?.cancel().catch(() => {});
      throw new TypeError('blocked: a redirect is not followed');
    }
    return response;
  };
}
