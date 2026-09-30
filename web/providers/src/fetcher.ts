/**
 * The one operation that leaves this origin: a request to a provider, made by the holder with the key it holds (design 3.2).
 *
 * What a page chooses is only a provider (by id), one path of the fixed list, a query, `content-type` and `accept`, a body and a
 * timeout. Everything else is the holder's: the address is the record's base plus the path (checked, `buildRequestUrl`), the
 * headers are the key, the dialect's and the record's own (`recordHeaders`), redirects are never followed (a key in `x-api-key` would
 * go with the request to wherever it was sent), and the answer comes back as a stream of events. An error a provider sends is
 * scrubbed of the key before it is passed on, because a provider's error text can quote the key it refused.
 */
import { buildRequestUrl, recordHeaders, RequestRefused } from '@oaiy/shared/providers/endpoints';
import { redactSecret } from '@oaiy/shared/providers/errors';
import { errorBody, type FetchRequest, type StreamBody } from '@oaiy/shared/broker/protocol';
import type { Budget } from './budget';
import { classifyFailure, type PageInfo } from './net';
import type { ProviderStore } from './store';

export interface FetcherDeps {
  store: ProviderStore;
  budget: Budget;
  fetchImpl: typeof fetch;
  page: PageInfo;
}

/** The most of an error answer that is read and scrubbed before it is passed on. */
const ERROR_BODY_MAX = 256 * 1024;

const OMITTED_HEADERS = new Set(['set-cookie', 'set-cookie2', 'content-encoding', 'content-length', 'transfer-encoding']);

export interface Fetcher {
  /** Run one request for `app`. Every outcome is an event to `emit`; nothing here rejects. */
  run(app: string, request: FetchRequest, emit: (event: StreamBody) => void, signal: AbortSignal): Promise<void>;
}

export function createFetcher(deps: FetcherDeps): Fetcher {
  return {
    async run(app, request, emit, signal) {
      const fail = (code: Parameters<typeof errorBody>[0], message: string, extra: Parameters<typeof errorBody>[2] = {}) => emit({ t: 'error', error: errorBody(code, message, extra) });

      const record = await deps.store.get(request.provider);
      if (!record || record.via !== 'broker') return fail('unknown-provider', 'There is no such provider.');
      let url: string;
      let key = '';
      let headers: Record<string, string>;
      try {
        url = buildRequestUrl(record, request.path, request.method, request.query);
        key = await deps.store.key(record.id);
        headers = recordHeaders(record, key, request.headers ?? []);
      } catch (e) {
        if (e instanceof RequestRefused) return fail(e.code, e.message);
        return fail('internal', 'The request could not be made.');
      }

      const taken = await deps.budget.take(app);
      if (!taken.ok) {
        return fail('budget', `This app has made its ${taken.limit} requests for this hour. It can go on when the hour has passed, or the limit can be raised on the Providers page.`, { retryAfterMs: taken.retryAfterMs });
      }

      let why: 'aborted' | 'timeout' | null = null;
      const controller = new AbortController();
      const timer = setTimeout(() => {
        why = 'timeout';
        controller.abort();
      }, request.timeoutMs);
      const onAbort = () => {
        why ??= 'aborted';
        controller.abort();
      };
      if (signal.aborted) onAbort();
      signal.addEventListener('abort', onAbort, { once: true });

      const stopped = () => (why === 'timeout' ? fail('timeout', 'The provider did not finish answering in time.') : fail('aborted', 'The request was stopped.'));

      try {
        let response: Response;
        try {
          response = await deps.fetchImpl(url, {
            method: request.method,
            headers,
            body: request.method === 'GET' ? undefined : (request.body as BodyInit | undefined),
            signal: controller.signal,
            // Never follow a redirect: a key sent in a header would go on to wherever the provider (or whoever answered for it) said.
            redirect: 'manual',
            credentials: 'omit',
            cache: 'no-store',
            referrerPolicy: 'no-referrer',
          });
        } catch {
          if (why) return stopped();
          const failure = await classifyFailure(record, url, deps.page, deps.fetchImpl);
          return fail('network', failure.message);
        }

        if (response.type === 'opaqueredirect' || (response.status >= 300 && response.status < 400)) {
          void response.body?.cancel().catch(() => {});
          return fail('redirect', 'The provider answered with a redirect. It is not followed: a request goes only to the address the provider was saved with.');
        }

        const heads: Array<[string, string]> = [];
        response.headers.forEach((value, name) => {
          if (!OMITTED_HEADERS.has(name.toLowerCase())) heads.push([name, value]);
        });

        if (response.status >= 400) {
          // An error is read whole (it is small) and scrubbed of the key before a page sees it.
          const bytes = await readCapped(response, ERROR_BODY_MAX, controller.signal).catch(() => null);
          if (bytes === null) return why ? stopped() : fail('network', 'The provider’s answer was cut off.');
          const scrubbed = new TextEncoder().encode(redactSecret(new TextDecoder().decode(bytes), key));
          emit({ t: 'head', status: response.status, statusText: response.statusText, headers: heads });
          if (scrubbed.byteLength > 0) emit({ t: 'chunk', bytes: scrubbed.buffer.slice(scrubbed.byteOffset, scrubbed.byteOffset + scrubbed.byteLength) as ArrayBuffer });
          return emit({ t: 'end' });
        }

        emit({ t: 'head', status: response.status, statusText: response.statusText, headers: heads });
        if (!response.body) return emit({ t: 'end' });
        const reader = response.body.getReader();
        // A stop reaches the body whether or not the browser ties it to the request's signal.
        const stopReading = () => void reader.cancel().catch(() => {});
        controller.signal.addEventListener('abort', stopReading, { once: true });
        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            if (why) break;
            if (value.byteLength > 0) emit({ t: 'chunk', bytes: value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength) as ArrayBuffer });
          }
        } catch {
          if (why) return stopped();
          return fail('network', 'The connection to the provider was lost before the answer was finished.');
        } finally {
          controller.signal.removeEventListener('abort', stopReading);
        }
        if (why) return stopped();
        emit({ t: 'end' });
      } finally {
        clearTimeout(timer);
        signal.removeEventListener('abort', onAbort);
      }
    },
  };
}

/** The body of `response`, up to `max` bytes. */
async function readCapped(response: Response, max: number, signal: AbortSignal): Promise<Uint8Array> {
  const reader = response.body?.getReader();
  if (!reader) return new Uint8Array(0);
  const parts: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    signal.throwIfAborted();
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.byteLength;
    if (total >= max) {
      void reader.cancel().catch(() => {});
      break;
    }
  }
  const out = new Uint8Array(Math.min(total, max));
  let at = 0;
  for (const part of parts) {
    const take = part.subarray(0, Math.max(0, out.length - at));
    out.set(take, at);
    at += take.length;
  }
  return out;
}
