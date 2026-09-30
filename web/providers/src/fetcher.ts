/**
 * The one operation that leaves this origin: a request to a provider, made by the holder with the key it holds (design 3.2).
 *
 * What a page chooses is only a provider (by id), one path of the fixed list, a query, `content-type` and `accept`, a body and a
 * timeout. Everything else is the holder's: the address is the record's base plus the path (checked, `buildRequestUrl`), the
 * headers are the key, the dialect's and the record's own (`recordHeaders`), redirects are never followed (a key in `x-api-key` would
 * go with the request to wherever it was sent), and the answer comes back as a stream of events. None of a provider's error text goes on
 * (fixed.ts): a scrub that depends on the key would tell an app which pieces the key holds.
 */
import { buildRequestUrl, recordHeaders, RequestRefused } from '@oaiy/shared/providers/endpoints';
import { errorBody, type FetchRequest, type StreamBody } from '@oaiy/shared/broker/protocol';
import type { Budget } from './budget';
import { fixedErrorBody, fixedStatusText, safeResponseHeaders } from './fixed';
import { classifyFailure, type PageInfo } from './net';
import type { ProviderStore } from './store';

export interface FetcherDeps {
  store: ProviderStore;
  budget: Budget;
  fetchImpl: typeof fetch;
  page: PageInfo;
}

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

        if (response.status >= 400) {
          // No word of a provider's error goes on: an app gets the status and wording that are a function of the status alone (fixed.ts).
          void response.body?.cancel().catch(() => {});
          const body = fixedErrorBody(record, response.status);
          emit({ t: 'head', status: response.status, statusText: fixedStatusText(response.status), headers: safeResponseHeaders(response.headers, true) });
          emit({ t: 'chunk', bytes: body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength) as ArrayBuffer });
          return emit({ t: 'end' });
        }

        emit({ t: 'head', status: response.status, statusText: fixedStatusText(response.status), headers: safeResponseHeaders(response.headers) });
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
