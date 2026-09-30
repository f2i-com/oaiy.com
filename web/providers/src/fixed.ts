/**
 * What an app is told about a provider's answer that is not the provider's own words (design 6).
 *
 * A provider's error text is provider-authored, and an app can steer parts of it (a model name, a body it sent that the provider echoes).
 * Removing the key from that text (a scrub) makes the text depend on the key: an app that chooses what is echoed reads, from what the
 * scrub removes, which pieces the key holds. So none of a provider's error text is forwarded: an app gets the HTTP status and a
 * classification and wording that are FIXED, a function of the status alone. The top-level Providers page and the modal are the holder's
 * own pages, whose DOM an app cannot read, and may show provider text as inert text after the same scrub (see `providerText` in test.ts).
 *
 * What is passed on from a provider's answer, then: the status; a `content-type`; `retry-after` and rate-limit counters when they are
 * plain numbers; and the body of a SUCCESS, which is the provider's answer to the request and cannot be otherwise. The holder trusts the
 * provider not to reflect its `Authorization` header into a success body; nothing filters one that does.
 */
import { providerTypeOf } from '@oaiy/shared/providers/adapters';
import { describeConnectionError, kindForStatus } from '@oaiy/shared/providers/errors';
import type { ProviderRecord } from '@oaiy/shared/providers/types';

const STATUS_TEXT: Record<number, string> = {
  400: 'Bad Request',
  401: 'Unauthorized',
  402: 'Payment Required',
  403: 'Forbidden',
  404: 'Not Found',
  408: 'Request Timeout',
  409: 'Conflict',
  413: 'Payload Too Large',
  422: 'Unprocessable Content',
  429: 'Too Many Requests',
  500: 'Internal Server Error',
  502: 'Bad Gateway',
  503: 'Service Unavailable',
  504: 'Gateway Timeout',
};

/** The reason phrase for a status: ours, never the provider's (an HTTP/1 reason phrase is free text). */
export function fixedStatusText(status: number): string {
  return status >= 200 && status < 300 ? 'OK' : (STATUS_TEXT[status] ?? (status >= 500 ? 'Server Error' : 'Error'));
}

/** The fixed body an app gets for an error status, in the shape its dialect's clients read. */
export function fixedErrorBody(record: ProviderRecord, status: number): Uint8Array {
  const kind = kindForStatus(status);
  const message = describeConnectionError(kind, { type: providerTypeOf(record), serverKind: record.serverKind, url: record.baseUrl }, status);
  const body =
    record.dialect === 'anthropic'
      ? { type: 'error', error: { type: kind, message } }
      : { error: { message, type: 'provider_error', code: kind } };
  return new TextEncoder().encode(JSON.stringify(body));
}

const MEDIA_TYPE = /^[A-Za-z0-9!#$&^_.+-]{1,64}\/[A-Za-z0-9!#$&^_.+-]{1,64}(?:; ?[A-Za-z0-9_.-]{1,32}=[A-Za-z0-9_.-]{1,32})*$/;
const COUNTER = /^\d{1,10}(?:\.\d{1,3})?(?:ms|s|m|h)?$/;

/** The headers of a provider's answer an app is given: a media type, and counters that are plain numbers. Nothing else, whatever it says. */
export function safeResponseHeaders(headers: Headers, error = false): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  headers.forEach((value, name) => {
    const lower = name.toLowerCase();
    const v = value.trim();
    if (lower === 'content-type' && !error && MEDIA_TYPE.test(v)) out.push([lower, v]);
    else if (lower === 'retry-after' && /^\d{1,7}$/.test(v)) out.push([lower, v]);
    else if (/^(?:x-)?ratelimit-(?:limit|remaining|reset)(?:-(?:requests|tokens))?$/.test(lower) && COUNTER.test(v)) out.push([lower, v]);
  });
  if (error) out.push(['content-type', 'application/json']);
  return out;
}
