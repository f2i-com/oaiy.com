/**
 * What the holder bounds in a request beyond an app's hourly count (design 3.2, threat 6): how large a body may be, and, where a record
 * sets a cap, how long a reply may be. These bound VOLUME. A request can still name any model the key may use, and what a reply costs is
 * the provider's price, which the holder cannot know: it does not bound cost, and the Providers page says so.
 */
import { DEFAULT_MAX_BODY_BYTES, MAX_BODY_BYTES, type RequestBody } from '@oaiy/shared/broker/protocol';
import type { ProviderRecord } from '@oaiy/shared/providers/types';

/** A body the holder will not send, and why (a fixed sentence, never the body's own words). */
export class BodyRefused extends Error {
  readonly code: 'too-large' | 'bad-body';

  constructor(code: 'too-large' | 'bad-body', message: string) {
    super(message);
    this.name = 'BodyRefused';
    this.code = code;
  }
}

/** The largest body a request to this provider may carry: what the record says (never above the ceiling), or 1 MiB. */
export function maxBodyBytes(record: Pick<ProviderRecord, 'limits'>): number {
  const own = record.limits?.maxBodyBytes;
  return typeof own === 'number' && Number.isInteger(own) && own > 0 ? Math.min(own, MAX_BODY_BYTES) : DEFAULT_MAX_BODY_BYTES;
}

/** The paths whose body asks for a reply of some length: a chat request. The other paths (images, speech, embeddings) have no reply length to cap. */
const GENERATION_PATH = { openai: '/chat/completions', anthropic: '/messages' } as const;

/**
 * The body with the reply length capped, when the record sets a cap and the request is a chat request. `max_tokens` (and, for the
 * OpenAI dialect, `max_completion_tokens`) is set to the cap when the request asks for more or names none; a smaller ask is kept. A chat
 * body that is not a JSON object is refused: the holder cannot cap what it cannot read. Anything else is returned as it was.
 */
export function capOutputTokens(record: Pick<ProviderRecord, 'dialect' | 'limits'>, path: string, body: RequestBody | undefined): RequestBody | undefined {
  const cap = record.limits?.maxOutputTokens;
  if (cap === undefined || body === undefined || path !== GENERATION_PATH[record.dialect]) return body;
  const text = typeof body === 'string' ? body : new TextDecoder('utf-8', { fatal: false }).decode(body as ArrayBufferView | ArrayBuffer);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new BodyRefused('bad-body', 'This provider caps the length of a reply, and the holder can only do that to a JSON body.');
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new BodyRefused('bad-body', 'This provider caps the length of a reply, and the holder can only do that to a JSON object.');
  const asked = parsed as Record<string, unknown>;
  const clamp = (name: string): void => {
    const now = asked[name];
    asked[name] = typeof now === 'number' && Number.isFinite(now) && now >= 1 ? Math.min(Math.floor(now), cap) : cap;
  };
  if (record.dialect === 'anthropic') clamp('max_tokens');
  else {
    // Either name asks for a reply length; one that is present is capped, and `max_tokens` is set when neither is.
    let any = false;
    for (const name of ['max_tokens', 'max_completion_tokens']) {
      if (name in asked) {
        clamp(name);
        any = true;
      }
    }
    if (!any) asked.max_tokens = cap;
  }
  return JSON.stringify(asked);
}
