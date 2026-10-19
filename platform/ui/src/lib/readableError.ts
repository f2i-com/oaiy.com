/**
 * An error as a person reads it. A service that fails answers with JSON (`{"error": {"message": "…"}}`, FastAPI's
 * `{"detail": …}`), and a run passed that on whole: the queue and the toast said
 * `HTTP 500: {"error":{"message":"Chat failed: …","type":"server_error"}}`. Here the JSON gives way to its message
 * (`HTTP 500: Chat failed: …`), and the `Error: ` an error gathers each time it is wrapped is said none times. Text with
 * no such JSON in it comes back as it was (less those prefixes).
 */
export function readableError(text: string): string {
  const plain = String(text ?? '').replace(/^(Error:\s*)+/, '').trim();
  const start = plain.indexOf('{');
  const end = plain.lastIndexOf('}');
  if (start < 0 || end <= start) return plain;
  let body: unknown;
  try {
    body = JSON.parse(plain.slice(start, end + 1));
  } catch {
    return plain;
  }
  const message = messageOf(body);
  if (!message) return plain;
  const before = plain.slice(0, start).trim();
  const after = plain.slice(end + 1).trim();
  return [before, message, after].filter(Boolean).join(' ').replace(/:\s*:/g, ':');
}

/** The words in a service's error body, where it has them. */
function messageOf(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null;
  const o = body as Record<string, unknown>;
  const err = o.error;
  if (typeof err === 'string' && err.trim()) return err.trim();
  if (err && typeof err === 'object') {
    const m = (err as Record<string, unknown>).message;
    if (typeof m === 'string' && m.trim()) return m.trim();
  }
  if (typeof o.message === 'string' && o.message.trim()) return o.message.trim();
  const detail = o.detail;
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) => (d && typeof d === 'object' ? (d as Record<string, unknown>).msg : d)).filter((m): m is string => typeof m === 'string');
    if (msgs.length) return msgs.join('; ');
  }
  return null;
}

/** A status said in words, for an error that came with nothing else (the person cannot act on "HTTP 503"). */
function statusInWords(status: number): string {
  if (status === 401 || status === 403) return 'the service refused it: a key or a permission is missing';
  if (status === 404) return 'the service has nothing at that address';
  if (status === 408 || status === 504) return 'the service took too long to answer';
  if (status === 413) return 'what was sent is too large for the service';
  if (status === 429) return 'the service is busy: too many requests at once';
  if (status === 503) return 'the service is not ready';
  if (status >= 500) return 'the service failed while doing it';
  if (status >= 400) return 'the service did not take the request';
  return 'the service answered with an error';
}

/**
 * An error as a toast says it: `readableError`, without the HTTP status a step's error carries ("Image Gen: HTTP 503 —
 * The picture engine is not running" says "Image Gen: The picture engine is not running"). The status stays in the run's
 * log, which has the whole message; where it was all the error said, a sentence says what it means.
 */
export function toastError(text: string): string {
  const said = readableError(text);
  return said
    // ("… (HTTP 503)" after a sentence: it says the rest already)
    .replace(/\s*\(HTTP \d{3}\)/g, '')
    // ("HTTP 503 — words", "HTTP 503: words": the words)
    .replace(/\bHTTP \d{3}\s*(?:—|–|-|:)\s*(?=\S)/g, '')
    // ("HTTP 503" with nothing after it: what it means)
    .replace(/\bHTTP (\d{3})\b\s*$/, (_m, code: string) => statusInWords(Number(code)))
    .trim();
}
