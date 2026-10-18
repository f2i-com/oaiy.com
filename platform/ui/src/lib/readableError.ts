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
