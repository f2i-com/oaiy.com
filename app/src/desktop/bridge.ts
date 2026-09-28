/**
 * OAIY Desktop, as the app reaches it: the bridge on this computer
 * (http://127.0.0.1:17972 by default) that runs the plugins (Aokie: the phone),
 * holds their events, and takes their commands. The same contract FormLogic
 * uses (docs/ecosystem/OAIY_PLATFORM.md): an open health check, pairing (the
 * person approves a code in the desktop's window, and the page gets a token),
 * then the event ring and connector commands with that token.
 */

export const DESKTOP_ORIGIN = 'http://127.0.0.1:17972';

export interface DesktopHealth {
  product: string;
  protocol: string;
  version: string;
}

/** An event as a plugin sent it (the desktop-event envelope). */
export interface DesktopEvent {
  seq: number;
  name: string;
  source: string;
  correlationId: string;
  idempotencyKey: string;
  occurredAt: string;
  data: Record<string, unknown>;
}

export class DesktopError extends Error {
  constructor(message: string, readonly status = 0, readonly code = '') {
    super(message);
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);

async function reply(resp: Response): Promise<unknown> {
  const text = await resp.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!resp.ok) {
    const err = isRecord(body) ? body.error : null;
    const code = isRecord(err) ? String(err.code ?? '') : '';
    const message = isRecord(err) ? String(err.message ?? code) : typeof err === 'string' ? err : `HTTP ${resp.status}`;
    throw new DesktopError(resp.status === 401 || resp.status === 403 ? `OAIY Desktop refused this page (${message}): pair it again` : message, resp.status, code);
  }
  return body;
}

/** Whether OAIY Desktop answers at `origin` (null when nothing does, or something else does). */
export async function desktopHealth(origin = DESKTOP_ORIGIN, signal?: AbortSignal): Promise<DesktopHealth | null> {
  try {
    const resp = await fetch(`${origin}/api/health`, { signal });
    if (!resp.ok) return null;
    const body = (await resp.json()) as Record<string, unknown>;
    if (body.product !== 'oaiy-desktop' || !String(body.protocol ?? '').startsWith('oaiy-bridge/1')) return null;
    return { product: String(body.product), protocol: String(body.protocol), version: String(body.version ?? '') };
  } catch {
    return null;
  }
}

/** Ask to pair: the desktop shows `code`, and the person approves it there. */
export async function startPairing(origin: string, label: string): Promise<{ pairingId: string; code: string }> {
  const body = await reply(await fetch(`${origin}/api/bridge/pairing`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ product: 'oaiy-app', label }),
  }));
  if (!isRecord(body) || typeof body.pairingId !== 'string' || typeof body.code !== 'string') throw new DesktopError('OAIY Desktop did not start pairing');
  return { pairingId: body.pairingId, code: body.code };
}

/** Where a pairing request stands; the token once it is approved. */
export async function pairingStatus(origin: string, pairingId: string): Promise<{ status: 'pending' | 'approved' | 'denied' | 'expired'; token?: string }> {
  const resp = await fetch(`${origin}/api/bridge/pairing/${encodeURIComponent(pairingId)}`);
  if (resp.status === 404) return { status: 'expired' };
  const body = await reply(resp);
  const status = isRecord(body) ? String(body.status) : '';
  const token = isRecord(body) && typeof body.token === 'string' ? body.token : undefined;
  return { status: (['pending', 'approved', 'denied', 'expired'].includes(status) ? status : 'pending') as 'pending', token };
}

/** A paired OAIY Desktop: its events and its plugins' commands. */
export class Desktop {
  constructor(readonly origin: string, readonly token: string) {}

  private headers(): Record<string, string> {
    return { authorization: `Bearer ${this.token}`, 'content-type': 'application/json' };
  }

  /** Events after `since` (a sequence number), and the position to ask from next. */
  async events(since: number, signal?: AbortSignal, limit = 200): Promise<{ events: DesktopEvent[]; next: number }> {
    const body = await reply(await fetch(`${this.origin}/api/bridge/events?since=${since}&limit=${limit}`, { headers: this.headers(), signal }));
    const list = isRecord(body) && Array.isArray(body.events) ? body.events : [];
    const events: DesktopEvent[] = [];
    for (const item of list) {
      if (!isRecord(item) || !isRecord(item.envelope)) continue;
      const e = item.envelope;
      events.push({
        seq: Number(item.seq) || 0,
        name: String(e.name ?? ''),
        source: String(e.source ?? ''),
        correlationId: String(e.correlationId ?? ''),
        idempotencyKey: String(e.idempotencyKey ?? ''),
        occurredAt: String(e.occurredAt ?? ''),
        data: isRecord(e.data) ? e.data : {},
      });
    }
    const next = isRecord(body) && typeof body.next === 'number' ? body.next : since;
    return { events, next };
  }

  /**
   * A plugin's command (such as Aokie's `sms.send`), and what it answered.
   * `idempotencyKey` makes a repeat harmless: the same key gets the same result.
   */
  async command(connector: string, command: string, payload: Record<string, unknown>, idempotencyKey: string, signal?: AbortSignal): Promise<unknown> {
    const body = await reply(await fetch(`${this.origin}/api/bridge/connectors/${encodeURIComponent(connector)}/request`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ command, payload, idempotencyKey }),
      signal,
    }));
    // {ok, result: {ok, data}}: the plugin's own answer inside the desktop's.
    const result = isRecord(body) ? body.result : null;
    if (isRecord(result) && result.ok === false) {
      const err = isRecord(result.error) ? result.error : {};
      throw new DesktopError(String(err.message ?? 'the command failed'), 0, String(err.code ?? 'command_failed'));
    }
    return isRecord(result) && 'data' in result ? result.data : result;
  }

  /**
   * Take (or renew) a lease: one holder at a time for a job several pages
   * could each do, such as answering text messages. `prefer` takes it over.
   */
  async lease(name: string, holder: string, ttlMs: number, prefer = false, signal?: AbortSignal): Promise<{ granted: boolean; holder: string }> {
    const body = await reply(await fetch(`${this.origin}/api/bridge/leases/${encodeURIComponent(name)}`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ holder, ttlMs, prefer }),
      signal,
    }));
    return { granted: isRecord(body) && body.granted === true, holder: isRecord(body) ? String(body.holder ?? '') : '' };
  }

  /** Give a lease up. */
  async release(name: string, holder: string): Promise<void> {
    await fetch(`${this.origin}/api/bridge/leases/${encodeURIComponent(name)}`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ holder, release: true }),
      keepalive: true,
    }).catch(() => {});
  }

  /**
   * The desktop's call events (server-sent: `call.started`, `call.caller`,
   * `call.said`, `call.speech_started`, `call.interrupted`, `call.error`,
   * `call.ended`) until `signal` aborts or the stream ends.
   */
  async voiceEvents(onEvent: (event: Record<string, unknown>) => void, signal: AbortSignal): Promise<void> {
    const resp = await fetch(`${this.origin}/api/voice/events`, { headers: this.headers(), signal });
    if (!resp.ok || !resp.body) throw new DesktopError(`OAIY Desktop's call events: HTTP ${resp.status}`, resp.status);
    const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value.replace(/\r\n/g, '\n');
      let end: number;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const data = block.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).replace(/^ /, '')).join('\n');
        if (!data) continue;
        try {
          const event = JSON.parse(data) as unknown;
          if (isRecord(event)) onEvent(event);
        } catch {
          /* not JSON: a comment or a keep-alive */
        }
      }
    }
  }

  private async voice(callId: string, action: string, body: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const reply_ = await reply(await fetch(`${this.origin}/api/voice/calls/${encodeURIComponent(callId)}/${action}`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(body),
      signal,
    }));
    return isRecord(reply_) && isRecord(reply_.result) ? reply_.result : {};
  }

  /** Speak `text` on a live call (after what is queued). */
  async say(callId: string, text: string, signal?: AbortSignal): Promise<void> {
    await this.voice(callId, 'say', { text }, signal);
  }

  /** One of the call's tools (`request_appointment`, `lookup_business_data`): what the phone answered. */
  async callTool(callId: string, name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<{ ok: boolean; output: unknown }> {
    const result = await this.voice(callId, 'tool', { name, arguments: args }, signal);
    return { ok: result.ok === true, output: result.output };
  }

  /** Say goodbye, then hang up. */
  async finishCall(callId: string, goodbye: string, signal?: AbortSignal): Promise<{ ok: boolean; output: unknown }> {
    const result = await this.voice(callId, 'finish', { goodbye }, signal);
    return { ok: result.ok === true, output: result.output };
  }

  /** Stop speaking on a call. */
  async hush(callId: string): Promise<void> {
    await this.voice(callId, 'hush', {}).catch(() => {});
  }

  /** The flows stored on the desktop. */
  async flows(signal?: AbortSignal): Promise<Array<{ id: string; name: string }>> {
    const body = await reply(await fetch(`${this.origin}/api/bridge/flows`, { headers: this.headers(), signal }));
    const list = isRecord(body) && Array.isArray(body.flows) ? body.flows : [];
    return list.filter(isRecord).map((f) => ({ id: String(f.flowId ?? ''), name: String(f.name ?? f.flowId ?? '') })).filter((f) => f.id);
  }

  /** A stored flow, as it was stored. */
  async flow(id: string, signal?: AbortSignal): Promise<unknown> {
    return reply(await fetch(`${this.origin}/api/bridge/flows/${encodeURIComponent(id)}`, { headers: this.headers(), signal }));
  }

  /** Run a stored flow with `input` (by its input nodes' labels) and wait for its result. */
  async runFlow(flowId: string, input: Record<string, unknown>, timeoutMs: number, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const key = `oaiy-app:${flowId}:${crypto.randomUUID()}`;
    const body = await reply(await fetch(`${this.origin}/api/bridge/runs`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ protocol: 'oaiy-bridge/1', caller: { product: 'oaiy-app', label: 'OAIY agent' }, flowId, input, mode: 'sync', timeoutMs, correlationId: key, idempotencyKey: key }),
      signal,
    }));
    return isRecord(body) ? body : {};
  }

  /** The plugins, and whether each runs. */
  async plugins(signal?: AbortSignal): Promise<Array<{ id: string; state: string }>> {
    const body = await reply(await fetch(`${this.origin}/api/plugins`, { headers: this.headers(), signal }));
    const list = Array.isArray(body) ? body : isRecord(body) && Array.isArray(body.plugins) ? body.plugins : [];
    return list.filter(isRecord).map((p) => ({ id: String(p.id ?? ''), state: String(p.state ?? p.status ?? '') }));
  }

  /** The desktop's calendar: its settings, and the appointments from `from` (YYYY-MM-DD) to before `to`. */
  async calendar(from?: string, to?: string, signal?: AbortSignal): Promise<{ settings: Record<string, unknown>; appointments: Array<Record<string, unknown>>; now: string }> {
    const q = new URLSearchParams({ ...(from ? { from } : {}), ...(to ? { to } : {}) }).toString();
    const body = await reply(await fetch(`${this.origin}/api/calendar${q ? `?${q}` : ''}`, { headers: this.headers(), signal }));
    return isRecord(body) ? (body as { settings: Record<string, unknown>; appointments: Array<Record<string, unknown>>; now: string }) : { settings: {}, appointments: [], now: '' };
  }

  /** Free times from `from` for `days` days, for `service` (a name) or its own length. */
  async calendarFree(from: string, days: number, service?: string, signal?: AbortSignal): Promise<{ minutes: number; service: string | null; days: Array<{ date: string; times: string[] }> }> {
    const q = new URLSearchParams({ ...(from ? { from } : {}), days: String(days), ...(service ? { service } : {}) }).toString();
    return (await reply(await fetch(`${this.origin}/api/calendar/free?${q}`, { headers: this.headers(), signal }))) as { minutes: number; service: string | null; days: Array<{ date: string; times: string[] }> };
  }

  /** A new appointment (`date` YYYY-MM-DD and `time` HH:MM, or `start`). */
  async calendarCreate(appointment: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const body = await reply(await fetch(`${this.origin}/api/calendar/appointments`, { method: 'POST', headers: this.headers(), body: JSON.stringify(appointment), signal }));
    return isRecord(body) ? body : {};
  }

  /** Change an appointment (status, start, service, name, phone, notes). */
  async calendarUpdate(id: string, change: Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const body = await reply(await fetch(`${this.origin}/api/calendar/appointments/${encodeURIComponent(id)}`, { method: 'PATCH', headers: this.headers(), body: JSON.stringify(change), signal }));
    return isRecord(body) ? body : {};
  }

  /** The words spoken in `wav` (16 kHz mono, 16-bit), by the desktop's speech-to-text. */
  async transcribe(wav: Uint8Array<ArrayBuffer>, signal?: AbortSignal): Promise<string> {
    const body = await reply(await fetch(`${this.origin}/api/voice/transcribe`, {
      method: 'POST',
      headers: { ...this.headers(), 'content-type': 'audio/wav' },
      body: wav,
      signal,
    }));
    return isRecord(body) ? String(body.text ?? '') : '';
  }
}
