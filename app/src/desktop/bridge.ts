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

/** Who named a contact, or wrote a fact: the person (`owner`) or the receptionist (`agent`). */
export type ContactBy = 'owner' | 'agent';

/**
 * A person as OAIY Desktop's Contacts keep them (the dashboard's AI
 * Receptionist → Contacts): their name (the person's own, `nameBy: owner`,
 * is never changed by the receptionist), the person's notes for the
 * receptionist, and what it remembered (`facts`).
 */
export interface Contact {
  /** The last nine digits of their number: the same person however it is written. */
  key: string;
  /** The number last seen for them ('' when only the key is known). */
  number: string;
  name: string;
  nameBy: ContactBy | null;
  notes: string;
  facts: Array<{ text: string; at: string; by: ContactBy }>;
  createdAt: string;
  updatedAt: string;
}

/** A contact as the desktop answered it (anything missing read as empty), or null when it is not one. */
export function readContact(value: unknown): Contact | null {
  if (!isRecord(value) || typeof value.key !== 'string' || !value.key) return null;
  const by = (v: unknown): ContactBy | null => (v === 'owner' || v === 'agent' ? v : null);
  const text = (v: unknown) => (typeof v === 'string' ? v : '');
  return {
    key: value.key,
    number: text(value.number),
    name: text(value.name),
    nameBy: by(value.nameBy),
    notes: text(value.notes),
    facts: (Array.isArray(value.facts) ? value.facts : []).filter(isRecord).filter((f) => typeof f.text === 'string' && f.text.trim()).map((f) => ({ text: String(f.text), at: text(f.at), by: by(f.by) ?? 'agent' })),
    createdAt: text(value.createdAt),
    updatedAt: text(value.updatedAt),
  };
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
   * One of a plugin's service-definition actions (such as `aokie.phone`'s
   * `sms.threads`), through the desktop's gated route, and what it answered.
   * `idempotencyKey` makes a repeat harmless (journalled actions need one).
   */
  async invokeAction(definition: string, actionId: string, input: Record<string, unknown>, idempotencyKey: string, signal?: AbortSignal): Promise<unknown> {
    const body = await reply(await fetch(`${this.origin}/api/services/actions/${encodeURIComponent(definition)}/${encodeURIComponent(actionId)}/invoke`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify({ input, idempotencyKey }),
      signal,
    }));
    // {ok, result: {ok, data}}: the plugin's own answer inside the desktop's.
    const result = isRecord(body) ? body.result : null;
    if (isRecord(result) && result.ok === false) {
      if (typeof result.error === 'string') throw new DesktopError(result.error, 0, 'action_failed');
      const err = isRecord(result.error) ? result.error : {};
      throw new DesktopError(String(err.message ?? 'the action failed'), 0, String(err.code ?? 'action_failed'));
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
    return this.follow('/api/voice/events', "OAIY Desktop's call events", onEvent, signal);
  }

  /**
   * Which of the desktop's modules (the phone, the calendar) are on: its
   * snapshot, as `modules.ts` reads it. An older desktop answers 404 (a
   * DesktopError with that status).
   */
  async modules(signal?: AbortSignal): Promise<unknown> {
    return reply(await fetch(`${this.origin}/api/modules`, { headers: this.headers(), signal }));
  }

  /** The modules' snapshot now, then again on each change (server-sent), until `signal` aborts or the stream ends. */
  async moduleEvents(onSnapshot: (snapshot: Record<string, unknown>) => void, signal: AbortSignal): Promise<void> {
    return this.follow('/api/modules/events', "OAIY Desktop's modules", onSnapshot, signal);
  }

  /** Flows' tasks for the agent (`agent.task`: id, from, task), the waiting ones first, until `signal` aborts. */
  async agentTasks(onEvent: (event: Record<string, unknown>) => void, signal: AbortSignal): Promise<void> {
    return this.follow('/api/agent/events', "OAIY Desktop's tasks for the agent", onEvent, signal);
  }

  /** The agent's answer to a flow's task (or why it could not). */
  async answerTask(id: string, answer: { reply: string } | { error: string }, signal?: AbortSignal): Promise<void> {
    await reply(await fetch(`${this.origin}/api/agent/tasks/${encodeURIComponent(id)}/reply`, { method: 'POST', headers: this.headers(), body: JSON.stringify(answer), signal }));
  }

  /** A server-sent event stream of the desktop's, event by event. */
  private async follow(path: string, what: string, onEvent: (event: Record<string, unknown>) => void, signal: AbortSignal): Promise<void> {
    const resp = await fetch(`${this.origin}${path}`, { headers: this.headers(), signal });
    if (!resp.ok || !resp.body) throw new DesktopError(`${what}: HTTP ${resp.status}`, resp.status);
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

  /**
   * The name a caller goes by, so the phone's greeting can use it (an empty
   * name forgets it): the receptionist's, never over a name the person gave
   * them in Contacts. Answers with the name kept ('' when the desktop did not say).
   */
  async rememberCaller(number: string, name: string): Promise<string> {
    const body = await reply(await fetch(`${this.origin}/api/voice/callers`, { method: 'PUT', headers: this.headers(), body: JSON.stringify({ number, name }) }));
    return isRecord(body) && typeof body.name === 'string' ? body.name : '';
  }

  /**
   * The contact for a number written any way, as OAIY Desktop's Contacts keep
   * it: null when they have none (`no_contact`). Any other failure (the
   * desktop out of reach, or one from before it kept contacts) throws.
   */
  async contact(number: string, signal?: AbortSignal): Promise<Contact | null> {
    try {
      return readContact(await reply(await fetch(`${this.origin}/api/contacts/${encodeURIComponent(number)}`, { headers: this.headers(), signal })));
    } catch (error) {
      if (error instanceof DesktopError && error.code === 'no_contact') return null;
      throw error;
    }
  }

  /** Every contact (by name), or those `q` finds in names, numbers, notes and facts. */
  async contacts(q = '', signal?: AbortSignal): Promise<Contact[]> {
    const body = await reply(await fetch(`${this.origin}/api/contacts${q ? `?q=${encodeURIComponent(q)}` : ''}`, { headers: this.headers(), signal }));
    const list = isRecord(body) && Array.isArray(body.contacts) ? body.contacts : [];
    return list.map(readContact).filter((c): c is Contact => !!c);
  }

  /**
   * Something the receptionist remembered about a person (made a contact if
   * they are not one): kept once (the same words again are not added), the
   * oldest of the receptionist's let go when the list is full.
   */
  async addContactFact(number: string, text: string, by: ContactBy = 'agent', signal?: AbortSignal): Promise<{ contact: Contact | null; added: boolean }> {
    const body = await reply(await fetch(`${this.origin}/api/contacts/${encodeURIComponent(number)}/facts`, { method: 'POST', headers: this.headers(), body: JSON.stringify({ text, by }), signal }));
    return { contact: isRecord(body) ? readContact(body.contact) : null, added: isRecord(body) && body.added === true };
  }

  /** Forget one of a contact's facts, by its place (from 0), only while it still says `text`. */
  async forgetContactFact(number: string, index: number, text: string, signal?: AbortSignal): Promise<Contact | null> {
    const body = await reply(await fetch(`${this.origin}/api/contacts/${encodeURIComponent(number)}/facts/${index}?text=${encodeURIComponent(text)}`, { method: 'DELETE', headers: this.headers(), signal }));
    return isRecord(body) ? readContact(body.contact) : null;
  }

  /** Stop speaking on a call. */
  async hush(callId: string): Promise<void> {
    await this.voice(callId, 'hush', {}).catch(() => {});
  }

  /** Store a flow on the desktop (the flow editor's store): made or replaced. */
  async putFlow(id: string, doc: Record<string, unknown>, signal?: AbortSignal): Promise<void> {
    await reply(await fetch(`${this.origin}/api/bridge/flows/${encodeURIComponent(id)}`, { method: 'PUT', headers: this.headers(), body: JSON.stringify(doc), signal }));
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

  /**
   * The desktop's calendar: its settings (with `business`, the business's
   * name), the appointments from `from` (YYYY-MM-DD) to before `to`, and the
   * receptionist's name (`receptionistName`, filled in: "Aokie" unless the
   * person set one; none from a desktop before it kept one).
   */
  async calendar(from?: string, to?: string, signal?: AbortSignal): Promise<{ available?: boolean; settings: Record<string, unknown>; appointments: Array<Record<string, unknown>>; now: string; receptionistName?: string }> {
    const q = new URLSearchParams({ ...(from ? { from } : {}), ...(to ? { to } : {}) }).toString();
    const body = await reply(await fetch(`${this.origin}/api/calendar${q ? `?${q}` : ''}`, { headers: this.headers(), signal }));
    return isRecord(body) ? (body as { settings: Record<string, unknown>; appointments: Array<Record<string, unknown>>; now: string; receptionistName?: string }) : { settings: {}, appointments: [], now: '' };
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
