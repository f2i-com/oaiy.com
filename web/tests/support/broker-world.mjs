/**
 * The holder's port protocol in one process: a broker over a fake IndexedDB, a `fetch` the test controls, and a client on the other
 * end of a real MessageChannel.
 */
import { KEY, M, input, makeHolder } from './holder.mjs';

export const AGENT = 'https://agent.example';
export const FLOWS = 'https://flows.example';
export const APPS = new Map([[AGENT, 'agent'], [FLOWS, 'flows']]);

/** A `fetch` that answers from `handler(url, init)` and keeps what it was asked. */
export function stubFetch(handler) {
  const calls = [];
  const impl = async (url, init = {}) => {
    const headers = Object.fromEntries(Object.entries(init.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]));
    const call = { url: String(url), method: init.method ?? 'GET', headers, body: init.body, signal: init.signal, mode: init.mode, redirect: init.redirect, credentials: init.credentials };
    calls.push(call);
    return handler(call);
  };
  return { impl, calls };
}

export const jsonResponse = (body, status = 200, headers = {}) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });

/** A response whose body is `parts`, one chunk each, `gapMs` apart. */
export function streamResponse(parts, { gapMs = 5, status = 200, onCancel } = {}) {
  let i = 0;
  const body = new ReadableStream({
    async pull(controller) {
      if (i >= parts.length) return controller.close();
      if (i > 0) await new Promise((resolve) => setTimeout(resolve, gapMs));
      controller.enqueue(new TextEncoder().encode(parts[i++]));
    },
    cancel() {
      onCancel?.();
    },
  });
  return new Response(body, { status, headers: { 'content-type': 'text/event-stream' } });
}

/** The client end of a connection. */
function makeClient(port) {
  const inbox = [];
  const waiters = [];
  let next = 1;
  port.onmessage = (event) => {
    inbox.push(event.data);
    for (const w of [...waiters]) w();
  };
  const waitFor = async (predicate, timeoutMs = 4000) => {
    const started = Date.now();
    for (;;) {
      const found = inbox.find(predicate);
      if (found) return found;
      if (Date.now() - started > timeoutMs) throw new Error(`no message matched in ${timeoutMs} ms; got ${JSON.stringify(inbox.map((m) => ({ t: m?.t, id: m?.id, ok: m?.ok })))}`);
      await new Promise((resolve) => {
        waiters.push(resolve);
        setTimeout(resolve, 20);
      });
    }
  };
  return {
    port,
    inbox,
    /** Send a request and wait for its reply (`{id, ok, ...}`). */
    async call(message, timeoutMs) {
      const id = next++;
      port.postMessage({ id, ...message });
      return waitFor((m) => m && m.id === id && 'ok' in m, timeoutMs);
    },
    /** Send a fetch and wait for its stream to end, or fail. */
    async fetch(message, { timeoutMs = 6000 } = {}) {
      const id = next++;
      port.postMessage({ id, op: 'fetch', ...message });
      await waitFor((m) => m && m.id === id && (m.t === 'end' || m.t === 'error'), timeoutMs);
      const events = inbox.filter((m) => m && m.id === id && typeof m.t === 'string');
      const head = events.find((e) => e.t === 'head');
      const error = events.find((e) => e.t === 'error')?.error;
      const bytes = events.filter((e) => e.t === 'chunk').map((e) => Buffer.from(e.bytes));
      return { id, events, head, error, text: Buffer.concat(bytes).toString('utf8'), chunks: bytes.length };
    },
    /** Send a fetch without waiting. */
    start(message) {
      const id = next++;
      port.postMessage({ id, op: 'fetch', ...message });
      return id;
    },
    raw: (data) => port.postMessage(data),
    waitFor,
    /** Everything received so far, as one string: for "the key is nowhere in it". */
    everything: () => JSON.stringify(inbox, (k, v) => (v instanceof ArrayBuffer ? Buffer.from(v).toString('latin1') : v)),
    close: () => port.close(),
  };
}

/**
 * @param {{ handler?: (call: object) => Response | Promise<Response>, record?: object, key?: string|null, apps?: Map<string,string>, now?: () => number }} [options]
 */
export async function brokerWorld(options = {}) {
  const fetchStub = stubFetch(options.handler ?? (() => jsonResponse({ data: [{ id: 'fake-chat' }] })));
  const h = makeHolder({ fetchImpl: fetchStub.impl, now: options.now, page: { protocol: 'https:', origin: 'https://providers.example' } });
  const key = options.key === undefined ? KEY : options.key;
  const saved = await h.store.save(input({ baseUrl: 'https://api.openai.com/v1', ...(options.record ?? {}) }), key ?? undefined);
  if (!saved.ok) throw new Error(`the test record was refused: ${JSON.stringify(saved)}`);
  const parent = { name: 'the parent window' };
  const broker = M.protocol.createBroker({ apps: options.apps ?? APPS, store: h.store, vault: h.vault, budget: h.budget, fetchImpl: fetchStub.impl, page: h.page, parent });
  // As the port's page does (broker-main.ts): a change to the list is told to every connected page.
  h.store.onChange(() => broker.notifyChanged());
  const clients = [];
  /** A hello from an allowed page (or, with `over`, from one that is not). */
  async function connect(over = {}) {
    const channel = new MessageChannel();
    const message = { origin: AGENT, source: parent, data: { op: 'hello', v: 1 }, ports: [channel.port2], ...over };
    const attached = broker.onWindowMessage(message);
    if (!attached) {
      channel.port1.close();
      channel.port2.close();
      return null;
    }
    const client = makeClient(channel.port1);
    clients.push(client);
    await client.waitFor((m) => m && m.t === 'hello');
    return client;
  }
  return {
    ...h,
    broker,
    parent,
    fetchStub,
    record: saved.record,
    connect,
    async close() {
      for (const c of clients) c.close();
      h.store.close();
    },
  };
}
