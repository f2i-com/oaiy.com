/**
 * The holder's side of the port protocol (design 3.2; the rules and shapes are in shared/broker/protocol.ts).
 *
 * A page speaks to it in two steps. A `hello` arrives as a window message, and is accepted only from a page whose ORIGIN is in the
 * compiled list (the `oaiy-apps` meta tag), only when its source is this frame's parent (not a sibling frame, not a sandboxed frame
 * whose origin is `null`), and only with exactly one port. Everything after goes over that port, and the operations are the nine in
 * `OPS`: none of them returns, edits or redirects a key, or adds, edits or deletes a provider. The one thing a page can change is a
 * provider's model, and only to one the provider itself listed.
 *
 * Each app has a request budget an hour, counted here at the one place a request leaves (a `fetch`, or a call the holder makes for
 * `models`, `test` or `probe`), so a page cannot spend more by asking another way.
 *
 * A compromised app can also flood the port, and what that costs is the holder's own work. So every connection is bounded (protocol.ts
 * in shared/): operations being worked on at once are capped and a further one is refused `busy` at once, not queued behind the rest;
 * every operation of any kind is rate limited; an app may hold only so many connections, the least recently active closed to make room;
 * a connection that has been quiet too long is closed (a MessagePort has no close event, so a page that has gone is only ever seen as
 * quiet); and a connection that keeps being refused is not answered past a number a second, which is cheaper than the refusals.
 */
import {
  IDLE_CLOSE_MS,
  MAX_CONNECTIONS_PER_APP,
  MAX_IN_FLIGHT,
  MAX_PENDING_OPS,
  OPS,
  OPS_BURST,
  OPS_PER_SECOND,
  PROTOCOL_VERSION,
  REFUSALS_ANSWERED_PER_SECOND,
  appForOrigin,
  errorBody,
  parseHello,
  parseRequest,
  type ErrorBody,
  type FetchRequest,
  type PortRequest,
  type Push,
  type Reply,
  type StatusBody,
  type StreamEvent,
} from '@oaiy/shared/broker/protocol';
import type { ProviderRecord } from '@oaiy/shared/providers/types';
import type { Budget } from './budget';
import { createFetcher } from './fetcher';
import type { PageInfo } from './net';
import type { ProviderStore } from './store';
import { createTester } from './test';
import type { ListableVault } from './vault';

/** The part of a MessagePort the holder uses (a MessageChannel's port in a browser, in a test, and in Node). */
export interface PortLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: { data: unknown }) => void) | null;
  start?(): void;
  close?(): void;
}

export interface BrokerDeps {
  /** Origin to app: who may speak. Empty allows nobody. */
  apps: ReadonlyMap<string, string>;
  store: ProviderStore;
  vault: ListableVault;
  budget: Budget;
  fetchImpl: typeof fetch;
  page: PageInfo;
  /** `window.parent`: the only window a `hello` may come from. */
  parent: unknown;
  /** The clock, in milliseconds, for the rate limit and the idle time. Default `performance.now()`: monotonic, unlike the wall clock. A test hands in its own. */
  now?: () => number;
}

/** A window `message` event, as far as the holder reads one. */
export interface WindowMessage {
  origin: string;
  source: unknown;
  data: unknown;
  ports: readonly PortLike[];
}

export interface Broker {
  /** A window message: true when it was a `hello` from an allowed page and a connection was made. */
  onWindowMessage(event: WindowMessage): boolean;
  /** Tell every connected app the list of providers changed. */
  notifyChanged(): void;
  /** Close the connections that have been quiet for `IDLE_CLOSE_MS`, telling each. The page calls it every minute. */
  sweep(): number;
  connections(): number;
}

interface Connection {
  app: string;
  lastActive: number;
  /** Whether a request of this connection is being worked on: a stream that runs for a long time is not a quiet connection. */
  working(): boolean;
  close(reason: 'idle' | 'replaced'): void;
}

export function createBroker(deps: BrokerDeps): Broker {
  // A clock that only goes forward: the wall clock (`Date.now`) can be set back an hour, and then the rate limit would see a negative
  // time (every connection locked for that hour) and the idle timer would never run out. What is counted here is elapsed time.
  const now = deps.now ?? (() => performance.now());
  const connections = new Map<PortLike, Connection>();
  const fetcher = createFetcher({ store: deps.store, budget: deps.budget, fetchImpl: deps.fetchImpl, page: deps.page });

  function attach(port: PortLike, app: string): void {
    // An app that holds too many connections loses its least recently active: a page that has gone leaves its port behind, and a
    // hostile one that opens many is held to its own share.
    const mine = [...connections.entries()].filter(([, c]) => c.app === app);
    if (mine.length >= MAX_CONNECTIONS_PER_APP) {
      mine.sort((a, b) => a[1].lastActive - b[1].lastActive);
      for (const [, c] of mine.slice(0, mine.length - MAX_CONNECTIONS_PER_APP + 1)) c.close('replaced');
    }

    const inFlight = new Map<number, AbortController>();
    let pending = 0;
    // A token bucket for every operation of any kind, and a count of the refusals answered in the current second.
    let tokens = OPS_BURST;
    let refilled = now();
    let refusalsSecond = Math.floor(refilled / 1000);
    let refusals = 0;

    const send = (message: Reply | StreamEvent | Push | { t: 'hello'; v: number; ops: readonly string[]; app: string }, transfer?: Transferable[]): void => {
      try {
        port.postMessage(message, transfer);
      } catch {
        // a port whose page has gone
      }
    };
    const ok = (id: number, result: unknown): void => send({ id, ok: true, result });
    const refuse = (id: number, error: ErrorBody): void => send({ id, ok: false, error });
    const refuseRequest = (request: PortRequest | null, id: number, error: ErrorBody): void => {
      // Past a number of refusals a second, a flood is not answered.
      const second = Math.floor(now() / 1000);
      if (second !== refusalsSecond) {
        refusalsSecond = second;
        refusals = 0;
      }
      if (++refusals > REFUSALS_ANSWERED_PER_SECOND) return;
      if (request?.op === 'fetch') send({ id, t: 'error', error });
      else refuse(id, error);
    };

    const connection: Connection = {
      app,
      lastActive: now(),
      working: () => inFlight.size > 0 || pending > 0,
      close(reason) {
        send({ t: 'closed', reason });
        connections.delete(port);
        port.onmessage = null;
        try {
          port.close?.();
        } catch {
          // already closed
        }
        for (const controller of inFlight.values()) controller.abort();
      },
    };
    connections.set(port, connection);

    const providerFor = async (id: number, providerId: string): Promise<ProviderRecord | null> => {
      const record = await deps.store.get(providerId);
      if (!record || record.via !== 'broker') {
        refuse(id, errorBody('unknown-provider', 'There is no such provider.'));
        return null;
      }
      return record;
    };

    const tester = createTester({
      fetchImpl: deps.fetchImpl,
      page: deps.page,
      key: (record) => deps.store.key(record.id),
      onModels: (record, ids) => deps.store.rememberModels(record.id, ids),
      take: async (bytes) => {
        const taken = await deps.budget.take(app, bytes);
        return taken.ok ? { ok: true } : taken;
      },
    });

    async function handle(id: number, request: PortRequest): Promise<void> {
      try {
        switch (request.op) {
          case 'list':
            return ok(id, await deps.store.summaries());
          case 'status': {
            const state = await deps.vault.ready();
            const body: StatusBody = { mode: state.mode, locked: state.locked, engine: { state: 'none', model: null, progress: null } };
            return ok(id, body);
          }
          case 'ui.open':
            // The holder opens nothing: it tells the page which of its own two ways to show the providers was asked for.
            return ok(id, { action: request.target });
          case 'setModel': {
            const outcome = await deps.store.setModel(request.provider, request.model, app);
            if (outcome === 'no-provider') return refuse(id, errorBody('unknown-provider', 'There is no such provider.'));
            // A model the provider has not listed is refused: what the Providers page then shows is the provider's words, not the app's.
            if (outcome === 'unknown-model') return refuse(id, errorBody('unknown-model', 'That model is not in the provider’s own list. Ask for the models first.'));
            return ok(id, {});
          }
          case 'abort':
            inFlight.get(request.target)?.abort();
            return ok(id, {});
          case 'models':
          case 'test':
          case 'probe': {
            const record = await providerFor(id, request.provider);
            if (!record) return;
            return ok(id, await tester[request.op](record));
          }
          case 'fetch':
            return startFetch(id, request);
        }
      } catch {
        // Whatever went wrong, a page is told that it did and nothing more: an exception's text is not for a page.
        if (request.op === 'fetch') send({ id, t: 'error', error: errorBody('internal', 'The request could not be made.') });
        else refuse(id, errorBody('internal', 'The operation could not be done.'));
      }
    }

    /** Whether a message is let in: it costs the holder something to answer, so it is counted first. */
    function admit(request: PortRequest): 'ok' | 'rate' | 'pending' {
      const at = now();
      tokens = Math.min(OPS_BURST, tokens + ((at - refilled) / 1000) * OPS_PER_SECOND);
      refilled = at;
      if (tokens < 1) return 'rate';
      tokens -= 1;
      // A fetch is long-lived and has its own cap; an abort is cheap and is what ends work: neither waits behind the others.
      if (request.op !== 'fetch' && request.op !== 'abort' && pending >= MAX_PENDING_OPS) return 'pending';
      return 'ok';
    }

    function startFetch(id: number, request: FetchRequest): void {
      if (inFlight.size >= MAX_IN_FLIGHT || inFlight.has(id)) {
        send({ id, t: 'error', error: errorBody('too-many', 'Too many requests are open at once.') });
        return;
      }
      const controller = new AbortController();
      inFlight.set(id, controller);
      void fetcher
        .run(
          app,
          request,
          (body) => {
            const event = { id, ...body } as StreamEvent;
            if (event.t === 'chunk') send(event, [event.bytes]);
            else send(event);
          },
          controller.signal,
        )
        .finally(() => inFlight.delete(id));
    }

    port.onmessage = (event) => {
      connection.lastActive = now();
      const parsed = parseRequest(event.data);
      if (!parsed.ok) {
        if (parsed.id !== null) refuseRequest(null, parsed.id, errorBody(parsed.code, parsed.message));
        return;
      }
      const { id, request } = parsed;
      const verdict = admit(request);
      if (verdict !== 'ok') {
        refuseRequest(request, id, errorBody('busy', verdict === 'rate' ? 'Too many requests a second. Slow down.' : 'Too many operations are being worked on. Try again in a moment.'));
        return;
      }
      if (request.op === 'fetch' || request.op === 'abort') {
        void handle(id, request);
        return;
      }
      pending++;
      void handle(id, request).finally(() => {
        pending--;
      });
    };
    port.start?.();
    send({ t: 'hello', v: PROTOCOL_VERSION, ops: OPS, app });
  }

  return {
    onWindowMessage(event) {
      // The origin is what the browser says the sender is: not a field of the message. `null` (a sandboxed page) is nobody.
      const app = appForOrigin(deps.apps, event.origin);
      if (app === null) return false;
      // Only this frame's parent: not another frame of the same page, which has the same origin.
      if (event.source !== deps.parent) return false;
      if (parseHello(event.data) === null) return false;
      if (event.ports.length !== 1) return false;
      attach(event.ports[0], app);
      return true;
    },

    notifyChanged() {
      for (const port of connections.keys()) {
        try {
          port.postMessage({ t: 'changed' });
        } catch {
          connections.delete(port);
        }
      }
    },

    sweep() {
      const at = now();
      let closed = 0;
      for (const connection of [...connections.values()]) {
        if (at - connection.lastActive > IDLE_CLOSE_MS && !connection.working()) {
          connection.close('idle');
          closed++;
        }
      }
      return closed;
    },

    connections: () => connections.size,
  };
}
