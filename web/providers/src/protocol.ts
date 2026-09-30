/**
 * The holder's side of the port protocol (design 3.2; the rules and shapes are in shared/broker/protocol.ts).
 *
 * A page speaks to it in two steps. A `hello` arrives as a window message, and is accepted only from a page whose ORIGIN is in the
 * compiled list (the `oaiy-apps` meta tag), only when its source is this frame's parent (not a sibling frame, not a sandboxed frame
 * whose origin is `null`), and only with exactly one port. Everything after goes over that port, and the operations are the nine in
 * `OPS`: none of them returns, edits or redirects a key, or adds, edits or deletes a provider. The one thing a page can change is a
 * provider's model.
 *
 * Each app has a request budget an hour, counted here at the one place a request leaves (a `fetch`, or a call the holder makes for
 * `models`, `test` or `probe`), so a page cannot spend more by asking another way.
 */
import {
  MAX_IN_FLIGHT,
  OPS,
  PROTOCOL_VERSION,
  appForOrigin,
  errorBody,
  parseHello,
  parseRequest,
  type ErrorBody,
  type FetchRequest,
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
  connections(): number;
}

export function createBroker(deps: BrokerDeps): Broker {
  const ports = new Set<PortLike>();
  const fetcher = createFetcher({ store: deps.store, budget: deps.budget, fetchImpl: deps.fetchImpl, page: deps.page });

  function attach(port: PortLike, app: string): void {
    ports.add(port);
    const inFlight = new Map<number, AbortController>();

    const send = (message: Reply | StreamEvent | { t: 'hello'; v: number; ops: readonly string[]; app: string } | { t: 'changed' }, transfer?: Transferable[]): void => {
      try {
        port.postMessage(message, transfer);
      } catch {
        // a port whose page has gone
      }
    };
    const ok = (id: number, result: unknown): void => send({ id, ok: true, result });
    const refuse = (id: number, error: ErrorBody): void => send({ id, ok: false, error });

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

    async function handle(data: unknown): Promise<void> {
      const parsed = parseRequest(data);
      if (!parsed.ok) {
        if (parsed.id !== null) refuse(parsed.id, errorBody(parsed.code, parsed.message));
        return;
      }
      const { id, request } = parsed;
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

    port.onmessage = (event) => void handle(event.data);
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
      for (const port of ports) {
        try {
          port.postMessage({ t: 'changed' });
        } catch {
          ports.delete(port);
        }
      }
    },

    connections: () => ports.size,
  };
}
