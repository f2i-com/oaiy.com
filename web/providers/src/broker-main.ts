/**
 * The port (`/broker.html`): the hidden frame an app embeds. It draws nothing. It waits for a `hello` and answers over the port it
 * comes with, under the rules in protocol.ts, and tells every connected app when the list of providers changes.
 */
import { createContext } from './context';
import { createBroker, type PortLike } from './protocol';

const ctx = createContext();
const broker = createBroker({
  apps: ctx.apps,
  store: ctx.store,
  vault: ctx.vault,
  budget: ctx.budget,
  fetchImpl: ctx.fetchImpl,
  page: ctx.page,
  parent: window.parent,
});

window.addEventListener('message', (event: MessageEvent) => {
  broker.onWindowMessage({ origin: event.origin, source: event.source, data: event.data, ports: event.ports as unknown as PortLike[] });
});

// A page that has gone leaves its port behind and there is no event to say so: the quiet ones are closed (and told).
setInterval(() => broker.sweep(), 60_000);

// The top-level Providers page (and any other frame of this origin) changed something.
ctx.store.onChange(() => broker.notifyChanged());
