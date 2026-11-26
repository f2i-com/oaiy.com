import type { PairingAccess } from './api';

/**
 * What a pairing grants, in the words the prompt and the Connections page both use (the desktop's
 * `bridge/pairing.rs` says what each kind reaches): an app, flows (which can run programs), plugins and
 * AI models; an Agent, those and the phone, the calendar, the contacts and OAIY's settings.
 */
export function pairingGrants(access: PairingAccess | undefined): string {
  return access === 'agent'
    ? 'wants to be your OAIY Agent: everything an app can do, and your calls, texts, calendar and contacts, and changing OAIY’s settings with the Agent’s tools. Approve only your own OAIY Agent page.'
    : 'wants to run your flows, use your plugins (the phone’s commands too) and your AI models through OAIY. A flow can run programs on this computer, so approve only an app you trust.';
}

/**
 * Who asks: the site, by the address the browser gave for it, or a program here when no site did (a native caller
 * sends no Origin). It is the one part of a request the asker did not write, so it is what a request is said to be
 * from, on a line of its own; the desktop refuses a request whose address is not one (`asking_origin`, bridge/pairing.rs).
 */
export function pairingWho(origin: string | null | undefined): string {
  return origin || 'a program on this computer';
}

/**
 * What the asker calls itself: its product's name, and its label where it gave one that says more. Its own words,
 * said as its own: they were the prompt's heading, where a page that called itself another site's name read as that site.
 */
export function pairingCalls(product: string, label: string | null | undefined, origin: string | null | undefined): string {
  return label && label !== product && label !== origin ? `It calls itself “${label}” (${product}).` : `It calls itself “${product}”.`;
}

/** The kind of a connection, for its row. */
export function accessName(access: PairingAccess | undefined): string {
  return access === 'agent' ? 'Agent' : 'App';
}
