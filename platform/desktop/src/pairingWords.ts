import type { PairingAccess } from './api';
import { isInvisible } from './visibleText';

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
 * A name an app gave itself, as it may be shown: without the characters a person cannot see (controls, a line break
 * among them; zero-width characters; the ones that turn the direction of the text around, with which a name reads as
 * something it does not say). The desktop does this to a label as a request comes in (`shown_label`, bridge/pairing.rs);
 * a connection made before it did is kept as it was written, so what is kept is cleaned here as it is shown. Empty
 * where nothing is left.
 */
export function shownName(name: string | null | undefined): string {
  let shown = '';
  for (const ch of name ?? '') {
    const code = ch.codePointAt(0) ?? 0;
    if (code <= 0x1f || (code >= 0x7f && code <= 0x9f) || isInvisible(code)) continue;
    shown += ch;
  }
  return shown.trim();
}

/**
 * The names the asker gave itself, in quotation marks as its own words: its label where it gave one that says more
 * than its product's name, and the product's name.
 */
export function pairingCalled(product: string, label: string | null | undefined, origin: string | null | undefined): string {
  const [named, called] = [shownName(product) || 'app', shownName(label)];
  return called && called !== named && called !== origin ? `“${called}” (${named})` : `“${named}”`;
}

/**
 * What the asker calls itself: its product's name, and its label where it gave one that says more. Its own words,
 * said as its own: they were the prompt's heading, where a page that called itself another site's name read as that site.
 */
export function pairingCalls(product: string, label: string | null | undefined, origin: string | null | undefined): string {
  return `It calls itself ${pairingCalled(product, label, origin)}.`;
}

/** The kind of a connection, for its row. */
export function accessName(access: PairingAccess | undefined): string {
  return access === 'agent' ? 'Agent' : 'App';
}
