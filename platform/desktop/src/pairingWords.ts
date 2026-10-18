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

/** Who asked: the site, or a program here when no site did (a native caller sends no Origin). */
export function pairingFrom(origin: string | null | undefined): string {
  return origin ? ` · ${origin}` : ' · a program on this computer';
}

/** The kind of a connection, for its row. */
export function accessName(access: PairingAccess | undefined): string {
  return access === 'agent' ? 'Agent' : 'App';
}
