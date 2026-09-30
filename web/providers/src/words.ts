/**
 * Words the Providers page and the embedded modal say about keys, in one place so that both say the same and a test can hold them to what
 * is true. (They are plain strings: they are set as text, never as markup.)
 */
import type { ProviderSummary } from '@oaiy/shared/providers/types';

/**
 * What the Providers page says an app can and cannot do with a key. An app can USE a key (a call made with it, within the limits), and
 * that is a spend; what it cannot do is read it, change it or point it somewhere else. "Never see" was too much: the key is in this
 * site's memory while a call is made, and the page must say what keeping it does not protect against (`KEYS_STORAGE`).
 */
export const KEYS_LEAD =
  'The AI services and servers you use, in one place. Your keys stay on this page’s own site. An app can ask this site to make a call with one of them, up to the limits below, but it is not given the key and cannot change where it goes.';

/** What keeping keys on this device does, and what it is not. */
export const KEYS_STORAGE =
  'Your keys are stored encrypted on this device, and only this site can open them, which keeps them apart from the apps. That is not protection against someone who has this computer or a copy of this browser’s profile, or against a browser extension that can read this site: while a call is being made the key is in this site’s memory. For a copied profile, set a passphrase (coming).';

/**
 * What a provider can do with a key it is given, which no page here can stop. A reply is passed to the app that asked as the provider sent
 * it (a stream, a model list fetched as any other path): the holder does not read it for a key.
 */
export const KEYS_PROVIDER =
  'A provider you give a key to can see it, so give a key only to a provider you trust with it. This page cannot stop a provider from echoing the key back in a reply: what a provider answers goes to the app that asked, as the provider sent it.';

/** What a provider's row says about its key. A key that is stored and cannot be opened says so, and what to do: it is not "stored". */
export function keyText(summary: Pick<ProviderSummary, 'hasKey' | 'keyUnreadable'>): string {
  if (summary.keyUnreadable) return 'key unreadable: re-enter it';
  return summary.hasKey ? 'key stored' : 'no key';
}
