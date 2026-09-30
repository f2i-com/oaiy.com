/**
 * What the editor can do here (shared/capabilities): from the window it is in and whether its person has linked it to OAIY Desktop.
 *
 * In OAIY's own window the desktop is given, so every feature of the desktop's is on, as it always was. In a tab it is on once the
 * person has linked the tab to a desktop (lib/desktopLink.ts: Connect, or an engine address given in Settings), and off before: no
 * Packages section, no dock, no palette entries for the nodes that drive the desktop's browser or ask its Agent, and the palette and
 * the node notices say what is missing. A link is what was saved, not whether the desktop answers this minute.
 */
import { derive, type Caps, type Link } from '@oaiy/shared/capabilities/derive';
import { pageHost } from '@oaiy/shared/capabilities/host';
import { hasSavedDesktopLink, subscribeDesktopLink } from './desktopLink';
import { getEngineBase } from './engineEndpoint';

// Which window this is, read as the page loads, before any flow or other script of the person's runs (shared/capabilities/host.ts).
pageHost();

let cached: { key: string; caps: Caps } | null = null;

/** The editor's capabilities now. The same object is returned until something that decides them changes (a React snapshot needs that). */
export function currentCaps(): Caps {
  const host = pageHost();
  const links: Link[] = hasSavedDesktopLink() ? [{ kind: 'desktop', base: getEngineBase() }] : [];
  const key = `${host.kind}|${host.hosted ?? ''}|${links.map((l) => `${l.kind}@${l.base}`).join(',')}`;
  if (cached?.key !== key) cached = { key, caps: derive({ host, links }) };
  return cached.caps;
}

/** Called when the link is made, forgotten or its address changed. Returns an unsubscribe. */
export function subscribeCaps(listener: () => void): () => void {
  return subscribeDesktopLink(listener);
}
