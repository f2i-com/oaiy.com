/**
 * Where the page looks for OAIY (the engine's gateway, port 8080) without being asked to, and where it does not.
 *
 * OAIY's own window and the desktop shell look at OAIY's usual address as the page loads, as they always have. A tab in a browser
 * does not: on a public address that look is a permission prompt for the visitor (Chrome's Local Network Access) and a request to
 * an address the page has no business with. A tab looks where the person said (the OAIY they found before, kept in the settings)
 * and, when they press Settings → Find OAIY, at the address they give. Nothing else goes out from the page to loopback or the LAN.
 *
 * The decisions are here, pure, so a test can hold them: main.ts asks and does what it is told.
 */
import { looksOnLoad, type Host } from '@oaiy/shared/capabilities/host';
import { OAIY_ORIGIN, originOf, type MediaSettings } from './media';
import type { ProviderConfig } from './providers/types';

/** Where to look, and whether the API key the person typed may go with the request. */
export interface LookTarget {
  origin: string;
  /**
   * False for an address that came from the page's own address (`?oaiy=`): a key the person typed for their own OAIY is not sent to
   * whatever a link names, and is not kept with what that address says about itself.
   */
  withKey: boolean;
}

/**
 * Where to look for OAIY as the page starts, or null for nowhere.
 *
 * - `asked`: an address in the page's own address (`?oaiy=<address>`). It is a seam for the tests, which point the page at a stand-in
 *   for OAIY, and it is honoured ONLY in an automated browser: anywhere else a link (one a model wrote into its reply, one someone
 *   sent) would choose where the page looks, and could carry the person's key there.
 * - `discovered`: the origin of the OAIY found before, which is a link the person saved.
 * - `setByHand`: the media service was typed in by hand, so what OAIY says is not looked for again.
 *
 * An automated browser looks only when asked, so a test never talks to a desktop it did not start.
 */
export function whereToLook(input: { host: Host; asked: string | null; discovered?: string; setByHand: boolean }): LookTarget | null {
  const { host } = input;
  if (host.automated) return input.asked ? { origin: input.asked, withKey: false } : null;
  const origin = input.discovered ?? (looksOnLoad(host) ? OAIY_ORIGIN : null);
  if (!origin || input.setByHand) return null;
  return { origin, withKey: true };
}

/** Where to look again, now and then, for what OAIY's Engines has chosen: the OAIY found before, else (in OAIY's own windows) its usual address. */
export function whereToFollow(input: { host: Host; discovered?: string }): string | null {
  if (input.host.automated) return null;
  return input.discovered ?? (looksOnLoad(input.host) ? OAIY_ORIGIN : null);
}

/** An origin, or '' for an address that is not one. */
const originOr = (url?: string | null): string => {
  try {
    return url ? originOf(url) : '';
  } catch {
    return '';
  }
};

/** Whether an origin is this computer's own (OAIY's engines on it). */
const loopback = (origin: string): boolean => /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$/i.test(origin);

/**
 * Where OAIY's engines now serve their gateway (`gateway`, as OAIY Desktop says), where the page knows this computer's OAIY
 * at another address (`known`: the one found before, else the usual one): the engines took a free port, theirs being
 * held by another program. Null where nothing moved, OAIY Desktop does not say, or the OAIY the page knows is another
 * computer's (the person's own choice, not moved).
 */
export function movedGateway(known: string | null | undefined, gateway: string | null | undefined): { from: string; to: string } | null {
  const to = originOr(gateway);
  const from = originOr(known ?? OAIY_ORIGIN);
  return to && from && to !== from && loopback(to) && loopback(from) ? { from, to } : null;
}

/**
 * What the page keeps of OAIY moved from `from` to `to` (an origin each): the OAIY it found, the media service's address
 * and the chat providers at it, each keeping its path (`/v1`). Whether anything moved.
 */
export function moveOaiy(media: MediaSettings, providers: ProviderConfig[], from: string, to: string): { media: MediaSettings; changed: boolean } {
  const moved = (url: string): string => (originOr(url) === from ? `${to}${url.slice(from.length)}` : url);
  let changed = false;
  for (const p of providers) {
    if (!p.baseUrl || originOr(p.baseUrl) !== from) continue;
    p.baseUrl = moved(p.baseUrl);
    changed = true;
  }
  let next = media;
  if (media.discovered?.origin === from || (media.baseUrl && originOr(media.baseUrl) === from)) {
    next = { ...media, baseUrl: media.baseUrl ? moved(media.baseUrl) : media.baseUrl, ...(media.discovered ? { discovered: { ...media.discovered, origin: media.discovered.origin === from ? to : media.discovered.origin } } : {}) };
    changed = true;
  }
  return { media: next, changed };
}
