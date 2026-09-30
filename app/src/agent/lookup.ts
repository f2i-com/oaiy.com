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
import { OAIY_ORIGIN } from './media';

/**
 * Where to look for OAIY as the page starts, or null for nowhere.
 *
 * - `asked`: an address in the page's own address (`?oaiy=<address>`), a way to point the page at OAIY somewhere else. It is honoured
 *   in OAIY's windows and in an automated browser (the tests use it to point the page at a stand-in); a visitor's tab ignores it, so
 *   that a link someone sends cannot make the page look at the visitor's network.
 * - `discovered`: the origin of the OAIY found before, which is a link the person saved.
 * - `setByHand`: the media service was typed in by hand, so what OAIY says is not looked for again.
 *
 * An automated browser looks only when asked, so a test never talks to a desktop it did not start.
 */
export function whereToLook(input: { host: Host; asked: string | null; discovered?: string; setByHand: boolean }): string | null {
  const { host } = input;
  const asked = input.asked !== null && (host.kind !== 'browser' || host.automated) ? input.asked : null;
  const where = asked ?? (host.automated ? null : (input.discovered ?? (looksOnLoad(host) ? OAIY_ORIGIN : null)));
  if (!where || (input.setByHand && !asked)) return null;
  return where;
}

/** Where to look again, now and then, for what OAIY's Engines has chosen: the OAIY found before, else (in OAIY's own windows) its usual address. */
export function whereToFollow(input: { host: Host; discovered?: string }): string | null {
  if (input.host.automated) return null;
  return input.discovered ?? (looksOnLoad(input.host) ? OAIY_ORIGIN : null);
}
