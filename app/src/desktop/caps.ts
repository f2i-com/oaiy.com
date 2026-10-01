/**
 * What the Agent can do here, from where it is and the desktop it is paired with (shared/capabilities).
 *
 * The phone, the calendar, flows as tools, OAIY's own tools and "Set up OAIY" are OAIY Desktop's: they are on only while there is one
 * behind the page. It is there in OAIY's own window (which is given the desktop) and in a page that was paired with one; a page with
 * neither has none of them, and shows the pairing chip instead. This used to be decided in several places, each by its own test
 * (`IN_OAIY`, `desktop != null`, the modules); now it is decided here, once, and each control asks.
 *
 * A control is `caps.x && <what it needed before>`: what the desktop says about its modules (whether a phone plugin is on, whether a
 * calendar is) still decides the rest, so nothing that showed before is hidden.
 */
import { derive, type Caps, type Link } from '@oaiy/shared/capabilities/derive';
import type { Host } from '@oaiy/shared/capabilities/host';

/** The desktop the page is paired with (given by OAIY's window, or paired from a tab), or null. */
export interface PairedDesktop {
  origin: string;
}

export function agentCaps(host: Host, desktop: PairedDesktop | null): Caps {
  const links: Link[] = desktop ? [{ kind: 'desktop', base: desktop.origin }] : [];
  return derive({ host, links });
}
