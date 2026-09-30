/**
 * What a page can do here, derived and not asked (design 3.4 and 3.5): from the window it is in (host.ts) and the links its person
 * has saved to OAIY Desktop or to an OAIY server, in the mode that follows.
 *
 *   standalone  no desktop and no server: the page runs on its own, in the browser
 *   paired      a desktop is behind the page: it is in OAIY's own window (given the desktop), or was paired with one
 *   server      OAIY's own server serves the page (`<meta name="oaiy-hosted" content="cookie">`), or a link to one is saved
 *
 * A feature is on when what it needs (features.ts) is there. The desktop and the server are looked for independently of the mode,
 * which is only a name for the sum: a page paired with a desktop AND signed in to a server has the features of both.
 *
 * A link is what the person saved, not a claim that it answers right now: a desktop that is switched off for a minute does not turn
 * the phone off (the page says it is offline; nothing is hidden and shown again). That the desktop answers, and to which scopes, is
 * asked of the desktop and is not decided here.
 *
 * Pure: the same input gives the same answer, and no global is read.
 */
import { looksOnLoad, type Host, type HostKind } from './host';
import { FEATURES, type FeatureId, type Requirement } from './features';

export type Mode = 'standalone' | 'paired' | 'server';

/** What a person saved: "Connect OAIY Desktop", "Add a server". */
export interface Link {
  kind: 'desktop' | 'server';
  /** Where it is (an origin). Not used to decide anything: the page reaches it, not this file. */
  base: string;
}

export interface DeriveInput {
  host: Host;
  links: readonly Link[];
}

export interface Caps {
  readonly mode: Mode;
  readonly host: HostKind;
  readonly hosted: Host['hosted'];
  /** May the page look for OAIY at its usual addresses as it loads (rule R1: only OAIY's own windows and the desktop shell)? */
  readonly looksOnLoad: boolean;
  /** Each feature: on, or hidden because it cannot work here. */
  readonly features: Readonly<Record<FeatureId, boolean>>;
  /**
   * What a flow's Database nodes store lasts as long as the tab: the browser's SQLite has no persistent storage on the main thread
   * (Chromium answers "Missing required OPFS APIs"), so it falls back to memory. OAIY's own windows run the same build.
   */
  readonly dataSessionOnly: boolean;
}

/** Whether a need is met, given what is behind the page. */
function met(need: Requirement, has: { desktop: boolean; server: boolean; ownWindow: boolean }): boolean {
  switch (need) {
    case 'desktop':
      return has.desktop;
    case 'desktop-or-server':
      return has.desktop || has.server;
    case 'own-window':
      return has.ownWindow;
  }
}

export function derive(input: DeriveInput): Caps {
  const { host, links } = input;
  const ownWindow = host.kind === 'oaiy-window';
  const has = {
    // OAIY's own window is given the desktop, so it needs no link.
    desktop: ownWindow || links.some((l) => l.kind === 'desktop'),
    server: host.hosted === 'cookie' || links.some((l) => l.kind === 'server'),
    ownWindow,
  };
  const features = Object.fromEntries(FEATURES.map((f) => [f.id, met(f.requires, has)])) as Record<FeatureId, boolean>;
  return Object.freeze({
    mode: has.server ? 'server' : has.desktop ? 'paired' : 'standalone',
    host: host.kind,
    hosted: host.hosted,
    looksOnLoad: looksOnLoad(host),
    features: Object.freeze(features),
    // Said of a tab. OAIY's own window runs the same build with the same limit, but it is left as it was: labelling it is a change to
    // the desktop's window, which this one does not make.
    dataSessionOnly: host.kind === 'browser',
  });
}

/** Whether a feature is on. */
export function has(caps: Caps, id: FeatureId): boolean {
  return caps.features[id];
}
