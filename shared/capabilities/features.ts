/**
 * The features the two apps show or hide by where they are, in one table (design 3.5 and 3.6).
 *
 * Each names what it NEEDS: OAIY Desktop behind a link (the person paired the page with it, or the page is in OAIY's own window,
 * which is given the desktop), a desktop or a server, or OAIY's own window itself (things that need the desktop's native commands,
 * which no link gives a browser tab). A feature whose need is not met cannot work in that mode, so the page does not show it: no
 * control that does nothing. Where the person can do something about it the page shows that instead (Connect), and a node that is
 * already in a flow says what it needs (`whyMissing`).
 *
 * Only the features an app looks at today are here. The providers, media and engine rows of design 3.6 join when the packages that
 * use them do (WA-05, WA-13).
 *
 * No feature reads a global: `derive` (derive.ts) turns this table and the page's host and links into what is on.
 */

/** What a feature needs. */
export type Requirement =
  /** OAIY Desktop: the page is in OAIY's own window, or was paired with one. */
  | 'desktop'
  /** OAIY Desktop or OAIY's server. */
  | 'desktop-or-server'
  /** OAIY's own window, whose page can call the desktop's native commands. No link gives a tab that. */
  | 'own-window';

export type FeatureId =
  // The Agent
  | 'phone'
  | 'calendar'
  | 'flowTools'
  | 'control'
  | 'setup'
  // The flow editor
  | 'browserNodes'
  | 'askAgent'
  | 'inputFolder'
  | 'packages'
  | 'dock';

export interface FeatureSpec {
  id: FeatureId;
  app: 'agent' | 'flows';
  requires: Requirement;
  /** What to say where it is missing: what it is, what it needs and why. One sentence. */
  says: string;
}

export const FEATURES: readonly FeatureSpec[] = [
  { id: 'phone', app: 'agent', requires: 'desktop', says: 'The phone needs OAIY Desktop: calls, texts and the Front desk run on the desktop, through its receptionist plugin.' },
  { id: 'calendar', app: 'agent', requires: 'desktop', says: 'The calendar tools need OAIY Desktop: the calendar is a plugin of the desktop.' },
  { id: 'flowTools', app: 'agent', requires: 'desktop-or-server', says: 'Flows as tools need OAIY Desktop or an OAIY server: the flows the Agent uses are kept and run there.' },
  { id: 'control', app: 'agent', requires: 'desktop', says: "OAIY's own tools need OAIY Desktop: they change its settings through its control interface." },
  { id: 'setup', app: 'agent', requires: 'desktop', says: 'Setting up OAIY needs OAIY Desktop: there is nothing to set up without it.' },
  { id: 'browserNodes', app: 'flows', requires: 'desktop-or-server', says: 'Browser nodes need OAIY Desktop or an OAIY server: they drive a browser that it runs.' },
  { id: 'askAgent', app: 'flows', requires: 'desktop-or-server', says: 'Ask the Agent needs OAIY Desktop or an OAIY server: the Agent that answers is its.' },
  { id: 'inputFolder', app: 'flows', requires: 'desktop-or-server', says: 'A folder input needs OAIY Desktop or an OAIY server: a browser page cannot read a folder of the computer by its path.' },
  { id: 'packages', app: 'flows', requires: 'own-window', says: "Packages need OAIY's own window: loading a flow package uses the desktop's own commands, which a browser tab does not have." },
  { id: 'dock', app: 'flows', requires: 'desktop', says: "The engine's dock needs OAIY Desktop: it shows the desktop's engine and its address." },
];

const BY_ID = new Map(FEATURES.map((f) => [f.id, f]));

export function featureSpec(id: FeatureId): FeatureSpec {
  const spec = BY_ID.get(id);
  if (!spec) throw new Error(`no feature called ${id}`);
  return spec;
}

/** What to say where a feature is missing: what it is and what it needs. The page adds what the person can do about it. */
export function whyMissing(id: FeatureId): string {
  return featureSpec(id).says;
}
