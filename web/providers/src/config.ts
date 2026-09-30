/**
 * What a deployment tells the providers origin: which app origins may embed it and speak to it. It is written into each page when
 * the folders are assembled (web/scripts/assemble.mjs, from the origins the operator chose), as
 * `<meta name="oaiy-apps" content="agent=https://agent.example flows=https://flows.example">`, because no domain is chosen in
 * this code. A page with no such tag, or a malformed one, allows nobody.
 */
import { parseAppOrigins } from '@oaiy/shared/broker/protocol';

export const APPS_META = 'oaiy-apps';

/** Origin to app name, from the page's own meta tag. */
export function readApps(doc: Pick<Document, 'querySelector'> = document): Map<string, string> {
  return parseAppOrigins(doc.querySelector(`meta[name="${APPS_META}"]`)?.getAttribute('content'));
}

/** The names of the apps, for the page that shows their budgets. */
export function appNames(apps: ReadonlyMap<string, string>): string[] {
  return [...new Set(apps.values())].sort();
}
