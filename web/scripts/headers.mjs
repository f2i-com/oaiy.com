/**
 * The response headers of the three hosts, from the templates in web/hosting/headers/ (design 3.8).
 *
 * A template is a `_headers` file (a path pattern, then indented `Name: value` lines; rules apply in the order written and a later
 * rule replaces the same header of an earlier one; a line `! Name` takes one away) with tokens for the origins the hosts have:
 * `{{AGENT_ORIGIN}}`, `{{FLOWS_ORIGIN}}` and `{{PROVIDERS_ORIGIN}}`, and `{{APP_ORIGINS}}`, the origins of every app allowed to embed
 * the providers origin, space-separated (`'none'` when there are none, so a mistake fails closed). The origins are a deployment's
 * choice and are never written into a template: a domain is not chosen yet, and the tests use `*.web.localhost`.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

export const HEADER_TEMPLATES = path.resolve(HERE, '..', 'hosting', 'headers');

const TOKENS = { AGENT_ORIGIN: 'agent', FLOWS_ORIGIN: 'flows', PROVIDERS_ORIGIN: 'providers' };

/** The template of one host (`agent`, `flows`, `providers`), unrendered. */
export function readTemplate(host) {
  return fs.readFileSync(path.join(HEADER_TEMPLATES, `${host}.headers`), 'utf8');
}

/**
 * A template with its origins filled in. `origins` names `agent`, `flows` and `providers`, and `apps` (a list) the origins allowed to
 * embed the providers origin; a token with no origin, or a `{{...}}` left over, is an error, so a template can never go out with a hole
 * in a policy.
 */
export function renderHeaders(text, origins) {
  const rendered = text.replace(/\{\{([A-Z_]+)\}\}/g, (whole, token) => {
    if (token === 'APP_ORIGINS') {
      const apps = origins?.apps;
      if (!Array.isArray(apps)) throw new Error(`no origins for ${whole}`);
      return apps.length > 0 ? apps.join(' ') : "'none'";
    }
    const key = TOKENS[token];
    const origin = key ? origins?.[key] : undefined;
    if (typeof origin !== 'string' || origin === '') throw new Error(`no origin for ${whole}`);
    return origin;
  });
  if (/\{\{|\}\}/.test(rendered)) throw new Error('a token is left in the headers');
  return rendered;
}
