/**
 * The browser harness of the web app's tests (design 8): the hosts server, a real Chromium, and the recorder that keeps a test
 * from ever reaching the product's own ports.
 *
 *     node --test tests/e2e/cases/
 *
 * Nothing here talks to OAIY Desktop, the engines or anything the owner runs: a request to one of the product's ports (17972,
 * 17872, 17973, 8080, 7860, 8783, 9333) on this computer is refused by the browser before it leaves, and recorded, so a test can
 * say "none was made" (the pattern of platform/ui/tests/pwa-e2e.mjs). Every scenario gets a browser context of its own with no
 * profile shared with any browser anyone uses.
 *
 * Which browser: Playwright's own Chromium, from %LOCALAPPDATA%\ms-playwright (the version the installed `playwright` package asks
 * for), in Chromium's new headless mode; `WEB_E2E_CHANNEL=chrome` (or `msedge`) runs the test on the installed Chrome or Edge instead.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { readTemplate, renderHeaders } from '../../scripts/headers.mjs';
import { startHosts } from './hosts.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const FIXTURES = path.join(HERE, 'fixtures');

/** The product's own ports: never reached from here. */
export const OWN_PORTS = new Set(['17972', '17872', '17973', '8080', '7860', '8783', '9333']);

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Poll `fn` until it returns something truthy (its value is returned), or fail with `what` after `timeout` ms. */
export async function waitFor(fn, { timeout = 8000, interval = 50, what = 'the condition' } = {}) {
  const started = Date.now();
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() - started > timeout) throw new Error(`timed out after ${timeout} ms waiting for ${what}`);
    await sleep(interval);
  }
}

/**
 * A browser. `isolateOrigins` asks Chromium to give those origins a process of their own (`--isolate-origins`): the providers
 * frame is then out of process from the app that embeds it, which is what a test that reads the APP's memory needs (two
 * same-site frames share a renderer, and so a heap, by default).
 */
export async function launchBrowser({ isolateOrigins = [], headless = true } = {}) {
  const args = [];
  if (isolateOrigins.length > 0) args.push(`--isolate-origins=${isolateOrigins.join(',')}`);
  const channel = process.env.WEB_E2E_CHANNEL || 'chromium';
  const browser = await chromium.launch({ headless, channel, args });
  return browser;
}

/** A fresh context with the recorder on. `blocked` lists what it refused. */
export async function newContext(browser, options = {}) {
  const context = await browser.newContext({ viewport: { width: 1200, height: 800 }, ...options });
  const blocked = [];
  await context.route(
    (url) => ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) && OWN_PORTS.has(url.port),
    (route) => {
      blocked.push(route.request().url());
      return route.abort();
    },
  );
  return { context, blocked };
}

/** A page that keeps what happened to it: console, page errors, requests and responses. */
export async function newPage(context) {
  const page = await context.newPage();
  const seen = { console: [], errors: [], requests: [], responses: [], failures: [] };
  page.on('console', (m) => seen.console.push({ type: m.type(), text: m.text(), location: m.location().url }));
  page.on('pageerror', (e) => seen.errors.push(e.message));
  page.on('request', (r) => seen.requests.push({ url: r.url(), method: r.method(), frame: r.frame()?.url() ?? null }));
  page.on('response', (r) => seen.responses.push({ url: r.url(), status: r.status(), headers: r.headers() }));
  page.on('requestfailed', (r) => seen.failures.push({ url: r.url(), reason: r.failure()?.errorText }));
  return { page, seen };
}

/** A folder holding the fixture shell (an app page that speaks the port protocol), copied to somewhere it can be served from. */
function shellFolder(root, name) {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  fs.cpSync(path.join(FIXTURES, 'shell'), dir, { recursive: true });
  return dir;
}

/**
 * The world of a test: the hosts server with the Agent, the flow editor, an evil page on the same site and an app on another
 * domain (the shells of fixtures/shell, each with the headers of its host as web/hosting/headers/ writes them), and the providers
 * origin from `assembleProviders`, given the origins that are now known.
 *
 * @param {{ assembleProviders?: (origins: Record<string,string>, dir: string) => Promise<string> | string, apps?: string[] }} [options]
 *   `apps` names the sites allowed to embed the providers origin (default agent and flows); the callback returns its folder.
 */
export async function startWorld(options = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oaiy-web-e2e-'));
  const hosts = await startHosts();
  const origins = { agent: hosts.origin('agent'), flows: hosts.origin('flows'), providers: hosts.origin('providers'), evil: hosts.origin('evil'), foreign: hosts.origin('foreign') };
  // The app on another domain has the Agent's headers: it is an Agent that is not on the providers origin's site.
  const withHeaders = (name) => ({ root: shellFolder(dir, name), headers: renderHeaders(readTemplate(name === 'foreign' ? 'agent' : name), origins) });
  hosts.setSite('agent', withHeaders('agent'));
  hosts.setSite('flows', withHeaders('flows'));
  hosts.setSite('foreign', withHeaders('foreign'));
  hosts.setSite('evil', { root: shellFolder(dir, 'evil'), headers: '' });
  if (options.assembleProviders) {
    const providersRoot = await options.assembleProviders(origins, path.join(dir, 'providers'));
    hosts.setSite('providers', { root: providersRoot });
  }
  return {
    hosts,
    origins,
    dir,
    async close() {
      await hosts.close();
      fs.rmSync(dir, { recursive: true, force: true });
    },
  };
}

/** The version of the browser, for the report. */
export async function browserVersion(browser) {
  return `${browser.browserType().name()} ${browser.version()}`;
}
