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
import { chromium, firefox } from 'playwright';
import { assembleProviders } from '../../scripts/assemble.mjs';
import { readTemplate, renderHeaders } from '../../scripts/headers.mjs';
import { startHosts } from './hosts.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const FIXTURES = path.join(HERE, 'fixtures');

/** The features Playwright 1.60 disables in the Chromium it launches (chromiumSwitches.ts), except ThirdPartyStoragePartitioning. */
const PLAYWRIGHT_DISABLED_FEATURES = [
  'AvoidUnnecessaryBeforeUnloadCheckSync',
  'BoundaryEventDispatchTracksNodeRemoval',
  'DestroyProfileOnBrowserClose',
  'DialMediaRouteProvider',
  'GlobalMediaControls',
  'HttpsUpgrades',
  'LensOverlay',
  'MediaRouter',
  'PaintHolding',
  'Translate',
  'AutoDeElevate',
  'RenderDocument',
  'OptimizationHints',
  'msForceBrowserSignIn',
  'msEdgeUpdateLaunchServicesPreferredVersion',
];

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
export async function launchBrowser({ isolateOrigins = [], headless = true, browserName = 'chromium' } = {}) {
  if (browserName === 'firefox') return launchFirefox({ headless });
  const args = [];
  if (isolateOrigins.length > 0) args.push(`--isolate-origins=${isolateOrigins.join(',')}`);
  // Playwright turns third-party storage partitioning OFF in the Chromium it launches (its issue 32230), whichever channel, and every
  // real Chrome and Edge has it on. With it off, a frame under an app on another domain shares the top-level window's storage, which
  // no visitor's browser does, and E1's "another registrable domain must NOT share the store" cannot be shown. This is Playwright's
  // own list of disabled features without that one; a later `--disable-features` replaces the earlier one.
  args.push(`--disable-features=${PLAYWRIGHT_DISABLED_FEATURES.join(',')}`);
  const channel = process.env.WEB_E2E_CHANNEL || 'chromium';
  const browser = await chromium.launch({ headless, channel, args });
  return browser;
}

/**
 * Firefox, for E1 (design 8 names Chromium and Firefox). Playwright's own Firefox build is what it wants; where the installed
 * one is another revision (Playwright 1.60 asks for 1522 and %LOCALAPPDATA%\ms-playwright may hold 1543) the newest installed one is
 * used. Nothing is downloaded. Returns null when no Firefox can be started, so a test can say it was not run.
 */
async function launchFirefox({ headless }) {
  // Playwright's Firefox starts with cookie behaviour 0 (no partitioning); a real Firefox has 5 (reject trackers and partition
  // third-party state) by default, which is what puts a frame under another site's storage in another partition.
  const firefoxUserPrefs = { 'network.cookie.cookieBehavior': 5 };
  try {
    return await firefox.launch({ headless, firefoxUserPrefs });
  } catch {
    // fall through to an installed revision
  }
  const root = path.join(process.env.LOCALAPPDATA ?? path.join(os.homedir(), '.cache'), 'ms-playwright');
  const found = fs.existsSync(root)
    ? fs.readdirSync(root).filter((d) => /^firefox-\d+$/.test(d)).sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))
    : [];
  for (const dir of found) {
    for (const exe of ['firefox/firefox.exe', 'firefox/firefox', 'firefox/Nightly.app/Contents/MacOS/firefox']) {
      const executablePath = path.join(root, dir, exe);
      if (!fs.existsSync(executablePath)) continue;
      try {
        return await firefox.launch({ headless, executablePath, firefoxUserPrefs });
      } catch {
        // try the next
      }
    }
  }
  return null;
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
 * origin assembled from its build (`npm run build`) with the origins that are now known.
 *
 * @param {{ providers?: boolean, apps?: string[], providersHeaders?: (rendered: string, origins: Record<string,string>) => string }} [options]
 *   `providers: false` leaves the providers site out (the harness tests); `apps` names the sites allowed to embed the providers
 *   origin (default agent and flows); `providersHeaders` changes its rendered `_headers` (a test that needs a variant).
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
  if (options.providers !== false) {
    const apps = Object.fromEntries((options.apps ?? ['agent', 'flows']).map((name) => [name, origins[name]]));
    const root = await assembleProviders({ outDir: path.join(dir, 'providers'), providers: origins.providers, apps, headers: options.providersHeaders ? (rendered) => options.providersHeaders(rendered, origins) : undefined });
    hosts.setSite('providers', { root });
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
