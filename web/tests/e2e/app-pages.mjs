/**
 * The two apps, real builds on their own hosts, and pages of them as a visitor's browser (or OAIY's window) would open them.
 * Shared by the cases that ask what the apps do where they are (E5: what they send; capabilities: what they show).
 *
 * `startAppWorld` builds the Agent and the flow editor (tests/e2e/apps.mjs), serves them on `agent.web.localhost` and
 * `flows.web.localhost` with the headers those hosts send, and starts a browser. `openApp` opens one of them in a browser context of
 * its own with the recorder on (tests/e2e/local-ranges.mjs), so every request to this computer or its network is seen and refused.
 */
import { readTemplate, renderHeaders } from '../../scripts/headers.mjs';
import { buildApps } from './apps.mjs';
import { browserVersion, launchBrowser, sleep } from './harness.mjs';
import { startHosts } from './hosts.mjs';
import { watchLocal } from './local-ranges.mjs';

export const DESKTOP = 'http://127.0.0.1:17972';
export const OAIY = 'http://127.0.0.1:8080';

/** What OAIY's window gives its pages before they run. */
export const OAIY_WINDOW = { origin: DESKTOP, token: 'window-token', theme: 'dark' };

/** How long a page is given to send whatever it is going to send as it loads (its probes fire in the first moments). */
export const SETTLE_MS = 3000;

export async function startAppWorld() {
  const apps = await buildApps();
  // The two apps on their own hosts, and each again at the name OAIY's own window is served from: a browser resolves every
  // `*.localhost` name to this computer, so a page at `oaiy.localhost:PORT` is whatever answers on that port. It is not OAIY's window
  // (which has no port and is given the desktop): the cases that show the apps do not take it for one open these two.
  const hosts = await startHosts({ hostNames: { oaiyAsAgent: 'oaiy.localhost', oaiyflowsAsFlows: 'oaiyflows.localhost' } });
  const names = ['agent', 'flows', 'providers', 'oaiyAsAgent', 'oaiyflowsAsFlows'];
  const origins = Object.fromEntries(names.map((name) => [name, hosts.origin(name)]));
  const headers = { agent: renderHeaders(readTemplate('agent'), { ...origins, apps: [] }), flows: renderHeaders(readTemplate('flows'), { ...origins, apps: [] }) };
  hosts.setSite('agent', { root: apps.agent, headers: headers.agent });
  hosts.setSite('flows', { root: apps.flows, headers: headers.flows });
  hosts.setSite('oaiyAsAgent', { root: apps.agent, headers: headers.agent });
  hosts.setSite('oaiyflowsAsFlows', { root: apps.flows, headers: headers.flows });
  const sites = names.map((name) => `${hosts.host(name)}:${hosts.port}`);
  const browser = await launchBrowser({});
  console.log(`# browser: ${await browserVersion(browser)}`);
  return {
    world: { hosts, origins },
    sites,
    browser,
    async close() {
      await browser.close();
      await hosts.close();
      apps.remove();
    },
  };
}

/**
 * A page of one of the apps, in a browser context of its own with the recorder on.
 * `desktop`: what OAIY's window gives its pages before they run. `storage`: localStorage set before the page runs.
 * `refuseAfterMs` and `answer`: see watchLocal. `at`: the host to open it at (`agent`, `flows`, or `oaiyAsAgent` / `oaiyflowsAsFlows`, the
 * app at `oaiy.localhost:PORT` / `oaiyflows.localhost:PORT`: OAIY's names, on a port).
 */
export async function openApp(env, app, { desktop = null, storage = {}, path = app === 'agent' ? '/' : '/app.html', refuseAfterMs = 0, answer = null, viewport = { width: 1440, height: 900 }, at = app } = {}) {
  const context = await env.browser.newContext({ viewport });
  const { attempts } = await watchLocal(context, { sites: env.sites, refuseAfterMs, answer });
  const errors = [];
  await context.addInitScript(
    ({ desktop, storage }) => {
      // A visitor's browser is not automated: the Agent looks for OAIY on its own only in a browser that is not.
      Object.defineProperty(Navigator.prototype, 'webdriver', { get: () => false, configurable: true });
      if (desktop) window.__OAIY_DESKTOP__ = Object.freeze(desktop);
      try {
        // No splash and no first-run wizard: they cover the pages a scenario presses buttons in.
        localStorage.setItem('skipSplash', 'true');
        localStorage.setItem('oaiy.wizard.completed', 'true');
        localStorage.setItem('oaiy_theme', 'dark');
        for (const [key, value] of Object.entries(storage)) localStorage.setItem(key, value);
      } catch {
        /* blocked storage: the page still loads */
      }
    },
    { desktop, storage },
  );
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${env.world.origins[at]}${path}`);
  return { context, page, attempts, errors };
}

/** The app is up: the Agent shows its project tree, the flow editor its workspace. Then it is given `settleMs` to send what it will. */
export async function appReady(app, page, settleMs = SETTLE_MS) {
  if (app === 'agent') await page.waitForSelector('.tree-row', { timeout: 60_000 });
  else await page.waitForSelector('#oaiy-main', { timeout: 60_000 });
  await sleep(settleMs);
}

export const healthOf = (attempts) => attempts.filter((a) => a.endsWith('/api/health'));

/** What OAIY Desktop answers to the routes a tab reads. It exists only in the browser: the request is answered there and goes no further. */
export const fakeDesktop = ({ url }) => {
  const route = new URL(url).pathname;
  if (route === '/api/health') return { body: { status: 'ok', product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: '9.9.9' } };
  if (route === '/api/services') return { body: { services: [{ id: 'py-rig', name: 'Python rig', description: 'A rig', category: 'llm', status: 'running', port: 8123, defaultPort: 8123, installed: true }] } };
  if (route === '/api/ai/engine/services') return { body: { services: [] } };
  return null;
};
