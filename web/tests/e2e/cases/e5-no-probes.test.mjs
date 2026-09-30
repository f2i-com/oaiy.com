/**
 * E5, no probes (design 8, 3.5 rule R1, 6 threat 8): a tab in a browser makes NO request to loopback or LAN ranges when an app loads;
 * pressing the button that looks (Connect) makes exactly the one request it says it makes. OAIY's own windows look as they always did.
 *
 * Against the REAL builds of the Agent and the flow editor, served on their own hosts with the headers those hosts send (tests/e2e/apps.mjs).
 * Every request the browser makes to this computer or its network, by a page, a worker or a service worker, is recorded and refused
 * (tests/e2e/local-ranges.mjs), so a probe is SEEN, and cannot reach the owner's desktop.
 *
 * Two things keep a "zero" from being an empty claim:
 *   - the Agent looks for OAIY on its own unless the browser is automated (`navigator.webdriver`, which every Playwright page is), so a
 *     zero under automation would hold for the code before this change too: each scenario here shows the page as a visitor's browser
 *     (webdriver false), the way the recorder's own test shows it sees a probe;
 *   - the same pages, given a desktop the way OAIY's window gives it, DO look, and the recorder sees every look.
 * The mutation checks in the change that made these pass (break the gate, see them fail, put it back) rest on both.
 *
 * What the Agent's "Connect" is: pressing the phone chip (the pairing dialog looks for OAIY Desktop: one `GET /api/health`) and
 * Settings -> Images, video and audio -> Find OAIY (one `GET /v1/discovery`; the fallback `/.well-known/oaiy.json` is asked only
 * after a 404, and a refused request is not one). The flow editor's is the Connect button of Settings -> Services and of the sidebar.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { readTemplate, renderHeaders } from '../../../scripts/headers.mjs';
import { buildApps } from '../apps.mjs';
import { browserVersion, launchBrowser, sleep, startWorld } from '../harness.mjs';
import { watchLocal } from '../local-ranges.mjs';

const DESKTOP = 'http://127.0.0.1:17972';
const OAIY = 'http://127.0.0.1:8080';
/** How long a page is given to send whatever it is going to send as it loads (its probes fire in the first moments). */
const SETTLE_MS = 3000;

let apps;
let world;
let browser;
let sites;

before(async () => {
  apps = await buildApps();
  world = await startWorld({ providers: false });
  const origins = { ...world.origins, apps: [] };
  world.hosts.setSite('agent', { root: apps.agent, headers: renderHeaders(readTemplate('agent'), origins) });
  world.hosts.setSite('flows', { root: apps.flows, headers: renderHeaders(readTemplate('flows'), origins) });
  sites = Object.keys(world.origins).map((name) => `${world.hosts.host(name)}:${world.hosts.port}`);
  browser = await launchBrowser({});
  console.log(`# browser: ${await browserVersion(browser)}`);
});

after(async () => {
  await browser?.close();
  await world?.close();
  apps?.remove();
});

/**
 * A page of one of the apps, in a browser context of its own with the recorder on.
 * `desktop`: what OAIY's window gives its pages before they run. `storage`: localStorage set before the page runs.
 */
async function open(app, { desktop = null, storage = {}, path = app === 'agent' ? '/' : '/app.html', refuseAfterMs = 0, answer = null } = {}) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const { attempts } = await watchLocal(context, { sites, refuseAfterMs, answer });
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
  await page.goto(`${world.origins[app]}${path}`);
  return { context, page, attempts, errors };
}

/** The app is up: the Agent shows its project tree, the flow editor its workspace. */
async function ready(app, page) {
  if (app === 'agent') await page.waitForSelector('.tree-row', { timeout: 60_000 });
  else await page.waitForSelector('#oaiy-main', { timeout: 60_000 });
  await sleep(SETTLE_MS);
}

const health = (attempts) => attempts.filter((a) => a.endsWith('/api/health'));

/** What OAIY Desktop answers to the routes a tab reads. It exists only in the browser: the request is answered there and goes no further. */
const fakeDesktop = ({ url }) => {
  const route = new URL(url).pathname;
  if (route === '/api/health') return { body: { status: 'ok', product: 'oaiy-desktop', protocol: 'oaiy-bridge/1', version: '9.9.9' } };
  if (route === '/api/services') return { body: { services: [{ id: 'py-rig', name: 'Python rig', description: 'A rig', category: 'llm', status: 'running', port: 8123, defaultPort: 8123, installed: true }] } };
  if (route === '/api/ai/engine/services') return { body: { services: [] } };
  return null;
};

describe('E5: the recorder sees what a page sends', () => {
  it('a request to loopback, to each private range, to link-local, to Tailscale and to a .local name is seen and refused, from a page and from a worker', async () => {
    const { context, page, attempts } = await open('flows');
    await ready('flows', page);
    assert.deepEqual(attempts, [], 'nothing before the test asks');
    const targets = [`${DESKTOP}/api/health`, 'http://192.168.1.5/x', 'http://[::1]:8080/', 'http://10.1.2.3/', 'http://172.20.0.9/', 'http://169.254.169.254/latest', 'http://100.101.102.103/', 'http://printer.local/', 'http://localhost:9/'];
    const outcomes = await page.evaluate(async (urls) => {
      const results = {};
      for (const url of urls) {
        try {
          await fetch(url, { mode: 'no-cors' });
          results[url] = 'sent';
        } catch {
          results[url] = 'refused';
        }
      }
      // And from a worker, which a page's own request hooks would not see.
      const source = `for (const u of ${JSON.stringify(['http://192.168.7.7/from-worker'])}) fetch(u, { mode: 'no-cors' }).catch(() => null); postMessage('done');`;
      const worker = new Worker(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })));
      await new Promise((resolve) => (worker.onmessage = resolve));
      return results;
    }, targets);
    await sleep(300);
    for (const url of targets) assert.equal(outcomes[url], 'refused', url);
    for (const url of [...targets, 'http://192.168.7.7/from-worker']) assert.ok(attempts.includes(`GET ${url}`), `${url} was seen: ${attempts.join(', ')}`);
    await context.close();
  });

  it('the sites under test are not probes, and an address that is not local is not one either', async () => {
    const { context, page, attempts } = await open('flows');
    await ready('flows', page);
    await page.evaluate(async (urls) => {
      for (const url of urls) await fetch(url, { mode: 'no-cors' }).catch(() => null);
    }, [`${world.origins.agent}/x`, `${world.origins.providers}/y`]);
    assert.deepEqual(attempts, []);
    await context.close();
  });
});

describe('E5: the Agent', () => {
  it('a fresh load in a visitor\'s browser sends nothing to this computer or its network', async () => {
    const { context, page, attempts, errors } = await open('agent');
    await ready('agent', page);
    assert.match(await page.locator('.chat-log').textContent(), /Welcome!/, 'it is the Agent, up and showing its welcome');
    assert.deepEqual(attempts, []);
    assert.deepEqual(errors, []);
    await context.close();
  });

  it('a link that carries an address for OAIY (?oaiy=) makes a visitor\'s tab look at nothing', async () => {
    for (const address of [OAIY, 'http://192.168.1.1', 'http://10.0.0.5:8080']) {
      const { context, page, attempts } = await open('agent', { path: `/?oaiy=${encodeURIComponent(address)}` });
      await ready('agent', page);
      assert.deepEqual(attempts, [], address);
      await context.close();
    }
  });

  it('a tab with an OAIY it found before (a link it saved) looks at that one, and only that', async () => {
    // The link is what the settings keep after Find OAIY: media.discovered. Saved through the page's own store, then reloaded.
    const again = await open('agent');
    await again.page.waitForSelector('.tree-row', { timeout: 60_000 });
    await again.page.evaluate(async () => {
      const open = indexedDB.open('bot.computer');
      const db = await new Promise((resolve, reject) => {
        open.onsuccess = () => resolve(open.result);
        open.onerror = () => reject(open.error);
      });
      const tx = db.transaction('kv', 'readwrite');
      const store = tx.objectStore('kv');
      const media = await new Promise((resolve) => {
        const get = store.get('media');
        get.onsuccess = () => resolve(get.result ?? { baseUrl: '', apiKey: '', enabled: true, imageModels: [], videoModels: [] });
        get.onerror = () => resolve({ baseUrl: '', apiKey: '', enabled: true, imageModels: [], videoModels: [] });
      });
      store.put({ ...media, baseUrl: 'http://192.168.1.20:8080/v1', discovered: { service: 'oaiy-studio', version: '0.1.0', origin: 'http://192.168.1.20:8080', at: Date.now() } }, 'media');
      await new Promise((resolve) => (tx.oncomplete = resolve));
      db.close();
    });
    again.attempts.length = 0;
    await again.page.reload();
    await ready('agent', again.page);
    assert.deepEqual(again.attempts, ['GET http://192.168.1.20:8080/v1/discovery'], 'the link the person saved, once, and nothing at the usual address');
    await again.context.close();
  });

  it('given a desktop (as OAIY\'s window gives it) the Agent looks for OAIY as it opens, as it always has, and the recorder sees it', async () => {
    const { context, page, attempts } = await open('agent', { desktop: { origin: DESKTOP, token: 'window-token', theme: 'dark' } });
    await ready('agent', page);
    assert.ok(attempts.includes(`GET ${OAIY}/v1/discovery`), `it looked for OAIY: ${attempts.join(', ')}`);
    assert.ok(attempts.some((a) => a.startsWith(`GET ${DESKTOP}/api/`)), `and at the desktop it was given: ${attempts.join(', ')}`);
    await context.close();
  });

  it('pressing the phone chip (the Agent\'s Connect) makes exactly one health request, and says it is looking', async () => {
    // The request is held for a moment before it is refused, so what the dialog says while it is out can be read.
    const { context, page, attempts } = await open('agent', { refuseAfterMs: 800 });
    await ready('agent', page);
    assert.deepEqual(attempts, []);
    await page.locator('button.chip.phone').click();
    await page.waitForSelector('dialog[open]');
    assert.match(await page.locator('dialog[open] p.muted').first().textContent(), /Looking for OAIY Desktop at http:\/\/127\.0\.0\.1:17972.*may ask whether it may connect to your network/);
    await page.waitForFunction(() => /is not running at/.test(document.querySelector('dialog[open]')?.textContent ?? ''), null, { timeout: 15_000 });
    await sleep(500);
    assert.deepEqual(attempts, [`GET ${DESKTOP}/api/health`]);
    await context.close();
  });

  it('Find OAIY (Settings, Images, video and audio) makes exactly one request, says so, and the page asked nothing before it', async () => {
    const { context, page, attempts } = await open('agent');
    await ready('agent', page);
    assert.deepEqual(attempts, []);
    await page.locator('button.settings-button').click();
    await page.waitForSelector('dialog.settings[open]');
    const section = page.locator('.media-settings');
    assert.match(await section.textContent(), /OAIY is looked for only when you press Find OAIY/);
    assert.match((await section.locator('button', { hasText: 'Find OAIY' }).getAttribute('title')) ?? '', /reaches out to your computer only when you press this/);
    await section.locator('button', { hasText: 'Find OAIY' }).click();
    await page.waitForFunction(() => /Nothing answered at/.test(document.querySelector('.media-settings .form-note')?.textContent ?? ''), null, { timeout: 15_000 });
    await sleep(500);
    assert.deepEqual(attempts, [`GET ${OAIY}/v1/discovery`]);
    await context.close();
  });
});

describe('E5: the flow editor', () => {
  it('a fresh load in a visitor\'s browser sends nothing to this computer or its network, and clears the services a desktop left in storage', async () => {
    const stale = JSON.stringify([{ id: 'companion:py-rig', name: 'Python rig (running)', endpoint: 'http://127.0.0.1:8123/v1/chat/completions', nodeTypes: ['ai_llm'], group: 'desktop', inUse: true }]);
    const { context, page, attempts, errors } = await open('flows', { storage: { 'oaiy.desktopServices': stale } });
    await ready('flows', page);
    assert.deepEqual(attempts, []);
    assert.equal(await page.evaluate(() => localStorage.getItem('oaiy.desktopServices')), null, 'the desktop\'s services are not this tab\'s');
    assert.deepEqual(errors, []);
    // Nothing keeps asking either: the poll runs every ten seconds where there is one.
    await sleep(11_000);
    assert.deepEqual(attempts, []);
    await context.close();
  });

  it('shows that the desktop has not been looked for, and Connect says what it asks and why the browser may put a question', async () => {
    const { context, page, attempts } = await open('flows');
    await ready('flows', page);
    await page.locator('button[aria-label="Settings"]').click();
    const card = page.locator('main .oaiy-connect');
    await card.waitFor();
    assert.match(await page.locator('main').textContent(), /not connected/);
    assert.match(await card.textContent(), /Connect asks OAIY Desktop, on this computer or at the address above, whether it is running: one request\. Until you press it this page sends nothing to your computer or your network\. Your browser may ask whether this site may connect to devices on your network/);
    assert.deepEqual(attempts, []);
    await context.close();
  });

  it('Connect in Settings makes exactly one health request, says nothing answered, and nothing keeps asking', async () => {
    const { context, page, attempts } = await open('flows');
    await ready('flows', page);
    await page.locator('button[aria-label="Settings"]').click();
    await page.locator('main [data-connect-desktop]').click();
    await page.waitForFunction(() => /Nothing answered at http:\/\/127\.0\.0\.1:17972/.test(document.querySelector('main .oaiy-connect')?.textContent ?? ''), null, { timeout: 15_000 });
    assert.deepEqual(attempts, [`GET ${DESKTOP}/api/health`]);
    assert.equal(await page.evaluate(() => localStorage.getItem('oaiy.desktopLinked')), null, 'nothing answered, so no link is kept');
    await sleep(11_000);
    assert.deepEqual(attempts, [`GET ${DESKTOP}/api/health`], 'and no poll was started');
    await context.close();
  });

  it('Connect to a desktop that answers: one health request, the link is kept, its services are listed once; Disconnect forgets it and nothing asks again', async () => {
    const { context, page, attempts } = await open('flows', { answer: fakeDesktop });
    await ready('flows', page);
    assert.deepEqual(attempts, []);
    await page.locator('button[aria-label="Settings"]').click();
    await page.locator('main [data-connect-desktop]').click();
    await page.waitForFunction(() => /connected.*9\.9\.9/.test(document.querySelector('main')?.textContent ?? ''), null, { timeout: 15_000 });
    await sleep(1500);
    assert.deepEqual(attempts, [`GET ${DESKTOP}/api/health`, `GET ${DESKTOP}/api/services`], 'one health request, and then the desktop\'s services once: not a second look');
    assert.equal(await page.evaluate(() => localStorage.getItem('oaiy.desktopLinked')), '1', 'the link is kept');
    assert.equal(JSON.parse(await page.evaluate(() => localStorage.getItem('oaiy.desktopServices'))).length, 1, 'and the desktop\'s service is in the palette\'s list');
    assert.match(await page.locator('aside').textContent(), /Linked to OAIY Desktop\./);
    // The link is what keeps the editor looking, so the poll runs: ten seconds on, it asks again.
    await sleep(10_500);
    assert.equal(health(attempts).length, 2, `the poll: ${attempts.join(', ')}`);
    await page.locator('main .oaiy-connect button', { hasText: 'Disconnect' }).click();
    await page.waitForFunction(() => /not connected/.test(document.querySelector('main')?.textContent ?? ''), null, { timeout: 5000 });
    assert.equal(await page.evaluate(() => localStorage.getItem('oaiy.desktopLinked')), null);
    assert.equal(await page.evaluate(() => localStorage.getItem('oaiy.desktopServices')), null, 'the desktop\'s services leave the palette');
    const before = attempts.length;
    await sleep(11_000);
    assert.equal(attempts.length, before, 'and nothing asks again');
    await context.close();
  });

  it('Connect in the sidebar makes exactly one health request too, and its tooltip says why', async () => {
    const { context, page, attempts } = await open('flows');
    await ready('flows', page);
    const button = page.locator('aside [data-connect-desktop]');
    assert.match((await button.getAttribute('title')) ?? '', /one request\. Until you press it this page sends nothing/);
    await button.click();
    await page.waitForFunction(() => /Nothing answered at/.test(document.querySelector('aside .oaiy-engine-get [role="status"]')?.textContent ?? ''), null, { timeout: 15_000 });
    assert.deepEqual(attempts, [`GET ${DESKTOP}/api/health`]);
    await context.close();
  });

  it('given a desktop (as OAIY\'s window gives it) the editor looks as it opens, as it always has: the two probes, and no Connect button', async () => {
    const { context, page, attempts } = await open('flows', { desktop: { origin: DESKTOP, token: 'window-token', theme: 'dark' } });
    await ready('flows', page);
    assert.equal(health(attempts).length, 2, `the detection and the service sync each look once: ${attempts.join(', ')}`);
    assert.ok(health(attempts).every((a) => a === `GET ${DESKTOP}/api/health`));
    assert.equal(await page.locator('[data-connect-desktop]').count(), 0, 'nothing to connect in OAIY\'s own window');
    await context.close();
  });

  it('a tab with a link (kept from Connect) looks as it opens, like the window does; one whose engine has an address of its own looks there', async () => {
    const linked = await open('flows', { storage: { 'oaiy.desktopLinked': '1' } });
    await ready('flows', linked.page);
    assert.equal(health(linked.attempts).length, 2, linked.attempts.join(', '));
    assert.ok(health(linked.attempts).every((a) => a === `GET ${DESKTOP}/api/health`));
    await linked.context.close();

    const addressed = await open('flows', { storage: { 'oaiy.engineBase': 'http://192.168.1.50:17972' } });
    await ready('flows', addressed.page);
    assert.equal(health(addressed.attempts).length, 2, addressed.attempts.join(', '));
    assert.ok(addressed.attempts.every((a) => a.startsWith('GET http://192.168.1.50:17972/')), addressed.attempts.join(', '));
    await addressed.context.close();
  });
});
