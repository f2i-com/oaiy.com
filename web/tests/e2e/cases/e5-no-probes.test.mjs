/**
 * E5, no probes (design 8, 3.5 rule R1, 6 threat 8): a tab in a browser makes NO request to loopback or LAN ranges when an app loads;
 * pressing the button that looks (Connect) makes exactly the one request it says it makes. OAIY's own windows look as they always did.
 *
 * Against the REAL builds of the Agent and the flow editor, served on their own hosts with the headers those hosts send (tests/e2e/app-pages.mjs).
 * Every request the browser makes to this computer or its network, by a page or a worker, is recorded and refused
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
import { sleep } from '../harness.mjs';
import { DESKTOP, OAIY, OAIY_WINDOW, appReady as ready, fakeDesktop, healthOf as health, openApp, readStored, seedOaiyLink, seedPairing, startAppWorld } from '../app-pages.mjs';

let env;

before(async () => {
  env = await startAppWorld();
});

after(async () => {
  await env?.close();
});

const open = (app, options) => openApp(env, app, options);
const world = { get origins() { return env.world.origins; } };

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

  it('a page at oaiy.localhost on a port, given no desktop, is a tab, not OAIY\'s window: it looks at nothing, and a link\'s address (?oaiy=) is ignored (the review\'s F5)', async () => {
    const seen = [];
    for (const path of ['/', `/?oaiy=${encodeURIComponent('http://192.168.1.1:8080')}`, `/?oaiy=${encodeURIComponent(OAIY)}`]) {
      const { context, page, attempts } = await open('agent', { at: 'oaiyAsAgent', path });
      await ready('agent', page);
      assert.match(await page.locator('.chat-log').textContent(), /Welcome!/);
      assert.equal(await page.evaluate(() => location.hostname), 'oaiy.localhost');
      assert.equal(await page.locator('button.chip.phone').evaluate((e) => e.hidden), false, 'the pairing chip: it is a tab');
      seen.push(...attempts);
      await context.close();
    }
    assert.deepEqual(seen, []);
  });

  it('OAIY\'s own window ignores an address in the page\'s own address too: only an automated browser is pointed elsewhere by one', async () => {
    const { context, page, attempts } = await open('agent', { desktop: OAIY_WINDOW, path: `/?oaiy=${encodeURIComponent('http://192.168.1.1:8080')}` });
    await ready('agent', page);
    assert.ok(attempts.includes(`GET ${OAIY}/v1/discovery`), `it looked at OAIY's usual address: ${attempts.join(', ')}`);
    assert.ok(attempts.every((a) => !a.includes('192.168.1.1')), `and never at the address the link named: ${attempts.join(', ')}`);
    await context.close();
  });

  it('an automated browser is pointed by ?oaiy= at the address the link names, and the key the person typed does not go there (the review\'s F5)', async () => {
    const STANDIN = 'http://192.168.1.99:8080';
    const s = await open('agent', { automated: true });
    await ready('agent', s.page);
    await seedOaiyLink(s.page, { origin: 'http://192.168.1.20:8080', key: 'sk-oaiy-review-key-0123456789' });
    s.attempts.length = 0;
    s.details.length = 0;
    await s.page.goto(`${env.world.origins.agent}/?oaiy=${encodeURIComponent(STANDIN)}`);
    await ready('agent', s.page);
    assert.deepEqual(s.attempts, [`GET ${STANDIN}/v1/discovery`], 'it looked where the link said, and at the saved OAIY not at all (an automated browser looks only where it is told)');
    assert.equal(s.details[0].headers.authorization, undefined, 'and sent no key to it');
    await s.context.close();
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
    const { context, page, attempts } = await open('agent', { desktop: OAIY_WINDOW });
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
    assert.match(await page.locator('dialog[open] p.muted').first().textContent(), /Looking for OAIY Desktop at http:\/\/127\.0\.0\.1:17972 now, because you opened this dialog.*may ask whether it may connect to your network/);
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
    assert.match(await section.textContent(), /OAIY is looked for only when you press Find OAIY: until you do, this page sends nothing to your computer or your network/);
    assert.match((await section.locator('button', { hasText: 'Find OAIY' }).getAttribute('title')) ?? '', /reaches out to your computer only when you press this/);
    await section.locator('button', { hasText: 'Find OAIY' }).click();
    await page.waitForFunction(() => /Nothing answered at/.test(document.querySelector('.media-settings .form-note')?.textContent ?? ''), null, { timeout: 15_000 });
    await sleep(500);
    assert.deepEqual(attempts, [`GET ${OAIY}/v1/discovery`]);
    await context.close();
  });
});

describe('E5: the Agent\'s saved links, and what its words say of each state', () => {
  const LAN = 'http://192.168.1.20:8080';
  const KEY = 'sk-oaiy-review-key-0123456789';

  /** The Agent as a tab that has found an OAIY at LAN (with a key), reloaded so the saved link is what it starts with. */
  async function withSavedOaiy(options = {}) {
    const s = await open('agent', options);
    await ready('agent', s.page);
    await seedOaiyLink(s.page, { origin: LAN, key: KEY });
    s.attempts.length = 0;
    s.details.length = 0;
    await s.page.reload();
    await ready('agent', s.page);
    return s;
  }
  const openSettings = async (page) => {
    await page.locator('button.settings-button').click();
    await page.waitForSelector('dialog.settings[open]');
    return page.locator('.media-settings');
  };

  it('a tab that saved an OAIY asks it there as it opens, with the key, and Settings says that: not "never reaches out", not "only when you press"', async () => {
    const { context, page, attempts, details } = await withSavedOaiy();
    assert.deepEqual(attempts, [`GET ${LAN}/v1/discovery`], 'it asked, before any button was pressed');
    assert.equal(details[0].headers.authorization, `Bearer ${KEY}`, 'and sent the key');
    const section = await openSettings(page);
    const words = await section.locator('p.muted').first().textContent();
    assert.match(words, /OAIY was found at http:\/\/192\.168\.1\.20:8080: this page asks it there each time it opens \(sending the key below\), and now and then to follow the model chosen in OAIY's Engines\. Forget OAIY, or clearing the address, ends that/);
    assert.doesNotMatch(words, /sends nothing|only when you press|never reaches/);
    assert.match((await section.locator('button', { hasText: 'Find OAIY' }).getAttribute('title')) ?? '', /also asks OAIY at http:\/\/192\.168\.1\.20:8080 each time it opens/);
    assert.equal(await section.locator('button', { hasText: 'Forget OAIY' }).count(), 1);
    await context.close();
  });

  it('a tab that has found none says it sends nothing until Find OAIY is pressed; and OAIY\'s own window says it is found on its own', async () => {
    const tab = await open('agent');
    await ready('agent', tab.page);
    const words = await (await openSettings(tab.page)).locator('p.muted').first().textContent();
    assert.match(words, /OAIY is looked for only when you press Find OAIY: until you do, this page sends nothing to your computer or your network/);
    assert.equal(await tab.page.locator('.media-settings button', { hasText: 'Forget OAIY' }).count(), 0, 'nothing found, nothing to forget');
    assert.deepEqual(tab.attempts, []);
    await tab.context.close();

    const own = await open('agent', { desktop: OAIY_WINDOW });
    await ready('agent', own.page);
    assert.match(await (await openSettings(own.page)).locator('p.muted').first().textContent(), /with a media service: OAIY is found on its own, and any server/);
    await own.context.close();
  });

  for (const how of ['clearing the address', 'Forget OAIY']) {
    it(`the Agent forgets the OAIY it found, by ${how} and Save: it stops asking it when it opens, and the key goes nowhere (the review's F1)`, async () => {
      const { context, page, attempts, details } = await withSavedOaiy();
      assert.deepEqual(attempts, [`GET ${LAN}/v1/discovery`], 'the control: before, it asks, with the key');
      assert.equal(details[0].headers.authorization, `Bearer ${KEY}`);
      const section = await openSettings(page);
      if (how === 'Forget OAIY') {
        await section.locator('button', { hasText: 'Forget OAIY' }).click();
        assert.match(await section.locator('.form-note').textContent(), /^Forgotten \(http:\/\/192\.168\.1\.20:8080\)\. Press Save to keep that/);
      } else {
        await section.locator('input').first().fill('');
        await section.locator('input').first().dispatchEvent('input');
      }
      await page.locator('dialog.settings button.primary').click();
      await page.waitForSelector('dialog.settings', { state: 'detached' });
      await sleep(500);
      const stored = await readStored(page);
      assert.equal(stored.discovered, null, 'what says it was found is gone');
      assert.equal(stored.endpoints, false);
      if (how === 'Forget OAIY') assert.deepEqual([stored.baseUrl, stored.keySealed], ['', false], 'and its address and key with it');
      attempts.length = 0;
      details.length = 0;
      await page.reload();
      await ready('agent', page);
      await sleep(3500);
      assert.deepEqual(attempts, [], 'a reload asks nothing at the old address, or anywhere');
      assert.deepEqual(details, []);
      // and the words are the words of a tab that has found none
      assert.match(await (await openSettings(page)).locator('p.muted').first().textContent(), /only when you press Find OAIY: until you do, this page sends nothing/);
      await context.close();
    });
  }

  it('a paired tab keeps in touch with its desktop, and the dialog says so; an unpaired one says it asks only because the dialog is open', async () => {
    const DESK = 'http://127.0.0.1:17972';
    const paired = await open('agent', { refuseAfterMs: 1500 });
    await ready('agent', paired.page);
    await seedPairing(paired.page, { origin: DESK, token: 'paired-token-0123456789' });
    paired.attempts.length = 0;
    await paired.page.reload();
    await ready('agent', paired.page);
    assert.ok(paired.attempts.some((a) => a.startsWith(`GET ${DESK}/api/`)), `it asks its desktop in the background, before any dialog: ${paired.attempts.join(', ')}`);
    await paired.page.locator('button.chip.phone').click();
    await paired.page.waitForSelector('dialog[open]');
    const words = await paired.page.locator('dialog[open] p.muted').first().textContent();
    assert.match(words, /^This page is paired with OAIY Desktop at http:\/\/127\.0\.0\.1:17972 and keeps in touch with it while it is open \(its calls, texts, settings and flows\)\. Checking it now\.$/);
    assert.doesNotMatch(words, /only from here|asks nothing|because you opened this dialog/);
    await paired.context.close();
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

  it('a page at oaiyflows.localhost on a port, given no desktop, is a tab too: it looks at nothing and offers Connect', async () => {
    const { context, page, attempts } = await open('flows', { at: 'oaiyflowsAsFlows' });
    await ready('flows', page);
    assert.equal(await page.evaluate(() => location.hostname), 'oaiyflows.localhost');
    assert.deepEqual(attempts, []);
    assert.equal(await page.locator('aside [data-connect-desktop]').count(), 1, 'Connect: it is a tab');
    await context.close();
  });

  it('shows that the desktop has not been looked for, and Connect says what it asks and why the browser may put a question', async () => {
    const { context, page, attempts } = await open('flows');
    await ready('flows', page);
    await page.locator('button[aria-label="Settings"]').click();
    const card = page.locator('main .oaiy-connect');
    await card.waitFor();
    assert.match(await page.locator('main').textContent(), /not connected/);
    assert.match(await card.textContent(), /Connect sends one request to http:\/\/127\.0\.0\.1:17972 \(GET \/api\/health\) to ask whether OAIY Desktop is running\. Until you press it this page sends nothing to your computer or your network\. Your browser may ask whether this site may connect to devices on your network/);
    assert.deepEqual(attempts, []);
    await context.close();
  });

  it('the card says what is true of each state (the review\'s F2 and F12): linked by Connect asks as it opens and every ten seconds, an address of its own does too, and after Disconnect it is never connected again', async () => {
    const card = (page) => page.locator('main [data-connect-words]');
    const openCard = async (page) => {
      await page.locator('button[aria-label="Settings"]').click();
      await card(page).waitFor();
      return card(page);
    };

    // A link kept by Connect, the desktop switched off: it asks anyway (as it opens, then every ten seconds), and says so.
    const linked = await open('flows', { storage: { 'oaiy.desktopLinked': '1' } });
    await ready('flows', linked.page);
    const c = await openCard(linked.page);
    assert.equal(await c.getAttribute('data-connect-words'), 'connected');
    const text = await c.textContent();
    assert.match(text, /^This browser has a link to OAIY Desktop at http:\/\/127\.0\.0\.1:17972, so the editor asks there each time it opens and every 10 seconds while it is open; nothing answers now\. Connect asks again now; Disconnect forgets the link and ends that\.$/);
    assert.doesNotMatch(text, /sends nothing|Until you press it/);
    assert.equal(await linked.page.locator('main .oaiy-connect button').count(), 2, 'Connect (again) and Disconnect: the link can be forgotten while the desktop is off');
    assert.equal(health(linked.attempts).length, 2, `it asked as it opened, without a button: ${linked.attempts.join(', ')}`);
    await sleep(10_500);
    assert.equal(health(linked.attempts).length, 3, 'and again after ten seconds: what the words say');
    // Disconnect: the words are the words of a tab that never connected, and it asks nothing more.
    await linked.page.locator('main .oaiy-connect button', { hasText: 'Disconnect' }).click();
    await linked.page.waitForFunction(() => document.querySelector('main [data-connect-words]')?.getAttribute('data-connect-words') === 'never', null, { timeout: 5000 });
    assert.match(await c.textContent(), /^Connect sends one request to http:\/\/127\.0\.0\.1:17972 \(GET \/api\/health\)/);
    const before = linked.attempts.length;
    await sleep(11_000);
    assert.equal(linked.attempts.length, before);
    await linked.context.close();

    // An address of its own in Settings, nothing answering there.
    const addressed = await open('flows', { storage: { 'oaiy.engineBase': 'http://192.168.1.50:17972' } });
    await ready('flows', addressed.page);
    const a = await openCard(addressed.page);
    assert.equal(await a.getAttribute('data-connect-words'), 'address');
    assert.match(await a.textContent(), /^The engine's address in Settings is http:\/\/192\.168\.1\.50:17972, so the editor asks there each time it opens and every 10 seconds while it is open; nothing answers now\. Reset above gives the default address back/);
    assert.equal(health(addressed.attempts).length, 2, 'and it did ask as it opened');
    await addressed.context.close();

    // Connected and answering: the words say what Connect started, and the requests are those.
    const up = await open('flows', { storage: { 'oaiy.desktopLinked': '1' }, answer: fakeDesktop });
    await ready('flows', up.page);
    const u = await openCard(up.page);
    assert.equal(await u.getAttribute('data-connect-words'), 'connected');
    assert.match(await u.textContent(), /^Connected to http:\/\/127\.0\.0\.1:17972\. This browser keeps the link: the editor asks OAIY Desktop whether it is running, and reads its services, each time it opens and every 10 seconds while it is open\. Disconnect ends that\.$/);
    assert.ok(up.attempts.includes(`GET ${DESKTOP}/api/services`), `it read the desktop's services as it opened: ${up.attempts.join(', ')}`);
    await up.context.close();
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
    assert.match((await button.getAttribute('title')) ?? '', /Connect sends one request to .* Until you press it this page sends nothing/);
    await button.click();
    await page.waitForFunction(() => /Nothing answered at/.test(document.querySelector('aside .oaiy-engine-get [role="status"]')?.textContent ?? ''), null, { timeout: 15_000 });
    assert.deepEqual(attempts, [`GET ${DESKTOP}/api/health`]);
    await context.close();
  });

  it('given a desktop (as OAIY\'s window gives it) the editor looks as it opens, as it always has: the two probes, and no Connect button', async () => {
    const { context, page, attempts } = await open('flows', { desktop: OAIY_WINDOW });
    await ready('flows', page);
    assert.equal(health(attempts).length, 2, `the detection and the service sync each look once: ${attempts.join(', ')}`);
    assert.ok(health(attempts).every((a) => a === `GET ${DESKTOP}/api/health`));
    assert.equal(await page.locator('[data-connect-desktop]').count(), 0, 'nothing to connect in OAIY\'s own window');
    // Where Connect would be, in Settings (the review's F3): the engine's card is there, and no Connect, no Disconnect and no words about
    // what a tab sends. (A count of the workspace only would pass whether or not the window is told from a tab.)
    await page.locator('.oaiy-sections-tabs button[aria-label="Settings"]').click();
    await page.getByText("OAIY's engine").first().waitFor({ timeout: 10_000 });
    assert.equal(await page.locator('[data-connect-desktop], [data-disconnect-desktop], .oaiy-connect, [data-connect-words]').count(), 0, 'no Connect anywhere on the page');
    assert.doesNotMatch(await page.locator('body').textContent(), /sends nothing to your computer or your network|Connect sends one request/);
    await context.close();
  });

  it('Reset in Settings, with no link kept from Connect, ends the looking: the poll stops and nothing goes out afterwards, not even to the default address', async () => {
    const { context, page, attempts } = await open('flows', { storage: { 'oaiy.engineBase': 'http://192.168.1.50:17972' } });
    await ready('flows', page);
    assert.equal(health(attempts).length, 2, `it looked at the address it was given: ${attempts.join(', ')}`);
    await page.locator('button[aria-label="Settings"]').click();
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    await page.waitForFunction(() => /not connected/.test(document.querySelector('main')?.textContent ?? ''), null, { timeout: 5000 });
    const before = attempts.length;
    await sleep(11_000);
    assert.equal(attempts.length, before, `nothing asked after Reset: ${attempts.slice(before).join(', ')}`);
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

describe('E5: the Agent\'s boot (the review\'s F9)', () => {
  const LAN = 'http://192.168.1.20:8080';
  const KEY = 'sk-oaiy-review-key-0123456789';

  // Every request to a local address is held this long before it is refused, the saved OAIY's among them: a network the person is not on.
  // The Agent gives up on a look at OAIY after 3 s (discoverOaiy), so a boot that waited for it would be 3 s late, and one that does not is not.
  const HELD_MS = 6000;
  const DESK = 'http://127.0.0.1:17972';

  it('the welcome does not wait for a saved OAIY that does not answer: it comes about a second after the look began, not after the look gave up (the review\'s F9)', async () => {
    const s = await open('agent', { refuseAfterMs: HELD_MS });
    await ready('agent', s.page);
    await seedOaiyLink(s.page, { origin: LAN, key: KEY });
    const asked = s.page.waitForRequest(`${LAN}/v1/discovery`, { timeout: 30_000 });
    await s.page.reload();
    await asked;
    const askedAt = Date.now();
    await s.page.waitForFunction(() => /Welcome!/.test(document.querySelector('.chat-log')?.textContent ?? ''), null, { timeout: 30_000 });
    const after = Date.now() - askedAt;
    assert.ok(after < 2000, `the welcome came ${after} ms after the saved OAIY was asked, which the look's 3 s limit would make 3000 or more: it waited for the look`);
    await s.context.close();
  });

  it('a paired tab asks its desktop while the saved OAIY is still being asked, not after the look has given up (the review\'s F9)', async () => {
    const s = await open('agent', { refuseAfterMs: HELD_MS });
    await ready('agent', s.page);
    await seedOaiyLink(s.page, { origin: LAN, key: KEY });
    await seedPairing(s.page, { origin: DESK, token: 'paired-token-0123456789' });
    s.details.length = 0;
    await s.page.reload();
    const first = (prefix) => s.details.find((d) => d.url.startsWith(prefix));
    for (let i = 0; i < 100 && !(first(`${LAN}/v1/discovery`) && first(`${DESK}/api/`)); i++) await sleep(100);
    const look = first(`${LAN}/v1/discovery`);
    const desk = first(`${DESK}/api/`);
    assert.ok(look, 'the saved OAIY was asked');
    assert.ok(desk, 'the desktop was asked');
    assert.ok(desk.at - look.at < 2000, `the desktop was asked ${desk.at - look.at} ms after the saved OAIY, which the look's 3 s limit would make 3000 or more: the boot waited for the look`);
    await s.context.close();
  });
});

describe('E5: the flow editor\'s media (the review\'s F4)', () => {
  // What a flow from someone else can carry: the addresses its nodes show. Each host is a different local range.
  const MEDIA = {
    outputImage: 'http://127.0.0.1:8188/view?filename=stale.png',
    outputVideo: 'http://192.168.77.7:8188/view?filename=stale.mp4',
    imageView: 'http://192.168.77.8/pic.png',
    imageSave: 'http://10.9.9.9/save.png',
    videoSave: 'http://172.16.4.4/clip.mp4',
  };
  // The picture an Input File node shows is not a run output (it is what the person chose; it is kept in the flow), so an import leaves it and
  // only the node's guard holds it back in a tab with no link.
  const PREVIEW = 'http://192.168.77.10/photo.png';
  const node = (id, type, x, y, data) => ({ id, type, position: { x, y }, data });
  const RUN_NODES = [
    node('out-image', 'output', 40, 40, { label: 'Result image', outputValue: MEDIA.outputImage }),
    node('out-video', 'output', 380, 40, { label: 'Result video', outputValue: MEDIA.outputVideo }),
    node('view', 'image_view', 720, 40, { imageUrl: MEDIA.imageView }),
    node('save', 'image_save', 40, 420, { imageUrl: MEDIA.imageSave, filename: 'kept' }),
    node('clip', 'video_save', 380, 420, { videoUrl: MEDIA.videoSave, filename: 'clip' }),
  ];
  const FILE_NODE = node('file', 'input_file', 720, 420, { fileName: 'photo.png', fileType: 'image', fileContent: 'x', imagePreview: PREVIEW });
  const project = (name = 'Shared flow', nodes = [...RUN_NODES, FILE_NODE]) => ({ version: '1.0.0', name, description: '', createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-01T00:00:00Z', flows: [{ id: 'shared-flow', name: 'Shared flow', description: '', createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-01T00:00:00Z', graph: { nodes, edges: [] } }], settings: {}, constants: [], llmEndpoints: [], imageGenEndpoints: [], httpPresets: [] });
  const asked = (attempts) => attempts.filter((a) => [...Object.values(MEDIA), PREVIEW].some((url) => a === `GET ${url}`));
  const shown = (page) => page.locator('.react-flow__node').count();

  it('a project that carries local addresses opens in a visitor\'s browser without one request to them, and the nodes say why', async () => {
    const { context, page, attempts, errors } = await open('flows', { storage: { oaiy_project: JSON.stringify(project()) } });
    await ready('flows', page);
    await page.waitForFunction(() => document.querySelectorAll('.react-flow__node').length >= 6, null, { timeout: 15_000 });
    await sleep(1500);
    assert.equal(await shown(page), 6, 'the flow is there');
    assert.deepEqual(asked(attempts), [], `nothing was asked of those addresses: ${attempts.join(', ')}`);
    assert.deepEqual(attempts, [], 'and nothing else of this computer or its network either');
    assert.ok((await page.locator('[data-blocked-media]').count()) >= 5, 'each node that would have loaded one says it did not');
    assert.match(await page.locator('[data-blocked-media]').first().textContent(), /^Blocked: this address is on your computer or network, and this page is not linked to OAIY Desktop\. Connect it in Settings/);
    assert.deepEqual(errors, []);
    await context.close();
  });

  it('the same project in a tab linked to a desktop, and in OAIY\'s own window, loads them as it always did (the control: the recorder sees each)', async () => {
    for (const how of [{ storage: { oaiy_project: JSON.stringify(project()), 'oaiy.desktopLinked': '1' } }, { desktop: OAIY_WINDOW, storage: { oaiy_project: JSON.stringify(project()) } }]) {
      const { context, page, attempts } = await open('flows', how);
      await ready('flows', page);
      await page.waitForFunction(() => document.querySelectorAll('.react-flow__node').length >= 6, null, { timeout: 15_000 });
      await sleep(1500);
      const got = asked(attempts);
      for (const url of [...Object.values(MEDIA), PREVIEW]) assert.ok(got.includes(`GET ${url}`), `${url} was asked for (${Object.keys(how).join('+')}): ${attempts.join(', ')}`);
      assert.equal(await page.locator('[data-blocked-media]').count(), 0, 'and nothing says it is blocked');
      await context.close();
    }
  });

  it('a project file imported into the editor arrives without the run outputs, so even a tab that IS linked asks nothing at the addresses it carried', async () => {
    const { context, page, attempts } = await open('flows', { storage: { 'oaiy.desktopLinked': '1' } });
    await ready('flows', page);
    await page.locator('button[aria-label="Import or export"]').click();
    await page.locator('input[type="file"][accept="application/json,.json"]').setInputFiles({ name: 'shared.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(project('Imported', RUN_NODES))) });
    await page.getByRole('button', { name: 'Replace and import' }).click();
    await page.waitForFunction(() => document.querySelectorAll('.react-flow__node').length >= 5, null, { timeout: 15_000 });
    await sleep(1500);
    assert.equal(await shown(page), 5, 'the flow came in, with all its nodes');
    assert.deepEqual(asked(attempts), [], `nothing was asked of the addresses it carried: ${attempts.join(', ')}`);
    assert.equal(await page.locator('[data-blocked-media]').count(), 0, 'they are not there to be blocked');
    const kept = await page.evaluate(() => JSON.parse(localStorage.getItem('oaiy_project') ?? '{}'));
    const carried = JSON.stringify(kept).match(/(127\.0\.0\.1:8188|192\.168\.77\.|10\.9\.9\.9|172\.16\.4\.4)/g);
    assert.equal(carried, null, 'nor kept in the project it was saved to');
    await context.close();
  });
});
