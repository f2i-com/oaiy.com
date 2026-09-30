/**
 * Capabilities (WA-03): what each app shows where it is. A tab that is not linked to OAIY Desktop shows no control that cannot work;
 * OAIY's own window (given the desktop) shows every control it always had.
 *
 * Against the real builds on their own hosts (tests/e2e/app-pages.mjs), with every request to this computer refused or, for the one
 * scenario that links a tab to a desktop, answered by a stand-in inside the browser (nothing leaves it).
 *
 *   - the Agent: the "Set up OAIY" button and the project list's Front desk and Set up entries belong to a desktop; a tab that is not
 *     paired shows the pairing chip instead, and OAIY's window shows the button and no pairing chip;
 *   - the flow editor: no Packages section, no dock and none of the nodes that drive OAIY Desktop in a tab that is not linked, and the
 *     Data page says it is kept for the session; Connect brings the dock and the nodes (never Packages) and Disconnect takes them away;
 *     OAIY's window has Packages, the nodes, and no session label, as it always did.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { sleep } from '../harness.mjs';
import { OAIY_WINDOW, appReady, fakeDesktop, openApp, startAppWorld } from '../app-pages.mjs';

let env;

before(async () => {
  env = await startAppWorld();
});

after(async () => {
  await env?.close();
});

const open = (app, options) => openApp(env, app, options);
const ready = (app, page) => appReady(app, page, 1000);
const BROWSER_NODES = /Browser (Session|Page|Extract|Action)/;

/** What the node palette lists for a search (the flow must be open). */
async function paletteFor(page, query) {
  const search = page.locator('[data-testid="node-palette"] input[placeholder="Search nodes"]');
  await search.fill(query);
  await sleep(300);
  return (await page.locator('[data-testid="node-palette"]').textContent()) ?? '';
}

/** A new flow, so the palette is open. `sidebar`: the web rail's button, else the sections' bar in OAIY's window. */
async function openAFlow(page, { sidebar }) {
  await page.locator(sidebar ? '.oaiy-new' : '.oaiy-sections-new').first().click();
  await page.getByRole('button', { name: /Create flow/i }).click();
  await page.waitForSelector('[data-testid="node-palette"]', { timeout: 15_000 });
  await sleep(500);
}

describe('the Agent', () => {
  it('a tab that is not paired shows no Set up OAIY, no Front desk and no Set up entry; it shows the pairing chip', async () => {
    const { context, page, errors } = await open('agent');
    await ready('agent', page);
    // (The project actions sit in a menu that is closed on OAIY's window's narrow header, so what is asked is the button's own `hidden`.)
    assert.equal(await page.locator('button.setup-oaiy').evaluate((e) => e.hidden), true, 'Set up OAIY');
    assert.equal(await page.locator('button.chip.phone').evaluate((e) => e.hidden), false);
    assert.equal((await page.locator('button.chip.phone').textContent()).trim(), 'desktop: not paired');
    const options = await page.evaluate(() => [...document.querySelectorAll('header.topbar select option')].map((o) => o.textContent));
    assert.ok(options.length > 0, 'the project list is there');
    assert.ok(options.every((text) => !/📞|⚙/.test(text)), `no Front desk or Set up entry: ${options.join(' | ')}`);
    assert.deepEqual(errors, []);
    await context.close();
  });

  it("OAIY's own window shows Set up OAIY and no pairing chip: nothing it had is hidden", async () => {
    const { context, page } = await open('agent', { desktop: OAIY_WINDOW });
    await ready('agent', page);
    assert.equal(await page.locator('button.setup-oaiy').evaluate((e) => e.hidden), false, 'Set up OAIY');
    assert.equal(await page.locator('button.chip.phone').evaluate((e) => e.hidden), true, 'nothing to pair in OAIY\'s own window');
    await context.close();
  });
});

describe('the flow editor in a tab', () => {
  it('has no Packages section and no dock, none of the nodes that drive OAIY Desktop, and says Data is kept for the session', async () => {
    const { context, page, errors } = await open('flows');
    await ready('flows', page);
    const nav = await page.locator('.oaiy-nav button').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
    assert.deepEqual(nav, ['Workflows', 'Data', 'Queue']);
    assert.equal(await page.locator('.oaiy-dock').count(), 0, 'no dock');
    await openAFlow(page, { sidebar: true });
    const browser = await paletteFor(page, 'Browser');
    assert.match(browser, /Browser Request/, 'a plain HTTP request stays');
    assert.doesNotMatch(browser, BROWSER_NODES);
    assert.doesNotMatch(await paletteFor(page, 'Ask the'), /Ask the Agent/);
    assert.doesNotMatch(await paletteFor(page, 'Folder'), /Folder Input/);
    await page.locator('.oaiy-nav button[aria-label="Data"]').click();
    assert.match(await page.locator('[data-testid="data-page"]').textContent(), /Kept for this session only: in a browser tab it lives in memory, and is gone when the tab is closed or reloaded\./);
    assert.deepEqual(errors, []);
    await context.close();
  });

  it('Connect brings the dock and the desktop\'s nodes, never Packages; Disconnect takes them away again', async () => {
    const { context, page } = await open('flows', { answer: fakeDesktop });
    await ready('flows', page);
    await openAFlow(page, { sidebar: true });
    assert.doesNotMatch(await paletteFor(page, 'Browser'), BROWSER_NODES, 'before Connect');
    await page.locator('aside [data-connect-desktop]').click();
    await page.locator('.oaiy-dock').waitFor({ timeout: 15_000 });
    assert.match(await page.locator('.oaiy-dock').textContent(), /companion connected/);
    assert.match(await paletteFor(page, 'Browser'), BROWSER_NODES, 'after Connect the browser nodes are offered');
    assert.match(await paletteFor(page, 'Ask the'), /Ask the Agent/);
    assert.match(await paletteFor(page, 'Folder'), /Folder Input/);
    const nav = await page.locator('.oaiy-nav button').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
    assert.deepEqual(nav, ['Workflows', 'Data', 'Queue'], 'Packages need the desktop\'s own commands, which no link gives a tab');
    assert.match(await paletteFor(page, 'Python'), /Python rig/, "and the desktop's service is in the palette");
    await page.locator('aside button', { hasText: 'Disconnect' }).click();
    await page.waitForFunction(() => document.querySelectorAll('.oaiy-dock').length === 0, null, { timeout: 5000 });
    assert.doesNotMatch(await paletteFor(page, 'Browser'), BROWSER_NODES, 'after Disconnect they are gone');
    assert.doesNotMatch(await paletteFor(page, 'Ask the'), /Ask the Agent/);
    await context.close();
  });
});

describe("the flow editor in OAIY's own window", () => {
  it('has the Packages section, the desktop\'s nodes and no session label on Data, as it always did, and nothing to connect', async () => {
    const { context, page, errors } = await open('flows', { desktop: OAIY_WINDOW });
    await ready('flows', page);
    const tabs = await page.locator('.oaiy-sections-tabs button').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
    assert.deepEqual(tabs, ['Workflows', 'Data', 'Queue', 'Packages', 'Settings']);
    assert.equal(await page.locator('.oaiy-dock').count(), 0, "OAIY's own window has never had the dock: its sidebar is beside the editor");
    assert.equal(await page.locator('[data-connect-desktop]').count(), 0);
    await openAFlow(page, { sidebar: false });
    assert.match(await paletteFor(page, 'Browser'), BROWSER_NODES);
    assert.match(await paletteFor(page, 'Ask the'), /Ask the Agent/);
    assert.match(await paletteFor(page, 'Folder'), /Folder Input/);
    await page.locator('.oaiy-sections-tabs button[aria-label="Data"]').click();
    const data = await page.locator('[data-testid="data-page"]').textContent();
    assert.doesNotMatch(data, /Kept for this session only/, "OAIY's window is left as it was");
    await page.locator('.oaiy-sections-tabs button[aria-label="Packages"]').click();
    await page.locator('[data-testid="packages-page"]').waitFor({ timeout: 5000 });
    assert.deepEqual(errors, []);
    await context.close();
  });
});
