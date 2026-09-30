/**
 * The Providers page as a person uses it, in a real browser (WA-04): adding, checking, editing and removing a provider, the
 * keyed-address rule of design 6 (threat 2) in the form itself, what the page says about where keys are kept, and the one place
 * `navigator.storage.persist()` is asked for (threat 12): the top-level page, never the hidden frame or the modal.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { browserVersion, launchBrowser, newContext, newPage, startWorld, waitFor } from '../harness.mjs';
import { startFakeProvider } from '../fake-provider.mjs';
import { addProvider, editProvider, listedNames } from '../providers-page.mjs';

const KEY = 'sk-pp-Zq7Rk2Lm9XvB4nT8wYc1HdJf6PsA3GuE5oIiN0';

let world;
let browser;
let fake;
let other;

before(async () => {
  world = await startWorld();
  fake = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] } });
  other = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] } });
  browser = await launchBrowser({ isolateOrigins: [world.origins.providers] });
  console.log(`# browser: ${await browserVersion(browser)}`);
});

after(async () => {
  await browser?.close();
  await fake?.close();
  await other?.close();
  await world?.close();
});

const details = (over = {}) => ({ name: 'Page fake', preset: 'A server on this computer', serverKind: 'other', baseUrl: fake.baseUrl, key: KEY, model: 'fake-chat', check: true, ...over });

describe('the Providers page', () => {
  it('says where keys are kept, honestly, and to type them only here', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.providers}/`);
    await page.getByText('sealed on this device').waitFor({ timeout: 8000 });
    const text = await page.locator('main').innerText();
    assert.match(text, /sealed on this device/);
    assert.match(text, /does not protect them from someone who copies this browser/);
    assert.match(text, /Type a key only on this page, at providers\.web\.localhost:\d+/);
    assert.match(text, /never see them/);
    await context.close();
  });

  it('a provider is added with its key typed in the password field, checked, listed, and then changed and removed', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await addProvider(page, world.origins.providers, details());
    assert.deepEqual(await listedNames(page), ['Page fake']);
    const row = page.locator('section[aria-label="Your providers"] li.row');
    assert.match(await row.innerText(), /key stored/);
    assert.match(await row.innerText(), /fake-chat/);
    assert.ok(!(await page.locator('body').innerHTML()).includes(KEY), 'the key is nowhere in the page after it is saved');
    assert.equal(await page.locator('input[type=password]').count(), 0, 'and the form is closed');

    // Check from the list uses the stored key.
    await row.getByRole('button', { name: 'Check' }).click();
    await row.locator('.result.good').waitFor({ timeout: 10000 });

    const form = await editProvider(page, 'Page fake');
    await form.locator('input[type=text]').first().fill('Renamed');
    assert.equal(await form.locator('input[type=password]').inputValue(), '', 'the stored key is not put back in the form');
    await form.getByRole('button', { name: 'Save changes' }).click();
    await page.locator('section[aria-label="Your providers"] li.row', { hasText: 'Renamed' }).waitFor();
    assert.match(await page.locator('section[aria-label="Your providers"] li.row').innerText(), /key stored/, 'renaming kept the key');

    const remove = page.locator('section[aria-label="Your providers"] li.row').getByRole('button', { name: 'Remove' });
    await remove.click();
    await page.getByRole('button', { name: 'Really remove?' }).click();
    await page.getByText('None yet').waitFor();
    await context.close();
  });

  it('a stored key is not sent to a new address without being typed again: not to check it, and not to save it', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await addProvider(page, world.origins.providers, details({ name: 'Keyed' }));
    const asked = other.log.length;
    const form = await editProvider(page, 'Keyed');
    await form.locator('input[type=url]').fill(other.baseUrl);

    await form.getByRole('button', { name: 'Check connection' }).click();
    await form.locator('.result.bad', { hasText: 'Type its key again' }).waitFor({ timeout: 8000 });
    assert.equal(other.log.length, asked, 'nothing was sent to the new address');

    await form.getByRole('button', { name: 'Save changes' }).click();
    await form.locator('.problems', { hasText: 'Type its key again' }).waitFor({ timeout: 8000 });
    assert.equal(other.log.length, asked, 'still nothing');
    assert.match(await page.locator('section[aria-label="Your providers"] li.row').innerText(), new RegExp(new URL(fake.baseUrl).host), 'and the provider still points where it did');

    // Typed again, both work, and the new address gets the key that was typed.
    await form.locator('input[type=password]').fill(KEY);
    await form.getByRole('button', { name: 'Check connection' }).click();
    await form.locator('.result.good').waitFor({ timeout: 10000 });
    assert.ok(other.log.some((r) => r.headers.authorization === `Bearer ${KEY}`));
    await form.getByRole('button', { name: 'Save changes' }).click();
    await page.locator('section[aria-label="Your providers"] li.row', { hasText: new URL(other.baseUrl).host }).waitFor({ timeout: 8000 });
    await context.close();
  });

  it('a form that is not valid says what is wrong and saves nothing: a service on the internet has to be https', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.providers}/`);
    await page.locator('button.tile', { hasText: 'Another OpenAI-compatible service' }).click();
    const form = page.locator('section[aria-label="Add or change a provider"]');
    await form.locator('input[type=text]').first().fill('Plain http');
    await form.locator('input[type=url]').fill('http://api.example.com/v1');
    await form.getByRole('button', { name: 'Add provider' }).click();
    await form.locator('.problems', { hasText: 'has to be https' }).waitFor({ timeout: 8000 });
    assert.deepEqual(await listedNames(page), []);
    await context.close();
  });

  it('each app has an hourly limit, shown with its usage and changed only here', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.providers}/`);
    const apps = page.locator('section[aria-label="Apps"]');
    await apps.getByText('0 of 600 requests used this hour').first().waitFor();
    const agent = apps.locator('li.row', { hasText: 'agent' });
    await agent.locator('input[type=number]').fill('7');
    await agent.getByRole('button', { name: 'Set' }).click();
    await agent.getByText('Saved.').waitFor();
    await page.reload();
    await apps.locator('li.row', { hasText: 'agent' }).getByText('of 7 requests used this hour').waitFor();
    await context.close();
  });
});

describe('storage.persist()', () => {
  it('is asked for once by the top-level Providers page, and never by the port or the modal', async () => {
    const { context } = await newContext(browser);
    await context.addInitScript(() => {
      const calls = (window.__persistCalls = []);
      if (navigator.storage) {
        const real = navigator.storage.persist?.bind(navigator.storage);
        navigator.storage.persist = (...args) => {
          calls.push(location.pathname);
          return real ? real(...args) : Promise.resolve(false);
        };
      }
    });
    const top = await newPage(context);
    await top.page.goto(`${world.origins.providers}/`);
    await waitFor(() => top.page.evaluate(() => window.__persistCalls.length > 0), { what: 'persist() from the top-level page' });
    assert.deepEqual(await top.page.evaluate(() => window.__persistCalls), ['/'], 'the Providers page asked, once');

    const app = await newPage(context);
    await app.page.goto(`${world.origins.flows}/`);
    await app.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
    await app.page.evaluate(() => {
      const frame = document.createElement('iframe');
      frame.src = `${location.protocol}//providers.web.localhost:${new URL(document.querySelector('iframe').src).port}/embed.html`;
      document.body.append(frame);
    });
    const frames = await waitFor(() => app.page.frames().filter((f) => f.url().startsWith(world.origins.providers)).length === 2 && app.page.frames().filter((f) => f.url().startsWith(world.origins.providers)), { what: 'both frames' });
    await new Promise((resolve) => setTimeout(resolve, 500));
    for (const frame of frames) assert.deepEqual(await frame.evaluate(() => window.__persistCalls), [], `no call from ${frame.url()}`);
    await context.close();
  });
});
