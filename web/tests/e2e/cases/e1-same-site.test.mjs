/**
 * E1, same site (design 8, 3.1 requirement 1): the three hosts share one store through the frame, in Chromium and in Firefox.
 *
 * The Agent, the flow editor and the providers origin are three ORIGINS of one registrable domain. A browser partitions a
 * cross-site embedded frame's storage by the top-level site and leaves a same-site frame's alone, so a provider saved at the
 * top-level providers window is the very one the hidden frame under the Agent and under the flow editor lists. The same test with
 * an app on ANOTHER registrable domain must fail: it is why the domain has to be one the owner registers, and not the default
 * hostnames of a hosting service (`pages.dev`, `netlify.app` and the rest are public suffixes, so each host there is its own site).
 *
 * The sharing case runs against the headers exactly as the template renders them. The other-domain case needs one change to them (the
 * providers documents say CORP `cross-origin` instead of `same-site`), because an app on another site could not embed the frame at all
 * otherwise, and what is being shown is that its STORAGE is another one, not that it is refused.
 *
 * WebKit is not a target (informational in the design) and is not run.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { browserVersion, launchBrowser, newContext, newPage, startWorld, waitFor } from '../harness.mjs';
import { startFakeProvider } from '../fake-provider.mjs';
import { addProvider, editProvider } from '../providers-page.mjs';

const KEY = 'sk-e1-Zq7Rk2Lm9XvB4nT8wYc1HdJf6PsA3GuE5oIiN0';

for (const browserName of ['chromium', 'firefox']) {
  describe(`E1 same site (${browserName})`, () => {
    let browser;
    let same; // the real headers
    let split; // the app on another domain may embed the frame
    let fakeSame;
    let fakeSplit;

    before(async () => {
      same = await startWorld({ apps: ['agent', 'flows'] });
      split = await startWorld({ apps: ['agent', 'flows', 'foreign'], providersHeaders: (rendered) => rendered.replaceAll('Cross-Origin-Resource-Policy: same-site', 'Cross-Origin-Resource-Policy: cross-origin') });
      fakeSame = await startFakeProvider({ key: KEY, cors: { allowOrigins: [same.origins.providers] } });
      fakeSplit = await startFakeProvider({ key: KEY, cors: { allowOrigins: [split.origins.providers] } });
      browser = await launchBrowser({ isolateOrigins: [same.origins.providers, split.origins.providers], browserName });
      if (browser) console.log(`# browser: ${await browserVersion(browser)}`);
      else console.log('# firefox could not be started here: E1 was not run in it');
    });

    after(async () => {
      await browser?.close();
      await fakeSame?.close();
      await fakeSplit?.close();
      await same?.close();
      await split?.close();
    });

    // Firefox starts slowly the first time and is slower still when other browsers are running: it is given time, not a different test.
    const details = (fake) => ({ name: 'Shared fake', preset: 'A server on this computer', serverKind: 'other', baseUrl: fake.baseUrl, key: KEY, model: 'fake-chat', check: true, timeout: browserName === 'firefox' ? 60000 : 10000 });

    async function appSees(world, context, name) {
      const { page, seen } = await newPage(context);
      await page.goto(`${world.origins[name]}/`);
      const hello = await page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
      return { page, seen, hello };
    }

    it('what the top-level providers window saves is what the frame lists under the Agent and under the flow editor, and a change is pushed to both', async (t) => {
      if (!browser) return t.skip('no Firefox is installed that Playwright can start');
      const { context } = await newContext(browser);
      const { page: top } = await newPage(context);
      await addProvider(top, same.origins.providers, details(fakeSame));

      for (const app of ['agent', 'flows']) {
        const { page, hello } = await appSees(same, context, app);
        assert.equal(hello?.t, 'hello', `${app}: the frame answered`);
        assert.equal(hello.app, app);
        const listed = await page.evaluate(() => window.oaiyTest.call({ op: 'list' }));
        assert.equal(listed.ok, true);
        assert.equal(listed.result.length, 1, `${app} sees the provider saved in the other window`);
        assert.deepEqual({ name: listed.result[0].name, hasKey: listed.result[0].hasKey, host: listed.result[0].host, kind: listed.result[0].kind }, { name: 'Shared fake', hasKey: true, host: new URL(fakeSame.baseUrl).host, kind: 'local-server' });
      }

      // A change in the top-level window reaches the apps that are open, through the frame.
      const agent = await appSees(same, context, 'agent');
      const flows = await appSees(same, context, 'flows');
      const form = await editProvider(top, 'Shared fake');
      await form.locator('input[type=text]').first().fill('Renamed fake');
      await form.getByRole('button', { name: 'Save changes' }).click();
      for (const { page } of [agent, flows]) {
        await waitFor(() => page.evaluate(() => window.oaiyTest.pushes().length > 0), { what: 'a changed push', timeout: browserName === 'firefox' ? 30000 : 8000 });
        const listed = await page.evaluate(() => window.oaiyTest.call({ op: 'list' }));
        assert.equal(listed.result[0].name, 'Renamed fake');
        assert.equal(listed.result[0].hasKey, true, 'and the key it was saved with is still there');
      }
      await context.close();
    });

    it('the frame under an app on ANOTHER registrable domain is partitioned: it does not see what the top-level window saved (why the domain must be one the owner registers)', async (t) => {
      if (!browser) return t.skip('no Firefox is installed that Playwright can start');
      const { context } = await newContext(browser);
      const { page: top } = await newPage(context);
      await addProvider(top, split.origins.providers, details(fakeSplit));

      const sameSite = await appSees(split, context, 'agent');
      assert.equal((await sameSite.page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result.length, 1, 'control: the same-site app sees it');

      const foreign = await appSees(split, context, 'foreign');
      assert.equal(foreign.hello?.t, 'hello', 'the frame loads and answers under the other-domain app');
      const listed = await foreign.page.evaluate(() => window.oaiyTest.call({ op: 'list' }));
      assert.equal(listed.ok, true);
      assert.deepEqual(listed.result, [], 'but its store is another one: nothing saved at the top-level window is in it');
      await context.close();
    });
  });
}
