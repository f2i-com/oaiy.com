/**
 * E2, isolation (design 8, 3.1 requirement 2 and 3.2's handshake): the Agent is cross-origin isolated (COOP same-origin, COEP
 * credentialless) and stays so with the providers frame embedded, because that frame sends COEP and a CORP that admits it; the flow
 * editor is not isolated and embeds the same frame without any of that.
 *
 * Inside the frame, `crossOriginIsolated` is a CHECK, not a verdict (the design says so): what is observed is asserted below. It is true
 * under the Agent only when the app delegates the feature with `allow="cross-origin-isolated"`, and false under the flow editor. The
 * apps do NOT delegate it in v0: an isolated holder has no use yet (the on-device engine, WA-09, may need one) and is easier for a
 * co-located Spectre-class attack to read, so the holder stays UNISOLATED, and the last test of the group below fails if a default
 * handshake ever delegates it.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { launchBrowser, newContext, newPage, startWorld } from '../harness.mjs';
import { startFakeProvider } from '../fake-provider.mjs';
import { addProvider } from '../providers-page.mjs';

let world;
let browser;

before(async () => {
  world = await startWorld();
  browser = await launchBrowser({ isolateOrigins: [world.origins.providers] });
});

after(async () => {
  await browser?.close();
  await world?.close();
});

async function frameOf(page) {
  const frame = page.frames().find((f) => f.url().startsWith(world.origins.providers));
  assert.ok(frame, 'the providers frame is there');
  return frame;
}

describe('E2 isolation', () => {
  it('the Agent is cross-origin isolated with the providers frame embedded, and the frame answers', async () => {
    const { context } = await newContext(browser);
    const { page, seen } = await newPage(context);
    await page.goto(`${world.origins.agent}/`);
    const hello = await page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
    assert.equal(hello?.t, 'hello', 'COEP and CORP let the frame in');
    assert.equal(await page.evaluate(() => crossOriginIsolated), true);
    assert.equal(await page.evaluate(() => typeof SharedArrayBuffer), 'function', 'so the sandbox\'s Atomics.wait has what it needs');
    const blocked = seen.failures.filter((f) => /BLOCKED|COEP|CORP/i.test(f.reason ?? ''));
    assert.deepEqual(blocked, [], 'nothing was blocked by the isolation policy');
    assert.ok(!seen.console.some((m) => /Cross-Origin-Embedder-Policy|Cross-Origin-Resource-Policy/i.test(m.text)), seen.console.map((m) => m.text).join(' | '));
    await context.close();
  });

  it('inside the frame: NOT isolated, under the Agent or under the flow editor (v0 does not delegate the feature)', async () => {
    const { context } = await newContext(browser);
    const agent = await newPage(context);
    await agent.page.goto(`${world.origins.agent}/`);
    await agent.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
    assert.equal(await (await frameOf(agent.page)).evaluate(() => crossOriginIsolated), false, 'under the Agent');
    assert.equal(await (await frameOf(agent.page)).evaluate(() => typeof SharedArrayBuffer), 'undefined', 'and with no SharedArrayBuffer to share a co-located reader');

    const flows = await newPage(context);
    await flows.page.goto(`${world.origins.flows}/`);
    const hello = await flows.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
    assert.equal(hello?.t, 'hello', 'the flow editor embeds it with no COEP of its own');
    assert.equal(await (await frameOf(flows.page)).evaluate(() => crossOriginIsolated), false, 'under the flow editor');
    await context.close();
  });

  it('the feature could be delegated (the engine, WA-09, may want it): with the token the frame IS isolated under the Agent, so it is the embedder\'s choice that keeps it off', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.agent}/`);
    const hello = await page.evaluate((origin) => window.oaiyTest.connect({ origin, allow: 'cross-origin-isolated' }), world.origins.providers);
    assert.equal(hello?.t, 'hello');
    assert.equal(await (await frameOf(page)).evaluate(() => crossOriginIsolated), true);
    await context.close();
  });
  it('an embedder that gives the frame the `credentialless` attribute gets an anonymous, empty store, so the frame must not be given it (3.1 requirement 2)', async () => {
    const fake = await startFakeProvider({ cors: { allowOrigins: [world.origins.providers] } });
    const { context } = await newContext(browser);
    try {
      const top = await newPage(context);
      await addProvider(top.page, world.origins.providers, { name: 'Saved at the top', preset: 'A server on this computer', serverKind: 'other', baseUrl: fake.baseUrl, key: '' });
      const { page } = await newPage(context);
      await page.goto(`${world.origins.agent}/`);
      const normal = await page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
      assert.equal(normal?.t, 'hello');
      assert.equal((await page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result.length, 1, 'control: an ordinary frame lists it');
      await page.evaluate(() => window.oaiyTest.disconnect());

      const anonymous = await page.evaluate((origin) => window.oaiyTest.connect({ origin, credentialless: true }), world.origins.providers);
      assert.equal(anonymous?.t, 'hello', 'it loads and answers');
      const listed = await page.evaluate(() => window.oaiyTest.call({ op: 'list' }));
      assert.deepEqual(listed.result, [], 'but its storage is another one, empty: nothing saved at the top-level window is in it');
    } finally {
      await context.close();
      await fake.close();
    }
  });

  it('a top-level visit of the providers origin is isolated by its own COOP and COEP', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    const response = await page.goto(`${world.origins.providers}/`);
    assert.equal(response.headers()['cross-origin-opener-policy'], 'same-origin');
    assert.equal(await page.evaluate(() => crossOriginIsolated), true);
    await context.close();
  });
});
