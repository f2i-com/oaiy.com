/**
 * E2, isolation (design 8, 3.1 requirement 2 and 3.2's handshake): the Agent is cross-origin isolated (COOP same-origin, COEP
 * credentialless) and stays so with the providers frame embedded, because that frame sends COEP and a CORP that admits it; the flow
 * editor is not isolated and embeds the same frame without any of that.
 *
 * Inside the frame, `crossOriginIsolated` is a CHECK, not a verdict (the design says so): it is true under the Agent when the app
 * delegates the feature with `allow="cross-origin-isolated"` and false under the flow editor. If it were true without the delegation
 * the token would go from the handshake; what is observed is asserted below, and the assertion says what to do if it changes.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { launchBrowser, newContext, newPage, startWorld } from '../harness.mjs';

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

  it('inside the frame: isolated under the Agent (the app delegates it), not under the flow editor', async () => {
    const { context } = await newContext(browser);
    const agent = await newPage(context);
    await agent.page.goto(`${world.origins.agent}/`);
    await agent.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
    assert.equal(await (await frameOf(agent.page)).evaluate(() => crossOriginIsolated), true, 'under the Agent');

    const flows = await newPage(context);
    await flows.page.goto(`${world.origins.flows}/`);
    const hello = await flows.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
    assert.equal(hello?.t, 'hello', 'the flow editor embeds it with no COEP of its own');
    assert.equal(await (await frameOf(flows.page)).evaluate(() => crossOriginIsolated), false, 'under the flow editor');
    await context.close();
  });

  it('without the delegation the frame is not isolated under the Agent: the token is needed (if this fails, the token goes from the handshake)', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.agent}/`);
    const hello = await page.evaluate((origin) => window.oaiyTest.connect({ origin, allow: 'local-network-access' }), world.origins.providers);
    assert.equal(hello?.t, 'hello');
    assert.equal(await (await frameOf(page)).evaluate(() => crossOriginIsolated), false);
    await context.close();
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
