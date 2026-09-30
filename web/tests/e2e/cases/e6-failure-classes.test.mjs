/**
 * E6, failure classes (design 8, 4.2 step 4): the browser reports a server that is down, a server that is up and refused CORS, and
 * mixed content as one and the same "Failed to fetch". The providers origin asks once more (a `no-cors` request, which a server
 * that is up answers with an opaque response), so a person is told which it is and what to change.
 *
 *   - a server that is up and does not allow the page says "allow <this origin>", with the exact line for the server;
 *   - one that is down says nothing answered there;
 *   - a wrong key on the model list is a readable "did not accept this key" in words that are ours (an app is never given the provider's);
 *   - mixed content (an https page calling plain http on another machine) names itself: it cannot happen on these http test pages,
 *     so it is checked by unit tests (net.test.mjs, broker.test.mjs) with the page's protocol given, and said so here.
 *
 * The same words are shown in the Providers page ("Check connection") and given to an app over the port.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { browserVersion, launchBrowser, newContext, newPage, startWorld } from '../harness.mjs';
import { deadPort, startFakeProvider } from '../fake-provider.mjs';
import { addProvider } from '../providers-page.mjs';

const KEY = 'sk-e6-Zq7Rk2Lm9XvB4nT8wYc1HdJf6PsA3GuE5oIiN0';

let world;
let browser;
let refusing;
let allowing;
let dead;

before(async () => {
  world = await startWorld();
  // Up, and it has not been told to allow the providers origin: no CORS headers at all.
  refusing = await startFakeProvider({ cors: 'none' });
  // Up, allows the page, and wants a key.
  allowing = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] } });
  dead = await deadPort();
  browser = await launchBrowser({ isolateOrigins: [world.origins.providers] });
  console.log(`# browser: ${await browserVersion(browser)}`);
});

after(async () => {
  await browser?.close();
  await refusing?.close();
  await allowing?.close();
  await world?.close();
});

const server = (over) => ({ preset: 'A server on this computer', serverKind: 'ollama', ...over });

async function connectedApp(context) {
  const app = await newPage(context);
  await app.page.goto(`${world.origins.flows}/`);
  await app.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
  return app;
}

describe('E6 failure classes', () => {
  it('a server that is up and does not allow the page: the message says to allow this origin, with the line for the server', async () => {
    const { context } = await newContext(browser);
    const top = await newPage(context);
    await addProvider(top.page, world.origins.providers, server({ name: 'No CORS', baseUrl: refusing.baseUrl, key: '' }));
    const app = await connectedApp(context);
    const provider = (await app.page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result[0];

    for (const op of ['models', 'test']) {
      const reply = await app.page.evaluate((message) => window.oaiyTest.call(message), { op, provider: provider.id });
      assert.equal(reply.ok, true, op);
      assert.equal(reply.result.ok, false, op);
      assert.equal(reply.result.error.kind, 'cors', op);
      assert.match(reply.result.error.message, new RegExp(`does not allow requests from ${world.origins.providers.replace(/[.:/]/g, '\\$&')}`), op);
      assert.match(reply.result.error.message, new RegExp(`OLLAMA_ORIGINS="${world.origins.providers.replace(/[.:/]/g, '\\$&')}"`), `${op}: the exact line for Ollama`);
    }
    // And a fetch through the port, as an app makes one.
    const events = await app.page.evaluate(async (id) => {
      window.oaiyTest.startStream('f', { provider: id, path: '/chat/completions', method: 'POST', body: '{}', headers: [['content-type', 'application/json']] });
      return window.oaiyTest.streamDone('f');
    }, provider.id);
    assert.equal(events.at(-1).error.code, 'network');
    assert.match(events.at(-1).error.message, /does not allow requests from/);
    // The second look was a no-cors GET of the origin, with nothing of ours in it.
    const probe = refusing.log.find((r) => r.method === 'GET' && r.path === '/' && !r.headers.authorization);
    assert.ok(probe, 'the server was asked once more, for its origin');
    assert.equal(probe.headers.cookie, undefined);
    assert.equal(probe.headers['x-api-key'], undefined);
    await context.close();
  });

  it('the Providers page says the same to the person who presses Check connection', async () => {
    const { context } = await newContext(browser);
    const top = await newPage(context);
    await top.page.goto(`${world.origins.providers}/`);
    await top.page.locator('button.tile', { hasText: 'A server on this computer' }).click();
    const form = top.page.locator('section[aria-label="Add or change a provider"]');
    await form.locator('select[aria-label="Which server"]').selectOption('ollama');
    await form.locator('input[type=url]').fill(refusing.baseUrl);
    await form.getByRole('button', { name: 'Check connection' }).click();
    const bad = form.locator('.result.bad');
    await bad.first().waitFor({ timeout: 10000 });
    const text = await bad.first().innerText();
    assert.match(text, new RegExp(`Allow ${world.origins.providers.replace(/[.:/]/g, '\\$&')}`));
    assert.match(text, /OLLAMA_ORIGINS=/);
    await context.close();
  });

  it('a server that is down: nothing answered there', async () => {
    const { context } = await newContext(browser);
    const top = await newPage(context);
    await addProvider(top.page, world.origins.providers, server({ name: 'Down', baseUrl: `http://127.0.0.1:${dead}/v1`, key: '' }));
    const app = await connectedApp(context);
    const provider = (await app.page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result[0];
    const reply = await app.page.evaluate((message) => window.oaiyTest.call(message), { op: 'test', provider: provider.id });
    assert.equal(reply.result.ok, false);
    assert.equal(reply.result.error.kind, 'unreachable');
    assert.match(reply.result.error.message, new RegExp(`Nothing answered at http://127\\.0\\.0\\.1:${dead}`));
    assert.match(reply.result.error.message, /local network/, 'and, for this computer, the permission a browser may want');
    assert.doesNotMatch(reply.result.error.message, /does not allow requests/);
    await context.close();
  });

  it('a wrong key on the model list is a readable "did not accept this key" in words that are ours: none of the provider\'s, and none of the key', async () => {
    const { context } = await newContext(browser);
    const top = await newPage(context);
    const WRONG = 'sk-e6-WRONG-Kx9Pq3Vn7Ld2Tb8Ye4Hs1Mj6Rc5Gf0Za';
    await addProvider(top.page, world.origins.providers, server({ name: 'Wrong key', serverKind: 'other', baseUrl: allowing.baseUrl, key: WRONG }));
    const app = await connectedApp(context);
    const provider = (await app.page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result[0];
    const reply = await app.page.evaluate((message) => window.oaiyTest.call(message), { op: 'models', provider: provider.id });
    assert.equal(reply.result.ok, false);
    assert.equal(reply.result.error.kind, 'auth');
    assert.equal(reply.result.error.status, 401);
    assert.match(reply.result.error.message, /did not accept (this API key|the one given) \(401\)/);
    assert.doesNotMatch(reply.result.error.message, /Incorrect API key provided|The provider said/, 'not the provider\'s words');
    assert.ok(!reply.result.error.message.includes(WRONG.slice(-6)) && !reply.result.error.message.includes(WRONG.slice(0, 8)), 'and not the key it quoted, masked');
    // The Providers page is the holder's own, and shows the provider's words as text, scrubbed.
    await top.page.goto(`${world.origins.providers}/`);
    const row = top.page.locator('section[aria-label="Your providers"] li.row', { hasText: 'Wrong key' });
    await row.getByRole('button', { name: 'Check' }).click();
    await row.locator('.result.bad').waitFor({ timeout: 10000 });
    const shown = await row.locator('.result.bad').innerText();
    assert.match(shown, /The provider said: “Incorrect API key provided: …/);
    assert.ok(!shown.includes(WRONG.slice(-6)) && !shown.includes(WRONG.slice(0, 8)));
    await context.close();
  });

  it('a key that is right works, lists the models, and the tools probe says what the model does', async () => {
    const { context } = await newContext(browser);
    const top = await newPage(context);
    await addProvider(top.page, world.origins.providers, server({ name: 'Right key', serverKind: 'other', baseUrl: allowing.baseUrl, key: KEY, model: 'fake-chat' }));
    const app = await connectedApp(context);
    const provider = (await app.page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result[0];
    const reply = await app.page.evaluate((message) => window.oaiyTest.call(message), { op: 'test', provider: provider.id });
    assert.equal(reply.result.ok, true, JSON.stringify(reply.result));
    assert.deepEqual(reply.result.models.map((m) => m.id).sort(), ['fake-chat', 'fake-embed-1', 'fake-tts']);
    assert.equal(reply.result.tools, 'yes');
    allowing.set({ rejectTools: true });
    const no = await app.page.evaluate((message) => window.oaiyTest.call(message), { op: 'test', provider: provider.id });
    allowing.set({ rejectTools: false });
    assert.equal(no.result.ok, true);
    assert.equal(no.result.tools, 'no', 'a model that rejects a tools array says so');
    await context.close();
  });
});
