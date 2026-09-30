/**
 * E3, keys never cross (design 8, 6 and 3.1's table of what a hostile party gets), in a real browser against the assembled folders.
 *
 * Script in the flow editor (an AI-written or imported flow runs in its main window) or in the Agent can USE a provider's key, within
 * a budget, and can never read it, redirect it or edit the provider:
 *
 *   1. the app origins' storage and memory hold no key, in any form, after the key was used by that very page;
 *   2. the port has no operation that returns one, and a hostile script that tries every name and every shape gets nothing;
 *   3. a `fetch` that carries its own `baseUrl`, or a path such as `//evil.example/x`, `/..%2f` or `/a@b`, goes nowhere new;
 *   4. only the allowed origins, from the parent window, are answered: not `evil.web.localhost`, not a sandboxed frame, not a
 *      sibling frame; and no other site may frame the holder, nor any app frame the Providers page where a key is typed;
 *   5. the embedded modal has no input at all, and the Providers page has the password and address inputs it should;
 *   6. the policy of every document is the one that makes 4 hold.
 *
 * `npm run test:mutations` breaks five things on purpose (drops frame-ancestors, accepts a baseUrl from the message, accepts
 * //evil.example/x, skips the origin check, adds a getKey operation) and requires the tests below to fail for each.
 *
 * A search that finds nothing proves nothing, so the storage and memory scans are shown to find a key that is there (the controls).
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';
import { browserVersion, launchBrowser, newContext, newPage, startWorld, waitFor } from '../harness.mjs';
import { parseHeaders, headersFor } from '../hosts.mjs';
import { startCollector, startFakeProvider } from '../fake-provider.mjs';
import { findInStorage, forms, heapHolds } from '../leakscan.mjs';
import { addProvider } from '../providers-page.mjs';

const KEY = 'sk-e3-Zq7Rk2Lm9XvB4nT8wYc1HdJf6PsA3GuE5oIiN0-QeUxVtWyMaS';
/** The key in every form a page might hold it in, and long pieces of it (a prefix, a suffix, a middle). */
const NEEDLES = [...forms(KEY), KEY.slice(0, 14), KEY.slice(-14), KEY.slice(12, 30)];

let world;
let browser;
let fake;
let collector;

before(async () => {
  world = await startWorld();
  fake = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] }, delayMs: 20 });
  collector = await startCollector();
  // No `--isolate-origins`: the holder's own header (Origin-Agent-Cluster) is what gives it a process of its own, and the memory scans
  // below mean what they say only because it does. The last test of E3.1 shows what happens without the header.
  browser = await launchBrowser();
  console.log(`# browser: ${await browserVersion(browser)}`);
});

after(async () => {
  await browser?.close();
  await collector?.close();
  await fake?.close();
  await world?.close();
});

const DETAILS = () => ({ name: 'Fake', preset: 'A server on this computer', serverKind: 'other', baseUrl: fake.baseUrl, key: KEY, model: 'fake-chat', check: true });

/** A context with the provider saved (its key typed at the top-level window) and a page for an app, connected to the frame. */
async function scenario(app, opts = {}) {
  const { context } = await newContext(browser);
  const top = await newPage(context);
  await addProvider(top.page, world.origins.providers, DETAILS());
  const appPage = await newPage(context);
  await appPage.page.goto(`${world.origins[app]}/`);
  const hello = await appPage.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
  const listed = await appPage.page.evaluate(() => window.oaiyTest.call({ op: 'list' }));
  return { context, top, app: appPage, hello, provider: listed.result?.[0], ...opts };
}

const call = (page, message) => page.evaluate((m) => window.oaiyTest.call(m), message);
const fetchAll = (page, name, message) =>
  page.evaluate(async ([n, m]) => {
    window.oaiyTest.startStream(n, m);
    return window.oaiyTest.streamDone(n);
  }, [name, message]);

describe('E3.1 no key in the app origins\' storage or memory', () => {
  for (const app of ['flows', 'agent']) {
    it(`${app}: after the page used the key through every operation, nothing of it is in its storage, its port traffic or its memory`, async () => {
      const s = await scenario(app);
      assert.equal(s.hello?.t, 'hello');
      const id = s.provider.id;
      const page = s.app.page;

      // Every operation there is, used for real: the fake provider receives the key from the HOLDER each time.
      assert.equal((await call(page, { op: 'status' })).ok, true);
      assert.equal((await call(page, { op: 'ui.open', target: 'pick' })).ok, true);
      const models = await call(page, { op: 'models', provider: id });
      assert.equal(models.result.ok, true, JSON.stringify(models));
      const tested = await call(page, { op: 'test', provider: id });
      assert.equal(tested.result.ok, true, JSON.stringify(tested));
      await call(page, { op: 'probe', provider: id });
      await call(page, { op: 'setModel', provider: id, model: 'fake-chat' });
      const stream = await fetchAll(page, 's1', { provider: id, path: '/chat/completions', method: 'POST', body: JSON.stringify({ model: 'fake-chat', stream: true, messages: [{ role: 'user', content: 'hi' }] }), headers: [['content-type', 'application/json']] });
      assert.equal(stream.at(-1).t, 'end');
      const streamed = stream.filter((e) => e.t === 'chunk').map((e) => e.text).join('');
      assert.ok(streamed.includes('"content":"Hello"') && streamed.includes('data: [DONE]'), streamed);
      const authed = fake.requests((r) => r.path === '/v1/chat/completions' || r.path === '/v1/models').filter((r) => r.method !== 'OPTIONS');
      assert.ok(authed.length >= 3 && authed.every((r) => r.headers.authorization === `Bearer ${KEY}`), 'the holder attached the key to every request');

      // Its storage, everything it received, and its memory.
      const dump = await page.evaluate(() => window.oaiyTest.dumpStorage());
      assert.deepEqual(findInStorage(dump, NEEDLES), [], 'storage');
      assert.deepEqual(findInStorage({ indexedDB: [], localStorage: {}, sessionStorage: {}, caches: [], cookies: await page.evaluate(() => window.oaiyTest.received()) }, NEEDLES), [], 'everything received over the port');
      assert.deepEqual(await heapHolds(s.context, page, NEEDLES), [], 'memory');
      // The scans above are not deaf: the same calls find the key in a page that was handed it.
      await page.evaluate((key) => {
        window.__planted = { key };
        localStorage.setItem('planted', key);
      }, KEY);
      assert.notDeepEqual(findInStorage(await page.evaluate(() => window.oaiyTest.dumpStorage()), NEEDLES), [], 'control: storage scan');
      assert.deepEqual(await heapHolds(s.context, page, [KEY]), [KEY], 'control: heap scan');
      await s.context.close();
    });
  }

  it('the holder is out of the app\'s process, so the app\'s heap is not the holder\'s (the memory scan means what it says)', async () => {
    const s = await scenario('flows');
    const frame = s.app.page.frames().find((f) => f.url().startsWith(world.origins.providers));
    assert.ok(frame);
    const session = await s.context.newCDPSession(frame).catch((e) => e);
    assert.ok(!(session instanceof Error), `the frame has a process of its own: ${session instanceof Error ? session.message : ''}`);
    await session.detach?.();
    await s.context.close();
  });

  it('and that is the doing of its Origin-Agent-Cluster header: without it two same-site frames share a process (control)', async () => {
    const plain = await startWorld({ providersHeaders: (rendered) => rendered.replaceAll('  Origin-Agent-Cluster: ?1\n', '') });
    const { context } = await newContext(browser);
    try {
      const app = await newPage(context);
      await app.page.goto(`${plain.origins.flows}/`);
      await app.page.evaluate((origin) => window.oaiyTest.connect({ origin }), plain.origins.providers);
      const frame = app.page.frames().find((f) => f.url().startsWith(plain.origins.providers));
      const outcome = await context.newCDPSession(frame).then((s) => s.detach().then(() => 'own process'), (e) => e.message);
      assert.match(outcome, /does not have a separate CDP session/, 'without the header the frame is part of the app\'s process');
    } finally {
      await context.close();
      await plain.close();
    }
  });

  it('and the key is in the providers origin only as ciphertext: not in its localStorage, its caches or its database', async () => {
    const s = await scenario('flows');
    const dump = await s.top.page.evaluate(async () => {
      const out = { indexedDB: [], localStorage: {}, sessionStorage: {}, caches: [], cookies: document.cookie };
      for (let i = 0; i < localStorage.length; i++) out.localStorage[localStorage.key(i)] = localStorage.getItem(localStorage.key(i));
      for (const { name } of await indexedDB.databases()) {
        const db = await new Promise((resolve, reject) => { const open = indexedDB.open(name); open.onsuccess = () => resolve(open.result); open.onerror = () => reject(open.error); });
        const one = { name, stores: {} };
        for (const store of db.objectStoreNames) {
          const values = await new Promise((resolve) => { const r = db.transaction(store).objectStore(store).getAll(); r.onsuccess = () => resolve(r.result); });
          one.stores[store] = values.map((v) => JSON.stringify(v, (k, x) => (x instanceof Uint8Array ? Array.from(x) : x)));
        }
        db.close();
        out.indexedDB.push(one);
      }
      return out;
    });
    assert.equal(dump.indexedDB.length, 1);
    assert.deepEqual(findInStorage(dump, NEEDLES), []);
    const vault = dump.indexedDB[0].stores.vault;
    assert.equal(vault.length, 1, 'a sealed record is there');
    assert.match(vault[0], /"wrappers"/);
    await s.context.close();
  });
});

describe('E3.2 a hostile script in the flow editor tries every operation', () => {
  it('the holder\'s hello lists exactly the nine operations, and every other name a script can think of is refused', async () => {
    const s = await scenario('flows');
    assert.deepEqual(s.hello.ops, ['abort', 'fetch', 'list', 'models', 'probe', 'setModel', 'status', 'test', 'ui.open']);
    const id = s.provider.id;
    const hostile = ['getKey', 'get', 'getSecret', 'getSecrets', 'secret', 'key', 'reveal', 'export', 'import', 'dump', 'backup', 'create', 'add', 'addProvider', 'edit', 'update', 'put', 'save', 'set', 'setKey', 'setBaseUrl', 'setUrl', 'delete', 'remove', 'deleteProvider', 'unlock', 'lock', 'setPassphrase', 'setMode', 'wrappers', 'vault', 'names', 'db', 'eval', 'hello', 'constructor', '__proto__', 'toString'];
    for (const op of hostile) {
      const reply = await call(s.app.page, { op, provider: id, name: `key:${id}`, names: [`key:${id}`], key: 'x', baseUrl: 'http://evil.example' });
      assert.equal(reply.ok, false, op);
      assert.equal(reply.error?.code, 'unknown-op', `${op} is not an operation: ${JSON.stringify(reply)}`);
    }
    for (const message of [{ op: 'list', includeKey: true }, { op: 'status', reveal: true }, { op: 'models', provider: id, includeKey: true }, { op: 'setModel', provider: id, model: 'm', baseUrl: `${collector.origin}/v1`, name: 'Hijacked', key: 'stolen' }, { op: 'ui.open', target: 'manage', edit: id }]) await call(s.app.page, message);
    const after = (await call(s.app.page, { op: 'list' })).result[0];
    assert.equal(after.name, 'Fake', 'no name was changed');
    assert.equal(after.host, new URL(fake.baseUrl).host, 'no address was changed');
    const text = await s.app.page.evaluate(() => window.oaiyTest.received());
    assert.deepEqual(NEEDLES.filter((n) => text.includes(n)), [], 'and nothing said back holds the key');
    assert.equal(collector.log.length, 0, 'nothing went anywhere else');
    await s.context.close();
  });

  it('a fetch that carries its own baseUrl (or url, or host) goes to the record\'s address and nowhere else', async () => {
    const s = await scenario('flows');
    const before = fake.log.length;
    const events = await fetchAll(s.app.page, 'b1', {
      provider: s.provider.id,
      path: '/models',
      method: 'GET',
      baseUrl: `${collector.origin}/v1`,
      url: `${collector.origin}/v1/models`,
      host: new URL(collector.origin).host,
    });
    assert.equal(events[0].t, 'head');
    assert.equal(events[0].status, 200);
    assert.equal(collector.log.length, 0, 'the collector was never asked');
    const mine = fake.log.slice(before).filter((r) => r.method === 'GET' && r.path === '/v1/models');
    assert.equal(mine.length, 1, 'the record\'s own provider was');
    await s.context.close();
  });

  it('a path of //evil.example/x, /..%2f, /a@b, or any other shape of an escape is refused, and nothing leaves', async () => {
    const s = await scenario('flows');
    const before = fake.log.length;
    for (const p of ['//evil.example/x', '/..%2f', '/a@b', '/../models', '/models/../../x', '/models?x=1', '/models#f', '\\evil', '/chat/completions/', '/v1/models', 'https://evil.example/models', '@evil.example/models', '/%2e%2e/models']) {
      const events = await fetchAll(s.app.page, `p-${p}`, { provider: s.provider.id, path: p, method: 'GET' });
      assert.equal(events.at(-1).t, 'error', p);
      assert.equal(events.at(-1).error.code, 'bad-path', `${p}: ${JSON.stringify(events.at(-1).error)}`);
    }
    assert.equal(fake.log.length, before, 'no request reached the provider for any of them');
    assert.equal(collector.log.length, 0, 'or anywhere else');
    await s.context.close();
  });

  it('the headers a page sends are the holder\'s to choose: the provider sees the key of the record and nothing the page put in', async () => {
    const s = await scenario('flows');
    const before = fake.log.length;
    await fetchAll(s.app.page, 'h1', {
      provider: s.provider.id,
      path: '/chat/completions',
      method: 'POST',
      body: JSON.stringify({ model: 'fake-chat', messages: [] }),
      headers: [['content-type', 'application/json'], ['authorization', 'Bearer stolen'], ['x-api-key', 'stolen'], ['x-evil', '1'], ['cookie', 'a=b']],
    });
    const seen = fake.log.slice(before).find((r) => r.method === 'POST');
    assert.equal(seen.headers.authorization, `Bearer ${KEY}`);
    assert.equal(seen.headers['x-api-key'], undefined);
    assert.equal(seen.headers['x-evil'], undefined);
    assert.equal(seen.headers.cookie, undefined);
    await s.context.close();
  });

  it('a redirect from the provider is not followed: the key does not go where it points', async () => {
    const redirecting = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] }, redirectTo: `${collector.origin}/leak` });
    const { context } = await newContext(browser);
    try {
      const top = await newPage(context);
      await addProvider(top.page, world.origins.providers, { ...DETAILS(), name: 'Redirecting', baseUrl: redirecting.baseUrl, check: false });
      const app = await newPage(context);
      await app.page.goto(`${world.origins.flows}/`);
      await app.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
      const provider = (await call(app.page, { op: 'list' })).result[0];
      const events = await fetchAll(app.page, 'r1', { provider: provider.id, path: '/chat/completions', method: 'POST', body: '{}' });
      assert.equal(events.at(-1).t, 'error');
      assert.equal(events.at(-1).error.code, 'redirect');
      assert.equal(collector.log.length, 0, 'nothing was sent to where the redirect pointed');
    } finally {
      await context.close();
      await redirecting.close();
    }
  });

  it('what the provider\'s own error says about the key is scrubbed before a page sees it', async () => {
    const s = await scenario('flows');
    // The fake quotes the key it was given, masked, when the key is wrong. Ask with a provider whose stored key is then wrong.
    fake.set({ key: 'sk-somebody-elses-key-0123456789abcdef' });
    const events = await fetchAll(s.app.page, 'e1', { provider: s.provider.id, path: '/models', method: 'GET' });
    fake.set({ key: KEY });
    assert.equal(events[0].status, 401);
    const text = events.filter((e) => e.t === 'chunk').map((e) => e.text).join('');
    assert.match(text, /Incorrect API key provided/);
    assert.deepEqual(NEEDLES.filter((n) => text.includes(n)), [], 'the provider\'s echo of the key is gone');
    assert.ok(!/sk-e3-Zq7/.test(text) && !text.includes(KEY.slice(-6)), 'and so is the masked start and end of it');
    await s.context.close();
  });
});

describe('E3.3 who is answered, and who can frame what', () => {
  it('a frame with an opaque origin (sandboxed) is not answered', async () => {
    const s = await scenario('flows');
    assert.equal(await s.app.page.evaluate(() => window.oaiyTest.sandboxHello()), 'none');
    await s.context.close();
  });

  it('a sibling frame of the app itself (its own origin, but not the parent window) is not answered', async () => {
    const s = await scenario('flows');
    assert.equal(await s.app.page.evaluate(() => window.oaiyTest.siblingHello()), 'none');
    await s.context.close();
  });

  it('another site cannot frame the holder: frame-ancestors refuses evil.web.localhost, and no hello is answered', async () => {
    const { context } = await newContext(browser);
    const evil = await newPage(context);
    await evil.page.goto(`${world.origins.evil}/`);
    const hello = await evil.page.evaluate((origin) => window.oaiyTest.connect({ origin, timeoutMs: 1500 }), world.origins.providers);
    assert.equal(hello, null, 'no answer');
    assert.ok(evil.seen.console.some((m) => /frame-ancestors/.test(m.text)), `the browser refused the frame: ${evil.seen.console.map((m) => m.text).join(' | ')}`);
    const frame = evil.page.frames().find((f) => f !== evil.page.mainFrame());
    assert.ok(!frame || !frame.url().startsWith(world.origins.providers) || (await frame.evaluate(() => document.querySelector('meta[name="oaiy-apps"]')).catch(() => null)) === null, 'and the document that loaded there is not the holder\'s');
    await context.close();
  });

  it('nor can an app frame the Providers page or the modal\'s manage routes: a key is typed only in a top-level window', async () => {
    for (const app of ['flows', 'agent']) {
      const { context } = await newContext(browser);
      const page = await newPage(context);
      await page.page.goto(`${world.origins[app]}/`);
      const loaded = await page.page.evaluate(async (origin) => {
        const frame = document.createElement('iframe');
        frame.src = `${origin}/`;
        document.body.append(frame);
        await new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }));
        return true;
      }, world.origins.providers);
      assert.equal(loaded, true);
      await waitFor(() => page.seen.console.some((m) => /frame-ancestors/.test(m.text)), { what: `${app}: the refusal to frame the Providers page` });
      const inner = page.page.frames().find((f) => f !== page.page.mainFrame());
      const inputs = inner ? await inner.locator('input').count().catch(() => 0) : 0;
      assert.equal(inputs, 0, `${app}: no form in a frame of the Providers page`);
      await context.close();
    }
  });

  it('a hello from an origin not in the list is dropped even where a misconfigured policy let it frame the holder (the message check stands alone)', async () => {
    // A second world: the header also names evil.web.localhost as a frame-ancestor, but the list of apps the page was built for does not.
    const misconfigured = await startWorld({ providersHeaders: (rendered, origins) => rendered.replace(/(frame-ancestors )(http[^\n]*)/g, `$1$2 ${origins.evil}`) });
    const { context } = await newContext(browser);
    try {
      const evil = await newPage(context);
      await evil.page.goto(`${misconfigured.origins.evil}/`);
      const hello = await evil.page.evaluate((origin) => window.oaiyTest.connect({ origin, timeoutMs: 1500 }), misconfigured.origins.providers);
      const frame = evil.page.frames().find((f) => f !== evil.page.mainFrame());
      assert.ok(frame?.url().startsWith(misconfigured.origins.providers), 'the frame DID load (the policy was wrong)');
      assert.equal(hello, null, 'and the holder still answered no one who is not an allowed origin');
      const control = await newPage(context);
      await control.page.goto(`${misconfigured.origins.agent}/`);
      assert.equal((await control.page.evaluate((origin) => window.oaiyTest.connect({ origin }), misconfigured.origins.providers))?.t, 'hello', 'control: an allowed origin is answered by the same holder');
    } finally {
      await context.close();
      await misconfigured.close();
    }
  });
});

describe('E3.4 the modal has nothing to type into', () => {
  it('the embedded modal has no input of any kind; the Providers page has the password and address inputs (control)', async () => {
    const s = await scenario('flows');
    const modal = await s.app.page.evaluate(async (origin) => {
      const frame = document.createElement('iframe');
      frame.src = `${origin}/embed.html`;
      frame.style.width = '600px';
      frame.style.height = '400px';
      document.body.append(frame);
      await new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }));
      return true;
    }, world.origins.providers);
    assert.equal(modal, true);
    const frame = await waitFor(() => s.app.page.frames().find((f) => f.url() === `${world.origins.providers}/embed.html`), { what: 'the modal frame' });
    await frame.locator('li.row').first().waitFor({ timeout: 8000 });
    assert.equal(await frame.locator('input, textarea').count(), 0, 'no input or textarea');
    assert.equal(await frame.locator('input[type=password], input[type=url]').count(), 0, 'no password or url input');
    assert.equal(await frame.locator('[contenteditable], [contenteditable=""], [contenteditable=true]').count(), 0, 'and nothing editable');
    const buttons = await frame.getByRole('button').allTextContents();
    assert.ok(buttons.length >= 3, `it has its buttons: ${buttons}`);
    assert.ok(!buttons.some((b) => /add|edit|remove|delete|save|key|password|address/i.test(b.replace('Manage providers…', ''))), `no button changes a provider: ${buttons}`);
    const text = await frame.locator('body').innerText();
    assert.match(text, /Fake/);
    assert.match(text, /key stored/);
    assert.deepEqual(NEEDLES.filter((n) => text.includes(n)), [], 'it shows that a key is stored, never the key');

    // Control: the same selector finds the inputs on the Providers page, in a top-level window.
    await s.top.page.reload();
    await s.top.page.locator('button.tile').first().click();
    assert.ok((await s.top.page.locator('input[type=password]').count()) >= 1, 'a password input on the Providers page');
    assert.ok((await s.top.page.locator('input[type=url]').count()) >= 1, 'and an address input');
    await s.context.close();
  });

  it('the Providers page draws no form if it is ever framed (defence in depth; the policy already forbids it)', async () => {
    const permissive = await startWorld({ providersHeaders: (rendered) => rendered.replace("frame-ancestors 'none'", 'frame-ancestors *') });
    const { context } = await newContext(browser);
    try {
      const page = await newPage(context);
      await page.page.goto(`${permissive.origins.flows}/`);
      await page.page.evaluate(async (origin) => {
        const frame = document.createElement('iframe');
        frame.src = `${origin}/`;
        document.body.append(frame);
        await new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }));
      }, permissive.origins.providers);
      const frame = await waitFor(() => page.page.frames().find((f) => f.url() === `${permissive.origins.providers}/`), { what: 'the framed Providers page' });
      await frame.getByText('managed in their own tab').waitFor({ timeout: 8000 });
      assert.equal(await frame.locator('input, select, textarea').count(), 0, 'it drew no form');
    } finally {
      await context.close();
      await permissive.close();
    }
  });
});

describe('E3.5 the policy of every document', () => {
  const folder = () => path.join(world.dir, 'providers');

  it('every HTML file of the assembled folder is named by exactly one policy: the two documents an app embeds allow the apps, every other forbids all framing', () => {
    const rules = parseHeaders(fs.readFileSync(path.join(folder(), '_headers'), 'utf8'));
    const htmls = fs.readdirSync(folder()).filter((f) => f.endsWith('.html'));
    assert.deepEqual(htmls.sort(), ['broker.html', 'embed.html', 'index.html']);
    const apps = `${world.origins.agent} ${world.origins.flows}`;
    for (const file of htmls) {
      const pathname = `/${file}`;
      const matching = rules.filter((r) => r.test.test(pathname) && r.set.some(([name]) => name === 'content-security-policy'));
      assert.equal(matching.length, 1, `${pathname} is named by exactly one CSP rule, not ${matching.length}`);
      const csp = headersFor(rules, pathname)['content-security-policy'];
      const ancestors = /frame-ancestors ([^;]+)$/.exec(csp)?.[1];
      assert.equal(ancestors, file === 'index.html' ? "'none'" : apps, pathname);
    }
    for (const clean of ['/broker', '/embed', '/']) assert.ok(headersFor(rules, clean)['content-security-policy'], clean);
  });

  it('served, each has that policy once: scripts and styles only from itself, connections to any provider, no framing but the allowed', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    for (const [p, ancestors] of [['/', "'none'"], ['/index.html', "'none'"], ['/broker.html', `${world.origins.agent} ${world.origins.flows}`], ['/embed.html', `${world.origins.agent} ${world.origins.flows}`], ['/broker', `${world.origins.agent} ${world.origins.flows}`]]) {
      const response = await page.goto(`${world.origins.providers}${p}`);
      const csp = response.headers()['content-security-policy'];
      assert.ok(csp, p);
      assert.ok(!csp.includes('unsafe-inline') && !csp.includes("'unsafe-eval'"), `${p}: no unsafe script`);
      assert.match(csp, /default-src 'none'/);
      assert.match(csp, /script-src 'self' 'wasm-unsafe-eval'/);
      assert.match(csp, /worker-src 'self' blob:/);
      assert.match(csp, /frame-src 'none'/);
      assert.match(csp, /object-src 'none'/);
      assert.match(csp, /style-src 'self';/);
      assert.match(csp, /connect-src \*/);
      assert.match(csp, /base-uri 'none'/);
      assert.match(csp, /form-action 'none'/);
      assert.ok(csp.endsWith(`frame-ancestors ${ancestors}`), `${p}: ${csp}`);
      assert.equal(csp.split('default-src').length, 2, `${p}: one policy, not two joined`);
      const headers = response.headers();
      assert.equal(headers['cross-origin-embedder-policy'], 'credentialless');
      assert.equal(headers['cross-origin-resource-policy'], 'same-site');
      assert.equal(headers['cross-origin-opener-policy'], 'same-origin');
      assert.equal(headers['x-content-type-options'], 'nosniff');
      assert.equal(headers['referrer-policy'], 'no-referrer');
    }
    assert.equal((await page.goto(`${world.origins.providers}/_headers`)).status(), 404, 'the headers file is not a page');
    await context.close();
  });

  it('the pages run no inline script and load nothing from any other site (the browser reports no policy violation while they run)', async () => {
    const s = await scenario('flows');
    const embed = await newPage(s.context);
    await embed.page.goto(`${world.origins.providers}/embed.html`);
    await embed.page.locator('li.row').first().waitFor();
    await s.top.page.reload();
    await s.top.page.locator('button.tile').first().waitFor();
    for (const seen of [embed.seen, s.top.seen, s.app.seen]) {
      assert.deepEqual(seen.console.filter((m) => /Content Security Policy|Refused to/i.test(m.text)), []);
      assert.deepEqual(seen.errors, []);
    }
    // The pages call the provider the person added (that is what they are for); everything else they load is their own.
    const foreign = [...embed.seen.requests, ...s.top.seen.requests].filter((r) => !r.url.startsWith(world.origins.providers) && !r.url.startsWith(fake.origin) && !r.url.startsWith('data:') && !r.url.startsWith('blob:'));
    assert.deepEqual(foreign, [], 'every request the two pages made was to their own origin or to the provider added');
    await s.context.close();
  });
});
