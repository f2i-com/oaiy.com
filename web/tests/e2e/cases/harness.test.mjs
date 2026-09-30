/**
 * The harness itself (WA-12): what every browser test in this folder stands on has to be shown to work first.
 *
 *   - the three hosts are three ORIGINS of one SITE (and another domain is another site), and each answers with its own headers;
 *   - the recorder refuses the product's own ports and says so;
 *   - the fake providers are as picky, and as awkward, as the real ones they imitate;
 *   - the leak scans FIND a secret that is there (a scan that cannot find anything proves nothing).
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';
import { browserVersion, launchBrowser, newContext, newPage, startWorld, waitFor } from '../harness.mjs';
import { deadPort, maskKey, startCollector, startFakeProvider } from '../fake-provider.mjs';
import { findInStorage, forms, heapHolds } from '../leakscan.mjs';

const KEY = 'sk-test-9f3c1b7a5d2e48a0b6c4d8e1f2a3b4c5';

let world;
let browser;

before(async () => {
  world = await startWorld();
  // The providers site is not built in this test: a shell stands in for it where a page needs something to fetch.
  world.hosts.setSite('providers', { root: path.join(world.dir, 'agent'), headers: '' });
  browser = await launchBrowser({});
  console.log(`# browser: ${await browserVersion(browser)}`);
});

after(async () => {
  await browser?.close();
  await world?.close();
});

describe('the hosts', () => {
  it('answer by Host: three names, three origins, and a name nobody is given is refused', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    for (const name of ['agent', 'flows', 'providers', 'evil', 'foreign']) {
      await page.goto(`${world.origins[name]}/`);
      assert.equal(await page.evaluate(() => location.origin), world.origins[name], name);
      assert.equal(await page.locator('#who').textContent(), `${world.hosts.host(name)}:${world.hosts.port}`, name);
    }
    const stranger = await page.goto(`http://nobody.web.localhost:${world.hosts.port}/`);
    assert.equal(stranger.status(), 421);
    await context.close();
  });

  it('each sends its own headers: the Agent is cross-origin isolated and the flow editor is not', async () => {
    const { context } = await newContext(browser);
    const { page, seen } = await newPage(context);
    const agent = await page.goto(`${world.origins.agent}/`);
    const headers = agent.headers();
    assert.equal(headers['cross-origin-opener-policy'], 'same-origin');
    assert.equal(headers['cross-origin-embedder-policy'], 'credentialless');
    assert.equal(headers['cross-origin-resource-policy'], 'same-origin');
    assert.equal(headers['x-content-type-options'], 'nosniff');
    assert.equal(headers['referrer-policy'], 'no-referrer');
    assert.match(headers['content-security-policy'], /frame-src 'self' http:\/\/providers\.web\.localhost:\d+;/);
    assert.match(headers['content-security-policy'], /frame-ancestors 'none'$/);
    assert.equal(await page.evaluate(() => crossOriginIsolated), true, 'the Agent host is isolated');

    const flows = await page.goto(`${world.origins.flows}/`);
    assert.equal(flows.headers()['cross-origin-opener-policy'], undefined);
    assert.equal(flows.headers()['cross-origin-embedder-policy'], undefined);
    assert.equal(await page.evaluate(() => crossOriginIsolated), false, 'the flow editor host is not');
    assert.deepEqual(seen.errors, []);
    await context.close();
  });

  it('the three are one SITE and another domain is another: what the providers frame\'s shared storage rests on', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.agent}/`);
    await page.evaluate(
      async (targets) => {
        // The answer may be blocked (a host that sends CORP same-origin); what is asked here is what the SERVER was told.
        for (const [name, url] of Object.entries(targets)) await fetch(`${url}/probe-${name}`, { mode: 'no-cors' }).catch(() => null);
      },
      { sameSite: world.origins.providers, crossSite: world.origins.foreign },
    );
    const asked = world.hosts.requests((r) => r.path.startsWith('/probe-'));
    assert.equal(asked.find((r) => r.path === '/probe-sameSite')?.fetchSite, 'same-site');
    assert.equal(asked.find((r) => r.path === '/probe-crossSite')?.fetchSite, 'cross-site');
    await context.close();
  });

  it('rules apply in order, a later one replaces an earlier one\'s header, and `!` takes one away', async () => {
    const { parseHeaders, headersFor } = await import('../hosts.mjs');
    const rules = parseHeaders('/*\n  A: 1\n  B: 1\n/x/*\n  A: 2\n  ! B\n/x/y\n  C: 3\n');
    assert.deepEqual(headersFor(rules, '/z'), { a: '1', b: '1' });
    assert.deepEqual(headersFor(rules, '/x/q'), { a: '2' });
    assert.deepEqual(headersFor(rules, '/x/y'), { a: '2', c: '3' });
  });
});

describe('the recorder', () => {
  it('refuses the product\'s own ports and records them; another port on this computer is let through', async () => {
    const fake = await startFakeProvider({});
    const { context, blocked } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.flows}/`);
    const outcomes = await page.evaluate(
      async (allowed) => {
        const results = {};
        for (const url of ['http://127.0.0.1:17972/api/health', 'http://localhost:8080/', 'http://127.0.0.1:9333/json', allowed]) {
          try {
            await fetch(url, { mode: 'no-cors' });
            results[url] = 'sent';
          } catch {
            results[url] = 'failed';
          }
        }
        return results;
      },
      `${fake.origin}/v1/models`,
    );
    assert.equal(outcomes['http://127.0.0.1:17972/api/health'], 'failed');
    assert.equal(outcomes['http://localhost:8080/'], 'failed');
    assert.equal(outcomes['http://127.0.0.1:9333/json'], 'failed');
    assert.equal(outcomes[`${fake.origin}/v1/models`], 'sent');
    assert.deepEqual(blocked.sort(), ['http://127.0.0.1:17972/api/health', 'http://127.0.0.1:9333/json', 'http://localhost:8080/']);
    await context.close();
    await fake.close();
  });

  it('a fake provider never takes one of the product\'s ports', async () => {
    for (let i = 0; i < 30; i++) {
      const fake = await startFakeProvider({});
      assert.ok(![17972, 17872, 17973, 8080, 7860, 8783, 9333].includes(fake.port));
      await fake.close();
    }
  });
});

describe('the fake providers', () => {
  it('OpenAI: a wrong key is a readable 401 on /models and an OPAQUE failure on chat, as OpenAI does it', async () => {
    const fake = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] } });
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.providers}/`);
    const result = await page.evaluate(
      async ({ base, key }) => {
        const out = {};
        const models = await fetch(`${base}/models`, { headers: { authorization: 'Bearer wrong-key-value-123456789' } });
        out.modelsStatus = models.status;
        out.modelsMessage = (await models.json()).error.message;
        out.modelsOk = (await (await fetch(`${base}/models`, { headers: { authorization: `Bearer ${key}` } })).json()).data.length;
        try {
          await fetch(`${base}/chat/completions`, { method: 'POST', headers: { authorization: 'Bearer wrong-key-value-123456789', 'content-type': 'application/json' }, body: '{}' });
          out.chat = 'answered';
        } catch (e) {
          out.chat = `${e.name}: ${e.message}`;
        }
        return out;
      },
      { base: fake.baseUrl, key: KEY },
    );
    assert.equal(result.modelsStatus, 401);
    assert.match(result.modelsMessage, /Incorrect API key provided: wrong-ke\*+6789/);
    assert.equal(result.modelsOk, 3);
    assert.match(result.chat, /TypeError: Failed to fetch/);
    assert.equal(maskKey('wrong-key-value-123456789'), `wrong-ke${'*'.repeat(13)}6789`);
    await context.close();
    await fake.close();
  });

  it('a server that refuses CORS is up (an opaque answer) and a browser cannot read it; a dead port is refused', async () => {
    const refusing = await startFakeProvider({ cors: 'none' });
    const dead = await deadPort();
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.providers}/`);
    const result = await page.evaluate(
      async ({ up, down }) => {
        const out = {};
        try {
          await fetch(`${up}/v1/models`);
          out.cors = 'readable';
        } catch (e) {
          out.cors = e.name;
        }
        try {
          const r = await fetch(`${up}/v1/models`, { mode: 'no-cors' });
          out.noCors = r.type;
        } catch (e) {
          out.noCors = e.name;
        }
        try {
          await fetch(`http://127.0.0.1:${down}/v1/models`, { mode: 'no-cors' });
          out.down = 'answered';
        } catch (e) {
          out.down = e.name;
        }
        return out;
      },
      { up: refusing.origin, down: dead },
    );
    assert.deepEqual(result, { cors: 'TypeError', noCors: 'opaque', down: 'TypeError' });
    await context.close();
    await refusing.close();
  });

  it('a stream arrives one chunk at a time, and a caller that goes away is seen going', async () => {
    const fake = await startFakeProvider({ delayMs: 150, chunks: ['a', 'b', 'c', 'd', 'e', 'f'] });
    const controller = new AbortController();
    const started = Date.now();
    const arrivals = [];
    const response = await fetch(`${fake.baseUrl}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ stream: true }), signal: controller.signal });
    const reader = response.body.getReader();
    while (arrivals.length < 3) {
      const { done } = await reader.read();
      if (done) break;
      arrivals.push(Date.now() - started);
    }
    controller.abort();
    assert.equal(arrivals.length, 3);
    assert.ok(arrivals[2] - arrivals[0] >= 200, `chunks were spread out: ${arrivals}`);
    const entry = await waitFor(() => fake.log.find((e) => e.aborted), { what: 'the fake to see the caller go' });
    assert.equal(entry.finished, false);
    assert.ok(entry.chunksWritten < 6, 'and it stopped writing');
    await fake.close();
  });

  it('Anthropic: the browser headers are required, the list is paged, a stream is named events', async () => {
    const fake = await startFakeProvider({ dialect: 'anthropic', key: KEY, modelPages: [['m1', 'm2'], ['m3']], chunks: ['x', 'y'], delayMs: 10, cors: { allowOrigins: [world.origins.providers] } });
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.providers}/`);
    const result = await page.evaluate(
      async ({ base, key }) => {
        const headers = { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' };
        const first = await (await fetch(`${base}/models?limit=2`, { headers })).json();
        const second = await (await fetch(`${base}/models?limit=2&after_id=${first.last_id}`, { headers })).json();
        const text = await (await fetch(`${base}/messages`, { method: 'POST', headers: { ...headers, 'content-type': 'application/json' }, body: JSON.stringify({ stream: true, model: 'm', messages: [] }) })).text();
        const bad = await fetch(`${base}/models`, { headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01' } });
        return { first: first.data.map((m) => m.id), more: first.has_more, second: second.data.map((m) => m.id), events: [...text.matchAll(/^event: (.+)$/gm)].map((m) => m[1]), badStatus: bad.status };
      },
      { base: `${fake.origin}/v1`, key: KEY },
    );
    assert.deepEqual(result.first, ['m1', 'm2']);
    assert.equal(result.more, true);
    assert.deepEqual(result.second, ['m3']);
    assert.deepEqual(result.events, ['message_start', 'content_block_start', 'content_block_delta', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop']);
    assert.equal(result.badStatus, 400, 'without the direct-browser header Anthropic refuses a page');
    await context.close();
    await fake.close();
  });

  it('a redirect is a 302 to where it is told, and the collector sees whatever reaches it', async () => {
    const collector = await startCollector();
    const fake = await startFakeProvider({ redirectTo: `${collector.origin}/leak` });
    const response = await fetch(`${fake.baseUrl}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}', redirect: 'manual' });
    assert.equal(response.status, 302);
    assert.equal(response.headers.get('location'), `${collector.origin}/leak`);
    assert.equal(collector.log.length, 0, 'nothing followed it');
    await fetch(`${collector.origin}/direct`);
    assert.equal(collector.log.length, 1);
    await fake.close();
    await collector.close();
  });
});

describe('the leak scans find a secret that is there', () => {
  it('storage: localStorage, sessionStorage, a cache, an IndexedDB store and a cookie; and nothing on a clean page', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.flows}/`);
    const needles = forms(KEY);
    const clean = findInStorage(await page.evaluate(() => window.oaiyTest.dumpStorage()), needles);
    assert.deepEqual(clean, [], 'a clean page holds nothing');
    await page.evaluate(async (key) => {
      localStorage.setItem('a', key);
      sessionStorage.setItem('b', `x${key}y`);
      document.cookie = `c=${encodeURIComponent(key)}; path=/`;
      const cache = await caches.open('leak');
      await cache.put('/leaked', new Response(`body ${key}`));
      await new Promise((resolve, reject) => {
        const open = indexedDB.open('leaky', 1);
        open.onupgradeneeded = () => open.result.createObjectStore('s');
        open.onsuccess = () => {
          const tx = open.result.transaction('s', 'readwrite');
          tx.objectStore('s').put({ secret: key }, 'k');
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error);
        };
      });
    }, KEY);
    const found = findInStorage(await page.evaluate(() => window.oaiyTest.dumpStorage()), needles);
    for (const where of ['localStorage[a]', 'sessionStorage[b]', 'cache leak', 'indexedDB leaky/s[0]', 'document.cookie']) {
      assert.ok(found.some((f) => f.startsWith(where)), `${where} was found: ${found}`);
    }
    await context.close();
  });

  it('memory: a heap snapshot of a page that holds the key shows it, and one that does not shows nothing', async () => {
    const { context } = await newContext(browser);
    const { page } = await newPage(context);
    await page.goto(`${world.origins.flows}/`);
    assert.deepEqual(await heapHolds(context, page, [KEY]), [], 'nothing before');
    await page.evaluate((key) => {
      window.__held = { key: `${key}` };
    }, KEY);
    assert.deepEqual(await heapHolds(context, page, [KEY]), [KEY], 'the search finds it once it is there');
    await page.evaluate(() => {
      window.__held = null;
    });
    await context.close();
  });
});
