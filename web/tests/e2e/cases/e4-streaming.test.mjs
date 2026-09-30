/**
 * E4, streaming and abort (design 8, 3.2): what an app sends through the holder to a provider comes back one chunk at a time, as it
 * is produced, and a request an app stops is stopped upstream too. The Agent's server-sent-events reader is written for a stream
 * that arrives as it is made; a holder that buffered the answer, or kept asking a provider whose caller had gone, would be wrong.
 *
 * The chunks are timed twice, by the fake provider as it WRITES each one and by the page as it RECEIVES each one over the port, and
 * the two are compared, so "incremental" is measured from both ends.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { browserVersion, launchBrowser, newContext, newPage, startWorld, waitFor } from '../harness.mjs';
import { startFakeProvider } from '../fake-provider.mjs';
import { addProvider } from '../providers-page.mjs';

const KEY = 'sk-e4-Zq7Rk2Lm9XvB4nT8wYc1HdJf6PsA3GuE5oIiN0';
const CHUNKS = ['one ', 'two ', 'three ', 'four ', 'five ', 'six ', 'seven ', 'eight'];
const DELAY = 180;

let world;
let browser;
let openai;
let claude;

before(async () => {
  world = await startWorld();
  openai = await startFakeProvider({ key: KEY, cors: { allowOrigins: [world.origins.providers] }, chunks: CHUNKS, delayMs: DELAY });
  claude = await startFakeProvider({ dialect: 'anthropic', key: KEY, cors: { allowOrigins: [world.origins.providers] }, chunks: CHUNKS, delayMs: DELAY });
  browser = await launchBrowser({ isolateOrigins: [world.origins.providers] });
  console.log(`# browser: ${await browserVersion(browser)}`);
});

after(async () => {
  await browser?.close();
  await openai?.close();
  await claude?.close();
  await world?.close();
});

async function connectedApp(details) {
  const { context } = await newContext(browser);
  const top = await newPage(context);
  await addProvider(top.page, world.origins.providers, details);
  const app = await newPage(context);
  await app.page.goto(`${world.origins.flows}/`);
  await app.page.evaluate((origin) => window.oaiyTest.connect({ origin }), world.origins.providers);
  const provider = (await app.page.evaluate(() => window.oaiyTest.call({ op: 'list' }))).result[0];
  return { context, top, app, provider };
}

const OPENAI = () => ({ name: 'Streaming OpenAI', preset: 'A server on this computer', serverKind: 'other', baseUrl: openai.baseUrl, key: KEY, model: 'fake-chat' });
const CLAUDE = () => ({ name: 'Streaming Claude', preset: 'Anthropic (Claude)', baseUrl: claude.baseUrl, key: KEY, model: 'fake-claude' });

describe('E4 streaming and abort', () => {
  for (const [label, details, path, body] of [
    ['OpenAI dialect', OPENAI, '/chat/completions', { model: 'fake-chat', stream: true, messages: [{ role: 'user', content: 'hi' }] }],
    ['Anthropic dialect', CLAUDE, '/messages', { model: 'fake-claude', stream: true, max_tokens: 64, messages: [{ role: 'user', content: 'hi' }] }],
  ]) {
    it(`${label}: chunks reach the page one at a time, as the provider writes them`, async () => {
      const fake = label.startsWith('Anthropic') ? claude : openai;
      const s = await connectedApp(details());
      const before = fake.log.length;
      const started = Date.now();
      await s.app.page.evaluate((id) => (window.__pid = id), s.provider.id);
      const events = await s.app.page.evaluate(
        async ([p, b]) => {
          window.oaiyTest.startStream('s', { provider: window.__pid, path: p, method: 'POST', body: JSON.stringify(b), headers: [['content-type', 'application/json']] });
          return window.oaiyTest.streamDone('s');
        },
        [path, body],
      );      const total = Date.now() - started;
      assert.equal(events[0].t, 'head');
      assert.equal(events[0].status, 200);
      assert.equal(events.at(-1).t, 'end');
      const chunks = events.filter((e) => e.t === 'chunk');
      assert.ok(chunks.length >= CHUNKS.length, `one message per piece the provider wrote: ${chunks.length}`);
      // As the page received them: spread over time, not delivered together at the end.
      const at = chunks.map((c) => c.at);
      const span = at.at(-1) - at[0];
      assert.ok(span >= (CHUNKS.length - 1) * DELAY * 0.7, `the chunks were spread over ${Math.round(span)} ms, not buffered`);
      const gaps = at.slice(1).map((t, i) => t - at[i]);
      assert.ok(gaps.filter((g) => g >= DELAY * 0.5).length >= CHUNKS.length - 3, `most gaps are the provider's own: ${gaps.map(Math.round)}`);
      // The first chunk came while the provider was still writing the rest.
      const entry = fake.log.slice(before).find((r) => r.method === 'POST');
      assert.ok(entry.finished);
      const firstAtPage = at[0] - events[0].at;
      assert.ok(firstAtPage < (CHUNKS.length * DELAY) / 2, `the first chunk did not wait for the last (${Math.round(firstAtPage)} ms after the head)`);
      assert.ok(total >= CHUNKS.length * DELAY * 0.8, 'and the whole took as long as the provider took');
      const text = events.filter((e) => e.t === 'chunk').map((e) => e.text).join('');
      for (const piece of CHUNKS) assert.ok(text.includes(piece.trim()), piece);
      // The holder gave it the key, in the header the dialect uses.
      if (label.startsWith('Anthropic')) {
        assert.equal(entry.headers['x-api-key'], KEY);
        assert.equal(entry.headers['anthropic-version'], '2023-06-01');
        assert.equal(entry.headers['anthropic-dangerous-direct-browser-access'], 'true');
        assert.ok(text.includes('event: content_block_delta'), 'named events come through whole');
      } else {
        assert.equal(entry.headers.authorization, `Bearer ${KEY}`);
      }
      await s.context.close();
    });
  }

  it('abort ends the stream on the page with `aborted` and closes the request to the provider, which then stops writing', async () => {
    const s = await connectedApp(OPENAI());
    await s.app.page.evaluate((id) => (window.__pid = id), s.provider.id);
    const before = openai.log.length;
    const id = await s.app.page.evaluate(() => window.oaiyTest.startStream('a', { provider: window.__pid, path: '/chat/completions', method: 'POST', body: JSON.stringify({ model: 'fake-chat', stream: true, messages: [] }), headers: [['content-type', 'application/json']] }));
    await waitFor(() => s.app.page.evaluate(() => window.oaiyTest.streamEvents('a').filter((e) => e.t === 'chunk').length >= 2), { what: 'two chunks' });
    const aborted = await s.app.page.evaluate((target) => window.oaiyTest.call({ op: 'abort', target }), id);
    assert.equal(aborted.ok, true);
    const last = await s.app.page.evaluate(() => window.oaiyTest.streamDone('a').then((events) => events.at(-1)));
    assert.equal(last.t, 'error');
    assert.equal(last.error.code, 'aborted');
    const upstream = await waitFor(() => openai.log.slice(before).find((r) => r.method === 'POST' && r.aborted), { timeout: 4000, what: 'the provider to see the request closed' });
    assert.equal(upstream.finished, false, 'it was closed before it was finished');
    assert.ok(upstream.chunksWritten < CHUNKS.length, `and the provider stopped writing after ${upstream.chunksWritten} of ${CHUNKS.length}`);
    const written = upstream.chunksWritten;
    await new Promise((resolve) => setTimeout(resolve, DELAY * 3));
    assert.equal(upstream.chunksWritten, written, 'it wrote nothing more');
    const chunksAfter = await s.app.page.evaluate(() => window.oaiyTest.streamEvents('a').filter((e) => e.t === 'chunk').length);
    assert.ok(chunksAfter < CHUNKS.length, 'and the page got no more');
    await s.context.close();
  });

  it('a page that closes its connection mid-stream (its frame removed) also stops the provider', async () => {
    const s = await connectedApp(OPENAI());
    await s.app.page.evaluate((id) => (window.__pid = id), s.provider.id);
    const before = openai.log.length;
    await s.app.page.evaluate(() => window.oaiyTest.startStream('c', { provider: window.__pid, path: '/chat/completions', method: 'POST', body: JSON.stringify({ model: 'fake-chat', stream: true, messages: [] }), headers: [['content-type', 'application/json']] }));
    await waitFor(() => s.app.page.evaluate(() => window.oaiyTest.streamEvents('c').filter((e) => e.t === 'chunk').length >= 1), { what: 'a chunk' });
    await s.app.page.close();
    // The holder's frame is gone with the page: its request to the provider goes with it.
    const upstream = await waitFor(() => openai.log.slice(before).find((r) => r.method === 'POST' && r.aborted), { timeout: 6000, what: 'the provider to see the request closed' });
    assert.equal(upstream.finished, false);
    await s.context.close();
  });
});
