/**
 * The leak scans are shown to find what they claim to (the review's F8: the scans read text, and a key handed over as a Uint8Array, or
 * reversed, was not found, so two planted leaks passed every test). A scan that finds nothing proves nothing until it is shown to find
 * something, for each shape it says it looks for. What no string scan can show is stated in leakscan.mjs: a key in a form that is not on
 * the list is not found, and the proof that there is none is the rule the port keeps, not the scans.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { dumpText, encodeForScan, findInStorage, findInText, needlesFor } from '../e2e/leakscan.mjs';
import { brokerWorld } from '../support/broker-world.mjs';
import { KEY } from '../support/holder.mjs';

const NEEDLES = needlesFor(KEY);
const reversed = (s) => [...s].reverse().join('');
const noise = (n) => 'x9Q'.repeat(n);
/** The pieces the scans look for, as well as the whole key. */
const PIECES = { whole: KEY, prefix: KEY.slice(0, 14), suffix: KEY.slice(-14), middle: KEY.slice(12, 30) };

/** Shapes a page might hold a key in: name to a function from a piece of the key to a value that holds it. */
const SHAPES = {
  'as it is': (p) => `before ${p} after`,
  reversed: (p) => `before ${reversed(p)} after`,
  'a Uint8Array': (p) => ({ note: new TextEncoder().encode(p) }),
  'a Uint8Array of the reversed key': (p) => ({ note: new TextEncoder().encode(reversed(p)) }),
  'an ArrayBuffer': (p) => new TextEncoder().encode(p).buffer,
  'a view into a larger buffer': (p) => {
    const bytes = new TextEncoder().encode(`${noise(5)}${p}${noise(5)}`);
    return new Uint8Array(bytes.buffer, 15, p.length);
  },
  'a DataView': (p) => new DataView(new TextEncoder().encode(p).buffer),
  'a Uint16Array of characters': (p) => Uint16Array.from([...p].map((c) => c.charCodeAt(0))),
  'a Node Buffer': (p) => Buffer.from(p),
  'a list of character codes': (p) => [...p].map((c) => c.charCodeAt(0)),
  'a text of character codes': (p) => [...p].map((c) => c.charCodeAt(0)).join(','),
  'a text of character codes with spaces': (p) => [...p].map((c) => c.charCodeAt(0)).join(', '),
  'an array of characters': (p) => [...p],
  hex: (p) => Buffer.from(p).toString('hex'),
  HEX: (p) => Buffer.from(p).toString('hex').toUpperCase(),
  'hex of the reversed key': (p) => Buffer.from(reversed(p)).toString('hex'),
  'in a Map': (p) => new Map([['k', p]]),
  'as a Map key': (p) => new Map([[p, 1]]),
  'in a Set': (p) => new Set([p]),
  'as an object key': (p) => ({ [p]: 1 }),
  nested: (p) => ({ a: [{ b: { c: [p] } }] }),
  'inside JSON text': (p) => JSON.stringify({ error: { message: `bad ${p}` } }),
};
// The alignment of a base64 run depends on where the key starts among the bytes before it: three cases, each with a different run.
for (const lead of [0, 1, 2]) {
  SHAPES[`base64 after ${lead} bytes`] = (p) => Buffer.from(`${'-'.repeat(lead)}${p}!!!`).toString('base64');
  SHAPES[`base64url after ${lead} bytes`] = (p) => Buffer.from(`${'-'.repeat(lead)}${p}!!!`).toString('base64url');
  SHAPES[`base64 of the reversed key after ${lead} bytes`] = (p) => Buffer.from(`${'-'.repeat(lead)}${reversed(p)}!!!`).toString('base64');
}

describe('planted keys: every shape the scans claim to look for is found', () => {
  for (const [shape, make] of Object.entries(SHAPES)) {
    for (const [name, piece] of Object.entries(PIECES)) {
      it(`${shape}: ${name}`, () => {
        const found = findInText(dumpText(make(piece)), NEEDLES);
        assert.ok(found.length > 0, `not found: ${dumpText(make(piece)).slice(0, 200)}`);
      });
    }
  }
});

describe('what the scans are not fooled by', () => {
  it('a message that holds nothing of the key is not a hit: the usual replies, another key, noise, a key\'s own look-alike', () => {
    const clean = [
      [{ id: 1, ok: true, result: [{ id: 'p_abc', name: 'Work', dialect: 'openai', host: 'api.openai.com', caps: ['chat'], model: 'gpt-x', hasKey: true, kind: 'external', locked: false }] }],
      { t: 'chunk', id: 3, bytes: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n').buffer },
      { id: 2, ok: false, error: { code: 'busy', message: 'Too many requests a second. Slow down.' } },
      'sk-proj-Aa1Bb2Cc3Dd4Ee5Ff6Gg7Hh8Ii9Jj0KkLlMmNnOoPpQqRrSsTt',
      new TextEncoder().encode(noise(400)),
      Array.from({ length: 300 }, (_, i) => i % 256),
      [...'a fairly ordinary sentence in single characters'],
    ];
    for (const value of clean) assert.deepEqual(findInText(dumpText(value), NEEDLES), [], dumpText(value).slice(0, 100));
  });

  it('a needle is long enough that a chance hit is not a risk, and there are needles for every shape', () => {
    assert.ok(NEEDLES.every((n) => n.length >= 12));
    assert.ok(NEEDLES.length > 150, `${NEEDLES.length} needles`);
    for (const n of [KEY, reversed(KEY), Buffer.from(KEY).toString('hex'), Buffer.from(KEY).toString('base64')]) assert.ok(NEEDLES.includes(n), n.slice(0, 20));
  });

  it('any run of 20 characters of the key, in any shape, is found; a run of a few is not looked for (stated in leakscan.mjs)', () => {
    for (let at = 0; at + 20 <= KEY.length; at++) {
      const run = KEY.slice(at, at + 20);
      assert.ok(findInText(dumpText(run), NEEDLES).length > 0, `${at}: ${run}`);
      assert.ok(findInText(dumpText(new TextEncoder().encode(reversed(run))), NEEDLES).length > 0, `${at}, reversed bytes`);
      assert.ok(findInText(Buffer.from(run).toString('base64'), NEEDLES).length > 0, `${at}, base64`);
    }
    assert.deepEqual(findInText(dumpText(KEY.slice(3, 10)), NEEDLES), []);
  });

  it('encodeForScan keeps what JSON.stringify loses: a Uint8Array becomes text and codes, not numbered keys (the review\'s H1)', () => {
    const bytes = new TextEncoder().encode(KEY);
    assert.deepEqual(findInText(JSON.stringify({ note: bytes }), NEEDLES), [], 'JSON.stringify alone finds nothing: that is the hole');
    assert.ok(findInText(dumpText({ note: bytes }), NEEDLES).length > 0);
    assert.deepEqual(encodeForScan({ a: 1, b: [true, null, 'x'] }), { a: 1, b: [true, null, 'x'] });
    const cyclic = {};
    cyclic.self = cyclic;
    assert.equal(dumpText(cyclic), '{"self":"[cycle]"}');
  });
});

describe('the storage scan looks in the places a page can keep bytes', () => {
  const base = () => ({ indexedDB: [], localStorage: {}, sessionStorage: {}, caches: [], cookies: '', opfs: [], serviceWorkers: [] });
  const asCodes = (text) => Array.from(Buffer.from(text)).join(',');

  it('IndexedDB, localStorage, sessionStorage, a cookie, Cache Storage (text and bytes), the origin private file system, a service worker\'s address', () => {
    assert.deepEqual(findInStorage(base(), NEEDLES), []);
    const dump = {
      ...base(),
      localStorage: { a: KEY },
      sessionStorage: { b: reversed(KEY) },
      cookies: `c=${encodeURIComponent(KEY)}`,
      caches: [{ name: 'k', items: [{ url: 'https://x.example/', body: '', bytes: KEY, codes: asCodes(KEY) }] }],
      indexedDB: [{ name: 'd', stores: { s: [dumpText({ v: new TextEncoder().encode(KEY) })] } }],
      opfs: [{ path: '/dir/file.bin', text: '', codes: asCodes(reversed(KEY)) }],
      serviceWorkers: [`https://x.example/${KEY}.js`],
    };
    const found = findInStorage(dump, NEEDLES);
    for (const where of ['localStorage[a]', 'sessionStorage[b]', 'document.cookie', 'cache k', 'indexedDB d/s[0]', 'opfs /dir/file.bin', 'service worker']) {
      assert.ok(found.some((f) => f.startsWith(where)), `${where} was found: ${found.join(' | ')}`);
    }
  });

  it('a file name in the file system, or a cache\'s name, is looked in too', () => {
    assert.ok(findInStorage({ ...base(), opfs: [{ path: `/${KEY}`, text: '', codes: '' }] }, NEEDLES).length > 0);
    assert.ok(findInStorage({ ...base(), caches: [{ name: KEY, items: [] }] }, NEEDLES).length > 0);
  });
});

describe('the port\'s traffic in the unit tests is searched the same way: the review\'s two planted leaks are found', () => {
  const worlds = [];
  after(async () => {
    for (const w of worlds) await w.close();
  });

  /** A holder that, on `list`, adds `mutate(key)` to every summary: a leak of the kind H1 (bytes) and H2 (reversed) were. */
  async function leaking(mutate) {
    const w = await brokerWorld();
    worlds.push(w);
    const summaries = w.store.summaries.bind(w.store);
    w.store.summaries = async () => (await summaries()).map((r) => ({ ...r, note: mutate(KEY) }));
    const client = await w.connect();
    const reply = await client.call({ op: 'list' });
    assert.equal(reply.ok, true);
    return client;
  }

  const LEAKS = {
    'H1: the key as a Uint8Array': (key) => new TextEncoder().encode(key),
    'H2: the key reversed': (key) => reversed(key),
    'the key in base64': (key) => Buffer.from(key).toString('base64'),
    'the key in hex': (key) => Buffer.from(key).toString('hex'),
    'the key as character codes': (key) => [...key].map((c) => c.charCodeAt(0)),
    'the key as an ArrayBuffer': (key) => new TextEncoder().encode(key).buffer,
    'the key in a Map': (key) => new Map([['k', key]]),
    'a piece of the key': (key) => key.slice(4, 26),
  };
  for (const [name, mutate] of Object.entries(LEAKS)) {
    it(`${name}`, async () => {
      const client = await leaking(mutate);
      assert.ok(findInText(client.everything(), NEEDLES).length > 0);
    });
  }

  it('and an honest holder is not a hit (the control of the ones above)', async () => {
    const w = await brokerWorld();
    worlds.push(w);
    const client = await w.connect();
    await client.call({ op: 'list' });
    await client.call({ op: 'status' });
    assert.deepEqual(findInText(client.everything(), NEEDLES), []);
  });
});
