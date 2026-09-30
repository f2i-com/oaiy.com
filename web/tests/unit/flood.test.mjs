/**
 * What a compromised app can cost the holder by flooding the port (design 3.2, the review's F5).
 *
 * The holder's own work is what is bounded here, apart from the provider: the review measured 5,000 `list` on one connection taking 3.6
 * seconds to drain and `setModel` a database write and a push to every connected page each. So on every connection: operations being
 * worked on at once are capped (a further one is refused `busy` at once), every operation is rate limited, a flood of refusals is not
 * answered past a number a second, an app holds only so many connections, and a quiet connection is closed and told.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { AGENT, FLOWS, brokerWorld, jsonResponse, streamResponse } from '../support/broker-world.mjs';
import { M, sleep } from '../support/holder.mjs';

const {
  HELLOS_PER_SECOND,
  HELLO_BURST,
  IDLE_CLOSE_MS,
  MAX_CONNECTIONS_PER_APP,
  MAX_PENDING_OPS,
  OPS_BURST,
  OPS_PER_SECOND,
  RECENT_ACTIVITY_MS,
  REFUSALS_ANSWERED_PER_SECOND,
} = M.sharedProtocol;

const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  for (const w of worlds) await w.close();
});

/** A clock the test moves. */
const clock = (start = 1_000_000) => {
  let at = start;
  const now = () => at;
  now.advance = (ms) => {
    at += ms;
  };
  return now;
};

const count = (client, predicate) => client.inbox.filter((m) => m && predicate(m)).length;

describe('a flood on one connection', () => {
  it('30,000 `list` are worked on a few hundred times at most, and an honest connection is answered at once (the review\'s probe: 5,000 took 3.6 seconds)', async () => {
    const w = await world();
    let executed = 0;
    const summaries = w.store.summaries.bind(w.store);
    w.store.summaries = async (...args) => {
      executed++;
      return summaries(...args);
    };
    const hostile = await w.connect();
    const honest = await w.connect({ origin: FLOWS });
    const started = Date.now();
    for (let i = 1; i <= 30_000; i++) hostile.raw({ id: i, op: 'list' });
    const status = await honest.call({ op: 'status' }, 8000);
    const latency = Date.now() - started;
    assert.equal(status.ok, true);
    await sleep(500);
    const seconds = (Date.now() - started) / 1000;
    assert.ok(executed <= OPS_BURST + OPS_PER_SECOND * seconds + 20, `${executed} lists were worked on in ${seconds.toFixed(1)} s`);
    assert.ok(executed < 1000, `${executed} lists were worked on`);
    assert.ok(hostile.inbox.length < 1000, `${hostile.inbox.length} messages came back: a flood is not answered in full`);
    assert.ok(latency < 3000, `the honest connection waited ${latency} ms`);
    // And the honest one is not held to the hostile one's account.
    assert.equal((await honest.call({ op: 'list' })).ok, true);
  });

  it('every operation is counted, not only `list`: 1,000 `setModel` are 100 writes and 100 pushes at most, not 1,000 of each (the review: a write and a push each)', async () => {
    const w = await world({ handler: () => jsonResponse({ data: [{ id: 'model-a' }, { id: 'model-b' }] }) });
    const watcher = await w.connect({ origin: FLOWS });
    const hostile = await w.connect();
    assert.equal((await hostile.call({ op: 'models', provider: w.record.id })).result.ok, true);
    const before = count(watcher, (m) => m.t === 'changed');
    for (let i = 1; i <= 1000; i++) hostile.raw({ id: 1000 + i, op: 'setModel', provider: w.record.id, model: i % 2 ? 'model-a' : 'model-b' });
    await sleep(600);
    const pushes = count(watcher, (m) => m.t === 'changed') - before;
    assert.ok(pushes >= 1, 'some were done');
    assert.ok(pushes <= OPS_BURST, `${pushes} pushes to another page`);
  });

  it('the rate is a burst and then a number a second: the 101st in a row is refused `busy`, and time gives more', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const client = await w.connect();
    for (let i = 0; i < OPS_BURST; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `#${i + 1}`);
    const refused = await client.call({ op: 'list' });
    assert.equal(refused.ok, false);
    assert.equal(refused.error.code, 'busy');
    at.advance(1000);
    for (let i = 0; i < OPS_PER_SECOND; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `after a second, #${i + 1}`);
    assert.equal((await client.call({ op: 'list' })).error.code, 'busy');
    at.advance(60_000);
    for (let i = 0; i < OPS_BURST; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `after a minute, #${i + 1}: the bucket holds a burst, not a minute of them`);
    assert.equal((await client.call({ op: 'list' })).error.code, 'busy');
  });

  it('the rate is the connection\'s own: another connection of the same app has its own burst', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const a = await w.connect();
    const b = await w.connect();
    for (let i = 0; i < OPS_BURST; i++) await a.call({ op: 'list' });
    assert.equal((await a.call({ op: 'list' })).error.code, 'busy');
    assert.equal((await b.call({ op: 'list' })).ok, true);
  });

  it('a flood of refusals is not answered past a number a second, and the connection is answered again once it is quiet', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const client = await w.connect();
    for (let i = 0; i < OPS_BURST; i++) await client.call({ op: 'list' });
    const before = client.inbox.length;
    for (let i = 0; i < 1000; i++) client.raw({ id: 5000 + i, op: 'list' });
    await sleep(300);
    const refusals = client.inbox.length - before;
    assert.equal(refusals, REFUSALS_ANSWERED_PER_SECOND, `${refusals} answers to 1,000 refused operations`);
    at.advance(2000);
    assert.equal((await client.call({ op: 'list' })).ok, true, 'a second later it is answered again');
  });
});

describe('operations being worked on at once', () => {
  it('a further one is refused `busy` at once, not queued; abort and the ones after are not held up', async () => {
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    let streaming = true;
    const w = await world({
      handler: (call) => (call.url.endsWith('/models') && call.method === 'GET' && streaming === false ? gate.then(() => jsonResponse({ data: [{ id: 'm' }] })) : streamResponse(Array.from({ length: 200 }, (_, i) => `d${i}`), { gapMs: 20 })),
    });
    const client = await w.connect();
    // A stream, which the pending cap does not count and an abort must always be able to end.
    const stream = client.start({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"stream":true}' });
    await client.waitFor((m) => m.id === stream && m.t === 'chunk');
    streaming = false;

    const pending = [];
    for (let i = 0; i < MAX_PENDING_OPS; i++) {
      const id = 100 + i;
      pending.push(id);
      client.raw({ id, op: 'models', provider: w.record.id });
    }
    await sleep(100);
    const started = Date.now();
    const seventeenth = await client.call({ op: 'list' }, 2000);
    assert.equal(seventeenth.ok, false);
    assert.equal(seventeenth.error.code, 'busy');
    assert.match(seventeenth.error.message, /Try again in a moment/);
    assert.ok(Date.now() - started < 1000, 'answered at once, not after the others');

    assert.equal((await client.call({ op: 'abort', target: stream })).ok, true, 'an abort is not held behind them');
    assert.equal((await client.waitFor((m) => m.id === stream && (m.t === 'error' || m.t === 'end'))).error.code, 'aborted');

    release();
    for (const id of pending) assert.equal((await client.waitFor((m) => m.id === id && 'ok' in m)).ok, true);
    assert.equal((await client.call({ op: 'list' })).ok, true, 'and when they are done, the connection works again');
  });
});

describe('connections an app holds', () => {
  it(`an app holds ${MAX_CONNECTIONS_PER_APP}: the next one closes the least recently active, and tells it; another app is not touched`, async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const flows = await w.connect({ origin: FLOWS });
    const mine = [];
    for (let i = 0; i < MAX_CONNECTIONS_PER_APP; i++) {
      mine.push(await w.connect());
      at.advance(1000);
    }
    assert.equal(w.broker.connections(), MAX_CONNECTIONS_PER_APP + 1);
    // They have all been quiet for longer than a connection in use is (a new hello does not push out one that was used lately).
    at.advance(RECENT_ACTIVITY_MS + 1000);
    // The first is used again: the second is now the one that has been quiet the longest.
    assert.equal((await mine[0].call({ op: 'list' })).ok, true);
    at.advance(10);
    const newest = await w.connect();
    assert.equal(w.broker.connections(), MAX_CONNECTIONS_PER_APP + 1, 'still the same number');
    const closed = await mine[1].waitFor((m) => m.t === 'closed');
    assert.deepEqual(closed, { t: 'closed', reason: 'replaced' });
    assert.equal(count(mine[0], (m) => m.t === 'closed'), 0);
    assert.equal(count(flows, (m) => m.t === 'closed'), 0);
    assert.equal((await mine[0].call({ op: 'list' })).ok, true);
    assert.equal((await newest.call({ op: 'list' })).ok, true);
    assert.equal((await flows.call({ op: 'list' })).ok, true);
    // The closed one is not answered, and no longer hears a change.
    const before = mine[1].inbox.length;
    mine[1].raw({ id: 9999, op: 'list' });
    await sleep(80);
    assert.equal(mine[1].inbox.length, before);
  });

  it('an app that opens connections without end holds sixteen', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    for (let i = 0; i < 100; i++) {
      await w.connect();
      at.advance(RECENT_ACTIVITY_MS + 1000);
    }
    assert.equal(w.broker.connections(), MAX_CONNECTIONS_PER_APP);
  });

  it('one that was used lately, or has a request open, is never closed to make room: the new hello is refused, and told', async () => {
    const at = clock();
    let opened = 0;
    const w = await world({ brokerNow: at, handler: () => { opened++; return streamResponse(Array.from({ length: 400 }, (_, i) => `d${i}`), { gapMs: 20 }); } });
    const mine = [];
    for (let i = 0; i < MAX_CONNECTIONS_PER_APP; i++) {
      mine.push(await w.connect());
      at.advance(1000);
    }
    // The first has a request open and has said nothing for a long while; the rest are quiet for less than that.
    const id = mine[0].start({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"stream":true}' });
    await mine[0].waitFor((m) => m.id === id && m.t === 'chunk');
    assert.equal(opened, 1);
    at.advance(IDLE_CLOSE_MS + 1000);
    assert.equal(w.broker.sweep(), MAX_CONNECTIONS_PER_APP - 1, 'the others were quiet for long: the sweep closes them; the one with a request open stays');
    // Fill the app's connections again, all in use.
    const again = [];
    for (let i = 0; i < MAX_CONNECTIONS_PER_APP - 1; i++) {
      again.push(await w.connect());
      at.advance(1000);
    }
    assert.equal(w.broker.connections(), MAX_CONNECTIONS_PER_APP);
    at.advance(RECENT_ACTIVITY_MS - 20_000);
    const channel = new MessageChannel();
    const got = [];
    channel.port1.onmessage = (event) => got.push(event.data);
    const accepted = w.broker.onWindowMessage({ origin: AGENT, source: w.parent, data: { op: 'hello', v: 1 }, ports: [channel.port2] });
    await sleep(60);
    assert.equal(accepted, false);
    assert.deepEqual(got, [{ t: 'refused', reason: 'too-many' }]);
    assert.equal(w.broker.connections(), MAX_CONNECTIONS_PER_APP, 'none was closed for it');
    for (const c of [mine[0], ...again]) assert.equal(count(c, (m) => m.t === 'closed'), 0);
    channel.port1.close();
    await mine[0].call({ op: 'abort', target: id });
  });
});

describe('a flood of hellos', () => {
  // Every port a flood made is closed when the test ends, pass or fail: a port the holder took keeps this process alive, and a failing test
  // (a mutant that takes all 20,000) must not leave the run hanging.
  const floods = [];
  after(() => floods.forEach((f) => f.done()));
  const flood = (w, times) => {
    const channels = [];
    let accepted = 0;
    for (let i = 0; i < times; i++) {
      const channel = new MessageChannel();
      channels.push(channel);
      if (w.broker.onWindowMessage({ origin: AGENT, source: w.parent, data: { op: 'hello', v: 1 }, ports: [channel.port2] })) accepted++;
    }
    const made = { accepted, channels, done: () => channels.forEach((c) => c.port1.close()) };
    floods.push(made);
    return made;
  };

  it('does not push out the honest connection of the same frame, and makes no more than a burst of connections (the review: 20,000 hellos in 110 ms)', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const honest = await w.connect();
    assert.equal((await honest.call({ op: 'list' })).ok, true);
    at.advance(RECENT_ACTIVITY_MS + 1000); // quiet for a while: the worst case for the honest one, the first to go on a least-recently-used rule
    const started = Date.now();
    const burst = flood(w, 20_000);
    const took = Date.now() - started;
    assert.ok(burst.accepted <= HELLO_BURST, `${burst.accepted} of 20,000 hellos made a connection (${took} ms)`);
    assert.ok(w.broker.connections() <= MAX_CONNECTIONS_PER_APP);
    assert.equal(count(honest, (m) => m.t === 'closed'), 0, 'the honest connection was not closed');
    assert.equal((await honest.call({ op: 'list' })).ok, true, 'and still answers');
    burst.done();
  });

  it('is a burst, then a few a second: time gives more, an hour gives a burst and not an hour of them', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const first = flood(w, 100);
    assert.equal(first.accepted, HELLO_BURST);
    at.advance(1000);
    const second = flood(w, 100);
    assert.equal(second.accepted, HELLOS_PER_SECOND);
    at.advance(60 * 60 * 1000);
    const third = flood(w, 100);
    assert.equal(third.accepted, HELLO_BURST);
    for (const f of [first, second, third]) f.done();
  });

  it('is counted for each app: one app\'s flood leaves another\'s hello alone', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const hostile = flood(w, 100);
    assert.equal(hostile.accepted, HELLO_BURST);
    const flows = await w.connect({ origin: FLOWS });
    assert.equal(flows.inbox[0].app, 'flows');
    hostile.done();
  });
});

describe('a quiet connection', () => {
  it('is closed after a quarter of an hour, and told; one that was used is not', async () => {
    const at = clock();
    const w = await world({ brokerNow: at });
    const quiet = await w.connect();
    const used = await w.connect();
    at.advance(IDLE_CLOSE_MS - 1000);
    assert.equal(w.broker.sweep(), 0, 'not yet');
    assert.equal((await used.call({ op: 'list' })).ok, true);
    at.advance(2000);
    assert.equal(w.broker.sweep(), 1);
    assert.deepEqual(await quiet.waitFor((m) => m.t === 'closed'), { t: 'closed', reason: 'idle' });
    assert.equal(count(used, (m) => m.t === 'closed'), 0);
    assert.equal(w.broker.connections(), 1);
    at.advance(IDLE_CLOSE_MS + 1000);
    assert.equal(w.broker.sweep(), 1);
    assert.equal(w.broker.connections(), 0);
  });

  it('is not one that has a request open: a stream that runs for an hour is not a quiet page', async () => {
    const at = clock();
    let cancelled = false;
    const w = await world({ brokerNow: at, handler: () => streamResponse(Array.from({ length: 400 }, (_, i) => `d${i}`), { gapMs: 20, onCancel: () => (cancelled = true) }) });
    const client = await w.connect();
    const id = client.start({ provider: w.record.id, path: '/chat/completions', method: 'POST', body: '{"stream":true}' });
    await client.waitFor((m) => m.id === id && m.t === 'chunk');
    at.advance(IDLE_CLOSE_MS * 2);
    assert.equal(w.broker.sweep(), 0);
    assert.equal(w.broker.connections(), 1);
    await client.call({ op: 'abort', target: id });
    await client.waitFor((m) => m.id === id && (m.t === 'error' || m.t === 'end'));
    await sleep(60);
    assert.equal(cancelled, true);
    // Now it is quiet like the others, and the clock says for how long.
    at.advance(IDLE_CLOSE_MS * 2);
    assert.equal(w.broker.sweep(), 1);
  });

});
