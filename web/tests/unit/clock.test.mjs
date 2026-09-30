/**
 * The port's rate limit and idle timer run on a clock that only goes forward (the review's low 1): with `Date.now`, setting the computer's
 * clock back an hour made the rate limiter see a negative time and lock every existing connection for about an hour, and the idle sweep
 * never closed anything. What is counted is elapsed time, so the wall clock is not asked.
 *
 * The wall clock here is a stand-in whose skew a test sets, installed before any broker is made (a broker that read `Date.now` would hold
 * this one), and the monotonic clock is `performance.now` advanced explicitly by a test: it only ever moves forward. Host scheduling
 * must not refill the burst while the MessageChannel loop runs on a busy or slower machine.
 */
import assert from 'node:assert/strict';
import { after, describe, it } from 'node:test';
import { M } from '../support/holder.mjs';
import { brokerWorld } from '../support/broker-world.mjs';

const { IDLE_CLOSE_MS, OPS_BURST, OPS_PER_SECOND } = M.sharedProtocol;

const realNow = Date.now;
let skew = 0;
Date.now = () => realNow() + skew;
let elapsed = 0;
performance.now = () => elapsed;

const worlds = [];
const world = async (options) => {
  const w = await brokerWorld(options);
  worlds.push(w);
  return w;
};
after(async () => {
  Date.now = realNow;
  delete performance.now;
  for (const w of worlds) await w.close();
});

describe('the clock the port counts time with', () => {
  it('a wall clock set back an hour does not lock the connections that exist: the rate limit does not read it', async () => {
    const w = await world();
    const client = await w.connect();
    for (let i = 0; i < 5; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `before #${i + 1}`);
    skew = -60 * 60 * 1000;
    for (let i = 0; i < 20; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `an hour back, #${i + 1}`);
    skew = 0;
    for (let i = 0; i < 5; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `back again #${i + 1}`);
  });

  it('a wall clock set forward is not a refill: the burst is still the burst', async () => {
    const w = await world();
    const client = await w.connect();
    skew = 24 * 60 * 60 * 1000;
    let answered = 0;
    // No monotonic time passes: the first hundred are the burst, the rest are refused regardless of host execution speed.
    for (let i = 0; i < OPS_BURST + OPS_PER_SECOND; i++) if ((await client.call({ op: 'list' })).ok) answered++;
    skew = 0;
    assert.equal(answered, OPS_BURST, 'a day forward on the wall clock does not refill the burst');
    // Only monotonic elapsed time refills it, at the configured sustained rate.
    elapsed += 1000;
    for (let i = 0; i < OPS_PER_SECOND; i++) assert.equal((await client.call({ op: 'list' })).ok, true, `refill #${i + 1}`);
    assert.equal((await client.call({ op: 'list' })).ok, false, 'the one-second refill is exhausted');
  });

  it('the idle sweep closes a quiet connection when the wall clock has been set back, and keeps a used one when it has been set forward', async () => {
    const w = await world();
    const quiet = await w.connect();
    const used = await w.connect();
    elapsed += IDLE_CLOSE_MS + 1000;
    skew = -60 * 60 * 1000;
    const reply = await used.call({ op: 'list' });
    assert.equal(reply.ok, true, JSON.stringify(reply));
    assert.equal(w.broker.sweep(), 1, 'the quiet one is closed although the wall clock says less time has gone by');
    assert.deepEqual(await quiet.waitFor((m) => m.t === 'closed'), { t: 'closed', reason: 'idle' });
    assert.equal(w.broker.connections(), 1);
    // A jump forward of the wall clock does not close a connection that was just used.
    skew = 3 * 60 * 60 * 1000;
    assert.equal((await used.call({ op: 'list' })).ok, true);
    assert.equal(w.broker.sweep(), 0);
    skew = 0;
    assert.equal(w.broker.connections(), 1);
  });
});
