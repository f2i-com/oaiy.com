/**
 * The request budget (web/providers/src/budget.ts, design 3.2, threat 6): what bounds the money a hostile flow can spend, since a
 * page can use a key but never read it.
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { M, anotherDocument, makeHolder } from '../support/holder.mjs';

const HOUR = 60 * 60 * 1000;

describe('the budget', () => {
  it('is 600 requests an hour by default, counted for each app on its own', async () => {
    let at = 1_000_000;
    const h = makeHolder({ now: () => at });
    assert.equal(await h.budget.limit('agent'), 600);
    assert.equal(M.sharedProtocol.DEFAULT_BUDGET_PER_HOUR, 600);
    const first = await h.budget.take('agent');
    assert.deepEqual(first, { ok: true, remaining: 599 });
    assert.deepEqual(await h.budget.usage('agent'), { used: 1, limit: 600 });
    assert.deepEqual(await h.budget.usage('flows'), { used: 0, limit: 600 }, 'the other app has its own');
  });

  it('refuses the request after the limit, says when the hour makes room, and gives the room back as the hour slides', async () => {
    let at = 5_000_000;
    const h = makeHolder({ now: () => at });
    await h.budget.setLimit('agent', 3);
    for (let i = 0; i < 3; i++) {
      assert.equal((await h.budget.take('agent')).ok, true);
      at += 10_000;
    }
    const refused = await h.budget.take('agent');
    assert.equal(refused.ok, false);
    assert.equal(refused.limit, 3);
    assert.equal(refused.retryAfterMs, HOUR - 30_000, 'the oldest request leaves the hour first');
    at += HOUR - 30_000 + 1;
    assert.equal((await h.budget.take('agent')).ok, true, 'the first request has left the window');
    assert.equal((await h.budget.take('agent')).ok, false, 'and only that one');
    at += 10_000;
    assert.equal((await h.budget.take('agent')).ok, true, 'the second leaves ten seconds later');
  });

  it('is kept in the database: a reload, another tab or a restart does not start again', async () => {
    let at = 9_000_000;
    const h = makeHolder({ now: () => at });
    await h.budget.setLimit('flows', 2);
    await h.budget.take('flows');
    await h.budget.take('flows');
    const reloaded = anotherDocument(h);
    assert.equal(await reloaded.budget.limit('flows'), 2);
    assert.equal((await reloaded.budget.take('flows')).ok, false);
    assert.deepEqual(await reloaded.budget.usage('flows'), { used: 2, limit: 2 });
  });

  it('two documents cannot both take the last request', async () => {
    const h = makeHolder();
    const other = anotherDocument(h);
    await h.budget.setLimit('agent', 10);
    const results = await Promise.all(Array.from({ length: 40 }, (_, i) => (i % 2 ? other : h).budget.take('agent')));
    assert.equal(results.filter((r) => r.ok).length, 10);
    assert.equal((await h.budget.usage('agent')).used, 10);
  });

  it('a limit is a whole number from 0 (shut out) to 100,000, and a garbled stored one is the default', async () => {
    const h = makeHolder();
    for (const bad of [-1, 1.5, 100_001, NaN, '5', undefined]) await assert.rejects(h.budget.setLimit('agent', bad), /whole number/, String(bad));
    await h.budget.setLimit('agent', 0);
    assert.equal((await h.budget.take('agent')).ok, false, 'a limit of 0 shuts the app out');
    await h.db.put('meta', 'limit:agent', 'lots');
    assert.equal(await h.budget.limit('agent'), 600);
    await h.db.put('budget', 'agent', 'not a list');
    assert.equal((await h.budget.take('agent')).ok, true, 'a damaged count is an empty one');
  });

  it('moments in the future (a clock that was wrong) are not counted against the app for ever', async () => {
    let at = 1_000_000;
    const h = makeHolder({ now: () => at });
    await h.db.put('budget', 'agent', [at + 10 * HOUR, at + 10 * HOUR]);
    assert.deepEqual(await h.budget.usage('agent'), { used: 0, limit: 600 });
  });
});
