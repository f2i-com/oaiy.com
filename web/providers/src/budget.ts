/**
 * How many requests an app may make through the port in an hour (design 3.2, threat 6): what bounds the money a hostile flow can
 * spend, since script in an app can use a key but never read it. Each app (Agent, flow editor) has its own count and limit, both
 * kept in this origin's database, so a page that reloads its frame, opens another tab or waits for a restart does not start again.
 *
 * The count is a sliding window: the moments of the requests of the last hour. Taking one is a single read-modify-write
 * transaction, so two documents cannot both take the last one.
 */
import { BUDGET_WINDOW_MS, DEFAULT_BUDGET_PER_HOUR } from '@oaiy/shared/broker/protocol';
import { request, type Db } from './db';

export type Taken = { ok: true; remaining: number } | { ok: false; retryAfterMs: number; limit: number };

export interface Budget {
  /** Count one request for `app`, if the hour has room. */
  take(app: string): Promise<Taken>;
  usage(app: string): Promise<{ used: number; limit: number }>;
  limit(app: string): Promise<number>;
  /** The most requests an hour `app` may make: 0 to 100,000 (0 shuts it out). Changed in the top-level page only. */
  setLimit(app: string, limit: number): Promise<void>;
}

const limitKey = (app: string): string => `limit:${app}`;
const MAX_LIMIT = 100_000;

export function createBudget(db: Db, now: () => number = Date.now): Budget {
  const readLimit = (value: unknown): number => (typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= MAX_LIMIT ? value : DEFAULT_BUDGET_PER_HOUR);
  const recent = (stamps: unknown, at: number): number[] => (Array.isArray(stamps) ? (stamps as unknown[]).filter((t): t is number => typeof t === 'number' && t > at - BUDGET_WINDOW_MS && t <= at + 60_000) : []);

  return {
    async take(app) {
      return db.transact(['budget', 'meta'], 'readwrite', async (tx) => {
        const budget = tx.objectStore('budget');
        const at = now();
        const limit = readLimit(await request(tx.objectStore('meta').get(limitKey(app))));
        const stamps = recent(await request(budget.get(app)), at);
        if (stamps.length >= limit) {
          return { ok: false as const, retryAfterMs: stamps.length > 0 ? Math.max(1000, stamps[0] + BUDGET_WINDOW_MS - at) : BUDGET_WINDOW_MS, limit };
        }
        stamps.push(at);
        budget.put(stamps, app);
        return { ok: true as const, remaining: limit - stamps.length };
      });
    },

    async usage(app) {
      return db.transact(['budget', 'meta'], 'readonly', async (tx) => {
        const at = now();
        const limit = readLimit(await request(tx.objectStore('meta').get(limitKey(app))));
        return { used: recent(await request(tx.objectStore('budget').get(app)), at).length, limit };
      });
    },

    async limit(app) {
      return readLimit(await db.get('meta', limitKey(app)));
    },

    async setLimit(app, limit) {
      if (!Number.isInteger(limit) || limit < 0 || limit > MAX_LIMIT) throw new Error(`The most requests an hour is a whole number from 0 to ${MAX_LIMIT}.`);
      await db.put('meta', limitKey(app), limit);
    },
  };
}
