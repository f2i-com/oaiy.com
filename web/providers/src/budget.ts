/**
 * How much an app may ask of the providers in an hour (design 3.2, threat 6): what bounds a hostile flow, since script in an app can
 * use a key but never read it. Each app (Agent, flow editor, and the embedded modal) has its own count of requests AND of bytes sent
 * and its own limits, all kept in this origin's database, so a page that reloads its frame, opens another tab or waits for a restart
 * does not start again.
 *
 * What this bounds is VOLUME: how many requests, how many bytes go out. It does not bound COST. A request can name any model the key may
 * use and ask for a long reply, and what that costs is the provider's price, not something the holder can know. What limits it further
 * is the size of a request (a record's `limits.maxBodyBytes`, 1 MiB unless it says more) and, where a record sets one, a cap on the
 * reply's length that the holder puts in the request itself. The Providers page shows the worst case these allow, in words.
 *
 * The count is a sliding window: the moment and size of each request of the last hour. Taking one is a single read-modify-write
 * transaction, so two documents cannot both take the last one.
 */
import { BUDGET_WINDOW_MS, DEFAULT_BUDGET_PER_HOUR, DEFAULT_BYTES_PER_HOUR, DEFAULT_MODAL_BUDGET_PER_HOUR, MODAL_APP } from '@oaiy/shared/broker/protocol';
import { request, type Db } from './db';

export type Taken =
  | { ok: true; remaining: number; bytesRemaining: number }
  | { ok: false; reason: 'requests' | 'bytes'; retryAfterMs: number; limit: number; byteLimit: number };

export interface Budget {
  /** Count one request of `bytes` bytes for `app`, if the hour has room for it. */
  take(app: string, bytes?: number): Promise<Taken>;
  usage(app: string): Promise<{ used: number; limit: number; bytes: number; byteLimit: number }>;
  limit(app: string): Promise<number>;
  byteLimit(app: string): Promise<number>;
  /** The most requests an hour `app` may make: 0 to 100,000 (0 shuts it out). Changed in the top-level page only. */
  setLimit(app: string, limit: number): Promise<void>;
  /** The most bytes an hour `app` may send: 0 to 4 GiB. */
  setByteLimit(app: string, limit: number): Promise<void>;
}

const limitKey = (app: string): string => `limit:${app}`;
const byteLimitKey = (app: string): string => `bytelimit:${app}`;
const MAX_LIMIT = 100_000;
const MAX_BYTE_LIMIT = 4 * 1024 * 1024 * 1024;

/** The requests an hour an app has until the owner says otherwise. */
export const defaultLimit = (app: string): number => (app === MODAL_APP ? DEFAULT_MODAL_BUDGET_PER_HOUR : DEFAULT_BUDGET_PER_HOUR);

type Stamp = [at: number, bytes: number];

export function createBudget(db: Db, now: () => number = Date.now): Budget {
  const readLimit = (app: string, value: unknown): number => (typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= MAX_LIMIT ? value : defaultLimit(app));
  const readByteLimit = (value: unknown): number => (typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= MAX_BYTE_LIMIT ? value : DEFAULT_BYTES_PER_HOUR);
  /** The requests still in the hour; an entry from before bytes were counted is a bare moment. */
  const recent = (stamps: unknown, at: number): Stamp[] => {
    if (!Array.isArray(stamps)) return [];
    const out: Stamp[] = [];
    for (const s of stamps as unknown[]) {
      const [t, b] = Array.isArray(s) ? (s as unknown[]) : [s, 0];
      if (typeof t === 'number' && t > at - BUDGET_WINDOW_MS && t <= at + 60_000) out.push([t, typeof b === 'number' && b >= 0 ? b : 0]);
    }
    return out;
  };
  const sum = (stamps: readonly Stamp[]): number => stamps.reduce((n, [, b]) => n + b, 0);

  return {
    async take(app, bytes = 0) {
      return db.transact(['budget', 'meta'], 'readwrite', async (tx) => {
        const budget = tx.objectStore('budget');
        const meta = tx.objectStore('meta');
        const at = now();
        const limit = readLimit(app, await request(meta.get(limitKey(app))));
        const byteLimit = readByteLimit(await request(meta.get(byteLimitKey(app))));
        const stamps = recent(await request(budget.get(app)), at);
        const retry = (needed: number): number => {
          // When enough of the oldest requests have left the hour to make room for `needed`.
          let freed = 0;
          for (const [t, b] of stamps) {
            freed += b;
            if (freed >= needed) return Math.max(1000, t + BUDGET_WINDOW_MS - at);
          }
          return BUDGET_WINDOW_MS;
        };
        if (stamps.length >= limit) return { ok: false as const, reason: 'requests' as const, retryAfterMs: stamps.length > 0 ? Math.max(1000, stamps[0][0] + BUDGET_WINDOW_MS - at) : BUDGET_WINDOW_MS, limit, byteLimit };
        const used = sum(stamps);
        if (used + bytes > byteLimit) return { ok: false as const, reason: 'bytes' as const, retryAfterMs: retry(used + bytes - byteLimit), limit, byteLimit };
        stamps.push([at, bytes]);
        budget.put(stamps, app);
        return { ok: true as const, remaining: limit - stamps.length, bytesRemaining: byteLimit - used - bytes };
      });
    },

    async usage(app) {
      return db.transact(['budget', 'meta'], 'readonly', async (tx) => {
        const at = now();
        const limit = readLimit(app, await request(tx.objectStore('meta').get(limitKey(app))));
        const byteLimit = readByteLimit(await request(tx.objectStore('meta').get(byteLimitKey(app))));
        const stamps = recent(await request(tx.objectStore('budget').get(app)), at);
        return { used: stamps.length, limit, bytes: sum(stamps), byteLimit };
      });
    },

    async limit(app) {
      return readLimit(app, await db.get('meta', limitKey(app)));
    },

    async byteLimit(app) {
      return readByteLimit(await db.get('meta', byteLimitKey(app)));
    },

    async setLimit(app, limit) {
      if (!Number.isInteger(limit) || limit < 0 || limit > MAX_LIMIT) throw new Error(`The most requests an hour is a whole number from 0 to ${MAX_LIMIT}.`);
      await db.put('meta', limitKey(app), limit);
    },

    async setByteLimit(app, limit) {
      if (!Number.isInteger(limit) || limit < 0 || limit > MAX_BYTE_LIMIT) throw new Error(`The most bytes an hour is a whole number from 0 to ${MAX_BYTE_LIMIT}.`);
      await db.put('meta', byteLimitKey(app), limit);
    },
  };
}
