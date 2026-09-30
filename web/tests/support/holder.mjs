/**
 * A holder for the unit tests: the providers origin's own modules on a fake IndexedDB, WebCrypto and a `fetch` the test controls.
 */
import { webcrypto } from 'node:crypto';
import { createFakeIdb } from './fakeIdb.mjs';
import { loadTs } from './load.mjs';

export const M = await loadTs('web/tests/support/entry.ts');

/** A key as long and as random-looking as a real one. The tests search for it (and for pieces of it) in everything a page could see. */
export const KEY = 'sk-proj-Zq7Rk2Lm9XvB4nT8wYc1HdJf6PsA3GuE5oIiN0-QeUxVtWyMaS';

const randomBytes = (n) => webcrypto.getRandomValues(new Uint8Array(n));

/**
 * @param {{ crypto?: object, fetchImpl?: typeof fetch, now?: () => number, page?: { protocol: string, origin: string } }} [options]
 */
export function makeHolder(options = {}) {
  const idb = createFakeIdb();
  const crypto = options.crypto ?? webcrypto;
  const db = M.db.openDb({ indexedDB: idb.factory });
  const vault = M.vault.createDeviceVault(db, { crypto });
  const channel = options.channel === undefined ? null : options.channel;
  const store = M.store.createStore(db, vault, { random: randomBytes, channel });
  const now = options.now ?? Date.now;
  const budget = M.budget.createBudget(db, now);
  const page = options.page ?? { protocol: 'https:', origin: 'https://providers.example' };
  return { idb, db, vault, store, budget, page, crypto, now, fetchImpl: options.fetchImpl ?? (async () => new Response('{}')), random: randomBytes };
}

/** A second document of the same origin: another vault, store and budget on the SAME database. */
export function anotherDocument(holder, options = {}) {
  const db = M.db.openDb({ indexedDB: holder.idb.factory });
  const vault = M.vault.createDeviceVault(db, { crypto: options.crypto ?? holder.crypto });
  const store = M.store.createStore(db, vault, { random: randomBytes, channel: null });
  const budget = M.budget.createBudget(db, holder.now);
  return { ...holder, db, vault, store, budget };
}

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** A record input for the add form: an OpenAI-dialect service. */
export const input = (over = {}) => ({ name: 'Work OpenAI', dialect: 'openai', kind: 'external', baseUrl: 'https://api.openai.com/v1', preset: 'openai', ...over });

/** WebCrypto whose calls take a little real time, as a browser's do: a transaction held open across one would be ended by a browser. */
export function slowCrypto(ms = 4) {
  const wait = () => new Promise((resolve) => setTimeout(resolve, ms));
  const subtle = new Proxy(webcrypto.subtle, {
    get(target, prop) {
      const value = target[prop];
      if (typeof value !== 'function') return value;
      return async (...args) => {
        await wait();
        return value.apply(target, args);
      };
    },
  });
  return { subtle, getRandomValues: (a) => webcrypto.getRandomValues(a) };
}
