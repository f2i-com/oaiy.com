/**
 * The parts of this origin, joined to the real browser: its database, its WebCrypto, its `fetch`, and the list of apps its page was
 * assembled for. Each page (the manage page, the embedded modal, the port) makes one.
 */
import { createBudget, type Budget } from './budget';
import { readApps } from './config';
import { openDb, type Db } from './db';
import type { PageInfo } from './net';
import { createStore, type ProviderStore } from './store';
import { createDeviceVault, type ListableVault } from './vault';

export interface Context {
  db: Db;
  vault: ListableVault;
  store: ProviderStore;
  budget: Budget;
  apps: Map<string, string>;
  page: PageInfo;
  fetchImpl: typeof fetch;
  random: (n: number) => Uint8Array;
}

export function createContext(): Context {
  const db = openDb();
  const vault = createDeviceVault(db);
  const random = (n: number): Uint8Array => crypto.getRandomValues(new Uint8Array(n));
  return {
    db,
    vault,
    store: createStore(db, vault, { random }),
    budget: createBudget(db),
    apps: readApps(),
    page: { protocol: location.protocol, origin: location.origin },
    // Called through a function: a bare `fetch` detached from the window throws.
    fetchImpl: (input, init) => fetch(input, init),
    random,
  };
}
