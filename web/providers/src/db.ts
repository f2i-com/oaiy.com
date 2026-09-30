/**
 * The providers origin's one IndexedDB database, `oaiy-providers`, and the two ways of using it: a single request, and a
 * transaction that reads and writes together.
 *
 * Stores: `records` (the provider records, by id, no key in them), `vault` (the sealed secret record), `keys` (the device
 * wrapper's non-extractable key), `meta` (settings such as a budget limit) and `budget` (the requests an app has made this hour).
 *
 * A transaction is only ever open across IndexedDB requests, never across a wait for anything else (WebCrypto included): a browser
 * ends a transaction that is left idle. So everything that needs a decrypted or encrypted value does its cryptography BEFORE it
 * opens the transaction, or AFTER it closes, and `transact` runs a callback that is all requests.
 */

export const DB_NAME = 'oaiy-providers';
export const DB_VERSION = 1;
export const STORES = ['records', 'vault', 'keys', 'meta', 'budget'] as const;
export type StoreName = (typeof STORES)[number];

/** What the database needs of the browser; tests hand in stand-ins. */
export interface DbEnv {
  indexedDB?: IDBFactory;
}

export interface Db {
  /** Run `work` on a transaction over `stores`. Resolves with what `work` returns once the transaction has COMMITTED. */
  transact<T>(stores: readonly StoreName[], mode: IDBTransactionMode, work: (tx: IDBTransaction) => Promise<T> | T): Promise<T>;
  /** One request. */
  get<T>(store: StoreName, key: string): Promise<T | undefined>;
  getAll<T>(store: StoreName): Promise<T[]>;
  put(store: StoreName, key: string, value: unknown): Promise<void>;
  delete(store: StoreName, key: string): Promise<void>;
}

export const request = <T>(r: IDBRequest<T>): Promise<T> =>
  new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error ?? new Error('IndexedDB request failed'));
  });

/** Settles when the transaction does. Made before the first await, so it cannot miss the event. */
function completion(tx: IDBTransaction): Promise<void> {
  const done = new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
  });
  done.catch(() => {}); // whoever awaits it hears; an early failure is not "unhandled"
  return done;
}

export function openDb(env: DbEnv = { indexedDB: globalThis.indexedDB }): Db {
  const factory = env.indexedDB;
  let opened: Promise<IDBDatabase> | null = null;

  const open = (): Promise<IDBDatabase> => {
    if (opened) return opened;
    const promise = new Promise<IDBDatabase>((resolve, reject) => {
      if (!factory) return reject(new Error('this browser has no IndexedDB'));
      let req: IDBOpenDBRequest;
      try {
        req = factory.open(DB_NAME, DB_VERSION);
      } catch (e) {
        return reject(e);
      }
      req.onupgradeneeded = () => {
        for (const name of STORES) if (!req.result.objectStoreNames.contains(name)) req.result.createObjectStore(name);
      };
      req.onsuccess = () => {
        const database = req.result;
        // Deleted, or upgraded by another tab: let go, and open afresh next time.
        database.onversionchange = () => {
          database.close();
          opened = null;
        };
        database.onclose = () => {
          opened = null;
        };
        resolve(database);
      };
      req.onerror = () => reject(req.error ?? new Error('indexedDB.open failed'));
      req.onblocked = () => reject(new Error('indexedDB.open was blocked'));
    });
    opened = promise;
    promise.catch(() => {
      if (opened === promise) opened = null; // try again next time
    });
    return promise;
  };

  const transact = async <T>(stores: readonly StoreName[], mode: IDBTransactionMode, work: (tx: IDBTransaction) => Promise<T> | T): Promise<T> => {
    const attempt = async (): Promise<T> => {
      const database = await open();
      const tx = database.transaction([...stores], mode);
      const done = completion(tx);
      let result: T;
      try {
        result = await work(tx);
      } catch (e) {
        try {
          tx.abort();
        } catch {
          // already finished
        }
        await done.catch(() => {});
        throw e;
      }
      await done;
      return result;
    };
    try {
      return await attempt();
    } catch (e) {
      // A connection closed under us (site data cleared, another tab upgraded): once more, on a fresh one.
      if ((e as { name?: string } | null)?.name !== 'InvalidStateError') throw e;
      opened = null;
      return attempt();
    }
  };

  return {
    transact,
    get: <T>(store: StoreName, key: string) => transact([store], 'readonly', (tx) => request<T | undefined>(tx.objectStore(store).get(key))),
    getAll: <T>(store: StoreName) => transact([store], 'readonly', (tx) => request<T[]>(tx.objectStore(store).getAll())),
    put: (store, key, value) => transact([store], 'readwrite', (tx) => void tx.objectStore(store).put(value, key)),
    delete: (store, key) => transact([store], 'readwrite', (tx) => void tx.objectStore(store).delete(key)),
  };
}
